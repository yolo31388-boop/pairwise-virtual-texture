"""
游戏内虚拟纹理与流式纹理系统 - 核心模块

功能：
- 虚拟纹理：纹理按页（tile）划分，带页表记录驻留页，按需分页加载
- 流式纹理：带优先级的流式请求队列 + 预取（prefetch）
- 纹理压缩：支持 BCn / ASTC 格式与压缩质量设置
- 纹理池：容量限制 + LRU 卸载最久未使用的页
- 纹理 LOD：mipmap 链生成、按距离选择 mip、各向异性过滤
"""
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import heapq
import math


SUPPORTED_COMPRESSION_FORMATS = ("NONE", "BC1", "BC3", "BC5", "BC7", "ASTC")

# 各压缩格式的 bits-per-pixel
_FORMAT_BPP = {
    "NONE": 32.0,   # RGBA8 未压缩
    "BC1": 4.0,
    "BC3": 8.0,
    "BC5": 8.0,
    "BC7": 8.0,
}

# ASTC 按质量选择块尺寸：质量越高块越小（码率越高）
_ASTC_QUALITY_BLOCKS = ((80, (4, 4)), (40, (6, 6)), (0, (8, 8)))


def _astc_bpp(quality: int) -> float:
    for min_quality, (bw, bh) in _ASTC_QUALITY_BLOCKS:
        if quality >= min_quality:
            return 128.0 / (bw * bh)
    return 128.0 / 64.0


@dataclass
class Config:
    page_size: int = 128
    pool_capacity_pages: int = 64
    compression_format: str = "BC7"
    compression_quality: int = 75
    max_anisotropy: int = 8
    prefetch_radius: int = 1
    stream_budget_per_update: int = 16
    lod_base_distance: float = 10.0


@dataclass
class Page:
    texture_id: str
    x: int
    y: int
    mip: int
    size_bytes: int
    last_used: int = 0


class VirtualTexture:
    def __init__(self, config: Optional[Dict] = None):
        if isinstance(config, Config):
            self.config = config
        else:
            self.config = Config(**(config or {}))
        if self.config.compression_format not in SUPPORTED_COMPRESSION_FORMATS:
            raise ValueError(
                "不支持的压缩格式: %s" % self.config.compression_format
            )
        self._textures: Dict[str, Dict] = {}
        self._pages: "OrderedDict[Tuple, Page]" = OrderedDict()
        self._stream_queue: List = []
        self._stream_seq = 0
        self._clock = 0
        self._anisotropy = 1
        self.stats = {
            "pages_loaded": 0,
            "pages_evicted": 0,
            "prefetches": 0,
            "stream_requests": 0,
        }

    # ------------------------------------------------------------------
    # 纹理注册与 mipmap（LOD）
    # ------------------------------------------------------------------
    def register_texture(self, texture_id: str, width: int, height: int) -> Dict:
        if width <= 0 or height <= 0:
            raise ValueError("纹理尺寸必须为正数")
        mip_levels = int(math.floor(math.log2(max(width, height)))) + 1
        info = {"width": width, "height": height, "mip_levels": mip_levels}
        self._textures[texture_id] = info
        return info

    def mip_chain(self, texture_id: str) -> List[Tuple[int, int]]:
        """生成 mipmap 链：每级宽高减半，直到 1x1。"""
        info = self._require_texture(texture_id)
        chain = []
        width, height = info["width"], info["height"]
        for _ in range(info["mip_levels"]):
            chain.append((width, height))
            width = max(1, width // 2)
            height = max(1, height // 2)
        return chain

    def mip_level_for_distance(self, distance: float, texture_id: Optional[str] = None) -> int:
        """按距离选择 mip 等级：距离每翻一倍，mip 升一级。"""
        base = max(self.config.lod_base_distance, 1e-6)
        level = int(math.floor(math.log2(max(distance, base) / base)))
        level = max(0, level)
        if texture_id is not None:
            level = min(level, self._require_texture(texture_id)["mip_levels"] - 1)
        return level

    # ------------------------------------------------------------------
    # 各向异性过滤
    # ------------------------------------------------------------------
    def set_anisotropy(self, level: int) -> int:
        self._anisotropy = max(1, min(int(level), self.config.max_anisotropy))
        return self._anisotropy

    @property
    def anisotropy(self) -> int:
        return self._anisotropy

    # ------------------------------------------------------------------
    # 虚拟纹理：分页 + 页表
    # ------------------------------------------------------------------
    def page_grid(self, texture_id: str, mip: int = 0) -> Tuple[int, int]:
        """返回 (列数, 行数)：纹理在某 mip 下的分页网格。"""
        info = self._require_texture(texture_id)
        mip_w = max(1, info["width"] >> mip)
        mip_h = max(1, info["height"] >> mip)
        cols = int(math.ceil(mip_w / self.config.page_size))
        rows = int(math.ceil(mip_h / self.config.page_size))
        return cols, rows

    def _page_size_bytes(self, mip: int) -> int:
        page_pixels = max(1, self.config.page_size >> mip)
        return page_pixels * page_pixels * 4

    def load_page(self, texture_id: str, x: int, y: int, mip: int = 0) -> Page:
        """按需加载一页进纹理池；超出容量时按 LRU 卸载。"""
        key = (texture_id, x, y, mip)
        self._clock += 1
        page = self._pages.get(key)
        if page is not None:
            page.last_used = self._clock
            self._pages.move_to_end(key)
            return page
        cols, rows = self.page_grid(texture_id, mip)
        if not (0 <= x < cols and 0 <= y < rows):
            raise ValueError("页坐标越界: (%d, %d) mip=%d" % (x, y, mip))
        page = Page(texture_id, x, y, mip, self._page_size_bytes(mip), self._clock)
        self._pages[key] = page
        self.stats["pages_loaded"] += 1
        self._evict_if_needed()
        return page

    def unload_page(self, texture_id: str, x: int, y: int, mip: int = 0) -> bool:
        return self._pages.pop((texture_id, x, y, mip), None) is not None

    def get_page(self, texture_id: str, x: int, y: int, mip: int = 0) -> Optional[Page]:
        """访问一页并刷新其 LRU 位置。"""
        key = (texture_id, x, y, mip)
        page = self._pages.get(key)
        if page is not None:
            self._clock += 1
            page.last_used = self._clock
            self._pages.move_to_end(key)
        return page

    def is_page_resident(self, texture_id: str, x: int, y: int, mip: int = 0) -> bool:
        return (texture_id, x, y, mip) in self._pages

    @property
    def page_table(self) -> Dict[Tuple, Dict]:
        """页表：记录哪些页当前驻留在内存中。"""
        return {
            key: {
                "mip": page.mip,
                "size_bytes": page.size_bytes,
                "last_used": page.last_used,
            }
            for key, page in self._pages.items()
        }

    @property
    def resident_page_count(self) -> int:
        return len(self._pages)

    def memory_usage(self) -> int:
        return sum(page.size_bytes for page in self._pages.values())

    # ------------------------------------------------------------------
    # 纹理池：容量限制 + LRU
    # ------------------------------------------------------------------
    def _evict_if_needed(self):
        capacity = max(1, self.config.pool_capacity_pages)
        while len(self._pages) > capacity:
            self._pages.popitem(last=False)  # 最久未使用的页
            self.stats["pages_evicted"] += 1

    # ------------------------------------------------------------------
    # 流式纹理：优先级队列 + 预取
    # ------------------------------------------------------------------
    def request_stream(
        self,
        texture_id: str,
        pages: Optional[List[Tuple[int, int, int]]] = None,
        priority: int = 0,
    ) -> int:
        """提交流式加载请求，按优先级（大者优先）在 update 中加载。"""
        self._require_texture(texture_id)
        if pages is None:
            cols, rows = self.page_grid(texture_id, 0)
            pages = [(x, y, 0) for y in range(rows) for x in range(cols)]
        for page in pages:
            x, y, mip = page
            self._stream_seq += 1
            heapq.heappush(
                self._stream_queue,
                (-priority, self._stream_seq, (texture_id, x, y, mip)),
            )
            self.stats["stream_requests"] += 1
        return len(pages)

    def prefetch_around(
        self,
        texture_id: str,
        center: Tuple[int, int],
        radius: Optional[int] = None,
        mip: int = 0,
        priority: int = -1,
    ) -> int:
        """预取：提前把中心页周围即将需要的页放入流式队列。"""
        radius = self.config.prefetch_radius if radius is None else radius
        cols, rows = self.page_grid(texture_id, mip)
        cx, cy = center
        queued = 0
        for y in range(max(0, cy - radius), min(rows, cy + radius + 1)):
            for x in range(max(0, cx - radius), min(cols, cx + radius + 1)):
                if (x, y) == (cx, cy) or self.is_page_resident(texture_id, x, y, mip):
                    continue
                queued += self.request_stream(
                    texture_id, pages=[(x, y, mip)], priority=priority
                )
        self.stats["prefetches"] += queued
        return queued

    @property
    def pending_stream_count(self) -> int:
        return len(self._stream_queue)

    # ------------------------------------------------------------------
    # 纹理压缩：格式 + 质量
    # ------------------------------------------------------------------
    def compress(
        self,
        texture_id: str,
        format: Optional[str] = None,
        quality: Optional[int] = None,
    ) -> Dict:
        info = self._require_texture(texture_id)
        fmt = format or self.config.compression_format
        if fmt not in SUPPORTED_COMPRESSION_FORMATS:
            raise ValueError("不支持的压缩格式: %s" % fmt)
        qual = self.config.compression_quality if quality is None else quality
        qual = max(0, min(100, int(qual)))
        pixels = info["width"] * info["height"]
        original_bytes = pixels * 4
        bpp = _astc_bpp(qual) if fmt == "ASTC" else _FORMAT_BPP[fmt]
        compressed_bytes = int(math.ceil(pixels * bpp / 8.0))
        return {
            "texture_id": texture_id,
            "format": fmt,
            "quality": qual,
            "original_bytes": original_bytes,
            "compressed_bytes": compressed_bytes,
            "ratio": compressed_bytes / original_bytes if original_bytes else 0.0,
        }

    # ------------------------------------------------------------------
    # 帧更新与重置
    # ------------------------------------------------------------------
    def update(self, dt: float):
        """每帧处理流式队列：按优先级加载，受每帧带宽预算限制。"""
        budget = max(0, self.config.stream_budget_per_update)
        processed = 0
        while self._stream_queue and processed < budget:
            _, _, (texture_id, x, y, mip) = heapq.heappop(self._stream_queue)
            self.load_page(texture_id, x, y, mip)
            processed += 1
        return processed

    def reset(self):
        """清空纹理池与流式队列（保留已注册纹理与配置）。"""
        self._pages.clear()
        self._stream_queue.clear()
        self._stream_seq = 0
        for key in self.stats:
            self.stats[key] = 0

    # ------------------------------------------------------------------
    def _require_texture(self, texture_id: str) -> Dict:
        info = self._textures.get(texture_id)
        if info is None:
            raise KeyError("纹理未注册: %s" % texture_id)
        return info
