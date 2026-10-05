"""
游戏内虚拟纹理与流式纹理系统 - 核心模块

包含五个子系统：
- 虚拟纹理：分页 + 页表，按需加载，不再全量驻留内存
- 流式纹理：优先级队列 + 视野预取 + 带宽限制
- 纹理压缩：BCn/ASTC 格式 + 可调质量
- 纹理池：容量限制 + LRU 卸载
- 纹理LOD：mipmap 链 + 各向异性过滤
"""
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import heapq
import math
import time


SUPPORTED_FORMATS = ("BC1", "BC3", "BC5", "BC7", "ASTC")

# 每种格式 4x4 像素块的字节数（ASTC 的块尺寸随质量变化，单独处理）
_FORMAT_BYTES_PER_BLOCK = {
    "BC1": 8,
    "BC3": 16,
    "BC5": 16,
    "BC7": 16,
}

# ASTC 质量区间 -> 块尺寸（质量越低块越大、体积越小）
_ASTC_BLOCK_SIZES = ((0.66, (4, 4)), (0.33, (6, 6)), (0.0, (8, 8)))

MAX_ANISOTROPY = 16


@dataclass
class Config:
    page_size: int = 128
    pool_capacity: int = 64
    stream_bandwidth: int = 4
    prefetch_radius: int = 1
    compression_format: str = "BC7"
    compression_quality: float = 0.8
    anisotropy: int = 4


class VirtualTexture:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        cfg = self.config
        self.page_size = max(1, int(cfg.get("page_size", 128)))
        self.pool_capacity = max(1, int(cfg.get("pool_capacity", 64)))
        self.stream_bandwidth = max(1, int(cfg.get("stream_bandwidth", 4)))
        self.prefetch_radius = max(0, int(cfg.get("prefetch_radius", 1)))
        self.compression_format = cfg.get("compression_format", "BC7")
        self.compression_quality = self._clamp_quality(
            cfg.get("compression_quality", 0.8))
        self.anisotropy = self._clamp_anisotropy(cfg.get("anisotropy", 4))

        # texture_id -> {width, height, pages_x, pages_y, mipmaps, compression}
        self._textures: Dict[str, Dict] = {}
        # texture_id -> {(px, py, mip): {"resident": bool, "last_used": int}}
        self._page_tables: Dict[str, Dict] = {}
        # 纹理池：page_key -> True，按 LRU 排序（队首最久未用）
        self._pool: "OrderedDict[Tuple, bool]" = OrderedDict()
        # 流式请求堆：(-priority, seq, page_key, is_prefetch)
        self._stream_queue: List = []
        self._stream_seq = 0
        self._pending = set()
        self._clock = 0
        self.stats = {"loaded": 0, "evicted": 0, "prefetched": 0}

    # ------------------------------------------------------------------
    # 虚拟纹理：分页 + 页表
    # ------------------------------------------------------------------
    def virtual_texture(self, texture_id: str, width: int, height: int) -> Dict:
        """创建虚拟纹理：只建立分页与页表，不把任何页加载进内存。"""
        if width <= 0 or height <= 0:
            raise ValueError("纹理尺寸必须为正数")
        mipmaps = self._build_mip_chain(width, height)
        info = {
            "width": width,
            "height": height,
            "pages_x": max(1, math.ceil(width / self.page_size)),
            "pages_y": max(1, math.ceil(height / self.page_size)),
            "mipmaps": mipmaps,
            "compression": None,
        }
        self._textures[texture_id] = info
        table: Dict[Tuple[int, int, int], Dict] = {}
        for mip, (mw, mh) in enumerate(mipmaps):
            for py in range(max(1, math.ceil(mh / self.page_size))):
                for px in range(max(1, math.ceil(mw / self.page_size))):
                    table[(px, py, mip)] = {"resident": False, "last_used": 0}
        self._page_tables[texture_id] = table
        return info

    def texture_info(self, texture_id: str) -> Dict:
        return self._textures[texture_id]

    def page_table(self, texture_id: str) -> Dict:
        """页表：记录每一页是否在内存中。"""
        return self._page_tables[texture_id]

    def is_page_resident(self, texture_id: str, px: int, py: int,
                         mip: int = 0) -> bool:
        entry = self._page_tables.get(texture_id, {}).get((px, py, mip))
        return bool(entry and entry["resident"])

    def pool_size(self) -> int:
        return len(self._pool)

    def pending_count(self) -> int:
        return len(self._stream_queue)

    # ------------------------------------------------------------------
    # 流式纹理：优先级 + 预取
    # ------------------------------------------------------------------
    def streaming_texture(self, texture_id: str, px: int, py: int,
                          mip: int = 0, priority: float = 0.0,
                          prefetch: bool = False) -> bool:
        """请求流式加载一页，按优先级入队；已驻留或已排队的页会被跳过。"""
        self._validate_page(texture_id, px, py, mip)
        key = (texture_id, px, py, mip)
        if self.is_page_resident(texture_id, px, py, mip) or key in self._pending:
            return False
        self._stream_seq += 1
        heapq.heappush(self._stream_queue,
                       (-priority, self._stream_seq, key, prefetch))
        self._pending.add(key)
        return True

    def set_view(self, texture_id: str, px: int, py: int, mip: int = 0):
        """设置视野中心页：可见页高优先级加载，周围页作为预取提前排队。"""
        self._validate_page(texture_id, px, py, mip)
        self.streaming_texture(texture_id, px, py, mip=mip, priority=100.0)
        info = self._textures[texture_id]
        mw, mh = info["mipmaps"][mip]
        max_px = max(1, math.ceil(mw / self.page_size)) - 1
        max_py = max(1, math.ceil(mh / self.page_size)) - 1
        for dy in range(-self.prefetch_radius, self.prefetch_radius + 1):
            for dx in range(-self.prefetch_radius, self.prefetch_radius + 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = px + dx, py + dy
                if 0 <= nx <= max_px and 0 <= ny <= max_py:
                    self.streaming_texture(texture_id, nx, ny, mip=mip,
                                           priority=10.0, prefetch=True)

    def update(self, dt: float):
        """每一帧按带宽限制从优先级队列中取页加载。"""
        budget = self.stream_bandwidth
        while budget > 0 and self._stream_queue:
            _, _, key, is_prefetch = heapq.heappop(self._stream_queue)
            self._pending.discard(key)
            texture_id, px, py, mip = key
            if self.is_page_resident(texture_id, px, py, mip):
                continue
            self._load_page(key)
            if is_prefetch:
                self.stats["prefetched"] += 1
            budget -= 1

    # ------------------------------------------------------------------
    # 纹理压缩：格式 + 质量
    # ------------------------------------------------------------------
    def texture_compression(self, texture_id: str,
                            format: Optional[str] = None,
                            quality: Optional[float] = None) -> Dict:
        """按指定格式与质量压缩纹理，返回压缩前后的体积信息。"""
        info = self._textures[texture_id]
        fmt = format or self.compression_format
        if fmt not in SUPPORTED_FORMATS:
            raise ValueError(
                "不支持的压缩格式: %r（支持 %s）" % (fmt, ", ".join(SUPPORTED_FORMATS)))
        q = self._clamp_quality(
            self.compression_quality if quality is None else quality)
        width, height = info["width"], info["height"]
        uncompressed = width * height * 4
        if fmt == "ASTC":
            block_w, block_h = self._astc_block_size(q)
            block_bytes = 16
        else:
            block_w = block_h = 4
            block_bytes = _FORMAT_BYTES_PER_BLOCK[fmt]
        compressed = (math.ceil(width / block_w)
                      * math.ceil(height / block_h) * block_bytes)
        result = {
            "format": fmt,
            "quality": q,
            "uncompressed_size": uncompressed,
            "compressed_size": compressed,
            "ratio": compressed / uncompressed,
        }
        info["compression"] = result
        return result

    # ------------------------------------------------------------------
    # 纹理池：容量限制 + LRU
    # ------------------------------------------------------------------
    def texture_pool(self) -> Dict:
        return {
            "capacity": self.pool_capacity,
            "size": len(self._pool),
            "pages": list(self._pool.keys()),
        }

    def touch_page(self, texture_id: str, px: int, py: int,
                   mip: int = 0) -> bool:
        """访问一页：将其标记为最近使用（LRU 中移到队尾）。"""
        key = (texture_id, px, py, mip)
        if key not in self._pool:
            return False
        self._pool.move_to_end(key)
        self._clock += 1
        self._page_tables[texture_id][(px, py, mip)]["last_used"] = self._clock
        return True

    def _load_page(self, key: Tuple):
        texture_id, px, py, mip = key
        while len(self._pool) >= self.pool_capacity:
            self._evict_lru()
        self._pool[key] = True
        self._clock += 1
        self._page_tables[texture_id][(px, py, mip)]["resident"] = True
        self._page_tables[texture_id][(px, py, mip)]["last_used"] = self._clock
        self.stats["loaded"] += 1

    def _evict_lru(self):
        """卸载最久未使用的页，并同步页表。"""
        key, _ = self._pool.popitem(last=False)
        texture_id, px, py, mip = key
        self._page_tables[texture_id][(px, py, mip)]["resident"] = False
        self.stats["evicted"] += 1

    # ------------------------------------------------------------------
    # 纹理LOD：mipmap + 各向异性过滤
    # ------------------------------------------------------------------
    def texture_lod(self, texture_id: str) -> List[Tuple[int, int]]:
        """返回 mipmap 链：从原始尺寸逐级减半直到 1x1。"""
        return list(self._textures[texture_id]["mipmaps"])

    def mip_count(self, texture_id: str) -> int:
        return len(self._textures[texture_id]["mipmaps"])

    def select_lod(self, texture_id: str, distance: float,
                   anisotropic: bool = True) -> int:
        """按距离选择 mip 等级；各向异性过滤会降低斜视角下的 mip 等级。"""
        mip_count = len(self._textures[texture_id]["mipmaps"])
        base = max(0, int(math.log2(max(distance, 1.0) / 100.0)))
        if anisotropic and self.anisotropy > 1:
            base = max(0, base - int(math.log2(self.anisotropy)))
        return min(base, mip_count - 1)

    def set_anisotropy(self, level: int):
        self.anisotropy = self._clamp_anisotropy(level)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def reset(self):
        self._textures = {}
        self._page_tables = {}
        self._pool = OrderedDict()
        self._stream_queue = []
        self._stream_seq = 0
        self._pending = set()
        self._clock = 0
        self.stats = {"loaded": 0, "evicted": 0, "prefetched": 0}

    def _validate_page(self, texture_id: str, px: int, py: int, mip: int):
        if texture_id not in self._textures:
            raise KeyError("未知纹理: %r" % texture_id)
        if (px, py, mip) not in self._page_tables[texture_id]:
            raise ValueError(
                "页坐标越界: (%d, %d, mip=%d)" % (px, py, mip))

    @staticmethod
    def _build_mip_chain(width: int, height: int) -> List[Tuple[int, int]]:
        levels = []
        w, h = width, height
        while True:
            levels.append((w, h))
            if w == 1 and h == 1:
                break
            w, h = max(1, w // 2), max(1, h // 2)
        return levels

    @staticmethod
    def _clamp_quality(q: float) -> float:
        return min(1.0, max(0.0, float(q)))

    @staticmethod
    def _clamp_anisotropy(level) -> int:
        return min(MAX_ANISOTROPY, max(1, int(level)))

    @staticmethod
    def _astc_block_size(quality: float) -> Tuple[int, int]:
        for threshold, block in _ASTC_BLOCK_SIZES:
            if quality >= threshold:
                return block
        return (8, 8)
