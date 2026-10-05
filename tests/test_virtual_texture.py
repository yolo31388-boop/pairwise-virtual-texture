"""
游戏内虚拟纹理与流式纹理系统 - 验收测试
运行方式：python -m pytest tests/test_virtual_texture.py -q
共 14 个测试用例
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rendering.virtual_texture import VirtualTexture, SUPPORTED_COMPRESSION_FORMATS


class TestVirtualTexture:
    def setup_method(self):
        self.system = VirtualTexture({"pool_capacity_pages": 64})

    # -- 虚拟纹理：分页 ------------------------------------------------
    def test_case_01(self):
        # 256x256 纹理按 128 的页大小应切成 2x2 的页网格
        self.system.register_texture("ground", 256, 256)
        assert self.system.page_grid("ground") == (2, 2)
        # 非整页尺寸向上取整
        self.system.register_texture("rocks", 300, 200)
        assert self.system.page_grid("rocks") == (3, 2)

    # -- 虚拟纹理：页表记录驻留页 -------------------------------------
    def test_case_02(self):
        self.system.register_texture("ground", 256, 256)
        self.system.load_page("ground", 0, 0)
        assert self.system.is_page_resident("ground", 0, 0)
        assert ("ground", 0, 0, 0) in self.system.page_table
        assert not self.system.is_page_resident("ground", 1, 1)
        assert self.system.resident_page_count == 1

    # -- 虚拟纹理：按需分页加载而非全量进内存 -------------------------
    def test_case_03(self):
        self.system.register_texture("atlas", 512, 512)  # 4x4 = 16 页
        self.system.load_page("atlas", 2, 3)
        assert self.system.resident_page_count == 1
        assert self.system.is_page_resident("atlas", 2, 3)
        assert not self.system.is_page_resident("atlas", 0, 0)

    # -- 流式纹理：不预加载，只有请求后才加载 -------------------------
    def test_case_04(self):
        self.system.register_texture("ground", 256, 256)
        assert self.system.resident_page_count == 0
        self.system.request_stream("ground", pages=[(0, 0, 0), (1, 1, 0)])
        assert self.system.resident_page_count == 0
        self.system.update(0.016)
        assert self.system.is_page_resident("ground", 0, 0)
        assert self.system.is_page_resident("ground", 1, 1)
        # 未请求的页不会被加载
        assert not self.system.is_page_resident("ground", 1, 0)

    # -- 流式纹理：优先级，高优先级先加载 -----------------------------
    def test_case_05(self):
        system = VirtualTexture({"pool_capacity_pages": 64,
                                 "stream_budget_per_update": 1})
        system.register_texture("ground", 256, 256)
        system.request_stream("ground", pages=[(0, 0, 0)], priority=0)
        system.request_stream("ground", pages=[(1, 1, 0)], priority=10)
        system.update(0.016)
        # 每帧只加载一页，高优先级的 (1,1) 必须先到
        assert system.is_page_resident("ground", 1, 1)
        assert not system.is_page_resident("ground", 0, 0)

    # -- 流式纹理：预取，提前加载即将需要的页 -------------------------
    def test_case_06(self):
        system = VirtualTexture({"pool_capacity_pages": 64,
                                 "stream_budget_per_update": 64})
        system.register_texture("ground", 512, 512)  # 4x4 页
        queued = system.prefetch_around("ground", center=(0, 0), radius=1)
        assert queued == 3  # (0,1)(1,0)(1,1)
        system.update(0.016)
        assert system.is_page_resident("ground", 1, 1)
        assert system.stats["prefetches"] == 3

    # -- 流式纹理：预取页优先级低于显式请求 ---------------------------
    def test_case_07(self):
        system = VirtualTexture({"pool_capacity_pages": 64,
                                 "stream_budget_per_update": 1})
        system.register_texture("ground", 512, 512)
        system.prefetch_around("ground", center=(0, 0), radius=1)
        system.request_stream("ground", pages=[(3, 3, 0)], priority=5)
        system.update(0.016)
        assert system.is_page_resident("ground", 3, 3)
        assert not system.is_page_resident("ground", 1, 1)

    # -- 纹理压缩：支持多种压缩格式 -----------------------------------
    def test_case_08(self):
        self.system.register_texture("albedo", 1024, 1024)
        for fmt in ("BC1", "BC3", "BC7", "ASTC"):
            assert fmt in SUPPORTED_COMPRESSION_FORMATS
            result = self.system.compress("albedo", format=fmt)
            assert result["format"] == fmt
            assert result["compressed_bytes"] < result["original_bytes"]

    # -- 纹理压缩：压缩后体积显著小于未压缩 ---------------------------
    def test_case_09(self):
        self.system.register_texture("albedo", 1024, 1024)
        raw = self.system.compress("albedo", format="NONE")
        bc1 = self.system.compress("albedo", format="BC1")
        # BC1 为 4bpp，相对 RGBA8(32bpp) 应为 1/8
        assert raw["compressed_bytes"] == 1024 * 1024 * 4
        assert bc1["compressed_bytes"] == 1024 * 1024 * 4 // 8
        assert bc1["ratio"] < raw["ratio"]

    # -- 纹理压缩：质量可调，影响 ASTC 码率 ---------------------------
    def test_case_10(self):
        self.system.register_texture("albedo", 1024, 1024)
        high = self.system.compress("albedo", format="ASTC", quality=100)
        low = self.system.compress("albedo", format="ASTC", quality=10)
        assert high["quality"] == 100 and low["quality"] == 10
        assert high["compressed_bytes"] > low["compressed_bytes"]
        # 质量越界会被钳制到 [0, 100]
        clamped = self.system.compress("albedo", format="ASTC", quality=999)
        assert clamped["quality"] == 100

    # -- 纹理池：容量限制，不会无限制占用内存 -------------------------
    def test_case_11(self):
        system = VirtualTexture({"pool_capacity_pages": 4})
        system.register_texture("ground", 512, 512)  # 16 页
        for y in range(4):
            for x in range(4):
                system.load_page("ground", x, y)
        assert system.resident_page_count == 4
        assert system.stats["pages_evicted"] == 12

    # -- 纹理池：LRU 卸载最久未使用的页 -------------------------------
    def test_case_12(self):
        system = VirtualTexture({"pool_capacity_pages": 2})
        system.register_texture("ground", 512, 512)
        system.load_page("ground", 0, 0)
        system.load_page("ground", 1, 0)
        system.get_page("ground", 0, 0)        # 刷新 (0,0)，(1,0) 成为最久未用
        system.load_page("ground", 2, 0)       # 应驱逐 (1,0)
        assert system.is_page_resident("ground", 0, 0)
        assert not system.is_page_resident("ground", 1, 0)
        assert system.is_page_resident("ground", 2, 0)

    # -- 纹理 LOD：mipmap 链逐级减半 ----------------------------------
    def test_case_13(self):
        self.system.register_texture("ground", 512, 256)
        chain = self.system.mip_chain("ground")
        assert chain[0] == (512, 256)
        assert chain[1] == (256, 128)
        assert chain[-1] == (1, 1)
        # 每级可独立分页加载
        self.system.load_page("ground", 0, 0, mip=1)
        assert self.system.is_page_resident("ground", 0, 0, mip=1)

    # -- 纹理 LOD：距离选 mip + 各向异性过滤 --------------------------
    def test_case_14(self):
        self.system.register_texture("ground", 512, 512)
        near = self.system.mip_level_for_distance(1.0, "ground")
        mid = self.system.mip_level_for_distance(20.0, "ground")
        far = self.system.mip_level_for_distance(1000.0, "ground")
        assert near == 0 and mid > near
        assert far <= 8  # 不超过 mip 链长度
        # 各向异性过滤可设置并钳制到硬件上限
        self.system.set_anisotropy(16)
        assert self.system.anisotropy <= 8
        self.system.set_anisotropy(0)
        assert self.system.anisotropy == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
