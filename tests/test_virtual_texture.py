"""
游戏内虚拟纹理与流式纹理系统 - 验收测试
运行方式：python -m pytest tests/test_virtual_texture.py -q
共 14 个测试用例
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rendering.virtual_texture import VirtualTexture


class TestVirtualTexture:
    def setup_method(self):
        self.system = VirtualTexture({
            "page_size": 128,
            "pool_capacity": 8,
            "stream_bandwidth": 2,
            "prefetch_radius": 1,
        })

    def test_case_01(self):
        # 虚拟纹理分页：纹理按 page_size 划分为页
        self.system.virtual_texture("hero", 512, 256)
        info = self.system.texture_info("hero")
        assert info["pages_x"] == 4
        assert info["pages_y"] == 2

    def test_case_02(self):
        # 页表：记录哪些页在内存中
        self.system.virtual_texture("hero", 256, 256)
        assert not self.system.is_page_resident("hero", 0, 0)
        self.system.streaming_texture("hero", 0, 0, priority=10)
        self.system.update(0.016)
        assert self.system.is_page_resident("hero", 0, 0)
        assert not self.system.is_page_resident("hero", 1, 1)
        table = self.system.page_table("hero")
        assert table[(0, 0, 0)]["resident"] is True
        assert table[(1, 1, 0)]["resident"] is False

    def test_case_03(self):
        # 创建虚拟纹理时只建页表，不全量加载到内存
        self.system.virtual_texture("big", 4096, 4096)
        assert self.system.pool_size() == 0
        assert self.system.stats["loaded"] == 0

    def test_case_04(self):
        # 流式优先级：带宽有限时高优先级的页先加载
        self.system.virtual_texture("t", 512, 512)
        for i in range(4):
            self.system.streaming_texture("t", i, 0, priority=1)
        self.system.streaming_texture("t", 3, 3, priority=100)
        self.system.update(0.016)
        assert self.system.is_page_resident("t", 3, 3)

    def test_case_05(self):
        # 流式预取：设置视野后相邻页被提前加载
        self.system.virtual_texture("t", 512, 512)
        self.system.set_view("t", 1, 1)
        self.system.update(0.016)
        assert self.system.is_page_resident("t", 1, 1)
        assert self.system.stats["prefetched"] >= 1

    def test_case_06(self):
        # 流式带宽限制：每次 update 最多加载 stream_bandwidth 页
        self.system.virtual_texture("t", 1024, 1024)
        for i in range(8):
            self.system.streaming_texture("t", i, 0, priority=5)
        self.system.update(0.016)
        assert self.system.pool_size() == 2
        self.system.update(0.016)
        assert self.system.pool_size() == 4

    def test_case_07(self):
        # 压缩格式：支持 BCn/ASTC，拒绝非法格式
        self.system.virtual_texture("t", 256, 256)
        for fmt in ("BC1", "BC3", "BC5", "BC7", "ASTC"):
            result = self.system.texture_compression("t", format=fmt)
            assert result["format"] == fmt
        with pytest.raises(ValueError):
            self.system.texture_compression("t", format="JPEG")

    def test_case_08(self):
        # 压缩真正减小体积：BC1 为 1/8，BC7 为 1/4
        self.system.virtual_texture("t", 256, 256)
        r1 = self.system.texture_compression("t", format="BC1")
        assert r1["uncompressed_size"] == 256 * 256 * 4
        assert r1["compressed_size"] < r1["uncompressed_size"]
        assert abs(r1["ratio"] - 0.125) < 1e-6
        r7 = self.system.texture_compression("t", format="BC7")
        assert abs(r7["ratio"] - 0.25) < 1e-6

    def test_case_09(self):
        # 压缩质量：可调、夹取到 [0,1]，且影响 ASTC 块尺寸与体积
        self.system.virtual_texture("t", 256, 256)
        result = self.system.texture_compression("t", format="ASTC", quality=0.3)
        assert result["quality"] == 0.3
        clamped = self.system.texture_compression("t", format="ASTC", quality=5.0)
        assert clamped["quality"] == 1.0
        hi = self.system.texture_compression("t", format="ASTC", quality=1.0)
        lo = self.system.texture_compression("t", format="ASTC", quality=0.0)
        assert lo["compressed_size"] < hi["compressed_size"]

    def test_case_10(self):
        # 池容量限制：驻留页数永不超过容量，超出时发生卸载
        self.system.virtual_texture("t", 1024, 1024)
        for i in range(8):
            for j in range(2):
                self.system.streaming_texture("t", i, j, priority=1)
        for _ in range(10):
            self.system.update(0.016)
        assert self.system.pool_size() <= 8
        assert self.system.stats["evicted"] > 0

    def test_case_11(self):
        # LRU：最久未使用的页先被卸载，刚访问过的页保留
        self.system.virtual_texture("t", 2048, 128)
        for i in range(8):
            self.system.streaming_texture("t", i, 0, priority=10)
        for _ in range(4):
            self.system.update(0.016)
        assert self.system.pool_size() == 8
        self.system.touch_page("t", 0, 0)
        self.system.streaming_texture("t", 8, 0, priority=10)
        self.system.update(0.016)
        assert self.system.is_page_resident("t", 0, 0)
        assert not self.system.is_page_resident("t", 1, 0)

    def test_case_12(self):
        # mipmap 链：尺寸逐级减半直到 1x1
        self.system.virtual_texture("t", 256, 256)
        levels = self.system.texture_lod("t")
        assert levels[0] == (256, 256)
        assert levels[-1] == (1, 1)
        assert len(levels) == 9
        for (w, h), (w2, h2) in zip(levels, levels[1:]):
            assert w2 == max(1, w // 2)
            assert h2 == max(1, h // 2)

    def test_case_13(self):
        # LOD：距离越远选择的 mip 等级越高
        self.system.virtual_texture("t", 1024, 1024)
        near = self.system.select_lod("t", 50.0)
        far = self.system.select_lod("t", 3200.0)
        assert near == 0
        assert far > near
        assert far <= self.system.mip_count("t") - 1

    def test_case_14(self):
        # 各向异性过滤：夹取到 [1,16]，斜视角下降低 mip 避免模糊
        self.system.virtual_texture("t", 1024, 1024)
        self.system.set_anisotropy(100)
        assert self.system.anisotropy == 16
        self.system.set_anisotropy(0)
        assert self.system.anisotropy == 1
        self.system.set_anisotropy(8)
        iso = self.system.select_lod("t", 800.0, anisotropic=False)
        aniso = self.system.select_lod("t", 800.0, anisotropic=True)
        assert aniso < iso


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
