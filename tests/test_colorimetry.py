"""色彩学测试：sRGB、Bradford 色适应、Lab、CIEDE2000（Sharma 标准向量）。"""

from __future__ import annotations

import unittest

import numpy as np

import tests  # noqa: F401  (确保 sys.path 已注入)
from aurhythm import colorimetry as cm


class TestSRGB(unittest.TestCase):
    def test_matrix_matches_published(self):
        published = np.array([
            [0.41239079926595934, 0.357584339383878, 0.1804807884018343],
            [0.21263900587151027, 0.715168678767756, 0.07219231536073371],
            [0.01933081871559182, 0.11919477979462598, 0.9505321522496607],
        ])
        np.testing.assert_allclose(cm.MATRIX_SRGB_TO_XYZ_D65, published, atol=1e-9)

    def test_tabulated_white_differs_only_slightly(self):
        """圆整表格白点与 xy 现算白点的差异必须很小（记录差异来源）。"""
        np.testing.assert_allclose(cm.WHITE_D65, cm.WHITE_D65_TABULATED, atol=3e-4)
        np.testing.assert_allclose(cm.WHITE_D50, cm.WHITE_D50_TABULATED, atol=3e-4)

    def test_matrix_inverse_is_inverse(self):
        np.testing.assert_allclose(
            cm.MATRIX_XYZ_D65_TO_SRGB @ cm.MATRIX_SRGB_TO_XYZ_D65,
            np.eye(3), atol=1e-12,
        )

    def test_white_maps_to_white(self):
        np.testing.assert_allclose(
            cm.srgb_to_xyz([1.0, 1.0, 1.0]), cm.WHITE_D65, atol=1e-9
        )

    def test_encode_decode_roundtrip(self):
        x = np.linspace(0.0, 1.0, 4096)
        back = cm.srgb_decode(cm.srgb_encode(x))
        np.testing.assert_allclose(back, x, atol=1e-12)

    def test_piecewise_continuity(self):
        t = cm.SRGB_LINEAR_THRESHOLD
        self.assertAlmostEqual(
            float(cm.srgb_encode(t)), float(cm.srgb_encode(np.nextafter(t, 1.0))), places=6
        )


class TestBradford(unittest.TestCase):
    def test_d50_to_d65_matches_published(self):
        # 已发表矩阵由圆整表格照明体导出；本模块由精确 xy 现算，
        # 差异 ≤3.3e-4（对 8-bit 码值无实质影响），故容差取 5e-4。
        published = np.array([
            [0.9555766, -0.0230393, 0.0631636],
            [-0.0282895, 1.0099416, 0.0210077],
            [0.0122982, -0.0204830, 1.3299098],
        ])
        np.testing.assert_allclose(cm.CAT_D50_TO_D65, published, atol=5e-4)

    def test_cat_roundtrip(self):
        np.testing.assert_allclose(
            cm.CAT_D65_TO_D50 @ cm.CAT_D50_TO_D65, np.eye(3), atol=1e-12
        )

    def test_cat_maps_white_points(self):
        np.testing.assert_allclose(cm.CAT_D50_TO_D65 @ cm.WHITE_D50, cm.WHITE_D65,
                                   atol=1e-12)


class TestLab(unittest.TestCase):
    def test_white_is_L100(self):
        lab = cm.xyz_to_lab(cm.WHITE_D65)
        self.assertAlmostEqual(float(lab[0]), 100.0, places=9)
        self.assertAlmostEqual(float(lab[1]), 0.0, places=9)
        self.assertAlmostEqual(float(lab[2]), 0.0, places=9)

    def test_black_is_L0(self):
        lab = cm.xyz_to_lab([0.0, 0.0, 0.0])
        np.testing.assert_allclose(lab, [0.0, 0.0, 0.0], atol=1e-12)

    def test_lab_roundtrip(self):
        rng = np.random.default_rng(1234)
        lab = np.column_stack([
            rng.uniform(5.0, 95.0, 256),
            rng.uniform(-90.0, 90.0, 256),
            rng.uniform(-90.0, 90.0, 256),
        ])
        back = cm.xyz_to_lab(cm.lab_to_xyz(lab))
        np.testing.assert_allclose(back, lab, atol=1e-9)

    def test_srgb_white_lab_under_d50(self):
        lab = cm.rgb_to_lab_srgb([1.0, 1.0, 1.0], white="D50")
        self.assertAlmostEqual(float(lab[0]), 100.0, places=7)
        self.assertLess(abs(float(lab[1])), 1e-6)
        self.assertLess(abs(float(lab[2])), 1e-6)

    def test_mid_gray_known_value(self):
        # 线性 0.5 灰在 D65 下的 L* ≈ 76.07
        lab = cm.rgb_to_lab_srgb([0.5, 0.5, 0.5])
        self.assertAlmostEqual(float(lab[0]), 76.0693, places=3)


class TestDeltaE(unittest.TestCase):
    def test_de76_simple(self):
        self.assertAlmostEqual(
            float(cm.delta_e_76([50.0, 0.0, 0.0], [50.0, 3.0, 4.0])), 5.0, places=12
        )

    #: Sharma, Wu & Dalal (2005) CIEDE2000 标准测试向量（34 组）
    SHARMA = [
        ((50.0000, 2.6772, -79.7751), (50.0000, 0.0000, -82.7485), 2.0425),
        ((50.0000, 3.1571, -77.2803), (50.0000, 0.0000, -82.7485), 2.8615),
        ((50.0000, 2.8361, -74.0200), (50.0000, 0.0000, -82.7485), 3.4412),
        ((50.0000, -1.3802, -84.2814), (50.0000, 0.0000, -82.7485), 1.0000),
        ((50.0000, -1.1848, -84.8006), (50.0000, 0.0000, -82.7485), 1.0000),
        ((50.0000, -0.9009, -85.5211), (50.0000, 0.0000, -82.7485), 1.0000),
        ((50.0000, 0.0000, 0.0000), (50.0000, -1.0000, 2.0000), 2.3669),
        ((50.0000, -1.0000, 2.0000), (50.0000, 0.0000, 0.0000), 2.3669),
        ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0009), 7.1792),
        ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0010), 7.1792),
        ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0011), 7.2195),
        ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0012), 7.2195),
        ((50.0000, -0.0010, 2.4900), (50.0000, 0.0009, -2.4900), 4.8045),
        ((50.0000, -0.0010, 2.4900), (50.0000, 0.0010, -2.4900), 4.8045),
        ((50.0000, -0.0010, 2.4900), (50.0000, 0.0011, -2.4900), 4.7461),
        ((50.0000, 2.5000, 0.0000), (50.0000, 0.0000, -2.5000), 4.3065),
        ((50.0000, 2.5000, 0.0000), (73.0000, 25.0000, -18.0000), 27.1492),
        ((50.0000, 2.5000, 0.0000), (61.0000, -5.0000, 29.0000), 22.8977),
        ((50.0000, 2.5000, 0.0000), (56.0000, -27.0000, -3.0000), 31.9030),
        ((50.0000, 2.5000, 0.0000), (58.0000, 24.0000, 15.0000), 19.4535),
        ((50.0000, 2.5000, 0.0000), (50.0000, 3.1736, 0.5854), 1.0000),
        ((50.0000, 2.5000, 0.0000), (50.0000, 3.2972, 0.0000), 1.0000),
        ((50.0000, 2.5000, 0.0000), (50.0000, 1.8634, 0.5757), 1.0000),
        ((50.0000, 2.5000, 0.0000), (50.0000, 3.2592, 0.3350), 1.0000),
        ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
        ((63.0109, -31.0961, -5.8663), (62.8187, -29.7946, -4.0864), 1.2630),
        ((61.2901, 3.7196, -5.3901), (61.4292, 2.2480, -4.9620), 1.8731),
        ((35.0831, -44.1164, 3.7933), (35.0232, -40.0716, 1.5901), 1.8645),
        ((22.7233, 20.0904, -46.6940), (23.0331, 14.9730, -42.5619), 2.0373),
        ((36.4612, 47.8580, 18.3852), (36.2715, 50.5065, 21.2231), 1.4146),
        ((90.8027, -2.0831, 1.4410), (91.1528, -1.6435, 0.0447), 1.4441),
        ((90.9257, -0.5406, -0.9208), (88.6381, -0.8985, -0.7239), 1.5381),
        ((6.7747, -0.2908, -2.4247), (5.8714, -0.0985, -2.2286), 0.6377),
        ((2.0776, 0.0795, -1.1350), (0.9033, -0.0636, -0.5514), 0.9082),
    ]

    def test_sharma_vectors(self):
        for i, (lab1, lab2, expected) in enumerate(self.SHARMA, start=1):
            got = float(cm.delta_e_2000(lab1, lab2))
            self.assertAlmostEqual(
                got, expected, places=4,
                msg=f"Sharma 向量 #{i}: 期望 {expected}, 得到 {got:.6f}",
            )

    def test_sharma_vectors_vectorized(self):
        """批量与逐条必须一致（向量化边界处理正确）。"""
        a = np.array([p[0] for p in self.SHARMA])
        b = np.array([p[1] for p in self.SHARMA])
        expected = np.array([p[2] for p in self.SHARMA])
        np.testing.assert_allclose(cm.delta_e_2000(a, b), expected, atol=1e-4)

    def test_symmetry(self):
        lab1, lab2 = self.SHARMA[25][0], self.SHARMA[25][1]
        self.assertAlmostEqual(
            float(cm.delta_e_2000(lab1, lab2)),
            float(cm.delta_e_2000(lab2, lab1)), places=12,
        )


class TestLCh(unittest.TestCase):
    def test_known(self):
        lch = cm.lab_to_lch([50.0, 3.0, 4.0])
        self.assertAlmostEqual(float(lch[0]), 50.0, places=12)
        self.assertAlmostEqual(float(lch[1]), 5.0, places=12)
        self.assertAlmostEqual(float(lch[2]), 53.130102, places=5)


if __name__ == "__main__":
    unittest.main()
