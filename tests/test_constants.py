"""标准常数测试：色卡参考值、Cineon、LogC3、H-D 参数、胶片预设。"""

from __future__ import annotations

import unittest

import numpy as np

import tests  # noqa: F401
from aurhythm import colorimetry as cm
from aurhythm.constants import (
    CINEON, COLORCHECKER_24_NAMES, COLORCHECKER_24_SRGB,
    COLORCHECKER_24_SRGB8, COLORCHECKER_NEUTRAL_SLICE,
    FILM_PRESETS, HDParams, LOGC3, PRESET_ORDER, CineonSpec,
)


class TestColorCheckerTable(unittest.TestCase):
    @staticmethod
    def _linear():
        """色卡参考值是 sRGB 编码域；用于 Lab/彩度计算前必须线性化。"""
        return cm.srgb_decode(COLORCHECKER_24_SRGB)

    def test_shape_and_names(self):
        self.assertEqual(COLORCHECKER_24_SRGB.shape, (24, 3))
        self.assertEqual(COLORCHECKER_24_SRGB8.shape, (24, 3))
        self.assertEqual(len(COLORCHECKER_24_NAMES), 24)

    def test_range(self):
        self.assertTrue(np.all(COLORCHECKER_24_SRGB >= 0.0))
        self.assertTrue(np.all(COLORCHECKER_24_SRGB <= 1.0))

    def test_not_placeholder_anymore(self):
        """R9 回归：旧表前 6 行是近重复占位值（存在完全相同的行）。"""
        head = COLORCHECKER_24_SRGB[:6]
        pairwise = np.abs(head[:, None, :] - head[None, :, :]).max(axis=2)
        off_diagonal = pairwise[~np.eye(6, dtype=bool)]
        self.assertGreater(float(off_diagonal.min()), 0.07,
                           "前 6 块看起来仍是占位重复值")

    def test_neutral_ramp_is_monotonic(self):
        neutral = self._linear()[COLORCHECKER_NEUTRAL_SLICE]
        L = cm.rgb_to_lab_srgb(neutral)[:, 0]
        self.assertTrue(np.all(np.diff(L) < 0.0), f"中性梯非单调: {L}")
        for row in neutral:
            self.assertLess(float(np.abs(row - row.mean()).max()), 0.01,
                            "中性梯色块应当接近中性")

    def test_white_and_black_anchors(self):
        labs = cm.rgb_to_lab_srgb(self._linear())
        self.assertGreater(float(labs[18, 0]), 94.0)   # white 9.5 → 95.8
        self.assertLess(float(labs[23, 0]), 26.0)      # black 2  → 21.7

    def test_saturated_patches_are_saturated(self):
        lch = cm.lab_to_lch(cm.rgb_to_lab_srgb(self._linear()))
        # 18 个色相块的最低彩度是 dark skin ≈ 18.05
        self.assertGreater(float(np.min(lch[:18, 1])), 15.0)
        # 6 个中性块彩度应当接近 0（最大 ≈ 0.58）
        self.assertLess(float(np.max(lch[18:, 1])), 2.0)


class TestCineon(unittest.TestCase):
    def test_anchor_codes(self):
        self.assertAlmostEqual(float(CINEON.code(0.0)), 95.0, places=12)
        self.assertAlmostEqual(float(CINEON.code(1.18)), 685.0, places=12)
        self.assertAlmostEqual(float(CINEON.code(2.048)), 1119.0, places=12)

    def test_codes_per_stop(self):
        self.assertAlmostEqual(CINEON.codes_per_stop, 150.514998, places=6)
        # 1 档曝光 = 150.5 codes
        a = float(CINEON.code(0.0))
        b = float(CINEON.code(np.log10(2.0)))
        self.assertAlmostEqual(b - a, CINEON.codes_per_stop, places=9)

    def test_reference_decades(self):
        self.assertAlmostEqual(CINEON.reference_decades, 1.18, places=12)

    def test_exposure_anchors(self):
        self.assertAlmostEqual(float(CINEON.exposure(1.18)), 0.90, places=12)
        self.assertAlmostEqual(
            float(CINEON.exposure(0.0)), 0.90 * 10.0 ** -1.18, places=12
        )

    def test_loge_exposure_roundtrip(self):
        log_e = np.linspace(-1.0, 2.0, 512)
        np.testing.assert_allclose(
            CINEON.log_e_from_exposure(CINEON.exposure(log_e)), log_e, atol=1e-12
        )

    def test_code_loge_roundtrip(self):
        code = np.linspace(0.0, 1023.0, 1024)
        np.testing.assert_allclose(CINEON.code(CINEON.log_e(code)), code, atol=1e-9)

    def test_normalize_denormalize(self):
        code = np.linspace(-100.0, 2000.0, 256)
        normalized = CINEON.normalize(code)
        self.assertGreaterEqual(float(normalized.min()), 0.0)
        self.assertLessEqual(float(normalized.max()), 1.0)
        np.testing.assert_allclose(CINEON.denormalize(CINEON.normalize(code)),
                                   np.clip(code, 0.0, 1023.0), atol=1e-9)

    def test_custom_spec_is_self_consistent(self):
        spec = CineonSpec(black_code=64.0, white_code=940.0, codes_per_decade=400.0)
        self.assertAlmostEqual(float(spec.code(spec.reference_decades)), 940.0, places=9)
        self.assertAlmostEqual(float(spec.code(0.0)), 64.0, places=9)


class TestLogC3(unittest.TestCase):
    def test_arri_reference_points(self):
        """ARRI 官方参考点：18% 中灰 → 0.391007，100% → 0.570632。"""
        self.assertAlmostEqual(float(LOGC3.encode(0.18)), 0.391007, places=6)
        self.assertAlmostEqual(float(LOGC3.encode(1.0)), 0.570632, places=6)

    def test_cut_continuity(self):
        """分段函数在 cut 点必须连续。"""
        left = float(LOGC3.encode(np.nextafter(LOGC3.cut, 0.0)))
        right = float(LOGC3.encode(LOGC3.cut))
        self.assertAlmostEqual(left, right, places=9)
        self.assertAlmostEqual(right, 0.149658, places=6)

    def test_linear_segment(self):
        self.assertAlmostEqual(float(LOGC3.encode(0.0)), LOGC3.f, places=12)
        self.assertAlmostEqual(
            float(LOGC3.encode(0.0)), 0.092809, places=6
        )

    def test_decode_roundtrip(self):
        exposure = np.concatenate([np.linspace(0.0, 0.0106, 64),
                                   np.linspace(0.0106, 20.0, 512)])
        encoded = LOGC3.encode(exposure)
        back = LOGC3.decode(encoded)
        np.testing.assert_allclose(back, exposure, rtol=1e-9, atol=1e-12)

    def test_monotonic(self):
        exposure = np.linspace(0.0, 40.0, 4096)
        self.assertTrue(np.all(np.diff(LOGC3.encode(exposure)) > 0.0))


class TestEncodingChainConsistency(unittest.TestCase):
    """Cineon 与 LogC3 必须是同一个 log 信号的两个视图。"""

    def test_18_percent_grey_code(self):
        log_e = float(CINEON.log_e_from_exposure(0.18))
        code = float(CINEON.code(log_e))
        self.assertAlmostEqual(code, 335.515, places=3)

    def test_that_code_reaches_arri_grey(self):
        code = float(CINEON.code(CINEON.log_e_from_exposure(0.18)))
        exposure = float(CINEON.exposure(CINEON.log_e(code)))
        self.assertAlmostEqual(exposure, 0.18, places=12)
        self.assertAlmostEqual(float(LOGC3.encode(exposure)), 0.391007, places=6)

    def test_reference_white_agrees(self):
        log_e = float(CINEON.log_e(685.0))
        self.assertAlmostEqual(float(CINEON.exposure(log_e)), 0.90, places=12)

    def test_one_stop_is_150_codes_on_both_sides(self):
        e1, e2 = 0.18, 0.36
        c1 = float(CINEON.code(CINEON.log_e_from_exposure(e1)))
        c2 = float(CINEON.code(CINEON.log_e_from_exposure(e2)))
        self.assertAlmostEqual(c2 - c1, CINEON.codes_per_stop, places=9)


class TestHDParams(unittest.TestCase):
    def test_derived_breakpoints(self):
        p = HDParams(a=0.5, x_mid=1.0)
        self.assertAlmostEqual(p.x_toe, 0.0, places=12)
        self.assertAlmostEqual(p.x_shoulder, 2.0, places=12)
        self.assertAlmostEqual(p.x_shoulder - p.x_toe, 1.0 / p.a, places=12)

    def test_dict_roundtrip(self):
        p = HDParams(a=0.83, x_mid=0.61, s_toe=0.11, s_sh=0.37, d_min=0.09, d_max=2.77)
        self.assertEqual(HDParams.from_dict(p.to_dict()), p)

    def test_v4_field_migration(self):
        """旧设置文件字段必须能读入。"""
        legacy = {
            "hd_slope": 4.5, "hd_mid": -0.8,
            "hd_clip_softness": 0.003,
            "hd_min": 0.12, "hd_max": 3.2,
        }
        p = HDParams.from_dict(legacy)
        self.assertAlmostEqual(p.a, 4.5)
        self.assertAlmostEqual(p.x_mid, -0.8)
        self.assertAlmostEqual(p.s_toe, 0.003)
        self.assertAlmostEqual(p.s_sh, 0.003)
        self.assertAlmostEqual(p.d_min, 0.12)
        self.assertAlmostEqual(p.d_max, 3.2)

    def test_partial_dict_uses_defaults(self):
        base = HDParams()
        p = HDParams.from_dict({"a": 0.5})
        self.assertAlmostEqual(p.a, 0.5)
        self.assertAlmostEqual(p.d_max, base.d_max)


class TestFilmPresets(unittest.TestCase):
    def test_no_preset_is_identity(self):
        preset = FILM_PRESETS[PRESET_ORDER[0]]
        self.assertTrue(preset.skip)
        np.testing.assert_allclose(preset.matrix_inv, np.eye(3))

    def test_all_presets_have_plausible_xtalk(self):
        """解串扰矩阵的物理约束。

        注意：旧 README 声称「行和 ≈1.3~1.5」，实测各预设行和为
        1.13~1.42 —— 该说法不准确。真正有意义的约束是行和为正
        （总增益 >0）、落在合理区间，且对角占优（通道被分离而非反转）。
        """
        for name in PRESET_ORDER[1:]:
            m = FILM_PRESETS[name].matrix_inv
            self.assertEqual(m.shape, (3, 3), name)
            self.assertTrue(np.all(np.diag(m) > 0), f"{name}: 对角必须 >0")
            off = m[~np.eye(3, dtype=bool)]
            self.assertTrue(np.all(off < 0), f"{name}: 非对角必须 <0（串扰消除）")
            rows = m.sum(axis=1)
            # 行和（总增益）不必 ≥1：物理来源的矩阵是 inv(Eᵀ)，而串扰是
            # **增加**密度，反解时要减掉其它染料的贡献，行和可以 <1
            # （真实的"中性 = 不等量 CMY"就是这样）。这里只拦退化情形。
            self.assertTrue(np.all(rows > 0.4) and np.all(rows < 1.6),
                            f"{name}: 行和（总增益）应在 0.4~1.6, 实际 {rows}")
            self.assertTrue(np.all(np.diag(m) > -off.reshape(3, 2).sum(axis=1)),
                            f"{name}: 必须对角占优")

    def test_all_presets_have_valid_hd(self):
        for name in PRESET_ORDER[1:]:
            hd = FILM_PRESETS[name].hd
            self.assertGreater(hd.a, 0.0, name)
            self.assertGreater(hd.d_max, hd.d_min + 0.2, name)
            self.assertGreaterEqual(hd.s_toe, 0.0, name)
            self.assertGreaterEqual(hd.s_sh, 0.0, name)

    def test_presets_are_distinct(self):
        hds = [FILM_PRESETS[n].hd for n in PRESET_ORDER[1:]]
        params = np.array([[h.a, h.x_mid, h.s_toe, h.s_sh, h.d_min, h.d_max]
                           for h in hds])
        self.assertGreater(float(np.std(params, axis=0).sum()), 0.0)

    def test_source_notes_are_honest(self):
        """没有官方曲线数据的预设不得声称「从柯达曲线采点」。"""
        for name, preset in FILM_PRESETS.items():
            if preset.skip:
                continue
            self.assertNotIn("官方H-D特性曲线采点拟合", preset.source_note, name)
            self.assertTrue(preset.source_note.strip(), f"{name}: 缺少来源标注")

    def test_preset_order_matches_dict(self):
        self.assertEqual(list(PRESET_ORDER), list(FILM_PRESETS.keys()))


if __name__ == "__main__":
    unittest.main()
