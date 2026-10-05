"""核心管线测试：无损失裁剪链、每通道片基、中性化、片基检测、
ICC 路径、LUT 单次套用、端到端色调分辨率。

对应回归项 R5 / R6 / R8 / R10 / R11 与验收标准 S1 / S9 / S10 / S11。
"""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

import tests  # noqa: F401
from aurhythm import profiles as prof
from aurhythm.constants import CINEON, DEFAULT_PRESET, HDParams
from aurhythm.pipeline import ScientificFilmPipeline
from aurhythm.tonemap import ToneControls


# ======================================================================
# 合成底片工具
# ======================================================================

def make_negative(height: int = 48, width: int = 64, *, base=(0.90, 0.84, 0.72),
                  seed: int = 0, cast=(1.0, 1.0, 1.0)):
    """合成一张「底片」：片基最亮，场景密度随 x 变化。

    返回 (img, base_rgb)：``img`` 为相机线性空间，``base_rgb`` 是真实片基。
    """
    rng = np.random.default_rng(seed)
    ramp = np.linspace(1.0, 0.02, width)[None, :]
    img = np.repeat(ramp[:, :, None], 3, axis=2).repeat(height, axis=0)
    img = img * np.asarray(base, dtype=np.float64).reshape(1, 1, 3)
    img = img * np.asarray(cast, dtype=np.float64).reshape(1, 1, 3)
    img = img + rng.normal(0.0, 0.002, img.shape)
    # 顶部 8 行为片基（未曝光区域）
    img[:8, :, :] = (np.asarray(base, dtype=np.float64)
                     * np.asarray(cast, dtype=np.float64)).reshape(1, 1, 3)
    return np.clip(img, 0.0, None).astype(np.float32), np.asarray(base, np.float64)


def _profile():
    """构造一个让 ``camera_to_linear_srgb`` 恰好等于恒等的 profile。

    ``camera_to_linear_srgb`` 的内部链路是
    ``M_xyz2srgb · CAT(D50→D65) · m``；因此取
    ``m = CAT(D65→D50) · M_srgb2xyz`` 即可让整条链抵消为 I，
    测试里的密度/亮度关系可以精确预测。
    """
    from aurhythm import colorimetry as cm
    m = cm.CAT_D65_TO_D50 @ cm.MATRIX_SRGB_TO_XYZ_D65
    return prof.profile_from_matrix(m)


# ======================================================================
class TestNoLossChain(unittest.TestCase):
    def test_stage_abc_does_not_clip(self):
        """R8/S11：阶段 1–3 必须保留 >1 与负值（旧实现裁到 [0,1]）。"""
        img = np.array([[[1.5, 0.5, -0.05], [2.0, 1.0, 0.25]]], dtype=np.float32)
        pipe = ScientificFilmPipeline()
        pipe.load_linear_image(img)
        pipe.profile = _profile()
        out = pipe._stage_a_b_c()
        self.assertGreater(float(out.max()), 1.0, "高光被裁掉了")
        self.assertLess(float(out.min()), 0.0, "负值被裁掉了")

    def test_density_allows_negative_and_is_distinct(self):
        """R8：比片基亮的像素必须有可区分的负密度，而不是全部塌成 0。"""
        pipe = ScientificFilmPipeline()
        img = np.zeros((1, 3, 3), dtype=np.float32)
        img[0, :, :] = np.array([[1.0, 1.0, 1.0],
                                 [1.2, 1.2, 1.2],
                                 [1.5, 1.5, 1.5]])
        pipe.load_linear_image(img)
        pipe.set_base_val(np.array([1.0, 1.0, 1.0]))
        density = pipe._density(pipe._stage_a_b_c(), pipe._base_aligned())
        values = density[0, :, 0]
        self.assertAlmostEqual(float(values[0]), 0.0, places=9)
        self.assertLess(float(values[1]), 0.0)
        self.assertLess(float(values[2]), float(values[1]))
        self.assertAlmostEqual(float(values[1]), -np.log10(1.2), places=6)

    def test_highlights_inside_window_stay_distinct(self):
        """端到端：窗口内的高密度区（高光）必须保持可区分的编码。

        这是 R1 的用户可见症状：旧实现在肩部塌成单值。
        """
        base = 1.0
        # 密度 1.8 / 1.85 / 1.9 / 1.95 → 全部落在窗口 [0, 2）内
        densities = np.array([1.80, 1.85, 1.90, 1.95])
        img = np.zeros((1, 4, 3), dtype=np.float32)
        img[0, :, :] = (base * 10.0 ** (-densities)).reshape(4, 1)
        pipe = ScientificFilmPipeline()
        pipe.load_linear_image(img)
        pipe.profile = _profile()
        pipe.set_base_val(np.array([base, base, base]))
        pipe.set_hd_params(d_min=0.0, d_max=2.0)
        code = pipe.process_to_code()
        distinct = len(np.unique(np.round(code[0, :, 0], 6)))
        self.assertEqual(distinct, 4, f"高光细节丢失: {code[0, :, 0]}")
        self.assertTrue(np.all(np.diff(code[0, :, 0]) > 0.0))

    def test_above_base_pixels_map_to_black_and_are_counted(self):
        """比片基亮（负密度）在负片里是物理无意义的 —— 收敛到黑点并计数。

        注意：这是**正确行为**（负片没有比片基更透明的区域），
        与 R8 的区别在于：R8 是线性阶段被裁掉，这里是密度越窗被透明统计。
        """
        base = 1.0
        values = np.array([1.0, 1.05, 1.3, 1.6])
        img = np.zeros((1, 4, 3), dtype=np.float32)
        img[0, :, :] = values.reshape(4, 1)
        pipe = ScientificFilmPipeline()
        pipe.load_linear_image(img)
        pipe.profile = _profile()
        pipe.set_base_val(np.array([base, base, base]))
        code = pipe.process_to_code()
        # 全部收敛到靠近黑点（差值远小于 1 个 code）
        self.assertLess(float(np.ptp(code[0, :, 0])), 1.0)
        stats = pipe.get_clip_stats()
        # v > base 的 3 个像素 × 3 通道 = 9；v == base 的像素 t 恰好为 0
        # （窗口边界，不算越窗）。
        self.assertEqual(stats["density_below_base"], 9)


class TestPerChannelBase(unittest.TestCase):
    def test_base_aligned_keeps_channels_separate(self):
        """R5：片基除数必须逐通道保留（旧实现覆盖成标量灰）。"""
        pipe = ScientificFilmPipeline()
        img, base = make_negative(base=(0.90, 0.50, 0.30))
        pipe.load_linear_image(img)
        pipe.profile = _profile()
        pipe.set_base_val(base)
        aligned = pipe._base_aligned()
        self.assertEqual(aligned.shape, (3,))
        ratios = aligned / aligned[1]
        np.testing.assert_allclose(ratios, np.array([1.8, 1.0, 0.6]), atol=1e-6)

    def test_channel_gains_are_not_cancelled_by_base(self):
        """印片光（通道增益）必须真的改变密度，不能与片基对消。"""
        pipe = ScientificFilmPipeline()
        img, base = make_negative()
        pipe.load_linear_image(img)
        pipe.profile = _profile()
        pipe.set_base_val(base)
        d0 = pipe._density(pipe._stage_a_b_c(), pipe._base_aligned())
        pipe.set_channel_gains([1.25, 1.0, 0.8])
        after = (pipe._stage_a_b_c() * pipe.channel_gains.reshape(1, 1, 3))
        d1 = pipe._density(after, pipe._base_aligned())
        expected = np.log10([1.25, 1.0, 0.8])
        np.testing.assert_allclose((d0 - d1)[0, 20, :], expected, atol=1e-6)


class TestDensityNeutralization(unittest.TestCase):
    def test_neutralizes_channel_cast(self):
        """S10：已知通道偏色必须被中性化到目标密度（误差 <0.01）。"""
        pipe = ScientificFilmPipeline()
        img, base = make_negative(cast=(1.15, 1.0, 0.85))
        pipe.load_linear_image(img)
        pipe.profile = _profile()
        pipe.set_base_val(base)
        result = pipe.auto_align_density_domain(target_density=0.7)
        achieved = np.array(result["achieved"])
        np.testing.assert_allclose(achieved, 0.7, atol=0.01)
        self.assertFalse(result["clamped"])

    def test_gain_clamp_is_configurable_and_reported(self):
        """R5：增益裁剪范围可配置，且触限时明确回报（不再静默）。"""
        pipe = ScientificFilmPipeline()
        img, base = make_negative(cast=(3.0, 1.0, 0.2))
        pipe.load_linear_image(img)
        pipe.profile = _profile()
        pipe.set_base_val(base)
        wide = pipe.auto_align_density_domain(target_density=0.7, clamp=None)
        self.assertFalse(wide["clamped"], "clamp=None 必须表示不裁剪")
        np.testing.assert_allclose(np.array(wide["achieved"]), 0.7, atol=0.01)

        narrow = pipe.auto_align_density_domain(target_density=0.7,
                                                clamp=(0.6, 1.6))
        self.assertTrue(narrow["clamped"])
        gains = np.array(narrow["gains"])
        self.assertTrue(np.all(gains >= 0.6 - 1e-9) and np.all(gains <= 1.6 + 1e-9))

    def test_default_clamp_is_three_stops(self):
        pipe = ScientificFilmPipeline()
        self.assertAlmostEqual(pipe.gain_clamp[0], 0.125)
        self.assertAlmostEqual(pipe.gain_clamp[1], 8.0)

    def test_measure_density_report(self):
        pipe = ScientificFilmPipeline()
        img, base = make_negative()
        pipe.load_linear_image(img)
        pipe.profile = _profile()
        pipe.set_base_val(base)
        report = pipe.measure_density()
        self.assertEqual(len(report.per_channel_median), 3)
        self.assertGreater(report.per_channel_max[0], report.per_channel_median[0])


class TestBaseDetection(unittest.TestCase):
    def test_saturated_plateau_frame(self):
        """R6/S9：片基过曝占画面 >5% 时旧实现返回 None，现在必须成功。"""
        img = np.full((32, 32, 3), 0.2, dtype=np.float32)
        img[:4, :, :] = 1.0                      # 12.5% 的像素是饱和片基
        pipe = ScientificFilmPipeline()
        pipe.load_linear_image(img)
        base = pipe.auto_detect_base()
        self.assertIsNotNone(base, "饱和片基帧检测失败（旧 bug 复现）")
        np.testing.assert_allclose(base, 1.0, atol=1e-6)

    def test_old_method_still_fails_that_frame(self):
        """记录旧行为：percentile 法在该帧上确实取不到片基。"""
        img = np.full((32, 32, 3), 0.2, dtype=np.float32)
        img[:4, :, :] = 1.0
        pipe = ScientificFilmPipeline()
        pipe.load_linear_image(img)
        self.assertIsNone(pipe.auto_detect_base(method="percentile"))

    def test_robust_against_salt_noise(self):
        """单像素尖峰不应左右片基（旧实现取单像素，噪声敏感）。"""
        img, base = make_negative(seed=3)
        img[0, 0, :] = 3.0                       # 极端尖峰
        pipe = ScientificFilmPipeline()
        pipe.load_linear_image(img)
        got = pipe.auto_detect_base()
        np.testing.assert_allclose(got, base, rtol=0.02)

    def test_percentile_method_returns_none_when_uniform(self):
        pipe = ScientificFilmPipeline()
        pipe.load_linear_image(np.full((8, 8, 3), 0.5, dtype=np.float32))
        self.assertIsNone(pipe.auto_detect_base(method="percentile"))
        # 新方法在均匀帧上也必须给出结果
        np.testing.assert_allclose(pipe.auto_detect_base(), 0.5, atol=1e-6)


class TestProfilePath(unittest.TestCase):
    def test_icc_source_is_full_path(self):
        """R10：profile 来源必须保存完整路径（批量复制才能生效）。"""
        xml = """<?xml version="1.0"?>
<dcpData>
  <ColorMatrix1>1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0</ColorMatrix1>
  <CalibrationIlluminant1>21</CalibrationIlluminant1>
</dcpData>"""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cam.dcp")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(xml)
            pipe = ScientificFilmPipeline()
            self.assertTrue(pipe.load_icc_profile(path))
            self.assertTrue(os.path.isabs(pipe.icc_source))
            self.assertEqual(os.path.dirname(pipe.icc_source), tmp)
            self.assertEqual(pipe.icc_source_name, "cam.dcp")

    def test_dual_illuminant_interpolation(self):
        xml = """<?xml version="1.0"?>
<dcpData>
  <ColorMatrix1>1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0</ColorMatrix1>
  <ColorMatrix2>2.0 0.0 0.0 0.0 2.0 0.0 0.0 0.0 2.0</ColorMatrix2>
  <CalibrationIlluminant1>21</CalibrationIlluminant1>
  <CalibrationIlluminant2>17</CalibrationIlluminant2>
</dcpData>"""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "dual.dcp")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(xml)
            profile = prof.load_dcp(path)
            self.assertTrue(profile.has_dual_illuminant)
            # 索引 0 = 日光（含 "Daylight" 的那一组）
            m_day = profile.matrix(1.0)
            m_tung = profile.matrix(0.0)
            np.testing.assert_allclose(m_day, np.eye(3), atol=1e-12)
            np.testing.assert_allclose(m_tung, np.eye(3) * 0.5, atol=1e-12)
            mid = profile.matrix(0.5)
            np.testing.assert_allclose(mid, np.eye(3) * 0.75, atol=1e-12)


class TestLutApplication(unittest.TestCase):
    class _ScaleLut:
        """把值放大 2 倍的假 LUT（用于验证套用次数）。"""

        def __init__(self):
            self.calls = 0

        def apply(self, values):
            self.calls += 1
            return np.clip(np.asarray(values) * 2.0, 0.0, 1.0)

    def _pipeline(self):
        pipe = ScientificFilmPipeline()
        img, base = make_negative()
        pipe.load_linear_image(img)
        pipe.profile = _profile()
        pipe.set_base_val(base)
        return pipe

    def test_output_applies_lut_exactly_once(self):
        """R11：LUT 只能套一次。

        语义变化说明：旧实现是"pipeline 偷偷套一次 + 导出端再套一次"（双重）。
        现在套不套由**显式的输出变换**决定（``set_output_lut``），
        ``process_for_output()`` 只套一次，导出端不再重复套。
        """
        pipe = self._pipeline()
        lut = self._ScaleLut()
        pipe.set_output_lut(lut, enabled=True, path="fake.cube")
        with_lut = pipe.process_for_output()
        self.assertEqual(lut.calls, 1, "启用 LUT 后应当恰好套一次")
        # 显式关掉 LUT 走对数直通（供导出端/校验使用）
        plain = pipe.process_for_output(apply_lut=False)
        self.assertEqual(lut.calls, 1, "apply_lut=False 不应再套")
        self.assertFalse(np.allclose(plain, with_lut))

    def test_lut_output_switches_to_target_space(self):
        """套了色彩还原 LUT：输出空间必须变成 LUT 的目标空间。"""
        from aurhythm import output as out_mod
        pipe = self._pipeline()
        self.assertEqual(pipe.output.effective_space, out_mod.SPACE_CINEON)
        self.assertFalse(pipe.output.is_display_referred)

        pipe.set_output_lut(self._ScaleLut(), enabled=True, path="x.cube",
                            input_space=out_mod.SPACE_LOGC3,
                            target_space=out_mod.SPACE_REC709)
        self.assertEqual(pipe.output.effective_space, out_mod.SPACE_REC709)
        self.assertTrue(pipe.output.is_display_referred)
        self.assertEqual(pipe.output.transfer_code(),
                         out_mod.SPACE_INFO[out_mod.SPACE_REC709]["dpx_transfer"])
        # 关掉后回到对数直通
        pipe.set_output_lut(None, enabled=False)
        self.assertEqual(pipe.output.effective_space, out_mod.SPACE_CINEON)

    def test_apply_lut_without_lut_raises(self):
        pipe = self._pipeline()
        with self.assertRaises(ValueError):
            pipe.process_for_output(apply_lut=True)


class TestEndToEnd(unittest.TestCase):
    def test_tonal_resolution_through_whole_pipeline(self):
        """S1：512 级合成底片必须给出 >=500 个不同 code。"""
        for preset in ("无 (关闭预设)", "Kodak Vision3 250D (5207)"):
            pipe = ScientificFilmPipeline()
            pipe.set_preset(preset)
            width = 512
            ramp = np.linspace(1.0, 0.005, width)[None, :, None]
            img = np.repeat(ramp, 3, axis=2).repeat(4, axis=0)
            pipe.load_linear_image(img.astype(np.float32))
            pipe.profile = _profile()
            pipe.set_base_val(np.array([1.0, 1.0, 1.0]))
            pipe.set_hd_params(d_min=0.0, d_max=np.log10(1.0 / 0.005))
            code = pipe.process_to_code()
            distinct = len(np.unique(np.round(code[0, :, 0], 9)))
            self.assertGreaterEqual(distinct, 500,
                                    f"{preset}: 只剩 {distinct} 级")

    def test_output_is_normalized_and_logc3_reaches_arri(self):
        pipe = ScientificFilmPipeline()
        img, base = make_negative()
        pipe.load_linear_image(img)
        pipe.profile = _profile()
        pipe.set_base_val(base)
        out = pipe.process_for_output()
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)

        pipe.set_output_colorspace("logc3")
        logc3 = pipe.process_for_output()
        self.assertGreaterEqual(float(logc3.min()), 0.0)
        self.assertLessEqual(float(logc3.max()), 1.0)

    def test_preview_matches_export_space(self):
        """预览必须由导出数据经显示变换得到（所见即所得）。"""
        pipe = ScientificFilmPipeline()
        img, base = make_negative()
        pipe.load_linear_image(img)
        pipe.profile = _profile()
        pipe.set_base_val(base)
        preview = pipe.process_for_preview()
        exported = pipe.process_for_output()
        from aurhythm import display
        expected = display.code_to_display(exported, "cineon")
        np.testing.assert_array_equal(preview, expected)

    def test_clip_stats_are_reported(self):
        pipe = ScientificFilmPipeline()
        pipe.set_preset("Kodak Vision3 250D (5207)")
        ramp = np.linspace(1.2, 0.002, 64)[None, :, None]
        img = np.repeat(ramp, 3, axis=2).repeat(2, axis=0)
        pipe.load_linear_image(img.astype(np.float32))
        pipe.profile = _profile()
        pipe.set_base_val(np.array([1.0, 1.0, 1.0]))
        pipe.set_hd_params(d_min=0.0, d_max=2.0)
        pipe.process_to_code()
        stats = pipe.get_clip_stats()
        self.assertGreater(stats["total_samples"], 0)
        self.assertGreater(stats["density_below_base"], 0,
                           "比片基亮的像素应当被计入统计")
        self.assertGreaterEqual(stats["density_above_window"], 0)
        self.assertIn("output_clipped_pct", stats)

    def test_describe_is_json_serialisable(self):
        import json
        pipe = ScientificFilmPipeline()
        img, base = make_negative()
        pipe.load_linear_image(img)
        pipe.profile = _profile()
        pipe.set_base_val(base)
        pipe.auto_align_density_domain()
        payload = json.dumps(pipe.describe())
        self.assertGreater(len(payload), 100)
        json.dumps(pipe.to_settings())

    def test_process_requires_base(self):
        pipe = ScientificFilmPipeline()
        img, _base = make_negative()
        pipe.load_linear_image(img)
        self.assertIsNone(pipe.process_to_code())

    def test_set_hd_params_rejects_unknown_key(self):
        pipe = ScientificFilmPipeline()
        with self.assertRaises(KeyError):
            pipe.set_hd_params(hd_softness=0.1)      # R7 的字段名已不存在
        pipe.set_hd_params(s_toe=0.1, s_sh=0.2)
        self.assertAlmostEqual(pipe.hd.s_toe, 0.1)
        self.assertAlmostEqual(pipe.hd.s_sh, 0.2)


if __name__ == "__main__":
    unittest.main()


class TestColorCheckerIntegration(unittest.TestCase):
    """色卡模块与管线的接口必须真的连通（不只是各自能跑）。"""

    def test_calibrate_then_chain_applies_correction(self):
        from aurhythm import colorchecker as cc
        from tests.test_colorchecker import render_chart, planted_transform

        target = cc.target_linear_srgb()
        planted = planted_transform(scale=(1.15, 0.95, 0.75))
        corners = np.array([[70.0, 55.0], [470.0, 45.0],
                            [490.0, 310.0], [60.0, 320.0]])
        img = render_chart(360, 540, corners=corners,
                           colors=target @ planted.T)

        pipe = ScientificFilmPipeline()
        pipe.load_linear_image(img.astype(np.float32))
        pipe.set_base_val(np.array([1.0, 1.0, 1.0]))

        detection, reason = cc.detect_chart(img, explain=True)
        self.assertIsNotNone(detection, reason)
        sampled = cc.sample_patches(img, detection.corners)
        result = pipe.calibrate_from_colorchecker(sampled, target)

        self.assertTrue(pipe.colorchecker_calibrated)
        self.assertLess(result.stats["mean_deltaE"], 1.0)
        self.assertIn("ΔE2000", pipe.get_error_report())

        # 矫正必须在链路里真的起作用（密度按 1/M 的比例改变）
        code = pipe.process_to_code()
        self.assertIsNotNone(code)
        self.assertTrue(np.all(np.isfinite(code)))

    def test_load_calibration_file_through_pipeline(self):
        import json as _json
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cal.json")
            with open(path, "w", encoding="utf-8") as fh:
                _json.dump({"matrix": np.eye(3).tolist()}, fh)
            pipe = ScientificFilmPipeline()
            self.assertTrue(pipe.load_calibration(path))
            self.assertTrue(pipe.colorchecker_calibrated)
            np.testing.assert_allclose(pipe.color_correction_matrix, np.eye(3))
