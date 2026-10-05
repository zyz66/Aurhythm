"""C-41 物理正向模型与"有画面"测试负片的测试。

核心断言：**用本工具自己的管线去还原合成负片，必须能对回真值**。
这是这些测试素材唯一有意义的验收标准 —— 否则给的测试图毫无价值。
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import numpy as np
from dataclasses import replace

import tests  # noqa: F401
from aurhythm import cli, colorimetry as cm, filmsim, io_read, io_write
from aurhythm import params as pm
from aurhythm import synth
from aurhythm.constants import (DYE_EXTINCTION_C41, FILM_PRESETS,
                                ORANGE_MASK_DENSITY_C41)
from aurhythm.pipeline import ScientificFilmPipeline

TEST_PRESET = "测试：C-41 合成负片（橙色罩+串扰）"


def _lab(rgb8):
    """8-bit sRGB → CIELAB（D65）。delta_e_2000 只接受 Lab。"""
    xyz = cm.srgb_to_xyz(np.asarray(rgb8, dtype=np.float64) / 255.0)
    return cm.xyz_to_lab(xyz)


class TestFilmPhysics(unittest.TestCase):
    def test_matching_matrix_inverts_the_generator(self):
        """配套预设的解串扰矩阵必须精确反掉生成侧的染料串扰。"""
        matrix = FILM_PRESETS[TEST_PRESET].matrix_inv
        product = DYE_EXTINCTION_C41.T @ matrix
        np.testing.assert_allclose(product, np.eye(3), atol=1e-12)

    def test_base_is_orange_and_matches_the_mask(self):
        """片基必须是橙色（蓝>绿>红），且与橙色罩密度一致。"""
        stock = filmsim.DEFAULT_STOCK
        base = stock.base_transmission()
        self.assertGreater(base[0], base[1])
        self.assertGreater(base[1], base[2])
        expected = np.power(10.0, -(np.asarray(ORANGE_MASK_DENSITY_C41)
                                    + stock.fog_density))
        np.testing.assert_allclose(base / base.max(), expected / expected.max(),
                                   atol=1e-12)

    def test_brighter_scene_gives_denser_negative(self):
        """物理极性：场景越亮 → 染料越多 → 负片越密（不能反）。"""
        scene, base = synth.make_negative(64, 96, noise=0.0)
        result = filmsim.expose(synth.make_scene(96, 64, seed=3))
        negative = result["negative"]
        density = -np.log10(np.clip(negative, 1e-9, None)
                            / result["base_rgb"].reshape(1, 1, 3))
        correlation = np.corrcoef(result["scene"].reshape(-1),
                                  density.reshape(-1))[0, 1]
        self.assertGreater(correlation, 0.5, "场景亮度与密度必须正相关")
        self.assertGreaterEqual(float(negative.min()), 0.0)
        self.assertLessEqual(float(negative.max()), 1.0)

    def test_no_pixel_is_brighter_than_the_film_base(self):
        """负片里不可能有比未曝光片基更亮的东西（片基已是最透射）。"""
        result = synth.make_film_negative(160, 120, seed=5)
        base = result["base_rgb"]
        self.assertLessEqual(float(result["negative"].max()),
                             float(base.max()) + 1e-9)

    def test_crosstalk_leaks_between_channels(self):
        """串扰必须真的存在：单通道场景会污染另外两个通道的密度。"""
        scene = np.zeros((16, 16, 3))
        scene[..., 0] = 0.5                     # 只有红光
        result = filmsim.expose(scene)
        density = -np.log10(np.clip(result["negative"], 1e-9, None)
                            / result["base_rgb"].reshape(1, 1, 3))
        mean = density.reshape(-1, 3).mean(axis=0)
        self.assertGreater(mean[0], 0.2, "主吸收通道必须有密度")
        self.assertGreater(mean[1], 0.02, "绿通道应当被青染料的不希望吸收污染")
        self.assertGreater(mean[2], 0.005, "蓝通道应当被污染")

    def test_grain_changes_the_image(self):
        clean = synth.make_film_negative(96, 64, seed=1, grain=0.0)
        noisy = synth.make_film_negative(96, 64, seed=1, grain=0.05)
        self.assertFalse(np.allclose(clean["negative"], noisy["negative"]))


class TestSceneLooksLikeAPicture(unittest.TestCase):
    def test_scene_has_content_not_a_ramp(self):
        scene = synth.make_scene(240, 160, seed=7)
        self.assertEqual(scene.shape, (160, 240, 3))
        # 不是渐变条：不同区域应当明显不同（天空/山/地/植被）
        sky = scene[10:30, 100:140].mean(axis=(0, 1))
        foliage = scene[150:, :60].mean(axis=(0, 1))
        self.assertGreater(float(np.abs(sky - foliage).max()), 0.1,
                           "天空与前景植被必须差异明显")
        # 有真实的高光（太阳）用于测试肩部
        self.assertGreater(float(scene.max()), 1.0)
        # 天空不是黑的（曾经的 bug：算好了 sky 却忘了写进 scene）
        self.assertGreater(float(scene[5:25, 5:60].mean()), 0.05)

    def test_scene_contains_a_colorchecker_card(self):
        scene = synth.make_scene(600, 400, seed=7)
        card = scene[210:330, 190:410]
        # 卡上有 24 个色块 → 颜色种类应当很多
        quantized = (card.reshape(-1, 3) * 8).astype(int)
        unique = {tuple(row) for row in quantized}
        self.assertGreater(len(unique), 12, "测试卡上应当有多个不同色块")

    def test_border_is_unexposed_film_base(self):
        """片基边框必须存在，否则「片基采样」无从下手。"""
        result = synth.make_film_negative(200, 300, seed=7, border=0.05)
        negative = result["negative"]
        base = result["base_rgb"]
        # 注意：assert_allclose 不广播形状，要显式 broadcast_to
        np.testing.assert_allclose(
            negative[:5], np.broadcast_to(base, negative[:5].shape), atol=1e-6)
        np.testing.assert_allclose(
            negative[:, -5:], np.broadcast_to(base, negative[:, -5:].shape),
            atol=1e-6)


class TestPipelineRecoversTheScene(unittest.TestCase):
    """最有意义的验收：管线还原合成负片，与真值比色差。"""

    @classmethod
    def setUpClass(cls):
        cls.width, cls.height = 700, 480
        result = synth.make_film_negative(cls.width, cls.height, seed=7,
                                          border=0.025)
        cls.result = result
        cls.pipe = ScientificFilmPipeline()
        cls.pipe.load_linear_image(result["negative"].astype(np.float32))
        cls.pipe.set_base_val(cls.pipe.auto_detect_base())
        cls.pipe.set_preset(TEST_PRESET)
        report = cls.pipe.measure_density()
        pm.apply_param(cls.pipe, "hd_d_min",
                       max(0.0, min(report.percentile_low)))
        pm.apply_param(cls.pipe, "hd_d_max",
                       max(0.3, max(report.percentile_high)))

    def test_auto_detected_base_matches_truth(self):
        detected = np.asarray(self.pipe.base_val_rgb)
        np.testing.assert_allclose(detected, self.result["base_rgb"],
                                   rtol=2e-3, atol=2e-3)

    def test_preset_matrix_recovers_layer_density_exactly(self):
        """配套预设必须能从通道密度精确解回三层染料密度。

        这是「解串扰」这一步最干净的判据：去掉橙色罩后用预设矩阵反解，
        应当与生成侧的真值逐像素一致（不加颗粒时是精确的）。
        """
        result = synth.make_film_negative(120, 160, seed=5, grain=0.0)
        matrix = FILM_PRESETS[TEST_PRESET].matrix_inv
        above_mask = (result["channel_density"]
                      - result["stock"].total_mask.reshape(1, 1, 3))
        recovered = above_mask.reshape(-1, 3) @ matrix.T
        truth = result["layer_density"].reshape(-1, 3)
        np.testing.assert_allclose(recovered, truth, atol=1e-9)

    def test_colorchecker_patches_recover_within_small_delta_e(self):
        """色卡 24 块的 ΔE2000 必须很小 —— 这是"测试图有意义"的判据。"""
        from aurhythm import display  # noqa: F401

        recovered = self.pipe.process_for_preview()
        truth = filmsim.to_srgb8(self.result["scene"])
        self.assertEqual(recovered.shape, truth.shape)

        # 卡片几何（与 synth.make_scene 中的排布一致）
        width, height = self.width, self.height
        x0, x1 = int(width * 0.30), int(width * 0.70)
        y0, y1 = int(height * 0.50), int(height * 0.88)
        pad_x = int((x1 - x0) * 0.05)
        pad_y = int((y1 - y0) * 0.05)
        ix0, ix1 = x0 + pad_x, x1 - pad_x
        iy0, iy1 = y0 + pad_y, y1 - pad_y
        wedge = int((iy1 - iy0) * 0.17)
        grid_y1 = iy1 - wedge - pad_y
        cell_w = (ix1 - ix0) / 6.0
        cell_h = (grid_y1 - iy0) / 4.0

        deltas = []
        for index in range(24):
            row, col = divmod(index, 6)
            ry0 = int(iy0 + row * cell_h + cell_h * 0.3)
            ry1 = int(iy0 + row * cell_h + cell_h * 0.7)
            rx0 = int(ix0 + col * cell_w + cell_w * 0.3)
            rx1 = int(ix0 + col * cell_w + cell_w * 0.7)
            a = truth[ry0:ry1, rx0:rx1].reshape(-1, 3).mean(axis=0).reshape(1, 1, 3)
            b = recovered[ry0:ry1, rx0:rx1].reshape(-1, 3).mean(axis=0).reshape(1, 1, 3)
            deltas.append(float(cm.delta_e_2000(_lab(a), _lab(b)).mean()))
        deltas = np.asarray(deltas)
        # CIEDE2000 必须喂 **Lab**。曾经误把 XYZ（0..1）喂进去，结果 ΔE 恒在
        # 0.1 量级 —— 那样的断言对任何两张图都会通过（等于没有测试）。
        self.assertLess(float(np.median(deltas)), 8.0,
                        f"色卡 ΔE2000 中位应 <8，实际 {np.median(deltas):.2f}")

    def test_delta_e_helper_is_called_with_lab(self):
        """防止再犯"把 XYZ 当 Lab 喂进 CIEDE2000"的错。"""
        black = np.zeros((4, 4, 3), np.uint8)
        white = np.full((4, 4, 3), 255, np.uint8)
        self.assertAlmostEqual(float(cm.delta_e_2000(_lab(black),
                                                     _lab(black)).max()), 0.0,
                              places=9)
        self.assertGreater(float(cm.delta_e_2000(_lab(black),
                                                 _lab(white)).mean()), 50.0)


class TestPresetWindowCoversTheStock(unittest.TestCase):
    """回归：配套预设自带的密度窗口必须覆盖本片种的真实密度范围。

    用户实测发现的 bug：预设手打的 ``d_max=2.60`` 低于真实值 3.08，
    于是 23.8% 的像素（天空/高光）被顶到窗口外，高光层次全丢。
    """

    def test_window_is_derived_not_hand_written(self):
        from aurhythm.constants import (C41_DENSITY_WINDOW,
                                        DYE_EXTINCTION_C41, LAYER_DMAX_C41)
        expected = float(np.max(DYE_EXTINCTION_C41.T
                                @ np.asarray(LAYER_DMAX_C41)))
        self.assertAlmostEqual(C41_DENSITY_WINDOW[0], 0.0)
        self.assertAlmostEqual(C41_DENSITY_WINDOW[1], expected, places=9)
        hd = FILM_PRESETS[TEST_PRESET].hd
        self.assertAlmostEqual(hd.d_min, C41_DENSITY_WINDOW[0])
        self.assertAlmostEqual(hd.d_max, C41_DENSITY_WINDOW[1])

    def test_preset_alone_does_not_clip_highlights(self):
        """只套预设（不做任何手动测量）也不该丢高光。"""
        result = synth.make_film_negative(400, 560, seed=7, border=0.025)
        pipe = ScientificFilmPipeline()
        pipe.load_linear_image(result["negative"].astype(np.float32))
        pipe.set_base_val(pipe.auto_detect_base())
        pipe.set_preset(TEST_PRESET)
        pipe.process_for_preview(count_stats=True)
        stats = pipe.get_clip_stats()
        over = stats["density_above_window"] / max(stats["total_samples"], 1)
        self.assertLess(over, 0.01,
                        f"预设自带的窗口不该把 {over * 100:.1f}% 的像素顶出窗外")
        below = stats["density_below_base"] / max(stats["total_samples"], 1)
        self.assertLess(below, 0.01)

    def test_missing_a_correct_window_loses_highlights(self):
        """对照：窗口上限**明显偏小**时必须复现出大量越窗（否则测试无约束力）。

        注意阈值要按实测密度范围来定：胶片参数修正为物理值之后，负片密度
        范围从 3.1D 降到约 2.2D，原来写死的 2.60 已经不再"偏小"了。
        """
        result = synth.make_film_negative(400, 560, seed=7, border=0.025)
        pipe = ScientificFilmPipeline()
        pipe.load_linear_image(result["negative"].astype(np.float32))
        pipe.set_base_val(pipe.auto_detect_base())
        pipe.set_preset(TEST_PRESET)
        report = pipe.measure_density()
        measured_max = max(report.percentile_high)
        # 把上限压到实测值的 ~55%
        pm.apply_param(pipe, "hd_d_max", max(0.3, measured_max * 0.55))
        pipe.process_for_preview(count_stats=True)
        stats = pipe.get_clip_stats()
        over = stats["density_above_window"] / max(stats["total_samples"], 1)
        self.assertGreater(over, 0.15,
                           f"上限压到实测的 55% 应当大量越窗，实际 {over:.1%}")


class TestPerChannelNeutrality(unittest.TestCase):
    """F4 验收：**中性必须中性**（用户实测抱怨的"所有预设最后都偏色"）。

    注意目标函数的界定：这里**不**用"还原场景的 ΔE"做判据 —— 胶片肩部
    已丢失高光信息，那在数学上不可能（实测 RMS 达 120 code）。可解且有意义
    的判据是：中性内容在三个通道上得到一致的响应。
    """

    def test_preset_carries_per_channel_calibration(self):
        hd = FILM_PRESETS[TEST_PRESET].hd
        self.assertTrue(hd.has_per_channel, "配套预设必须带逐通道校准")
        self.assertEqual(len(hd.d_max_rgb), 3)
        self.assertEqual(len(hd.a_rgb), 3)

    def test_derivation_reproduces_the_baked_values(self):
        """预设里的逐通道值是**推导**出来的；改了生成器就必须同步更新。"""
        from aurhythm import filmsim
        derived = filmsim.derive_per_channel_calibration()
        hd = FILM_PRESETS[TEST_PRESET].hd
        for key in ("a_rgb", "d_min_rgb", "d_max_rgb"):
            np.testing.assert_allclose(getattr(hd, key), derived[key],
                                       rtol=2e-3, atol=2e-3,
                                       err_msg=f"{key} 与推导结果不一致（漂移了）")

    def test_neutral_ladder_spread_is_small(self):
        """无彩色阶梯：三通道 code 的离散度必须很小。"""
        from aurhythm import filmsim
        from aurhythm.tonemap import TransferFunction
        hd = FILM_PRESETS[TEST_PRESET].hd
        _x, layers = filmsim.neutral_ladder()
        codes, _outside = TransferFunction(hd=hd).density_to_code(layers)
        spread = np.std(codes, axis=1)
        self.assertLess(float(spread.mean()), 2.0,
                        f"中性阶梯平均离散度应 <2 code，实际 {spread.mean():.2f}")

    def test_neutral_regions_are_neutral_in_the_pipeline(self):
        """端到端：合成图上多个中性区域的 B/R 必须接近真值。"""
        result = synth.make_film_negative(700, 480, seed=7, border=0.025)
        pipe = ScientificFilmPipeline()
        pipe.load_linear_image(result["negative"].astype(np.float32))
        pipe.set_base_val(pipe.auto_detect_base())
        pipe.set_preset(TEST_PRESET)
        recovered = pipe.process_for_preview()
        truth = filmsim.to_srgb8(result["scene"])
        height, width = recovered.shape[:2]
        regions = ((0.805, 0.855, 0.33, 0.40),
                   (0.805, 0.855, 0.44, 0.52),
                   (0.53, 0.56, 0.33, 0.66))
        for y0, y1, x0, x1 in regions:
            got = recovered[int(height * y0):int(height * y1),
                            int(width * x0):int(width * x1)]
            ref = truth[int(height * y0):int(height * y1),
                        int(width * x0):int(width * x1)]
            got_ratio = float(got[..., 2].mean()) / max(float(got[..., 0].mean()), 1e-6)
            ref_ratio = float(ref[..., 2].mean()) / max(float(ref[..., 0].mean()), 1e-6)
            self.assertAlmostEqual(
                got_ratio, ref_ratio, delta=0.025,
                msg=f"区域({y0},{x0}) 偏色：B/R={got_ratio:.3f}，真值 {ref_ratio:.3f}")

    def test_shared_curve_is_measurably_worse(self):
        """回归护栏：退回共享曲线必须能被判为更差（否则测试没有约束力）。"""
        from aurhythm import filmsim
        from aurhythm.tonemap import TransferFunction
        _x, layers = filmsim.neutral_ladder()
        good = FILM_PRESETS[TEST_PRESET].hd
        bad = replace(hd_without_channels(good), a_rgb=None, d_min_rgb=None,
                      d_max_rgb=None)
        good_spread = float(np.std(
            TransferFunction(hd=good).density_to_code(layers)[0], axis=1).mean())
        bad_spread = float(np.std(
            TransferFunction(hd=bad).density_to_code(layers)[0], axis=1).mean())
        self.assertGreater(bad_spread, good_spread * 3.0,
                           f"共享曲线应当明显更差：{bad_spread:.2f} vs {good_spread:.2f}")


def hd_without_channels(hd):
    from dataclasses import replace as _replace
    return _replace(hd)


class TestAllPresetsLeaveNeutralNeutral(unittest.TestCase):
    """用户实测抱怨的核心：「其他胶片预设最后都偏色」。

    这里用**与真值的偏差**（不是绝对中性）判定：合成图上有三块已知中性区，
    真值本身带一点点色偏（b*≈+0.2），所以只有"与真值的偏差"才是有意义的量。

    说明：预设里的逐通道数值目前是**临时值**（按片种窗口推导，见 constants
    的注释），等按柯达官方 RGB 吸收曲线反推后替换；届时可收紧这个阈值。
    """

    #: 临时阈值（CIEDE2000 之外的简单色差：Δ(a*,b*) 的模）
    MAX_DEVIATION = {None: 3.0, "Kodak Vision3 250D (5207)": 3.0,
                     "Kodak Vision3 500T (5219)": 3.0}

    @classmethod
    def setUpClass(cls):
        cls.width, cls.height = 700, 480
        result = synth.make_film_negative(cls.width, cls.height, seed=7,
                                          border=0.025)
        cls.negative = result["negative"].astype(np.float32)
        cls.truth = filmsim.to_srgb8(result["scene"])

    def _deviation(self, preset):
        from aurhythm.constants import DEFAULT_PRESET

        pipe = ScientificFilmPipeline()
        pipe.load_linear_image(self.negative)
        pipe.set_base_val(pipe.auto_detect_base())
        if preset != DEFAULT_PRESET:
            pipe.set_preset(preset)
        report = pipe.measure_density()
        pm.apply_param(pipe, "hd_d_min", max(0.0, min(report.percentile_low)))
        pm.apply_param(pipe, "hd_d_max", max(0.3, max(report.percentile_high)))
        recovered = pipe.process_for_preview()
        height, width = recovered.shape[:2]
        regions = ((0.805, 0.86, 0.33, 0.40), (0.805, 0.86, 0.44, 0.52),
                   (0.53, 0.56, 0.33, 0.66))
        worst = 0.0
        for y0, y1, x0, x1 in regions:
            got = _lab_of(recovered[int(height * y0):int(height * y1),
                                    int(width * x0):int(width * x1)])
            ref = _lab_of(self.truth[int(height * y0):int(height * y1),
                                     int(width * x0):int(width * x1)])
            worst = max(worst, float(np.hypot(got[1] - ref[1], got[2] - ref[2])))
        return worst

    def test_every_preset_keeps_neutral_neutral(self):
        from aurhythm.constants import PRESET_ORDER
        failures = []
        for name in PRESET_ORDER:
            if name.startswith("无"):
                continue                     # 不套预设本来就该偏色（未做串扰校正）
            deviation = self._deviation(name)
            if deviation > self.MAX_DEVIATION.get(name, 3.0):
                failures.append(f"{name}={deviation:.2f}")
        self.assertFalse(failures, "以下预设让中性区偏色了：" + ", ".join(failures))

    def test_no_preset_is_much_worse_than_a_derived_one(self):
        """对照：不套预设必须明显更差，否则这些预设没有存在意义。"""
        none_dev = self._deviation("无 (关闭预设)")
        best = min(self._deviation(n) for n in
                   ("Kodak Portra 400", "Kodak Gold 200", TEST_PRESET))
        self.assertGreater(none_dev, best * 3.0,
                           f"不套预设 {none_dev:.2f} 应当明显差于最佳预设 {best:.2f}")


def _lab_of(rgb8):
    """8-bit sRGB → CIELAB。"""
    xyz = cm.srgb_to_xyz(np.asarray(rgb8, dtype=np.float64) / 255.0)
    return cm.xyz_to_lab(xyz).reshape(-1, 3).mean(axis=0)


class TestSynthCliScene(unittest.TestCase):
    def test_cli_scene_writes_negative_truth_and_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "scene.tif")
            code = cli.main(["synth", "--kind", "scene", "--width", "240",
                             "--height", "160", "--out", out, "--json"])
            self.assertEqual(code, cli.EXIT_OK)
            self.assertTrue(os.path.exists(out))
            truth = os.path.join(tmp, "scene_truth.png")
            self.assertTrue(os.path.exists(truth), "必须同时给对照正片")

            array, _description = io_write.read_tiff_baseline(out)
            self.assertEqual(array.shape, (160, 240, 3))
            self.assertGreater(int(array.max()), 30000)   # 16-bit 且不是全黑

            sidecar = os.path.join(tmp, "scene.json")
            self.assertTrue(os.path.exists(sidecar))
            with open(sidecar, encoding="utf-8") as fh:
                payload = json.load(fh)
            self.assertEqual(payload["kind"], "scene")
            self.assertEqual(payload["matching_preset"], TEST_PRESET)
            np.testing.assert_allclose(payload["base_rgb"],
                                       self.__class__._base_of(out), atol=1e-3)

    @staticmethod
    def _base_of(path):
        linear, _info = io_read.load_any(path)
        pipe = ScientificFilmPipeline()
        pipe.load_linear_image(linear)
        return pipe.auto_detect_base()

    def test_negative_kind_still_works(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "n.tif")
            self.assertEqual(cli.main(["synth", "--out", path, "--kind",
                                       "negative", "--width", "64",
                                       "--height", "48", "--json"]),
                             cli.EXIT_OK)
            self.assertTrue(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
