"""非线性校正测试：H-D/S 曲线、逆解、色调控制、单调样条。

对应验收标准 S1（512 级不失真）、S2（逆解往返）、S3（参数扫掠单调性）。

术语：曲线的自变量是**归一化输入密度** ``t ∈ [0,1]``，因变量是归一化
输出密度 ``t'``；``characteristic`` 做 t→t'，``inverse_characteristic``
做 t'→t。
"""

from __future__ import annotations

import math
import unittest

import numpy as np

import tests  # noqa: F401
from aurhythm.constants import CINEON, FILM_PRESETS, HDParams, PRESET_ORDER
from aurhythm import tonemap as tm


def _sweep_params(seed: int = 7, count: int = 400) -> list:
    """在参数空间里撒点（含极端值），用于单调性与可逆性扫掠。"""
    rng = np.random.default_rng(seed)
    combos = [HDParams()]
    for name in PRESET_ORDER[1:]:
        combos.append(FILM_PRESETS[name].hd)
    a_choices = [0.2, 0.5, 1.0, 2.0, 6.0]
    s_choices = [1e-4, 0.01, 0.2, 0.6, 1.0]
    for _ in range(count):
        combos.append(HDParams(
            a=float(rng.choice(a_choices)),
            x_mid=float(rng.uniform(-3.0, 3.0)),
            s_toe=float(rng.choice(s_choices)),
            s_sh=float(rng.choice(s_choices)),
            d_min=float(rng.uniform(0.0, 0.4)),
            d_max=float(rng.uniform(1.0, 4.0)),
        ))
    return combos


def _three_channel(values) -> np.ndarray:
    """把 1-D 数据扩成 (N,3)：CDL/曲线阶段要求通道维存在。"""
    return np.repeat(np.asarray(values, dtype=np.float64)[:, None], 3, axis=1)


class TestNumerics(unittest.TestCase):
    def test_softplus_stability(self):
        z = np.array([-1e6, -800.0, -100.0, 0.0, 1.0, 800.0, 1e6])
        out = tm.softplus(z)
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertTrue(np.all(out[:3] < 1e-40), "小 z 时 softplus 应趋 0")
        self.assertAlmostEqual(float(out[3]), math.log(2.0), places=12)
        self.assertAlmostEqual(float(out[4]), math.log1p(math.e), places=12)
        np.testing.assert_allclose(out[5:], z[5:], rtol=1e-12)

    def test_sigmoid_stability(self):
        z = np.array([-1e6, -800.0, 0.0, 800.0, 1e6])
        out = tm.sigmoid(z)
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertEqual(float(out[0]), 0.0)
        self.assertEqual(float(out[-1]), 1.0)
        self.assertAlmostEqual(float(out[2]), 0.5, places=15)

    def test_log_expm1_stability(self):
        t = np.array([1e-9, 1e-3, 1.0, 5.0, 100.0, 5000.0])
        out = tm.log_expm1(t)
        self.assertTrue(np.all(np.isfinite(out)))
        np.testing.assert_allclose(np.exp(out) + 1.0, np.exp(t), rtol=1e-9)

    def test_sanitize_is_idempotent(self):
        for p in _sweep_params(count=60):
            once = tm.sanitize(p)
            self.assertEqual(tm.sanitize(once), once)

    def test_sanitize_repairs_garbage(self):
        p = tm.sanitize(HDParams(a=0.0, x_mid=float("nan"), s_toe=-1.0,
                                 s_sh=99.0, d_min=2.0, d_max=1.0))
        self.assertGreaterEqual(p.a, tm.MIN_SLOPE)
        self.assertGreaterEqual(p.s_toe, tm.MIN_SOFTNESS)
        self.assertLessEqual(p.s_sh, tm.MAX_SOFTNESS)
        self.assertGreater(p.d_max, p.d_min)
        self.assertTrue(math.isfinite(p.x_mid))


class TestCharacteristic(unittest.TestCase):
    def test_midpoint_is_exactly_half(self):
        """t'(x_mid) = 0.5 精确成立（闭式归一化的结果）。"""
        for p in _sweep_params(count=120):
            prep = tm.prepare(p)
            got = float(tm.characteristic(prep.x_mid, prep))
            self.assertAlmostEqual(got, 0.5, places=9,
                                   msg=f"参数 {p} 的中点偏移 {got - 0.5}")

    def test_mid_slope_is_exactly_a(self):
        """中段斜率精确等于 a（闭式归一化的结果）。"""
        for p in _sweep_params(count=120):
            prep = tm.prepare(p)
            got = float(tm.characteristic_derivative(prep.x_mid, prep))
            self.assertAlmostEqual(got, p.a, places=8,
                                   msg=f"参数 {p} 的中段斜率 {got} != {p.a}")

    def test_identity_is_exact(self):
        """a=1、x_mid=0.5、s→0 必须精确退化为恒等映射。"""
        tf = tm.TransferFunction(hd=HDParams(a=1.0, x_mid=0.5,
                                             s_toe=tm.MIN_SOFTNESS,
                                             s_sh=tm.MIN_SOFTNESS))
        t = np.linspace(0.0, 1.0, 257)
        # 软度取到数值下限（1e-6）时残余偏差约 1e-6 个 t 单位（≈4e-4 code），
        # 真正精确的恒等只在 s→0 的极限成立。
        np.testing.assert_allclose(tm.characteristic(t, tf.hd), t, atol=1e-5)
        self.assertAlmostEqual(tf.effective_slope(), 1.0, places=4)

    def test_asymptotes(self):
        prep = tm.prepare(HDParams(s_toe=0.3, s_sh=0.3))
        self.assertLess(float(tm.characteristic(prep.x_min, prep)), 1e-6)
        self.assertGreater(float(tm.characteristic(prep.x_max, prep)), 1.0 - 1e-9)

    def test_strict_monotonic_over_parameter_grid(self):
        """S3：400+ 组参数（含极端值）下必须严格单调。"""
        combos = _sweep_params(count=400)
        checked = 0
        for p in combos:
            prep = tm.prepare(p)
            x_grid = np.linspace(prep.x_floor, prep.x_ceil, 257)
            d = tm.characteristic_derivative(x_grid, prep)
            if not np.all(d > 0.0):
                bad = x_grid[d <= 0.0]
                self.fail(f"参数 {p} 在 x={bad[:3]} 处导数 <= 0")
            t = tm.characteristic(x_grid, prep)
            if not np.all(np.diff(t) > 0.0):
                self.fail(f"参数 {p} 非严格单调")
            checked += 1
        self.assertGreater(checked, 400)

    def test_softness_primarily_controls_its_own_end(self):
        """趾/肩软度的主效应必须在各自那一端（x 空间、固定输出电平）。

        如实记录：两个软度并非完全独立，增大会轻微整体位移曲线；
        但主效应的比值必须 >2.5。
        """
        base = tm.prepare(HDParams(a=1.0, x_mid=0.5, s_toe=0.06, s_sh=0.06))
        soft_toe = tm.prepare(HDParams(a=1.0, x_mid=0.5, s_toe=0.45, s_sh=0.06))
        soft_sh = tm.prepare(HDParams(a=1.0, x_mid=0.5, s_toe=0.06, s_sh=0.45))

        def x_of(prep, levels):
            return tm.inverse_characteristic(np.array(levels), prep)[0]

        low, mid, high = 0.05, 0.5, 0.95
        xb = x_of(base, [low, mid, high])
        xt = x_of(soft_toe, [low, mid, high])
        xs = x_of(soft_sh, [low, mid, high])
        d_toe = np.abs(xt - xb)
        d_sh = np.abs(xs - xb)
        self.assertGreater(d_toe[0], 2.5 * d_toe[1],
                           f"趾部软度对低端影响不足: {d_toe}")
        self.assertGreater(d_sh[2], 2.5 * d_sh[1],
                           f"肩部软度对高端影响不足: {d_sh}")
        # 高端对趾部软度的敏感性低于低端
        self.assertGreater(d_toe[0], d_toe[2])

    def test_chord_slope_tracks_contrast(self):
        """弦斜率随 a 单调增，且 a=1 时接近 1。"""
        slopes = []
        for a in (0.6, 1.0, 1.5, 2.5):
            tf = tm.TransferFunction(hd=HDParams(a=a, x_mid=0.5,
                                                 s_toe=0.05, s_sh=0.05))
            slopes.append(tf.effective_slope())
        self.assertTrue(np.all(np.diff(slopes) > 0.0), slopes)
        self.assertLess(abs(slopes[1] - 1.0), 5e-3, slopes)

    def test_array_and_scalar_agree(self):
        prep = tm.prepare(HDParams())
        xs = np.linspace(-2.0, 4.0, 33)
        arr = tm.characteristic(xs, prep)
        for i, x in enumerate(xs):
            self.assertAlmostEqual(float(arr[i]),
                                   float(tm.characteristic(x, prep)), places=12)


class TestInverse(unittest.TestCase):
    def test_roundtrip_forward_then_inverse(self):
        """S2：x → t → x 误差 <= 2e-5。

        极低反差（a=0.2）叠加宽软度时，端部 dt/dx 极小，x 空间的往返误差
        会上升到 ~1e-5 —— 这是参数本身的条件数，不是求解器缺陷；
        t 空间的往返（下一个测试）对所有参数都必须精确。
        """
        checked = 0
        for p in _sweep_params(count=200):
            prep = tm.prepare(p)
            # 端点本身正好是 t_floor / t_ceil（会被判为夹取），内缩一个极小量
            eps = 1e-9 * (prep.x_ceil - prep.x_floor)
            x = np.linspace(prep.x_floor + eps, prep.x_ceil - eps, 512)
            t = tm.characteristic(x, prep)
            back, clamped = tm.inverse_characteristic(t, prep)
            self.assertFalse(bool(np.any(clamped)), f"参数 {p} 出现意外夹取")
            np.testing.assert_allclose(back, x, atol=2e-5,
                                       err_msg=f"参数 {p} 前向-逆向不一致")
            checked += 1
        self.assertGreater(checked, 200)

    def test_roundtrip_on_full_output_level_range(self):
        """输出电平 0.02..0.98 全域往返（多数参数下 dt/dx 都很小）。"""
        for p in _sweep_params(count=80):
            prep = tm.prepare(p)
            levels = np.linspace(0.02, 0.98, 128)
            x, clamped = tm.inverse_characteristic(levels, prep)
            self.assertFalse(bool(np.any(clamped)), f"参数 {p} 意外夹取")
            np.testing.assert_allclose(tm.characteristic(x, prep), levels,
                                       atol=1e-9, err_msg=f"参数 {p}")

    def test_roundtrip_inverse_then_forward(self):
        """t' → t 必须精确（逆解的真正契约）。"""
        for p in _sweep_params(count=200):
            prep = tm.prepare(p)
            eps = 1e-9 * (prep.x_ceil - prep.x_floor)
            x_grid = np.linspace(prep.x_floor + eps, prep.x_ceil - eps, 400)
            t = tm.characteristic(x_grid, prep)
            x_back, clamped = tm.inverse_characteristic(t, prep)
            self.assertFalse(bool(np.any(clamped)), f"参数 {p} 出现意外夹取")
            np.testing.assert_allclose(tm.characteristic(x_back, prep), t,
                                       atol=1e-9,
                                       err_msg=f"参数 {p} 逆解不精确")

    def test_levels_roundtrip(self):
        """输出电平 0.02..0.98 必须都能精确反解回输入电平。"""
        for p in _sweep_params(count=80):
            prep = tm.prepare(p)
            levels = np.linspace(0.02, 0.98, 128)
            x, clamped = tm.inverse_characteristic(levels, prep)
            self.assertFalse(bool(np.any(clamped)))
            np.testing.assert_allclose(tm.characteristic(x, prep), levels,
                                       atol=1e-9, err_msg=f"参数 {p}")

    def test_endpoints_clamp(self):
        """t'=0 与 t'=1 是渐近线，必须被夹取到有限端点。"""
        prep = tm.prepare(HDParams())
        x, clamped = tm.inverse_characteristic(np.array([0.0, 1.0]), prep)
        self.assertTrue(bool(np.all(clamped)))
        self.assertTrue(np.all(np.isfinite(x)))
        self.assertAlmostEqual(float(x[0]), prep.x_floor, places=9)
        self.assertAlmostEqual(float(x[1]), prep.x_ceil, places=9)

    def test_monotone_inverse(self):
        prep = tm.prepare(HDParams())
        t = np.linspace(0.0, 1.0, 2048)
        x, _ = tm.inverse_characteristic(t, prep)
        self.assertTrue(np.all(np.diff(x) > 0.0))

    def test_convergence_is_fast(self):
        """中段初值很准：6 次迭代即达到完整解的精度。"""
        prep = tm.prepare(HDParams())
        t = np.linspace(0.02, 0.98, 64)
        x6, _ = tm.inverse_characteristic(t, prep, max_iter=6)
        x_full, _ = tm.inverse_characteristic(t, prep)
        np.testing.assert_allclose(x6, x_full, atol=1e-6)


class TestToneControls(unittest.TestCase):
    def test_identity(self):
        self.assertTrue(tm.ToneControls().is_identity())

    def test_exposure_one_stop_is_150_codes(self):
        tc = tm.ToneControls(exposure_ev=1.0)
        code = np.array([0.0, 100.0, 500.0])
        np.testing.assert_allclose(tc.apply_code(code),
                                   code + CINEON.codes_per_stop, atol=1e-9)
        np.testing.assert_allclose(tc.invert_code(tc.apply_code(code)), code,
                                   atol=1e-9)

    def test_contrast_about_pivot(self):
        pivot = tm.PIVOT_18_GREY_CODE
        tc = tm.ToneControls(contrast=2.0)
        self.assertAlmostEqual(float(tc.apply_code(np.array(pivot))), pivot,
                               places=9)
        high = pivot + 100.0
        self.assertAlmostEqual(float(tc.apply_code(np.array(high))),
                               pivot + 200.0, places=9)

    def test_tone_roundtrip(self):
        tc = tm.ToneControls(exposure_ev=-1.5, contrast=1.7)
        code = np.linspace(0.0, 1023.0, 128)
        np.testing.assert_allclose(tc.invert_code(tc.apply_code(code)), code,
                                   atol=1e-9)

    def test_pivot_default_is_18_grey(self):
        self.assertAlmostEqual(tm.PIVOT_18_GREY_CODE, 335.515, places=3)

    def test_cdl_identity_and_effect(self):
        n = np.linspace(0.0, 1.0, 32)
        rgb = np.stack([n, n, n], axis=-1)
        np.testing.assert_allclose(tm.ToneControls().apply_cdl(rgb), rgb,
                                   atol=1e-15)
        brighter = tm.ToneControls(gain=(1.2, 1.2, 1.2)).apply_cdl(rgb)
        self.assertTrue(np.all(brighter >= rgb - 1e-15))
        lifted = tm.ToneControls(lift=(0.1, 0.1, 0.1)).apply_cdl(rgb)
        self.assertAlmostEqual(float(lifted[0, 0]), 0.1, places=12)
        gamma = tm.ToneControls(gamma=(2.0, 2.0, 2.0)).apply_cdl(rgb)
        self.assertLess(float(gamma[16, 0]), float(rgb[16, 0]))

    def test_cdl_per_channel(self):
        rgb = np.full((1, 3), 0.5)
        out = tm.ToneControls(gain=(2.0, 1.0, 1.0)).apply_cdl(rgb)
        self.assertAlmostEqual(float(out[0, 0]), 1.0, places=12)
        self.assertAlmostEqual(float(out[0, 1]), 0.5, places=12)

    def test_cdl_rejects_bad_shape(self):
        with self.assertRaises(ValueError):
            tm.ToneControls(gain=(1.1, 1.0, 1.0)).apply_cdl(np.zeros(64))

    def test_dict_roundtrip(self):
        tc = tm.ToneControls(exposure_ev=0.3, contrast=1.2, master_gamma=1.1,
                             lift=(0.01, 0.02, 0.03), gain=(1.1, 1.0, 0.9))
        self.assertEqual(tm.ToneControls.from_dict(tc.to_dict()), tc)

    def test_v5_legacy_field_migration(self):
        legacy = {"exposure_ev": 0.5, "contrast": 1.1, "pivot_loge": 0.48}
        tc = tm.ToneControls.from_dict(legacy)
        self.assertAlmostEqual(tc.exposure_ev, 0.5)
        self.assertAlmostEqual(tc.pivot_code, tm.PIVOT_18_GREY_CODE)


class TestCurves(unittest.TestCase):
    def test_identity(self):
        c = tm.MonotoneCurve.identity()
        self.assertTrue(c.is_identity())
        x = np.linspace(0.0, 1.0, 64)
        np.testing.assert_allclose(c.apply(x), x, atol=1e-12)

    def test_five_point_identity_is_recognised(self):
        """回归：5 点共线恒等曲线必须被判为恒等（曾导致数组塌成 (3,)）。"""
        c = tm.MonotoneCurve([(0.0, 0.0), (0.25, 0.25), (0.5, 0.5),
                              (0.75, 0.75), (1.0, 1.0)])
        self.assertTrue(c.is_identity())

    def test_monotone_for_arbitrary_points(self):
        rng = np.random.default_rng(3)
        for _ in range(200):
            xs = np.sort(rng.uniform(0.0, 1.0, 6))
            xs[0], xs[-1] = 0.0, 1.0
            ys = np.sort(rng.uniform(0.0, 1.0, 6))
            curve = tm.MonotoneCurve(list(zip(xs, ys)))
            out = curve.apply(np.linspace(0.0, 1.0, 512))
            self.assertTrue(np.all(np.diff(out) >= -1e-12), "单调样条出现下降")
            self.assertGreaterEqual(float(out.min()), 0.0)
            self.assertLessEqual(float(out.max()), 1.0)

    def test_interpolates_nodes(self):
        pts = [(0.0, 0.05), (0.4, 0.3), (0.7, 0.9), (1.0, 0.95)]
        curve = tm.MonotoneCurve(pts)
        for x, y in pts:
            self.assertAlmostEqual(float(curve.apply(np.array(x))), y, places=9)

    def test_list_roundtrip(self):
        curve = tm.MonotoneCurve([(0.0, 0.0), (0.5, 0.7), (1.0, 1.0)])
        again = tm.MonotoneCurve.from_list(curve.to_list())
        np.testing.assert_allclose(again.apply(np.linspace(0, 1, 32)),
                                   curve.apply(np.linspace(0, 1, 32)), atol=1e-12)

    def test_curveset_per_channel(self):
        cs = tm.CurveSet(red=tm.MonotoneCurve([(0.0, 0.1), (1.0, 1.0)]))
        self.assertFalse(cs.is_identity())
        out = cs.apply(np.full((4, 3), 0.5))
        self.assertAlmostEqual(float(out[0, 0]), 0.55, places=9)
        self.assertAlmostEqual(float(out[0, 1]), 0.5, places=12)

    def test_curveset_rejects_bad_shape(self):
        with self.assertRaises(ValueError):
            tm.CurveSet(red=tm.MonotoneCurve([(0.0, 0.1), (1.0, 1.0)])).apply(
                np.zeros(64))


class TestTransferFunction(unittest.TestCase):
    def test_identity_preset_is_linear_in_density(self):
        """无预设：code = 95 + 590·t（标准参考黑/白窗口）。"""
        tf = tm.TransferFunction(hd=FILM_PRESETS["无 (关闭预设)"].hd)
        code, _ = tf.density_to_code(_three_channel(np.linspace(0.0, 1.0, 9)))
        expected = CINEON.black_code + tm.REFERENCE_SPAN * np.linspace(0, 1, 9)
        # 软度取数值下限时残余 ≈4e-4 code（见 test_identity_is_exact）
        np.testing.assert_allclose(code[:, 0], expected, atol=1e-3)

    def test_tonal_resolution(self):
        """S1：512 级输入密度必须得到 >=500 个不同 code（原先只有 3 级）。"""
        for name in PRESET_ORDER:
            tf = tm.TransferFunction(hd=FILM_PRESETS[name].hd)
            p = tf.params
            density = _three_channel(np.linspace(p.d_min, p.d_max, 512))
            code, _ = tf.density_to_code(density)
            distinct = len(np.unique(np.round(code[:, 0], 9)))
            self.assertGreaterEqual(distinct, 500,
                                    f"{name}: 512 级输入只剩 {distinct} 级输出")
            self.assertTrue(np.all(np.diff(code[:, 0]) > 0.0),
                            f"{name}: 输出非严格单调")

    def test_tonal_resolution_with_tones_and_curves(self):
        tf = tm.TransferFunction(
            hd=FILM_PRESETS["Kodak Vision3 250D (5207)"].hd,
            tone=tm.ToneControls(exposure_ev=0.5, contrast=1.3,
                                 gain=(1.1, 1.0, 0.95)),
            curves=tm.CurveSet(master=tm.MonotoneCurve([(0.0, 0.0), (0.5, 0.55),
                                                        (1.0, 1.0)])),
        )
        p = tf.params
        density = _three_channel(np.linspace(p.d_min, p.d_max, 512))
        code, _ = tf.density_to_code(density)
        self.assertGreaterEqual(len(np.unique(np.round(code[:, 0], 9))), 500)

    def test_output_fits_reference_window(self):
        """auto_fit 下整个输入窗口必须落在 95..685 之内。"""
        for name in PRESET_ORDER:
            tf = tm.TransferFunction(hd=FILM_PRESETS[name].hd)
            p = tf.params
            density = _three_channel(np.linspace(p.d_min, p.d_max, 64))
            code, _ = tf.density_to_code(density)
            self.assertGreaterEqual(float(code.min()), CINEON.black_code - 1e-6)
            self.assertLessEqual(float(code.max()), CINEON.white_code + 1e-6)

    def test_standard_encoding_when_auto_fit_disabled(self):
        """auto_fit=False 时按 Cineon 标准 500 codes/密度单位编码。"""
        tf = tm.TransferFunction(hd=FILM_PRESETS["无 (关闭预设)"].hd,
                                 auto_fit=False)
        p = tf.params
        density = _three_channel(np.linspace(p.d_min, p.d_max, 5))
        code, _ = tf.density_to_code(density)
        span = float(code[-1, 0] - code[0, 0])
        # 恒等曲线的残余（软度数值下限）使跨度略小于理论值
        self.assertAlmostEqual(span, 500.0 * p.density_range, delta=0.1)

    def test_out_of_window_is_reported_not_clipped(self):
        tf = tm.TransferFunction(hd=FILM_PRESETS["通用负片 (默认)"].hd)
        p = tf.params
        # 逐通道参数启用时，三个通道的窗口**各不相同**，所以每个通道都要取
        # 自己的边界（用同一个数会让某些通道恰好落到窗外）。
        if p.has_per_channel:
            lo = np.asarray(p.d_min_rgb, dtype=np.float64)
            hi = np.asarray(p.d_max_rgb, dtype=np.float64)
        else:
            lo = np.full(3, float(p.d_min))
            hi = np.full(3, float(p.d_max))
        density = np.stack([lo - 0.5, lo, hi, hi + 0.5])
        code, outside = tf.density_to_code(density)
        self.assertTrue(bool(outside[0].all()) and bool(outside[3].all()))
        self.assertFalse(bool(outside[1].any()) or bool(outside[2].any()))
        # 计数单位是通道样本（2 个越窗像素 × 3 通道）
        self.assertEqual(tf.last_out_of_window, 6)
        # 越窗不应造成逆序（趾/肩是平滑压缩）
        self.assertTrue(np.all(np.diff(code[:, 0]) > 0.0))

    def test_density_to_code_is_monotone(self):
        tf = tm.TransferFunction(hd=FILM_PRESETS["通用负片 (默认)"].hd)
        p = tf.params
        density = _three_channel(np.linspace(p.d_min - 0.3, p.d_max + 0.3, 2048))
        code, _ = tf.density_to_code(density)
        self.assertTrue(np.all(np.isfinite(code)))
        self.assertTrue(np.all(np.diff(code[:, 0]) >= 0.0), "出现逆序")
        # 片基以下（t<0）不是硬裁，而是沿趾部平滑收敛到黑点：
        # 必须严格位于 [黑点, 窗口黑点] 之间，并随密度下降单调趋近黑点。
        d = np.linspace(p.d_min - 0.3, p.d_max + 0.3, 2048)
        below = d < p.d_min
        self.assertGreaterEqual(float(code[below, 0].min()), CINEON.black_code - 0.01)
        self.assertLessEqual(float(code[below, 0].max()),
                             float(code[~below, 0][0]) + 1e-9)
        self.assertTrue(np.all(np.diff(code[below, 0]) > 0.0))
        # 远低于片基时收敛到黑点
        far, _ = tf.density_to_code(_three_channel(np.array([p.d_min - 5.0])))
        self.assertLess(abs(float(far[0, 0]) - CINEON.black_code), 0.01)

    def test_code_to_density_roundtrip(self):
        tf = tm.TransferFunction(hd=FILM_PRESETS["Kodak Gold 200"].hd)
        codes = np.linspace(CINEON.black_code + 5.0,
                            CINEON.white_code - 5.0, 64)
        # 该预设带逐通道参数，code→密度必须按通道给出
        density = tf.code_to_density(_three_channel(codes))[:, 0]
        self.assertTrue(np.all(np.diff(density) > 0.0), "code→密度必须单调")
        back, _ = tf.density_to_code(_three_channel(density))
        np.testing.assert_allclose(back[:, 0], codes, atol=1e-6)

    def test_code_to_density_with_tones_and_curves(self):
        tf = tm.TransferFunction(
            hd=FILM_PRESETS["通用负片 (默认)"].hd,
            tone=tm.ToneControls(exposure_ev=0.3, contrast=1.2),
        )
        codes = np.linspace(200.0, 600.0, 32)
        density = tf.code_to_density(_three_channel(codes))[:, 0]
        back, _ = tf.density_to_code(_three_channel(density))
        np.testing.assert_allclose(back[:, 0], codes, atol=1e-4)

    def test_exposure_control_shifts_code(self):
        hd = FILM_PRESETS["通用负片 (默认)"].hd
        base = tm.TransferFunction(hd=hd)
        lifted = tm.TransferFunction(hd=hd, tone=tm.ToneControls(exposure_ev=1.0))
        p = base.params
        density = _three_channel(np.linspace(p.d_min, p.d_max, 64))
        c0, _ = base.density_to_code(density)
        c1, _ = lifted.density_to_code(density)
        np.testing.assert_allclose(c1 - c0, CINEON.codes_per_stop, atol=1e-6)

    def test_contrast_control_increases_slope(self):
        hd = FILM_PRESETS["通用负片 (默认)"].hd
        p = tm.sanitize(hd)
        density = _three_channel(np.linspace(p.d_min, p.d_max, 256))
        plain, _ = tm.TransferFunction(hd=hd).density_to_code(density)
        punchy, _ = tm.TransferFunction(
            hd=hd, tone=tm.ToneControls(contrast=1.5)).density_to_code(density)
        slope_plain = float(np.ptp(plain[:, 0]))
        slope_punchy = float(np.ptp(punchy[:, 0]))
        self.assertGreater(slope_punchy, slope_plain)

    def test_logc3_output_reaches_arri_reference(self):
        """code 335.515（18% 中灰）必须给出 ARRI LogC3 的 0.391007。"""
        tf = tm.TransferFunction()
        self.assertAlmostEqual(float(tf.code_to_logc3(
            tm.PIVOT_18_GREY_CODE)), 0.391007, places=6)
        self.assertAlmostEqual(float(tf.code_to_exposure(CINEON.white_code)),
                               0.90, places=12)

    def test_requires_channel_axis_when_per_channel_stages_active(self):
        tf = tm.TransferFunction(hd=FILM_PRESETS["通用负片 (默认)"].hd,
                                 tone=tm.ToneControls(gain=(1.1, 1.0, 1.0)))
        with self.assertRaises(ValueError):
            tf.density_to_code(np.linspace(0.2, 3.0, 16))

    def test_describe_and_sample(self):
        tf = tm.TransferFunction()
        d = tf.describe()
        for key in ("hd", "tone", "curves", "codes_per_unit",
                    "effective_slope", "auto_fit"):
            self.assertIn(key, d)
        density, code = tf.sample(64)
        self.assertEqual(density.shape, (64,))
        self.assertEqual(code.shape, (64, 3))
        self.assertTrue(np.all(np.diff(code[:, 0]) > 0.0))


if __name__ == "__main__":
    unittest.main()
