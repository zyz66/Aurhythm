"""LUT 测试：R2（list 相减必崩）、R3（R/B 轴交换）、1D/3D、.cube 往返、
以及「校准变换」.cube 烘焙。

R11（双重套用）在 tests/test_pipeline.py 里验证。
"""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

import tests  # noqa: F401
from aurhythm.lut import CubeLUT, LutError


def write_cube(path, size, entries, header=None, domain_min=None,
               domain_max=None, one_d=False):
    lines = list(header or [])
    lines.append(f"LUT_1D_SIZE {size}" if one_d else f"LUT_3D_SIZE {size}")
    if domain_min:
        lines.append("DOMAIN_MIN " + " ".join(str(v) for v in domain_min))
    if domain_max:
        lines.append("DOMAIN_MAX " + " ".join(str(v) for v in domain_max))
    for row in entries:
        lines.append(" ".join(str(v) for v in row))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def identity_cube_entries(size):
    """按 .cube 规范顺序（R 最快）生成恒等 LUT 的数据行。"""
    rows = []
    for b in range(size):
        for g in range(size):
            for r in range(size):
                rows.append([r / (size - 1), g / (size - 1), b / (size - 1)])
    return rows


class TestDefaultDomainNoLongerCrashes(unittest.TestCase):
    def test_apply_with_default_domain(self):
        """R2：默认 domain（过去是 list）下 apply 必须能跑。"""
        lut = CubeLUT()
        lut.size = 2
        lut.table = np.zeros((2, 2, 2, 3))
        out = lut.apply(np.zeros((1, 1, 3)))
        self.assertEqual(out.shape, (1, 1, 3))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_default_domain_is_ndarray(self):
        lut = CubeLUT()
        self.assertIsInstance(lut.domain_min, np.ndarray)
        self.assertIsInstance(lut.domain_max, np.ndarray)
        self.assertEqual((lut.domain_max - lut.domain_min).tolist(), [1.0, 1.0, 1.0])

    def test_list_domain_would_have_failed(self):
        """记录旧 bug：两个 Python list 相减必然 TypeError。"""
        with self.assertRaises(TypeError):
            [1.0, 1.0, 1.0] - [0.0, 0.0, 0.0]      # noqa: B018


class TestAxisOrder(unittest.TestCase):
    def test_identity_lut_preserves_rgb(self):
        """R3：恒等 LUT 必须原样返回（旧实现把 R 与 B 对调）。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "id.cube")
            write_cube(path, 3, identity_cube_entries(3))
            lut = CubeLUT()
            lut.load(path)
            probe = np.array([[[0.9, 0.1, 0.1], [0.1, 0.9, 0.1],
                               [0.1, 0.1, 0.9], [0.5, 0.25, 0.75]]])
            out = lut.apply(probe)
            np.testing.assert_allclose(out, probe, atol=1e-6)

    def test_table_layout_is_r_g_b(self):
        """规范布局必须是 table[r,g,b]。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "id.cube")
            write_cube(path, 2, identity_cube_entries(2))
            lut = CubeLUT()
            lut.load(path)
            # table[1,0,0] 应当是「只有 R 最大」→ 接近 (1,0,0)
            np.testing.assert_allclose(lut.table[1, 0, 0], [1.0, 0.0, 0.0], atol=1e-9)
            np.testing.assert_allclose(lut.table[0, 0, 1], [0.0, 0.0, 1.0], atol=1e-9)

    def test_save_load_roundtrip_preserves_axes(self):
        lut = CubeLUT.identity(5)
        rng = np.random.default_rng(0)
        lut.table = np.clip(lut.table * 0.8 + rng.random(lut.table.shape) * 0.1,
                            0.0, 1.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.cube")
            lut.save(path)
            again = CubeLUT()
            again.load(path)
            np.testing.assert_allclose(again.table, lut.table, atol=1e-5)


class TestParsing(unittest.TestCase):
    def test_title_and_domain(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "t.cube")
            write_cube(path, 2, identity_cube_entries(2),
                       header=['TITLE "my lut"'],
                       domain_min=[-0.1, 0.0, 0.0], domain_max=[1.1, 1.0, 1.0])
            lut = CubeLUT()
            lut.load(path)
            self.assertEqual(lut.title, "my lut")
            np.testing.assert_allclose(lut.domain_min, [-0.1, 0.0, 0.0])
            # 域扩展后，输入 1.1 应当达到 LUT 的最大值
            out = lut.apply(np.array([[[1.1, 1.0, 1.0]]]))
            np.testing.assert_allclose(out[0, 0], [1.0, 1.0, 1.0], atol=1e-6)

    def test_comments_and_blank_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "c.cube")
            write_cube(path, 2, identity_cube_entries(2),
                       header=["# comment", ""])
            lut = CubeLUT()
            lut.load(path)
            self.assertEqual(lut.size, 2)

    def test_missing_size_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.cube")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("0 0 0\n1 1 1\n")
            with self.assertRaises(LutError):
                CubeLUT().load(path)

    def test_insufficient_data_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "short.cube")
            write_cube(path, 3, [[0, 0, 0], [1, 1, 1]])
            with self.assertRaises(LutError):
                CubeLUT().load(path)

    def test_both_sizes_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "both.cube")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("LUT_3D_SIZE 2\nLUT_1D_SIZE 2\n0 0 0\n")
            with self.assertRaises(LutError):
                CubeLUT().load(path)

    def test_bad_domain_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "dom.cube")
            write_cube(path, 2, identity_cube_entries(2),
                       domain_min=[1.0, 0.0, 0.0], domain_max=[0.0, 1.0, 1.0])
            with self.assertRaises(LutError):
                CubeLUT().load(path)

    def test_missing_file_raises(self):
        with self.assertRaises(LutError):
            CubeLUT().load("/nonexistent/definitely-not-here.cube")


class TestOneDLut(unittest.TestCase):
    def test_1d_lut_applies_shared_curve(self):
        entries = [[i / 4, (i / 4) ** 2, 1.0 - i / 4] for i in range(5)]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "one.cube")
            write_cube(path, 5, entries, one_d=True)
            lut = CubeLUT()
            lut.load(path)
            self.assertTrue(lut.is_1d)
            out = lut.apply(np.array([[0.5, 0.5, 0.5]]))
            np.testing.assert_allclose(out[0, 0], 0.5, atol=1e-9)
            np.testing.assert_allclose(out[0, 1], 0.25, atol=1e-9)
            np.testing.assert_allclose(out[0, 2], 0.5, atol=1e-9)

    def test_1d_save_load_roundtrip(self):
        lut = CubeLUT()
        lut.size = 3
        lut.is_1d = True
        lut.table = np.array([[0.0, 0.0, 0.0], [0.3, 0.4, 0.5], [1.0, 1.0, 1.0]])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "one.cube")
            lut.save(path)
            again = CubeLUT()
            again.load(path)
            self.assertTrue(again.is_1d)
            np.testing.assert_allclose(again.table, lut.table, atol=1e-5)


class TestInterpolation(unittest.TestCase):
    def test_trilinear_midpoint(self):
        lut = CubeLUT.identity(3)
        # 把中心点改成 (0.2, 0.4, 0.6)，检查中点插值命中
        lut.table[1, 1, 1] = [0.2, 0.4, 0.6]
        out = lut.apply(np.array([[[0.5, 0.5, 0.5]]]))
        np.testing.assert_allclose(out[0, 0], [0.2, 0.4, 0.6], atol=1e-9)

    def test_clamps_outside_domain(self):
        lut = CubeLUT.identity(3)
        out = lut.apply(np.array([[-1.0, -1.0, -1.0], [2.0, 2.0, 2.0]]))
        np.testing.assert_allclose(out[0], [0.0, 0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(out[1], [1.0, 1.0, 1.0], atol=1e-9)

    def test_shape_preserved(self):
        lut = CubeLUT.identity(4)
        for shape in [(5, 7, 3), (13, 3), (3,)]:
            out = lut.apply(np.random.default_rng(0).random(shape))
            self.assertEqual(out.shape, shape)

    def test_empty_lut_is_noop(self):
        lut = CubeLUT()
        values = np.random.default_rng(0).random((4, 4, 3))
        np.testing.assert_allclose(lut.apply(values), values)


def _calibration_codes(pipe, density):
    """与 ``CubeLUT.from_pipeline(include_look=False)`` 同一路径的期望值。"""
    from aurhythm.tonemap import characteristic
    params = pipe.transfer.params
    density = np.asarray(density, dtype=np.float64)
    if params.has_per_channel:
        code = np.empty_like(density)
        for i in range(3):
            pc = params.channel(i)
            t = (density[..., i] - pc.d_min) / pc.density_range
            code[..., i] = pipe.transfer.encode_density(characteristic(t, pc))
    else:
        t = (density - params.d_min) / params.density_range
        code = pipe.transfer.encode_density(characteristic(t, params))
    return np.asarray(pipe.transfer.tone.apply_code(code), dtype=np.float64)


class TestCalibrationLut(unittest.TestCase):
    def _pipeline(self):
        from aurhythm.pipeline import ScientificFilmPipeline
        pipe = ScientificFilmPipeline()
        pipe.set_preset("Kodak Vision3 250D (5207)")
        return pipe

    def test_bakes_calibration_transform(self):
        """烘焙的 .cube 必须与管线自身的校准链一致（误差 <=1e-3）。"""
        pipe = self._pipeline()
        lut = CubeLUT.from_pipeline(pipe, size=17)
        transfer = pipe.transfer
        params = transfer.params
        axis = np.linspace(0.0, 1.0, 17)
        r, g, b = np.meshgrid(axis, axis, axis, indexing="ij")
        uv = np.stack([r, g, b], axis=-1).reshape(-1, 3)
        # 逐通道参数启用时，域是逐通道窗口（见 CubeLUT.from_pipeline 的说明）
        if params.has_per_channel:
            lo = np.asarray(params.d_min_rgb).reshape(1, 3)
            hi = np.asarray(params.d_max_rgb).reshape(1, 3)
        else:
            lo = np.full((1, 3), params.d_min)
            hi = np.full((1, 3), params.d_max)
        density = lo + uv * (hi - lo)
        expected, _ = transfer.density_to_code(density,
                                              count_out_of_window=False)
        expected = np.clip(expected, 0.0, 1023.0) / 1023.0
        got = lut.table.reshape(-1, 3)
        np.testing.assert_allclose(got, expected, atol=1e-3)

    def test_default_excludes_look(self):
        """默认烘焙必须**不含**调色层（CDL/曲线）。"""
        from aurhythm.tonemap import CurveSet, MonotoneCurve, ToneControls
        pipe = self._pipeline()
        pipe.set_tone_controls(ToneControls(gain=(1.5, 1.5, 1.5)))
        pipe.set_curves(CurveSet(master=MonotoneCurve([(0.0, 0.2), (1.0, 1.0)])))
        calibration = CubeLUT.from_pipeline(pipe, size=9, include_look=False)
        with_look = CubeLUT.from_pipeline(pipe, size=9, include_look=True)
        self.assertFalse(np.allclose(calibration.table, with_look.table))
        # 校准版应当等于「不含 CDL/曲线」的结果
        plain = self._pipeline()
        reference = CubeLUT.from_pipeline(plain, size=9)
        np.testing.assert_allclose(calibration.table, reference.table, atol=1e-9)

    def test_identity_preset_lut_is_linear_ramp(self):
        from aurhythm.constants import CINEON
        from aurhythm.pipeline import ScientificFilmPipeline
        pipe = ScientificFilmPipeline()
        pipe.set_preset("无 (关闭预设)")
        lut = CubeLUT.from_pipeline(pipe, size=5)
        expected = (CINEON.black_code + 590.0 * np.linspace(0, 1, 5)) / 1023.0
        # 沿 r 轴取 R 输出、沿 g 轴取 G 输出，都应当是同一条线性斜坡
        ramp_r = lut.table[:, 0, 0, 0]
        ramp_g = lut.table[0, :, 0, 1]
        ramp_b = lut.table[0, 0, :, 2]
        np.testing.assert_allclose(ramp_r, expected, atol=2e-3)
        np.testing.assert_allclose(ramp_g, expected, atol=2e-3)
        np.testing.assert_allclose(ramp_b, expected, atol=2e-3)

    def test_baked_lut_over_domain_range(self):
        """套用烘焙 LUT 后再套回管线的归一化导出，误差应很小。"""
        import numpy as np
        pipe = self._pipeline()
        lut = CubeLUT.from_pipeline(pipe, size=33)
        params = pipe.transfer.params
        uv = np.linspace(0.0, 1.0, 13)
        uv3 = np.repeat(uv[:, None], 3, axis=1)
        if params.has_per_channel:
            lo = np.asarray(params.d_min_rgb).reshape(1, 3)
            hi = np.asarray(params.d_max_rgb).reshape(1, 3)
        else:
            lo = np.full((1, 3), params.d_min)
            hi = np.full((1, 3), params.d_max)
        density = lo + uv3 * (hi - lo)
        direct = _calibration_codes(pipe, density)
        direct = np.clip(direct, 0.0, 1023.0) / 1023.0
        via_lut = lut.apply(uv3)
        np.testing.assert_allclose(via_lut, direct, atol=2e-3)


if __name__ == "__main__":
    unittest.main()
