"""输出变换测试：对数直通 vs 经色彩还原 LUT 到显示色彩空间。

直接验证用户提出的需求：套上 ARRI 那种 ``LogC3 → Rec.709`` 色彩还原 LUT 后，
导出**不再是 log 素材**，而是 Rec.709 —— 容器标注、DPX 字段、预览显示
全部随之改变。
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import numpy as np

import tests  # noqa: F401
from aurhythm import cli, display, io_write
from aurhythm import output as om
from aurhythm import synth
from aurhythm.lut import CubeLUT
from aurhythm.pipeline import ScientificFilmPipeline
from tests.test_pipeline import _profile


def logc3_to_rec709_lut(size: int = 17) -> CubeLUT:
    """合成一个"LogC3 → Rec.709"色彩还原 LUT（模拟 ARRI 官网那只）。

    逐通道、与通道无关：``out = bt709_encode(logc3_decode(in))``。
    """
    axis = np.linspace(0.0, 1.0, size)
    r, g, b = np.meshgrid(axis, axis, axis, indexing="ij")
    uv = np.stack([r, g, b], axis=-1)
    linear = display.bt709_decode  # 占位，实际用 LogC3 解码
    from aurhythm.constants import LOGC3
    values = display.bt709_encode(np.clip(LOGC3.decode(uv), 0.0, None))
    lut = CubeLUT()
    lut.size = size
    lut.title = "synthetic LogC3 to Rec.709"
    lut.table = np.clip(values, 0.0, 1.0)
    return lut


def _pipeline_with_image():
    image, base = synth.make_negative(24, 32, noise=0.0)
    pipe = ScientificFilmPipeline()
    pipe.load_linear_image(image)
    pipe.profile = _profile()
    pipe.set_base_val(base)
    pipe.set_output_colorspace("logc3")
    return pipe


class TestOutputTransform(unittest.TestCase):
    def test_defaults_are_log_passthrough(self):
        t = om.OutputTransform()
        self.assertEqual(t.mode, om.MODE_LOG)
        self.assertEqual(t.effective_space, om.SPACE_CINEON)
        self.assertFalse(t.is_display_referred)
        self.assertEqual(t.transfer_code(),
                         om.DPX_TRANSFER["printing_density"])
        self.assertIn("对数直通", t.label())

    def test_lut_mode_reports_target_space_and_dpx_codes(self):
        t = om.OutputTransform(mode=om.MODE_LUT,
                               lut_input_space=om.SPACE_LOGC3,
                               lut_target_space=om.SPACE_REC709)
        self.assertEqual(t.effective_space, om.SPACE_REC709)
        self.assertTrue(t.is_display_referred)
        self.assertEqual(t.transfer_code(), om.DPX_TRANSFER["itu_r_709"])
        self.assertEqual(t.colorimetric_code(),
                         om.DPX_COLORIMETRIC["itu_r_bt709"])
        self.assertIn("经 LUT", t.label())

    def test_manual_override_wins(self):
        t = om.OutputTransform(mode=om.MODE_LUT, dpx_transfer=3,
                               dpx_colorimetric=5)
        self.assertEqual(t.transfer_code(), 3)
        self.assertEqual(t.colorimetric_code(), 5)

    def test_sanitize_rejects_garbage(self):
        t = om.OutputTransform(mode="nope", log_space="xyz",
                               lut_target_space="???", range="bogus").sanitize()
        self.assertEqual(t.mode, om.MODE_LOG)
        self.assertEqual(t.log_space, om.SPACE_CINEON)
        self.assertEqual(t.lut_target_space, om.SPACE_REC709)
        self.assertEqual(t.range, om.RANGE_FULL)

    def test_dict_roundtrip(self):
        t = om.OutputTransform(mode=om.MODE_LUT, lut_target_space=om.SPACE_SRGB,
                               range=om.RANGE_LEGAL, lut_path="/tmp/a.cube")
        again = om.OutputTransform.from_dict(t.to_dict())
        self.assertEqual(again.mode, t.mode)
        self.assertEqual(again.lut_target_space, t.lut_target_space)
        self.assertEqual(again.range, t.range)
        self.assertEqual(again.lut_path, t.lut_path)


class TestRangeMapping(unittest.TestCase):
    def test_legal_bounds(self):
        self.assertEqual(om.legal_bounds(10), (64, 940))
        self.assertEqual(om.legal_bounds(12), (256, 3760))
        self.assertEqual(om.legal_bounds(16), (4096, 60160))
        with self.assertRaises(ValueError):
            om.legal_bounds(8)

    def test_full_and_legal_scaling(self):
        data = np.array([0.0, 0.5, 1.0])
        self.assertEqual(io_write.to_codes(data).tolist(), [0, 512, 1023])
        self.assertEqual(io_write.to_codes(data, "legal").tolist(),
                         [64, 502, 940])
        u16 = io_write.to_uint16(data, "legal")
        self.assertEqual(u16.tolist(), [4096, 32128, 60160])
        self.assertEqual(u16.dtype, np.uint16)

    def test_clipping(self):
        self.assertEqual(io_write.to_codes([-1.0, 2.0]).tolist(), [0, 1023])


class TestLutExportPath(unittest.TestCase):
    """套上色彩还原 LUT 之后的完整导出语义。"""

    def test_output_values_are_display_referred(self):
        pipe = _pipeline_with_image()
        log_values = pipe.process_for_output()
        lut = logc3_to_rec709_lut()
        pipe.set_output_lut(lut, enabled=True, path="x.cube",
                            input_space=om.SPACE_LOGC3,
                            target_space=om.SPACE_REC709)
        display_values = pipe.process_for_output()
        self.assertFalse(np.allclose(log_values, display_values))
        # 中灰附近：LogC3 的 0.391 应当映射到 Rec.709 的约 0.4~0.5
        probe = np.full((1, 1, 3), 0.391007)
        mapped = lut.apply(probe)[0, 0, 0]
        self.assertGreater(mapped, 0.35)
        self.assertLess(mapped, 0.55)

    def test_dpx_header_follows_target_space(self):
        pipe = _pipeline_with_image()
        lut = logc3_to_rec709_lut()
        pipe.set_output_lut(lut, enabled=True, path="x.cube",
                            target_space=om.SPACE_REC709)
        data = pipe.process_for_output()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.dpx")
            io_write.write_image(
                path, data, "dpx10", metadata=pipe.to_settings(),
                range_mode=pipe.output.effective_range,
                dpx_transfer=pipe.output.transfer_code(),
                dpx_colorimetric=pipe.output.colorimetric_code())
            _codes, header = io_write.read_dpx10(path)
            self.assertEqual(header["transfer"], om.DPX_TRANSFER["itu_r_709"])
            self.assertEqual(header["colorimetric"],
                             om.DPX_COLORIMETRIC["itu_r_bt709"])
            with open(os.path.splitext(path)[0] + ".json", encoding="utf-8") as fh:
                meta = json.load(fh)
            self.assertEqual(meta["dpx_transfer"], 6)
            self.assertEqual(meta["output"]["effective_space"], "rec709")
            self.assertTrue(meta["output"]["is_display_referred"])
            self.assertEqual(meta["output"]["lut_path"], "x.cube")

    def test_log_path_keeps_log_codes(self):
        """对数直通时 DPX 字段必须反映**实际编码**：
        LogC3 → logarithmic(3)，Cineon → printing density(1)。"""
        for space, expected in ((om.SPACE_LOGC3, om.DPX_TRANSFER["logarithmic"]),
                                (om.SPACE_CINEON,
                                 om.DPX_TRANSFER["printing_density"])):
            pipe = _pipeline_with_image()
            pipe.set_output_colorspace(space)
            data = pipe.process_for_output()
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, f"{space}.dpx")
                io_write.write_image(
                    path, data, "dpx10", metadata=pipe.to_settings(),
                    range_mode=pipe.output.effective_range,
                    dpx_transfer=pipe.output.transfer_code(),
                    dpx_colorimetric=pipe.output.colorimetric_code())
                _codes, header = io_write.read_dpx10(path)
                self.assertEqual(header["transfer"], expected, space)
                self.assertEqual(header["colorimetric"],
                                 om.DPX_COLORIMETRIC["user_defined"], space)

    def test_legal_range_limits_written_codes(self):
        pipe = _pipeline_with_image()
        pipe.set_output_lut(logc3_to_rec709_lut(), enabled=True, path="x.cube",
                            target_space=om.SPACE_REC709,
                            range_mode=om.RANGE_LEGAL)
        self.assertEqual(pipe.output.effective_range, om.RANGE_LEGAL)
        data = pipe.process_for_output()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "legal.dpx")
            io_write.write_image(path, data, "dpx10",
                                 range_mode=pipe.output.effective_range)
            codes, _header = io_write.read_dpx10(path)
            self.assertGreaterEqual(int(codes.min()), 64)
            self.assertLessEqual(int(codes.max()), 940)

    def test_tiff_metadata_records_transform(self):
        pipe = _pipeline_with_image()
        pipe.set_output_lut(logc3_to_rec709_lut(), enabled=True, path="x.cube",
                            target_space=om.SPACE_REC709)
        data = pipe.process_for_output()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.tif")
            io_write.write_tiff(path, data, 16, metadata=pipe.to_settings())
            _array, description = io_write.read_tiff_baseline(path)
            meta = json.loads(description)
            self.assertEqual(meta["output"]["effective_space"], "rec709")
            self.assertEqual(meta["output"]["mode"], "lut")


class TestPreviewWithLut(unittest.TestCase):
    def test_preview_shows_display_space_directly(self):
        """套 LUT 后预览必须按显示域显示，而不是再解一次 log。"""
        pipe = _pipeline_with_image()
        lut = logc3_to_rec709_lut()
        pipe.set_output_lut(lut, enabled=True, path="x.cube",
                            target_space=om.SPACE_REC709)
        from aurhythm.constants import CINEON
        code = pipe.process_to_code(count_stats=False)
        values = lut.apply(pipe._encode_log(code, om.SPACE_LOGC3))
        expected = display.code_to_display(values, om.SPACE_REC709)
        got = pipe.process_for_preview()
        np.testing.assert_array_equal(got, expected)

    def test_preview_differs_from_log_path(self):
        pipe = _pipeline_with_image()
        log_preview = pipe.process_for_preview()
        pipe.set_output_lut(logc3_to_rec709_lut(), enabled=True, path="x.cube",
                            target_space=om.SPACE_REC709)
        lut_preview = pipe.process_for_preview()
        self.assertFalse(np.array_equal(log_preview, lut_preview))

    def test_rec709_to_srgb_response_is_not_naive_passthrough(self):
        """Rec.709 与 sRGB 差一个 OETF：不能直接当 sRGB 显示。"""
        values = np.array([[0.1, 0.5, 0.9]])
        response = display.display_response(values, om.SPACE_REC709)
        self.assertFalse(np.allclose(response, values, atol=1e-3))
        # 线性数据则应当被 sRGB 编码（0.5 → ≈0.735）
        linear_response = display.display_response(np.array([[0.5]]),
                                                   om.SPACE_LINEAR)
        self.assertAlmostEqual(float(linear_response[0, 0]), 0.7354, places=3)


class TestParamsAndCli(unittest.TestCase):
    def test_registry_covers_output_params(self):
        from aurhythm import params as pm
        pipe = _pipeline_with_image()
        for key in ("lut_enabled", "lut_input_space", "lut_target_space",
                    "output_range", "dpx_transfer", "dpx_colorimetric"):
            value = pm.read_param(pipe, key)
            pm.apply_param(pipe, key, value)
            self.assertEqual(pm.read_param(pipe, key), value)

    def test_cli_convert_with_color_lut(self):
        lut = logc3_to_rec709_lut()
        with tempfile.TemporaryDirectory() as tmp:
            lut_path = os.path.join(tmp, "logc3_to_rec709.cube")
            lut.save(lut_path)
            source = os.path.join(tmp, "neg.tif")
            io_write.write_tiff(source, synth.make_negative(32, 48,
                                                           noise=0.0)[0], 16)
            out = os.path.join(tmp, "out")
            code = cli.main(["convert", "--in", source, "--out", out,
                             "--format", "dpx10", "--colorspace", "logc3",
                             "--lut", lut_path, "--lut-target", "rec709",
                             "--json"])
            self.assertEqual(code, cli.EXIT_OK)
            produced = [n for n in os.listdir(out) if n.endswith(".dpx")]
            self.assertEqual(len(produced), 1)
            _codes, header = io_write.read_dpx10(os.path.join(out, produced[0]))
            self.assertEqual(header["transfer"], om.DPX_TRANSFER["itu_r_709"])
            with open(os.path.join(out, os.path.splitext(produced[0])[0]
                                   + ".json"), encoding="utf-8") as fh:
                meta = json.load(fh)
            self.assertEqual(meta["output"]["effective_space"], "rec709")

    def test_cli_range_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "n.tif")
            io_write.write_tiff(source, synth.make_negative(32, 48)[0], 16)
            out = os.path.join(tmp, "o")
            code = cli.main(["convert", "--in", source, "--out", out,
                             "--format", "dpx10", "--range", "legal", "--json"])
            self.assertEqual(code, cli.EXIT_OK)
            produced = [n for n in os.listdir(out) if n.endswith(".dpx")]
            codes, _header = io_write.read_dpx10(
                os.path.join(out, produced[0]))
            self.assertGreaterEqual(int(codes.min()), 64)
            self.assertLessEqual(int(codes.max()), 940)

    def test_cli_rejects_bad_dpx_transfer(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "n.tif")
            io_write.write_tiff(source, synth.make_negative(16, 24)[0], 16)
            code = cli.main(["convert", "--in", source, "--out",
                             os.path.join(tmp, "o"), "--dpx-transfer", "nope"])
            self.assertEqual(code, cli.EXIT_USAGE)


if __name__ == "__main__":
    unittest.main()
