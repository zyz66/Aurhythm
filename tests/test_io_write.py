"""导出测试：基线 TIFF（16/32-bit）、真 DPX 10-bit、EXR 缺失时的报错、
provenance、以及格式分派。

DPX 与 TIFF 在本环境（无 tifffile / imageio）也能完整验证；
EXR 只验证「依赖缺失时给出可读错误」。
"""

from __future__ import annotations

import json
import os
import struct
import tempfile
import unittest

import numpy as np

import tests  # noqa: F401
from aurhythm import io_write as w


def sample_image(height: int = 13, width: int = 21, dtype=np.float64):
    rng = np.random.default_rng(4)
    return np.clip(rng.random((height, width, 3)), 0.0, 1.0).astype(dtype)


class TestConversionHelpers(unittest.TestCase):
    def test_uint16_rounding_and_clip(self):
        arr = np.array([0.0, 0.5, 1.0, -0.2, 1.4])
        out = w.to_uint16(arr)
        self.assertEqual(out.dtype, np.uint16)
        self.assertEqual(out[0], 0)
        self.assertEqual(out[2], 65535)
        self.assertEqual(out[3], 0)
        self.assertEqual(out[4], 65535)
        self.assertEqual(int(out[1]), int(round(0.5 * 65535)))

    def test_codes_from_normalized(self):
        out = w.to_codes(np.array([0.0, 1.0, 0.5, 2.0]))
        self.assertEqual(out.tolist(), [0, 1023, int(round(0.5 * 1023)), 1023])

    def test_rejects_wrong_shape(self):
        bad = np.zeros((4, 4, 4), dtype=np.float64)      # 4 通道，非法
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(w.WriteError):
                w.write_image(os.path.join(tmp, "x.tif"), bad, "tiff16")

    def test_accepts_2d_grayscale_by_broadcast(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "g.tif")
            w.write_image(path, np.full((4, 5), 0.5), "tiff16")
            back, _ = w.read_tiff_baseline(path)
            self.assertEqual(back.shape, (4, 5, 3))


class TestBaselineTiff(unittest.TestCase):
    def test_roundtrip_16bit(self):
        arr = sample_image()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.tif")
            w._write_tiff_baseline(path, arr, 16, None)
            back, description = w.read_tiff_baseline(path)
            self.assertIsNone(description)
            self.assertEqual(back.shape, arr.shape)
            self.assertEqual(back.dtype, np.uint16)
            expected = w.to_uint16(arr)
            np.testing.assert_array_equal(back, expected)

    def test_roundtrip_32bit_float(self):
        arr = sample_image().astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "f.tif")
            w.write_tiff(path, arr, 32)
            back, _ = w.read_tiff_baseline(path)
            # 内置写出器在无 tifffile 时使用；有 tifffile 时读回由它负责，
            # 这里只比较数值（float32 精度）
            np.testing.assert_allclose(back.astype(np.float32), arr, atol=1e-6)

    def test_description_embedding(self):
        arr = sample_image()
        meta = {"preset": "Vision3 250D", "d_min": 0.12, "nested": {"a": 1}}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.tif")
            w._write_tiff_baseline(path, arr, 16,
                                   json.dumps(meta, ensure_ascii=False))
            _back, description = w.read_tiff_baseline(path)
            self.assertEqual(json.loads(description), meta)

    def test_pillow_can_open_it(self):
        """独立校验：Pillow 能打开内置写出器产出的 TIFF（结构合法）。"""
        try:
            from PIL import Image
        except ImportError:                     # pragma: no cover
            self.skipTest("Pillow 不可用")
        arr = sample_image()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "p.tif")
            w._write_tiff_baseline(path, arr, 16, None)
            with Image.open(path) as img:
                self.assertEqual(img.size, (arr.shape[1], arr.shape[0]))

    def test_odd_width_and_height(self):
        arr = sample_image(7, 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "odd.tif")
            w.write_tiff(path, arr, 16)
            back, _ = w.read_tiff_baseline(path)
            np.testing.assert_array_equal(back, w.to_uint16(arr))

    def test_bad_bit_depth(self):
        with self.assertRaises(w.WriteError):
            w.write_tiff("/tmp/x.tif", sample_image(), 24)


class TestDpx(unittest.TestCase):
    def test_header_fields(self):
        arr = sample_image(9, 11)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.dpx")
            w.write_dpx10(path, arr)
            with open(path, "rb") as fh:
                blob = fh.read()
            self.assertEqual(blob[:4], b"SDPX")
            self.assertEqual(blob[8:12], b"V2.0")
            (data_offset,) = struct.unpack(">I", blob[4:8])
            (total_size,) = struct.unpack(">I", blob[16:20])
            self.assertEqual(data_offset, w.DPX_DATA_OFFSET)
            self.assertEqual(total_size, len(blob))
            self.assertEqual(struct.unpack(">I", blob[772:776])[0], 11)
            self.assertEqual(struct.unpack(">I", blob[776:780])[0], 9)
            element = w.DPX_ELEMENT_OFFSET
            self.assertEqual(blob[element + 20], 50)          # descriptor RGB
            # transfer 默认 = printing density（DPX 常见映射里 1 = printing
            # density；旧代码写 2 却命名为 printing density，已修正）
            from aurhythm.output import SPACE_CINEON, SPACE_INFO
            self.assertEqual(blob[element + 21],
                             SPACE_INFO[SPACE_CINEON]["dpx_transfer"])
            self.assertEqual(blob[element + 23], 10)          # bit size
            self.assertEqual(struct.unpack(">H", blob[element + 24:element + 26])[0], 1)
            self.assertEqual(struct.unpack(">I", blob[element + 12:element + 16])[0], 1023)

    def test_bit_exact_roundtrip(self):
        """码值必须位精确往返（10-bit 不能有量化漂移）。"""
        arr = sample_image(8, 6)
        codes_expected = w.to_codes(arr)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "r.dpx")
            w.write_dpx10(path, arr)
            codes, header = w.read_dpx10(path)
            np.testing.assert_array_equal(codes, codes_expected)
            self.assertEqual(header["width"], 6)
            self.assertEqual(header["height"], 8)
            self.assertEqual(header["bit_size"], 10)

    def test_line_packing_is_word_aligned(self):
        """packing=1：每像素 3 个 32-bit 字，行末不需要额外填充。"""
        width, height = 5, 3
        arr = sample_image(height, width)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "w.dpx")
            w.write_dpx10(path, arr)
            size = os.path.getsize(path)
            self.assertEqual(size, w.DPX_DATA_OFFSET + width * height * 3 * 4)

    def test_extreme_values_clamped_to_code_range(self):
        arr = np.array([[[0.0, 1.0, -1.0], [2.0, 0.5, 0.5]]], dtype=np.float64)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "c.dpx")
            w.write_dpx10(path, arr)
            codes, _ = w.read_dpx10(path)
            self.assertEqual(int(codes[0, 0, 0]), 0)
            self.assertEqual(int(codes[0, 0, 1]), 1023)
            self.assertEqual(int(codes[0, 0, 2]), 0)
            self.assertEqual(int(codes[0, 1, 0]), 1023)

    def test_read_rejects_other_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "not.dpx")
            with open(path, "wb") as fh:
                fh.write(b"XPDS" + b"\x00" * 100)
            with self.assertRaises(w.WriteError):
                w.read_dpx10(path)


class TestExr(unittest.TestCase):
    def test_missing_dependency_gives_readable_error(self):
        """EXR 依赖缺失时必须给出可读错误（旧实现静默失败）。"""
        try:
            import imageio.v2  # noqa: F401
            self.skipTest("本环境有 imageio，跳过缺失路径")
        except ImportError:
            pass
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.exr")
            with self.assertRaises(w.WriteError) as ctx:
                w.write_exr(path, sample_image(), 32)
            message = str(ctx.exception)
            self.assertIn("EXR", message)
            self.assertIn("imageio", message)


class TestDispatchAndProvenance(unittest.TestCase):
    def test_extension_map(self):
        self.assertEqual(w.extension_for("tiff16"), "tif")
        self.assertEqual(w.extension_for("dpx10"), "dpx")
        self.assertEqual(w.extension_for("exr32"), "exr")
        self.assertEqual(len(w.SUPPORTED_FORMATS), 5)

    def test_write_image_dispatch(self):
        arr = sample_image(6, 6)
        with tempfile.TemporaryDirectory() as tmp:
            for fmt in ("tiff16", "tiff32", "dpx10"):
                path = os.path.join(tmp, f"out.{w.extension_for(fmt)}")
                w.write_image(path, arr, fmt)
                self.assertTrue(os.path.getsize(path) > 0)
            with self.assertRaises(w.WriteError):
                w.write_image(os.path.join(tmp, "x.bin"), arr, "jpeg")

    def test_dpx_writes_provenance_sidecar(self):
        arr = sample_image(6, 6)
        meta = {"preset": "Vision3 500T", "base": [0.9, 0.84, 0.72]}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "p.dpx")
            w.write_dpx10(path, arr, metadata=meta, transfer=6, colorimetric=2)
            sidecar = os.path.splitext(path)[0] + ".json"
            self.assertTrue(os.path.exists(sidecar))
            with open(sidecar, encoding="utf-8") as fh:
                payload = json.load(fh)
            # 原始参数必须原样保留，并附上实际写入的 DPX 字段（可审计）
            for key, value in meta.items():
                self.assertEqual(payload[key], value)
            self.assertEqual(payload["dpx_transfer"], 6)
            self.assertEqual(payload["dpx_colorimetric"], 2)
            self.assertEqual(payload["range"], "full")

    def test_tiff_embeds_provenance(self):
        arr = sample_image(5, 5)
        meta = {"schema_version": 1, "preset": "无 (关闭预设)"}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "t.tif")
            w.write_tiff(path, arr, 16, metadata=meta)
            _back, description = w.read_tiff_baseline(path)
            self.assertEqual(json.loads(description), meta)


class TestPipelineIntegration(unittest.TestCase):
    def test_export_normalized_output_to_all_local_formats(self):
        from aurhythm.pipeline import ScientificFilmPipeline
        from tests.test_pipeline import make_negative, _profile

        pipe = ScientificFilmPipeline()
        img, base = make_negative(16, 24)
        pipe.load_linear_image(img)
        pipe.profile = _profile()
        pipe.set_base_val(base)
        data = pipe.process_for_output()
        self.assertIsNotNone(data)
        with tempfile.TemporaryDirectory() as tmp:
            tif = os.path.join(tmp, "out.tif")
            w.write_image(tif, data, "tiff16", metadata=pipe.describe())
            back, description = w.read_tiff_baseline(tif)
            self.assertEqual(back.shape, data.shape)
            self.assertIn("transfer", json.loads(description))

            dpx = os.path.join(tmp, "out.dpx")
            w.write_image(dpx, data, "dpx10", metadata=pipe.to_settings())
            codes, _header = w.read_dpx10(dpx)
            np.testing.assert_array_equal(codes, w.to_codes(data))


if __name__ == "__main__":
    unittest.main()
