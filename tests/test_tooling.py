"""参数注册表、设置存取、图像读取、批量引擎、CLI 的测试。

重点：
* S13 —— 注册表里每个键都能被管线读写（R7 类死控件的结构性根治）；
* R10 —— 参数复制使用完整 profile 路径；
* R12 —— 无 rawpy 时模块仍可导入、只有真读 RAW 才报错；
* 设置文件的旧版本迁移；
* CLI 的 synth → convert → info 链路与退出码。
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import struct
import tempfile
import unittest

import numpy as np

import tests  # noqa: F401
from aurhythm import batch as batch_mod
from aurhythm import cli
from aurhythm import io_read, io_write
from aurhythm import params as pm
from aurhythm import settings as sm
from aurhythm import synth
from aurhythm.constants import SCHEMA_VERSION
from aurhythm.pipeline import ScientificFilmPipeline


def _pipeline_with_image(height=24, width=32, cast=(1.0, 1.0, 1.0)):
    img, base = synth.make_negative(height=height, width=width, cast=cast,
                                    noise=0.0)
    pipe = ScientificFilmPipeline()
    pipe.load_linear_image(img)
    pipe.set_base_val(base)
    return pipe


class TestParamRegistry(unittest.TestCase):
    def test_every_key_is_readable_and_writable(self):
        """S13：注册表里每个键都必须能被管线读写（R7 的根治）。"""
        pipe = _pipeline_with_image()
        for key, spec in pm.PARAM_SPECS.items():
            if key == "export_format":
                continue          # 纯 UI 状态
            value = pm.read_param(pipe, key)
            self.assertIsNotNone(value, key)
            pm.apply_param(pipe, key, value)          # 必须不抛错
            self.assertEqual(pm.read_param(pipe, key), value, key)

    def test_unknown_key_raises(self):
        pipe = _pipeline_with_image()
        for key in ("hd_softness", "nope", "hd_clip_softness"):
            with self.assertRaises(KeyError):
                pm.apply_param(pipe, key, 0.1)
            with self.assertRaises(KeyError):
                pm.read_param(pipe, key)

    def test_values_are_clamped_to_registry_range(self):
        pipe = _pipeline_with_image()
        pm.apply_param(pipe, "hd_a", 999.0)
        self.assertLessEqual(pm.read_param(pipe, "hd_a"), pm.spec("hd_a").hi)
        pm.apply_param(pipe, "tone_contrast", -5.0)
        self.assertGreaterEqual(pm.read_param(pipe, "tone_contrast"),
                                pm.spec("tone_contrast").lo)

    def test_choice_validation(self):
        pipe = _pipeline_with_image()
        pm.apply_param(pipe, "output_colorspace", "logc3")
        self.assertEqual(pm.read_param(pipe, "output_colorspace"), "logc3")
        with self.assertRaises(ValueError):
            pm.apply_param(pipe, "output_colorspace", "aces")

    def test_channel_gain_writes_only_its_channel(self):
        pipe = _pipeline_with_image()
        pm.apply_param(pipe, "gain_r", 1.4)
        gains = np.asarray(pipe.channel_gains)
        self.assertAlmostEqual(gains[0], 1.4)
        self.assertAlmostEqual(gains[1], 1.0)
        self.assertAlmostEqual(gains[2], 1.0)

    def test_cdl_writes_only_its_channel_and_field(self):
        pipe = _pipeline_with_image()
        pm.apply_param(pipe, "cdl_gain_b", 1.7)
        self.assertAlmostEqual(pipe.tone.gain[2], 1.7)
        self.assertAlmostEqual(pipe.tone.gain[0], 1.0)
        self.assertAlmostEqual(pipe.tone.gamma[2], 1.0)

    def test_snapshot_and_restore(self):
        pipe = _pipeline_with_image()
        pm.apply_param(pipe, "hd_a", 1.8)
        pm.apply_param(pipe, "gain_g", 1.25)
        snapshot = pm.snapshot(pipe)
        pm.apply_param(pipe, "hd_a", 1.0)
        pm.restore(pipe, snapshot)
        self.assertAlmostEqual(pm.read_param(pipe, "hd_a"), 1.8)
        self.assertAlmostEqual(pm.read_param(pipe, "gain_g"), 1.25)

    def test_look_keys_are_marked(self):
        """调色层必须被显式标注，UI 才能分区并默认恒等。"""
        look = pm.look_keys()
        self.assertIn("tone_contrast", look)
        self.assertIn("cdl_gain_r", look)
        self.assertNotIn("hd_a", look)
        self.assertNotIn("icc_weight", look)
        self.assertGreater(len(look), 10)
        for key in look:
            self.assertTrue(pm.spec(key).look)

    def test_groups_cover_all_keys(self):
        covered = set()
        for group in pm.GROUP_ORDER:
            covered |= {s.key for s in pm.specs_for_group(group)}
        self.assertEqual(covered, set(pm.PARAM_SPECS))
        for key in pm.PARAM_SPECS:
            self.assertIn(pm.spec(key).group, pm.GROUP_ORDER)

    def test_defaults_are_valid(self):
        pipe = _pipeline_with_image()
        for key, value in pm.defaults().items():
            if key == "export_format":
                continue
            pm.apply_param(pipe, key, value)


class TestSettings(unittest.TestCase):
    def test_roundtrip(self):
        pipe = _pipeline_with_image(cast=(1.1, 1.0, 0.9))
        pipe.auto_align_density_domain()
        data = pipe.to_settings()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "s.json")
            sm.save_settings(path, data)
            loaded = sm.load_settings(path)
        self.assertEqual(loaded["schema_version"], SCHEMA_VERSION)
        self.assertEqual(loaded["preset"], data["preset"])
        np.testing.assert_allclose(loaded["channel_gains"],
                                   data["channel_gains"])

    def test_apply_settings_to_pipeline(self):
        source = _pipeline_with_image()
        source.set_preset("Kodak Vision3 500T (5219)")
        source.set_channel_gains([1.1, 0.95, 1.2])
        pm.apply_param(source, "hd_a", 1.4)

        target = ScientificFilmPipeline()
        report = sm.apply_settings_to_pipeline(target, source.to_settings(),
                                               copy_pixels=True)
        self.assertIsInstance(report, list)
        self.assertEqual(target.preset_name, "Kodak Vision3 500T (5219)")
        np.testing.assert_allclose(target.channel_gains, [1.1, 0.95, 1.2])
        self.assertAlmostEqual(target.hd.a, 1.4)

    def test_profile_copy_uses_full_path(self):
        """R10：复制参数必须用完整路径重新加载 profile。"""
        xml = """<?xml version="1.0"?>
<dcpData><ColorMatrix1>1 0 0 0 1 0 0 0 1</ColorMatrix1></dcpData>"""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cam.dcp")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(xml)
            source = ScientificFilmPipeline()
            source.load_icc_profile(path)
            data = source.to_settings()
            self.assertEqual(data["icc_source"], os.path.abspath(path))

            target = ScientificFilmPipeline()
            report = sm.apply_settings_to_pipeline(target, data)
            self.assertIsNotNone(target.profile, f"profile 未复制: {report}")
            self.assertTrue(target.icc_loaded)

    def test_v4_migration(self):
        legacy = {
            "hd_slope": 4.5, "hd_mid": -0.8, "hd_clip_softness": 0.003,
            "hd_min": 0.12, "hd_max": 3.2,
            "preset": "通用负片 (默认)", "icc_weight": 0.25,
            "channel_gains": [1.0, 1.0, 1.0], "output_colorspace": "cineon",
            "format": "exr", "bit_depth": 32,
        }
        report: list = []
        migrated = sm.migrate(dict(legacy), report=report)
        self.assertEqual(migrated["schema_version"], SCHEMA_VERSION)
        self.assertAlmostEqual(migrated["hd"]["d_min"], 0.12)
        self.assertEqual(migrated["export_format"], "exr32")
        self.assertTrue(any("迁移" in line for line in report))

    def test_migration_reports_unknown_keys(self):
        report: list = []
        sm.migrate({"some_unknown_flag": 1}, report=report)
        self.assertTrue(any("未知参数" in line for line in report))

    def test_settings_to_param_values(self):
        pipe = _pipeline_with_image()
        pm.apply_param(pipe, "gain_b", 1.3)
        values = sm.settings_to_param_values(pipe.to_settings())
        self.assertAlmostEqual(values["gain_b"], 1.3)
        self.assertIn("hd_a", values)


class TestIoRead(unittest.TestCase):
    def test_tiff16_roundtrip_is_assumed_linear(self):
        arr = np.linspace(0, 1, 16 * 16 * 3).reshape(16, 16, 3).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.tif")
            io_write.write_tiff(path, arr, 16)
            data, info = io_read.load_any(path)
            self.assertFalse(info.is_raw)
            self.assertTrue(info.assumed_linear)
            self.assertEqual(data.shape, (16, 16, 3))
            np.testing.assert_allclose(data, arr, atol=2e-5)

    def test_tiff32_float(self):
        arr = np.linspace(0, 1.5, 8 * 8 * 3).reshape(8, 8, 3).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "f.tif")
            io_write.write_tiff(path, arr, 32)
            data, info = io_read.load_any(path, assume_linear=True)
            self.assertEqual(info.source_bits, 32)
            np.testing.assert_allclose(data, arr, atol=1e-6)

    def test_png_is_srgb_decoded_by_default(self):
        try:
            from PIL import Image
        except ImportError:                            # pragma: no cover
            self.skipTest("Pillow 不可用")
        encoded = np.array([[[0, 0, 0], [128, 128, 128], [255, 255, 255]]],
                           dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.png")
            Image.fromarray(encoded).save(path)
            data, info = io_read.load_any(path)
            self.assertFalse(info.assumed_linear)
            self.assertAlmostEqual(float(data[0, 0, 0]), 0.0, places=4)
            self.assertAlmostEqual(float(data[0, 1, 0]), 0.2158, places=3)
            self.assertAlmostEqual(float(data[0, 2, 0]), 1.0, places=4)

    def test_png_can_be_forced_linear(self):
        try:
            from PIL import Image
        except ImportError:                            # pragma: no cover
            self.skipTest("Pillow 不可用")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "b.png")
            Image.fromarray(np.full((4, 4, 3), 128, dtype=np.uint8)).save(path)
            data, info = io_read.load_any(path, assume_linear=True)
            self.assertTrue(info.assumed_linear)
            self.assertAlmostEqual(float(data[0, 0, 0]), 128 / 255, places=5)

    def test_raw_without_rawpy_gives_readable_error(self):
        """R12：缺 rawpy 时只在真正读 RAW 时报错，且消息可读。"""
        try:
            import rawpy  # noqa: F401
            self.skipTest("本环境有 rawpy")
        except ImportError:
            pass
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.nef")
            with open(path, "wb") as fh:
                fh.write(b"\x00" * 64)
            with self.assertRaises(io_read.ReadError) as ctx:
                io_read.load_any(path)
            self.assertIn("rawpy", str(ctx.exception))

    def test_missing_file(self):
        with self.assertRaises(io_read.ReadError):
            io_read.load_any("/nonexistent/nope.tif")

    def test_probe_tiff(self):
        arr = np.zeros((9, 7, 3), dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "p.tif")
            io_write.write_tiff(path, arr, 16, metadata={"a": 1})
            info = io_read.probe(path)
            self.assertEqual(info["width"], 7)
            self.assertEqual(info["height"], 9)
            self.assertEqual(info["source_bits"], 16)

    def test_module_imports_without_rawpy(self):
        """R12：模块本身必须能在没有 rawpy 的环境导入。"""
        import importlib
        module = importlib.import_module("aurhythm.io_read")
        self.assertTrue(hasattr(module, "load_any"))


class TestSynth(unittest.TestCase):
    def test_negative_shape_and_base(self):
        image, base = synth.make_negative(height=16, width=24, seed=1)
        self.assertEqual(image.shape, (16, 24, 3))
        self.assertEqual(base.shape, (3,))
        np.testing.assert_allclose(image[:8, 0, :], np.tile(base, (8, 1)),
                                   atol=0.01)

    def test_chart_has_expected_structure(self):
        image = synth.make_chart(height=180, width=270)
        self.assertEqual(image.shape, (180, 270, 3))
        # 背景与色块都应当存在
        self.assertGreater(float(image.max()), 0.5)
        self.assertLess(float(image.min()), 0.05)

    def test_density_gamma_creates_neutralisable_cast(self):
        """逐通道密度反差：片基处不偏色，中间调分通道分离。"""
        neutral = synth.make_negative(120, 160, base_rows=24, noise=0.0)
        cast = synth.make_negative(120, 160, base_rows=24, noise=0.0,
                                   density_gamma=(0.90, 1.0, 1.12))
        self.assertEqual(cast[0].shape, neutral[0].shape)
        # 片基（顶部若干行）在两种模型下都应当是同一个 base*cast
        np.testing.assert_allclose(cast[0][:8], neutral[0][:8], atol=1e-6)

        def reference(image):
            """返回中性化**实际使用**的参考密度（已排除片基区）。"""
            pipe = ScientificFilmPipeline()
            pipe.load_linear_image(image)
            pipe.set_base_val(pipe.auto_detect_base())
            return np.array(pipe.auto_align_density_domain(
                target_density=0.7)["reference"])

        neutral_ref = reference(neutral[0])
        cast_ref = reference(cast[0])
        # 中性样片三通道参考密度相同；带逐通道反差的必须分离
        np.testing.assert_allclose(neutral_ref, neutral_ref[0], atol=1e-6)
        self.assertGreater(float(cast_ref.max() - cast_ref.min()), 0.05,
                           f"逐通道反差未被体现: {cast_ref}")

        # 中性化必须能把它对齐到目标
        pipe = ScientificFilmPipeline()
        pipe.load_linear_image(cast[0])
        pipe.set_base_val(pipe.auto_detect_base())
        result = pipe.auto_align_density_domain(target_density=0.7)
        np.testing.assert_allclose(np.array(result["achieved"]), 0.7, atol=0.01)

    def test_neutralization_ignores_large_blank_base_area(self):
        """大片未曝光片基不得被当成"中间调"参考（真实扫描件的常见情形）。

        60% 片基 + 40% 场景时，若用整幅中位数当参考，参考点会落在片基上
        （密度≈0），于是拿**片基**去对齐目标密度 —— 整幅被推亮 0.7。
        """
        image, base = synth.make_negative(200, 200, base_rows=120, noise=0.0,
                                          density_gamma=(0.90, 1.0, 1.12))
        pipe = ScientificFilmPipeline()
        pipe.load_linear_image(image)
        pipe.set_base_val(base)
        result = pipe.auto_align_density_domain(target_density=0.7)
        self.assertGreater(result["excluded_base_pixels"], 0, "应当排除片基像素")
        self.assertFalse(result["used_fallback"])

        gains = np.asarray(pipe.channel_gains)
        density = pipe._density(pipe._stage_a_b_c() * gains.reshape(1, 1, 3),
                               pipe._base_aligned())
        scene = density[120:, :, :].reshape(-1, 3)
        np.testing.assert_allclose(np.median(scene, axis=0), 0.7, atol=0.05)

    def test_density_window_is_robust_to_outliers(self):
        """测量密度范围必须抗离群点：单个热像素不得把窗口拉爆。"""
        image, base = synth.make_negative(120, 160, base_rows=24, noise=0.0)
        clean = ScientificFilmPipeline()
        clean.load_linear_image(image)
        clean.set_base_val(base)
        clean_report = clean.measure_density()

        dirty = image.copy()
        dirty[60, 80, :] = 0.0005           # 一个极暗的坏点（=极大密度）
        noisy = ScientificFilmPipeline()
        noisy.load_linear_image(dirty)
        noisy.set_base_val(base)
        noisy_report = noisy.measure_density()

        # 绝对最大值被离群点拉高……
        self.assertGreater(max(noisy_report.per_channel_max),
                           max(clean_report.per_channel_max) + 0.5)
        # ……但 99.9% 分位基本不动（这才是窗口该用的量）
        np.testing.assert_allclose(noisy_report.percentile_high,
                                   clean_report.percentile_high, atol=0.15)

    def test_cli_synth_density_gamma_option(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cast.tif")
            code, out, _err = run_cli(["synth", "--out", path, "--kind",
                                       "negative", "--density-gamma",
                                       "0.9", "1.0", "1.1", "--json"])
            self.assertEqual(code, cli.EXIT_OK)
            self.assertTrue(os.path.exists(path))
            self.assertIn("密度反差", out)

    def test_unknown_kind(self):
        with self.assertRaises(ValueError):
            synth.make("nope")


class TestBatch(unittest.TestCase):
    def _write_negatives(self, directory, count=3, cast=(1.0, 1.0, 1.0)):
        paths = []
        for index in range(count):
            image, _base = synth.make_negative(height=24, width=32,
                                              cast=cast, noise=0.0,
                                              seed=index)
            path = os.path.join(directory, f"neg{index}.tif")
            io_write.write_tiff(path, image, 16)
            paths.append(path)
        return paths

    def test_run_batch_exports_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._write_negatives(tmp)
            out = os.path.join(tmp, "out")
            os.makedirs(out)
            seen = []
            result = batch_mod.run_batch(
                paths, {"schema_version": 1, "preset": "通用负片 (默认)"},
                out_dir=out, export_format="tiff16",
                progress=lambda i, n, item: seen.append((i, n, item.status)))
            self.assertEqual(result.summary["ok"], 3)
            self.assertEqual(result.summary["failed"], 0)
            self.assertEqual(len(seen), 3)
            for item in result.items:
                self.assertTrue(os.path.exists(item.output))
                self.assertIsNotNone(item.density)
            json.dumps(result.to_dict())

    def test_cancel_marks_remaining(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._write_negatives(tmp, count=4)
            calls = {"n": 0}

            def cancel():
                calls["n"] += 1
                return calls["n"] > 2

            result = batch_mod.run_batch(paths, {}, cancel=cancel)
            self.assertGreaterEqual(result.count("canceled"), 1)
            self.assertLess(result.count("ok"), 4)

    def test_failure_is_reported_not_silent(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = os.path.join(tmp, "broken.tif")
            with open(bad, "wb") as fh:
                fh.write(b"not a tiff at all")
            result = batch_mod.run_batch([bad], {})
            self.assertEqual(result.summary["failed"], 1)
            self.assertTrue(result.failure_report())

    def test_unique_output_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._write_negatives(tmp, count=1)
            out = os.path.join(tmp, "out")
            os.makedirs(out)
            settings = {"schema_version": 1}
            first = batch_mod.process_one(paths[0], settings, out_dir=out)
            second = batch_mod.process_one(paths[0], settings, out_dir=out)
            self.assertNotEqual(first.output, second.output)
            third = batch_mod.process_one(paths[0], settings, out_dir=out,
                                          overwrite=True)
            self.assertEqual(third.output, first.output)

    def test_collect_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_negatives(tmp, count=2)
            os.makedirs(os.path.join(tmp, "sub"))
            io_write.write_tiff(os.path.join(tmp, "sub", "deep.tif"),
                                np.zeros((4, 4, 3), np.float32), 16)
            self.assertEqual(len(batch_mod.collect_paths(tmp)), 3)
            self.assertEqual(len(batch_mod.collect_paths(tmp, recursive=False)), 2)


def run_cli(argv):
    """调用 CLI 并吞掉 stdout/stderr，返回退出码（保持套件输出干净）。"""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(argv)
    return code, out.getvalue(), err.getvalue()


class TestCli(unittest.TestCase):
    def test_synth_convert_info_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "neg.tif")
            code, _out, _err = run_cli(["synth", "--out", source, "--kind", "negative",
                             "--width", "48", "--height", "32", "--json"])
            self.assertEqual(code, cli.EXIT_OK)
            self.assertTrue(os.path.exists(source))

            out = os.path.join(tmp, "out")
            code, _out, _err = run_cli(["convert", "--in", source, "--out", out,
                             "--format", "dpx10",
                             "--preset", "Kodak Vision3 250D (5207)",
                             "--json"])
            self.assertEqual(code, cli.EXIT_OK)
            produced = sorted(os.listdir(out))
            # DPX 会额外写一份 provenance sidecar（.json）
            dpx = [name for name in produced if name.endswith(".dpx")]
            self.assertEqual(len(dpx), 1, produced)
            self.assertEqual(len(produced), 2, produced)
            codes, header = io_write.read_dpx10(os.path.join(out, dpx[0]))
            self.assertEqual(header["bit_size"], 10)

            code, _out, _err = run_cli(["info", source, "--json"])
            self.assertEqual(code, cli.EXIT_OK)

    def test_convert_partial_failure_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = os.path.join(tmp, "bad.tif")
            with open(bad, "wb") as fh:
                fh.write(b"nope")
            out = os.path.join(tmp, "out")
            code, _out, _err = run_cli(["convert", "--in", bad, "--out", out, "--json"])
            self.assertEqual(code, cli.EXIT_PARTIAL)

    def test_missing_input_gives_unsupported(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, _out, _err = run_cli(["convert", "--in", "/nonexistent/x.tif",
                             "--out", tmp, "--json"])
            self.assertEqual(code, cli.EXIT_UNSUPPORTED)

    def test_unknown_preset_is_usage_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, _out, _err = run_cli(["convert", "--in", "/nonexistent/x.tif",
                                        "--out", tmp, "--preset", "Nope"])
            self.assertEqual(code, cli.EXIT_USAGE)

    def test_lut_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cal.cube")
            code, _out, _err = run_cli(["lut", "--out", path, "--size", "9",
                             "--preset", "通用负片 (默认)", "--json"])
            self.assertEqual(code, cli.EXIT_OK)
            from aurhythm.lut import CubeLUT
            lut = CubeLUT()
            lut.load(path)
            self.assertEqual(lut.size, 9)

    def test_calibrate_with_manual_corners(self):
        with tempfile.TemporaryDirectory() as tmp:
            corners = [80, 60, 460, 45, 480, 300, 65, 315]
            chart = os.path.join(tmp, "chart.tif")
            io_write.write_tiff(
                chart, synth.make_chart(360, 540, corners=np.array(
                    [[corners[0], corners[1]], [corners[2], corners[3]],
                     [corners[4], corners[5]], [corners[6], corners[7]]],
                    dtype=np.float64)), 16)
            out = os.path.join(tmp, "matrix.json")
            code, _out, _err = run_cli(["calibrate", "--chart", chart, "--corners",
                             ",".join(str(v) for v in corners),
                             "--out", out, "--json"])
            self.assertEqual(code, cli.EXIT_OK)
            with open(out, encoding="utf-8") as fh:
                payload = json.load(fh)
            self.assertEqual(np.array(payload["matrix"]).shape, (3, 3))
            self.assertLess(payload["stats"]["mean_deltaE"], 1.5)

    def test_calibrate_rejects_uniform_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            chart = os.path.join(tmp, "flat.tif")
            io_write.write_tiff(chart, np.full((120, 160, 3), 0.5, np.float32), 16)
            code, _out, _err = run_cli(["calibrate", "--chart", chart, "--json"])
            self.assertEqual(code, cli.EXIT_PARTIAL)

    def test_no_command_prints_help(self):
        self.assertEqual(run_cli([])[0], cli.EXIT_USAGE)

    def test_report_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "n.tif")
            io_write.write_tiff(source, synth.make_negative(24, 32)[0], 16)
            report = os.path.join(tmp, "report.json")
            code, _out, _err = run_cli(["convert", "--in", source, "--out",
                             os.path.join(tmp, "o"), "--report", report,
                             "--json"])
            self.assertEqual(code, cli.EXIT_OK)
            with open(report, encoding="utf-8") as fh:
                payload = json.load(fh)
            self.assertIn("summary", payload)


if __name__ == "__main__":
    unittest.main()
