"""R1–R15 回归项显式映射。

每一条都记录 **v4.2 的实测症状** 与 v5.0 的判据，便于审计"回归是否真的被覆盖"。
（同样的断言在各自的模块测试里也有更细的版本；这里的作用是让映射可追溯。）
"""

from __future__ import annotations

import ast
import importlib
import inspect
import os
import tempfile
import unittest

import numpy as np

import tests  # noqa: F401
from aurhythm import colorchecker as cc, io_read, params as pm, synth
from aurhythm import lut as lutmod, profiles as prof, settings as sm
from aurhythm.constants import CINEON, FILM_PRESETS, LOGC3, PRESET_ORDER
from aurhythm.pipeline import ScientificFilmPipeline
from aurhythm.tonemap import TransferFunction
from tests.test_pipeline import _profile


class TestRegressions(unittest.TestCase):
    def test_r1_preset_no_longer_posterises(self):
        """R1：旧实现开启预设后 512 级灰阶只剩 3 级（Sigmoid 退化成砖墙）。"""
        for name in PRESET_ORDER:
            tf = TransferFunction(hd=FILM_PRESETS[name].hd)
            p = tf.params
            density = np.repeat(
                np.linspace(p.d_min, p.d_max, 512)[:, None], 3, axis=1)
            code, _ = tf.density_to_code(density)
            distinct = len(np.unique(np.round(code[:, 0], 9)))
            self.assertGreaterEqual(distinct, 500, f"{name}: 只剩 {distinct} 级")

    def test_r2_default_lut_domain_no_crash(self):
        """R2：旧实现 domain 是 list，`list - list` → TypeError，LUT 必崩。"""
        lut = lutmod.CubeLUT()
        lut.size = 2
        lut.table = np.zeros((2, 2, 2, 3))
        out = lut.apply(np.zeros((1, 1, 3)))
        self.assertEqual(out.shape, (1, 1, 3))
        self.assertIsInstance(lut.domain_max, np.ndarray)

    def test_r3_identity_lut_keeps_axes(self):
        """R3：旧实现对恒等 LUT 会把 R 与 B 对调。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "id.cube")
            rows = [[r / 2, g / 2, b / 2] for b in range(3) for g in range(3)
                    for r in range(3)]
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("LUT_3D_SIZE 3\n"
                         + "\n".join(" ".join(map(str, row)) for row in rows))
            lut = lutmod.CubeLUT()
            lut.load(path)
            probe = np.array([[[0.9, 0.1, 0.1], [0.1, 0.1, 0.9]]])
            np.testing.assert_allclose(lut.apply(probe), probe, atol=1e-9)

    def test_r4_logc3_matches_arri(self):
        """R4：旧公式 18% 灰给 0.5497（ARRI 参考 0.391007），1% 曝光差 +0.67。"""
        self.assertAlmostEqual(float(LOGC3.encode(0.18)), 0.391007, places=6)
        self.assertAlmostEqual(float(LOGC3.encode(1.0)), 0.570632, places=6)
        self.assertAlmostEqual(float(LOGC3.encode(LOGC3.cut)), 0.149658, places=6)

    def test_r5_per_channel_base_and_configurable_clamp(self):
        """R5：旧实现把逐通道片基覆盖成标量灰，且增益静默裁到 [0.5, 2.0]。"""
        img, base = synth.make_negative(32, 48, base=(0.90, 0.50, 0.30),
                                        noise=0.0)
        pipe = ScientificFilmPipeline()
        pipe.load_linear_image(img)
        pipe.profile = _profile()
        pipe.set_base_val(base)
        aligned = pipe._base_aligned()
        ratios = aligned / aligned[1]
        np.testing.assert_allclose(ratios, [1.8, 1.0, 0.6], atol=1e-6)

        result = pipe.auto_align_density_domain(clamp=(0.9, 1.1))
        self.assertTrue(result["clamped"], "触限必须被回报")
        wide = pipe.auto_align_density_domain(clamp=None)
        self.assertFalse(wide["clamped"], "clamp=None 表示不裁剪")

    def test_r6_saturated_base_frame_detected(self):
        """R6：旧判据 `luminance > percentile(95)` 在片基过曝 >5% 时返回 None。"""
        img = np.full((32, 32, 3), 0.2, dtype=np.float32)
        img[:4] = 1.0
        pipe = ScientificFilmPipeline()
        pipe.load_linear_image(img)
        self.assertIsNone(pipe.auto_detect_base(method="percentile"))
        base = pipe.auto_detect_base()
        self.assertIsNotNone(base)
        np.testing.assert_allclose(base, 1.0, atol=1e-6)

    def test_r7_ui_key_mismatch_is_structurally_impossible(self):
        """R7：旧 UI 写 hd_softness、管线读 hd_clip_softness，滑块完全无效。"""
        pipe = ScientificFilmPipeline()
        with self.assertRaises(KeyError):
            pm.apply_param(pipe, "hd_softness", 0.1)
        with self.assertRaises(KeyError):
            pm.read_param(pipe, "hd_clip_softness")
        tree = ast.parse(inspect.getsource(
            importlib.import_module("aurhythm.app")))
        used = set()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "attr", None) == "_param_rows"):
                for arg in node.args[1:]:
                    if isinstance(arg, ast.List):
                        used |= {e.value for e in arg.elts
                                 if isinstance(e, ast.Constant)}
        self.assertEqual(used | {"export_format"}, set(pm.PARAM_SPECS))

    def test_r8_linear_stages_do_not_clip(self):
        """R8：旧实现 apply_icc / 密度转换都 clip，高光永久损失。"""
        pipe = ScientificFilmPipeline()
        pipe.load_linear_image(np.array([[[1.5, 0.5, -0.05]]], dtype=np.float32))
        pipe.profile = _profile()
        out = pipe._stage_a_b_c()
        self.assertGreater(float(out.max()), 1.0)
        self.assertLess(float(out.min()), 0.0)

    def test_r9_no_random_detection_and_real_table(self):
        """R9：旧 detect_colorchecker 是 `target * random.uniform(...)`。"""
        offenders = [node.attr for node in ast.walk(ast.parse(
            inspect.getsource(cc)))
            if isinstance(node, ast.Attribute)
            and node.attr in ("random", "uniform", "normal", "seed")]
        self.assertEqual(offenders, [])
        lab = cc.target_linear_srgb()
        from aurhythm import colorimetry as cm
        chroma = cm.lab_to_lch(cm.rgb_to_lab_srgb(lab))[:, 1]
        self.assertGreater(float(np.min(chroma[:18])), 15.0)
        self.assertLess(float(np.max(chroma[18:])), 2.0)

    def test_r10_profile_copied_by_full_path(self):
        """R10：旧实现只传 basename，批量复制 profile 静默失败。"""
        xml = ('<?xml version="1.0"?><dcpData>'
               '<ColorMatrix1>1 0 0 0 1 0 0 0 1</ColorMatrix1></dcpData>')
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cam.dcp")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(xml)
            source = ScientificFilmPipeline()
            source.load_icc_profile(path)
            data = source.to_settings()
            self.assertEqual(data["icc_source"], os.path.abspath(path))
            target = ScientificFilmPipeline()
            sm.apply_settings_to_pipeline(target, data)
            self.assertTrue(target.icc_loaded)

    def test_r11_lut_applied_at_most_once(self):
        """R11：旧实现导出时内外各套一次 LUT（双重套用）。

        v5 语义：套不套由**显式输出变换**决定，``process_for_output()``
        恰好套一次；导出端不再重复套。这里断言"绝不出现 2 次"。
        """
        class Scale:
            calls = 0

            def apply(self, values):
                Scale.calls += 1
                return np.clip(np.asarray(values) * 2, 0, 1)

        img, base = synth.make_negative(16, 24, noise=0.0)
        pipe = ScientificFilmPipeline()
        pipe.load_linear_image(img)
        pipe.profile = _profile()
        pipe.set_base_val(base)
        pipe.set_output_lut(Scale(), enabled=True, path="x",
                            target_space="rec709")
        pipe.process_for_output()
        self.assertEqual(Scale.calls, 1, "启用 LUT 后应当恰好套一次")
        pipe.process_for_output(apply_lut=False)     # 对数直通，不再套
        self.assertEqual(Scale.calls, 1, "显式关掉后不得再套")
        # 关掉 LUT 后任何导出都不得再套
        pipe.set_output_lut(None, enabled=False)
        pipe.process_for_output()
        self.assertEqual(Scale.calls, 1)

    def test_r12_optional_dependencies_are_lazy(self):
        """R12：旧实现在模块顶层 import rawpy，缺依赖直接崩溃。"""
        for module in (io_read,):
            tree = ast.parse(inspect.getsource(module))
            imports = []
            for node in tree.body:
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module)
            self.assertNotIn("rawpy", imports)
        # app.py 只在函数内 import tkinter（这里验证它可以无 GUI 导入）
        self.assertTrue(hasattr(importlib.import_module("aurhythm.app"),
                                "AurhythmApp"))

    def test_r13_batch_is_cancellable_with_progress(self):
        """R13：旧批量导出跑在主线程且无进度回调（界面冻结）。"""
        from aurhythm import batch as batch_mod
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for index in range(3):
                path = os.path.join(tmp, f"n{index}.tif")
                from aurhythm import io_write
                io_write.write_tiff(path, synth.make_negative(16, 24,
                                                             noise=0.0)[0], 16)
                paths.append(path)
            seen = []
            calls = {"n": 0}

            def progress(index, total, item):
                seen.append((index, total))

            def cancel():
                calls["n"] += 1
                return calls["n"] > 1

            result = batch_mod.run_batch(paths, {}, progress=progress,
                                         cancel=cancel)
            self.assertTrue(seen, "必须回调进度")
            self.assertGreaterEqual(result.count("canceled"), 1)

    def test_r14_profile_chain_applies_cat(self):
        """R14：相机链必须含 D50→D65 适应与 XYZ→sRGB。"""
        # 用非单位矩阵构造 profile，链的输出必须等于手工计算的组合
        from aurhythm import colorimetry as cm
        cam_to_xyz = np.array([[0.3, 0.4, 0.2],
                               [0.2, 0.7, 0.1],
                               [0.0, 0.1, 0.8]])
        profile = prof.profile_from_matrix(cam_to_xyz)
        rgb = np.array([[[0.2, 0.3, 0.4]]])
        got = prof.camera_to_linear_srgb(rgb, profile)
        expected = (cm.MATRIX_XYZ_D65_TO_SRGB @ cm.CAT_D50_TO_D65
                    @ cam_to_xyz) @ rgb.reshape(3)
        np.testing.assert_allclose(got.reshape(3), expected, atol=1e-12)
        self.assertFalse(np.allclose(got.reshape(3), rgb.reshape(3)))

    def test_r14_lut_type_icc_raises(self):
        """R14 附带：LUT 型 ICC 必须明确报错，而不是静默返回单位矩阵。"""
        # 伪造一个带 "mAB "（4 字符，含尾空格）标签的最小 ICC
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lut.icc")
            data_offset = 128 + 4 + 12
            blob = bytearray(128)
            blob[0:4] = (data_offset + 8).to_bytes(4, "big")
            blob[36:40] = b"acsp"
            blob += (1).to_bytes(4, "big")                       # tag count
            blob += b"mAB " + data_offset.to_bytes(4, "big") \
                + (8).to_bytes(4, "big")
            blob += b"mAB \x00\x00\x00\x00"
            with open(path, "wb") as fh:
                fh.write(bytes(blob))
            with self.assertRaises(Exception) as ctx:
                prof.load_icc(path)
            self.assertIn("LUT", str(ctx.exception))

    def test_r15_shoulder_is_not_hard_clipped(self):
        """R15：旧实现先 `t = clip(...)` 再进 sigmoid，软裁剪完全失效。"""
        tf = TransferFunction(hd=FILM_PRESETS["通用负片 (默认)"].hd)
        p = tf.params
        # 超出 d_max 的密度必须仍然单调、且逐步逼近白点（软肩，不是硬裁）
        density = np.repeat(
            np.linspace(p.d_max, p.d_max + 1.0, 64)[:, None], 3, axis=1)
        code, _ = tf.density_to_code(density)
        self.assertTrue(np.all(np.diff(code[:, 0]) >= 0.0))
        self.assertGreater(float(code[32, 0]), float(code[0, 0]),
                           "肩部应当是软压缩而不是立刻平掉")
        # 渐近行为：远离窗口后收敛到参考白
        far, _ = tf.density_to_code(
            np.repeat(np.array([[p.d_max + 50.0]]), 3, axis=1))
        self.assertLess(abs(float(far[0, 0]) - CINEON.white_code), 1.0)


if __name__ == "__main__":
    unittest.main()
