"""GUI 测试。

本环境有真实的 tkinter（Tk 9.0.4）但没有显示器，因此：

* 能建真实 ``tk.Tk()`` 时（``withdraw()`` 掉），**真实构建整套界面并驱动交互**
  —— 这能抓到真正的 tkinter API 错误（错误的控件参数、缺失属性、布局问题）；
* 建不了 Tk 时自动跳过，仍然保留两组**结构测试**：
  1. 用 AST 断言 UI 为每个「进管线」的注册参数生成了控件（S13/R7 的 UI 侧保证）；
  2. 断言调色控件只出现在「调色」标签页里（工具定位不漂移）。
"""

from __future__ import annotations

import ast
import importlib
import inspect
import os
import sys
import tempfile
import types
import unittest

import numpy as np

import tests  # noqa: F401
from aurhythm import io_write, params as pm, settings as sm, synth

#: 纯 UI 状态键（不进管线，UI 不需要生成控件）
UI_ONLY_KEYS = {"export_format"}


#: Tk 可用性只探测一次 —— 在同一进程里反复创建/销毁 Tk 根窗口
#: 在 macOS 上是已知的崩溃源，因此这里缓存结果。
_TK_AVAILABLE = None


def _real_tk_available() -> bool:
    global _TK_AVAILABLE
    if _TK_AVAILABLE is None:
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            root.destroy()
            _TK_AVAILABLE = True
        except Exception:                               # noqa: BLE001
            _TK_AVAILABLE = False
    return _TK_AVAILABLE


def _install_tkinter_stubs():
    """仅在真实 tkinter 不可用时安装桩。"""
    for name in ("tkinter", "tkinter.ttk", "tkinter.filedialog",
                 "tkinter.messagebox", "tkinter.simpledialog"):
        sys.modules.setdefault(name, types.ModuleType(name))
    tk = sys.modules["tkinter"]
    ttk = sys.modules["tkinter.ttk"]
    tk.ttk = ttk
    for sub in ("filedialog", "messagebox", "simpledialog"):
        setattr(tk, sub, sys.modules[f"tkinter.{sub}"])


def _stub_dialogs(module):
    """把弹窗替换成记录器，避免测试阻塞。"""
    calls = []

    class _Box:
        @staticmethod
        def showinfo(title, message, **kwargs):
            calls.append(("info", title, message))

        @staticmethod
        def showwarning(title, message, **kwargs):
            calls.append(("warning", title, message))

        @staticmethod
        def showerror(title, message, **kwargs):
            calls.append(("error", title, message))

        @staticmethod
        def askyesno(title, message, **kwargs):
            calls.append(("askyesno", title, message))
            return True

    module.messagebox = _Box
    return calls


class TestAppStructure(unittest.TestCase):
    """不依赖 Tk 的结构性断言（任何环境都能跑）。"""

    @classmethod
    def setUpClass(cls):
        if not _real_tk_available():
            _install_tkinter_stubs()
        sys.modules.pop("aurhythm.app", None)
        cls.module = importlib.import_module("aurhythm.app")

    def test_public_api(self):
        self.assertTrue(hasattr(self.module, "AurhythmApp"))
        self.assertTrue(callable(self.module.main))
        self.assertIn("look", self.module.DARK_THEME)

    def test_no_top_level_heavy_imports(self):
        tree = ast.parse(inspect.getsource(self.module))
        top_level = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_level.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_level.append(node.module)
        for forbidden in ("rawpy", "tifffile", "imageio"):
            self.assertNotIn(forbidden, top_level)

        from aurhythm import io_read
        tree = ast.parse(inspect.getsource(io_read))
        imports = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        self.assertNotIn("rawpy", imports)
        self.assertNotIn("PIL", [name.split(".")[0] for name in imports])

    def test_ui_covers_every_registered_param(self):
        """S13：UI 必须为每个「进管线」的注册参数生成控件。"""
        tree = ast.parse(inspect.getsource(self.module))
        used = set()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "attr", None) == "_param_rows"):
                for arg in node.args[1:]:
                    if isinstance(arg, ast.List):
                        used |= {e.value for e in arg.elts
                                 if isinstance(e, ast.Constant)
                                 and isinstance(e.value, str)}
        expected = set(pm.PARAM_SPECS) - UI_ONLY_KEYS
        self.assertEqual(expected - used, set(),
                         f"UI 漏了这些参数控件: {sorted(expected - used)}")
        self.assertEqual(used - set(pm.PARAM_SPECS), set())

    def test_look_controls_only_in_look_tab(self):
        tree = ast.parse(inspect.getsource(self.module))
        look_lines = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_build_tab_look":
                look_lines = {child.lineno for child in ast.walk(node)
                              if isinstance(child, ast.Call)}
        used_in_look = set()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "attr", None) == "_param_rows"
                    and node.lineno in look_lines):
                for arg in node.args[1:]:
                    if isinstance(arg, ast.List):
                        used_in_look |= {e.value for e in arg.elts
                                         if isinstance(e, ast.Constant)}
        self.assertEqual(used_in_look, set(pm.look_keys()))


@unittest.skipUnless(_real_tk_available(), "本环境无法创建 Tk 根窗口")
class TestRealGui(unittest.TestCase):
    """真实构建界面并驱动交互。"""

    @classmethod
    def setUpClass(cls):
        import tkinter as tk
        from aurhythm.app import AurhythmApp

        cls.tk = tk
        cls.root = tk.Tk()
        cls.root.withdraw()
        cls.dialogs = _stub_dialogs(sys.modules["aurhythm.app"])
        cls.app = AurhythmApp(root=cls.root, start_mainloop=False)

    @classmethod
    def tearDownClass(cls):
        try:
            cls.app.on_closing()
        except Exception:                               # noqa: BLE001
            pass

    def _load_synthetic(self, cast=(1.1, 1.0, 0.9)):
        image, base = synth.make_negative(height=48, width=64, cast=cast,
                                          noise=0.0, seed=1)
        image_id = max(self.app.images, default=-1) + 1
        self.app.images[image_id] = {
            "path": f"/tmp/synthetic_{image_id}.tif", "name": "synthetic.tif",
            "pipeline": None, "loaded": True, "status": "就绪",
            "linear": image, "info": None,
        }
        self.app.tree.insert("", "end", iid=str(image_id), text="synthetic.tif",
                             values=("就绪", "❌", "❌"))
        self.app._on_loaded(image_id)
        self.app.current_id = image_id
        self.app.tree.selection_set(str(image_id))
        return image_id, base

    def test_widgets_were_built(self):
        self.assertTrue(self.app.param_vars)
        for key in set(pm.PARAM_SPECS) - UI_ONLY_KEYS:
            self.assertIn(key, self.app.param_vars, key)
        self.assertTrue(hasattr(self.app, "curve_canvas"))
        self.assertTrue(hasattr(self.app, "hist_canvas"))
        self.assertTrue(hasattr(self.app, "tree"))

    def test_load_image_and_sync(self):
        self._load_synthetic()
        pipe = self.app._pipeline()
        self.assertIsNotNone(pipe)
        self.app._sync_all_widgets()
        self.assertAlmostEqual(self.app.param_vars["hd_a"].get(),
                               float(pipe.hd.a), places=6)

    def test_pipeline_roundtrip_through_widgets(self):
        """改控件 → 写进管线 → 再从管线读回控件。"""
        self._load_synthetic()
        self.app.param_vars["hd_a"].set(1.6)
        self.app.on_param_slider("hd_a")
        self.assertAlmostEqual(self.app._pipeline().hd.a, 1.6, places=6)

        self.app.param_vars["gain_r"].set(1.3)
        self.app.on_param_slider("gain_r")
        self.assertAlmostEqual(
            float(np.asarray(self.app._pipeline().channel_gains)[0]), 1.3,
            places=6)
        self.app.reset_gains()

        self.app.param_vars["cdl_gain_b"].set(1.4)
        self.app.on_param_slider("cdl_gain_b")
        self.assertAlmostEqual(self.app._pipeline().tone.gain[2], 1.4, places=6)
        self.app.reset_look()

    def test_entry_widget_updates_pipeline(self):
        self._load_synthetic()
        entry = self.app.param_entries["tone_contrast"]
        entry.delete(0, self.tk.END)
        entry.insert(0, "1.75")
        self.app.on_param_entry("tone_contrast")
        self.assertAlmostEqual(self.app._pipeline().tone.contrast, 1.75,
                               places=6)
        self.app.reset_look()

    def test_preset_selection_updates_curve_and_plot(self):
        self._load_synthetic()
        self.app.preset_var.set("Kodak Vision3 250D (5207)")
        self.app.on_preset_change()
        self.assertEqual(self.app._pipeline().preset_name,
                         "Kodak Vision3 250D (5207)")
        self.assertTrue(self.app.curve_canvas.find_all())

    def test_base_detection_and_neutralization(self):
        self._load_synthetic(cast=(1.15, 1.0, 0.85))
        self.app.auto_base()
        pipe = self.app._pipeline()
        self.assertIsNotNone(pipe.base_val_rgb)
        self.app.neutralize()
        achieved = np.array(pipe.density_report.achieved)
        np.testing.assert_allclose(achieved, pipe.target_density, atol=0.01)

    def test_base_auto_detected_on_load(self):
        """导入即自动检测片基 —— 否则导入后预览是空的（黑屏 bug）。"""
        image_id, _base = self._load_synthetic()
        pipe = self.app._pipeline()
        self.assertIsNotNone(pipe.base_val_rgb, "导入后应当已自动检测片基")
        self.assertTrue(self.app.images[image_id].get("base_auto"))
        self.assertIn("自动检测", self.app.base_label.cget("text"))

    def test_preview_never_blank_after_load(self):
        """回归：导入后必须立刻能画出预览（曾经因为没片基直接返回 None）。"""
        image_id, _base = self._load_synthetic()
        preview = self.app._render_preview(self.app.images[image_id])
        self.assertIsNotNone(preview, "导入后预览不应为空")
        self.assertGreater(int(preview.max()), 30)

    def test_preview_without_base_uses_provisional(self):
        """即使片基被清空，也要用临时估计渲染，而不是给空画布。"""
        image_id, _base = self._load_synthetic()
        pipe = self.app._pipeline()
        pipe.base_val_rgb = None
        pipe.base_samples = []
        preview = self.app._render_preview(self.app.images[image_id])
        self.assertIsNotNone(preview)
        self.assertGreater(int(preview.max()), 30)
        self.assertIsNone(pipe.base_val_rgb, "临时估计不得写回管线状态")

    def test_black_frame_gives_placeholder_not_white(self):
        """全黑帧必须给中性提示图，而不是被算成纯白。"""
        image_id = max(self.app.images, default=-1) + 1
        black = np.zeros((32, 32, 3), dtype=np.float32)
        self.app.images[image_id] = {
            "path": "/tmp/black.tif", "name": "black.tif", "pipeline": None,
            "loaded": True, "status": "就绪", "linear": black, "info": None}
        self.app.tree.insert("", "end", iid=str(image_id), text="black.tif")
        self.app._on_loaded(image_id)
        self.app.current_id = image_id
        pipe = self.app._pipeline()
        self.assertIsNone(pipe.base_val_rgb, "无信号帧不应产生片基")
        preview = self.app._render_preview(self.app.images[image_id])
        self.assertIsNotNone(preview)
        self.assertLess(int(preview.max()), 100, "不应被算成高光/纯白")

    def test_manual_base_sampling_replaces_auto_value(self):
        """手动采样应当重新开始一组，而不是与自动检测值混在一起。"""
        self._load_synthetic()
        self.app.sampling_mode = True
        self.app._sample_base(2, 2)
        self.app._sample_base(4, 4)
        pipe = self.app._pipeline()
        self.assertEqual(len(pipe.base_samples), 2)
        self.assertIsNotNone(pipe.base_val_rgb)
        self.assertFalse(self.app.images[self.app.current_id].get("base_auto"))

    def test_measure_density_range_updates_the_params(self):
        """「测量密度范围 → 填入上下限」必须真的写进参数（不是只改状态栏）。"""
        self._load_synthetic()
        pipe = self.app._pipeline()
        pm.apply_param(pipe, "hd_d_min", 0.55)
        pm.apply_param(pipe, "hd_d_max", 0.60)
        self.app.measure_density_range()
        self.assertNotAlmostEqual(pipe.transfer.params.d_max, 0.60, places=3,
                                  msg="按钮没有更新密度上限")
        self.assertAlmostEqual(pipe.transfer.params.d_min, 0.0, places=3)
        # 控件也要同步（否则界面显示的还是旧值）
        self.assertAlmostEqual(
            float(self.app.param_entries["hd_d_max"].get()), 
            pipe.transfer.params.d_max, places=3)

    def test_sampling_black_pixel_is_rejected(self):
        """点到纯黑区域必须被拒绝，而不是把整条密度链弄坏。"""
        image_id, _base = self._load_synthetic()
        self.app.images[image_id]["linear"][20, 20] = 0.0
        before = self.app._pipeline().base_val_rgb
        self.app._sample_base(20, 20)
        after = self.app._pipeline().base_val_rgb
        np.testing.assert_allclose(after, before)

    def test_preview_render_and_display(self):
        image_id, _base = self._load_synthetic()
        self.app.auto_base()
        record = self.app.images[image_id]
        preview = self.app._render_preview(record)
        self.assertIsNotNone(preview)
        self.assertEqual(preview.dtype, np.uint8)
        self.app._show_preview(preview, 12.0, image_id)
        self.assertIsNotNone(self.app.canvas_photo)
        self.assertTrue(self.app.canvas.find_all())

    def test_preview_discards_stale_results(self):
        image_id, _base = self._load_synthetic()
        self.app.auto_base()
        preview = self.app._render_preview(self.app.images[image_id])
        self.app.canvas_photo = None
        self.app._show_preview(preview, 1.0, image_id + 999)   # 过期
        self.assertIsNone(self.app.canvas_photo)

    def test_analysis_and_plots(self):
        image_id, _base = self._load_synthetic()
        self.app.auto_base()
        preview = self.app._render_preview(self.app.images[image_id])
        self.app._show_preview(preview, 5.0, image_id)
        self.app.refresh_analysis()
        text = self.app.stats_text.get("1.0", self.tk.END)
        self.assertIn("管线耗时", text)
        self.assertTrue(self.app.hist_canvas.find_all())
        self.assertTrue(self.app.curve_canvas.find_all())

    def test_measure_density_range_button(self):
        self._load_synthetic()
        self.app.auto_base()
        self.app.measure_density_range()
        pipe = self.app._pipeline()
        self.assertGreater(pipe.hd.d_max, pipe.hd.d_min)

    def test_reset_view(self):
        self._load_synthetic()
        self.app.param_vars["view_exposure_ev"].set(1.5)
        self.app.on_param_slider("view_exposure_ev")
        self.app.reset_view()
        self.assertAlmostEqual(self.app._pipeline().view.exposure_ev, 0.0,
                               places=6)

    def test_undo_redo(self):
        self._load_synthetic()
        self.app.param_vars["hd_a"].set(2.0)
        self.app.on_param_slider("hd_a")
        self.assertAlmostEqual(self.app._pipeline().hd.a, 2.0, places=6)
        self.app.undo()
        self.assertNotAlmostEqual(self.app._pipeline().hd.a, 2.0, places=3)
        self.app.redo()
        self.assertAlmostEqual(self.app._pipeline().hd.a, 2.0, places=6)

    def test_settings_save_and_load_through_gui_paths(self):
        self._load_synthetic()
        self.app.auto_base()
        self.app.neutralize()
        pipe = self.app._pipeline()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "s.json")
            sm.save_settings(path, pipe.to_settings())
            pipe.set_preset("无 (关闭预设)")
            report = pipe.apply_settings(sm.load_settings(path))
            self.assertIsInstance(report, list)
        self.app._sync_all_widgets()

    def test_export_current_writes_file(self):
        """走与「导出当前」相同的代码路径，但不弹保存对话框。"""
        self._load_synthetic()
        self.app.auto_base()
        pipe = self.app._pipeline()
        data = pipe.process_for_output()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.dpx")
            io_write.write_image(path, data, "dpx10",
                                 metadata=pipe.to_settings())
            self.assertTrue(os.path.exists(path))
            self.assertTrue(os.path.exists(os.path.join(tmp, "out.json")))

    def test_chart_detection_and_solve_from_gui(self):
        from aurhythm import colorchecker as cc
        from tests.test_colorchecker import render_chart, planted_transform

        target = cc.target_linear_srgb()
        corners = np.array([[70.0, 55.0], [470.0, 45.0],
                            [490.0, 310.0], [60.0, 320.0]])
        chart = render_chart(360, 540, corners=corners,
                             colors=target @ planted_transform().T)
        image_id = max(self.app.images, default=-1) + 1
        self.app.images[image_id] = {
            "path": "/tmp/chart.tif", "name": "chart.tif", "pipeline": None,
            "loaded": True, "status": "就绪", "linear": chart, "info": None,
        }
        self.app.tree.insert("", "end", iid=str(image_id), text="chart.tif")
        self.app._on_loaded(image_id)
        self.app.current_id = image_id

        self.app.detect_chart()
        pipe = self.app._pipeline()
        self.assertTrue(pipe.colorchecker_calibrated)
        self.assertLess(pipe.error_analysis["mean_deltaE"], 1.5)
        self.assertTrue(self.app.calib_label.cget("text"))

    def test_batch_copy_settings(self):
        first, _b1 = self._load_synthetic()
        second, _b2 = self._load_synthetic()
        self.app.current_id = first
        self.app.param_vars["hd_a"].set(1.9)
        self.app.on_param_slider("hd_a")
        self.app.batch_copy_settings()
        self.assertAlmostEqual(self.app.images[second]["pipeline"].hd.a, 1.9,
                               places=6)

    def test_remove_and_clear(self):
        image_id, _base = self._load_synthetic()
        self.app.tree.selection_set(str(image_id))
        self.app.remove_selected()
        self.assertNotIn(image_id, self.app.images)
        self._load_synthetic()
        self.app.clear_images()
        self.assertEqual(self.app.images, {})


class TestLegacyEntry(unittest.TestCase):
    def test_launcher_imports_and_reexports(self):
        module = importlib.import_module("Aurhythm")
        for name in ("ScientificFilmPipeline", "CubeLUT", "ICCLoader",
                     "ColorCheckerCalibration", "FILM_PRESETS",
                     "COLORCHECKER_24_D50", "apply_dark_theme"):
            self.assertTrue(hasattr(module, name), name)
        self.assertEqual(module.COLORCHECKER_24_D50.shape, (24, 3))

    def test_cli_help_via_launcher(self):
        import Aurhythm
        self.assertEqual(Aurhythm.main(["info", "--help"]), 0)


if __name__ == "__main__":
    unittest.main()
