"""Aurhythm 图形界面（唯一允许 import tkinter 的模块）。

设计原则
--------
* **校准优先**：默认路径是纯物理映射（白平衡 → 相机 profile → 色卡 →
  密度 → 中性化 → 解串扰 → 胶片特性曲线 → Cineon/LogC3）。所有调色类
  控件集中在「调色（可选·非校准）」标签页，默认全部恒等，并在界面上
  明确标注，避免工具定位漂移。
* **控件由注册表驱动**：所有参数来自 :mod:`aurhythm.params`，
  UI 不手写任何「键名 ↔ 属性名」映射，因此不会出现滑块接线错误
  （旧版 ``hd_softness`` / ``hd_clip_softness`` 错配导致控件完全无效）。
* **预览 latest-wins**：渲染线程只处理最新请求，滑杆拖动有防抖，
  在预览分辨率上渲染 —— 拖动不再卡顿，也不会显示过期结果。
* **批量不冻结界面**：批量处理与批量导出都在工作线程里跑，
  进度经 ``root.after`` 回主线程。
* 所有界面更新都发生在主线程（工作线程一律用 ``root.after`` 投递）。
"""

from __future__ import annotations

import os
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import numpy as np

from . import batch as batch_mod
from . import colorchecker as cc
from . import display as display_mod
from . import io_read
from . import io_write
from . import params as pm
from . import settings as sm
from .constants import FILM_PRESETS, PRESET_ORDER

DARK_THEME = {
    "bg": "#1a1a1a", "fg": "#e0e0e0", "frame_bg": "#2d2d2d",
    "panel_bg": "#232323", "button_bg": "#3c3c3c", "button_fg": "#ffffff",
    "canvas_bg": "#0a0a0a", "accent": "#00b4d8", "warning": "#ffaa00",
    "warn": "#ffaa00",
    "success": "#00d47a", "error": "#ff4444", "look": "#c792ea",
    "muted": "#8a8a8a",
}

#: 预览渲染的最大边长（超出则降采样渲染）
MAX_PREVIEW_EDGE = 1600


def apply_dark_theme(root):
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:                                 # pragma: no cover
        pass
    style.configure(".", background=DARK_THEME["bg"], foreground=DARK_THEME["fg"],
                    fieldbackground=DARK_THEME["panel_bg"])
    style.configure("TLabel", background=DARK_THEME["bg"], foreground=DARK_THEME["fg"])
    style.configure("TFrame", background=DARK_THEME["bg"])
    style.configure("Panel.TFrame", background=DARK_THEME["panel_bg"])
    style.configure("TLabelframe", background=DARK_THEME["panel_bg"],
                    foreground=DARK_THEME["fg"], relief="flat", borderwidth=1)
    style.configure("TLabelframe.Label", background=DARK_THEME["panel_bg"],
                    foreground=DARK_THEME["fg"])
    style.configure("TButton", background=DARK_THEME["button_bg"],
                    foreground=DARK_THEME["button_fg"], padding=5)
    style.map("TButton", background=[("active", DARK_THEME["accent"])])
    style.configure("TEntry", fieldbackground=DARK_THEME["panel_bg"],
                    foreground=DARK_THEME["fg"])
    style.configure("TCombobox", fieldbackground=DARK_THEME["panel_bg"],
                    foreground=DARK_THEME["fg"])
    style.configure("TCheckbutton", background=DARK_THEME["panel_bg"],
                    foreground=DARK_THEME["fg"])
    style.configure("TNotebook", background=DARK_THEME["bg"])
    style.configure("TNotebook.Tab", background=DARK_THEME["button_bg"],
                    foreground=DARK_THEME["fg"], padding=(8, 4))
    style.map("TNotebook.Tab", background=[("selected", DARK_THEME["accent"])])
    style.configure("TScale", background=DARK_THEME["panel_bg"],
                    troughcolor=DARK_THEME["button_bg"])
    style.configure("Treeview", background=DARK_THEME["panel_bg"],
                    fieldbackground=DARK_THEME["panel_bg"],
                    foreground=DARK_THEME["fg"])
    return style


class AurhythmApp:
    """主窗口。"""

    def __init__(self, root: tk.Tk | None = None, *, start_mainloop: bool = True):
        self.root = root or tk.Tk()
        self.root.title("Aurhythm 胶片 Cineon 校准器 v5.0")
        self.root.geometry("1680x1020")
        self.root.configure(bg=DARK_THEME["bg"])
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        apply_dark_theme(self.root)

        # ---------- 图像管理 ----------
        self.images: dict[int, dict] = {}
        self.current_id: int | None = None
        self.render_scale = 0.25
        self.display_scale = 1.0
        self.display_offset = (0, 0)
        self.canvas_photo = None
        self._preview_source: np.ndarray | None = None

        # ---------- 交互状态 ----------
        self.sampling_mode = False
        self.sampling_points: list = []
        self.corner_points: list = []
        self.corner_mode = False

        # ---------- 渲染线程（latest-wins + 防抖） ----------
        self._render_request = None
        self._render_lock = threading.Lock()
        self._render_event = threading.Event()
        self._render_stop = False
        self._alive = True
        self._render_thread = None
        self._debounce_ms = 120
        self._debounce_job = None
        self._render_busy = False
        self._ui_queue = queue.Queue()
        self._ui_job = None

        # ---------- 撤销栈 ----------
        self._undo: list = []
        self._redo: list = []
        self._undo_limit = 50

        self.param_vars: dict[str, tk.Variable] = {}
        self.param_entries: dict[str, ttk.Entry] = {}

        self._build_ui()
        self._start_render_thread()
        self._ui_job = self.root.after(40, self._poll_ui)
        self.root.bind("<Control-z>", lambda _e: self.undo())
        self.root.bind("<Control-y>", lambda _e: self.redo())
        self.root.bind("<Control-Z>", lambda _e: self.redo())
        if start_mainloop:
            self.root.mainloop()

    # ==================================================================
    # 界面搭建
    # ==================================================================

    def _build_ui(self):
        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        left = ttk.Frame(main, width=290)
        self._build_image_panel(left)
        main.add(left, weight=0)

        center = ttk.Frame(main)
        self._build_preview_panel(center)
        main.add(center, weight=1)

        right = ttk.Frame(main, width=560)
        self._build_tabs(right)
        main.add(right, weight=0)

        self._sync_all_widgets()
        self._redraw_curve()

    # ---------------- 左：图像列表 ----------------
    def _build_image_panel(self, parent):
        ttk.Label(parent, text="📁 图像", font=("Helvetica", 14, "bold")).pack(anchor=tk.W)
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=(4, 6))
        ttk.Button(row, text="导入", command=self.import_images, width=8).pack(side=tk.LEFT)
        ttk.Button(row, text="移除", command=self.remove_selected, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="清空", command=self.clear_images, width=8).pack(side=tk.LEFT)

        box = ttk.Labelframe(parent, text="图像列表", padding=4)
        box.pack(fill=tk.BOTH, expand=True)
        self.tree = ttk.Treeview(box, columns=("status", "calib", "preset"),
                                 show="tree headings", height=22, selectmode="extended")
        self.tree.heading("#0", text="文件")
        self.tree.column("#0", width=140)
        for key, label, width in (("status", "状态", 60), ("calib", "色卡", 50),
                                  ("preset", "预设", 50)):
            self.tree.heading(key, text=label)
            self.tree.column(key, width=width)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        bar = ttk.Scrollbar(box, orient=tk.VERTICAL, command=self.tree.yview)
        bar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=bar.set)
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self.on_select_image())

        actions = ttk.Labelframe(parent, text="批量", padding=4)
        actions.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(actions, text="复制参数到全部", width=18,
                   command=self.batch_copy_settings).pack(fill=tk.X, pady=2)
        ttk.Button(actions, text="批量处理（检测片基+中性化）", width=18,
                   command=self.batch_process).pack(fill=tk.X, pady=2)
        self.progress = ttk.Progressbar(parent, orient=tk.HORIZONTAL, mode="determinate")
        self.progress.pack(fill=tk.X, pady=(6, 2))
        self.status_label = ttk.Label(parent, text="就绪", foreground=DARK_THEME["muted"])
        self.status_label.pack(anchor=tk.W)

    # ---------------- 中：预览 ----------------
    def _build_preview_panel(self, parent):
        bar = ttk.Frame(parent)
        bar.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(bar, text="预览").pack(side=tk.LEFT)
        self.scale_var = tk.StringVar(value="25%")
        ttk.OptionMenu(bar, self.scale_var, "25%", "100%", "50%", "25%", "12.5%",
                       command=lambda _v: self._on_scale_change()).pack(side=tk.LEFT, padx=4)
        self.sample_var = tk.StringVar(value="点击采样模式：关闭")
        ttk.Label(bar, textvariable=self.sample_var,
                  foreground=DARK_THEME["accent"]).pack(side=tk.LEFT, padx=10)
        self.info_label = ttk.Label(bar, text="", foreground=DARK_THEME["muted"])
        self.info_label.pack(side=tk.RIGHT)

        self.canvas = tk.Canvas(parent, bg=DARK_THEME["canvas_bg"],
                                highlightthickness=0, height=560)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Motion>", self.on_canvas_move)

        bottom = ttk.Frame(parent)
        bottom.pack(fill=tk.X, pady=(4, 0))
        self.cursor_label = ttk.Label(bottom, text="", relief="sunken")
        self.cursor_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.stats_label = ttk.Label(bottom, text="", relief="sunken")
        self.stats_label.pack(side=tk.RIGHT)

    # ---------------- 右：标签页 ----------------
    def _build_tabs(self, parent):
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)

        tabs = [
            ("🎯 色彩校准", self._build_tab_color),
            ("🎞 非线性校正", self._build_tab_film),
            ("👁 显示", self._build_tab_view),
            ("📊 分析", self._build_tab_analysis),
            ("📤 导出", self._build_tab_export),
            ("🎨 调色（可选·非校准）", self._build_tab_look),
        ]
        for title, builder in tabs:
            frame = ttk.Frame(notebook)
            builder(frame)
            notebook.add(frame, text=title)

    @staticmethod
    def _scrollable(parent):
        canvas = tk.Canvas(parent, bg=DARK_THEME["panel_bg"], highlightthickness=0)
        bar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>",
                   lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=bar.set)
        canvas.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        return inner

    # ---- 色彩校准 ----
    def _build_tab_color(self, parent):
        inner = self._scrollable(parent)
        profile = ttk.Labelframe(inner, text="相机 profile（ICC / DCP）", padding=8)
        profile.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(profile, text="导入 .dcp/.icc", command=self.import_profile,
                   width=16).pack(anchor=tk.W)
        self.profile_label = ttk.Label(profile, text="未加载",
                                       foreground=DARK_THEME["muted"])
        self.profile_label.pack(anchor=tk.W, pady=4)
        self._param_rows(profile, ["icc_weight"])

        wb = ttk.Labelframe(inner, text="白平衡", padding=8)
        wb.pack(fill=tk.X, pady=(0, 8))
        row = ttk.Frame(wb)
        row.pack(fill=tk.X)
        ttk.Button(row, text="按 AsShotNeutral", width=16,
                   command=self.wb_from_neutral).pack(side=tk.LEFT)
        ttk.Button(row, text="关闭白平衡", width=12,
                   command=lambda: self._set_wb("none")).pack(side=tk.LEFT, padx=4)
        self.wb_label = ttk.Label(wb, text="未启用", foreground=DARK_THEME["muted"])
        self.wb_label.pack(anchor=tk.W, pady=4)

        base = ttk.Labelframe(inner, text="片基采样", padding=8)
        base.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(base,
                  text="导入时已自动检测片基；如需重采样：先点「采样模式」，"
                       "再在预览里点未曝光的透明边缘（可多次点击取中位数）",
                  foreground=DARK_THEME["muted"], wraplength=470,
                  justify=tk.LEFT).pack(anchor=tk.W)
        row = ttk.Frame(base)
        row.pack(fill=tk.X, pady=4)
        ttk.Button(row, text="采样模式", width=12,
                   command=self.toggle_sampling).pack(side=tk.LEFT)
        ttk.Button(row, text="自动检测片基", width=14,
                   command=self.auto_base).pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text="清除采样", width=12,
                   command=self.clear_base).pack(side=tk.LEFT)
        self.base_label = ttk.Label(base, text="未采样", foreground=DARK_THEME["muted"])
        self.base_label.pack(anchor=tk.W)

        align = ttk.Labelframe(inner, text="密度域中性化（逐通道）", padding=8)
        align.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(align, text="按图像密度中位数把逐通道密度对齐到目标值，消除中间调偏色",
                  foreground=DARK_THEME["muted"], wraplength=470).pack(anchor=tk.W)
        row = ttk.Frame(align)
        row.pack(fill=tk.X, pady=4)
        ttk.Button(row, text="中性化", width=12,
                   command=self.neutralize).pack(side=tk.LEFT)
        ttk.Button(row, text="增益归零", width=12,
                   command=self.reset_gains).pack(side=tk.LEFT, padx=4)
        self._param_rows(align, ["target_density", "gain_r", "gain_g", "gain_b",
                                 "clamp_cmy"])
        self.density_label = ttk.Label(align, text="", foreground=DARK_THEME["muted"])
        self.density_label.pack(anchor=tk.W, pady=4)

        checker = ttk.Labelframe(inner, text="色卡矫正（测量，非调色）", padding=8)
        checker.pack(fill=tk.X)
        ttk.Button(checker, text="导入 .ccmx/.json", width=16,
                   command=self.import_calibration).pack(anchor=tk.W, pady=2)
        row = ttk.Frame(checker)
        row.pack(fill=tk.X, pady=2)
        ttk.Button(row, text="自动检测色卡", width=14,
                   command=self.detect_chart).pack(side=tk.LEFT)
        ttk.Button(row, text="手动四角", width=10,
                   command=self.toggle_corners).pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text="求解并应用", width=12,
                   command=self.solve_from_corners).pack(side=tk.LEFT)
        self.calib_label = ttk.Label(checker, text="未校准",
                                     foreground=DARK_THEME["muted"])
        self.calib_label.pack(anchor=tk.W, pady=4)

    # ---- 非线性校正 ----
    def _build_tab_film(self, parent):
        inner = self._scrollable(parent)
        preset = ttk.Labelframe(inner, text="胶片预设", padding=8)
        preset.pack(fill=tk.X, pady=(0, 8))
        self.preset_var = tk.StringVar(value=FILM_PRESETS[PRESET_ORDER[0]].name)
        combo = ttk.Combobox(preset, textvariable=self.preset_var,
                             values=list(PRESET_ORDER), state="readonly", width=42)
        combo.pack(fill=tk.X)
        combo.bind("<<ComboboxSelected>>", lambda _e: self.on_preset_change())
        self.preset_note = ttk.Label(preset, text="", foreground=DARK_THEME["muted"],
                                     wraplength=470)
        self.preset_note.pack(anchor=tk.W, pady=4)

        curve = ttk.Labelframe(inner, text="胶片特性曲线（密度域）", padding=8)
        curve.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(curve,
                  text="曲线作用在归一化密度上；反差 1.0 = 保持原反差。"
                       "密度窗口请先用下方「测量密度范围 → 填入上下限」按本图实测值填好",
                  foreground=DARK_THEME["muted"], wraplength=470,
                  justify=tk.LEFT).pack(anchor=tk.W)
        self._param_rows(curve, ["hd_a", "hd_x_mid", "hd_s_toe", "hd_s_sh",
                                 "hd_d_min", "hd_d_max"])
        row = ttk.Frame(curve)
        row.pack(fill=tk.X, pady=4)
        ttk.Button(row, text="测量密度范围 → 填入上下限", width=26,
                   command=self.measure_density_range).pack(side=tk.LEFT)
        ttk.Button(row, text="重置曲线", width=10,
                   command=self.reset_curve).pack(side=tk.LEFT, padx=4)

        plot = ttk.Labelframe(inner, text="特性曲线预览", padding=8)
        plot.pack(fill=tk.X)
        self.curve_canvas = tk.Canvas(plot, bg=DARK_THEME["canvas_bg"], height=220,
                                      highlightthickness=0)
        self.curve_canvas.pack(fill=tk.X)

    # ---- 显示 ----
    def _build_tab_view(self, parent):
        inner = self._scrollable(parent)
        box = ttk.Labelframe(inner, text="预览显示（只影响屏幕，不影响导出）",
                             padding=8)
        box.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(box, text="预览由**导出数据**经这一变换得到，因此所见即所得",
                  foreground=DARK_THEME["muted"], wraplength=470).pack(anchor=tk.W)
        self._param_rows(box, ["view_exposure_ev", "view_gamma", "view_mode"])
        # 显示码率：由片种 γ 自动算出，这里给一个微调倍率。
        # 用标准 500 codes/decade 去解一条 γ=0.65 的负片会把画面压扁**发灰**，
        # 所以这个量是"看得对"的关键，且它**只影响屏幕**、不影响导出。
        self._param_rows(box, ["view_tone_scale"])
        ttk.Button(box, text="复位显示", command=self.reset_view).pack(
            anchor=tk.W, pady=6)

    # ---- 分析 ----
    def _build_tab_analysis(self, parent):
        inner = self._scrollable(parent)
        dE = ttk.Labelframe(inner, text="色卡 ΔE2000", padding=8)
        dE.pack(fill=tk.X, pady=(0, 8))
        self.error_text = tk.Text(dE, height=8, bg=DARK_THEME["canvas_bg"],
                                  fg=DARK_THEME["fg"], font=("Courier", 10))
        self.error_text.pack(fill=tk.X)
        ttk.Button(dE, text="刷新", command=self.refresh_analysis).pack(anchor=tk.W, pady=4)

        hist = ttk.Labelframe(inner, text="密度 / 直方图", padding=8)
        hist.pack(fill=tk.X, pady=(0, 8))
        self.hist_canvas = tk.Canvas(hist, bg=DARK_THEME["canvas_bg"], height=180,
                                     highlightthickness=0)
        self.hist_canvas.pack(fill=tk.X)

        stats = ttk.Labelframe(inner, text="统计与裁剪", padding=8)
        stats.pack(fill=tk.X)
        self.stats_text = tk.Text(stats, height=12, bg=DARK_THEME["canvas_bg"],
                                  fg=DARK_THEME["fg"], font=("Courier", 10))
        self.stats_text.pack(fill=tk.X)

    # ---- 导出 ----
    def _build_tab_export(self, parent):
        inner = self._scrollable(parent)
        box = ttk.Labelframe(inner, text="导出设置", padding=8)
        box.pack(fill=tk.X, pady=(0, 8))
        self._param_rows(box, ["output_colorspace", "export_format", "auto_fit",
                               "codes_per_density"])
        ttk.Label(box, text="TIFF 会内嵌完整参数（provenance）；DPX/EXR 另写 .json sidecar",
                  foreground=DARK_THEME["muted"], wraplength=470).pack(anchor=tk.W)

        # ---- 色彩还原 LUT：输出变成 LUT 的目标色彩空间 ----
        restore = ttk.Labelframe(
            inner, text="色彩还原 LUT（输出到 LUT 的目标色彩空间）", padding=8)
        restore.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(restore,
                  text="例：ARRI 官网的 LogC3 → Rec.709。套用后导出的**不再是"
                       "对数素材**，而是 Rec.709 显示素材（容器标注与 DPX "
                       "transfer 会随之改变）",
                  foreground=DARK_THEME["look"], wraplength=470,
                  justify=tk.LEFT).pack(anchor=tk.W)
        row = ttk.Frame(restore)
        row.pack(fill=tk.X, pady=4)
        ttk.Button(row, text="加载 .cube", width=12,
                   command=self.load_lut).pack(side=tk.LEFT)
        ttk.Button(row, text="卸载", width=8,
                   command=self.unload_lut).pack(side=tk.LEFT, padx=4)
        self.lut_label = ttk.Label(restore, text="未加载 LUT",
                                   foreground=DARK_THEME["muted"], wraplength=470)
        self.lut_label.pack(anchor=tk.W, pady=(0, 4))
        self._param_rows(restore, ["lut_enabled", "lut_input_space",
                                  "lut_target_space", "output_range"])
        self.output_label = ttk.Label(restore, text="", foreground=DARK_THEME["accent"],
                                      wraplength=470, justify=tk.LEFT)
        self.output_label.pack(anchor=tk.W, pady=(4, 0))

        # ---- 校准变换 .cube 导出 ----
        lut = ttk.Labelframe(inner, text="导出校准变换 .cube", padding=8)
        lut.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(lut, text="烘焙「密度→Cineon」校准变换，便于在 Resolve/Nuke 复用",
                  foreground=DARK_THEME["muted"], wraplength=470).pack(anchor=tk.W)
        row = ttk.Frame(lut)
        row.pack(fill=tk.X, pady=4)
        ttk.Button(row, text="导出 .cube（仅校准）", width=18,
                   command=lambda: self.export_lut(False)).pack(side=tk.LEFT)
        ttk.Button(row, text="含调色层", width=10,
                   command=lambda: self.export_lut(True)).pack(side=tk.LEFT, padx=4)

        # ---- 高级：DPX 字段覆盖 ----
        adv = ttk.Labelframe(inner, text="高级：DPX 字段（默认按输出空间自动）",
                             padding=8)
        adv.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(adv, text="默认值按输出空间自动给出；在 Resolve 里核对后如需"
                            "对齐机构规范可手动覆盖",
                  foreground=DARK_THEME["muted"], wraplength=470).pack(anchor=tk.W)
        self._param_rows(adv, ["dpx_transfer", "dpx_colorimetric"])

        actions = ttk.Labelframe(inner, text="执行", padding=8)
        actions.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(actions, text="导出当前图像", width=20,
                   command=self.export_current).pack(anchor=tk.W, pady=2)
        ttk.Button(actions, text="批量导出所选图像", width=20,
                   command=self.export_batch).pack(anchor=tk.W, pady=2)

        settings_box = ttk.Labelframe(inner, text="参数快照", padding=8)
        settings_box.pack(fill=tk.X)
        row = ttk.Frame(settings_box)
        row.pack(fill=tk.X, pady=2)
        ttk.Button(row, text="保存参数", width=12,
                   command=self.save_settings).pack(side=tk.LEFT)
        ttk.Button(row, text="加载参数", width=12,
                   command=self.load_settings).pack(side=tk.LEFT, padx=4)
        row = ttk.Frame(settings_box)
        row.pack(fill=tk.X, pady=2)
        ttk.Button(row, text="撤销 (Ctrl+Z)", width=14,
                   command=self.undo).pack(side=tk.LEFT)
        ttk.Button(row, text="重做", width=10, command=self.redo).pack(side=tk.LEFT, padx=4)

    # ---- 调色（可选·非校准） ----
    def _build_tab_look(self, parent):
        inner = self._scrollable(parent)
        warn = ttk.Label(
            inner,
            text="以下控件属于**调色**，不是校准。默认全部恒等；"
                 "导出 .cube 时默认不会烘入这些参数。",
            foreground=DARK_THEME["look"], wraplength=470, justify=tk.LEFT)
        warn.pack(anchor=tk.W, pady=(0, 8))

        tone = ttk.Labelframe(inner, text="曝光 / 对比度", padding=8)
        tone.pack(fill=tk.X, pady=(0, 8))
        self._param_rows(tone, ["tone_exposure_ev", "tone_contrast"])

        cdl = ttk.Labelframe(inner, text="ASC-CDL（Lift / Gamma / Gain）", padding=8)
        cdl.pack(fill=tk.X, pady=(0, 8))
        self._param_rows(cdl, ["tone_master_lift", "tone_master_gamma",
                               "tone_master_gain"])
        self._param_rows(cdl, ["cdl_lift_r", "cdl_lift_g", "cdl_lift_b",
                               "cdl_gamma_r", "cdl_gamma_g", "cdl_gamma_b",
                               "cdl_gain_r", "cdl_gain_g", "cdl_gain_b"])

        reset = ttk.Frame(inner)
        reset.pack(fill=tk.X)
        ttk.Button(reset, text="全部复位为恒等", width=18,
                   command=self.reset_look).pack(anchor=tk.W)

    # ---------------- 参数行工厂（注册表驱动） ----------------
    def _param_rows(self, parent, keys):
        for key in keys:
            spec = pm.spec(key)
            row = ttk.Frame(parent)
            row.pack(fill=tk.X, pady=2)
            colour = DARK_THEME["look"] if spec.look else DARK_THEME["fg"]
            ttk.Label(row, text=spec.label, width=16, foreground=colour).pack(side=tk.LEFT)
            if spec.kind == "bool":
                var = tk.BooleanVar(value=bool(spec.default))
                ttk.Checkbutton(row, variable=var,
                                command=lambda k=key: self.on_param_change(k)).pack(side=tk.LEFT)
            elif spec.kind == "choice":
                var = tk.StringVar(value=str(spec.default))
                combo = ttk.Combobox(row, textvariable=var, values=list(spec.choices),
                                     state="readonly", width=12)
                combo.pack(side=tk.LEFT)
                combo.bind("<<ComboboxSelected>>",
                           lambda _e, k=key: self.on_param_change(k))
            else:
                var = tk.DoubleVar(value=float(spec.default))
                ttk.Scale(row, from_=spec.lo, to=spec.hi, variable=var,
                          orient=tk.HORIZONTAL, length=220,
                          command=lambda _v, k=key: self.on_param_slider(k)).pack(
                    side=tk.LEFT, fill=tk.X, expand=True, padx=4)
                entry = ttk.Entry(row, width=9)
                entry.pack(side=tk.RIGHT)
                entry.bind("<Return>", lambda _e, k=key: self.on_param_entry(k))
                entry.bind("<FocusOut>", lambda _e, k=key: self.on_param_entry(k))
                self.param_entries[key] = entry
            self.param_vars[key] = var
            if spec.doc:
                ttk.Label(parent, text=f"　{spec.doc}", foreground=DARK_THEME["muted"],
                          wraplength=470, justify=tk.LEFT).pack(anchor=tk.W)

    # ==================================================================
    # 参数读写
    # ==================================================================

    def _pipeline(self):
        if self.current_id is None:
            return None
        record = self.images.get(self.current_id)
        return None if record is None else record["pipeline"]

    def on_param_slider(self, key):
        spec = pm.spec(key)
        var = self.param_vars[key]
        value = var.get()
        entry = self.param_entries.get(key)
        if entry is not None:
            entry.delete(0, tk.END)
            entry.insert(0, f"{value:.{spec.decimals}f}")
        self._push_undo()
        self._apply(key, value)

    def on_param_entry(self, key):
        spec = pm.spec(key)
        entry = self.param_entries.get(key)
        if entry is None:
            return
        try:
            value = float(entry.get())
        except ValueError:
            return
        self.param_vars[key].set(value)
        self._push_undo()
        self._apply(key, value)

    def on_param_change(self, key):
        self._push_undo()
        self._apply(key, self.param_vars[key].get())

    def _apply(self, key, value):
        pipe = self._pipeline()
        if pipe is None:
            return
        try:
            pm.apply_param(pipe, key, value)
        except (KeyError, ValueError) as exc:
            messagebox.showerror("参数错误", str(exc))
            return
        self._sync_labels()
        self.request_preview()
        if key.startswith("hd_"):
            self._redraw_curve()

    def _sync_all_widgets(self):
        pipe = self._pipeline()
        if pipe is None:
            return
        for key, var in self.param_vars.items():
            if key == "export_format":
                continue
            try:
                value = pm.read_param(pipe, key)
            except KeyError:
                continue
            var.set(value)
            entry = self.param_entries.get(key)
            if entry is not None:
                spec = pm.spec(key)
                entry.delete(0, tk.END)
                entry.insert(0, f"{float(value):.{spec.decimals}f}")
        if pipe.preset_name in FILM_PRESETS:
            self.preset_var.set(pipe.preset_name)
        self._sync_labels()

    def _sync_labels(self):
        pipe = self._pipeline()
        if pipe is None:
            return
        if pipe.profile is not None:
            self.profile_label.config(
                text=f"✅ {pipe.profile.source_name}（{pipe.profile.kind}）",
                foreground=DARK_THEME["success"])
        else:
            self.profile_label.config(text="未加载（按线性 sRGB 处理）",
                                      foreground=DARK_THEME["muted"])
        self.wb_label.config(
            text=f"启用：{'×'.join(f'{g:.3f}' for g in pipe.wb_gains)}"
            if pipe.wb_mode == "manual" else "未启用",
            foreground=DARK_THEME["success"] if pipe.wb_mode == "manual"
            else DARK_THEME["muted"])
        if pipe.base_val_rgb is not None:
            rgb = np.asarray(pipe.base_val_rgb)
            record = self.images.get(self.current_id) or {}
            source = "自动检测" if record.get("base_auto") else "手动采样"
            spread = float(rgb.max() - rgb.min())
            self.base_label.config(
                text=f"R={rgb[0]:.3f} G={rgb[1]:.3f} B={rgb[2]:.3f}"
                     f"（{source}，{len(pipe.base_samples)} 点，差异 {spread:.3f}）",
                foreground=DARK_THEME["warn"] if spread > 0.15
                else DARK_THEME["fg"])
        else:
            self.base_label.config(text="未采样", foreground=DARK_THEME["muted"])
        self.calib_label.config(
            text=f"✅ {pipe.calibration_source}｜ΔE2000 均值 "
                 f"{pipe.error_analysis['mean_deltaE']:.3f}"
            if pipe.colorchecker_calibrated and pipe.error_analysis
            else ("✅ 已加载矩阵" if pipe.colorchecker_calibrated else "未校准"),
            foreground=DARK_THEME["success"] if pipe.colorchecker_calibrated
            else DARK_THEME["muted"])
        if hasattr(self, "output_label"):
            self.output_label.config(
                text=f"当前导出：**{pipe.output.label()}**"
                     f"\nDPX transfer={pipe.output.transfer_code()} "
                     f"colorimetric={pipe.output.colorimetric_code()}")
        report = pipe.density_report
        if any(abs(v) > 1e-9 for v in report.per_channel_median):
            self.density_label.config(
                text=f"参考中位密度 "
                     f"{tuple(round(v, 3) for v in report.per_channel_median)}"
                     f" → 达成 {tuple(round(v, 3) for v in report.achieved)}"
                     f"（目标 {report.target}）")
        else:
            # 还没中性化：不要显示 (0,0,0) —— 那看着像坏了
            self.density_label.config(
                text="未中性化（点「中性化」按图像中位密度逐通道对齐到目标）")

    # ==================================================================
    # 图像管理
    # ==================================================================

    def import_images(self):
        paths = filedialog.askopenfilenames(
            title="选择图像（RAW / TIFF / PNG）",
            filetypes=[("所有支持", "*.nef *.nrw *.dng *.cr2 *.cr3 *.arw *.raf "
                                   "*.orf *.rw2 *.pef *.srw *.tif *.tiff *.png "
                                   "*.jpg *.jpeg"),
                       ("RAW", "*.nef *.nrw *.dng *.cr2 *.cr3 *.arw *.raf *.orf "
                               "*.rw2 *.pef *.srw"),
                       ("TIFF/PNG", "*.tif *.tiff *.png *.jpg *.jpeg"),
                       ("所有文件", "*.*")])
        if not paths:
            return
        for path in paths:
            self._add_image(path)

    def _add_image(self, path: str):
        image_id = max(self.images, default=-1) + 1
        self.images[image_id] = {
            "path": path, "name": os.path.basename(path),
            "pipeline": None, "loaded": False, "status": "加载中",
            "linear": None, "info": None,
        }
        self.tree.insert("", "end", iid=str(image_id),
                         text=os.path.basename(path),
                         values=("加载中", "❌", "❌"))
        threading.Thread(target=self._load_worker, args=(image_id,),
                         daemon=True).start()

    def _load_worker(self, image_id: int):
        record = self.images[image_id]
        try:
            linear, info = io_read.load_any(record["path"])
            record["linear"] = linear
            record["info"] = info
            record["loaded"] = True
            record["status"] = "就绪"
            self._post(lambda: self._on_loaded(image_id))
        except Exception as exc:                        # noqa: BLE001
            record["status"] = f"失败: {exc}"
            self._post(lambda: self._refresh_row(image_id))

    def _on_loaded(self, image_id: int):
        from .pipeline import ScientificFilmPipeline
        record = self.images[image_id]
        pipe = ScientificFilmPipeline()
        pipe.load_linear_image(record["linear"])
        # 导入即自动检测一次片基：这是**测量**而不是编辑，用户可以随时手动
        # 重新采样覆盖它。旧行为是"没采样前预览一片黑"，那是纯粹的坏体验。
        base = pipe.auto_detect_base()
        if base is not None:
            pipe.set_base_val(base)
            record["base_auto"] = True
        record["pipeline"] = pipe
        self._refresh_row(image_id)
        if self.current_id is None:
            self.tree.selection_set(str(image_id))
            self.on_select_image()

    def _refresh_row(self, image_id: int):
        record = self.images.get(image_id)
        if record is None:
            return
        pipe = record.get("pipeline")
        self.tree.item(str(image_id), text=record["name"], values=(
            record["status"],
            "✅" if pipe is not None and pipe.colorchecker_calibrated else "❌",
            "✅" if pipe is not None and not pipe.skip_preset else "❌",
        ))

    def on_select_image(self):
        selection = self.tree.selection()
        if not selection:
            return
        self.current_id = int(selection[0])
        record = self.images.get(self.current_id)
        if record is None:
            return
        if not record["loaded"]:
            return
        self._sync_all_widgets()
        self.refresh_analysis()
        self.request_preview(immediate=True)

    def remove_selected(self):
        for item in self.tree.selection():
            image_id = int(item)
            self.tree.delete(item)
            self.images.pop(image_id, None)
        if self.current_id not in self.images:
            self.current_id = None
            self.canvas.delete("all")
            self.canvas_photo = None

    def clear_images(self):
        if not self.images:
            return
        if not messagebox.askyesno("清空", "移除全部图像？"):
            return
        self.tree.delete(*self.tree.get_children())
        self.images.clear()
        self.current_id = None
        self.canvas.delete("all")
        self.canvas_photo = None

    # ==================================================================
    # 预览渲染（latest-wins + 防抖）
    # ==================================================================

    def _on_scale_change(self):
        self.render_scale = float(self.scale_var.get().rstrip("%")) / 100.0
        self.request_preview(immediate=True)

    def request_preview(self, immediate: bool = False):
        if self.current_id is None:
            return
        if immediate:
            if self._debounce_job is not None:
                self.root.after_cancel(self._debounce_job)
                self._debounce_job = None
            self._enqueue_render()
            return
        if self._debounce_job is not None:
            self.root.after_cancel(self._debounce_job)
        self._debounce_job = self.root.after(self._debounce_ms,
                                            self._enqueue_render)

    def _enqueue_render(self):
        self._debounce_job = None
        with self._render_lock:
            self._render_request = self.current_id
        self._render_event.set()

    def _post(self, callback):
        """工作线程 → 主线程：只投递到线程安全队列，**绝不直接碰 Tk**。

        为什么不能在工作线程里直接 ``root.after(...)``：Tcl 解释器不是
        线程安全的，窗口销毁或与主线程竞争时会直接**段错误**
        （实测间歇性崩溃）。正确做法是队列 + 主线程轮询。
        """
        if self._render_stop or not self._alive:
            return
        self._ui_queue.put(callback)

    def _poll_ui(self):
        """主线程定时排空 UI 队列（唯一的界面更新入口）。"""
        if not self._alive:
            return
        for _ in range(64):
            try:
                callback = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            try:
                callback()
            except Exception:                           # noqa: BLE001
                pass
        try:
            self._ui_job = self.root.after(40, self._poll_ui)
        except Exception:                               # noqa: BLE001
            self._alive = False

    def _start_render_thread(self):
        self._render_thread = threading.Thread(target=self._render_worker,
                                               daemon=True)
        self._render_thread.start()

    def _render_worker(self):
        while not self._render_stop:
            self._render_event.wait(timeout=0.2)
            if self._render_stop:
                break
            with self._render_lock:
                image_id = self._render_request
                self._render_request = None
            self._render_event.clear()
            if image_id is None:
                continue
            record = self.images.get(image_id)
            if record is None or not record["loaded"]:
                continue
            self._render_busy = True
            started = time.perf_counter()
            try:
                preview = self._render_preview(record)
            except Exception as exc:                    # noqa: BLE001
                preview = None
                self._post(lambda e=exc: self.status_label.config(
                    text=f"渲染失败: {e}", foreground=DARK_THEME["error"]))
            elapsed = (time.perf_counter() - started) * 1000.0
            self._render_busy = False
            if preview is None or self._render_stop:
                continue
            self._post(lambda p=preview, t=elapsed, i=image_id:
                       self._show_preview(p, t, i))

    def _render_preview(self, record):
        """渲染预览。

        **绝不返回 None**：只要图像有像素就一定要画出东西。旧实现在
        ``base_val_rgb is None`` 时直接 ``return None``，于是导入后画布一直
        是空的（就是用户看到的"黑屏"）。
        """
        pipe = record["pipeline"]
        if pipe is None or pipe.linear_img is None:
            return None
        step = 1
        height, width = pipe.linear_img.shape[:2]
        longest = max(height, width)
        if longest * self.render_scale < MAX_PREVIEW_EDGE:
            step = max(1, int(round(1.0 / max(self.render_scale, 1e-3))))
        render_pipe = self._downscaled_pipeline(pipe, step)

        if render_pipe.base_val_rgb is None:
            # 还没片基：用临时估计值渲染，只影响这一次渲染，**不改管线状态**
            provisional = render_pipe.auto_detect_base()
            if provisional is None or not np.any(provisional > 0):
                # 连片基都估不出来（例如全黑帧）：给中灰提示图，而不是黑屏
                shape = render_pipe.linear_img.shape[:2]
                return np.full(shape + (3,), 40, dtype=np.uint8)
            render_pipe.set_base_val(provisional)
        preview = render_pipe.process_for_preview(scale=1.0, count_stats=True)
        # 把（降采样副本上算出的）统计回传给真实管线：分析页与窗口提示需要它。
        # 统计量是比例，抽样不影响结论；界面会标明样本分辨率。
        pipe.clip_stats = render_pipe.clip_stats
        pipe.preview_shape = preview.shape[:2]
        return preview

    @staticmethod
    def _downscaled_pipeline(pipe, step: int):
        """克隆一条管线用于渲染；``step > 1`` 时在降采样副本上渲染。

        克隆是必要的：预览可能需要一个**临时片基**（还没采样时），
        而且降采样渲染能显著提速大图。克隆里的可变对象（tone/curves/view）
        只被读取，不会被修改。
        """
        from .pipeline import ScientificFilmPipeline
        clone = ScientificFilmPipeline()
        clone.__dict__.update({k: v for k, v in pipe.__dict__.items()
                               if k != "linear_img"})
        image = pipe.linear_img
        clone.linear_img = (np.ascontiguousarray(image[::step, ::step, :])
                            if step > 1 else image)
        clone.image_loaded = True
        return clone

    def _show_preview(self, preview: np.ndarray, elapsed_ms: float, image_id: int):
        if image_id != self.current_id:
            return                                    # 过期结果直接丢弃
        self._preview_source = preview
        from PIL import Image, ImageTk
        image = Image.fromarray(preview, mode="RGB")
        canvas_w = max(self.canvas.winfo_width(), 10)
        canvas_h = max(self.canvas.winfo_height(), 10)
        iw, ih = image.size
        scale = min(canvas_w / iw, canvas_h / ih) * 0.98
        if scale <= 0:
            scale = 1.0
        new_size = (max(1, int(iw * scale)), max(1, int(ih * scale)))
        resized = image.resize(new_size, Image.LANCZOS)
        self.display_scale = scale
        self.canvas_photo = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(canvas_w // 2, canvas_h // 2, anchor=tk.CENTER,
                                 image=self.canvas_photo)
        self.display_offset = ((canvas_w - new_size[0]) // 2,
                               (canvas_h - new_size[1]) // 2)
        self.info_label.config(text=f"{preview.shape[1]}×{preview.shape[0]}  "
                                    f"{elapsed_ms:.0f} ms")
        self._redraw_overlays()
        self.refresh_analysis()

    def _canvas_to_image(self, event):
        if self._preview_source is None or self.canvas_photo is None:
            return None
        height, width = self._preview_source.shape[:2]
        x = int((event.x - self.display_offset[0]) / max(self.display_scale, 1e-9))
        y = int((event.y - self.display_offset[1]) / max(self.display_scale, 1e-9))
        if 0 <= x < width and 0 <= y < height:
            return x, y
        return None

    def _redraw_overlays(self):
        for index, (x, y) in enumerate(self.sampling_points[-12:]):
            cx = self.display_offset[0] + x * self.display_scale
            cy = self.display_offset[1] + y * self.display_scale
            self.canvas.create_oval(cx - 4, cy - 4, cx + 4, cy + 4,
                                    outline=DARK_THEME["accent"], width=2)
        for index, (x, y) in enumerate(self.corner_points):
            cx = self.display_offset[0] + x * self.display_scale
            cy = self.display_offset[1] + y * self.display_scale
            self.canvas.create_oval(cx - 6, cy - 6, cx + 6, cy + 6,
                                    outline=DARK_THEME["warning"], width=2)
            self.canvas.create_text(cx + 12, cy - 10, text=str(index + 1),
                                    fill=DARK_THEME["warning"])
        if len(self.corner_points) == 4:
            points = []
            for x, y in self.corner_points:
                points.extend([self.display_offset[0] + x * self.display_scale,
                               self.display_offset[1] + y * self.display_scale])
            points.extend(points[:2])
            self.canvas.create_line(*points, fill=DARK_THEME["warning"], width=1)

    # ==================================================================
    # 画布交互
    # ==================================================================

    def toggle_sampling(self):
        self.sampling_mode = not self.sampling_mode
        self.corner_mode = False
        self.sample_var.set("点击采样模式：开启（点片基）" if self.sampling_mode
                            else "点击采样模式：关闭")

    def toggle_corners(self):
        self.corner_mode = not self.corner_mode
        self.sampling_mode = False
        self.corner_points = []
        self.sample_var.set("手动四角：依次点左上/右上/右下/左下"
                            if self.corner_mode else "点击采样模式：关闭")
        self._redraw_overlays()

    def on_canvas_click(self, event):
        coords = self._canvas_to_image(event)
        if coords is None:
            return
        if self.sampling_mode:
            self._sample_base(*coords)
        elif self.corner_mode:
            self.corner_points.append(coords)
            self._redraw_overlays()
            if len(self.corner_points) == 4:
                self.corner_mode = False
                self.sample_var.set("四角已选，点「求解并应用」")

    def _sample_base(self, x, y):
        record = self.images.get(self.current_id)
        if record is None:
            return
        linear = record["linear"]
        rgb = np.asarray(linear[y, x, :3], dtype=np.float64)
        pipe = record["pipeline"]
        if float(np.max(rgb)) <= 1e-6:
            self.status_label.config(
                text="该点没有任何信号（纯黑），不是有效的片基采样",
                foreground=DARK_THEME["error"])
            return
        # 用户开始手动采样 → 丢弃"自动检测"的那一个值：
        # 它是整幅统计量，和手动点混在一起取中位数会污染结果。
        if record.get("base_auto"):
            pipe.base_samples = []
            self.sampling_points = []
        record["base_auto"] = False
        pipe.base_samples.append(rgb.copy())
        if len(pipe.base_samples) == 1:
            pipe.set_base_val(rgb, (x, y))
        else:
            pipe.set_base_val_multisample(pipe.base_samples, (x, y))
        self.sampling_points.append((x, y))
        spread = float(np.max(pipe.base_val_rgb) - np.min(pipe.base_val_rgb))
        # 不再弹模态确认：彩色负片的片基本来就是**橙色**（橙色 mask），
        # 三通道差异大是物理正常的，消偏色交给「中性化」。这里只做信息提示。
        if spread > 0.4:
            self.status_label.config(
                text=f"片基三通道差异 {spread:.3f} 偏大，请确认采样点确实是"
                     "未曝光的片基边缘（而不是画面里的物体）",
                foreground=DARK_THEME["warn"])
        else:
            self.status_label.config(text=f"已采样片基（{len(pipe.base_samples)} 点）")
        self._sync_labels()
        self._redraw_overlays()
        self.request_preview(immediate=True)

    def on_canvas_move(self, event):
        coords = self._canvas_to_image(event)
        record = self.images.get(self.current_id)
        if coords is None or record is None or not record["loaded"]:
            self.cursor_label.config(text="")
            return
        x, y = coords
        rgb = np.asarray(record["linear"][y, x, :3], dtype=np.float64)
        self.cursor_label.config(
            text=f"({x}, {y})  输入线性 R={rgb[0]:.3f} G={rgb[1]:.3f} B={rgb[2]:.3f}")

    # ==================================================================
    # 各动作
    # ==================================================================

    def import_profile(self):
        pipe = self._pipeline()
        if pipe is None:
            messagebox.showwarning("提示", "请先导入图像")
            return
        path = filedialog.askopenfilename(
            title="选择相机 profile",
            filetypes=[("相机 profile", "*.dcp *.icc *.icm"), ("所有文件", "*.*")])
        if not path:
            return
        try:
            pipe.load_icc_profile(path)
        except Exception as exc:                        # noqa: BLE001
            messagebox.showerror("加载失败", str(exc))
            return
        self._sync_labels()
        self.request_preview(immediate=True)

    def wb_from_neutral(self):
        record = self.images.get(self.current_id)
        if record is None:
            return
        neutral, reason = io_read.read_as_shot_neutral(record["path"])
        if neutral is None:
            messagebox.showinfo("白平衡", f"无法获取 AsShotNeutral：{reason}\n"
                                          "可手动设置通道增益。")
            return
        record["pipeline"].wb_from_neutral(neutral)
        self._sync_labels()
        self.request_preview(immediate=True)

    def _set_wb(self, mode):
        pipe = self._pipeline()
        if pipe is None:
            return
        pipe.set_wb(mode)
        self._sync_labels()
        self.request_preview(immediate=True)

    def auto_base(self):
        pipe = self._pipeline()
        if pipe is None:
            return
        base = pipe.auto_detect_base()
        if base is None:
            messagebox.showwarning("自动检测", "未能检测到片基，请手动采样")
            return
        pipe.set_base_val(base)
        self._sync_labels()
        self.request_preview(immediate=True)

    def clear_base(self):
        pipe = self._pipeline()
        if pipe is None:
            return
        pipe.base_val_rgb = None
        pipe.base_samples = []
        self.sampling_points = []
        self._sync_labels()
        self.canvas.delete("all")
        self.canvas_photo = None

    def neutralize(self):
        pipe = self._pipeline()
        if pipe is None:
            return
        if pipe.base_val_rgb is None:
            messagebox.showwarning("密度域中性化", "请先采样片基")
            return
        result = pipe.auto_align_density_domain()
        if result is None:
            messagebox.showwarning("密度域中性化", "无法完成（缺少片基或图像）")
            return
        self._sync_all_widgets()
        self._redraw_curve()
        self.request_preview(immediate=True)
        if result["clamped"]:
            self.status_label.config(
                text="增益触及裁剪范围，达成密度未完全到目标",
                foreground=DARK_THEME["warning"])

    def reset_gains(self):
        pipe = self._pipeline()
        if pipe is None:
            return
        self._push_undo()
        pipe.reset_gains()
        self._sync_all_widgets()
        self.request_preview(immediate=True)

    def on_preset_change(self):
        pipe = self._pipeline()
        if pipe is None:
            return
        name = self.preset_var.get()
        if not pipe.set_preset(name):
            return
        self._push_undo()
        preset = FILM_PRESETS[name]
        self.preset_note.config(text=f"{preset.description}\n来源：{preset.source_note}")
        self._sync_all_widgets()
        self._redraw_curve()
        self.request_preview(immediate=True)
        self._refresh_row(self.current_id)

    def measure_density_range(self):
        pipe = self._pipeline()
        if pipe is None or pipe.base_val_rgb is None:
            messagebox.showwarning("测量密度范围", "请先采样片基")
            return
        report = pipe.measure_density()
        if report is None:
            return
        # 用 0.1% / 99.9% 分位而不是绝对极值：单个热像素/灰尘点就会把窗口
        # 拉爆（整幅对比被压扁）。极值仍然显示出来供判断。
        low = min(report.percentile_low)
        high = max(report.percentile_high)
        pm.apply_param(pipe, "hd_d_min", max(0.0, float(low)))
        pm.apply_param(pipe, "hd_d_max", max(0.3, float(high)))
        self._sync_all_widgets()
        self._redraw_curve()
        self.request_preview(immediate=True)
        self.status_label.config(
            text=f"密度窗口已按实测 0.1%–99.9% 分位填入：{low:.3f} … {high:.3f}"
                 f"（绝对极值 {min(report.per_channel_min):.3f}"
                 f" … {max(report.per_channel_max):.3f}，已排除离群点）",
            foreground=DARK_THEME["fg"])

    def reset_curve(self):
        pipe = self._pipeline()
        if pipe is None:
            return
        self._push_undo()
        for key, value in (("hd_a", 1.0), ("hd_x_mid", 0.5), ("hd_s_toe", 0.06),
                           ("hd_s_sh", 0.06)):
            pm.apply_param(pipe, key, value)
        self._sync_all_widgets()
        self._redraw_curve()
        self.request_preview(immediate=True)

    def reset_view(self):
        pipe = self._pipeline()
        if pipe is None:
            return
        for key, value in pm.defaults().items():
            if key.startswith("view_"):
                pm.apply_param(pipe, key, value)
        self._sync_all_widgets()
        self.request_preview(immediate=True)

    def reset_look(self):
        pipe = self._pipeline()
        if pipe is None:
            return
        self._push_undo()
        for key, value in pm.defaults().items():
            if pm.spec(key).look:
                pm.apply_param(pipe, key, value)
        self._sync_all_widgets()
        self.request_preview(immediate=True)

    # ---- 色卡 ----
    def import_calibration(self):
        pipe = self._pipeline()
        if pipe is None:
            return
        path = filedialog.askopenfilename(
            title="选择校准文件",
            filetypes=[("校准", "*.ccmx *.json"), ("所有文件", "*.*")])
        if not path:
            return
        try:
            pipe.load_calibration(path)
        except Exception as exc:                        # noqa: BLE001
            messagebox.showerror("加载失败", str(exc))
            return
        self._push_undo()
        self._sync_labels()
        self.refresh_analysis()
        self.request_preview(immediate=True)
        self._refresh_row(self.current_id)

    def detect_chart(self):
        record = self.images.get(self.current_id)
        if record is None:
            return
        detection, reason = cc.detect_chart(record["linear"], explain=True)
        if detection is None:
            messagebox.showinfo("自动检测失败",
                                f"{reason}\n\n请改用「手动四角」。")
            return
        self.corner_points = [tuple(p) for p in detection.corners]
        self._redraw_overlays()
        self._solve_with_points(detection.corners)

    def solve_from_corners(self):
        if len(self.corner_points) != 4:
            messagebox.showinfo("手动四角", "请先在预览上依次点四个角")
            return
        self._solve_with_points(np.array(self.corner_points, dtype=np.float64))

    def _solve_with_points(self, corners):
        record = self.images.get(self.current_id)
        if record is None:
            return
        try:
            sampled = cc.sample_patches(record["linear"], corners)
            result = record["pipeline"].calibrate_from_colorchecker(sampled)
        except Exception as exc:                        # noqa: BLE001
            messagebox.showerror("求解失败", str(exc))
            return
        self._push_undo()
        self._sync_labels()
        self.refresh_analysis()
        self.request_preview(immediate=True)
        self._refresh_row(self.current_id)
        messagebox.showinfo("色卡矫正完成", cc.format_error_report(result.stats))

    # ---- 分析与绘图 ----
    def refresh_analysis(self):
        pipe = self._pipeline()
        if pipe is None:
            return
        self.error_text.delete("1.0", tk.END)
        self.error_text.insert("1.0", pipe.get_error_report())
        stats = pipe.get_clip_stats()
        density = pipe.get_density_report()
        shape = getattr(pipe, "preview_shape", None)
        scale_note = (f"（统计样本 {shape[1]}×{shape[0]}，预览分辨率）"
                      if shape else "")
        perf = pipe.get_performance_stats()
        self.stats_text.delete("1.0", tk.END)
        self.stats_text.insert("1.0", (
            f"管线耗时: profile {perf['profile']:.1f}ms  密度 {perf['density']:.1f}ms  "
            f"曲线 {perf['curve']:.1f}ms  合计 {perf['total']:.1f}ms\n"
            f"编码斜率: {pipe.transfer.codes_per_unit:.1f} code/单位密度"
            f"（自动匹配: {'开' if pipe.auto_fit else '关'}）\n"
            f"曲线弦斜率: {pipe.transfer.effective_slope():.3f}\n"
            f"越窗: 片基以下 {stats['density_below_base']}  超出上限 "
            f"{stats['density_above_window']}  末端裁剪 "
            f"{stats['output_clipped']} ({stats['output_clipped_pct']}%)"
            f"{scale_note}\n"
            f"profile 越界通道样本: {stats['profile_out_of_gamut']}\n"
            f"密度中位: {density['per_channel_median']}\n"
            f"达成密度: {density['achieved']}  目标 {density['target_density']}\n"
            f"片基: {density['base']}  片基差异 {density['base_spread']:.3f}"))
        # 用 stats_label 显示"窗口匹配度"：这是最容易让人误以为"程序坏了"
        # 的地方 —— 数据超出密度窗口时高光会正常地压到白，但用户不知道原因。
        total = max(stats["total_samples"], 1)
        over = stats["density_above_window"] / total
        under = stats["density_below_base"] / total
        hints = []
        if over > 0.02:
            hints.append(f"⚠ {over * 100:.1f}% 像素密度超窗口上限"
                         "（点「测量密度范围」按实测自动匹配）")
        if under > 0.02:
            hints.append(f"{under * 100:.1f}% 像素比片基更亮（已收敛到黑点）")
        gamut = stats["profile_out_of_gamut"] / total
        if gamut > 0.02:
            hints.append(f"{gamut * 100:.1f}% 通道样本超出 sRGB 色域")
        self.stats_label.config(
            text="  |  ".join(hints) if hints else "窗口匹配良好",
            foreground=DARK_THEME["warn"] if hints else DARK_THEME["muted"])
        self._draw_histogram()
        self._redraw_curve()

    def _draw_histogram(self):
        self.hist_canvas.delete("all")
        if self._preview_source is None:
            return
        width = max(self.hist_canvas.winfo_width(), 100)
        height = max(self.hist_canvas.winfo_height(), 60)
        counts = display_mod.histogram(self._preview_source, bins=64)
        peak = max(1, int(counts.max()))
        colours = ("#ff5555", "#55ff77", "#5588ff")
        for channel in range(3):
            points = []
            for index in range(counts.shape[0]):
                x = index / (counts.shape[0] - 1) * width
                y = height - counts[index, channel] / peak * (height - 6)
                points.extend([x, y])
            if len(points) >= 4:
                self.hist_canvas.create_line(*points, fill=colours[channel], width=1)

    def _redraw_curve(self):
        canvas = getattr(self, "curve_canvas", None)
        if canvas is None:
            return
        pipe = self._pipeline()
        canvas.delete("all")
        width = max(canvas.winfo_width(), 100)
        height = max(canvas.winfo_height(), 60)
        canvas.create_rectangle(0, 0, width, height, outline=DARK_THEME["button_bg"])
        if pipe is None:
            return
        density, code = pipe.transfer.sample(128)
        span = max(float(code.max() - code.min()), 1e-6)
        points = []
        for index in range(len(density)):
            x = index / (len(density) - 1) * width
            y = height - (code[index, 0] - code.min()) / span * (height - 8) - 4
            points.extend([x, y])
        canvas.create_line(*points, fill=DARK_THEME["accent"], width=2)
        canvas.create_text(6, 10, anchor=tk.W,
                           text=f"{pipe.preset_name}｜{density[0]:.2f}…{density[-1]:.2f} D"
                                f"｜{code.min():.0f}…{code.max():.0f} code",
                           fill=DARK_THEME["muted"])

    # ==================================================================
    # 导出
    # ==================================================================

    def export_current(self):
        pipe = self._pipeline()
        if pipe is None:
            messagebox.showwarning("导出", "请先选择图像")
            return
        if pipe.base_val_rgb is None:
            messagebox.showwarning("导出", "请先采样片基")
            return
        fmt = pm.read_param(pipe, "export_format")
        if fmt == "export_format":
            fmt = "tiff16"
        stem = os.path.splitext(self.images[self.current_id]["name"])[0]
        path = filedialog.asksaveasfilename(
            defaultextension="." + io_write.extension_for(fmt),
            initialfile=f"{stem}_{pipe.output_colorspace}.{io_write.extension_for(fmt)}")
        if not path:
            return
        try:
            # 不在导出端再套 LUT：套不套由 pipe.output.mode 决定（只套一次）
            data = pipe.process_for_output()
            io_write.write_image(
                path, data, fmt, metadata=pipe.to_settings(),
                range_mode=pipe.output.effective_range,
                dpx_transfer=pipe.output.transfer_code(),
                dpx_colorimetric=pipe.output.colorimetric_code())
        except Exception as exc:                        # noqa: BLE001
            messagebox.showerror("导出失败", str(exc))
            return
        messagebox.showinfo("导出完成", f"已写出\n{path}")

    def export_batch(self):
        selection = self.tree.selection()
        ids = [int(item) for item in selection] if selection else list(self.images)
        paths = [self.images[i]["path"] for i in ids
                 if i in self.images and self.images[i]["loaded"]]
        if not paths:
            messagebox.showwarning("批量导出", "没有可导出的图像")
            return
        source = self._pipeline()
        if source is None:
            return
        out_dir = filedialog.askdirectory(title="选择导出目录")
        if not out_dir:
            return
        settings = source.to_settings()
        fmt = pm.read_param(source, "export_format")

        def progress(index, total, item):
            self._post(lambda: self.progress.configure(
                maximum=total, value=index))
            self._post(lambda i=item, n=index, t=total:
                       self.status_label.config(
                           text=f"导出中 {n}/{t}：{i.name} — {i.status}"))

        def worker():
            result = batch_mod.run_batch(paths, settings, out_dir=out_dir,
                                         export_format=fmt, progress=progress,
                                         auto_base=True, neutralize=True)
            self._post(lambda: self._on_batch_done(result))

        threading.Thread(target=worker, daemon=True).start()

    def _on_batch_done(self, result):
        self.progress.configure(value=0)
        summary = result.summary
        self.status_label.config(
            text=f"批量完成：{summary['ok']}/{summary['total']} 成功，"
                 f"{summary['failed']} 失败（{summary['elapsed_s']}s）",
            foreground=DARK_THEME["success"] if summary["failed"] == 0
            else DARK_THEME["warning"])
        report = result.failure_report()
        if report:
            messagebox.showwarning("批量导出", report[:1500])

    def export_lut(self, include_look: bool):
        from .lut import CubeLUT
        pipe = self._pipeline()
        if pipe is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".cube",
                                           initialfile="aurhythm_calibration.cube")
        if not path:
            return
        try:
            CubeLUT.from_pipeline(pipe, size=33,
                                  include_look=include_look).save(path)
        except Exception as exc:                        # noqa: BLE001
            messagebox.showerror("导出 LUT 失败", str(exc))
            return
        messagebox.showinfo("LUT", f"已写出 {path}\n"
                                   f"{'含调色层' if include_look else '仅校准变换'}")

    def load_lut(self):
        from .lut import CubeLUT
        pipe = self._pipeline()
        if pipe is None:
            return
        path = filedialog.askopenfilename(filetypes=[("Cube LUT", "*.cube"),
                                                     ("所有文件", "*.*")])
        if not path:
            return
        lut = CubeLUT()
        try:
            lut.load(path)
            # 统一入口：这一步同时把输出变换切到 "lut → 目标空间"
            pipe.set_output_lut(lut, enabled=True, path=path)
        except Exception as exc:                        # noqa: BLE001
            messagebox.showerror("LUT 加载失败", str(exc))
            return
        self.lut_label.config(text=f"✅ {os.path.basename(path)}"
                                   f"（{lut.size}³）",
                              foreground=DARK_THEME["success"])
        self._push_undo()
        self._sync_all_widgets()
        self.request_preview(immediate=True)

    def unload_lut(self):
        pipe = self._pipeline()
        if pipe is None:
            return
        pipe.set_output_lut(None, enabled=False)
        self.lut_label.config(text="未加载 LUT", foreground=DARK_THEME["muted"])
        self._sync_all_widgets()
        self.request_preview(immediate=True)

    # ---- 参数快照 / 撤销 ----
    def save_settings(self):
        pipe = self._pipeline()
        if pipe is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".json",
                                           initialfile="aurhythm_settings.json")
        if not path:
            return
        sm.save_settings(path, pipe.to_settings())
        messagebox.showinfo("参数", f"已保存\n{path}")

    def load_settings(self):
        pipe = self._pipeline()
        if pipe is None:
            return
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json"),
                                                     ("所有文件", "*.*")])
        if not path:
            return
        try:
            report = pipe.apply_settings(sm.load_settings(path))
        except Exception as exc:                        # noqa: BLE001
            messagebox.showerror("加载参数失败", str(exc))
            return
        self._sync_all_widgets()
        self._redraw_curve()
        self.request_preview(immediate=True)
        if report:
            messagebox.showinfo("参数", "\n".join(report))

    def _push_undo(self):
        pipe = self._pipeline()
        if pipe is None:
            return
        self._undo.append(pm.snapshot(pipe))
        if len(self._undo) > self._undo_limit:
            self._undo.pop(0)
        self._redo.clear()

    def undo(self):
        pipe = self._pipeline()
        if pipe is None or not self._undo:
            return
        self._redo.append(pm.snapshot(pipe))
        pm.restore(pipe, self._undo.pop())
        self._sync_all_widgets()
        self._redraw_curve()
        self.request_preview(immediate=True)

    def redo(self):
        pipe = self._pipeline()
        if pipe is None or not self._redo:
            return
        self._undo.append(pm.snapshot(pipe))
        pm.restore(pipe, self._redo.pop())
        self._sync_all_widgets()
        self._redraw_curve()
        self.request_preview(immediate=True)

    # ---- 批量（参数复制） ----
    def batch_copy_settings(self):
        source = self._pipeline()
        if source is None:
            messagebox.showwarning("批量", "请先选择一张参考图像")
            return
        settings = source.to_settings()
        count = 0
        for image_id, record in self.images.items():
            if image_id == self.current_id or not record["loaded"]:
                continue
            sm.apply_settings_to_pipeline(record["pipeline"], settings,
                                          copy_pixels=True)
            count += 1
        if not count:
            messagebox.showinfo("批量", "没有其他已加载的图像")
            return
        messagebox.showinfo("批量", f"已把参考参数复制到 {count} 张图像")

    def batch_process(self):
        source = self._pipeline()
        if source is None:
            messagebox.showwarning("批量", "请先选择一张参考图像")
            return
        settings = source.to_settings()
        targets = [record["path"] for image_id, record in self.images.items()
                   if image_id != self.current_id and record["loaded"]]
        if not targets:
            messagebox.showinfo("批量", "没有其他已加载的图像")
            return

        def progress(index, total, item):
            self._post(lambda: self.progress.configure(
                maximum=total, value=index))
            self._post(lambda i=item, n=index, t=total:
                       self.status_label.config(
                           text=f"处理中 {n}/{t}：{i.name} — {i.status}"))

        def worker():
            result = batch_mod.run_batch(targets, settings, progress=progress,
                                         auto_base=True, neutralize=True)
            self._post(lambda: self._on_batch_done(result))

        threading.Thread(target=worker, daemon=True).start()

    # ==================================================================
    # 生命周期
    # ==================================================================

    def on_closing(self):
        """先停线程、再销毁窗口 —— 顺序反了会在工作线程里打到死 Tcl 上。"""
        self._render_stop = True
        self._render_event.set()
        thread = self._render_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.5)
        self._alive = False
        if self._ui_job is not None:
            try:
                self.root.after_cancel(self._ui_job)
            except Exception:                           # noqa: BLE001
                pass
            self._ui_job = None
        try:
            self.root.destroy()
        except Exception:                               # noqa: BLE001
            pass


def main(argv=None):
    """GUI 入口（``Aurhythm.py`` 与 ``python -m aurhythm`` 都走这里）。"""
    if argv:
        from .cli import main as cli_main
        return cli_main(argv)
    app = AurhythmApp()
    return 0


__all__ = ["AurhythmApp", "main", "DARK_THEME", "apply_dark_theme"]
