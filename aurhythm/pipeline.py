"""核心管线：输入线性 RGB → 色彩校准 → 密度 → 非线性校正 → 编码。

阶段链（每个阶段的域与裁剪策略都是硬约定，测试逐条守护）
------------------------------------------------------
====  ==========================  ==================  ==========================
#     阶段                        输入 → 输出         裁剪
====  ==========================  ==================  ==========================
1     白平衡（对角增益）            相机线性 → 相机线性  无
2     相机 profile → 线性 sRGB     相机线性 → 线性 sRGB 无（可负、可 >1）
3     色卡矫正（可选）              线性 sRGB → 线性 sRGB 无
4     通道增益（印片光）            线性 sRGB → 线性 sRGB 无
5     密度 D = −log10(v / base)     线性 sRGB → 密度      仅下界保护 log
6     解串扰矩阵（可选）            密度 RGB → 密度 CMY   max(·, 0)（可关）
7     归一化 + 非线性校正           密度 → 归一化密度    趾/肩平滑压缩
8     编码（Cineon / LogC3）        → code               末端统一裁剪
====  ==========================  ==================  ==========================

**阶段 1–6 一律不裁剪**：这是旧实现 ``apply_icc`` / ``_linear_to_density``
里 ``np.clip(..., 0, 1)`` 造成高光永久损失的根因修复（R8）。

片基（film base）语义
--------------------
用户/自动采样得到的是**输入线性空间**的片基 RGB（UI 能直接访问的
``linear_img``）；管线把它推过 阶段 1–3（**不含**通道增益，增益是印片光，
本就应该相对片基产生偏移），得到与密度计算同一空间的除数::

    base_aligned = profile_chain(色卡(base_raw))

每通道分别保留（旧实现把它覆盖成标量灰，是 R5 的一半病因）。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np

from . import display
from . import output as output_mod
from . import profiles as profiles_mod
from .constants import (
    CINEON, DEFAULT_GAIN_CLAMP, DEFAULT_PRESET, DEFAULT_TARGET_DENSITY,
    FILM_PRESETS, HDParams, SCHEMA_VERSION,
)
from .output import OutputTransform
from .tonemap import (REFERENCE_SPAN, CurveSet, ToneControls, TransferFunction,
                      sanitize)

#: 密度计算里 log 的下界保护（透射率不会取到 0）
TRANSMISSION_FLOOR = 1e-6
#: ``auto_align_density_domain(clamp=...)`` 的哨兵：表示「用 self.gain_clamp」；
#: 显式传 ``None`` 表示**不裁剪**（两者语义必须可区分）
USE_DEFAULT_CLAMP = object()


@dataclass
class StageStats:
    """每个阶段耗时（秒）。"""

    wb: float = 0.0
    profile: float = 0.0
    colorchecker: float = 0.0
    gains: float = 0.0
    density: float = 0.0
    crosstalk: float = 0.0
    curve: float = 0.0
    encode: float = 0.0
    total: float = 0.0

    def to_dict(self, scale: float = 1000.0) -> dict:
        return {k: round(v * scale, 3) for k, v in self.__dict__.items()}


@dataclass
class ClipStats:
    """裁剪/越界统计（透明化用：绝不静默丢信息）。"""

    profile_out_of_gamut: int = 0      # 阶段 2 后 <0 或 >1 的通道样本
    total_samples: int = 0
    density_below_base: int = 0        # 比片基亮（D < 0）
    density_above_window: int = 0      # 超出 d_max
    output_clipped: int = 0            # 末端裁剪到 [0, 1023]

    def to_dict(self) -> dict:
        total = max(self.total_samples, 1)
        return {
            "profile_out_of_gamut": self.profile_out_of_gamut,
            "profile_out_of_gamut_pct": round(100.0 * self.profile_out_of_gamut / total, 4),
            "density_below_base": self.density_below_base,
            "density_above_window": self.density_above_window,
            "output_clipped": self.output_clipped,
            "output_clipped_pct": round(100.0 * self.output_clipped / total, 4),
            "total_samples": self.total_samples,
        }


@dataclass
class DensityReport:
    """密度域读数（分析页与自动曝光用）。"""

    per_channel_median: tuple = (0.0, 0.0, 0.0)
    per_channel_min: tuple = (0.0, 0.0, 0.0)
    per_channel_max: tuple = (0.0, 0.0, 0.0)
    #: 0.1% / 99.9% 分位（用于自动填密度窗口，抗离群点）
    percentile_low: tuple = (0.0, 0.0, 0.0)
    percentile_high: tuple = (0.0, 0.0, 0.0)
    base: tuple = (0.0, 0.0, 0.0)
    base_spread: float = 0.0
    target: float = DEFAULT_TARGET_DENSITY
    achieved: tuple = (0.0, 0.0, 0.0)

    def to_dict(self) -> dict:
        return {
            "per_channel_median": [round(v, 4) for v in self.per_channel_median],
            "per_channel_min": [round(v, 4) for v in self.per_channel_min],
            "per_channel_max": [round(v, 4) for v in self.per_channel_max],
            "percentile_low": [round(v, 4) for v in self.percentile_low],
            "percentile_high": [round(v, 4) for v in self.percentile_high],
            "base": [round(v, 4) for v in self.base],
            "base_spread": round(self.base_spread, 4),
            "target_density": self.target,
            "achieved": [round(v, 4) for v in self.achieved],
        }


class ScientificFilmPipeline:
    """单张图像的完整校准管线（不含 GUI 依赖）。"""

    def __init__(self):
        # ---------- 输入 ----------
        self.linear_img: np.ndarray | None = None
        self.image_loaded = False

        # ---------- 白平衡 ----------
        self.wb_mode = "none"                     # 'none' | 'manual'
        self.wb_gains = np.ones(3, dtype=np.float64)

        # ---------- 相机 profile ----------
        self.profile = None                       # CameraProfile | None
        self.icc_weight = 1.0                     # 0=钨丝灯, 1=日光
        self.icc_source = None                    # **完整路径**（R10 修复）
        self.icc_source_name = None

        # ---------- 色卡矫正 ----------
        self.color_correction_matrix: np.ndarray | None = None
        self.colorchecker_calibrated = False
        self.calibration_source = None
        self.colorchecker_data = None
        self.error_analysis = None

        # ---------- 片基 ----------
        self.base_val_rgb: np.ndarray | None = None      # 输入线性空间
        self.sample_coords = None
        self.base_samples: list = []
        self.base_report: dict = {}

        # ---------- 通道增益（印片光） ----------
        self.channel_gains = np.ones(3, dtype=np.float64)
        self.gain_clamp = tuple(DEFAULT_GAIN_CLAMP)
        self.target_density = DEFAULT_TARGET_DENSITY
        self.density_report = DensityReport()

        # ---------- 非线性校正 ----------
        self.preset_name = DEFAULT_PRESET
        self.hd = FILM_PRESETS[DEFAULT_PRESET].hd
        self.skip_preset = FILM_PRESETS[DEFAULT_PRESET].skip
        self.matrix_inv = np.eye(3)
        self.crosstalk_enabled = False
        self.tone = ToneControls()
        self.curves = CurveSet()
        self.auto_fit = True
        self.codes_per_density = 500.0
        self.clamp_cmy = True

        # ---------- 显示 ----------
        self.view = display.ViewSettings()

        # ---------- 输出 ----------
        #: 末端定义：对数直通 or 经色彩还原 LUT 到显示域
        self.output = OutputTransform()
        self.lut = None
        self.lut_path = None
        self.lut_enabled = False

        # ---------- 统计 ----------
        self.stats = StageStats()
        self.clip_stats = ClipStats()

        self.set_preset(self.preset_name)

    # ==================================================================
    # 输入 / 白平衡
    # ==================================================================

    def load_linear_image(self, img_array) -> bool:
        if img_array is None:
            return False
        arr = np.asarray(img_array, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.ndim != 3 or arr.shape[-1] < 3:
            raise ValueError(f"线性图像形状必须是 (H,W,3)，收到 {arr.shape}")
        self.linear_img = np.ascontiguousarray(arr[..., :3], dtype=np.float32)
        self.image_loaded = True
        return True

    def set_wb(self, mode: str = "manual", gains=None):
        """白平衡：'none' 或 'manual'（对角增益）。"""
        if mode not in ("none", "manual"):
            raise ValueError("wb mode 必须是 'none' 或 'manual'")
        self.wb_mode = mode
        if gains is not None:
            g = np.asarray(gains, dtype=np.float64).ravel()
            if g.size != 3:
                raise ValueError("白平衡增益必须是 3 个数")
            self.wb_gains = np.maximum(g, 1e-6)
        return self.wb_gains

    def wb_from_neutral(self, as_shot_neutral):
        """由 DNG ``AsShotNeutral`` 得到白平衡增益（归一化到绿通道）。"""
        neutral = np.asarray(as_shot_neutral, dtype=np.float64).ravel()
        if neutral.size != 3 or np.any(neutral <= 0):
            raise ValueError("AsShotNeutral 必须是 3 个正数")
        gains = neutral[1] / neutral
        return self.set_wb("manual", gains)

    # ==================================================================
    # 相机 profile
    # ==================================================================

    def load_icc_profile(self, filepath: str) -> bool:
        """加载 ICC/DCP 相机 profile（真实解析；失败抛 ProfileError）。"""
        profile = profiles_mod.load_profile(filepath)
        self.profile = profile
        self.icc_source = profile.source              # 完整路径（R10）
        self.icc_source_name = profile.source_name
        return True

    def unload_icc(self):
        self.profile = None
        self.icc_source = None
        self.icc_source_name = None

    def set_icc_weight(self, weight: float):
        """0 = 钨丝灯（矩阵 2），1 = 日光（矩阵 1）。"""
        self.icc_weight = float(np.clip(weight, 0.0, 1.0))

    @property
    def icc_loaded(self) -> bool:
        return self.profile is not None

    # ==================================================================
    # 色卡矫正
    # ==================================================================

    def load_calibration(self, filepath: str) -> bool:
        """加载 ``.ccmx`` / ``.json`` 里的 3x3 矫正矩阵。"""
        from .colorchecker import load_calibration_file
        result = load_calibration_file(filepath)
        self.color_correction_matrix = result["matrix"]
        self.colorchecker_calibrated = True
        self.calibration_source = result["type"]
        self.colorchecker_data = result
        return True

    def set_color_matrix(self, matrix):
        m = np.asarray(matrix, dtype=np.float64)
        if m.shape != (3, 3):
            raise ValueError("色卡矫正矩阵必须是 3x3")
        self.color_correction_matrix = m
        self.colorchecker_calibrated = True
        self.calibration_source = self.calibration_source or "manual"
        return m

    def calibrate_from_colorchecker(self, detected_colors, target_colors=None):
        """由测得的 24 色块求解 3x3 矫正矩阵 + ΔE2000 报告。"""
        from .colorchecker import solve_correction
        result = solve_correction(detected_colors, target_colors)
        self.color_correction_matrix = result.matrix
        self.colorchecker_calibrated = True
        self.calibration_source = "auto"
        self.colorchecker_data = result
        self.error_analysis = result.stats
        return result

    def get_error_report(self) -> str:
        from .colorchecker import format_error_report
        return format_error_report(self.error_analysis)

    # ==================================================================
    # 片基
    # ==================================================================

    def set_base_val(self, rgb_values, coords=None):
        """设置**单点**片基采样（输入线性空间）。

        语义为「把片基设为这个值」：会**重置**采样列表。
        多点统计请用 :meth:`set_base_val_multisample`（那才负责累积）。
        """
        rgb = np.asarray(rgb_values, dtype=np.float64).ravel()
        if rgb.size != 3:
            raise ValueError("片基采样必须是 3 个通道值")
        if float(np.max(rgb)) <= TRANSMISSION_FLOOR:
            raise ValueError(
                "片基采样全为 0：该点没有任何信号（点到纯黑区域了？）\n"
                "请点击未曝光的片基边缘。")
        self.base_val_rgb = rgb
        self.sample_coords = coords
        self.base_samples = [rgb.copy()]
        spread = float(np.max(rgb) - np.min(rgb))
        self.base_report = {
            "rgb": rgb.tolist(),
            "range": spread,
            "mean": float(np.mean(rgb)),
            "status": "balanced" if spread < 0.1 else "unbalanced",
        }
        return self.base_report

    def set_base_val_multisample(self, samples, coords=None):
        """多点采样：取每通道中位数（抗噪，同时保留每通道信息）。"""
        arr = np.asarray(list(samples), dtype=np.float64).reshape(-1, 3)
        if arr.size == 0:
            raise ValueError("没有可用的片基采样")
        median = np.median(arr, axis=0)
        self.base_samples = [row.copy() for row in arr]
        self.base_val_rgb = median
        self.sample_coords = coords
        self.base_report = {
            "rgb": median.tolist(),
            "range": float(np.max(median) - np.min(median)),
            "mean": float(np.mean(median)),
            "samples": int(arr.shape[0]),
            "spread": float(np.max(np.std(arr, axis=0))),
            "status": "balanced" if float(np.max(median) - np.min(median)) < 0.1
                      else "unbalanced",
        }
        return self.base_report

    def auto_detect_base(self, img=None, method: str = "robust"):
        """自动检测片基。

        ``method='robust'``（默认）：亮度掩膜用 ``>=`` 分位（修 R6：旧实现
        用 ``>``，片基过曝/明亮区域超过 5% 时会得到空掩膜而返回 None），
        再对掩膜内像素取**每通道中位数**（旧实现取单像素，噪声敏感）。
        ``method='percentile'`` 保留旧行为用于对比。
        """
        img = self.linear_img if img is None else np.asarray(img)
        if img is None:
            return None
        arr = np.asarray(img, dtype=np.float64)
        luminance = arr.mean(axis=2)

        if float(np.max(arr)) <= TRANSMISSION_FLOOR:
            # 全黑/无有效信号：0 片基会让密度算成 +6 并渲染成**纯白**，
            # 明确返回 None 让上层显示"无法检测片基"。
            return None

        if method == "percentile":
            # 旧实现的原始逻辑（严格 >），保留用于对比与回归记录
            threshold = np.percentile(luminance, 95)
            mask = luminance > threshold
            if not np.any(mask):
                return None
            pixels = arr[mask]
            distances = np.std(pixels, axis=1)
            return pixels[int(np.argmin(distances))]

        threshold = np.percentile(luminance, 95)
        mask = luminance >= threshold
        min_pixels = max(16, int(1e-4 * mask.size))
        if int(np.count_nonzero(mask)) < min_pixels:
            # 掩膜太小 → 退化为每通道高分位（依然给出可用结果）
            base = np.percentile(arr.reshape(-1, 3), 99.5, axis=0)
            method_used = "percentile-99.5"
        else:
            base = np.median(arr[mask], axis=0)
            method_used = "mask-median"
        self.base_report = {
            "rgb": base.tolist(),
            "method": method_used,
            "mask_pixels": int(np.count_nonzero(mask)),
            "range": float(np.max(base) - np.min(base)),
            "mean": float(np.mean(base)),
        }
        return base

    # ==================================================================
    # 密度域中性化（每通道）
    # ==================================================================

    def auto_align_density_domain(self, target_density=None,
                                  clamp=USE_DEFAULT_CLAMP,
                                  reference: str = "median",
                                  base_floor: float = 0.02):
        """密度域中性化：让参考点的**逐通道**密度对齐到目标值。

        与旧实现的区别（R5）：

        * 参考点是图像自身的密度中位数（默认），不是「最亮区域」；
        * 逐通道增益**不再被覆盖成标量灰**，且除数使用
          ``base_aligned = 逐通道片基``；
        * 增益裁剪范围可配置（默认 ±3 档），并把**达成值**回报出来。

        返回 ``dict``：``gains`` / ``reference`` / ``achieved`` / ``target``。
        """
        if self.base_val_rgb is None or self.linear_img is None:
            return None
        target = float(self.target_density if target_density is None
                       else target_density)
        limits = self.gain_clamp if clamp is USE_DEFAULT_CLAMP else clamp

        linear = self._stage_a_b_c()                       # 不含通道增益
        base = self._base_aligned()
        density = self._density(linear, base)

        flat = density.reshape(-1, 3)
        # 排除片基区域：真实扫描件常有大片未曝光片基/齿孔（密度≈0）。
        # 若把整幅中位数直接当"中间调"，片基占多数时参考点会落在片基上，
        # 结果是拿片基去对齐目标密度（整幅被推亮 0.7 档），完全错。
        scene_mask = flat.max(axis=1) > base_floor
        min_scene = max(64, int(0.005 * len(flat)))
        used_fallback = int(np.count_nonzero(scene_mask)) < min_scene
        sample = flat if used_fallback else flat[scene_mask]

        if reference == "median":
            ref = np.median(sample, axis=0)
        elif reference == "mean":
            ref = sample.mean(axis=0)
        elif reference == "trimmed":
            lo = int(0.1 * len(sample))
            hi = max(lo + 1, int(0.9 * len(sample)))
            ref = np.mean(np.sort(sample, axis=0)[lo:hi], axis=0)
        else:
            raise ValueError("reference 必须是 'median' / 'mean' / 'trimmed'")

        gains = np.power(10.0, ref - target)               # D' = D − log10(g)
        if limits is not None:
            lo, hi = float(limits[0]), float(limits[1])
            gains = np.clip(gains, lo, hi)
        self.channel_gains = gains
        self.target_density = target

        achieved = ref - np.log10(gains)
        self.density_report = DensityReport(
            per_channel_median=tuple(ref.tolist()),
            per_channel_min=tuple(flat.min(axis=0).tolist()),
            per_channel_max=tuple(flat.max(axis=0).tolist()),
            percentile_low=tuple(np.percentile(flat, 0.1, axis=0).tolist()),
            percentile_high=tuple(np.percentile(flat, 99.9, axis=0).tolist()),
            base=tuple(base.tolist()),
            base_spread=float(np.max(base) - np.min(base)),
            target=target,
            achieved=tuple(achieved.tolist()),
        )
        return {
            "gains": tuple(gains.tolist()),
            "reference": tuple(ref.tolist()),
            "achieved": tuple(achieved.tolist()),
            "target": target,
            "reference_pixels": int(len(sample)),
            "excluded_base_pixels": (0 if used_fallback
                                     else int(len(flat) - len(sample))),
            "used_fallback": bool(used_fallback),
            "clamped": bool(limits is not None and (
                np.any(np.isclose(gains, float(limits[0])))
                or np.any(np.isclose(gains, float(limits[1]))))),
        }

    neutralize = auto_align_density_domain

    def set_channel_gains(self, gains):
        g = np.asarray(gains, dtype=np.float64).ravel()
        if g.size != 3:
            raise ValueError("通道增益必须是 3 个数")
        self.channel_gains = g
        return g

    def reset_gains(self):
        self.channel_gains = np.ones(3, dtype=np.float64)

    def solve_per_channel_from_neutral(self, reference=None):
        """【未完成 · 实测更差 · 尚未接入界面】逐通道窗口求解。

        **实测记录（诚实标注）**：在合成 C-41 测试图上，本方法让结果**更差**：
        全图 ΔE2000 由 9.05 升到 11.85（全图分位版）与 13.59（中性参照版），
        中性灰块的 B/R 由 0.903 变成 0.827 / 0.855（偏得更厉害）。

        原因分析（已确认的部分）：管线的**解串扰输出是正确的** —— 场景为中性
        灰时，它解出的是胶片**真实的**不等量逐层染料密度（2.120/2.050/1.859），
        那就是彩色负片固有的"层间反差偏色"。但**只调逐通道的密度窗口**
        （= 增益+偏移）并不能把它修回去：密度到 code 之间还有一层非线性
        S 曲线，而三层的差异同时体现在**斜率**上（生成侧 contrast =
        1.00/1.06/0.94）。要正确反演，需要同时解逐通道的 ``a``（曲线斜率），
        并对**已知中性阶梯**做最小二乘拟合，而不是只对齐两个分位。

        **结论：本函数在修正前不要接入界面、不要用于导出。**
        正确做法（下一步）：在合成图上用已知真值做最小二乘，解出
        ``(a_c, d_min_c, d_max_c)``，再用真实扫描件交叉验证。

        为什么必须用"已知中性"而不是全图分位：实测证明按全图分位对齐三通道
        （等于假设灰世界）会让偏色**更严重**——彩色画面的三通道密度分布本来
        就不同，强行拉平等于改色。正确做法是只用**已知中性的内容**做参照。

        ``reference`` 为 ``(N,3)`` 的中性参照密度采样（如灰阶楔/用户点的中性区）；
        为 ``None`` 时用整幅的中位密度（退化为现有「中性化」的偏移部分）。

        解的形式是密度域上的逐通道仿射 ``d' = (d - d_min_c)/range_c``：
        对参照的每个通道取 (低分位, 高分位)，令其映射到三通道共同的
        ``[u_lo, u_hi]`` —— 于是中性在三通道得到相同的归一化响应。
        """
        if not self.image_loaded:
            return None
        density = self._density(self._stage_a_b_c(), self._base_aligned())
        flat = density.reshape(-1, 3)
        if reference is None:
            sample = flat[flat.max(axis=1) > 0.02]
            if len(sample) < 64:
                sample = flat
        else:
            sample = np.asarray(reference, dtype=np.float64).reshape(-1, 3)
            if len(sample) < 8:
                return None
        p_lo = np.percentile(sample, 5.0, axis=0)
        p_hi = np.percentile(sample, 95.0, axis=0)

        span = max(float(self.hd.d_max) - float(self.hd.d_min), 1e-6)
        u_lo = (float(np.mean(p_lo)) - float(self.hd.d_min)) / span
        u_hi = (float(np.mean(p_hi)) - float(self.hd.d_min)) / span
        if u_hi - u_lo < 1e-6:
            return None
        ranges = (p_hi - p_lo) / (u_hi - u_lo)
        d_mins = p_lo - u_lo * ranges

        from dataclasses import replace
        self.hd = replace(
            self.hd,
            d_min_rgb=tuple(float(v) for v in d_mins),
            d_max_rgb=tuple(float(v) for v in d_mins + ranges))
        return {
            "reference_pixels": int(len(sample)),
            "d_min_rgb": tuple(round(float(v), 4) for v in d_mins),
            "d_max_rgb": tuple(round(float(v), 4)
                               for v in (d_mins + ranges)),
        }

    def clear_per_channel_window(self):
        """回到共享的单一窗口。"""
        from dataclasses import replace
        self.hd = replace(self.hd, d_min_rgb=None, d_max_rgb=None, a_rgb=None)

    def measure_density(self):
        """测量当前图像的逐通道密度读数（不改增益）。"""
        if self.base_val_rgb is None or self.linear_img is None:
            return None
        linear = self._stage_a_b_c()
        base = self._base_aligned()
        density = self._density(linear, base)
        flat = density.reshape(-1, 3)
        return DensityReport(
            per_channel_median=tuple(np.median(flat, axis=0).tolist()),
            per_channel_min=tuple(flat.min(axis=0).tolist()),
            per_channel_max=tuple(flat.max(axis=0).tolist()),
            percentile_low=tuple(np.percentile(flat, 0.1, axis=0).tolist()),
            percentile_high=tuple(np.percentile(flat, 99.9, axis=0).tolist()),
            base=tuple(base.tolist()),
            base_spread=float(np.max(base) - np.min(base)),
            target=self.target_density,
            achieved=tuple((np.median(flat, axis=0)
                            - np.log10(self.channel_gains)).tolist()),
        )

    # ==================================================================
    # 胶片预设 / 曲线 / 色调
    # ==================================================================

    def set_preset(self, preset_name: str) -> bool:
        if preset_name not in FILM_PRESETS:
            return False
        preset = FILM_PRESETS[preset_name]
        self.preset_name = preset_name
        self.hd = preset.hd
        self.matrix_inv = np.asarray(preset.matrix_inv, dtype=np.float64).copy()
        self.skip_preset = preset.skip
        self.crosstalk_enabled = not preset.skip
        return True

    def set_hd_params(self, **kwargs):
        """按字典更新曲线参数（未知键直接报错，避免静默无效控件）。"""
        data = self.hd.to_dict()
        for key, value in kwargs.items():
            if key not in data:
                raise KeyError(f"未知曲线参数: {key}")
            data[key] = float(value)
        self.hd = sanitize(HDParams(**data))
        return self.hd

    def set_tone_controls(self, tone: ToneControls):
        self.tone = tone
        return self.tone

    def set_curves(self, curves: CurveSet):
        self.curves = curves
        return self.curves

    @property
    def output_colorspace(self) -> str:
        """兼容旧名：对数直通时的输出空间。"""
        return self.output.log_space

    @output_colorspace.setter
    def output_colorspace(self, colorspace: str):
        self.set_output_colorspace(colorspace)

    def set_output_colorspace(self, colorspace: str):
        """设置**对数直通**时的输出空间（不影响 LUT 模式）。"""
        if colorspace not in ("cineon", "logc3"):
            raise ValueError("输出色彩空间必须是 'cineon' 或 'logc3'")
        from dataclasses import replace
        self.output = replace(self.output, log_space=colorspace).sanitize()

    def set_output_transform(self, transform: OutputTransform):
        self.output = transform.sanitize()
        return self.output

    def set_output_lut(self, lut, enabled: bool = True, path: str | None = None,
                       input_space: str | None = None,
                       target_space: str | None = None,
                       range_mode: str | None = None):
        """启用「色彩还原 LUT」：输出变成 LUT 的**目标色彩空间**。"""
        from dataclasses import replace

        self.lut = lut
        self.lut_enabled = bool(enabled and lut is not None)
        self.lut_path = path
        changes = {"lut_path": path}
        if self.lut_enabled:
            changes["mode"] = output_mod.MODE_LUT
            if input_space:
                changes["lut_input_space"] = input_space
            if target_space:
                changes["lut_target_space"] = target_space
            if range_mode:
                changes["range"] = range_mode
        else:
            changes["mode"] = output_mod.MODE_LOG
        self.output = replace(self.output, **changes).sanitize()
        return self.output

    def set_lut(self, lut, enabled: bool = True, path: str | None = None,
                **kwargs):
        """兼容旧名 —— 内部统一走 :meth:`set_output_lut`，只有一条代码路径。"""
        return self.set_output_lut(lut, enabled=enabled, path=path, **kwargs)

    @property
    def transfer(self) -> TransferFunction:
        """当前的非线性传递函数（预览/导出/LUT 导出共用）。"""
        return TransferFunction(hd=self.hd, tone=self.tone, curves=self.curves,
                                auto_fit=self.auto_fit,
                                codes_per_density=self.codes_per_density)

    # ==================================================================
    # 阶段计算
    # ==================================================================

    def _stage_a_b_c(self) -> np.ndarray:
        """阶段 1–3：白平衡 → 相机 profile → 色卡矫正（**不裁剪**）。"""
        t0 = time.perf_counter()
        arr = np.asarray(self.linear_img, dtype=np.float64)
        if self.wb_mode == "manual":
            arr = arr * self.wb_gains.reshape(1, 1, 3)
        self.stats.wb = time.perf_counter() - t0

        t0 = time.perf_counter()
        arr = profiles_mod.camera_to_linear_srgb(arr, self.profile,
                                                 self.icc_weight)
        self.stats.profile = time.perf_counter() - t0

        t0 = time.perf_counter()
        if self.colorchecker_calibrated and self.color_correction_matrix is not None:
            flat = arr.reshape(-1, 3)
            arr = (flat @ self.color_correction_matrix.T).reshape(arr.shape)
        self.stats.colorchecker = time.perf_counter() - t0
        return arr

    def _base_aligned(self) -> np.ndarray:
        """把片基采样推过阶段 1–3，得到与密度同空间的除数（逐通道）。"""
        if self.base_val_rgb is None:
            raise ValueError("尚未采样片基")
        base = np.asarray(self.base_val_rgb, dtype=np.float64).reshape(1, 1, 3)
        if self.wb_mode == "manual":
            base = base * self.wb_gains.reshape(1, 1, 3)
        base = profiles_mod.camera_to_linear_srgb(base, self.profile,
                                                  self.icc_weight)
        if self.colorchecker_calibrated and self.color_correction_matrix is not None:
            flat = base.reshape(-1, 3)
            base = (flat @ self.color_correction_matrix.T).reshape(base.shape)
        return np.maximum(base.reshape(3), TRANSMISSION_FLOOR)

    @staticmethod
    def _density(linear: np.ndarray, base: np.ndarray) -> np.ndarray:
        """D = −log10(v / base)，只在下界保护 log；**允许 D < 0**。"""
        ratio = np.asarray(linear, dtype=np.float64) / base.reshape(1, 1, 3)
        ratio = np.maximum(ratio, TRANSMISSION_FLOOR)
        return -np.log10(ratio)

    def _crosstalk(self, density: np.ndarray) -> np.ndarray:
        if not self.crosstalk_enabled or self.matrix_inv is None:
            return density
        flat = density.reshape(-1, 3)
        out = flat @ np.asarray(self.matrix_inv, dtype=np.float64).T
        if self.clamp_cmy:
            out = np.maximum(out, 0.0)
        return out.reshape(density.shape)

    # ==================================================================
    # 处理入口
    # ==================================================================

    def process_to_code(self, count_stats: bool = True):
        """跑完整条链，返回**未裁剪**的 Cineon code (H,W,3)。"""
        if not self.image_loaded or self.base_val_rgb is None:
            return None
        t_start = time.perf_counter()

        linear = self._stage_a_b_c()

        t0 = time.perf_counter()
        if self.channel_gains is not None:
            linear = linear * self.channel_gains.reshape(1, 1, 3)
        self.stats.gains = time.perf_counter() - t0

        t0 = time.perf_counter()
        density = self._density(linear, self._base_aligned())
        self.stats.density = time.perf_counter() - t0

        t0 = time.perf_counter()
        density_cmy = self._crosstalk(density)
        self.stats.crosstalk = time.perf_counter() - t0
        # 越窗统计用**解串扰之前**的密度：解串扰会把负密度钳到 0
        # （染料密度非负，物理正确），那样就看不出「比片基亮」了。
        density_for_stats = density

        t0 = time.perf_counter()
        code, _outside = self.transfer.density_to_code(
            density_cmy, count_out_of_window=False)
        self.stats.curve = time.perf_counter() - t0

        self.stats.encode = 0.0
        self.stats.total = time.perf_counter() - t_start

        if count_stats:
            self._update_clip_stats(code, density_for_stats, linear)
        return code

    def _update_clip_stats(self, code: np.ndarray, density: np.ndarray,
                           linear: np.ndarray | None = None):
        """统计必须从**密度**判定越窗，而不是从 code 判定。

        （code 的趾部会被曲线抬起，``code < 黑点`` 并不能代表「比片基亮」。）
        """
        params = self.transfer.params
        t = (density - params.d_min) / params.density_range
        stats = ClipStats()
        stats.total_samples = int(code.size)
        stats.density_below_base = int(np.count_nonzero(t < 0.0))
        stats.density_above_window = int(np.count_nonzero(t > 1.0))
        stats.output_clipped = int(np.count_nonzero(
            (code < 0.0) | (code > CINEON.max_code)))
        buffer = self._stage_a_b_c() if linear is None else linear
        stats.profile_out_of_gamut = int(np.count_nonzero(
            (buffer < 0.0) | (buffer > 1.0)))
        self.clip_stats = stats

    def display_rate(self) -> float | None:
        """预览解码用的**码率**（codes/decade）。

        ``None`` = 用 Cineon 标准的 500。已知片种 γ 时返回
        ``REFERENCE_SPAN · γ / 窗口跨度`` —— 这样编码时压进密度窗口的信息
        会被正确展开，预览才不会"发灰"。未知 γ 时保持标准值（不猜）。
        """
        gamma = getattr(self.hd, "density_per_decade", None)
        if not gamma or gamma <= 0:
            return None
        hd = self.hd
        if hd.has_per_channel and hd.d_max_rgb is not None \
                and hd.d_min_rgb is not None:
            span = float(np.mean([hi - lo for lo, hi in
                                  zip(hd.d_min_rgb, hd.d_max_rgb)]))
        else:
            span = float(hd.density_range)
        if span <= 1e-6:
            return None
        scale = float(getattr(self.view, "tone_scale", 1.0) or 1.0)
        scale = min(max(scale, 0.2), 5.0)
        return float(REFERENCE_SPAN) * float(gamma) / span * scale

    def _encode_log(self, code, space: str):
        """把 code 编码成指定的对数空间（0..1）。"""
        if space == "logc3":
            return self.transfer.code_to_logc3(code)
        return CINEON.normalize(code)

    def process_for_output(self, apply_lut: bool | None = None):
        """归一化 [0,1] 的导出数据（按 ``output_colorspace``）。

        **不主动套用 LUT**：``apply_lut=True`` 时也只套用一次，
        由调用方负责不再重复套用（旧实现内外各套一次，R11）。
        """
        code = self.process_to_code()
        if code is None:
            return None
        transform = self.output
        use_lut = (transform.mode == output_mod.MODE_LUT
                   if apply_lut is None else bool(apply_lut))
        if use_lut:
            if self.lut is None:
                raise ValueError("输出变换要求套用 LUT，但没有可用的 LUT")
            # LUT 按它期望的输入空间取数据（ARRI 的 LogC3→Rec.709 要 LogC3）
            source = self._encode_log(code, transform.lut_input_space)
            return self.lut.apply(source)
        return self._encode_log(code, transform.log_space)

    def process_for_preview(self, view: display.ViewSettings | None = None,
                            scale: float = 1.0, count_stats: bool = False):
        """预览用 uint8（与导出色域严格一致）。

        ``count_stats=True`` 时顺带更新越窗/裁剪统计。调用方通常传的是
        **降采样副本**（例如预览分辨率），因此这一步很便宜，却让分析页
        与窗口提示有真实数字可用 —— 之前 GUI 从不计算统计，导致分析页
        一直显示 0、窗口提示永远说"匹配良好"。
        """
        view = self.view if view is None else view
        code = self.process_to_code(count_stats=count_stats)
        if code is None:
            return None
        transform = self.output
        if transform.mode == output_mod.MODE_LUT and self.lut is not None:
            # 套了色彩还原 LUT：数据已经是**显示域**（如 Rec.709），
            # 不能再按 log 解一遍，否则等于二次伽马。
            values = self.lut.apply(
                self._encode_log(code, transform.lut_input_space))
            space = transform.lut_target_space
        else:
            values = self._encode_log(code, transform.log_space)
            space = transform.log_space
        preview = display.code_to_display(values, space, view,
                                         codes_per_decade=self.display_rate())
        if scale < 1.0:
            step = max(1, int(round(1.0 / scale)))
            preview = preview[::step, ::step]
        return np.ascontiguousarray(preview)

    # ==================================================================
    # 报告 / 序列化
    # ==================================================================

    def get_performance_stats(self) -> dict:
        return self.stats.to_dict()

    def get_clip_stats(self) -> dict:
        return self.clip_stats.to_dict()

    def get_density_report(self) -> dict:
        return self.density_report.to_dict()

    def describe(self) -> dict:
        return {
            "image": None if self.linear_img is None else list(self.linear_img.shape),
            "profile": None if self.profile is None else self.profile.describe(),
            "icc_weight": self.icc_weight,
            "wb": {"mode": self.wb_mode, "gains": self.wb_gains.tolist()},
            "colorchecker": {
                "calibrated": self.colorchecker_calibrated,
                "source": self.calibration_source,
                "matrix": None if self.color_correction_matrix is None
                          else self.color_correction_matrix.tolist(),
            },
            "base": None if self.base_val_rgb is None else self.base_val_rgb.tolist(),
            "base_samples": len(self.base_samples),
            "channel_gains": self.channel_gains.tolist(),
            "density": self.density_report.to_dict(),
            "preset": self.preset_name,
            "hd": self.hd.to_dict(),
            "tone": self.tone.to_dict(),
            "curves": self.curves.to_dict(),
            "transfer": {"auto_fit": self.auto_fit,
                         "codes_per_density": self.codes_per_density,
                         "effective_slope": self.transfer.effective_slope()},
            "output_colorspace": self.output_colorspace,
            "output": self.output.to_dict(),
            "output_label": self.output.label(),
            "lut": self.lut_path,
            "view": self.view.to_dict(),
            "clip_stats": self.get_clip_stats(),
        }

    def to_settings(self) -> dict:
        """完整参数快照（可 JSON 序列化，用于保存/复制到批处理）。"""
        return {
            "schema_version": SCHEMA_VERSION,
            "icc_source": self.icc_source,
            "icc_weight": self.icc_weight,
            "wb_mode": self.wb_mode,
            "wb_gains": self.wb_gains.tolist(),
            "colorchecker": {
                "calibrated": self.colorchecker_calibrated,
                "source": self.calibration_source,
                "matrix": (None if self.color_correction_matrix is None
                           else self.color_correction_matrix.tolist()),
            },
            "base_val_rgb": (None if self.base_val_rgb is None
                             else self.base_val_rgb.tolist()),
            "channel_gains": self.channel_gains.tolist(),
            "gain_clamp": list(self.gain_clamp),
            "target_density": self.target_density,
            "preset": self.preset_name,
            "hd": self.hd.to_dict(),
            "tone": self.tone.to_dict(),
            "curves": self.curves.to_dict(),
            "auto_fit": self.auto_fit,
            "codes_per_density": self.codes_per_density,
            "crosstalk_enabled": self.crosstalk_enabled,
            "clamp_cmy": self.clamp_cmy,
            "output_colorspace": self.output_colorspace,
            "output": self.output.to_dict(),
            "lut_path": self.lut_path,
            "view": self.view.to_dict(),
        }

    def apply_settings(self, data: dict, copy_pixels: bool = False):
        """应用参数快照（``copy_pixels=True`` 时也复制色卡矫正矩阵等）。"""
        from .settings import apply_settings_to_pipeline
        return apply_settings_to_pipeline(self, data, copy_pixels=copy_pixels)


__all__ = [
    "ScientificFilmPipeline", "StageStats", "ClipStats", "DensityReport",
    "TRANSMISSION_FLOOR", "USE_DEFAULT_CLAMP",
]
