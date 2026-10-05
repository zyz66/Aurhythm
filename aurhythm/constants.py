"""标准常数：色卡参考值、Cineon / LogC3 编码规格、胶片预设。

本模块是纯数据 + 纯函数，只依赖 numpy 与标准库。

编码约定（全文唯一约定，编码与解码共用）
--------------------------------------
Cineon 10-bit（SMPTE/柯达参考实现）

* ``code 95``  = 黑（负片未曝光片基 / 趾部渐近线）
* ``code 685`` = 90% 白卡（参考白）
* ``500`` codes per decade of scene exposure
  → 1 档曝光 = 500·log10(2) = **150.515** codes

因此在「相对片基趾部的 log10 曝光」坐标 ``logE``（趾部 = 0）上有::

    code(logE) = 95 + 500 * logE
    logE(code) = (code - 95) / 500
    E(logE)    = 0.90 * 10 ** (logE - (685 - 95) / 500)

``0.90`` 与参考白码 685 的对应关系来自 Cineon 工具的通行定义
（参考白 = 90% 白卡）。18% 中灰因此落在 ``code 335.5``，并且
``LogC3(0.18) = 0.391007``，即 ARRI 官方参考点 —— 两条输出链路
互为同一 log 信号的不同视图，天然一致。

参考来源
--------
* ARRI LogC 规范（LogC3 / EI800 常数）
  https://www.arri.com/resource/blob/390448/871b733469a2f2c2099ca51b1516bcd1/arri-logc-false-color-specification-data.pdf
* Cineon 参考黑 95 / 参考白 685 = 90% 白卡
  https://download.autodesk.com/us/maya/2013help-composite/files/GUID-B6CDA21A-DCC7-4094-9732-DF858026BF93.htm
* 参考白 log 曝光归一化（SMPTE Journal 102/12）
  https://journal.smpte.org/periodicals/SMPTE%20Journal/102/12/6/07238748.pdf
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

#: 印片光（通道增益）的默认裁剪范围：**±3 档**（2³ = 8、2⁻³ = 0.125）。
#: 触限时会明确回报 ``clamped=True``，不再静默夹住（见回归 R5）。
DEFAULT_GAIN_CLAMP = (0.125, 8.0)

#: 密度域中性化的默认目标中间调密度（彩色负片印片的常用值）
DEFAULT_TARGET_DENSITY = 0.7

SCHEMA_VERSION = 1

# ======================================================================
# ColorChecker Classic 24 —— X-Rite 官方 sRGB(8-bit) 参考值
# ======================================================================
# 用途：色卡矫正的目标值（线性化后使用）与 ΔE 参考。
# 现行代码里的 COLORCHECKER_24_D50 是占位假数据（前 6 行近重复），已废弃。

COLORCHECKER_24_NAMES = (
    "dark skin", "light skin", "blue sky", "foliage",
    "blue flower", "bluish green", "orange", "purplish blue",
    "moderate red", "purple", "yellow green", "orange yellow",
    "blue", "green", "red", "yellow",
    "magenta", "cyan", "white 9.5", "neutral 8",
    "neutral 6.5", "neutral 5", "neutral 3.5", "black 2",
)

COLORCHECKER_24_SRGB8 = np.array([
    [115,  82,  68],   # 1  dark skin
    [194, 150, 130],   # 2  light skin
    [ 98, 122, 157],   # 3  blue sky
    [ 87, 108,  67],   # 4  foliage
    [133, 128, 177],   # 5  blue flower
    [103, 189, 170],   # 6  bluish green
    [214, 126,  44],   # 7  orange
    [ 80,  91, 166],   # 8  purplish blue
    [193,  90,  99],   # 9  moderate red
    [ 94,  60, 108],   # 10 purple
    [157, 188,  64],   # 11 yellow green
    [224, 163,  46],   # 12 orange yellow
    [ 56,  61, 150],   # 13 blue
    [ 70, 148,  73],   # 14 green
    [175,  54,  60],   # 15 red
    [231, 199,  31],   # 16 yellow
    [187,  86, 149],   # 17 magenta
    [  8, 133, 161],   # 18 cyan
    [243, 243, 242],   # 19 white 9.5
    [200, 200, 200],   # 20 neutral 8
    [160, 160, 160],   # 21 neutral 6.5
    [122, 122, 121],   # 22 neutral 5
    [ 85,  85,  85],   # 23 neutral 3.5
    [ 52,  52,  52],   # 24 black 2
], dtype=np.float64)

#: sRGB 编码域 [0,1] 的参考值（未线性化）
COLORCHECKER_24_SRGB = COLORCHECKER_24_SRGB8 / 255.0

#: **兼容别名**：旧版本用过 ``COLORCHECKER_24_D50`` 这个名字（当年是占位
#: 假数据，前 6 行近重复，已废弃）。这里指向现行的 X-Rite sRGB 参考值，
#: 保证旧脚本仍能拿到 (24,3) 的数组。**它不是真正的 D50 值**，
#: 需要 D50/XYZ 请用 :mod:`aurhythm.colorimetry` 现算。
COLORCHECKER_24_D50_COMPAT = COLORCHECKER_24_SRGB

#: 中性梯索引（19..24，1-based → 18..23）与色块索引
COLORCHECKER_NEUTRAL_SLICE = slice(18, 24)


# ======================================================================
# Cineon 10-bit
# ======================================================================

@dataclass(frozen=True)
class CineonSpec:
    """Cineon 10-bit 对数编码规格（编码与解码共用，符号约定唯一）。"""

    black_code: float = 95.0
    white_code: float = 685.0
    codes_per_decade: float = 500.0
    #: 参考白（code = white_code）对应的线性曝光量，0.90 = 90% 白卡
    white_exposure: float = 0.90
    #: 满量程上限（10-bit）
    max_code: float = 1023.0

    # ---------- 派生量 ----------
    @property
    def reference_decades(self) -> float:
        """参考白相对黑所占的 decade 数（标准 = 1.18）。"""
        return (self.white_code - self.black_code) / self.codes_per_decade

    @property
    def codes_per_stop(self) -> float:
        """1 档曝光对应的 code 数（标准 = 150.515）。"""
        return self.codes_per_decade * math.log10(2.0)

    @property
    def log_e_black(self) -> float:
        return 0.0

    # ---------- 编解码 ----------
    def code(self, log_e):
        """相对片基趾部的 log10 曝光 → Cineon code（不裁剪）。"""
        return self.black_code + self.codes_per_decade * np.asarray(log_e, dtype=np.float64)

    def log_e(self, code):
        """Cineon code → 相对片基趾部的 log10 曝光。"""
        return (np.asarray(code, dtype=np.float64) - self.black_code) / self.codes_per_decade

    def exposure(self, log_e):
        """相对 log10 曝光 → 线性曝光量（参考白 = white_exposure）。"""
        log_e = np.asarray(log_e, dtype=np.float64)
        return self.white_exposure * np.power(10.0, log_e - self.reference_decades)

    def log_e_from_exposure(self, exposure):
        """线性曝光量 → 相对 log10 曝光（``exposure`` 的逆）。"""
        exposure = np.asarray(exposure, dtype=np.float64)
        safe = np.maximum(exposure, np.finfo(np.float64).tiny)
        return self.reference_decades + np.log10(safe / self.white_exposure)

    def normalize(self, code):
        """code → [0,1] 归一化（导出用）。"""
        return np.clip(np.asarray(code, dtype=np.float64), 0.0, self.max_code) / self.max_code

    def denormalize(self, value):
        """[0,1] 归一化 → code。"""
        return np.asarray(value, dtype=np.float64) * self.max_code


CINEON = CineonSpec()


# ======================================================================
# ARRI LogC3 (EI 800)
# ======================================================================

@dataclass(frozen=True)
class LogC3Spec:
    """ARRI LogC3（EI 800）传递函数常数。

    ``y = c*log10(a*E + b) + d``（E > cut），否则 ``y = e*E + f``。
    ``E`` 为线性场景曝光，18% 中灰 = 0.18 → y = 0.391007。
    """

    a: float = 5.555556
    b: float = 0.052272
    c: float = 0.247190
    d: float = 0.385537
    e: float = 5.367655
    f: float = 0.092809
    cut: float = 0.010591

    def encode(self, exposure):
        """线性曝光 → LogC3（[0,1] 量级，不裁剪）。"""
        e_in = np.maximum(np.asarray(exposure, dtype=np.float64), 0.0)
        log_branch = self.c * np.log10(self.a * e_in + self.b) + self.d
        lin_branch = self.e * e_in + self.f
        return np.where(e_in > self.cut, log_branch, lin_branch)

    def decode(self, value):
        """LogC3 → 线性曝光（``encode`` 的数值逆，用于测试与显示）。"""
        v = np.asarray(value, dtype=np.float64)
        lin_branch = (v - self.f) / self.e
        log_branch = (np.power(10.0, (v - self.d) / self.c) - self.b) / self.a
        return np.where(v > self.cut * self.e + self.f, log_branch, lin_branch)


LOGC3 = LogC3Spec()


# ======================================================================
# 胶片特性曲线参数（H-D）
# ======================================================================

@dataclass(frozen=True)
class HDParams:
    """非线性校正曲线的参数（作用在**归一化密度域**上）。

    曲线把归一化输入密度 ``t`` 映射为归一化输出密度 ``t'``，是一条单调
    S 曲线（实现见 :mod:`aurhythm.tonemap`）::

        toe(t) = s_t · softplus( a_in · (t − t_c) / s_t )
        t'(t)  = 1 − s_s · softplus( (1 − toe(t)) / s_s )

    参数含义（都在归一化密度 0..1 上）::

        a     反差倍数。1.0 = 保持原反差；>1 增加反差（S 曲线更陡）。
              a_in 经过闭式归一化，使中段斜率精确等于 a。
        x_mid 曲线中点（t = 0.5）所在的归一化密度位置。0.5 = 正中，
              调低会把中灰往上推（提亮中间调）。
        s_toe 趾部圆角半径（低密度端，片基/暗部）。
        s_sh  肩部圆角半径（高密度端，高光）。
        d_min 输入密度窗口下限（片基 + 灰雾）。
        d_max 输入密度窗口上限（最大密度）。

    ``a=1.0、x_mid=0.5、s→0`` 时曲线退化为恒等映射（"无预设"）。
    """

    a: float = 1.0
    x_mid: float = 0.5
    s_toe: float = 0.06
    s_sh: float = 0.06
    d_min: float = 0.18
    d_max: float = 3.10

    # ---------- 逐通道覆盖 ----------
    #: 真实彩色负片的三个感光层，其**反差**与**最大密度**都不同
    #: （实测级别的差异：逐层 d_max 差约 12%、逐层 γ 差约 13%）。
    #: 一条共享曲线在数学上无法反演逐层差异 —— 残余的逐通道影调差就
    #: 表现为**偏色**（这正是"所有预设最后都偏蓝/偏暖"的根因）。
    #: ``None`` = 该字段沿用上面的共享值。
    d_min_rgb: tuple | None = None
    d_max_rgb: tuple | None = None
    a_rgb: tuple | None = None

    #: ``True`` = **线性化模式**：曲线改用胶片特性曲线的**逆**
    #: （``inverse_characteristic``），即"把胶片自身的非线性去掉"。
    #: 这才是"非线性校正"该做的事；``a_rgb`` 此时作为**输出标定**
    #: （逐通道增益），而不是曲线内部斜率。
    #:
    #: 为什么必须有这个模式：用正向 S 曲线去**逼近**曲线之逆时，两者不在
    #: 同一形状族，两端必然留残差（实测中性阶梯离散度只能压到 ~0.8 code、
    #: 边缘区仍偏 1~3%）。改用精确的逆之后，中性在数学上精确成立。
    linearize: bool = False

    #: 片种的 **γ**（每 decade 曝光产生多少密度，datasheet 上的物理量）。
    #: 用途：显示端要按 ``REFERENCE_SPAN·γ/range`` 的码率去解，才能把编码时
    #: 压进去的密度"展开"回曝光。用标准的 500 codes/decade 去解一条 γ=0.65
    #: 的负片，画面会被压扁 → **发灰**（实测彩度只剩真值的 55%）。
    density_per_decade: float | None = None

    @property
    def has_per_channel(self) -> bool:
        return any(v is not None
                   for v in (self.d_min_rgb, self.d_max_rgb, self.a_rgb))

    def channel(self, index: int) -> "HDParams":
        """取第 ``index`` 个通道的单通道参数（未覆盖的沿用共享值）。"""
        index = int(index) % 3
        changes = {}
        for name in ("a", "d_min", "d_max"):
            values = getattr(self, f"{name}_rgb")
            if values is not None:
                changes[name] = float(values[index])
        return replace(self, d_min_rgb=None, d_max_rgb=None, a_rgb=None,
                       linearize=self.linearize, **changes)

    def with_channels(self, a=None, d_min=None, d_max=None) -> "HDParams":
        """返回带逐通道覆盖的副本（None 表示该字段不启用逐通道）。"""
        return replace(
            self,
            a_rgb=None if a is None else tuple(float(v) for v in a),
            d_min_rgb=None if d_min is None else tuple(float(v) for v in d_min),
            d_max_rgb=None if d_max is None else tuple(float(v) for v in d_max))

    # ---------- 派生断点 ----------
    @property
    def x_toe(self) -> float:
        return self.x_mid - 0.5 / self.a

    @property
    def x_shoulder(self) -> float:
        return self.x_mid + 0.5 / self.a

    @property
    def density_range(self) -> float:
        return self.d_max - self.d_min

    def to_dict(self) -> dict:
        return {
            "a": self.a, "x_mid": self.x_mid,
            "s_toe": self.s_toe, "s_sh": self.s_sh,
            "d_min": self.d_min, "d_max": self.d_max,
            "d_min_rgb": None if self.d_min_rgb is None else list(self.d_min_rgb),
            "d_max_rgb": None if self.d_max_rgb is None else list(self.d_max_rgb),
            "a_rgb": None if self.a_rgb is None else list(self.a_rgb),
            "linearize": bool(self.linearize),
            "density_per_decade": self.density_per_decade,
        }

    @classmethod
    def from_dict(cls, data) -> "HDParams":
        base = cls()
        values = {}
        for key in ("a", "x_mid", "s_toe", "s_sh", "d_min", "d_max"):
            if key in data and data[key] is not None:
                values[key] = float(data[key])
        # v4 兼容：hd_slope / hd_mid / hd_clip_softness
        if "a" not in values and "hd_slope" in data:
            values["a"] = float(data["hd_slope"])
        if "x_mid" not in values and "hd_mid" in data:
            values["x_mid"] = float(data["hd_mid"])
        if "hd_clip_softness" in data:
            soft = float(data["hd_clip_softness"])
            values.setdefault("s_toe", soft)
            values.setdefault("s_sh", soft)
        if "hd_min" in data:
            values.setdefault("d_min", float(data["hd_min"]))
        if "hd_max" in data:
            values.setdefault("d_max", float(data["hd_max"]))
        if data.get("linearize") is not None:
            values["linearize"] = bool(data["linearize"])
        if data.get("density_per_decade") is not None:
            values["density_per_decade"] = float(data["density_per_decade"])
        for key in ("d_min_rgb", "d_max_rgb", "a_rgb"):
            raw = data.get(key)
            if raw is not None and len(raw) == 3:
                values[key] = tuple(float(v) for v in raw)
        return replace(base, **values)


# ======================================================================
# 胶片预设
# ======================================================================

@dataclass(frozen=True)
class FilmPreset:
    """胶片预设 = 解串扰矩阵 + H-D 曲线参数。"""

    name: str
    description: str
    matrix_inv: np.ndarray          # RGB 密度 → CMY 密度
    hd: HDParams
    skip: bool = False
    #: 诚实标注参数来源；没有官方数据的不得声称「从柯达曲线采点」
    source_note: str = "风格化估计（无官方 H-D 曲线数据，参数为相机负片典型值）"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "matrix_inv": self.matrix_inv.tolist(),
            "hd": self.hd.to_dict(),
            "skip": self.skip,
            "source_note": self.source_note,
        }


def _xtalk(r1, r2, r3):
    """构造满足约束的解串扰矩阵：对角 >0、非对角 <0、行和 ≈1.3~1.5。"""
    return np.array([r1, r2, r3], dtype=np.float64)


_ESTIMATE = "风格化估计（无官方 H-D 曲线数据；解串扰矩阵与曲线参数为负片典型值）"


def _identity_hd() -> HDParams:
    """恒等曲线：反差 1、中点居中、趾肩取数值下限（无胶片模型）。

    软度取 1e-6（:data:`aurhythm.tonemap.MIN_SOFTNESS`）：残余偏差约
    4e-4 个 code，远低于任何可观测或可用的精度。
    """
    return HDParams(a=1.0, x_mid=0.5, s_toe=1e-6, s_sh=1e-6, d_min=0.0, d_max=1.0)


FILM_PRESETS: dict[str, FilmPreset] = {
    "无 (关闭预设)": FilmPreset(
        name="无 (关闭预设)",
        description="跳过胶片曲线：密度线性映射到编码范围（仅做曝光/对比度）",
        matrix_inv=np.eye(3),
        hd=_identity_hd(),
        skip=True,
        source_note="线性（无胶片模型）",
    ),
    "Kodak Portra 400": FilmPreset(
        name="Kodak Portra 400",
        description="柯达 Portra 400 负片，人像肤色倾向；低反差、宽肩部",
        matrix_inv=_xtalk([1.842, -0.356, -0.124],
                          [-0.187, 1.423, -0.089],
                          [-0.031, -0.213, 1.568]),
        hd=HDParams(a=1.18, x_mid=0.46, s_toe=0.10, s_sh=0.15,
                    d_min=0.18, d_max=3.05),
        source_note=_ESTIMATE,
    ),
    "Kodak Portra 800": FilmPreset(
        name="Kodak Portra 800",
        description="柯达 Portra 800 负片，高感光度；反差更低、肩部更软",
        matrix_inv=_xtalk([1.867, -0.361, -0.131],
                          [-0.192, 1.445, -0.096],
                          [-0.038, -0.221, 1.592]),
        hd=HDParams(a=1.12, x_mid=0.47, s_toe=0.11, s_sh=0.17,
                    d_min=0.20, d_max=3.00),
        source_note=_ESTIMATE,
    ),
    "Kodak Vision3 250D (5207)": FilmPreset(
        name="Kodak Vision3 250D (5207)",
        description="柯达 Vision3 250D 电影负片，日光平衡(5500K)；标准反差",
        matrix_inv=_xtalk([1.873, -0.398, -0.152],
                          [-0.221, 1.472, -0.108],
                          [-0.048, -0.241, 1.635]),
        hd=HDParams(a=1.32, x_mid=0.45, s_toe=0.08, s_sh=0.09,
                    d_min=0.12, d_max=3.20),
        source_note=_ESTIMATE,
    ),
    "Kodak Vision3 500T (5219)": FilmPreset(
        name="Kodak Vision3 500T (5219)",
        description="柯达 Vision3 500T 电影负片，钨丝灯平衡(3200K)；标准反差",
        matrix_inv=_xtalk([1.945, -0.378, -0.148],
                          [-0.185, 1.534, -0.119],
                          [-0.041, -0.228, 1.672]),
        hd=HDParams(a=1.28, x_mid=0.46, s_toe=0.09, s_sh=0.10,
                    d_min=0.13, d_max=3.25),
        source_note=_ESTIMATE,
    ),
    "Kodak Gold 200": FilmPreset(
        name="Kodak Gold 200",
        description="柯达 Gold 200 负片，日常卷；中等反差、暖调",
        matrix_inv=_xtalk([1.780, -0.340, -0.115],
                          [-0.175, 1.410, -0.085],
                          [-0.028, -0.205, 1.540]),
        hd=HDParams(a=1.22, x_mid=0.48, s_toe=0.09, s_sh=0.12,
                    d_min=0.20, d_max=3.00),
        source_note=_ESTIMATE,
    ),
    "Fuji C200": FilmPreset(
        name="Fuji C200",
        description="富士 C200 负片，绿色表现倾向；反差略高",
        matrix_inv=_xtalk([1.756, -0.312, -0.108],
                          [-0.165, 1.387, -0.078],
                          [-0.025, -0.198, 1.512]),
        hd=HDParams(a=1.25, x_mid=0.47, s_toe=0.09, s_sh=0.11,
                    d_min=0.22, d_max=2.95),
        source_note=_ESTIMATE,
    ),
    "通用负片 (默认)": FilmPreset(
        name="通用负片 (默认)",
        description="通用彩色负片模型，适用于未知胶片类型",
        matrix_inv=_xtalk([1.700, -0.350, -0.120],
                          [-0.180, 1.400, -0.090],
                          [-0.030, -0.200, 1.500]),
        hd=HDParams(a=1.20, x_mid=0.47, s_toe=0.09, s_sh=0.11,
                    d_min=0.18, d_max=3.10),
        source_note=_ESTIMATE,
    ),
}

DEFAULT_PRESET = "无 (关闭预设)"

#: 兼容旧引用
NO_PRESET = FILM_PRESETS[DEFAULT_PRESET]

#: 预设的规范顺序（UI 下拉框）
PRESET_ORDER = tuple(FILM_PRESETS.keys())

# ======================================================================
# C-41 片种预设的统一修正（**临时值，待你按柯达官方吸收曲线参数化后替换**）
# ======================================================================

#: 各 C-41 片种的**典型 γ**（密度/decade，公开资料量级）—— 片种之间真正客观
#: 的差异之一（另一个是 D-min/D-max）。**随手填的典型值，不是实测**。
_STOCK_GAMMA_TYPICAL = {
    "Kodak Portra 400": 0.60,
    "Kodak Portra 800": 0.62,
    "Kodak Vision3 250D (5207)": 0.55,
    "Kodak Vision3 500T (5219)": 0.55,
    "Kodak Gold 200": 0.62,
    "Fuji C200": 0.60,
    "通用负片 (默认)": 0.60,
}

#: 各片种的逐通道窗口 ``(a_rgb, d_min_rgb, d_max_rgb)``。
#:
#: **重要：临时值（"随手填"）**。做法：C-41 工艺的染料与成色剂是**工艺级
#: 常量**，所以所有 C-41 片种共用同一组染料吸收（→ 同一个解串扰矩阵，
#: 见 ``_c41_matching_matrix``）；然后按**每个预设自己的密度窗口**用
#: ``filmsim.derive_per_channel_calibration`` 推导逐通道值。
#: 等按柯达官方 RGB 吸收曲线反推出每只胶片的真实矩阵后，应整表替换。
#: ``tests/test_filmsim.py`` 里有测试重新推导并比对，防止悄悄漂移。
_PER_CHANNEL_BY_PRESET = {
    "Kodak Portra 400": ((1.0, 1.0, 1.0),
                         (0.3588, 0.3335, 0.3646), (1.3804, 1.3783, 1.2876)),
    "Kodak Portra 800": ((1.0, 1.0, 1.0),
                         (0.3617, 0.3357, 0.369), (1.3963, 1.3948, 1.3014)),
    "Kodak Vision3 250D (5207)": ((1.0, 1.0, 1.0),
                                  (0.3509, 0.3284, 0.3578),
                                  (1.3917, 1.3871, 1.2974)),
    "Kodak Vision3 500T (5219)": ((1.0, 1.0, 1.0),
                                  (0.3298, 0.3072, 0.3421),
                                  (1.3612, 1.357, 1.2717)),
    "Kodak Gold 200": ((1.0, 1.0, 1.0),
                       (0.3617, 0.3357, 0.369), (1.3963, 1.3948, 1.3014)),
    "Fuji C200": ((1.0, 1.0, 1.0),
                  (0.3808, 0.3581, 0.3828), (1.3895, 1.3859, 1.2966)),
    "通用负片 (默认)": ((1.0, 1.0, 1.0),
                        (0.3523, 0.327, 0.3614), (1.3735, 1.3722, 1.2829)),
}

# ======================================================================
# C-41 彩色负片的物理参数（合成测试负片与配套预设共用）
# ======================================================================

#: 染料消光矩阵：行 = 染料（青 C / 品红 M / 黄 Y），列 = 对 (R,G,B) 的密度贡献。
#: 对角线是主吸收，非对角是**不想要的吸收**（相邻波段串扰）—— 这正是管线里
#: 「解串扰矩阵」要反掉的东西。数量级按 C-41 染料侧吸收的典型比例取。
DYE_EXTINCTION_C41 = np.array([
    [1.00, 0.25, 0.10],   # 青染料（红敏层）：主吸红，另吸绿 25%、蓝 10%
    [0.05, 1.00, 0.20],   # 品红染料（绿敏层）：主吸绿，另吸蓝 20%、红 5%
    [0.02, 0.15, 1.00],   # 黄染料（蓝敏层）：主吸蓝，另吸绿 15%、红 2%
], dtype=np.float64)

#: 橙色罩（未曝光片基的成色剂密度，Status M 量级）：**蓝 > 绿 > 红**，
#: 所以片基看起来是橙色的。管线里的「片基采样」正是要量它。
ORANGE_MASK_DENSITY_C41 = (0.20, 0.50, 0.75)

#: 逐层最大染料密度与逐层反差 —— filmsim 与配套预设**共用同一份**。
#: **为什么是这些数值**：``characteristic`` 的直线段是 ``t ∈ [0,1]``，而
#: ``t = contrast·x + 0.5``，所以**直线段宽度 = 1/contrast 个 decade**、
#: ``γ = d_max·contrast``。早期取 contrast≈1.0、d_max≈2.2 → 直线段仅 3.3 档、
#: γ 高达 2.18（真实彩色负片 8~10 档、0.55~0.75）→ 8.5 档的场景一半落进
#: 趾/肩饱和区 → 合成负片必然掉饱和度发灰。
#: 现在：contrast≈0.40 → 直线段 8.3 档；d_max≈1.6 → γ = 0.59~0.69 ✓
LAYER_DMAX_C41 = (1.66, 1.63, 1.58)
LAYER_CONTRAST_C41 = (0.400, 0.424, 0.376)


def _c41_matching_matrix() -> np.ndarray:
    """反掉生成侧染料串扰的 3×3（= ``inv(Eᵀ)``）。"""
    return np.linalg.inv(DYE_EXTINCTION_C41.T)


def _c41_density_window() -> tuple:
    """本片种在**通道密度**上的真实范围（相对片基）：``d_min=0``（片基即黑场），
    ``d_max = max(Eᵀ · 逐层最大染料密度)``。**不要手打这两个数**：手打偏低会
    把大量像素顶出窗口、丢掉高光层次。"""
    return 0.0, float(np.max(DYE_EXTINCTION_C41.T
                             @ np.asarray(LAYER_DMAX_C41, dtype=np.float64)))


C41_DENSITY_WINDOW = _c41_density_window()

FILM_PRESETS["测试：C-41 合成负片（橙色罩+串扰）"] = FilmPreset(
    name="测试：C-41 合成负片（橙色罩+串扰）",
    description="与 synth 生成的合成测试负片严格配套：反掉三层染料串扰，"
                "密度窗口与逐通道曲线按该片种物理参数推导",
    matrix_inv=_c41_matching_matrix(),
    hd=HDParams(a=1.0, x_mid=0.5, s_toe=0.08, s_sh=0.10,
                d_min=C41_DENSITY_WINDOW[0], d_max=C41_DENSITY_WINDOW[1],
                a_rgb=(1.0, 1.0, 1.0),
                d_min_rgb=(0.3488, 0.3231, 0.3558),
                d_max_rgb=(1.392, 1.3892, 1.2969),
                density_per_decade=0.650),
    source_note="矩阵由 DYE_EXTINCTION_C41 求逆、窗口与逐通道曲线由物理参数"
                "推导（目标=中性一致，非手打）",
)

for _name, _gamma in _STOCK_GAMMA_TYPICAL.items():
    _preset = FILM_PRESETS[_name]
    _a_rgb, _d_min_rgb, _d_max_rgb = _PER_CHANNEL_BY_PRESET[_name]
    FILM_PRESETS[_name] = replace(
        _preset,
        matrix_inv=_c41_matching_matrix(),
        hd=replace(_preset.hd, a_rgb=_a_rgb, d_min_rgb=_d_min_rgb,
                   d_max_rgb=_d_max_rgb, density_per_decade=_gamma),
        source_note="矩阵与逐通道结构暂用 C-41 共用（工艺级常量），逐通道值按"
                    "本预设窗口推导；γ 为公开典型值 —— 均为**临时值**，"
                    "待按柯达官方 RGB 吸收曲线反推后整表替换")

PRESET_ORDER = tuple(FILM_PRESETS.keys())
