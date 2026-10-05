"""非线性校正：胶片 H-D 特性曲线、解析逆解、色调控制、单调样条。

设计要点（为什么不是原来那个退化的 Sigmoid）
------------------------------------------
原实现 ``k = 8/ε``（ε≈0.002~0.005 → k≈1600~4000）把 logistic 的过渡区
压到 ``|t−0.5| < 0.01``，于是「S 曲线」退化成砖墙：实测 512 级灰阶只剩
**3 个不同输出值**。

本模块采用**嵌套软裁剪**（compositional soft-clip）::

    toe(x) = s_t · softplus( a_in · (x − x_c) / s_t )
    t(x)   = 1 − s_s · softplus( (1 − toe(x)) / s_s )

性质（全部有测试守护）：

* **严格单调性由构造保证**：两个 softplus 都严格递增，严格增函数的复合
  仍严格递增，因此 ``dt/dx > 0`` **与参数取值无关**。这是硬要求：逆解
  必须有定义。备选的「差式 softplus」（``t = a·s_t·softplus(…) −
  a·s_s·softplus(…)``）中段更"标准"，但实测在 3000 组随机参数中有
  **94%** 会在实用值域内出现零/负斜率，故弃用。
* **中点精确**：``x_c`` 与 ``a_in`` 由闭式解确定，使 ``t(x_mid) = 0.5``
  与 ``dt/dx|_{x_mid} = a`` 都精确成立。
* **值域端点有限**：``t = T_FLOOR`` 与 ``t = 1 − T_FLOOR`` 都有闭式解，
  分别给出 ``x_floor`` / ``x_ceil``；超出范围的密度映射到这两个**有限**
  端点而不是 ∓∞，从而在 code 域保持连续。
* **参数耦合（如实记录）**：``s_toe`` / ``s_sh`` 并非完全独立 —— 增大趾部
  软度会让整条曲线轻微整体位移（实测在 18% 灰处约 0.05 decade，在趾部
  约 0.33 decade），因此 UI 中标注为「趾/肩圆角半径（会轻微影响整体）」。
  主效应依然明确：改趾部软度时低端响应是中段的 6 倍以上。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from functools import lru_cache

import numpy as np

from .constants import CINEON, HDParams

#: 软度的数值下限（=0 会退化为折线，公式中有除法）
MIN_SOFTNESS = 1e-6
MAX_SOFTNESS = 1.0
MIN_SLOPE = 0.2
MAX_SLOPE = 6.0
#: 曲线值域的实用下限：``t`` 小于此值视为「已达片基黑」。
#: ``t`` 的真正渐近线是 0（对应 x→−∞），必须给定一个有限参考点，
#: 否则超范围像素会映射到 −∞，在 code 域产生巨大跳变。
T_FLOOR = 1e-6


# ======================================================================
# 基础数值函数
# ======================================================================

def softplus(z):
    """数值稳定的 ``log(1 + exp(z))``（大 |z| 不溢出）。"""
    z = np.asarray(z, dtype=np.float64)
    return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))


def sigmoid(z):
    """数值稳定的 logistic。"""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def log_expm1(t):
    """数值稳定的 ``log(exp(t) − 1)``（t>0，大 t 不溢出）。"""
    t = np.asarray(t, dtype=np.float64)
    out = np.empty_like(t)
    small = t < 18.0
    out[small] = np.log(np.expm1(t[small]))
    big = ~small
    out[big] = t[big] + np.log1p(-np.exp(-t[big]))
    return out


# ======================================================================
# 参数净化
# ======================================================================

def sanitize(params: HDParams) -> HDParams:
    """把参数夹到数值安全区间。

    这是**唯一**允许修改用户参数的地方；非法参数不抛错而是夹取，
    并保证 ``sanitize(sanitize(p)) == sanitize(p)``（幂等）。

    嵌套式构造下严格单调性由数学结构保证，因此这里只需夹取数值范围，
    不需要额外的耦合约束（也就不存在不幂等的风险）。
    """
    a = float(np.clip(params.a, MIN_SLOPE, MAX_SLOPE))
    s_toe = float(np.clip(params.s_toe, MIN_SOFTNESS, MAX_SOFTNESS))
    s_sh = float(np.clip(params.s_sh, MIN_SOFTNESS, MAX_SOFTNESS))
    d_min = float(params.d_min)
    d_max = float(params.d_max)
    if not math.isfinite(d_min):
        d_min = HDParams().d_min
    if not math.isfinite(d_max):
        d_max = HDParams().d_max
    if d_max <= d_min + 0.05:
        d_max = d_min + 0.05
    x_mid = float(params.x_mid)
    if not math.isfinite(x_mid):
        x_mid = HDParams().x_mid
    x_mid = float(np.clip(x_mid, -20.0, 20.0))
    def _clean_triple(raw, fallback):
        if raw is None:
            return None
        out = []
        for index in range(3):
            try:
                value = float(raw[index])
            except (TypeError, IndexError, ValueError):
                value = float(fallback)
            out.append(value if math.isfinite(value) else float(fallback))
        return tuple(out)

    a_rgb = _clean_triple(getattr(params, "a_rgb", None), a)
    d_min_rgb = _clean_triple(getattr(params, "d_min_rgb", None), d_min)
    d_max_rgb = _clean_triple(getattr(params, "d_max_rgb", None), d_max)
    if a_rgb is not None:
        a_rgb = tuple(min(max(v, MIN_SLOPE), MAX_SLOPE) for v in a_rgb)
    if d_min_rgb is not None and d_max_rgb is not None:
        fixed = []
        for lo, hi in zip(d_min_rgb, d_max_rgb):
            fixed.append(hi if hi > lo + 0.05 else lo + 0.05)
        d_max_rgb = tuple(fixed)
    # 注意：新增字段必须在这里显式传递，否则 sanitize 会把它们丢掉
    # （曾因此让 density_per_decade / linearize 静默失效）
    gamma = getattr(params, "density_per_decade", None)
    if gamma is not None:
        try:
            gamma = float(gamma)
            gamma = gamma if (math.isfinite(gamma) and gamma > 0) else None
        except (TypeError, ValueError):
            gamma = None
    return HDParams(a=a, x_mid=x_mid, s_toe=s_toe, s_sh=s_sh,
                    d_min=d_min, d_max=d_max,
                    a_rgb=a_rgb, d_min_rgb=d_min_rgb, d_max_rgb=d_max_rgb,
                    linearize=bool(getattr(params, "linearize", False)),
                    density_per_decade=gamma)


# ======================================================================
# 曲线准备（闭式常量）
# ======================================================================

@dataclass(frozen=True)
class PreparedCurve:
    """H-D 曲线的全部闭式常量（由参数一次算出，之后逐像素复用）。"""

    a: float            # 中段精确斜率（t 单位 / decade）
    x_mid: float        # t = 0.5 处的 log10 曝光
    x_center: float     # 内层斜坡的零点
    a_inner: float      # 内层斜坡斜率（已归一化，使中段斜率 = a）
    s_toe: float
    s_sh: float
    x_min: float
    x_max: float
    t_floor: float
    t_ceil: float
    x_floor: float
    x_ceil: float

    @property
    def params(self) -> HDParams:
        return HDParams(a=self.a, x_mid=self.x_mid, s_toe=self.s_toe,
                        s_sh=self.s_sh, d_min=0.0, d_max=1.0)


def _prepare_uncached(a: float, x_mid: float, s_toe: float, s_sh: float) -> PreparedCurve:
    """由净化后的参数算出闭式常量。

    归一化（两个闭式条件同时精确满足）：

    1. ``t(x_mid) = 0.5``
       ⟺ ``softplus((1−toe)/s_s) = 0.5/s_s``
       ⟹ ``toe_at_mid = 1 − s_s·log_expm1(0.5/s_s)``
    2. ``dt/dx|x_mid = a_in·σ(z)·σ(w) = a``
       ⟹ ``a_in = a / [(1 − e^{−toe_at_mid/s_t})·(1 − e^{−0.5/s_s})]``

    值域端点（``T_FLOOR``）：

    * ``t = T_FLOOR``     ⟹ ``toe = 1 − s_s·log_expm1((1−T_FLOOR)/s_s)``
    * ``t = 1 − T_FLOOR`` ⟹ ``toe = 1 − s_s·log_expm1(T_FLOOR/s_s)``（> 1）

    两者都严格为正（因为 ``0 < T_FLOOR < 1``），故 ``x_floor`` /
    ``x_ceil`` 都是有限值。
    """
    w_outer = float(log_expm1(0.5 / s_sh))
    toe_at_mid = 1.0 - s_sh * w_outer
    z_inner = float(log_expm1(toe_at_mid / s_toe))
    d_toe = 1.0 - math.exp(-toe_at_mid / s_toe)     # = σ(z_inner)
    d_out = 1.0 - math.exp(-0.5 / s_sh)             # = σ(w_outer)
    a_inner = a / max(d_toe * d_out, 1e-12)
    x_center = x_mid - s_toe * z_inner / a_inner

    toe_floor = max(1.0 - s_sh * float(log_expm1((1.0 - T_FLOOR) / s_sh)), 1e-12)
    x_floor = x_center + s_toe * float(log_expm1(toe_floor / s_toe)) / a_inner
    toe_ceil = 1.0 - s_sh * float(log_expm1(T_FLOOR / s_sh))
    x_ceil = x_center + s_toe * float(log_expm1(toe_ceil / s_toe)) / a_inner

    span = 20.0 / a_inner
    return PreparedCurve(
        a=a, x_mid=x_mid, x_center=x_center, a_inner=a_inner,
        s_toe=s_toe, s_sh=s_sh,
        x_min=min(x_center - span, x_floor - 1.0),
        x_max=max(x_center + span, x_ceil + 1.0),
        t_floor=T_FLOOR, t_ceil=1.0 - T_FLOOR,
        x_floor=x_floor, x_ceil=x_ceil,
    )


@lru_cache(maxsize=256)
def _prepare_cached(a: float, x_mid: float, s_toe: float, s_sh: float) -> PreparedCurve:
    return _prepare_uncached(a, x_mid, s_toe, s_sh)


def prepare(params: HDParams) -> PreparedCurve:
    """由（已净化的）参数得到闭式常量。"""
    p = sanitize(params)
    return _prepare_cached(p.a, p.x_mid, p.s_toe, p.s_sh)


# ======================================================================
# 正向曲线 / 导数 / 逆解
# ======================================================================

def characteristic(x, params_or_prepared):
    """正向 H-D 曲线：log10 曝光 → 归一化密度 ``t``（严格单调）。"""
    prep = _as_prepared(params_or_prepared)
    x = np.asarray(x, dtype=np.float64)
    toe = prep.s_toe * softplus(prep.a_inner * (x - prep.x_center) / prep.s_toe)
    return 1.0 - prep.s_sh * softplus((1.0 - toe) / prep.s_sh)


def characteristic_derivative(x, params_or_prepared):
    """``dt/dx``：两个 sigmoid 之积，恒 > 0（仅在浮点下溢时变 0）。"""
    prep = _as_prepared(params_or_prepared)
    x = np.asarray(x, dtype=np.float64)
    inner = prep.a_inner * (x - prep.x_center) / prep.s_toe
    toe = prep.s_toe * softplus(inner)
    return sigmoid(inner) * prep.a_inner * sigmoid((1.0 - toe) / prep.s_sh)


def inverse_characteristic(t, params_or_prepared, max_iter: int = 48,
                           tol: float = 1e-11):
    """逆解：归一化密度 ``t`` → log10 曝光。

    带二分保护的牛顿法（``rtsafe`` 风格）：

    * 初值取中段线性模型的闭式解（中段几乎精确，实测 3~6 次收敛）
    * 牛顿步越界或收缩过慢时回退二分
    * ``t`` 超出实用值域 ``[t_floor, t_ceil]`` 时映射到**有限**端点
      ``x_floor`` / ``x_ceil``（用 ``clamped`` 报告）

    返回 ``(x, clamped_mask)``；``x`` 形状与 ``t`` 相同。
    """
    prep = _as_prepared(params_or_prepared)
    t_in = np.asarray(t, dtype=np.float64)
    below = t_in < prep.t_floor
    above = t_in > prep.t_ceil
    t_clamped = np.where(below, prep.t_floor,
                         np.where(above, prep.t_ceil, t_in))
    clamped = below | above

    lo = np.full(t_clamped.shape, prep.x_min, dtype=np.float64)
    hi = np.full(t_clamped.shape, prep.x_max, dtype=np.float64)

    # 初值：中段线性模型 t ≈ a·(x − x_mid) + 0.5
    x = prep.x_mid + (t_clamped - 0.5) / prep.a
    x = np.clip(x, prep.x_floor, prep.x_ceil)

    for _ in range(max_iter):
        f = characteristic(x, prep) - t_clamped
        if np.max(np.abs(f)) <= tol:
            break
        d = characteristic_derivative(x, prep)
        lo = np.where(f < 0.0, x, lo)
        hi = np.where(f > 0.0, x, hi)
        step = np.where(d > 1e-300, -f / np.maximum(d, 1e-300), 0.0)
        x_new = x + step
        bad = (x_new <= lo) | (x_new >= hi) | (np.abs(step) > 0.5 * (hi - lo))
        x_new = np.where(bad, 0.5 * (lo + hi), x_new)
        if np.all(x_new == x):
            break
        x = x_new
    # 单调性不变量：t ∈ [t_floor, t_ceil] 的真解必然落在 [x_floor, x_ceil] 内。
    # 求解器接近端点时可能过冲，这里按数学事实夹取，保证整条映射在 code
    # 域上单调不减（不会出现求解器伪影造成的逆序）。
    x = np.clip(x, prep.x_floor, prep.x_ceil)
    x = np.where(below, prep.x_floor, np.where(above, prep.x_ceil, x))
    return x, clamped


# ======================================================================
# 色调控制
# ======================================================================

LOG10_2 = math.log10(2.0)
#: 18% 中灰在 Cineon 曝光标度上的 code（= 685 + 500·log10(0.18/0.90)）
PIVOT_18_GREY_CODE = float(CINEON.code(CINEON.log_e_from_exposure(0.18)))
#: 参考白与参考黑之间的 code 跨度（590）
REFERENCE_SPAN = float(CINEON.white_code - CINEON.black_code)
#: Cineon 标准的编码密度：500 codes / 1.0 密度单位
DEFAULT_CODES_PER_DENSITY = 500.0


@dataclass(frozen=True)
class ToneControls:
    """色调控制：code 域的曝光/对比度 + ASC-CDL。

    * ``exposure_ev``：按 Cineon 的曝光标度平移 code
      （1 档 = ``codes_per_stop`` = 150.515 codes）
    * ``contrast``：围绕 ``pivot_code``（默认 18% 中灰的 code 335.515）旋转
    * ``lift``(offset) / ``gamma``(power) / ``gain``(slope)：归一化 code 上的
      ASC-CDL ``out = (in·slope + offset)^power``

    为什么全在 code 域：Cineon/LogC3 是**对数编码**，曝光与对比度在这个
    域上就是平移与旋转；而且 LUT / CDL 本来就定义在编码域，预览、导出与
    ``.cube`` 导出因此可以共用同一段代码。
    """

    exposure_ev: float = 0.0
    contrast: float = 1.0
    pivot_code: float = PIVOT_18_GREY_CODE
    master_lift: float = 0.0
    master_gamma: float = 1.0
    master_gain: float = 1.0
    lift: tuple = (0.0, 0.0, 0.0)
    gamma: tuple = (1.0, 1.0, 1.0)
    gain: tuple = (1.0, 1.0, 1.0)

    # ---------- code 域：曝光 / 对比度 ----------
    def apply_code(self, code):
        code = np.asarray(code, dtype=np.float64)
        code = code + self.exposure_ev * CINEON.codes_per_stop
        if self.contrast != 1.0:
            code = self.pivot_code + (code - self.pivot_code) * self.contrast
        return code

    def invert_code(self, code):
        code = np.asarray(code, dtype=np.float64)
        if self.contrast != 1.0 and self.contrast != 0.0:
            code = self.pivot_code + (code - self.pivot_code) / self.contrast
        return code - self.exposure_ev * CINEON.codes_per_stop

    # ---------- code 域的 ASC-CDL（归一化） ----------
    def cdl_slope(self) -> np.ndarray:
        return np.asarray(self.gain, dtype=np.float64) * self.master_gain

    def cdl_offset(self) -> np.ndarray:
        return np.asarray(self.lift, dtype=np.float64) + self.master_lift

    def cdl_power(self) -> np.ndarray:
        return np.asarray(self.gamma, dtype=np.float64) * self.master_gamma

    def apply_cdl(self, n):
        """归一化 code 上的 ASC-CDL（输入必须为 (...,3)）。"""
        n = np.asarray(n, dtype=np.float64)
        if n.ndim == 0 or n.shape[-1] != 3:
            raise ValueError(
                f"ToneControls.apply_cdl 需要形状 (..., 3) 的输入，收到 {n.shape}")
        v = n * self.cdl_slope() + self.cdl_offset()
        v = np.maximum(v, 0.0)
        return np.power(v, self.cdl_power())

    def is_identity_cdl(self) -> bool:
        """仅判断 CDL 部分（曝光/对比度在 code 域另行处理）。"""
        return (self.master_lift == 0.0 and self.master_gamma == 1.0
                and self.master_gain == 1.0
                and tuple(self.lift) == (0.0, 0.0, 0.0)
                and tuple(self.gamma) == (1.0, 1.0, 1.0)
                and tuple(self.gain) == (1.0, 1.0, 1.0))

    def is_identity(self) -> bool:
        return (self.exposure_ev == 0.0 and self.contrast == 1.0
                and self.is_identity_cdl())

    def to_dict(self) -> dict:
        return {
            "exposure_ev": self.exposure_ev, "contrast": self.contrast,
            "pivot_code": self.pivot_code,
            "master_lift": self.master_lift, "master_gamma": self.master_gamma,
            "master_gain": self.master_gain,
            "lift": list(self.lift), "gamma": list(self.gamma),
            "gain": list(self.gain),
        }

    @classmethod
    def from_dict(cls, data) -> "ToneControls":
        out = {}
        for key in ("exposure_ev", "contrast", "pivot_code",
                    "master_lift", "master_gamma", "master_gain"):
            if key in data and data[key] is not None:
                out[key] = float(data[key])
        # v5 早期字段名兼容
        if "pivot_loge" in data and "pivot_code" not in out:
            out["pivot_code"] = PIVOT_18_GREY_CODE
        for key in ("lift", "gamma", "gain"):
            if data.get(key) is not None:
                vals = tuple(float(v) for v in data[key])
                if len(vals) == 3:
                    out[key] = vals
        return replace(cls(), **out)


# ======================================================================
# 单调样条曲线（Fritsch–Carlson PCHIP）
# ======================================================================

def pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fritsch–Carlson 单调三次 Hermite 的节点导数。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    if n == 2:
        return np.full(2, (y[1] - y[0]) / (x[1] - x[0]))
    h = np.diff(x)
    delta = np.diff(y) / h
    m = np.zeros(n)
    m[0] = delta[0]
    m[-1] = delta[-1]
    for i in range(1, n - 1):
        d0, d1 = delta[i - 1], delta[i]
        if d0 * d1 <= 0.0:
            m[i] = 0.0
        else:
            w1 = 2.0 * h[i] + h[i - 1]
            w2 = h[i] + 2.0 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / d0 + w2 / d1)
    # 端点单调性修正
    if delta[0] != 0.0:
        m[0] = (np.clip(m[0], 0.0, 3.0 * delta[0]) if delta[0] > 0
                else np.clip(m[0], 3.0 * delta[0], 0.0))
    if delta[-1] != 0.0:
        m[-1] = (np.clip(m[-1], 0.0, 3.0 * delta[-1]) if delta[-1] > 0
                 else np.clip(m[-1], 3.0 * delta[-1], 0.0))
    return m


def pchip_eval(x: np.ndarray, y: np.ndarray, m: np.ndarray, t):
    """在节点外做常数延拓的单调三次 Hermite 求值。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    idx = np.clip(np.searchsorted(x, t, side='right') - 1, 0, len(x) - 2)
    x0, x1 = x[idx], x[idx + 1]
    y0, y1 = y[idx], y[idx + 1]
    h = x1 - x0
    s = np.clip((t - x0) / h, 0.0, 1.0)
    h00 = (1 + 2 * s) * (1 - s) ** 2
    h10 = s * (1 - s) ** 2
    h01 = s ** 2 * (3 - 2 * s)
    h11 = s ** 2 * (s - 1)
    out = h00 * y0 + h10 * h * m[idx] + h01 * y1 + h11 * h * m[idx + 1]
    return np.where(t < x[0], y[0], np.where(t > x[-1], y[-1], out))


@dataclass
class MonotoneCurve:
    """归一化 code 域上的单调曲线（默认 5 点恒等）。"""

    points: list = field(default_factory=lambda: [(0.0, 0.0), (0.25, 0.25),
                                                  (0.5, 0.5), (0.75, 0.75),
                                                  (1.0, 1.0)])
    _x: np.ndarray = field(init=False, repr=False, default=None)
    _y: np.ndarray = field(init=False, repr=False, default=None)
    _m: np.ndarray = field(init=False, repr=False, default=None)

    def __post_init__(self):
        self._rebuild()

    def _rebuild(self):
        pts = sorted((float(px), float(py)) for px, py in self.points)
        if len(pts) < 2:
            pts = [(0.0, 0.0), (1.0, 1.0)]
        x = np.array([p[0] for p in pts], dtype=np.float64)
        y = np.array([p[1] for p in pts], dtype=np.float64)
        keep = np.concatenate([[True], np.diff(x) > 1e-9])
        x, y = x[keep], y[keep]
        if len(x) < 2:
            x = np.array([0.0, 1.0])
            y = np.array([0.0, 1.0])
        y = np.clip(y, 0.0, 1.0)
        self._x, self._y = x, y
        self._m = pchip_slopes(x, y)

    def set_points(self, points):
        self.points = list(points)
        self._rebuild()
        return self

    def apply(self, t):
        return pchip_eval(self._x, self._y, self._m, t)

    def is_identity(self) -> bool:
        """恒等判定必须看**函数**而不是控制点个数。

        （曾经的 bug：默认 5 点恒等曲线被判为非恒等，于是曲线分支被激活；
        若输入是 1-D 数组，``out[..., 0]`` 会退化成标量并把整个数组
        塌成 ``(3,)`` —— 表现为「512 级灰阶只剩 3 级」。）
        """
        return (len(self._x) >= 2
                and self._x[0] == 0.0 and self._x[-1] == 1.0
                and bool(np.allclose(self._y, self._x, atol=1e-12)))

    def to_list(self) -> list:
        return [[float(a), float(b)] for a, b in zip(self._x, self._y)]

    @classmethod
    def from_list(cls, data) -> "MonotoneCurve":
        if not data:
            return cls()
        return cls(points=[(float(p[0]), float(p[1])) for p in data])

    @classmethod
    def identity(cls) -> "MonotoneCurve":
        return cls()


@dataclass
class CurveSet:
    """主曲线 + R/G/B 分通道曲线。"""

    master: MonotoneCurve = field(default_factory=MonotoneCurve.identity)
    red: MonotoneCurve = field(default_factory=MonotoneCurve.identity)
    green: MonotoneCurve = field(default_factory=MonotoneCurve.identity)
    blue: MonotoneCurve = field(default_factory=MonotoneCurve.identity)

    def apply(self, n):
        """``n`` 必须为 (...,3) 归一化 code。

        形状契约必须严格：最后一维是 3 个通道。传入 1-D 数组会直接报错，
        而不是静默塌成 ``(3,)``（历史 bug，见 ``is_identity`` 的说明）。
        """
        n = np.asarray(n, dtype=np.float64)
        if n.ndim == 0 or n.shape[-1] != 3:
            raise ValueError(
                f"CurveSet.apply 需要形状 (..., 3) 的输入，收到 {n.shape}")
        out = self.master.apply(n)
        return np.stack([self.red.apply(out[..., 0]),
                         self.green.apply(out[..., 1]),
                         self.blue.apply(out[..., 2])], axis=-1)

    def is_identity(self) -> bool:
        return all(c.is_identity() for c in (self.master, self.red,
                                             self.green, self.blue))

    def to_dict(self) -> dict:
        return {
            "master": self.master.to_list(), "red": self.red.to_list(),
            "green": self.green.to_list(), "blue": self.blue.to_list(),
        }

    @classmethod
    def from_dict(cls, data) -> "CurveSet":
        if not data:
            return cls()
        return cls(
            master=MonotoneCurve.from_list(data.get("master")),
            red=MonotoneCurve.from_list(data.get("red")),
            green=MonotoneCurve.from_list(data.get("green")),
            blue=MonotoneCurve.from_list(data.get("blue")),
        )


# ======================================================================
# 完整传递函数（密度 → code）
# ======================================================================

@dataclass
class TransferFunction:
    """整条非线性链：归一化密度 → Cineon code。

    数据流（关键：**曲线作用在密度域**）::

        ① t  = (D − d_min) / (d_max − d_min)     归一化输入密度（可越界）
        ② t' = characteristic(t, hd)             非线性校正（单调 S 曲线）
        ③ code = 95 + codes_per_unit · t'        编码（默认 auto-fit → 95..685）
        ④ code = tone.apply_code(code)           曝光(EV) / 对比度
        ⑤ n = code/1023 → CDL → 分通道曲线 → code
        ⑥ 可选：code → 线性曝光 → LogC3

    为什么曲线在密度域而不是曝光域：Cineon/DPX 编码的是**胶片密度**
    （标准 500 codes / 1.0 密度单位），不是场景曝光。早期实现把曲线当
    「密度→曝光」用，实测 code 会超出 10-bit 量程 **4.5 倍**（30 位
    Cineon code），完全不可用。改为密度域后：

    * ``auto_fit=True``（默认）把整个输入密度窗口映射到标准参考黑/白
      （95 / 685），输出天然落在 10-bit 可用范围内；
    * ``auto_fit=False`` 时按 Cineon 标准的 500 codes/密度单位编码，
      便于与其它 Cineon 工具对齐（可能超范围，末端统一裁剪）。

    预览、导出与 ``.cube`` 导出共用这一个对象，保证三者严格一致。
    """

    hd: HDParams = field(default_factory=HDParams)
    tone: ToneControls = field(default_factory=ToneControls)
    curves: CurveSet = field(default_factory=CurveSet)
    #: 把归一化输入密度 [0,1] 线性映射到参考黑/白（95..685）
    auto_fit: bool = True
    #: ``auto_fit=False`` 时使用的标准编码密度
    codes_per_density: float = DEFAULT_CODES_PER_DENSITY

    #: 上一次 ``density_to_code`` 中落在输入密度窗口之外的**通道样本数**
    #: （(H,W,3) 输入时即像素数 × 越窗通道数）
    last_out_of_window: int = field(default=0, init=False)

    @property
    def prepared(self) -> PreparedCurve:
        return prepare(self.hd)

    @property
    def params(self) -> HDParams:
        """净化后的参数（所有数值计算都必须用它）。"""
        return sanitize(self.hd)

    @property
    def codes_per_unit(self) -> float:
        """归一化密度 ``t`` → code 的斜率。"""
        if self.auto_fit:
            return REFERENCE_SPAN
        return self.codes_per_density * self.params.density_range

    # ---------- 密度 → code ----------
    def density_to_code(self, density, count_out_of_window: bool = True):
        """归一化密度（任意实数）→ Cineon code（不裁剪到 [0,1023]）。

        返回 ``(code, outside_mask)``：``outside_mask`` 标记密度落在声明
        的输入窗口 ``[d_min, d_max]`` 之外的像素（片基以下 / 最大密度
        以上）—— 它们会被曲线的趾/肩平滑压缩，而不是硬裁。
        """
        p = self.params
        d = np.asarray(density, dtype=np.float64)
        if p.has_per_channel and d.ndim >= 1 and d.shape[-1] == 3:
            # 逐通道曲线：真实胶片每层的反差/最大密度都不同，共享一条曲线
            # 无法反演，残余就成了偏色。
            code = np.empty(d.shape, dtype=np.float64)
            outside = np.zeros(d.shape, dtype=bool)
            for index in range(3):
                pc = p.channel(index)
                tc = (d[..., index] - pc.d_min) / pc.density_range
                outside[..., index] = (tc < 0.0) | (tc > 1.0)
                if p.linearize:
                    # 线性化：先取胶片特性曲线的**逆**（去掉胶片自身的
                    # 非线性），再按逐通道增益标定输出。
                    shape = replace(pc, a=1.0, d_min=0.0, d_max=1.0)
                    u = inverse_characteristic(np.clip(tc, -1.0, 2.0), shape)
                    gain = REFERENCE_SPAN / max(float(pc.a), 1e-6)
                    code[..., index] = (CINEON.black_code
                                        + gain * (u - pc.x_mid))
                else:
                    code[..., index] = self.encode_density(
                        characteristic(tc, pc))
        elif p.linearize:
            t = (d - p.d_min) / p.density_range
            outside = (t < 0.0) | (t > 1.0)
            shape = replace(p, a=1.0, d_min=0.0, d_max=1.0)
            u = inverse_characteristic(np.clip(t, -1.0, 2.0), shape)
            code = (CINEON.black_code
                    + (REFERENCE_SPAN / max(float(p.a), 1e-6))
                    * (u - p.x_mid))
        else:
            t = (d - p.d_min) / p.density_range
            outside = (t < 0.0) | (t > 1.0)
            code = self.encode_density(characteristic(t, p))
        code = self.apply_code_domain(code)
        if count_out_of_window:
            self.last_out_of_window = int(np.count_nonzero(outside))
        return code, outside

    def encode_density(self, t_curve):
        """归一化（已校正）密度 → code（不裁剪）。"""
        return CINEON.black_code + self.codes_per_unit * np.asarray(
            t_curve, dtype=np.float64)

    def decode_code(self, code):
        """code → 归一化（已校正）密度（``encode_density`` 的逆）。"""
        return (np.asarray(code, dtype=np.float64) - CINEON.black_code) \
            / self.codes_per_unit

    def apply_code_domain(self, code):
        """code 域的曝光/对比度 + CDL + 分通道曲线。"""
        code = self.tone.apply_code(code)
        if self.tone.is_identity_cdl() and self.curves.is_identity():
            return code
        n = code / CINEON.max_code
        if n.ndim == 0 or n.shape[-1] != 3:
            raise ValueError(
                "启用 CDL/曲线时，density_to_code 需要形状 (..., 3) 的输入；"
                f"收到 {n.shape}（请把单通道数据扩展为 3 通道）")
        if not self.tone.is_identity_cdl():
            n = self.tone.apply_cdl(n)
        if not self.curves.is_identity():
            n = self.curves.apply(n)
        return n * CINEON.max_code

    # ---------- code → 密度（分析与测试用） ----------
    def code_to_density(self, code, clip_to_range: bool = True):
        """Cineon code → 归一化密度（``density_to_code`` 的数值逆）。"""
        code = np.asarray(code, dtype=np.float64)
        if clip_to_range:
            code = np.clip(code, 0.0, CINEON.max_code)
        if not (self.tone.is_identity_cdl() and self.curves.is_identity()):
            n = code / CINEON.max_code
            n = _invert_monotone(self._forward_code_domain, n)
            code = n * CINEON.max_code
        code = self.tone.invert_code(code)
        t_curve = self.decode_code(code)
        p = self.params
        if p.has_per_channel:
            shape = np.shape(t_curve)
            if not (len(shape) >= 1 and shape[-1] == 3):
                raise ValueError(
                    "启用逐通道参数时 code_to_density 需要形状 (..., 3) 的输入："
                    f"单通道 code 无法确定属于哪一层（收到 {shape}）。"
                    "请显式扩展为 3 通道。")
        if p.has_per_channel and np.ndim(t_curve) >= 1 \
                and np.shape(t_curve)[-1] == 3:
            # 逐通道的逆：density_to_code 在逐通道模式下会走逐通道曲线，
            # 逆函数必须用**同一组**逐通道参数，否则往返不一致
            # （测试 test_code_to_density_roundtrip 抓到的就是这个问题）。
            out = np.empty(np.shape(t_curve), dtype=np.float64)
            for index in range(3):
                pc = p.channel(index)
                t = inverse_characteristic(t_curve[..., index], pc)[0]
                out[..., index] = t * pc.density_range + pc.d_min
            return out
        t = inverse_characteristic(t_curve, p)[0]
        return t * p.density_range + p.d_min

    def _forward_code_domain(self, n):
        if not self.tone.is_identity_cdl():
            n = self.tone.apply_cdl(n)
        if not self.curves.is_identity():
            n = self.curves.apply(n)
        return n

    # ---------- 输出空间 ----------
    @staticmethod
    def code_to_exposure(code):
        """Cineon code → 线性曝光（参考白 code 685 = 0.90）。"""
        return CINEON.exposure(CINEON.log_e(code))

    def code_to_logc3(self, code):
        """Cineon code → ARRI LogC3（经线性曝光，标准锚点）。"""
        from .constants import LOGC3
        return LOGC3.encode(self.code_to_exposure(code))

    # ---------- 分析 ----------
    def effective_slope(self, t_lo: float = 0.25, t_hi: float = 0.75) -> float:
        """密度域曲线在 [0.25, 0.75] 的弦斜率。

        恒等曲线（``a=1``、``s``→0）时为 1.0；越大反差越高。
        """
        c_lo = float(characteristic(np.array(t_lo), self.hd))
        c_hi = float(characteristic(np.array(t_hi), self.hd))
        return float((c_hi - c_lo) / (t_hi - t_lo))

    def describe(self) -> dict:
        return {
            "hd": self.hd.to_dict(),
            "tone": self.tone.to_dict(),
            "curves": self.curves.to_dict(),
            "auto_fit": self.auto_fit,
            "codes_per_unit": self.codes_per_unit,
            "effective_slope": self.effective_slope(),
            "pivot_18_grey_code": PIVOT_18_GREY_CODE,
        }

    def sample(self, n: int = 256):
        """返回 ``(density, code)`` 采样，供曲线绘制；``code`` 恒为 (n,3)。"""
        p = self.params
        t = np.linspace(0.0, 1.0, n)
        t_curve = characteristic(t, p)
        code = self.encode_density(t_curve)
        code3 = np.repeat(code[:, None], 3, axis=1)
        code3 = self.apply_code_domain(code3)
        density = t * p.density_range + p.d_min
        return density, code3


def _invert_monotone(func, target, lo=0.0, hi=1.0, iters: int = 60):
    """对单调函数做二分求逆（``func`` 需支持数组输入）。"""
    target = np.asarray(target, dtype=np.float64)
    lo = np.full(target.shape, lo, dtype=np.float64)
    hi = np.full(target.shape, hi, dtype=np.float64)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f = func(mid)
        lo = np.where(f < target, mid, lo)
        hi = np.where(f >= target, mid, hi)
    return 0.5 * (lo + hi)


def _as_prepared(params_or_prepared) -> PreparedCurve:
    if isinstance(params_or_prepared, PreparedCurve):
        return params_or_prepared
    return prepare(params_or_prepared)


__all__ = [
    "softplus", "sigmoid", "log_expm1", "sanitize", "prepare", "PreparedCurve",
    "characteristic", "characteristic_derivative", "inverse_characteristic",
    "ToneControls", "MonotoneCurve", "CurveSet", "TransferFunction",
    "pchip_slopes", "pchip_eval", "PIVOT_18_GREY_CODE", "REFERENCE_SPAN",
    "DEFAULT_CODES_PER_DENSITY", "LOG10_2",
    "MIN_SOFTNESS", "MAX_SOFTNESS", "MIN_SLOPE", "MAX_SLOPE", "T_FLOOR",
]
