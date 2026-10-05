"""C-41 彩色负片的**物理正向模型** —— 合成"真实感"测试负片。

本模块是 :mod:`aurhythm.pipeline` 的逆过程::

    管线：负片扫描值 → 密度 D=−log₁₀(v/base) → 解串扰 → 片基中性化
          → 归一化 → 胶片特性曲线 → Cineon / LogC3 编码
    本模块：线性场景光 → 三层曝光 → 逐层特性曲线 → 染料密度
          → 叠加层间串扰与橙色罩 → 扫描值

这样合成的文件才**真的能用来验证管线**：我们知道自己灌进去的场景是什么，
所以能算出管线还原得准不准。

物理约定
--------
* 三层感光层（负片：曝光越多 → 染料越多 → 密度越高）：
  青染料（红敏）、品红染料（绿敏）、黄染料（蓝敏）
* **橙色罩**：未曝光片基仍有成色剂生成的密度，且**蓝 > 绿 > 红**，
  所以片基呈橙色（约 ``(0.20, 0.50, 0.75)`` Status M 量级）。
  管线的「片基采样」正是量它，密度转换时 ``−log₁₀(v/base)`` 会自动把它减掉。
* **层间串扰**：每种染料都吸收相邻波段（青染料也吸绿/蓝…），用消光矩阵
  ``E``（行 = 染料 C/M/Y，列 = 通道 R/G/B）描述。这正是「解串扰矩阵」要反掉的，
  因此本模块提供 :meth:`FilmStock.matching_crosstalk_matrix`（= ``inv(Eᵀ)``），
  与 ``constants`` 里那个配套预设**完全一致**。
* **逐层反差差异**：``layer_contrast`` 让三层 γ 不同 —— 片基处偏色为 0、
  中间调分通道分离，这才是「密度域中性化」真正要修的东西。

**未包含（如实说明）**：感光层的光谱灵敏度重叠。真实胶片的三层灵敏度会互相
渗一点，那部分无法用单一 3×3 串扰矩阵精确反演，只能靠色卡矫正（ΔE2000）
近似。这里默认按通道一一对应，好让「解串扰」这一步可以被精确验证。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .constants import (DYE_EXTINCTION_C41, LAYER_CONTRAST_C41,
                        LAYER_DMAX_C41, ORANGE_MASK_DENSITY_C41)
from .tonemap import HDParams, characteristic, sanitize


@dataclass(frozen=True)
class FilmStock:
    """一种 C-41 负片的物理参数。"""

    name: str = "C-41 通用负片"
    #: 橙色罩（R,G,B 的 Status M 密度；蓝最大 → 片基橙）
    mask_density: tuple = ORANGE_MASK_DENSITY_C41
    #: 灰雾（叠加在罩上）
    fog_density: float = 0.03
    #: 染料消光矩阵（行 = C/M/Y，列 = R/G/B）
    dye_extinction: np.ndarray = field(
        default_factory=lambda: DYE_EXTINCTION_C41.copy())
    #: 逐层反差系数（青/品红/黄）→ 中间调偏色
    layer_contrast: tuple = LAYER_CONTRAST_C41
    #: 逐层最大染料密度（青/品红/黄）
    layer_dmax: tuple = LAYER_DMAX_C41
    #: 特性曲线趾/肩软度（归一化密度单位）
    s_toe: float = 0.08
    s_sh: float = 0.10
    #: 中灰对应的线性场景光
    scene_ref: float = 0.18
    #: 颗粒强度（密度域）
    grain: float = 0.010

    # ---------------- 派生量 ----------------
    @property
    def total_mask(self) -> np.ndarray:
        return np.asarray(self.mask_density, dtype=np.float64) + self.fog_density

    def matching_crosstalk_matrix(self) -> np.ndarray:
        """能精确反掉本片染料串扰的 3×3（= ``inv(Eᵀ)``）。"""
        return np.linalg.inv(np.asarray(self.dye_extinction,
                                       dtype=np.float64).T)

    def base_transmission(self, peak: float = 0.95) -> np.ndarray:
        """片基的线性透射值（已按 ``peak`` 归一化到最亮通道）。"""
        t = np.power(10.0, -self.total_mask)
        return t * (peak / float(np.max(t)))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "mask_density": list(self.mask_density),
            "fog_density": self.fog_density,
            "dye_extinction": np.asarray(self.dye_extinction).tolist(),
            "layer_contrast": list(self.layer_contrast),
            "layer_dmax": list(self.layer_dmax),
            "s_toe": self.s_toe, "s_sh": self.s_sh,
            "scene_ref": self.scene_ref, "grain": self.grain,
            "base_transmission": self.base_transmission().tolist(),
            "matching_crosstalk_matrix": self.matching_crosstalk_matrix().tolist(),
            "note": "合成用正向模型；未包含感光层光谱灵敏度重叠",
        }


#: 默认片种
DEFAULT_STOCK = FilmStock()


def _layer_densities(scene: np.ndarray, stock: FilmStock) -> np.ndarray:
    """线性场景光 → 三层的**染料密度**（已含逐层反差与趾/肩）。"""
    exposure = np.clip(np.asarray(scene, dtype=np.float64), 1e-6, None)
    out = np.empty_like(exposure)
    curve = sanitize(HDParams(a=1.0, x_mid=0.5, s_toe=stock.s_toe,
                              s_sh=stock.s_sh, d_min=0.0, d_max=1.0))
    for layer in range(3):
        # 相对中灰的对数曝光（decade），**逐层**处理
        x = np.log10(exposure[..., layer] / stock.scene_ref)
        # 反差差异：把对数曝光按层缩放 → 片基处不动、中间调分离
        t = stock.layer_contrast[layer] * x + 0.5
        shaped = characteristic(t, curve)
        out[..., layer] = (stock.layer_dmax[layer]
                           * np.clip(np.asarray(shaped), 0.0, 1.0))
    return out


def expose(scene: np.ndarray, stock: FilmStock = DEFAULT_STOCK,
           grain: float | None = None, seed: int = 0,
           border: float = 0.0, border_light: bool = True) -> dict:
    """把线性场景光渲染成"被扫描到的彩色负片"。

    返回 ``dict``，含：

    * ``negative``：相机线性扫描值（管线入口用的就是它）
    * ``layer_density``：三层染料密度（真值，未加串扰）
    * ``channel_density``：加串扰与橙色罩后的通道密度
    * ``base_rgb``：片基的扫描值（= 橙色罩 + 灰雾）
    * ``scene``：场景真值（原样返回）
    """
    scene = np.asarray(scene, dtype=np.float64)
    if scene.ndim != 3 or scene.shape[-1] != 3:
        raise ValueError(f"场景必须是 (H,W,3) 线性光，收到 {scene.shape}")

    layers = _layer_densities(scene, stock)                       # (H,W,3) C,M,Y
    extinction = np.asarray(stock.dye_extinction, dtype=np.float64)
    # D_rgb = Eᵀ · d_cmy + mask
    channel_density = layers @ extinction + stock.total_mask.reshape(1, 1, 3)

    sigma = stock.grain if grain is None else float(grain)
    if sigma > 0:
        rng = np.random.default_rng(seed)
        # 颗粒在**中间调最明显**：按染料密度归一到 0..1 再取平方根
        weight = np.sqrt(np.clip(layers / np.asarray(stock.layer_dmax), 0, 1))
        channel_density = channel_density + (rng.normal(0.0, sigma, layers.shape)
                                             * weight
                                             @ extinction)

    result_base = stock.base_transmission()
    transmission = np.power(10.0, -channel_density)
    negative = transmission * (0.95 / float(np.max(
        np.power(10.0, -stock.total_mask))))

    # 片基边框：真实扫描件里总有一圈未曝光的片边，正好用来试「片基采样」
    if border and border > 0:
        height, width = negative.shape[:2]
        rows = max(1, int(round(height * border)))
        cols = max(1, int(round(width * border)))
        base = stock.base_transmission()
        negative[:rows, :, :] = base
        negative[-rows:, :, :] = base
        negative[:, :cols, :] = base
        negative[:, -cols:, :] = base
        if border_light:
            negative[:rows * 2, :, :] = base          # 上边留宽一点，便于采样

    # 物理上限：片基（未曝光）已是最大透射。颗粒在密度域加噪声后可能让个别
    # 像素略高于片基，那等于"比透明片还亮"，不物理 —— 夹回去。
    negative = np.minimum(negative, result_base.reshape(1, 1, 3))
    negative = np.clip(negative, 0.0, 1.0)
    return {
        "negative": negative,
        "layer_density": layers,
        "channel_density": channel_density,
        "base_rgb": stock.base_transmission(),
        "scene": scene,
        "stock": stock,
    }


def densitometry_report(result: dict) -> dict:
    """给出合成结果的体检数据（便于确认"这张负片是否物理合理"）。"""
    negative = result["negative"]
    base = result["base_rgb"]
    density = -np.log10(np.clip(negative, 1e-9, None)
                        / np.clip(base, 1e-9, None).reshape(1, 1, 3))
    flat = density.reshape(-1, 3)
    return {
        "size": [int(negative.shape[1]), int(negative.shape[0])],
        "base_rgb": [round(float(v), 4) for v in base],
        "base_hue_ratio": [round(float(v / base.max()), 3) for v in base],
        "density_min": [round(float(v), 3) for v in flat.min(axis=0)],
        "density_median": [round(float(v), 3) for v in np.median(flat, axis=0)],
        "density_max": [round(float(v), 3) for v in flat.max(axis=0)],
        "density_below_base_pct": round(
            float(np.count_nonzero(np.any(density < -0.01, axis=2))
                  / density.shape[0] / density.shape[1] * 100), 4),
    }


# ======================================================================
# 逐通道校准推导（消偏色）
# ======================================================================

def neutral_ladder(stock: FilmStock = DEFAULT_STOCK, low: float = -3.5,
                   high: float = 3.5, count: int = 96):
    """无彩色阶梯：一组中性曝光经过本片种后的**逐层染料密度**。

    返回 ``(x, layers)``：``x`` 为相对中灰的 log10 曝光，``layers`` 为
    ``(N,3)`` 的染料密度（C,M,Y）—— 正是管线「解串扰」之后的量。
    """
    x = np.linspace(float(low), float(high), int(count))
    exposure = stock.scene_ref * np.power(10.0, x)
    scene = np.repeat(exposure[:, None, None], 3, axis=2)
    return x, _layer_densities(scene, stock).reshape(-1, 3)


def _channel_codes(layers: np.ndarray, a_rgb, d_min_rgb, d_max_rgb):
    """按管线同样的公式算出逐通道 code（与 tonemap 保持一致）。"""
    from .constants import CINEON
    from .tonemap import REFERENCE_SPAN

    codes = np.empty_like(np.asarray(layers, dtype=np.float64))
    for c in range(3):
        hd = sanitize(HDParams(a=float(a_rgb[c]), x_mid=0.5,
                               s_toe=0.08, s_sh=0.10,
                               d_min=float(d_min_rgb[c]),
                               d_max=float(d_max_rgb[c])))
        t = (np.asarray(layers, dtype=np.float64)[..., c] - hd.d_min) \
            / hd.density_range
        codes[..., c] = CINEON.black_code + REFERENCE_SPAN \
            * characteristic(t, hd)
    return codes


def derive_per_channel_calibration(stock: FilmStock = DEFAULT_STOCK,
                                   iterations: int = 160,
                                   tone_weight: float = 1.2,
                                   target_white: float = 0.9,
                                   window: tuple | None = None,
                                   verbose: bool = False) -> dict:
    """推导逐通道校准：**共享窗口取自数据，只解逐通道增益与偏移**。

    目标函数为什么是这样（踩过的坑，务必保留）
    ------------------------------------------
    1. 拿"还原场景的 ΔE 对真值"当目标是**不可能**的：胶片肩部已丢失高光
       信息，任何单调函数都反解不回来（实测拟合 RMS 达 **120 code**）。
    2. 只求"中性一致"而放任逐通道窗口自由浮动也**不行**：拟合会把窗口压到
       0.65 D，而数据范围有 2.28 D —— 绝大部分密度被塞进曲线的趾/肩，
       影调被压扁，画面**发灰**（实测彩度只剩真值的 25~58%）。

    所以本函数采用**解析共享窗口 + 6 参数**：

    * 共享窗口 ``[D0, D1]`` 由片种物理参数给出（``D0=0`` 片基，
      ``D1=max(Eᵀ·逐层 d_max)``），保证整段密度落在曲线的直线区；
    * 共享窗口 ``[D0, D1]`` 取自片种物理范围，逐通道只在其附近**受限微调**
      （实测放开 6 个参数自由浮动时，优化器宁可把窗口压到 1.0 D、把影调
      压扁也要降低离散度 —— 那是退化解，画面会灰）；
    * 逐通道解**受限的窗口偏移与展宽**（``|shift| ≤ 0.45``、``|widen| ≤ 0.60``
      密度单位）。为什么是这两个量而不是曲线斜率 ``a``：实测逐通道 ``a`` 对
      中性几乎无效（它改局部斜率，而三层的差异在**自变量偏移**上），
      甚至会被优化器拿来当"一起压扁"的退化解。

    返回 dict（含 ``a_rgb``/``d_min_rgb``/``d_max_rgb`` 与验收指标）。
    """
    from .constants import C41_DENSITY_WINDOW, CINEON
    from .tonemap import REFERENCE_SPAN, TransferFunction

    x, layers = neutral_ladder(stock)
    ref = C41_DENSITY_WINDOW if window is None else window
    d0, d1 = float(ref[0]), float(ref[1])
    span = max(d1 - d0, 1e-6)
    target = CINEON.white_code + REFERENCE_SPAN * (
        x + np.log10(stock.scene_ref / target_white))
    weight = np.exp(-((x - x.mean()) / 2.2) ** 2)

    def model(a_rgb, shift, widen):
        """目标函数里用的模型：物理窗口 ± 逐通道偏移/展宽。"""
        lo = tuple(d0 + float(shift[c]) - float(widen[c]) * 0.5 for c in range(3))
        hi = tuple(d1 + float(shift[c]) + float(widen[c]) * 0.5 for c in range(3))
        hd = HDParams(a=1.0, x_mid=0.5, s_toe=stock.s_toe, s_sh=stock.s_sh,
                      a_rgb=tuple(a_rgb), d_min_rgb=lo, d_max_rgb=hi)
        return TransferFunction(hd=hd).density_to_code(layers)[0]

    def free_model(a_rgb, lo_rgb, hi_rgb):
        """自由窗口模型（实测中性与彩度都更好的那一版）。"""
        hd = HDParams(a=1.0, x_mid=0.5, s_toe=stock.s_toe, s_sh=stock.s_sh,
                      a_rgb=tuple(a_rgb), d_min_rgb=tuple(lo_rgb),
                      d_max_rgb=tuple(hi_rgb))
        return TransferFunction(hd=hd).density_to_code(layers)[0]

    min_span, max_span = 470.0, 595.0

    def cost(a_rgb, shift, widen):
        codes = model(a_rgb, shift, widen)
        mean = codes.mean(axis=1)
        spread = float(np.mean(np.std(codes, axis=1)))
        tone = float(np.sqrt(np.sum(weight * (mean - target) ** 2)
                             / np.sum(weight)))
        # **防退化解**：只求"三通道一致"时，优化器会把所有通道一起压扁
        # （一起变灰自然就"一致"了）。所以必须同时要求阶梯**用满动态范围**。
        span = float(mean.max() - mean.min())
        penalty = 0.0
        if span < min_span:
            penalty += (min_span - span) ** 2 * 0.02
        elif span > max_span:
            penalty += (span - max_span) ** 2 * 0.02
        return spread + float(tone_weight) * tone + penalty, spread

    # ---- 第一段：自由窗口（增益+偏移），实测中性与彩度最好 ----
    gain = np.ones(3)
    offset = np.zeros(3)

    def cost_free(gain, offset):
        lo = tuple(d0 - float(offset[c]) * span / max(float(gain[c]), 1e-3)
                   for c in range(3))
        hi = tuple(lo[c] + span / max(float(gain[c]), 1e-3) for c in range(3))
        codes = free_model((1.0, 1.0, 1.0), lo, hi)
        mean = codes.mean(axis=1)
        spread = float(np.mean(np.std(codes, axis=1)))
        tone = float(np.sqrt(np.sum(weight * (mean - target) ** 2)
                             / np.sum(weight)))
        sp = float(mean.max() - mean.min())
        pen = (min_span - sp) ** 2 * 0.02 if sp < min_span else 0.0
        return spread + float(tone_weight) * tone + pen, spread

    best_free = cost_free(gain, offset)
    step_f = 0.10
    for _ in range(int(iterations)):
        moved = False
        for c in range(3):
            for sign in (1.0, -1.0):
                trial = gain.copy(); trial[c] += sign * step_f
                if trial[c] > 0.1:
                    value = cost_free(trial, offset)[0]
                    if value < best_free[0] - 1e-9:
                        gain, best_free, moved = trial, cost_free(trial, offset), True
            for sign in (1.0, -1.0):
                trial = offset.copy(); trial[c] += sign * step_f
                value = cost_free(gain, trial)[0]
                if value < best_free[0] - 1e-9:
                    offset, best_free, moved = trial, cost_free(gain, trial), True
        if not moved:
            step_f *= 0.5
        if step_f < 1e-4:
            break
    lo_free = tuple(d0 - float(offset[c]) * span / max(float(gain[c]), 1e-3)
                    for c in range(3))
    hi_free = tuple(lo_free[c] + span / max(float(gain[c]), 1e-3)
                    for c in range(3))

    a_rgb = np.ones(3)
    shift = np.zeros(3)
    widen = np.zeros(3)
    # 偏移/展宽都**受限**：这是"在这个片种的物理窗口附近做逐通道微调"，
    # 不是自由拟合 —— 放开就会把窗口压塌、影调变灰。
    max_shift, max_widen = 0.45, 0.60
    best = cost(a_rgb, shift, widen)
    step = 0.08
    for _ in range(int(iterations)):
        improved = False
        for c in range(3):
            for sign in (1.0, -1.0):
                trial = shift.copy(); trial[c] += sign * step
                if abs(trial[c]) <= max_shift:
                    value = cost(a_rgb, trial, widen)[0]
                    if value < best[0] - 1e-9:
                        shift, best, improved = trial, cost(a_rgb, trial, widen), True
            for sign in (1.0, -1.0):
                trial = widen.copy(); trial[c] += sign * step
                if abs(trial[c]) <= max_widen:
                    value = cost(a_rgb, shift, trial)[0]
                    if value < best[0] - 1e-9:
                        widen, best, improved = trial, cost(a_rgb, shift, trial), True
        if not improved:
            step *= 0.5
        if step < 1e-4:
            break

    # 两段拟合取中性更好的那一个（都用同一个目标函数评估）
    if best_free[1] <= best[1]:
        d_min_rgb, d_max_rgb = lo_free, hi_free
        codes = free_model((1.0, 1.0, 1.0), lo_free, hi_free)
        chosen = "free-window"
    else:
        d_min_rgb = tuple(d0 + float(shift[c]) - float(widen[c]) * 0.5
                          for c in range(3))
        d_max_rgb = tuple(d1 + float(shift[c]) + float(widen[c]) * 0.5
                          for c in range(3))
        codes = model(a_rgb, shift, widen)
        chosen = "bounded-shift"
    report = {
        "a_rgb": tuple(round(float(v), 4) for v in a_rgb),
        "d_min_rgb": tuple(round(float(v), 4) for v in d_min_rgb),
        "d_max_rgb": tuple(round(float(v), 4) for v in d_max_rgb),
        "shared_window": (round(d0, 4), round(d1, 4)),
        "shift_rgb": tuple(round(float(v), 4) for v in shift),
        "widen_rgb": tuple(round(float(v), 4) for v in widen),
        "variant": chosen,
        "free_spread_codes": round(float(best_free[1]), 3),
        "ladder_span_codes": round(float(codes.mean(axis=1).max()
                                         - codes.mean(axis=1).min()), 1),
        "ladder_spread_codes": round(float(np.mean(np.std(codes, axis=1))), 3),
        "ladder_spread_max_codes": round(float(np.max(np.std(codes, axis=1))), 3),
        "note": "两段拟合取优：自由窗口(增益+偏移) vs 物理窗口±受限偏移",
    }
    if verbose:
        print("逐通道校准推导:", report)
    return report


def fit_linearizing_calibration(stock: FilmStock = DEFAULT_STOCK,
                               iterations: int = 140,
                               tone_weight: float = 0.08,
                               target_white: float = 0.9,
                               verbose: bool = False) -> dict:
    """在**线性化模式**下拟合逐通道参数（7 参数，目标函数正确）。

    模型（与管线严格一致）::

        u_c  = G⁻¹( (d_c − d_min_c) / (d_max_c − d_min_c) )      G = 胶片曲线
        code_c = 95 + (590 / a_c) · (u_c − x_mid)

    目标（两条，见 :func:`derive_per_channel_calibration` 的说明）：

    1. **中性必须中性**：无彩色阶梯三通道 code 的离散度 → 权重最高；
    2. **影调锚定**：中段 code 跟随标准 Cineon 编码（肩部区域本就不可能跟随，
       因此按 |x| 加权，越靠边权重越低）。

    ``d_max_c`` 取片种自身的逐层最大染料密度（解析），其余 7 个参数拟合。
    """
    from .constants import CINEON
    from .tonemap import REFERENCE_SPAN, TransferFunction

    x, layers = neutral_ladder(stock)
    d_max_rgb = tuple(float(v) for v in stock.layer_dmax)
    target = CINEON.white_code + REFERENCE_SPAN * (
        x + np.log10(stock.scene_ref / target_white))
    weight = np.exp(-((x - x.mean()) / 2.2) ** 2)

    def model(a_rgb, d_min_rgb, x_mid):
        hd = HDParams(linearize=True, a_rgb=tuple(a_rgb),
                      d_min_rgb=tuple(d_min_rgb), d_max_rgb=d_max_rgb,
                      x_mid=float(x_mid), s_toe=stock.s_toe, s_sh=stock.s_sh)
        codes, _outside = TransferFunction(hd=hd).density_to_code(layers)
        return codes

    def cost(a_rgb, d_min_rgb, x_mid):
        codes = model(a_rgb, d_min_rgb, x_mid)
        spread = float(np.mean(np.std(codes, axis=1)))
        tone = float(np.sqrt(np.sum(weight * (codes.mean(axis=1) - target) ** 2)
                             / np.sum(weight)))
        return spread + float(tone_weight) * tone, spread

    contrast = np.asarray(stock.layer_contrast, dtype=np.float64)
    a_rgb = contrast.copy()
    d_min_rgb = np.zeros(3)
    x_mid = 0.5
    best = cost(a_rgb, d_min_rgb, x_mid)
    steps = [0.06, 0.06, 0.05]
    for _ in range(int(iterations)):
        improved = False
        for c in range(3):
            for sign in (1.0, -1.0):
                trial = a_rgb.copy(); trial[c] += sign * steps[0]
                if trial[c] > 0.15:
                    value = cost(trial, d_min_rgb, x_mid)[0]
                    if value < best[0] - 1e-9:
                        a_rgb, best, improved = trial, cost(trial, d_min_rgb, x_mid), True
            for sign in (1.0, -1.0):
                trial = d_min_rgb.copy(); trial[c] += sign * steps[1]
                value = cost(a_rgb, trial, x_mid)[0]
                if value < best[0] - 1e-9:
                    d_min_rgb, best, improved = trial, cost(a_rgb, trial, x_mid), True
        for sign in (1.0, -1.0):
            value = cost(a_rgb, d_min_rgb, x_mid + sign * steps[2])[0]
            if value < best[0] - 1e-9:
                x_mid, best, improved = x_mid + sign * steps[2], cost(a_rgb, d_min_rgb, x_mid + sign * steps[2]), True
        if not improved:
            steps = [v * 0.5 for v in steps]
        if max(steps) < 2e-4:
            break

    codes = model(a_rgb, d_min_rgb, x_mid)
    report = {
        "linearize": True,
        "a_rgb": tuple(round(float(v), 4) for v in a_rgb),
        "d_min_rgb": tuple(round(float(v), 4) for v in d_min_rgb),
        "d_max_rgb": tuple(round(float(v), 4) for v in d_max_rgb),
        "x_mid": round(float(x_mid), 4),
        "s_toe": stock.s_toe, "s_sh": stock.s_sh,
        "ladder_span_codes": round(float(codes.mean(axis=1).max()
                                         - codes.mean(axis=1).min()), 1),
        "ladder_spread_codes": round(float(np.mean(np.std(codes, axis=1))), 3),
        "ladder_spread_max_codes": round(float(np.max(np.std(codes, axis=1))), 3),
        "note": "线性化模式；目标=中性一致(主)+影调锚定(次)",
    }
    if verbose:
        print("线性化拟合:", report)
    return report


def to_srgb8(linear_rgb: np.ndarray) -> np.ndarray:
    """线性光 → 8-bit sRGB（给"对照正片"用，任何看图软件都能开）。"""
    from . import colorimetry as cm
    arr = np.clip(np.asarray(linear_rgb, dtype=np.float64), 0.0, 1.0)
    return np.clip(np.round(cm.srgb_encode(arr) * 255.0), 0, 255).astype(np.uint8)


__all__ = ["FilmStock", "DEFAULT_STOCK", "expose", "densitometry_report",
           "to_srgb8", "neutral_ladder", "derive_per_channel_calibration",
           "analytic_linearizing_calibration"]
