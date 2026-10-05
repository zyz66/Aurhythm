"""显示变换与分析读数：导出空间 → 屏幕 sRGB（所见即所得）。

为什么必须显式定义
------------------
旧实现 ``process_for_preview`` 用固定 Reinhard 色调映射，与导出的对数域
**没有任何关系**，于是「看到的」和「导出的」不是同一个东西。在校准工具
里这是致命的：你必须能确认导出的编码就是你看到的那张图。

本模块把显示链路写成显式的、可复现的函数::

    Cineon code ──(还原曝光)──> 线性曝光 E ──(显示曝光)──> sRGB 编码 ──> uint8
    LogC3       ──(ARRI 逆函数)──> 线性曝光 E ──(同上)

另外提供示波器/直方图所需的**数据**（不画图，只算数组），
让 UI 与测试都能复用同一份数值。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import colorimetry as cm
from .constants import CINEON, LOGC3

#: ARRI 曝光伪色分区（log10 曝光，相对 18% 中灰）：(下界, RGB)
#: 依据 ARRI 曝光伪色规范的分区思路，用离散色带指示曝光水平。
FALSE_COLOR_ZONES = (
    (-4.0, (0.20, 0.00, 0.35)),   # 深紫：严重欠曝
    (-2.0, (0.00, 0.00, 0.60)),   # 蓝：欠曝
    (-1.0, (0.00, 0.45, 0.00)),   # 绿：略欠
    (-0.33, (0.55, 0.55, 0.55)),  # 灰：正常
    (0.33, (0.90, 0.90, 0.00)),   # 黄：略过
    (1.0, (1.00, 0.45, 0.00)),    # 橙：过曝
    (2.0, (1.00, 0.00, 0.00)),    # 红：严重过曝
)


def bt709_encode(linear):
    """线性光 → BT.709 OETF 编码（Rec.709 视频值）。"""
    x = np.asarray(linear, dtype=np.float64)
    low = 4.5 * x
    high = 1.099 * np.power(np.maximum(x, 0.0), 0.45) - 0.099
    return np.where(x < 0.018, low, high)


def bt709_decode(encoded):
    """BT.709 OETF 编码 → 线性光。"""
    v = np.asarray(encoded, dtype=np.float64)
    low = v / 4.5
    high = np.power((np.maximum(v, 0.0) + 0.099) / 1.099, 1.0 / 0.45)
    return np.where(v < 0.081, low, high)


def display_response(values, space: str) -> np.ndarray:
    """把某个空间的**显示域**编码值转成屏幕 sRGB 值（0..1）。

    只对显示域空间有意义（Rec.709 / sRGB / 线性）；
    对数域请走 :func:`code_to_display`（需要先解出曝光）。
    """
    arr = np.asarray(values, dtype=np.float64)
    if space == "rec709":
        # 监视器是 sRGB：先按 BT.709 解出线性，再按 sRGB 编码。
        # （两者基色相同，差的是 OETF —— 暗部会差几个码值，直接当 sRGB
        #  显示会略微压暗阴影。）
        return cm.srgb_encode(np.maximum(bt709_decode(arr), 0.0))
    if space == "srgb":
        return np.clip(arr, 0.0, 1.0)
    if space == "linear":
        return cm.srgb_encode(np.maximum(arr, 0.0))
    raise ValueError(f"display_response 不支持的空间: {space}")


@dataclass(frozen=True)
class ViewSettings:
    """预览显示设置（与导出无关，只影响屏幕上看到的样子）。"""

    exposure_ev: float = 0.0
    gamma: float = 1.0
    mode: str = "video"        # 'video' | 'linear' | 'false_color'
    clip: bool = True
    #: 解码码率的倍率（1.0 = 按片种 γ 自动算出的码率）。
    #: 这是一个**观看变换**参数：解码码率越大，同样的码值被解成越亮的曝光，
    #: 画面被"展开"得越多。用标准 500 去解会发灰，按片种 γ 解开才正常。
    tone_scale: float = 1.0

    def to_dict(self) -> dict:
        return {"exposure_ev": self.exposure_ev, "gamma": self.gamma,
                "mode": self.mode, "clip": self.clip,
                "tone_scale": self.tone_scale}


def to_exposure(values, colorspace: str = "cineon",
                codes_per_decade: float | None = None) -> np.ndarray:
    """导出空间的值 → 线性曝光（E，参考白 90% 白卡 = 0.90）。

    ``codes_per_decade`` 默认用 Cineon 标准的 500。但对**已知 γ 的片种**，
    正确码率是 ``REFERENCE_SPAN·γ/窗口跨度`` —— 用标准码率去解一条 γ=0.65
    的负片，编码时压进去的密度没有被展开，画面会被压扁**发灰**（实测彩度
    只剩真值的 55%；按片种 γ 解开后恢复到 ~89%）。
    """
    arr = np.asarray(values, dtype=np.float64)
    if colorspace == "logc3":
        return LOGC3.decode(arr)
    code = arr * CINEON.max_code
    if codes_per_decade is None:
        return CINEON.exposure(CINEON.log_e(code))
    rate = max(float(codes_per_decade), 1e-6)
    log_e = (code - CINEON.black_code) / rate
    return CINEON.exposure(log_e)


def exposure_to_code(exposure, colorspace: str = "cineon") -> np.ndarray:
    """线性曝光 → 导出空间的值（``to_exposure`` 的逆）。"""
    arr = np.asarray(exposure, dtype=np.float64)
    if colorspace == "logc3":
        return LOGC3.encode(arr)
    code = CINEON.code(CINEON.log_e_from_exposure(arr))
    return code / CINEON.max_code


def false_color(values, colorspace: str = "cineon") -> np.ndarray:
    """ARRI 风格曝光伪色（返回 [0,1] 的 RGB，供显示）。"""
    exposure = np.maximum(to_exposure(values, colorspace, codes_per_decade=codes_per_decade), 1e-9)
    log_e = np.log10(exposure / 0.18)          # 相对 18% 中灰
    out = np.zeros(log_e.shape + (3,), dtype=np.float64)
    bounds = np.array([z[0] for z in FALSE_COLOR_ZONES], dtype=np.float64)
    idx = np.clip(np.searchsorted(bounds, log_e, side="right") - 1,
                  0, len(FALSE_COLOR_ZONES) - 1)
    for i, (_lo, rgb) in enumerate(FALSE_COLOR_ZONES):
        mask = idx == i
        if np.any(mask):
            out[mask] = rgb
    return out


def code_to_display(values, colorspace: str = "cineon",
                    view: ViewSettings | None = None,
                    codes_per_decade: float | None = None) -> np.ndarray:
    """导出空间的值 → 显示用 uint8 (H,W,3)（所见即所得）。

    ``colorspace`` 为 ``rec709`` / ``srgb`` / ``linear`` 时按**显示域**
    处理（不再解 log），因为那正是「套了色彩还原 LUT」之后的输出。
    """
    view = view or ViewSettings()
    if colorspace in ("rec709", "srgb", "linear"):
        rgb = display_response(values, colorspace)
        if view.clip:
            rgb = np.clip(rgb, 0.0, 1.0)
        return np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
    if view.mode == "false_color":
        rgb = false_color(values, colorspace)
    else:
        exposure = to_exposure(values, colorspace, codes_per_decade=codes_per_decade)
        if view.mode == "linear":
            rgb = exposure
        else:                                   # 'video'（默认）
            rgb = exposure * (2.0 ** view.exposure_ev)
            rgb = cm.srgb_encode(rgb)
        if view.mode == "video" and view.gamma != 1.0:
            rgb = np.power(np.maximum(rgb, 0.0), 1.0 / view.gamma)
    if view.clip:
        rgb = np.clip(rgb, 0.0, 1.0)
    return np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)


# ======================================================================
# 示波器 / 直方图数据（只算数组，由 UI 绘制）
# ======================================================================

def histogram(rgb_uint8, bins: int = 256) -> np.ndarray:
    """返回 (bins, 3) 的直方图计数（R/G/B）。"""
    arr = np.asarray(rgb_uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    out = np.zeros((bins, 3), dtype=np.int64)
    for c in range(3):
        counts, _ = np.histogram(arr[..., c], bins=bins, range=(0, 255))
        out[:, c] = counts
    return out


def waveform(rgb_uint8, columns: int = 256) -> np.ndarray:
    """波形图数据：返回 (columns, 256) 的亮度累积矩阵。"""
    arr = np.asarray(rgb_uint8, dtype=np.float64)
    if arr.ndim == 3:
        arr = cm.luminance(arr) / 255.0
    h, w = arr.shape
    xs = np.clip((np.arange(w) * columns // max(w, 1)), 0, columns - 1)
    ys = np.clip((arr * 255.0).astype(np.int64), 0, 255)
    out = np.zeros((columns, 256), dtype=np.int64)
    np.add.at(out, (xs[None, :].repeat(h, axis=0).ravel(), ys.ravel()), 1)
    return out


def parade(rgb_uint8) -> np.ndarray:
    """RGB parade：把三通道并排（返回 uint8 图像）。"""
    arr = np.asarray(rgb_uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    h, w, _ = arr.shape
    out = np.zeros((h, w * 3), dtype=np.uint8)
    for c in range(3):
        out[:, c * w:(c + 1) * w] = arr[..., c]
    return out


def vectorscope(rgb_uint8, size: int = 256) -> np.ndarray:
    """矢量示波器数据：返回 (size, size) 的色度累积矩阵（Cb/Cr 平面）。"""
    arr = np.asarray(rgb_uint8, dtype=np.float64)
    if arr.ndim == 2:
        return np.zeros((size, size), dtype=np.int64)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b
    x = np.clip((cr / 255.0 + 0.5) * (size - 1), 0, size - 1).astype(np.int64)
    y = np.clip((0.5 - cb / 255.0) * (size - 1), 0, size - 1).astype(np.int64)
    out = np.zeros((size, size), dtype=np.int64)
    np.add.at(out, (y.ravel(), x.ravel()), 1)
    return out


__all__ = [
    "ViewSettings", "to_exposure", "exposure_to_code", "false_color",
    "bt709_encode", "bt709_decode", "display_response",
    "code_to_display", "histogram", "waveform", "parade", "vectorscope",
    "FALSE_COLOR_ZONES",
]
