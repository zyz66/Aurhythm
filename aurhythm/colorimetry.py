"""色彩学：矩阵链、色适应、sRGB、Lab、ΔE76 / ΔE2000。

只依赖 numpy 与标准库。所有矩阵都从色度坐标**运行时推导**（而不是抄常数），
这样白点与基色的一致性由构造保证；测试再与已发表的参考矩阵对拍，
既能抓住转写错误，也不会把错误固化成「真理」。
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

# ======================================================================
# 白点 / 基色 / 矩阵
# ======================================================================

#: Bradford 锥响应矩阵
MATRIX_BRADFORD = np.array([
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
], dtype=np.float64)


def xy_to_xyz(x: float, y: float) -> np.ndarray:
    """CIE xy 色度 → 归一化 XYZ（Y = 1）。"""
    return np.array([x / y, 1.0, (1.0 - x - y) / y], dtype=np.float64)


#: CIE 标准照明体 XYZ —— **由精确 xy 现算**，作为全模块唯一自洽来源。
#:
#: 为什么不抄常见表格值 (0.95047,1,1.08883)/(0.96422,1,0.82521)：
#: 那些圆整值并非由 (0.3127,0.3290)/(0.34567,0.35850) 导出（Z 分量差
#: 约 2.3e-4），混用会导致 sRGB 矩阵与白点不自洽。现算的好处是
#: ``srgb_to_xyz([1,1,1]) == WHITE_D65`` 精确成立，且 sRGB 矩阵与已发表
#: 矩阵吻合到 1e-16。代价：Bradford 矩阵与「用圆整白点算出的」已发表
#: 版本相差 ≤3.3e-4（远小于 8-bit 一个码值的影响，测试中已记录）。
WHITE_D65 = xy_to_xyz(0.3127, 0.3290)
WHITE_D50 = xy_to_xyz(0.34567, 0.35850)

#: 常见圆整表格值，仅用于测试对照
WHITE_D65_TABULATED = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
WHITE_D50_TABULATED = np.array([0.96422, 1.0, 0.82521], dtype=np.float64)


def rgb_to_xyz_matrix(primaries_xy, white_xyz: np.ndarray) -> np.ndarray:
    """由基色色度坐标与白点构造 RGB→XYZ 矩阵（列为各基色 XYZ）。"""
    primaries = [xy_to_xyz(x, y) for x, y in primaries_xy]
    p = np.array(primaries, dtype=np.float64).T          # 3x3，列 = 基色 XYZ
    s = np.linalg.solve(p, np.asarray(white_xyz, dtype=np.float64))
    return p * s


#: sRGB (IEC 61966-2-1) 基色
SRGB_PRIMARIES = ((0.64, 0.33), (0.30, 0.60), (0.15, 0.06))
#: Rec.709 与 sRGB 同基色
REC709_PRIMARIES = SRGB_PRIMARIES

MATRIX_SRGB_TO_XYZ_D65 = rgb_to_xyz_matrix(SRGB_PRIMARIES, WHITE_D65)
MATRIX_XYZ_D65_TO_SRGB = np.linalg.inv(MATRIX_SRGB_TO_XYZ_D65)


@lru_cache(maxsize=64)
def _bradford_cat_cached(src: tuple, dst: tuple) -> np.ndarray:
    src_cone = MATRIX_BRADFORD @ np.array(src, dtype=np.float64)
    dst_cone = MATRIX_BRADFORD @ np.array(dst, dtype=np.float64)
    return np.linalg.inv(MATRIX_BRADFORD) @ np.diag(dst_cone / src_cone) @ MATRIX_BRADFORD


def bradford_cat(src_white, dst_white) -> np.ndarray:
    """Bradford 色适应矩阵：XYZ(src 白点) → XYZ(dst 白点)。"""
    src = tuple(np.asarray(src_white, dtype=np.float64).ravel())
    dst = tuple(np.asarray(dst_white, dtype=np.float64).ravel())
    return _bradford_cat_cached(src, dst)


#: 常用色适应矩阵（D50 PCS ↔ D65 显示）
CAT_D50_TO_D65 = bradford_cat(WHITE_D50, WHITE_D65)
CAT_D65_TO_D50 = bradford_cat(WHITE_D65, WHITE_D50)


# ======================================================================
# 应用辅助
# ======================================================================

def _apply_matrix(matrix: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """把 3x3 矩阵应用到 (H,W,3) 或 (N,3) 数组，**不裁剪**。"""
    arr = np.asarray(pixels, dtype=np.float64)
    flat = arr.reshape(-1, 3)
    out = flat @ np.asarray(matrix, dtype=np.float64).T
    return out.reshape(arr.shape)


def srgb_to_xyz(rgb, white: str = "D65") -> np.ndarray:
    """线性 sRGB → XYZ（默认 D65；``white='D50'`` 时再做 Bradford 适应）。"""
    xyz = _apply_matrix(MATRIX_SRGB_TO_XYZ_D65, rgb)
    if white.upper() == "D50":
        xyz = _apply_matrix(CAT_D65_TO_D50, xyz)
    return xyz


def xyz_to_srgb(xyz, white: str = "D65") -> np.ndarray:
    """XYZ → 线性 sRGB（可越界，不裁剪）。"""
    arr = np.asarray(xyz, dtype=np.float64)
    if white.upper() == "D50":
        arr = _apply_matrix(CAT_D50_TO_D65, arr)
    return _apply_matrix(MATRIX_XYZ_D65_TO_SRGB, arr)


# ======================================================================
# sRGB 传递函数
# ======================================================================

SRGB_LINEAR_THRESHOLD = 0.0031308


def srgb_encode(linear):
    """线性光 → sRGB 编码（IEC 61966-2-1 精确分段，不裁剪）。"""
    x = np.asarray(linear, dtype=np.float64)
    low = x * 12.92
    high = 1.055 * np.power(np.maximum(x, 0.0), 1.0 / 2.4) - 0.055
    return np.where(x <= SRGB_LINEAR_THRESHOLD, low, high)


def srgb_decode(encoded):
    """sRGB 编码 → 线性光（精确分段，不裁剪）。"""
    x = np.asarray(encoded, dtype=np.float64)
    low = x / 12.92
    high = np.power((np.maximum(x, 0.0) + 0.055) / 1.055, 2.4)
    return np.where(x <= 0.04045, low, high)


# ======================================================================
# Lab / LCh
# ======================================================================

LAB_EPSILON = 216.0 / 24389.0
LAB_KAPPA = 24389.0 / 27.0


def _lab_f(t):
    t = np.asarray(t, dtype=np.float64)
    return np.where(t > LAB_EPSILON, np.cbrt(t), (LAB_KAPPA * t + 16.0) / 116.0)


def xyz_to_lab(xyz, white=WHITE_D65) -> np.ndarray:
    """XYZ → CIELAB（``white`` 必须与 XYZ 的参考白一致）。"""
    arr = np.asarray(xyz, dtype=np.float64)
    ref = np.asarray(white, dtype=np.float64)
    ratio = arr / ref.reshape((1,) * (arr.ndim - 1) + (3,))
    fx, fy, fz = _lab_f(ratio[..., 0]), _lab_f(ratio[..., 1]), _lab_f(ratio[..., 2])
    return np.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], axis=-1)


def lab_to_xyz(lab, white=WHITE_D65) -> np.ndarray:
    """CIELAB → XYZ（``xyz_to_lab`` 的逆）。"""
    arr = np.asarray(lab, dtype=np.float64)
    L, a, b = arr[..., 0], arr[..., 1], arr[..., 2]
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0

    def inv_f(f):
        f = np.asarray(f, dtype=np.float64)
        f3 = f ** 3
        return np.where(f3 > LAB_EPSILON, f3, (116.0 * f - 16.0) / LAB_KAPPA)

    ref = np.asarray(white, dtype=np.float64)
    return np.stack([inv_f(fx), inv_f(fy), inv_f(fz)], axis=-1) * ref.reshape(
        (1,) * (arr.ndim - 1) + (3,)
    )


def rgb_to_lab_srgb(rgb, white: str = "D65") -> np.ndarray:
    """线性 sRGB → Lab。默认 D65（与 sRGB 基色一致）；``'D50'`` 先做色适应。"""
    w = WHITE_D50 if white.upper() == "D50" else WHITE_D65
    return xyz_to_lab(srgb_to_xyz(rgb, white=white), white=w)


def lab_to_lch(lab) -> np.ndarray:
    """Lab → LCh（h 单位：度）。"""
    arr = np.asarray(lab, dtype=np.float64)
    L, a, b = arr[..., 0], arr[..., 1], arr[..., 2]
    C = np.hypot(a, b)
    h = np.degrees(np.arctan2(b, a)) % 360.0
    return np.stack([L, C, h], axis=-1)


# ======================================================================
# 色差
# ======================================================================

def delta_e_76(lab1, lab2) -> np.ndarray:
    """CIE76 色差（欧氏距离）。"""
    a = np.asarray(lab1, dtype=np.float64)
    b = np.asarray(lab2, dtype=np.float64)
    return np.sqrt(np.sum((a - b) ** 2, axis=-1))


def delta_e_2000(lab1, lab2, k_l: float = 1.0, k_c: float = 1.0, k_h: float = 1.0) -> np.ndarray:
    """CIEDE2000 色差（Sharma / CIEDE2000 标准实现，向量化）。

    实现严格遵循 Sharma, Wu & Dalal (2005) 的公式（含 a' 缩放、h' 平均
    的环形处理与 T 项），用于验证的标准测试向量见
    ``tests/test_colorimetry.py``。
    """
    a1 = np.asarray(lab1, dtype=np.float64)
    a2 = np.asarray(lab2, dtype=np.float64)
    L1, A1, B1 = a1[..., 0], a1[..., 1], a1[..., 2]
    L2, A2, B2 = a2[..., 0], a2[..., 1], a2[..., 2]

    C1 = np.hypot(A1, B1)
    C2 = np.hypot(A2, B2)
    C_bar = 0.5 * (C1 + C2)
    C_bar7 = C_bar ** 7
    G = 0.5 * (1.0 - np.sqrt(C_bar7 / (C_bar7 + 25.0 ** 7)))

    A1p = (1.0 + G) * A1
    A2p = (1.0 + G) * A2
    C1p = np.hypot(A1p, B1)
    C2p = np.hypot(A2p, B2)

    H1p = np.degrees(np.arctan2(B1, A1p)) % 360.0
    H2p = np.degrees(np.arctan2(B2, A2p)) % 360.0
    # atan2(0,0) 定义为 0
    H1p = np.where((np.abs(A1p) + np.abs(B1)) == 0.0, 0.0, H1p)
    H2p = np.where((np.abs(A2p) + np.abs(B2)) == 0.0, 0.0, H2p)

    dLp = L2 - L1
    dCp = C2p - C1p

    zero_chroma = (C1p * C2p) == 0.0
    dh_angle = H2p - H1p
    dh_angle = np.where(dh_angle > 180.0, dh_angle - 360.0, dh_angle)
    dh_angle = np.where(dh_angle < -180.0, dh_angle + 360.0, dh_angle)
    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh_angle / 2.0))
    dHp = np.where(zero_chroma, 0.0, dHp)

    Lp_bar = 0.5 * (L1 + L2)
    Cp_bar = 0.5 * (C1p + C2p)

    h_sum = H1p + H2p
    h_diff = np.abs(H1p - H2p)
    Hp_bar = np.where(
        zero_chroma, h_sum,
        np.where(h_diff <= 180.0, 0.5 * h_sum,
                 np.where(h_sum < 360.0, 0.5 * (h_sum + 360.0), 0.5 * (h_sum - 360.0))),
    )

    T = (1.0
         - 0.17 * np.cos(np.radians(Hp_bar - 30.0))
         + 0.24 * np.cos(np.radians(2.0 * Hp_bar))
         + 0.32 * np.cos(np.radians(3.0 * Hp_bar + 6.0))
         - 0.20 * np.cos(np.radians(4.0 * Hp_bar - 63.0)))

    d_theta = 30.0 * np.exp(-(((Hp_bar - 275.0) / 25.0) ** 2))
    Cp_bar7 = Cp_bar ** 7
    R_C = 2.0 * np.sqrt(Cp_bar7 / (Cp_bar7 + 25.0 ** 7))
    S_L = 1.0 + (0.015 * (Lp_bar - 50.0) ** 2) / np.sqrt(20.0 + (Lp_bar - 50.0) ** 2)
    S_C = 1.0 + 0.045 * Cp_bar
    S_H = 1.0 + 0.015 * Cp_bar * T
    R_T = -np.sin(np.radians(2.0 * d_theta)) * R_C

    term_L = dLp / (k_l * S_L)
    term_C = dCp / (k_c * S_C)
    term_H = dHp / (k_h * S_H)

    return np.sqrt(term_L ** 2 + term_C ** 2 + term_H ** 2 + R_T * term_C * term_H)


#: 兼容旧命名
delta_e = delta_e_2000


def lab_string(lab) -> str:
    """格式化单个 Lab 值。"""
    v = np.asarray(lab, dtype=np.float64).ravel()
    return f"L*={v[0]:6.2f} a*={v[1]:6.2f} b*={v[2]:6.2f}"


def luminance(rgb, coefficients=(0.2126, 0.7152, 0.0722)):
    """线性 sRGB 亮度（Rec.709 系数）。"""
    arr = np.asarray(rgb, dtype=np.float64)
    return arr @ np.asarray(coefficients, dtype=np.float64)


__all__ = [
    "MATRIX_BRADFORD", "WHITE_D50", "WHITE_D65",
    "MATRIX_SRGB_TO_XYZ_D65", "MATRIX_XYZ_D65_TO_SRGB",
    "CAT_D50_TO_D65", "CAT_D65_TO_D50",
    "bradford_cat", "rgb_to_xyz_matrix", "xy_to_xyz",
    "srgb_to_xyz", "xyz_to_srgb", "srgb_encode", "srgb_decode",
    "xyz_to_lab", "lab_to_xyz", "rgb_to_lab_srgb", "lab_to_lch",
    "delta_e_76", "delta_e_2000", "delta_e", "lab_string", "luminance",
    "LAB_EPSILON", "LAB_KAPPA", "SRGB_LINEAR_THRESHOLD",
]
