"""合成测试素材：没有 RAW 也能跑通整条管线。

两类素材：

* :func:`make_negative` —— 一张「彩色负片」：片基最亮、场景密度随位置变化，
  可注入逐通道偏色与噪声。用于验证密度链、片基检测、中性化、曲线。
* :func:`make_chart` —— 把 24 色卡按给定四角（支持透视）渲染进画面，
  可注入已知的 3x3 相机变换。用于验证色卡检测、采样与 ΔE2000 报告。

CLI（``aurhythm synth``）与测试共用这些生成器。
"""

from __future__ import annotations

import numpy as np

from . import colorchecker as cc

#: 合成素材的种类
KINDS = ("negative", "chart")


#: 可合成的目标（CLI 与参数校验共用，避免两处漂移）
KINDS = ("negative", "chart", "scene")


def make_negative(height: int = 512, width: int = 768, *,
                  base=(0.90, 0.84, 0.72), cast=(1.0, 1.0, 1.0),
                  density_gamma=(1.0, 1.0, 1.0),
                  noise: float = 0.002, base_rows: int = 40,
                  min_transmission: float = 0.004, seed: int = 0):
    """合成彩色负片。

    返回 ``(image, base_rgb)``；``image`` 是**相机线性**空间，
    顶部 ``base_rows`` 行是未曝光的片基（真实片基 = ``base * cast``）。

    两种"偏色"模型，物理含义不同：

    * ``cast``：**乘性**整体偏色。片基与画面被同等缩放，因此在密度域里
      **必然对消**（这正是"以片基为参照"要消掉的东西，例如橙色 mask）。
      它只会改变片基颜色，不会制造需要中性化的中间调偏色。
    * ``density_gamma``：**逐通道密度反差**。``v_c = base_c · ramp**gamma_c``，
      片基处（ramp=1）为 0、中间调按通道分离 —— 这才是真实彩色负片
      （三层染料反差不同）造成的偏色，也是「密度域中性化」真正要修的东西。
    """
    rng = np.random.default_rng(seed)
    ramp = np.linspace(1.0, min_transmission, width)[None, :, None]
    ramp = np.repeat(ramp, 3, axis=2)                     # (1, W, 3)
    gamma = np.asarray(density_gamma, dtype=np.float64).reshape(1, 1, 3)
    ramp = np.power(ramp, gamma)                          # 逐通道密度反差
    image = np.repeat(ramp, height, axis=0)
    image = image * np.asarray(base, dtype=np.float64).reshape(1, 1, 3)
    image = image * np.asarray(cast, dtype=np.float64).reshape(1, 1, 3)
    if noise:
        image = image + rng.normal(0.0, noise, image.shape)
    base_rows = max(0, min(int(base_rows), height))
    if base_rows:
        film_base = (np.asarray(base, dtype=np.float64)
                     * np.asarray(cast, dtype=np.float64)).reshape(1, 1, 3)
        image[:base_rows, :, :] = film_base
    return np.clip(image, 0.0, None).astype(np.float32), np.asarray(base, np.float64)


def make_chart(height: int = 512, width: int = 768, *, corners=None,
               colors=None, camera_matrix=None, background: float = 0.85,
               grid: float = 0.03, grid_frac: float = 0.06,
               margin: float = 0.12, noise: float = 0.0, seed: int = 0):
    """把 24 色卡渲染进画面（支持透视/旋转）。

    ``camera_matrix`` 为相机→标准空间的 3x3；给定时色块会被它「污染」，
    于是检测+求解应当恢复其逆矩阵。
    """
    if colors is None:
        colors = cc.target_linear_srgb()
    colors = np.asarray(colors, dtype=np.float64)
    if camera_matrix is not None:
        colors = colors @ np.asarray(camera_matrix, dtype=np.float64).T

    if corners is None:
        m = margin
        corners = np.array([
            [m * width, m * height],
            [(1 - m) * width, m * height * 0.95],
            [(1 - m * 0.9) * width, (1 - m) * height],
            [m * width * 1.1, (1 - m * 0.95) * height],
        ], dtype=np.float64)
    corners = np.asarray(corners, dtype=np.float64)

    image = np.full((height, width, 3), background, dtype=np.float64)
    h_inv = np.linalg.inv(cc.homography_from_unit_square(corners))
    ys, xs = np.mgrid[0:height, 0:width]
    uv = cc.apply_homography(h_inv, np.stack([xs.ravel(), ys.ravel()], axis=-1))
    u, v = uv[:, 0], uv[:, 1]
    inside = (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
    if np.any(inside):
        col = np.clip((u * cc.CHART_COLUMNS).astype(np.int64), 0,
                      cc.CHART_COLUMNS - 1)
        row = np.clip((v * cc.CHART_ROWS).astype(np.int64), 0, cc.CHART_ROWS - 1)
        fu = u * cc.CHART_COLUMNS - col
        fv = v * cc.CHART_ROWS - row
        is_patch = ((fu > grid_frac) & (fu < 1 - grid_frac)
                    & (fv > grid_frac) & (fv < 1 - grid_frac))
        flat = image.reshape(-1, 3)
        index = row * cc.CHART_COLUMNS + col
        on_patch = inside & is_patch
        flat[on_patch] = colors[index[on_patch]]
        flat[inside & ~is_patch] = grid
    if noise:
        rng = np.random.default_rng(seed)
        image = image + rng.normal(0.0, noise, image.shape)
    return np.clip(image, 0.0, None).astype(np.float32)


def default_chart_corners(width: int, height: int, margin: float = 0.12):
    """给出一组缺省的四角（左上/右上/右下/左下）。"""
    return np.array([
        [margin * width, margin * height],
        [(1 - margin) * width, margin * height],
        [(1 - margin) * width, (1 - margin) * height],
        [margin * width, (1 - margin) * height],
    ], dtype=np.float64)


def make(kind: str = "negative", width: int = 768, height: int = 512, **kwargs):
    """按 ``kind`` 生成素材。"""
    if kind == "negative":
        return make_negative(height=height, width=width, **kwargs)
    if kind == "chart":
        return make_chart(height=height, width=width, **kwargs)
    raise ValueError(f"未知素材种类 {kind!r}，可选 {KINDS}")


__all__ = ["KINDS", "make_negative", "make_chart", "make",
           "default_chart_corners"]


# ======================================================================
# 程序化"有画面"的场景（供合成真实感测试负片）
# ======================================================================

def smooth_noise(width: int, height: int, scale: int, seed: int) -> np.ndarray:
    """用 PIL 双三次放大的低分辨率随机网格做平滑噪声（0..1）。"""
    from PIL import Image

    rng = np.random.default_rng(seed)
    sw, sh = max(2, int(width // max(scale, 1))), max(2, int(height // max(scale, 1)))
    grid = (rng.random((sh, sw)) * 255.0).astype(np.uint8)
    image = Image.fromarray(grid, mode="L").resize((width, height),
                                                   Image.BICUBIC)
    return np.asarray(image, dtype=np.float64) / 255.0


def fbm(width: int, height: int, seed: int,
        octaves=((96, 0.50), (48, 0.26), (24, 0.14), (12, 0.10))) -> np.ndarray:
    """多倍频平滑噪声（0..1）。"""
    total = np.zeros((height, width))
    weight = 0.0
    for index, (scale, amp) in enumerate(octaves):
        total += amp * smooth_noise(width, height, scale, seed + index * 17)
        weight += amp
    return total / max(weight, 1e-9)


def _paint_rect(target, x0, y0, x1, y1, color):
    x0, x1 = max(0, int(x0)), min(target.shape[1], int(x1))
    y0, y1 = max(0, int(y0)), min(target.shape[0], int(y1))
    if x1 > x0 and y1 > y0:
        target[y0:y1, x0:x1] = np.asarray(color, dtype=np.float64).reshape(1, 1, 3)


def make_scene(width: int = 1200, height: int = 800, seed: int = 7,
               with_card: bool = True) -> np.ndarray:
    """程序化生成一张"有画面"的线性场景光图（风景 + 场景中的测试卡）。

    包含：天空渐变与云、太阳高光（可到 4.0，用来试胶片的肩部）、
    远山（大气透视）、带纹理的地面、前景深色植被（试趾部）、
    以及立在场景中的**测试卡**（24 色块 + 21 级灰阶 + 分辨率元件）。
    """
    from . import colorimetry as cm
    from .constants import COLORCHECKER_24_SRGB

    yy, xx = np.mgrid[0:height, 0:width]
    u = xx / float(width)
    v = yy / float(height)
    aspect = width / float(height)
    scene = np.zeros((height, width, 3), dtype=np.float64)
    horizon = 0.47

    # ---------- 天空 ----------
    t = np.clip(v / horizon, 0.0, 1.0)[..., None]
    sky_top = np.array([0.13, 0.25, 0.56])
    sky_horizon = np.array([0.62, 0.74, 0.86])
    sky = sky_top * (1 - t) + sky_horizon * t

    clouds = fbm(width, height, seed)
    cloud_band = np.clip((clouds - 0.48) * 3.2, 0.0, 1.0)
    cloud_fade = np.clip(1.0 - (v / horizon) ** 1.5, 0.0, 1.0)
    cloud_amt = cloud_band * cloud_fade
    sky = sky + cloud_amt[..., None] * (np.array([1.05, 1.02, 1.00]) - sky)

    # 太阳 + 辉光（高光：故意超过 1.0 以测试肩部压缩）
    du = (u - 0.80) * aspect
    dv = v - 0.15
    dist = np.hypot(du, dv)
    glow = np.exp(-(dist / 0.075) ** 2) * 1.1 + np.exp(-(dist / 0.22) ** 2) * 0.35
    sky = sky + glow[..., None] * np.array([1.00, 0.86, 0.62])
    sky[dist < 0.016] = np.array([4.0, 3.7, 3.2])
    # 天空是底层：后面的远山/地面/植被按 mask 覆盖上去
    scene[...] = sky

    # ---------- 远山（三层，大气透视）----------
    ridges = ((0.44, np.array([0.30, 0.34, 0.40]), 3),
              (0.52, np.array([0.16, 0.20, 0.20]), 11),
              (0.60, np.array([0.07, 0.11, 0.09]), 23))
    for base_v, color, off in ridges:
        noise = fbm(width, height, seed + off, octaves=((64, 0.6), (24, 0.4)))
        ridge = base_v + (noise - 0.5) * 0.10
        mask = v > ridge
        shade = (noise[..., None] - 0.5) * 0.05
        scene[mask] = (color + shade)[mask]

    # ---------- 地面 ----------
    ground = np.zeros_like(scene)
    texture = fbm(width, height, seed + 41, octaves=((48, 0.4), (12, 0.35), (5, 0.25)))
    warmth = fbm(width, height, seed + 77, octaves=((32, 0.6), (8, 0.4)))
    ground[..., 0] = 0.10 + 0.16 * texture + 0.10 * warmth
    ground[..., 1] = 0.14 + 0.20 * texture
    ground[..., 2] = 0.05 + 0.06 * texture
    ground_mask = v > 0.60
    scene[ground_mask] = ground[ground_mask]

    # ---------- 前景：深色植被（试趾部）+ 几块暖色石头 ----------
    foliage = np.zeros_like(scene)
    detail = fbm(width, height, seed + 101, octaves=((24, 0.5), (6, 0.3), (3, 0.2)))
    foliage[..., 0] = 0.020 + 0.030 * detail
    foliage[..., 1] = 0.035 + 0.055 * detail
    foliage[..., 2] = 0.015 + 0.020 * detail
    foliage_mask = v > 0.80
    scene[foliage_mask] = foliage[foliage_mask]
    rock_noise = fbm(width, height, seed + 13, octaves=((40, 0.6), (10, 0.4)))
    rocks = (v > 0.86) & (rock_noise > 0.62)
    scene[rocks] = np.array([0.16, 0.12, 0.10]) + rock_noise[rocks][..., None] * 0.06

    # ---------- 场景中的测试卡 ----------
    if with_card:
        card_x0, card_x1 = int(width * 0.30), int(width * 0.70)
        card_y0, card_y1 = int(height * 0.50), int(height * 0.88)
        # 接触阴影
        shadow = np.exp(-(((xx - (card_x0 + card_x1) / 2)
                           / (0.30 * width)) ** 2
                          + ((yy - (card_y1 + 0.02 * height))
                             / (0.045 * height)) ** 2))
        scene *= (1.0 - 0.55 * shadow)[..., None]
        _paint_rect(scene, card_x0, card_y0, card_x1, card_y1,
                    np.array([0.42, 0.42, 0.41]))

        pad_x = int((card_x1 - card_x0) * 0.05)
        pad_y = int((card_y1 - card_y0) * 0.05)
        inner_x0, inner_x1 = card_x0 + pad_x, card_x1 - pad_x
        inner_y0, inner_y1 = card_y0 + pad_y, card_y1 - pad_y
        wedge_h = int((inner_y1 - inner_y0) * 0.17)
        grid_y1 = inner_y1 - wedge_h - pad_y

        # 24 色块（6 列 × 4 行）
        patch_linear = cm.srgb_decode(COLORCHECKER_24_SRGB)
        cell_w = (inner_x1 - inner_x0) / 6.0
        cell_h = (grid_y1 - inner_y0) / 4.0
        gap = max(1, int(min(cell_w, cell_h) * 0.07))
        for index, color in enumerate(patch_linear):
            row, col = divmod(index, 6)
            x0 = inner_x0 + col * cell_w + gap
            x1 = inner_x0 + (col + 1) * cell_w - gap
            y0 = inner_y0 + row * cell_h + gap
            y1 = inner_y0 + (row + 1) * cell_h - gap
            _paint_rect(scene, x0, y0, x1, y1, color)

        # 21 级灰阶（等密度步进）
        steps = 21
        step_w = (inner_x1 - inner_x0) / float(steps)
        for index in range(steps):
            value = 10.0 ** (-2.0 * index / (steps - 1))
            _paint_rect(scene,
                        inner_x0 + index * step_w, inner_y1 - wedge_h,
                        inner_x0 + (index + 1) * step_w, inner_y1,
                        np.array([value, value, value]))

        # 分辨率元件：棋盘（放在卡片右下角）
        detail_x0 = inner_x1 - int((inner_x1 - inner_x0) * 0.22)
        detail_x1 = inner_x1 - pad_x
        detail_y0 = inner_y0
        detail_y1 = inner_y0 + int((grid_y1 - inner_y0) * 0.10)
        size = max(2, int((grid_y1 - inner_y0) * 0.012))
        block = np.indices((detail_y1 - detail_y0,
                            detail_x1 - detail_x0)).sum(0)
        checker = ((block // max(1, size)) % 2 == 0).astype(np.float64)
        region = scene[detail_y0:detail_y1, detail_x0:detail_x1]
        region[...] = 0.02 + checker[..., None] * 0.72

    return np.clip(scene, 0.0, None)


def make_film_negative(width: int = 1200, height: int = 800, *,
                       stock=None, seed: int = 7, border: float = 0.03,
                       with_card: bool = True, grain: float | None = None,
                       scene=None):
    """合成一张"扫描到的彩色负片"：场景 → C-41 正向模型。

    返回 :func:`aurhythm.filmsim.expose` 的 dict（含 negative / base_rgb /
    scene 真值 / 三层染料密度）。
    """
    from . import filmsim

    if scene is None:
        scene = make_scene(width, height, seed=seed, with_card=with_card)
    stock = stock or filmsim.DEFAULT_STOCK
    return filmsim.expose(scene, stock, grain=grain, seed=seed, border=border)
