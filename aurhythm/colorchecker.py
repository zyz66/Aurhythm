"""色卡矫正：棋盘检测、色块采样、岭回归求解、ΔE2000 报告。

为什么要有这个模块
------------------
旧实现 ``detect_colorchecker`` 是::

    detected = COLORCHECKER_24_D50 * np.random.uniform(0.85, 1.15, (24, 3))

—— 随机数冒充「自动检测」，而且参考表本身还是占位假数据。本模块把它换成
两条**确定性**路径：

* **手动四角**（保证路径）：在预览上点 4 个角，单应变换后按 6×4 网格采样。
  一定可用、一定可复现、可被合成色卡精确验证。
* **自动检测**（best-effort）：降采样 → Otsu 前景 → 最大连通域 →
  四边形拟合 → 用「长宽比 / 矩形度 / 面积占比 / 彩度」打分。
  置信度不足时返回 ``None`` 并给出**具体原因**，绝不编造数据。

ΔE 一律使用 **CIEDE2000**，在 Lab(D65) 下计算；参考值用 X-Rite 官方
sRGB 表（``constants.COLORCHECKER_24_SRGB``）线性化后使用。
"""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

from . import colorimetry as cm
from .constants import COLORCHECKER_24_NAMES, COLORCHECKER_24_SRGB

#: ColorChecker Classic 的排布
CHART_COLUMNS = 6
CHART_ROWS = 4


class ChartDetectionError(Exception):
    """检测失败（消息面向用户）。"""


@dataclass
class CorrectionResult:
    """色卡矫正结果（矩阵 + 逐色块 ΔE2000 + 统计）。"""

    matrix: np.ndarray
    offset: np.ndarray | None
    measured: np.ndarray
    target: np.ndarray
    predicted: np.ndarray
    delta_e_before: np.ndarray
    delta_e_after: np.ndarray
    names: tuple = COLORCHECKER_24_NAMES

    @property
    def stats(self) -> dict:
        after = self.delta_e_after
        before = self.delta_e_before
        worst = int(np.argmax(after))
        return {
            "mean_deltaE": float(np.mean(after)),
            "max_deltaE": float(np.max(after)),
            "min_deltaE": float(np.min(after)),
            "std_deltaE": float(np.std(after)),
            "p95_deltaE": float(np.percentile(after, 95)),
            "per_patch": [float(v) for v in after],
            "per_patch_before": [float(v) for v in before],
            "patch_names": list(self.names),
            "worst_patch": worst,
            "worst_patch_name": self.names[worst],
            "mean_before": float(np.mean(before)),
            "max_before": float(np.max(before)),
            # 兼容旧字段名
            "mean_deltaE_before": float(np.mean(before)),
            "max_deltaE_before": float(np.max(before)),
        }

    def to_dict(self) -> dict:
        return {
            "matrix": self.matrix.tolist(),
            "offset": None if self.offset is None else self.offset.tolist(),
            "stats": self.stats,
        }


@dataclass
class ChartDetection:
    """自动检测结果。"""

    corners: np.ndarray          # (4,2) 顺序：左上, 右上, 右下, 左下
    confidence: float
    reason: str = ""
    details: dict = field(default_factory=dict)


# ======================================================================
# 参考值 / 文件加载
# ======================================================================

def target_linear_srgb() -> np.ndarray:
    """ColorChecker 24 色的**线性 sRGB** 参考值。"""
    return cm.srgb_decode(COLORCHECKER_24_SRGB)


def load_calibration_file(filepath: str) -> dict:
    """加载 ``.ccmx`` / ``.json`` 校准文件里的 3x3 矩阵。"""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".ccmx":
        return _load_ccmx(filepath)
    if ext == ".json":
        return _load_json(filepath)
    raise ValueError(f"不支持的校准文件格式: {ext or '(无扩展名)'}")


def _load_ccmx(filepath: str) -> dict:
    try:
        root = ET.parse(filepath).getroot()
    except Exception as exc:                        # noqa: BLE001
        raise ValueError(f"{os.path.basename(filepath)}: XML 解析失败 ({exc})") from exc
    for elem in root.iter("ColorMatrix"):
        if elem.get("type") in ("XYZtoCamera", "XYZToCamera"):
            values = (elem.text or "").split()
            if len(values) != 9:
                raise ValueError("ColorMatrix 需要 9 个数")
            xyz_to_cam = np.array([float(v) for v in values]).reshape(3, 3)
            return {
                "matrix": np.linalg.inv(xyz_to_cam),
                "type": "ccmx",
                "source": os.path.basename(filepath),
            }
    raise ValueError(f"{os.path.basename(filepath)}: 未找到 XYZtoCamera 矩阵")


def _load_json(filepath: str) -> dict:
    with open(filepath, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if "matrix" not in data:
        raise ValueError(f"{os.path.basename(filepath)}: JSON 缺少 'matrix'")
    matrix = np.array(data["matrix"], dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"matrix 必须是 3x3，收到 {matrix.shape}")
    return {
        "matrix": matrix,
        "type": "json",
        "source": os.path.basename(filepath),
        "description": data.get("description", ""),
    }


# ======================================================================
# 单应变换与色块采样（手动四角 = 保证路径）
# ======================================================================

def homography_from_unit_square(corners) -> np.ndarray:
    """把单位正方形 (0,0)(1,0)(1,1)(0,1) 映射到给定四边形的 3x3 单应矩阵。

    用标准 DLT：``corners`` 顺序必须是 左上 / 右上 / 右下 / 左下。
    """
    quad = np.asarray(corners, dtype=np.float64)
    if quad.shape != (4, 2):
        raise ValueError("四角必须是 4 个 (x, y)")
    if not np.all(np.isfinite(quad)):
        raise ValueError("四角包含非有限值")
    if _polygon_area(quad) < 1e-6:
        raise ValueError("四角退化（面积为零或共线）")
    src = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    a = []
    for (u, v), (x, y) in zip(src, quad):
        a.append([u, v, 1, 0, 0, 0, -x * u, -x * v, -x])
        a.append([0, 0, 0, u, v, 1, -y * u, -y * v, -y])
    a = np.array(a, dtype=np.float64)
    _u, s, vt = np.linalg.svd(a)
    h = vt[-1].reshape(3, 3)
    if abs(h[2, 2]) < 1e-12:
        raise ValueError("单应矩阵退化（四角共线？）")
    return h / h[2, 2]


def apply_homography(h_matrix, uv) -> np.ndarray:
    uv = np.atleast_2d(np.asarray(uv, dtype=np.float64))
    ones = np.ones((uv.shape[0], 1))
    homo = np.hstack([uv, ones]) @ np.asarray(h_matrix).T
    return homo[:, :2] / homo[:, 2:3]


def _bilinear(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    x = np.clip(xs, 0, w - 1.001)
    y = np.clip(ys, 0, h - 1.001)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = (x - x0)[..., None]
    fy = (y - y0)[..., None]
    c00 = img[y0, x0]
    c10 = img[y0, np.minimum(x0 + 1, w - 1)]
    c01 = img[np.minimum(y0 + 1, h - 1), x0]
    c11 = img[np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)]
    top = c00 * (1 - fx) + c10 * fx
    bottom = c01 * (1 - fx) + c11 * fx
    return top * (1 - fy) + bottom * fy


def sample_patches(img: np.ndarray, corners, margin: float = 0.25,
                   grid: int = 7, columns: int = CHART_COLUMNS,
                   rows: int = CHART_ROWS) -> np.ndarray:
    """按单应网格采样 24 个色块，返回 (24,3) 的**线性**中位数值。

    ``margin`` 是每块边缘的回退比例（0.25 = 只取中央 50%），
    目的是避开块间黑格与印字；取中位数以抗高光/灰尘点。
    """
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    h_matrix = homography_from_unit_square(corners)

    offsets = (np.arange(grid) + 0.5) / grid            # 块内采样网格 0..1
    patch_values = []
    for row in range(rows):
        for col in range(columns):
            u0, u1 = col / columns, (col + 1) / columns
            v0, v1 = row / rows, (row + 1) / rows
            du, dv = (u1 - u0) * margin, (v1 - v0) * margin
            us = u0 + du + offsets * (u1 - u0 - 2 * du)
            vs = v0 + dv + offsets * (v1 - v0 - 2 * dv)
            uu, vv = np.meshgrid(us, vs)
            xy = apply_homography(h_matrix, np.stack([uu.ravel(), vv.ravel()],
                                                     axis=-1))
            samples = _bilinear(arr, xy[:, 0], xy[:, 1])
            patch_values.append(np.median(samples, axis=0))
    return np.array(patch_values, dtype=np.float64)


# ======================================================================
# 自动检测（best-effort，失败即明确拒绝）
# ======================================================================

def _otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    hist, edges = np.histogram(values, bins=bins)
    total = hist.sum()
    if total == 0:
        return 0.0
    centers = (edges[:-1] + edges[1:]) / 2.0
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    valid = (weight_bg > 0) & (weight_fg > 0)
    if not np.any(valid):
        return float(np.mean(values))
    mean_bg = np.cumsum(hist * centers) / np.maximum(weight_bg, 1)
    mean_total = np.sum(hist * centers) / total
    mean_fg = (mean_total - np.cumsum(hist * centers)) / np.maximum(weight_fg, 1)
    between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    between[~valid] = -1.0
    return float(centers[int(np.argmax(between))])


def _label_components(mask: np.ndarray):
    """两遍扫描 + 并查集的连通域标记（返回标签图与各标签像素数）。"""
    h, w = mask.shape
    parent: list[int] = []

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    runs = []                     # [(row, [(start, end, label), ...]), ...]
    previous_runs: list = []
    for y in range(h):
        row = mask[y]
        if not row.any():
            previous_runs = []
            continue
        diff = np.diff(row.astype(np.int8))
        starts = list(np.flatnonzero(diff == 1) + 1)
        ends = list(np.flatnonzero(diff == -1) + 1)
        if row[0]:
            starts.insert(0, 0)
        if row[-1]:
            ends.append(w)
        current = []
        for start, end in zip(starts, ends):
            label = len(parent)
            parent.append(label)
            for p_start, p_end, p_label in previous_runs:
                if p_start < end and start < p_end:
                    union(label, p_label)
            current.append((start, end, label))
        runs.append((y, current))
        previous_runs = current

    labels = np.zeros((h, w), dtype=np.int64)
    sizes: dict[int, int] = {}
    for y, current in runs:
        for start, end, label in current:
            root = find(label)
            labels[y, start:end] = root + 1
            sizes[root + 1] = sizes.get(root + 1, 0) + (end - start)
    return labels, sizes


def _quad_from_mask(mask: np.ndarray):
    """由掩膜求四个极值角点（顺序：左上, 右上, 右下, 左下）。"""
    ys, xs = np.nonzero(mask)
    if len(xs) < 4:
        return None
    pts = np.stack([xs, ys], axis=-1).astype(np.float64)
    s = pts[:, 0] + pts[:, 1]
    d = pts[:, 0] - pts[:, 1]
    corners = np.array([pts[np.argmin(s)], pts[np.argmax(d)],
                        pts[np.argmax(s)], pts[np.argmin(d)]])
    return corners


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Andrew 单调链凸包（返回逆时针顶点，不重复首点）。"""
    pts = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return ((a[0] - o[0]) * (b[1] - o[1])
                - (a[1] - o[1]) * (b[0] - o[0]))

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def _polygon_area(corners) -> float:
    x, y = corners[:, 0], corners[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _border_ring(arr: np.ndarray, width: int = 3) -> np.ndarray:
    h, w = arr.shape[:2]
    width = max(1, min(width, h // 4, w // 4))
    return np.concatenate([
        arr[:width].reshape(-1, 3), arr[-width:].reshape(-1, 3),
        arr[:, :width].reshape(-1, 3), arr[:, -width:].reshape(-1, 3),
    ])


def detect_chart(img: np.ndarray, max_size: int = 1024,
                 min_confidence: float = 0.55, explain: bool = False):
    """best-effort 自动检测色卡外框，返回 :class:`ChartDetection` 或 ``None``。

    算法（确定性）：

    1. 降采样到最长边 <= ``max_size``；
    2. 由**边框环**估计背景色，求每像素到背景的距离图，用 Otsu 分割
       —— 色卡本体（含块间黑格）因此成为一个连通块，不受个别色块明暗影响；
    3. 取最大连通域，用四个极值点拟合成四边形；
    4. 评分：面积占比 8%–92%、长宽比 1.0–2.0（6×4 方块的 ≈1.5）、
       矩形度 >0.75、掩膜内彩度不低于外部。

    几何项（面积占比 / 长宽比 / 凸包填充率 / 掩膜密度）是**硬门槛**，
    任一项不过直接拒绝；彩度项只影响置信度。绝不返回编造的四角。

    背景不均匀（边框环颜色方差大）时直接拒绝并提示改用手动四角。
    ``explain=True`` 时返回 ``(detection, reason)``，便于 UI 告诉用户
    为什么被拒绝、该怎么做。
    """
    def _reject(reason):
        return (None, reason) if explain else None

    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    step = 1
    if max(arr.shape[:2]) > max_size:
        step = max(1, int(round(max(arr.shape[:2]) / max_size)))
        arr = arr[::step, ::step]
    scale = 1.0 / step

    ring = _border_ring(arr)
    background = np.median(ring, axis=0)
    ring_spread = float(np.mean(np.std(ring, axis=0)))
    if ring_spread > 0.15:
        return _reject(f"背景不均匀（边框颜色标准差 {ring_spread:.2f}），"
                       "请改用手动四角")

    distance = np.abs(arr - background.reshape(1, 1, 3)).sum(axis=2)
    threshold = max(_otsu_threshold(distance.ravel()), 0.02)
    mask = distance > threshold
    if mask.sum() == 0 or mask.all():
        return _reject("画面几乎没有可分割的前景，请改用手动四角")

    labels, sizes = _label_components(mask)
    if not sizes:
        return _reject("未找到任何连通前景")
    largest = max(sizes, key=sizes.get)
    component = labels == largest
    area_fraction = component.sum() / component.size

    corners = _quad_from_mask(component)
    if corners is None:
        return _reject("前景太小，无法拟合四边形")
    quad_area = _polygon_area(corners)
    if quad_area <= 0:
        return _reject("四边形面积为零（前景共线）")
    # 色卡的浅色块（尤其白色块）与背景几乎同色，掩膜必然是「网格状」的，
    # 因此不能用「掩膜像素 / 四边形面积」当矩形度 —— 那会恒低。
    # 改用**凸包填充率**（色卡的凸包≈四边形）＋**掩膜密度下限**（防空散点）。
    hull = convex_hull(np.stack(np.nonzero(component)[::-1], axis=-1))
    hull_area = _polygon_area(hull) if len(hull) >= 3 else 0.0
    rectangularity = hull_area / quad_area if quad_area > 0 else 0.0
    fill_density = float(component.sum()) / quad_area

    edges = [np.linalg.norm(corners[(i + 1) % 4] - corners[i]) for i in range(4)]
    width = 0.5 * (edges[0] + edges[2])
    height = 0.5 * (edges[1] + edges[3])
    aspect = (width / height) if height > 1e-6 else 0.0

    inside = arr[component]
    outside = arr[~component]
    chroma_inside = float(np.mean(np.abs(inside - inside.mean(axis=1, keepdims=True))))
    chroma_outside = (float(np.mean(np.abs(outside - outside.mean(axis=1,
                                                                  keepdims=True))))
                      if outside.size else 0.0)

    hard_failures = []
    if not 0.08 <= area_fraction <= 0.92:
        hard_failures.append(f"面积占比 {area_fraction:.0%} 不在 8%–92%")
    if not 1.0 <= aspect <= 2.0:
        hard_failures.append(f"长宽比 {aspect:.2f} 不像 6×4 色卡(≈1.5)")
    if rectangularity <= 0.80:
        hard_failures.append(f"凸包填充率 {rectangularity:.2f} 过低")
    if fill_density <= 0.08:
        hard_failures.append(f"掩膜密度 {fill_density:.2f} 过低（可能只是零散特征）")
    if hard_failures:
        return _reject("；".join(hard_failures))

    # 几何已通过，彩度项只影响置信度
    reasons = []
    score = 0.75
    if chroma_inside >= chroma_outside * 0.9:
        score += 0.25
    else:
        reasons.append("掩膜内彩度低于外部（几何通过，但请目视确认）")

    details = {
        "area_fraction": round(area_fraction, 4),
        "aspect": round(aspect, 3),
        "rectangularity": round(rectangularity, 3),
        "fill_density": round(fill_density, 4),
        "chroma_inside": round(chroma_inside, 4),
        "chroma_outside": round(chroma_outside, 4),
        "component_pixels": int(component.sum()),
        "background": [round(float(v), 4) for v in background],
        "ring_spread": round(ring_spread, 4),
        "downscale": step,
    }
    if score < min_confidence:
        return _reject("；".join(reasons) or "置信度不足")
    detection = ChartDetection(corners=corners / scale, confidence=score,
                               reason="；".join(reasons), details=details)
    return (detection, "") if explain else detection


# ======================================================================
# 求解与报告
# ======================================================================

def _to_lab(linear_rgb: np.ndarray) -> np.ndarray:
    safe = np.clip(np.asarray(linear_rgb, dtype=np.float64), 0.0, None)
    return cm.rgb_to_lab_srgb(safe, white="D65")


def solve_correction(measured, target=None, mode: str = "3x3",
                     ridge: float = 1e-3) -> CorrectionResult:
    """岭回归求解 ``measured → target`` 的线性矫正。

    ``M = (AᵀA + λI)⁻¹AᵀB``，``corrected = measured · Mᵀ``。
    ``mode='3x3+offset'`` 时额外求解每通道偏移（仿射，4 列）。

    返回带逐色块 ΔE2000（矫正前/后）的结果对象。
    """
    a = np.asarray(measured, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"测得色块必须是 (N,3)，收到 {a.shape}")
    if target is None:
        target = target_linear_srgb()
    b = np.asarray(target, dtype=np.float64)
    if b.shape != a.shape:
        raise ValueError(f"目标色块形状 {b.shape} 与测得 {a.shape} 不一致")

    # 岭回归的 λ 相对**设计矩阵尺度**归一化：λ 表示「平均平方量级的比例」，
    # 因此与像素数值量级（0..1 还是 0..65535）无关。
    scale_sq = float(np.trace(a.T @ a)) / max(a.shape[1], 1)
    ridge_abs = max(ridge, 0.0) * max(scale_sq, 1e-12)

    if mode == "3x3":
        design = a
        matrix = (np.linalg.solve(design.T @ design + ridge_abs * np.eye(3),
                                  design.T @ b)).T
        offset = None
        predicted = a @ matrix.T
    elif mode == "3x3+offset":
        design = np.hstack([a, np.ones((a.shape[0], 1))])
        solution = np.linalg.solve(design.T @ design + ridge_abs * np.eye(4),
                                   design.T @ b)
        matrix = solution[:3].T
        offset = solution[3]
        predicted = a @ matrix.T + offset
    else:
        raise ValueError("mode 必须是 '3x3' 或 '3x3+offset'")

    lab_measured = _to_lab(a)
    lab_target = _to_lab(b)
    lab_predicted = _to_lab(predicted)
    return CorrectionResult(
        matrix=matrix,
        offset=offset,
        measured=a,
        target=b,
        predicted=predicted,
        delta_e_before=cm.delta_e_2000(lab_measured, lab_target),
        delta_e_after=cm.delta_e_2000(lab_predicted, lab_target),
    )


def apply_correction(linear_rgb, matrix, offset=None) -> np.ndarray:
    """应用矫正矩阵（不裁剪，交由末端处理）。"""
    arr = np.asarray(linear_rgb, dtype=np.float64)
    flat = arr.reshape(-1, 3) @ np.asarray(matrix, dtype=np.float64).T
    if offset is not None:
        flat = flat + np.asarray(offset, dtype=np.float64)
    return flat.reshape(arr.shape)


def format_error_report(stats) -> str:
    """把 ΔE 统计格式化成分析页/CLI 用的文本。"""
    if not stats:
        return "未进行色卡校准"
    return (
        f"ΔE2000 平均: {stats['mean_deltaE']:.3f}\n"
        f"ΔE2000 最大: {stats['max_deltaE']:.3f} "
        f"({stats['worst_patch_name']})\n"
        f"ΔE2000 最小: {stats['min_deltaE']:.3f}\n"
        f"ΔE2000 P95 : {stats['p95_deltaE']:.3f}\n"
        f"ΔE2000 标准差: {stats['std_deltaE']:.3f}\n"
        f"矫正前平均: {stats['mean_before']:.3f} "
        f"(最大 {stats['max_before']:.3f})"
    )


def error_table(stats) -> list:
    """返回 (色块名, 矫正前 ΔE, 矫正后 ΔE) 列表，供表格展示。"""
    if not stats:
        return []
    return [(stats["patch_names"][i], stats["per_patch_before"][i],
             stats["per_patch"][i]) for i in range(len(stats["per_patch"]))]


__all__ = [
    "CorrectionResult", "ChartDetection", "ChartDetectionError",
    "target_linear_srgb", "load_calibration_file", "sample_patches",
    "homography_from_unit_square", "apply_homography", "detect_chart",
    "solve_correction", "apply_correction", "format_error_report",
    "error_table", "convex_hull", "CHART_COLUMNS", "CHART_ROWS",
]
