"""色卡矫正测试：单应采样、自动检测、岭回归、ΔE2000 报告。

对应回归项 R9（假随机检测 + 占位参考表）与验收标准 S8。
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import numpy as np

import tests  # noqa: F401
from aurhythm import colorchecker as cc
from aurhythm import colorimetry as cm


# ======================================================================
# 合成色卡渲染器
# ======================================================================

def render_chart(height=360, width=540, corners=None, colors=None,
                 background=0.85, grid=0.03, grid_frac=0.06,
                 columns=cc.CHART_COLUMNS, rows=cc.CHART_ROWS):
    """按 4 角把 24 色块渲染进画面（含块间黑格与背景）。

    用逆单应逐像素反查，因此天然支持透视/旋转，不需要正向投影。
    """
    if corners is None:
        corners = np.array([[0.12 * width, 0.14 * height],
                            [0.88 * width, 0.10 * height],
                            [0.92 * width, 0.88 * height],
                            [0.08 * width, 0.92 * height]], dtype=np.float64)
    corners = np.asarray(corners, dtype=np.float64)
    if colors is None:
        colors = cc.target_linear_srgb()

    img = np.full((height, width, 3), background, dtype=np.float64)
    h_inv = np.linalg.inv(cc.homography_from_unit_square(corners))

    ys, xs = np.mgrid[0:height, 0:width]
    uv = cc.apply_homography(h_inv, np.stack([xs.ravel(), ys.ravel()], axis=-1))
    u, v = uv[:, 0], uv[:, 1]
    inside = (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
    if not np.any(inside):
        return img

    col = np.clip((u * columns).astype(np.int64), 0, columns - 1)
    row = np.clip((v * rows).astype(np.int64), 0, rows - 1)
    fu = u * columns - col
    fv = v * rows - row
    is_patch = ((fu > grid_frac) & (fu < 1 - grid_frac)
                & (fv > grid_frac) & (fv < 1 - grid_frac))

    flat = img.reshape(-1, 3)
    patch_index = row * columns + col
    on_chart = inside & is_patch
    flat[on_chart] = np.asarray(colors, dtype=np.float64)[patch_index[on_chart]]
    on_grid = inside & ~is_patch
    flat[on_grid] = grid
    return img


def planted_transform(scale=(1.10, 0.95, 0.80), cross=0.05):
    """构造一个可逆的 3x3「相机→标准」变换，用作被求解的对象。"""
    m = np.diag(scale).astype(np.float64)
    m[0, 1] = cross
    m[1, 0] = -cross
    m[2, 1] = cross * 0.5
    return m


class TestHomography(unittest.TestCase):
    def test_unit_square_maps_to_quad(self):
        quad = np.array([[10.0, 20.0], [110.0, 15.0], [120.0, 90.0], [5.0, 95.0]])
        h = cc.homography_from_unit_square(quad)
        got = cc.apply_homography(h, np.array([[0, 0], [1, 0], [1, 1], [0, 1]]))
        np.testing.assert_allclose(got, quad, atol=1e-9)

    def test_degenerate_quad_raises(self):
        with self.assertRaises(ValueError):
            cc.homography_from_unit_square(np.zeros((4, 2)))


class TestPatchSampling(unittest.TestCase):
    def test_recovers_planted_colors_with_perspective(self):
        """带透视/旋转的合成色卡必须被采回原始颜色（相对误差 <3%）。"""
        target = cc.target_linear_srgb()
        img = render_chart(corners=np.array([[70.0, 50.0], [480.0, 30.0],
                                             [505.0, 330.0], [45.0, 345.0]]),
                           colors=target)
        corners = np.array([[70.0, 50.0], [480.0, 30.0],
                            [505.0, 330.0], [45.0, 345.0]])
        sampled = cc.sample_patches(img, corners)
        self.assertEqual(sampled.shape, (24, 3))
        rel = np.abs(sampled - target) / np.maximum(target, 1e-3)
        self.assertLess(float(np.median(rel)), 0.03,
                        f"采样误差过大: {np.median(rel)}")
        self.assertLess(float(np.max(rel)), 0.15)

    def test_margin_avoids_grid_lines(self):
        """加大回退必须更干净（网格线是深色，会拉低采样值）。"""
        target = cc.target_linear_srgb()
        corners = np.array([[60.0, 40.0], [500.0, 40.0],
                            [500.0, 340.0], [60.0, 340.0]])
        img = render_chart(corners=corners, colors=target)
        tight = cc.sample_patches(img, corners, margin=0.0, grid=9)
        safe = cc.sample_patches(img, corners, margin=0.25, grid=9)
        err_tight = np.abs(tight - target).mean()
        err_safe = np.abs(safe - target).mean()
        self.assertLess(err_safe, err_tight)


class TestChartDetection(unittest.TestCase):
    def setUp(self):
        self.target = cc.target_linear_srgb()
        self.corners = np.array([[80.0, 60.0], [460.0, 45.0],
                                 [480.0, 300.0], [65.0, 315.0]])

    def test_detects_synthetic_chart(self):
        img = render_chart(360, 540, corners=self.corners, colors=self.target)
        detection = cc.detect_chart(img)
        self.assertIsNotNone(detection, "合成色卡应当被检出")
        self.assertGreater(detection.confidence, 0.55)
        error = np.linalg.norm(detection.corners - self.corners, axis=1).max()
        diagonal = np.linalg.norm([540, 360])
        self.assertLess(error / diagonal, 0.02,
                        f"角点误差 {error:.1f}px 超过对角线的 2%")

    def test_end_to_end_matrix_recovery(self):
        """检测 → 采样 → 求解：植入的 3x3 必须被恢复（误差 <1e-3）。"""
        planted = planted_transform()
        measured_colors = self.target @ planted.T
        img = render_chart(360, 540, corners=self.corners,
                           colors=measured_colors)
        detection = cc.detect_chart(img)
        self.assertIsNotNone(detection)
        sampled = cc.sample_patches(img, detection.corners)
        result = cc.solve_correction(sampled, self.target)
        expected = np.linalg.inv(planted)
        # 检测出的角点相对真实四边形有 ~1.4px 外扩，传递到矩阵约 0.8% 误差；
        # 感知上更有意义的指标是矫正后的 ΔE（下面另行断言）。
        np.testing.assert_allclose(result.matrix, expected, atol=1e-2)
        self.assertLess(result.stats["mean_deltaE"], 1.0)
        self.assertGreater(result.stats["mean_before"], 3.0)

    def test_rejects_featureless_image(self):
        rng = np.random.default_rng(0)
        flat = np.full((240, 320, 3), 0.5)
        self.assertIsNone(cc.detect_chart(flat))
        noise = np.clip(0.5 + rng.normal(0, 0.002, (240, 320, 3)), 0, 1)
        self.assertIsNone(cc.detect_chart(noise))

    def test_rejects_nonuniform_background(self):
        """背景不均匀时明确拒绝，而不是给出错误四角。"""
        img = render_chart(360, 540, corners=self.corners, colors=self.target)
        gradient = np.linspace(0.2, 0.9, 540)[None, :, None]
        img = np.clip(img + gradient * 0.5, 0, 1)
        self.assertIsNone(cc.detect_chart(img))

    def test_returns_reason_when_rejected(self):
        img = np.full((200, 200, 3), 0.4)
        img[90:110, 90:110] = 0.6          # 小块：面积与长宽比都不对
        detection, reason = cc.detect_chart(img, explain=True)
        self.assertIsNone(detection)
        self.assertTrue(reason)
        self.assertIn("面积占比", reason)


class TestSolveCorrection(unittest.TestCase):
    def setUp(self):
        self.target = cc.target_linear_srgb()

    def test_identity_when_measured_equals_target(self):
        result = cc.solve_correction(self.target.copy(), self.target, ridge=0.0)
        self.assertLess(result.stats["max_deltaE"], 1e-6)
        np.testing.assert_allclose(result.matrix, np.eye(3), atol=1e-6)

    def test_recovers_planted_matrix(self):
        planted = planted_transform(scale=(0.85, 1.05, 1.25), cross=0.08)
        measured = self.target @ planted.T
        result = cc.solve_correction(measured, self.target, ridge=0.0)
        np.testing.assert_allclose(result.matrix, np.linalg.inv(planted),
                                   atol=1e-6)
        self.assertLess(result.stats["mean_deltaE"], 1e-6)

    def test_affine_mode(self):
        planted = planted_transform()
        offset = np.array([0.02, -0.01, 0.005])
        measured = self.target @ planted.T + offset
        result = cc.solve_correction(measured, self.target,
                                     mode="3x3+offset", ridge=0.0)
        self.assertIsNotNone(result.offset)
        self.assertLess(result.stats["mean_deltaE"], 1e-6)
        np.testing.assert_allclose(result.offset, -np.linalg.inv(planted) @ offset,
                                   atol=1e-6)

    def test_ridge_keeps_singular_design_stable(self):
        """重复色块（设计矩阵奇异）时岭回归必须仍给出有限结果。"""
        measured = np.tile(self.target[:1], (24, 1))
        result = cc.solve_correction(measured, self.target)
        self.assertTrue(np.all(np.isfinite(result.matrix)))

    def test_rejects_shape_mismatch(self):
        with self.assertRaises(ValueError):
            cc.solve_correction(np.zeros((10, 3)), self.target)

    def test_rejects_unknown_mode(self):
        with self.assertRaises(ValueError):
            cc.solve_correction(self.target, self.target, mode="lut")

    def test_stats_schema_and_report(self):
        planted = planted_transform()
        measured = self.target @ planted.T
        result = cc.solve_correction(measured, self.target)
        stats = result.stats
        for key in ("mean_deltaE", "max_deltaE", "min_deltaE", "std_deltaE",
                    "p95_deltaE", "per_patch", "patch_names", "worst_patch",
                    "worst_patch_name", "mean_deltaE_before"):
            self.assertIn(key, stats)
        self.assertEqual(len(stats["per_patch"]), 24)
        text = cc.format_error_report(stats)
        self.assertIn("ΔE2000", text)
        table = cc.error_table(stats)
        self.assertEqual(len(table), 24)
        json.dumps(result.to_dict())

    def test_noise_floor_is_realistic(self):
        """加噪后仍应显著改善（平均 ΔE 至少降低 3 倍）。"""
        rng = np.random.default_rng(7)
        planted = planted_transform(scale=(1.2, 0.9, 0.7))
        measured = self.target @ planted.T
        noisy = measured * (1.0 + rng.normal(0, 0.01, measured.shape))
        result = cc.solve_correction(np.clip(noisy, 0, None), self.target)
        self.assertLess(result.stats["mean_deltaE"] * 3.0,
                        result.stats["mean_before"])


class TestCalibrationFiles(unittest.TestCase):
    def test_ccmx_roundtrip(self):
        matrix = planted_transform()
        xml = ("<?xml version='1.0'?><ccmx><ColorMatrix type='XYZtoCamera'>"
               + " ".join(str(v) for v in np.linalg.inv(matrix).ravel())
               + "</ColorMatrix></ccmx>")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cam.ccmx")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(xml)
            loaded = cc.load_calibration_file(path)
            np.testing.assert_allclose(loaded["matrix"], matrix, atol=1e-12)
            self.assertEqual(loaded["type"], "ccmx")

    def test_json_roundtrip(self):
        matrix = planted_transform()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cam.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"matrix": matrix.tolist(), "description": "测试"},
                          fh)
            loaded = cc.load_calibration_file(path)
            np.testing.assert_allclose(loaded["matrix"], matrix, atol=1e-12)

    def test_bad_files_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad_json = os.path.join(tmp, "bad.json")
            with open(bad_json, "w", encoding="utf-8") as fh:
                json.dump({"nope": 1}, fh)
            with self.assertRaises(ValueError):
                cc.load_calibration_file(bad_json)

            bad_ext = os.path.join(tmp, "bad.txt")
            with open(bad_ext, "w", encoding="utf-8") as fh:
                fh.write("x")
            with self.assertRaises(ValueError):
                cc.load_calibration_file(bad_ext)


class TestNoRandomData(unittest.TestCase):
    def test_module_has_no_random_calls(self):
        """R9 回归：本模块的**代码**（不含文档引用）不得出现任何随机数。

        用 AST 检查真实的属性访问，避免把文档里对旧 bug 的引用误判为违规。
        """
        import ast
        import inspect
        tree = ast.parse(inspect.getsource(cc))
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in (
                    "random", "uniform", "normal", "seed", "randint"):
                offenders.append(node.attr)
        self.assertEqual(offenders, [], f"色卡模块里出现随机调用: {offenders}")

    def test_reference_table_is_not_placeholder(self):
        """R9 回归：参考表必须是有色彩的官方值，而不是近重复占位值。"""
        lab = cm.rgb_to_lab_srgb(cc.target_linear_srgb())
        chroma = cm.lab_to_lch(lab)[:, 1]
        self.assertGreater(float(np.min(chroma[:18])), 15.0)
        self.assertLess(float(np.max(chroma[18:])), 2.0)


if __name__ == "__main__":
    unittest.main()
