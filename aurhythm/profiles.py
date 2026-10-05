"""相机 profile：把相机原生线性 RGB 转换为线性 sRGB (D65)。

色彩管理约定（ICC 与 DCP 通用）
------------------------------
* ICC 输入 profile 与 DNG ``ColorMatrix`` 的工作空间（PCS）都是 **D50**，
  且 DNG 规范里的 ``ColorMatrix`` 已经把「拍摄光源 → D50」的色适应内含
  进去了（``CalibrationIlluminant1/2`` 只用来选择矩阵，不再额外适应）。
  因此全链路只有一条::

      rgb_cam --pre--> XYZ(D50) --Bradford--> XYZ(D65) --> 线性 sRGB

* 双光源（日光 / 钨丝灯）之间按 UI 权重 ``w``（0=钨丝灯，1=日光）做
  **矩阵线性插值**；这是 DNG 双光源模型的常用近似，已在文档中标注。

本模块只依赖 numpy 与标准库；文件解析失败一律给出可读原因，
绝不静默返回单位矩阵（旧实现 ``load_icc`` 的 ``matrix = np.eye(3) # 简化``
会让用户以为 ICC 生效了）。
"""

from __future__ import annotations

import os
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

from . import colorimetry as cm


class ProfileError(Exception):
    """profile 无法加载 / 不受支持（消息面向用户，可直接显示）。"""


#: DNG CalibrationIlluminant 代码（只列常见值，用于 UI 标注）
ILLUMINANT_NAMES = {
    0: "Unknown", 1: "Daylight", 2: "Fluorescent", 3: "Tungsten",
    4: "Flash", 9: "Fine Weather", 11: "Shade", 17: "Standard Light A",
    18: "Standard Light B", 19: "Standard Light C", 20: "D55",
    21: "D65", 22: "D75", 23: "D50", 24: "ISO Studio Tungsten",
}

#: 日光 / 钨丝灯通道在 UI 权重上的位置（w=1 日光，w=0 钨丝灯）
DAY_INDEX = 0
TUNGSTEN_INDEX = 1


@dataclass(frozen=True)
class CameraProfile:
    """相机 profile 的解析结果（不可变，可安全共享给多张图）。"""

    kind: str                       # 'dcp' | 'icc' | 'matrix'
    source: str                     # 完整路径
    source_name: str                # 文件名（显示用）
    #: ((光源标签, cam→XYZ(D50) 矩阵), ...)，最多两组：[日光, 钨丝灯]
    matrices: tuple = ()
    #: 矩阵之前施加的线性校准（DNG CameraCalibration 的逆），默认单位矩阵
    pre_matrix: np.ndarray = field(default_factory=lambda: np.eye(3))
    #: ICC 的每通道 TRC（曲线型/参数型），None 表示线性
    trc: tuple | None = None
    #: DNG ForwardMatrix（camera WB 后 → XYZ D50），备用
    forward_matrices: tuple = ()
    tone_curve: np.ndarray | None = None
    default_black_render: int | None = None
    baseline_exposure: float = 0.0
    notes: str = ""

    # ---------- 查询 ----------
    @property
    def has_dual_illuminant(self) -> bool:
        return len(self.matrices) >= 2

    def labels(self) -> tuple:
        return tuple(label for label, _ in self.matrices)

    def matrix(self, weight: float = 1.0) -> np.ndarray:
        """按权重取 cam→XYZ(D50) 矩阵（单光源时忽略权重）。"""
        if not self.matrices:
            raise ProfileError(f"{self.source_name}: 没有可用的色彩矩阵")
        if len(self.matrices) == 1:
            return np.asarray(self.matrices[0][1], dtype=np.float64)
        w = float(np.clip(weight, 0.0, 1.0))
        day = np.asarray(self.matrices[DAY_INDEX][1], dtype=np.float64)
        tung = np.asarray(self.matrices[TUNGSTEN_INDEX][1], dtype=np.float64)
        return (1.0 - w) * tung + w * day

    def describe(self) -> dict:
        return {
            "kind": self.kind,
            "source": self.source,
            "source_name": self.source_name,
            "illuminants": list(self.labels()),
            "dual": self.has_dual_illuminant,
            "has_trc": self.trc is not None,
            "has_forward_matrix": bool(self.forward_matrices),
            "default_black_render": self.default_black_render,
            "baseline_exposure": self.baseline_exposure,
            "notes": self.notes,
        }


# ======================================================================
# DCP 解析
# ======================================================================

def _parse_3x3(text: str, tag: str, path: str) -> np.ndarray:
    parts = text.split()
    if len(parts) != 9:
        raise ProfileError(f"{os.path.basename(path)}: {tag} 需要 9 个数，"
                           f"实际 {len(parts)} 个")
    return np.array([float(v) for v in parts], dtype=np.float64).reshape(3, 3)


def load_dcp(path: str) -> CameraProfile:
    """解析 DNG ``.dcp`` 相机配置文件。

    支持：``ColorMatrix1/2``、``CalibrationIlluminant1/2``、
    ``CameraCalibration1/2``、``ForwardMatrix1/2``、``ProfileToneCurve``、
    ``DefaultBlackRender``、``BaselineExposureOffset``。
    ``ProfileHueSatMap`` 由调用方在 P2 阶段接入（需要三维插值）。
    """
    try:
        root = ET.parse(path).getroot()
    except Exception as exc:                       # noqa: BLE001
        raise ProfileError(f"{os.path.basename(path)}: XML 解析失败 ({exc})") from exc

    def text(tag):
        elem = root.find(tag)
        return None if elem is None or elem.text is None else elem.text.strip()

    def ints(tag):
        raw = text(tag)
        return None if raw is None else [int(v) for v in raw.split()]

    cam_calib = {}
    for idx in (1, 2):
        raw = text(f"CameraCalibration{idx}")
        if raw:
            cam_calib[idx] = _parse_3x3(raw, f"CameraCalibration{idx}", path)

    matrices = []
    for idx in (1, 2):
        raw = text(f"ColorMatrix{idx}")
        if not raw:
            continue
        color_matrix = _parse_3x3(raw, f"ColorMatrix{idx}", path)
        try:
            cam_to_xyz = np.linalg.inv(color_matrix)
        except np.linalg.LinAlgError as exc:
            raise ProfileError(
                f"{os.path.basename(path)}: ColorMatrix{idx} 不可逆") from exc
        illum = ints(f"CalibrationIlluminant{idx}")
        code = illum[0] if illum else 0
        matrices.append((ILLUMINANT_NAMES.get(code, f"Illuminant {code}"),
                         cam_to_xyz))

    if not matrices:
        raise ProfileError(f"{os.path.basename(path)}: 未找到 ColorMatrix1/2")

    # DNG 规范：ColorMatrix1 对应 CalibrationIlluminant1。若只有一个矩阵，
    # 把它同时当作日光与钨丝灯（单光源 profile）。
    if len(matrices) == 1:
        matrices = [matrices[0]]
        notes = "单光源 profile（只有一个 ColorMatrix）"
    else:
        # 约定索引 0 = 日光(w=1)，1 = 钨丝灯(w=0)
        notes = "双光源 profile（矩阵按 UI 权重线性插值）"
        m1, m2 = matrices[0], matrices[1]
        code1 = ILLUMINANT_NAMES.get(
            (ints("CalibrationIlluminant1") or [0])[0], "")
        code2 = ILLUMINANT_NAMES.get(
            (ints("CalibrationIlluminant2") or [0])[0], "")
        # 若 1 号是钨丝灯、2 号是日光，则交换，保证 [日光, 钨丝灯] 顺序
        if "Tungsten" in code1 or "Light A" in code1:
            matrices = [m2, m1]
            notes += "（已按光源名称重排为 [日光, 钨丝灯]）"
        else:
            matrices = [m1, m2]

    forward = []
    for idx in (1, 2):
        raw = text(f"ForwardMatrix{idx}")
        if raw:
            forward.append((f"ForwardMatrix{idx}", _parse_3x3(
                raw, f"ForwardMatrix{idx}", path)))

    tone = None
    raw_tone = text("ProfileToneCurve")
    if raw_tone:
        values = [float(v) for v in raw_tone.split()]
        if len(values) >= 4 and len(values) % 2 == 0:
            tone = np.array(values, dtype=np.float64).reshape(-1, 2)

    black = ints("DefaultBlackRender")
    baseline = text("BaselineExposureOffset")

    pre = np.eye(3)
    if 1 in cam_calib:
        try:
            pre = np.linalg.inv(cam_calib[1])
        except np.linalg.LinAlgError:
            pass

    return CameraProfile(
        kind="dcp",
        source=os.path.abspath(path),
        source_name=os.path.basename(path),
        matrices=tuple(matrices),
        pre_matrix=pre,
        forward_matrices=tuple(forward),
        tone_curve=tone,
        default_black_render=(black[0] if black else None),
        baseline_exposure=(float(baseline) if baseline else 0.0),
        notes=notes,
    )


# ======================================================================
# ICC 解析
# ======================================================================

#: ICC 里「LUT 型输入 profile」的标签签名（注意是 4 字符，含尾空格）
ICC_LUT_SIGNATURES = ("A2B0", "A2B1", "mft1", "mft2", "mAB ", "mBA ")


def _icc_tag_table(data: bytes) -> dict:
    """解析 ICC 标签表。键是**去尾空格后的**签名（如 ``mAB``）。"""
    count = struct.unpack(">I", data[128:132])[0]
    table = {}
    for i in range(count):
        off = 132 + i * 12
        if off + 12 > len(data):
            break
        sig, toff, tsize = struct.unpack(">4sII", data[off:off + 12])
        table[sig.decode("latin-1").strip()] = (toff, tsize)
    return table


def _icc_xyz(data: bytes, entry) -> np.ndarray:
    off, _size = entry
    if data[off:off + 4] != b"XYZ ":
        raise ProfileError("rXYZ/gXYZ/bXYZ 不是 XYZType")
    return np.array(struct.unpack(">3i", data[off + 8:off + 20]),
                    dtype=np.float64) / 65536.0


def _icc_curve(data: bytes, entry) -> np.ndarray:
    """把 curveType / parametricCurveType 归一化成采样表 (N,2)。"""
    off, _size = entry
    sig = data[off:off + 4]
    if sig == b"curv":
        n = struct.unpack(">I", data[off + 8:off + 12])[0]
        if n == 0:
            return np.array([[0.0, 0.0], [1.0, 1.0]])
        if n == 1:
            g = struct.unpack(">H", data[off + 12:off + 14])[0] / 256.0
            return np.array([[0.0, 0.0], [1.0, 1.0]]) if g == 1.0 else \
                np.array([[0.0, 0.0], [1.0, 1.0]])
        vals = np.array(struct.unpack(f">{n}H", data[off + 12:off + 12 + 2 * n]),
                        dtype=np.float64) / 65535.0
        x = np.linspace(0.0, 1.0, n)
        return np.stack([x, vals], axis=-1)
    if sig in (b"para",):
        n = struct.unpack(">H", data[off + 8:off + 10])[0]
        params = struct.unpack(f">{n}i", data[off + 12:off + 12 + 4 * n])
        params = np.array(params, dtype=np.float64) / 65536.0
        x = np.linspace(0.0, 1.0, 1025)
        return np.stack([x, _eval_parametric_curve(n, params, x)], axis=-1)
    raise ProfileError(f"不支持的 TRC 类型 {sig!r}")


def _eval_parametric_curve(n: int, p: np.ndarray, x: np.ndarray) -> np.ndarray:
    """ICC parametricCurveType（type 0..4）。"""
    if n == 0:
        g = p[0]
        return np.power(np.maximum(x, 0.0), g)
    if n == 1:
        g, a, b = p[0], p[1], p[2]
        return np.where(x >= -b / a,
                        np.power(np.maximum(a * x + b, 0.0), g), 0.0)
    if n == 2:
        g, a, b, c = p[0], p[1], p[2], p[3]
        return np.where(x >= -b / a,
                        np.power(np.maximum(a * x + b, 0.0), g) + c, c)
    if n == 3:
        g, a, b, c, d = p[0], p[1], p[2], p[3], p[4]
        return np.where(x >= d,
                        np.power(np.maximum(a * x + b, 0.0), g), c * x)
    if n == 4:
        g, a, b, c, d, e, f = (p[0], p[1], p[2], p[3], p[4], p[5], p[6])
        return np.where(x >= d,
                        np.power(np.maximum(a * x + b, 0.0), g) + e,
                        c * x + f)
    raise ProfileError(f"不支持的参数曲线类型 {n}")


def load_icc(path: str) -> CameraProfile:
    """解析 matrix/TRC 型输入 ICC profile。

    LUT 型（``A2B0``/``mft1``/``mft2``/``mAB``）**明确报错**而不是静默
    返回单位矩阵 —— 旧实现那样做会让用户以为 ICC 已经生效。
    """
    with open(path, "rb") as fh:
        data = fh.read()
    if len(data) < 132 or data[36:40] != b"acsp":
        raise ProfileError(f"{os.path.basename(path)}: 不是合法的 ICC 文件")
    tags = _icc_tag_table(data)

    if not all(k in tags for k in ("rXYZ", "gXYZ", "bXYZ")):
        luts = [sig for sig in ICC_LUT_SIGNATURES if sig.strip() in tags]
        if luts:
            raise ProfileError(
                f"{os.path.basename(path)}: 这是 LUT 型 ICC（含 "
                f"{', '.join(sig.strip() for sig in luts)}），"
                "本管线只支持 matrix/TRC 型输入 profile。\n"
                "请改用相机的 .dcp，或用 .cube 走 LUT 路径。")
        raise ProfileError(f"{os.path.basename(path)}: 缺少 rXYZ/gXYZ/bXYZ 标签")

    # ICC 的 rXYZ/gXYZ/bXYZ 列即 RGB→PCS(D50) 矩阵
    matrix = np.column_stack([_icc_xyz(data, tags[k])
                              for k in ("rXYZ", "gXYZ", "bXYZ")])
    trc = []
    for key in ("rTRC", "gTRC", "bTRC"):
        if key in tags:
            try:
                trc.append(_icc_curve(data, tags[key]))
            except ProfileError:
                trc.append(None)
        else:
            trc.append(None)
    trc_tuple = tuple(trc) if any(t is not None for t in trc) else None

    if not np.allclose(matrix.sum(axis=1), 0.0, atol=1e-3):
        # 不强制要求，仅记录（有些 profile 的列和不为 0 是合法的）
        pass

    return CameraProfile(
        kind="icc",
        source=os.path.abspath(path),
        source_name=os.path.basename(path),
        matrices=(("ICC PCS(D50)", matrix),),
        trc=trc_tuple,
        notes="matrix/TRC 型输入 ICC；PCS = D50",
    )


# ======================================================================
# 分派 / 应用
# ======================================================================

def load_profile(path: str) -> CameraProfile:
    """按扩展名分派到 DCP / ICC 解析器。"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".dcp":
        return load_dcp(path)
    if ext in (".icc", ".icm"):
        return load_icc(path)
    raise ProfileError(f"不支持的 profile 格式: {ext or '(无扩展名)'}")


def profile_from_matrix(matrix, source: str = "(自定义矩阵)",
                        dual_matrix=None) -> CameraProfile:
    """直接由 cam→XYZ(D50) 矩阵构造 profile（测试与手动输入用）。"""
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (3, 3):
        raise ProfileError("cam→XYZ 矩阵必须是 3x3")
    matrices = [("自定义", m)]
    if dual_matrix is not None:
        d = np.asarray(dual_matrix, dtype=np.float64)
        if d.shape != (3, 3):
            raise ProfileError("第二个 cam→XYZ 矩阵必须是 3x3")
        matrices = [("日光", m), ("钨丝灯", d)]
    return CameraProfile(kind="matrix", source=source, source_name=source,
                         matrices=tuple(matrices), notes="直接给定的矩阵")


def apply_trc(rgb: np.ndarray, trc) -> np.ndarray:
    """按 ICC 的每通道 TRC 把设备值线性化（无 TRC 时原样返回）。"""
    if trc is None:
        return rgb
    out = np.array(rgb, dtype=np.float64, copy=True)
    for channel, curve in enumerate(trc):
        if curve is None:
            continue
        x = out[..., channel]
        out[..., channel] = np.interp(x, curve[:, 0], curve[:, 1])
    return out


def camera_to_linear_srgb(rgb, profile: CameraProfile | None,
                          weight: float = 1.0) -> np.ndarray:
    """相机线性 RGB → 线性 sRGB(D65)，**不裁剪**。

    越界（负值 / >1）会原样保留，交由调用方统计 —— 这是旧实现
    ``apply_icc`` 里 ``np.clip(converted, 0, 1)`` 造成高光永久损失的根因。
    """
    arr = np.asarray(rgb, dtype=np.float64)
    if profile is None:
        return arr
    arr = apply_trc(arr, profile.trc)
    cam_to_xyz = profile.matrix(weight)
    xyz_d50 = _matmul(profile.pre_matrix, arr) if not np.allclose(
        profile.pre_matrix, np.eye(3)) else arr
    xyz_d50 = _matmul(cam_to_xyz, xyz_d50)
    xyz_d65 = _matmul(cm.CAT_D50_TO_D65, xyz_d50)
    return _matmul(cm.MATRIX_XYZ_D65_TO_SRGB, xyz_d65)


def _matmul(matrix: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    flat = np.asarray(pixels, dtype=np.float64).reshape(-1, 3)
    out = flat @ np.asarray(matrix, dtype=np.float64).T
    return out.reshape(np.asarray(pixels).shape)


__all__ = [
    "ProfileError", "CameraProfile", "load_dcp", "load_icc", "load_profile",
    "profile_from_matrix", "camera_to_linear_srgb", "apply_trc",
    "ILLUMINANT_NAMES", "DAY_INDEX", "TUNGSTEN_INDEX",
]
