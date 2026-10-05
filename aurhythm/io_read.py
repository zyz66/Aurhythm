"""图像读取：RAW（懒加载 rawpy）、TIFF（8/16/32）、PNG/JPEG。

设计约定
--------
* **RAW** 一律按 ``gamma=(1,1)``、``no_auto_bright=True``、
  ``output_color=raw`` 解码 —— 拿到的就是**相机原生线性**数据，
  与后续的相机 profile 链路配套。``use_camera_wb`` 默认关闭，
  白平衡由 :meth:`pipeline.set_wb` / ``AsShotNeutral`` 显式控制。
* **TIFF 16/32** 默认视为**线性**（扫描仪/浮点中间片的常见约定），
  **PNG / 8-bit / JPEG** 默认视为 **sRGB 编码**并线性化。
  两种情况都可以用 ``assume_linear`` 覆盖，避免"猜"。
* ``rawpy`` 是**懒加载**的：没有它时 GUI 仍可启动、TIFF/PNG 仍可用，
  只有真的去读 RAW 才会报出可读错误（旧实现在模块顶层 import rawpy，
  缺依赖时直接崩溃，R12）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from . import colorimetry as cm

#: 常见 RAW 扩展名
RAW_EXTENSIONS = (".nef", ".nrw", ".dng", ".cr2", ".cr3", ".crw", ".arw",
                  ".srf", ".sr2", ".raf", ".orf", ".rw2", ".raw", ".pef",
                  ".srw", ".kdc", ".dcr", ".erf", ".mrw", ".3fr", ".fff",
                  ".iiq", ".rwl", ".x3f")
#: 位图扩展名
BITMAP_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")


class ReadError(Exception):
    """读取失败（消息面向用户）。"""


@dataclass
class ImageInfo:
    path: str
    name: str
    width: int
    height: int
    channels: int
    source_bits: int
    is_raw: bool
    assumed_linear: bool
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "path": self.path, "name": self.name,
            "width": self.width, "height": self.height,
            "channels": self.channels, "source_bits": self.source_bits,
            "is_raw": self.is_raw, "assumed_linear": self.assumed_linear,
            "note": self.note,
        }


def is_raw(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in RAW_EXTENSIONS


def is_bitmap(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in BITMAP_EXTENSIONS


# ======================================================================
# RAW
# ======================================================================

def load_raw(path: str, use_camera_wb: bool = False) -> np.ndarray:
    """解码 RAW → 相机原生线性 float32（[0,1]，16-bit 满量程为 1.0）。"""
    try:
        import rawpy
    except ImportError as exc:
        raise ReadError(
            "读取 RAW 需要 rawpy：pip install rawpy\n"
            "（TIFF/PNG 输入不需要 rawpy）") from exc
    try:
        with rawpy.imread(path) as raw:
            rgb = raw.postprocess(gamma=(1, 1), no_auto_bright=True,
                                  output_bps=16, use_camera_wb=use_camera_wb,
                                  output_color=rawpy.ColorSpace.raw)
    except Exception as exc:                            # noqa: BLE001
        raise ReadError(f"{os.path.basename(path)}: RAW 解码失败 ({exc})") from exc

    arr = rgb.astype(np.float32) / 65535.0
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return np.ascontiguousarray(arr[..., :3])


def read_as_shot_neutral(path: str):
    """尽力读取 ``AsShotNeutral``（返回 ``(neutral, note)`` 或 ``(None, 原因)``）。

    不同 rawpy 版本暴露的元数据字段不一致，因此这里**尽力而为**并如实
    报告失败原因，绝不编造白平衡值。
    """
    try:
        import rawpy
    except ImportError:
        return None, "缺少 rawpy，无法读取 AsShotNeutral"
    try:
        with rawpy.imread(path) as raw:
            candidates = []
            for attr in ("camera_whitebalance", "daylight_whitebalance"):
                values = getattr(raw, attr, None)
                if values:
                    candidates.append(np.asarray(values, dtype=np.float64)[:3])
            for attr in ("rgb_xyz_matrix", "color_matrix"):
                if getattr(raw, attr, None) is not None:
                    break
            if not candidates:
                return None, "该 RAW 未提供白平衡元数据"
            wb = candidates[0]
            if np.any(wb <= 0):
                return None, "白平衡元数据非法（含非正值）"
            neutral = wb[1] / wb          # 归一化到绿通道
            return neutral, ""
    except Exception as exc:                            # noqa: BLE001
        return None, f"读取白平衡失败: {exc}"


# ======================================================================
# 位图
# ======================================================================

def load_tiff(path: str, assume_linear: bool | None = None):
    """读 TIFF，返回 ``(float32 线性 RGB, ImageInfo)``。"""
    array = None
    note = ""
    try:
        import tifffile
        array = np.asarray(tifffile.imread(path))
        note = "tifffile"
    except ImportError:
        pass
    except Exception as exc:                            # noqa: BLE001
        note = f"tifffile 读取失败: {exc}"

    if array is None:
        from .io_write import WriteError, read_tiff
        try:
            array, _description = read_tiff(path)
            note = note or "内置 TIFF 读取器"
        except WriteError as exc:
            array = _load_with_pillow(path)
            note = f"Pillow 读取（{exc}）"

    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    array = array[..., :3]
    bits = int(array.dtype.itemsize) * 8
    linear_by_default = bits >= 16
    linear = linear_by_default if assume_linear is None else bool(assume_linear)

    if np.issubdtype(array.dtype, np.integer):
        data = array.astype(np.float32) / float(np.iinfo(array.dtype).max)
    else:
        data = array.astype(np.float32)
    if not linear:
        data = cm.srgb_decode(data).astype(np.float32)
    info = ImageInfo(path=os.path.abspath(path), name=os.path.basename(path),
                     width=array.shape[1], height=array.shape[0],
                     channels=3, source_bits=bits, is_raw=False,
                     assumed_linear=linear,
                     note=f"{note}；" + ("按线性" if linear else "按 sRGB 解码"))
    return data, info


def load_bitmap(path: str, assume_linear: bool | None = None):
    """用 Pillow 读 PNG/JPEG/BMP，返回 ``(float32 线性 RGB, ImageInfo)``。"""
    try:
        from PIL import Image
    except ImportError as exc:
        raise ReadError("读取位图需要 Pillow：pip install Pillow") from exc
    try:
        with Image.open(path) as img:
            img = img.convert("RGB") if img.mode not in ("RGB", "L") else img
            array = np.asarray(img)
    except Exception as exc:                            # noqa: BLE001
        raise ReadError(f"{os.path.basename(path)}: 读取失败 ({exc})") from exc

    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    bits = int(array.dtype.itemsize) * 8
    linear = False if assume_linear is None else bool(assume_linear)
    data = array.astype(np.float32) / float(np.iinfo(array.dtype).max)
    if not linear:
        data = cm.srgb_decode(data).astype(np.float32)
    info = ImageInfo(path=os.path.abspath(path), name=os.path.basename(path),
                     width=array.shape[1], height=array.shape[0], channels=3,
                     source_bits=bits, is_raw=False, assumed_linear=linear,
                     note="Pillow；" + ("按线性" if linear else "按 sRGB 解码"))
    return data, info


def _load_with_pillow(path: str) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ReadError("需要 tifffile 或 Pillow 才能读取该 TIFF") from exc
    with Image.open(path) as img:
        return np.asarray(img)


# ======================================================================
# 分派
# ======================================================================

def load_any(path: str, assume_linear: bool | None = None, **raw_kwargs):
    """按扩展名读入任意支持的图像，返回 ``(float32 线性 RGB, ImageInfo)``。"""
    if not os.path.exists(path):
        raise ReadError(f"文件不存在: {path}")
    if is_raw(path):
        data = load_raw(path, **raw_kwargs)
        info = ImageInfo(path=os.path.abspath(path), name=os.path.basename(path),
                         width=data.shape[1], height=data.shape[0], channels=3,
                         source_bits=16, is_raw=True, assumed_linear=True,
                         note="rawpy；相机原生线性（未做白平衡）")
        return data, info
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        return load_tiff(path, assume_linear=assume_linear)
    if ext in BITMAP_EXTENSIONS:
        return load_bitmap(path, assume_linear=assume_linear)
    # 未知扩展名：先试位图，失败再试 RAW
    try:
        return load_bitmap(path, assume_linear=assume_linear)
    except ReadError:
        data = load_raw(path)
        info = ImageInfo(path=os.path.abspath(path), name=os.path.basename(path),
                         width=data.shape[1], height=data.shape[0], channels=3,
                         source_bits=16, is_raw=True, assumed_linear=True,
                         note="按 RAW 解码（扩展名未知）")
        return data, info


def probe(path: str) -> dict:
    """只读取元信息（尽量不解码整幅图）。"""
    if is_raw(path):
        try:
            import rawpy
        except ImportError as exc:
            raise ReadError("探测 RAW 需要 rawpy") from exc
        with rawpy.imread(path) as raw:
            sizes = raw.sizes
            return ImageInfo(path=os.path.abspath(path),
                             name=os.path.basename(path),
                             width=int(sizes.width), height=int(sizes.height),
                             channels=3, source_bits=int(sizes.bits),
                             is_raw=True, assumed_linear=True,
                             note=f"相机: {getattr(raw, 'camera_manufacturer', '?')} "
                                  f"{getattr(raw, 'camera_model', '?')}").to_dict()
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        from .io_write import read_tiff
        try:
            array, description = read_tiff(path)
            return ImageInfo(path=os.path.abspath(path),
                             name=os.path.basename(path),
                             width=array.shape[1], height=array.shape[0],
                             channels=3,
                             source_bits=int(array.dtype.itemsize) * 8,
                             is_raw=False, assumed_linear=True,
                             note=(description or "")[:200]).to_dict()
        except Exception:                               # noqa: BLE001
            pass
    from PIL import Image
    with Image.open(path) as img:
        bits = 16 if img.mode.startswith("I;16") else 8
        return ImageInfo(path=os.path.abspath(path),
                         name=os.path.basename(path),
                         width=img.width, height=img.height, channels=3,
                         source_bits=bits, is_raw=False, assumed_linear=False,
                         note=f"Pillow 模式 {img.mode}").to_dict()


__all__ = [
    "ReadError", "ImageInfo", "RAW_EXTENSIONS", "BITMAP_EXTENSIONS",
    "is_raw", "is_bitmap", "load_raw", "load_tiff", "load_bitmap",
    "load_any", "probe", "read_as_shot_neutral",
]
