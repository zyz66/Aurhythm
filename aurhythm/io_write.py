"""图像写出：基线 TIFF（16/32-bit）、真 DPX 10-bit、EXR、provenance。

为什么自己写 TIFF
-----------------
``tifffile`` 不在核心依赖里，而 **Pillow 无法写 16-bit / float32 的 RGB
TIFF**（实测 ``TypeError: Cannot handle this data type``）。为了保证
「TIFF16/TIFF32 导出」在没有额外依赖的环境里也能工作并**可被自动化验证**，
这里实现一个最小但严格的基线 TIFF 写出器（小端、单 strip、无压缩、
SampleFormat 1=uint / 3=IEEE float），并配一个同规格的读取器供测试与
``info`` 命令使用。装了 ``tifffile`` 时优先用 ``tifffile``。

DPX 10-bit（SMPTE 268M）
------------------------
真正按规范写出（旧实现只是把文件改名成 ``.tif`` 存 TIFF）：
big-endian ``SDPX``、2048 字节头区（这里用 8192 偏移，留出用户数据区）、
ImageElement descriptor 50 (RGB)、bit size 10、packing 1（filled method A：
10-bit 值放在 32-bit 字的**高位**）、transfer 2（printing density）。
每行 3 个 32-bit 字/像素，天然 4 字节对齐，不需要行填充。

provenance
----------
TIFF 写进 ``ImageDescription``；DPX/EXR 另写同名 ``.json`` sidecar
（DPX 头里没有可靠的自由文本区，硬塞进去会破坏兼容性）。
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import struct

import numpy as np

from .output import (RANGE_FULL, RANGE_LEGAL, full_scale, legal_bounds,
                     scale_to_int)

#: 支持的导出格式标识
SUPPORTED_FORMATS = ("tiff16", "tiff32", "dpx10", "exr16", "exr32")

#: DPX 头部各段偏移（SMPTE 268M）
DPX_MAGIC = b"SDPX"
DPX_VERSION = b"V2.0"
DPX_HEADER_SIZE = 1664
DPX_DATA_OFFSET = 8192
DPX_ELEMENT_OFFSET = 780
DPX_DESCRIPTOR_RGB = 50
#: 见 aurhythm/output.py 的说明：DPX 字段代码按常见实现映射，
#: 且**全部可由 OutputTransform 覆盖**（不再写死在写入路径里）。
DPX_TRANSFER_PRINTING_DENSITY = 1
DPX_COLORIMETRIC_USER_DEFINED = 0
DPX_PACKING_FILLED_A = 1
DPX_MAX_CODE = 1023.0


class WriteError(Exception):
    """写出失败（消息面向用户）。"""


# ======================================================================
# 通用
# ======================================================================

def _as_hwc(data) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise WriteError(f"导出数据必须是 (H,W,3)，收到 {arr.shape}")
    return arr


def to_uint16(data, range_mode: str = RANGE_FULL) -> np.ndarray:
    """[0,1] → 16-bit。``range_mode='legal'`` 时映射到视频合法范围
    （4096..60160），否则用满量程。"""
    return scale_to_int(data, 16, range_mode).astype(np.uint16)


def to_float32(data) -> np.ndarray:
    return np.asarray(data, dtype=np.float32)


def to_codes(data, range_mode: str = RANGE_FULL) -> np.ndarray:
    """[0,1] → 10-bit code。``range_mode='legal'`` 时映射到 64..940。"""
    return scale_to_int(data, 10, range_mode).astype(np.uint16)


def _sidecar_path(path: str) -> str:
    return os.path.splitext(path)[0] + ".json"


def write_provenance(path: str, metadata: dict | None, embedded: bool):
    """写 provenance：``embedded=True`` 由调用方嵌入文件内，否则写 sidecar。"""
    if not metadata:
        return None
    payload = json.dumps(metadata, ensure_ascii=False, indent=2,
                         default=str).encode("utf-8")
    if embedded:
        return payload
    sidecar = _sidecar_path(path)
    with open(sidecar, "wb") as fh:
        fh.write(payload)
    return sidecar


# ======================================================================
# 基线 TIFF
# ======================================================================

_TIFF_TYPES = {"SHORT": 3, "LONG": 4, "ASCII": 2}
_TIFF_TYPE_SIZES = {2: 1, 3: 2, 4: 4}


def _tiff_tag(tag: int, type_name: str, count: int, values) -> tuple:
    return tag, _TIFF_TYPES[type_name], count, values


def write_tiff(path: str, data, bit_depth: int = 16, metadata: dict | None = None,
               prefer_tifffile: bool = True, range_mode: str = RANGE_FULL) -> str:
    """写 RGB TIFF（``bit_depth`` 为 16 或 32）。

    有 ``tifffile`` 时优先使用（多 strip、更完整的元数据）；
    否则退回内置基线写出器。
    """
    arr = _as_hwc(data)
    if bit_depth not in (16, 32):
        raise WriteError("TIFF 位深只支持 16 或 32")

    description = None
    if metadata:
        description = json.dumps(metadata, ensure_ascii=False, default=str)

    if prefer_tifffile:
        try:
            import tifffile
        except ImportError:
            tifffile = None
        if tifffile is not None:
            payload = (to_uint16(arr, range_mode) if bit_depth == 16
                       else to_float32(arr))
            tifffile.imwrite(path, payload, photometric="rgb",
                             description=description or "")
            return path

    return _write_tiff_baseline(path, arr, bit_depth, description, range_mode)


def _write_tiff_baseline(path: str, arr: np.ndarray, bit_depth: int,
                         description: str | None,
                         range_mode: str = RANGE_FULL) -> str:
    height, width = arr.shape[:2]
    if bit_depth == 16:
        payload = to_uint16(arr, range_mode).astype("<u2")
        sample_format = 1
        bytes_per_sample = 2
    else:
        payload = np.ascontiguousarray(arr, dtype="<f4")
        sample_format = 3
        bytes_per_sample = 4

    tags = [
        _tiff_tag(256, "LONG", 1, width),
        _tiff_tag(257, "LONG", 1, height),
        _tiff_tag(258, "SHORT", 3, [bit_depth] * 3),
        _tiff_tag(259, "SHORT", 1, 1),
        _tiff_tag(262, "SHORT", 1, 2),
        _tiff_tag(273, "LONG", 1, 0),          # 占位，写 IFD 时替换为真实偏移
        _tiff_tag(277, "SHORT", 1, 3),
        _tiff_tag(278, "LONG", 1, height),
        _tiff_tag(279, "LONG", 1,
                  width * height * 3 * bytes_per_sample),
        _tiff_tag(284, "SHORT", 1, 1),
        _tiff_tag(339, "SHORT", 3, [sample_format] * 3),
    ]
    if description:
        raw = description.encode("utf-8") + b"\x00"
        tags.append(_tiff_tag(270, "ASCII", len(raw), raw))
    tags.append(_tiff_tag(305, "ASCII", 9, b"Aurhythm"))
    tags.sort(key=lambda t: t[0])

    ifd_offset = 8
    ifd_size = 2 + 12 * len(tags) + 4
    extra_offset = ifd_offset + ifd_size
    extra = bytearray()
    entries = []

    for tag, type_id, count, values in tags:
        size = _TIFF_TYPE_SIZES[type_id] * count
        if type_id == 2:                       # ASCII
            blob = bytes(values)
        elif type_id == 3:
            seq = values if isinstance(values, (list, tuple)) else [values]
            blob = struct.pack("<" + "H" * len(seq), *seq)
        else:
            seq = values if isinstance(values, (list, tuple)) else [values]
            blob = struct.pack("<" + "I" * len(seq), *seq)
        if size <= 4:
            entry_value = blob.ljust(4, b"\x00")
        else:
            entry_value = struct.pack("<I", extra_offset + len(extra))
            extra.extend(blob)
            if len(extra) % 2:
                extra.append(0)
        entries.append((tag, type_id, count, entry_value))

    pixel_offset = extra_offset + len(extra)
    pixel_bytes = payload.tobytes()

    with open(path, "wb") as fh:
        fh.write(b"II")
        fh.write(struct.pack("<H", 42))
        fh.write(struct.pack("<I", ifd_offset))
        fh.write(struct.pack("<H", len(entries)))
        for tag, type_id, count, value in entries:
            if tag == 273:                     # StripOffsets 需要最终的像素偏移
                value = struct.pack("<I", pixel_offset)
            fh.write(struct.pack("<HHI", tag, type_id, count))
            fh.write(value)
        fh.write(struct.pack("<I", 0))
        fh.write(bytes(extra))
        fh.write(pixel_bytes)
    return path


def read_tiff(path: str):
    """读取**未压缩** TIFF（小端/大端、单/多 strip、PlanarConfiguration=1）。

    支持 BitsPerSample 8/16/32 与 SampleFormat 1(uint)/2(int)/3(float)，
    覆盖扫描仪与图像软件最常产出的那一类文件。返回 ``(array, description)``；
    未压缩之外的压缩方式会明确报错，而不是悄悄读错。

    这既是内置写出器的配套读取器（测试与 ``info`` 命令），
    也是 ``io_read`` 在缺少 ``tifffile`` / Pillow 读不出 16-bit RGB 时的兜底。
    """
    with open(path, "rb") as fh:
        blob = fh.read()
    if blob[:2] == b"II":
        endian = "<"
    elif blob[:2] == b"MM":
        endian = ">"
    else:
        raise WriteError("不是合法的 TIFF（缺少 II/MM 字节序标记）")
    (magic,) = struct.unpack(endian + "H", blob[2:4])
    if magic != 42:
        raise WriteError("TIFF 魔数不正确（可能是 BigTIFF，暂不支持）")
    (ifd_offset,) = struct.unpack(endian + "I", blob[4:8])
    (count,) = struct.unpack(endian + "H", blob[ifd_offset:ifd_offset + 2])
    sizes = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 11: 4, 12: 8}
    tags = {}
    for i in range(count):
        base = ifd_offset + 2 + 12 * i
        tag, type_id, cnt = struct.unpack(endian + "HHI", blob[base:base + 8])
        raw = blob[base + 8:base + 12]
        size = sizes.get(type_id, 1) * cnt
        if size > 4:
            (offset,) = struct.unpack(endian + "I", raw)
            raw = blob[offset:offset + size]
        tags[tag] = (type_id, cnt, raw)

    def numbers(tag, default=None):
        if tag not in tags:
            return default
        type_id, cnt, raw = tags[tag]
        fmt = {1: "B", 3: "H", 4: "I", 11: "f", 12: "d"}[type_id]
        return list(struct.unpack(endian + fmt * cnt,
                                  raw[:cnt * sizes[type_id]]))

    compression = (numbers(259) or [1])[0]
    if compression != 1:
        raise WriteError(f"只支持未压缩 TIFF（Compression={compression}）")
    planar = (numbers(284) or [1])[0]
    if planar != 1:
        raise WriteError("只支持 PlanarConfiguration=1（交错）")

    width = numbers(256)[0]
    height = numbers(257)[0]
    bits_list = numbers(258) or [8]
    bit_depth = bits_list[0]
    samples = (numbers(277) or [1])[0]
    sample_format = (numbers(339) or [1])[0]
    rows_per_strip = (numbers(278) or [height])[0] or height
    offsets = numbers(273) or []
    counts = numbers(279) or []

    if sample_format == 3 and bit_depth == 32:
        dtype = np.dtype(endian + "f4")
    elif sample_format == 3 and bit_depth == 16:
        dtype = np.dtype(endian + "f2")
    elif bit_depth == 8:
        dtype = np.dtype("u1")
    elif bit_depth == 16 and sample_format == 2:
        dtype = np.dtype(endian + "i2")
    elif bit_depth == 16:
        dtype = np.dtype(endian + "u2")
    elif bit_depth == 32 and sample_format == 2:
        dtype = np.dtype(endian + "i4")
    elif bit_depth == 32:
        dtype = np.dtype(endian + "u4")
    else:
        raise WriteError(f"不支持的位深/SampleFormat: {bit_depth}/{sample_format}")

    if not offsets:
        raise WriteError("TIFF 缺少 StripOffsets")
    chunks = []
    for index, offset in enumerate(offsets):
        byte_count = (counts[index] if index < len(counts)
                      else width * rows_per_strip * samples * (bit_depth // 8))
        chunks.append(np.frombuffer(blob[offset:offset + byte_count], dtype=dtype))
    pixels = np.concatenate(chunks)
    expected = width * height * samples
    if pixels.size < expected:
        raise WriteError(f"数据不足：需要 {expected} 个采样，只有 {pixels.size}")
    array = pixels[:expected].reshape(height, width, samples)
    if samples == 1:
        array = np.repeat(array, 3, axis=2)

    description = None
    if 270 in tags:
        description = tags[270][2].split(b"\x00")[0].decode("utf-8", "replace")
    return array, description


#: 兼容旧名字
read_tiff_baseline = read_tiff


# ======================================================================
# DPX 10-bit
# ======================================================================

def write_dpx10(path: str, data, metadata: dict | None = None,
                transfer: int | None = None,
                colorimetric: int | None = None,
                range_mode: str = RANGE_FULL) -> str:
    """写 10-bit RGB DPX（SMPTE 268M，packing 1 = filled method A）。

    ``transfer`` / ``colorimetric`` 由 :class:`aurhythm.output.OutputTransform`
    按输出空间给出（对数 → logarithmic / printing density；Rec.709 → ITU-R 709）。
    传 ``None`` 时退回 printing density / user-defined。
    """
    arr = _as_hwc(data)
    height, width = arr.shape[:2]
    codes = to_codes(arr, range_mode).astype(np.uint32)
    transfer_code = (DPX_TRANSFER_PRINTING_DENSITY if transfer is None
                     else int(transfer))
    colorimetric_code = (DPX_COLORIMETRIC_USER_DEFINED
                         if colorimetric is None else int(colorimetric))

    line_words = width * 3
    line_bytes = line_words * 4
    payload = np.zeros((height, line_words), dtype=">u4")
    payload[:, 0::3] = codes[:, :, 0] << 22
    payload[:, 1::3] = codes[:, :, 1] << 22
    payload[:, 2::3] = codes[:, :, 2] << 22
    image_bytes = payload.tobytes()
    total_size = DPX_DATA_OFFSET + len(image_bytes)

    header = bytearray(DPX_DATA_OFFSET)
    header[0:4] = DPX_MAGIC
    struct.pack_into(">I", header, 4, DPX_DATA_OFFSET)
    header[8:8 + len(DPX_VERSION)] = DPX_VERSION
    struct.pack_into(">I", header, 16, total_size)
    struct.pack_into(">I", header, 20, 1)
    struct.pack_into(">I", header, 24, DPX_HEADER_SIZE)     # generic header size
    struct.pack_into(">I", header, 28, 384)                 # industry header size
    struct.pack_into(">I", header, 32, 0)                   # user data size
    names = os.path.basename(path).encode("ascii", "replace")[:99]
    header[36:36 + len(names)] = names
    now = _dt.datetime.now()
    struct.pack_into(">HHHHHHI", header, 136, now.year, now.month, now.day,
                     now.hour, now.minute, now.second, now.microsecond // 1000)
    creator = b"Aurhythm 5.0"[:99]
    header[160:160 + len(creator)] = creator
    project = b"Aurhythm film calibration"[:199]
    header[260:260 + len(project)] = project

    base = 768
    struct.pack_into(">H", header, base, 0)                 # orientation
    struct.pack_into(">H", header, base + 2, 1)             # number of elements
    struct.pack_into(">I", header, base + 4, width)
    struct.pack_into(">I", header, base + 8, height)
    element = base + 12
    struct.pack_into(">I", header, element + 0, 0)                        # data sign
    struct.pack_into(">I", header, element + 4, 0)                        # ref low data
    struct.pack_into(">f", header, element + 8, 0.0)                      # ref low qty
    struct.pack_into(">I", header, element + 12, int(DPX_MAX_CODE))       # ref high data
    struct.pack_into(">f", header, element + 16, 2.048)                   # ref high qty
    header[element + 20] = DPX_DESCRIPTOR_RGB
    header[element + 21] = transfer_code
    header[element + 22] = colorimetric_code
    header[element + 23] = 10                                             # bit size
    struct.pack_into(">H", header, element + 24, DPX_PACKING_FILLED_A)
    struct.pack_into(">H", header, element + 26, 0)                       # encoding
    struct.pack_into(">I", header, element + 28, DPX_DATA_OFFSET)         # data offset
    struct.pack_into(">I", header, element + 32, 0)                       # EOL padding
    struct.pack_into(">I", header, element + 36, 0)                       # EOI padding
    label = b"Aurhythm Cineon 10-bit"[:31]
    header[element + 40:element + 40 + len(label)] = label

    with open(path, "wb") as fh:
        fh.write(bytes(header))
        fh.write(image_bytes)
    payload = dict(metadata or {})
    payload["dpx_transfer"] = transfer_code
    payload["dpx_colorimetric"] = colorimetric_code
    payload["range"] = range_mode
    write_provenance(path, payload or None, embedded=False)
    return path


def read_dpx10(path: str):
    """按规范读回 :func:`write_dpx10` 写出的 DPX，返回 ``(codes, header)``。"""
    with open(path, "rb") as fh:
        blob = fh.read()
    if blob[:4] != DPX_MAGIC:
        raise WriteError("不是 big-endian SDPX 文件")
    (data_offset,) = struct.unpack(">I", blob[4:8])
    (total_size,) = struct.unpack(">I", blob[16:20])
    width = struct.unpack(">I", blob[772:776])[0]
    height = struct.unpack(">I", blob[776:780])[0]
    element = DPX_ELEMENT_OFFSET
    header = {
        "data_offset": data_offset,
        "total_size": total_size,
        "version": blob[8:16].rstrip(b"\x00").decode("ascii", "replace"),
        "width": width,
        "height": height,
        "descriptor": blob[element + 20],
        "transfer": blob[element + 21],
        "colorimetric": blob[element + 22],
        "bit_size": blob[element + 23],
        "packing": struct.unpack(">H", blob[element + 24:element + 26])[0],
        "ref_high": struct.unpack(">I", blob[element + 12:element + 16])[0],
    }
    words = np.frombuffer(blob[data_offset:data_offset + width * height * 3 * 4],
                          dtype=">u4").reshape(height, width, 3)
    codes = (words >> 22).astype(np.uint16)
    return codes, header


# ======================================================================
# EXR
# ======================================================================

def write_exr(path: str, data, bit_depth: int = 32,
              metadata: dict | None = None) -> str:
    """写 EXR（16 → float16，32 → float32）。

    EXR 写出依赖 ``imageio`` 的 freeimage 插件或 ``OpenEXR``；两者都不可用时
    抛出**可读**错误（旧实现在导出时静默失败）。
    """
    arr = _as_hwc(data)
    payload = (np.asarray(arr, dtype=np.float16) if bit_depth == 16
               else np.asarray(arr, dtype=np.float32))
    errors = []
    try:
        import imageio.v2 as imageio
        imageio.imwrite(path, payload, format="EXR")
        write_provenance(path, metadata, embedded=False)
        return path
    except ImportError as exc:
        errors.append(f"imageio 不可用 ({exc})")
    except Exception as exc:                        # noqa: BLE001
        errors.append(f"imageio 写 EXR 失败 ({exc})")
    try:
        import OpenEXR                              # noqa: F401
        import Imath                                # noqa: F401
        raise WriteError("检测到 OpenEXR，但本版本尚未接入其写出路径；"
                         "请安装 imageio 以启用 EXR 导出")
    except ImportError:
        errors.append("OpenEXR 不可用")
    raise WriteError("无法写出 EXR：" + "；".join(errors)
                     + "。请安装 imageio（含 freeimage 插件）后重试。")


# ======================================================================
# 分派
# ======================================================================

def write_png(path: str, data) -> str:
    """写 8-bit PNG（给"对照正片"预览用；PIL 延迟导入）。"""
    from PIL import Image

    arr = np.asarray(data)
    if arr.dtype != np.uint8:
        arr = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)
    return path


def write_image(path: str, data, fmt: str = "tiff16",
                metadata: dict | None = None, range_mode: str = RANGE_FULL,
                dpx_transfer: int | None = None,
                dpx_colorimetric: int | None = None) -> str:
    """按 ``fmt`` 写出（``SUPPORTED_FORMATS`` 之一）。

    ``range_mode='legal'`` 时按视频合法范围映射（10-bit 64..940）。
    """
    if fmt not in SUPPORTED_FORMATS:
        raise WriteError(f"不支持的导出格式 {fmt!r}，可选：{SUPPORTED_FORMATS}")
    if range_mode not in (RANGE_FULL, RANGE_LEGAL):
        raise WriteError(f"不支持的范围 {range_mode!r}")
    arr = _as_hwc(data)
    if fmt == "tiff16":
        return write_tiff(path, arr, 16, metadata, range_mode=range_mode)
    if fmt == "tiff32":
        return write_tiff(path, arr, 32, metadata, range_mode=range_mode)
    if fmt == "dpx10":
        return write_dpx10(path, arr, metadata, transfer=dpx_transfer,
                           colorimetric=dpx_colorimetric, range_mode=range_mode)
    if fmt == "exr16":
        return write_exr(path, arr, 16, metadata)
    return write_exr(path, arr, 32, metadata)


def extension_for(fmt: str) -> str:
    return {"tiff16": "tif", "tiff32": "tif", "dpx10": "dpx",
            "exr16": "exr", "exr32": "exr"}[fmt]


def describe_format(fmt: str) -> str:
    return {
        "tiff16": "TIFF 16-bit (Cineon code → 0..65535)",
        "tiff32": "TIFF 32-bit float",
        "dpx10": "DPX 10-bit (SMPTE 268M, printing density)",
        "exr16": "EXR half float",
        "exr32": "EXR 32-bit float",
    }.get(fmt, fmt)


__all__ = [
    "WriteError", "SUPPORTED_FORMATS", "to_uint16", "to_float32", "to_codes",
    "write_tiff", "read_tiff", "read_tiff_baseline", "write_dpx10", "read_dpx10",
    "write_png",
    "legal_bounds", "full_scale",
    "write_exr", "write_image", "extension_for", "describe_format",
    "write_provenance",
]
