"""输出变换：**对数直通** vs **经色彩还原 LUT 到显示色彩空间**。

为什么需要这一层
----------------
「套了 LUT」之后，导出的东西**不再是 log 素材**：

* 未套 LUT → 输出是对数域（Cineon / ARRI LogC3），是给下游调色用的中间片；
* 套了 ARRI 官网那种 ``LogC3 → Rec.709`` 色彩还原 LUT → 输出是
  **Rec.709 显示域**（已含 BT.709 OETF），是能直接看的成片。

两者的容器标注、DPX 的 transfer/colorimetric 字段、以及**预览该怎么显示**
都不一样。旧实现把 LUT 结果照旧按 log 去标注与显示，等于在 TIFF 里塞了
一份 Rec.709 却写着 "Cineon" —— 下游会再套一次 log 解码，画面全错。

关于 DPX 字段代码（如实说明）
----------------------------
``SPACE_INFO`` 里的 ``dpx_transfer`` / ``dpx_colorimetric`` 按 DPX/SMPTE 268M
的**常见实现**填写。本项目的开发环境无法联网核对规范原文，因此：

1. 它们全部是**可覆盖的具名常量**（CLI ``--dpx-transfer`` / UI 高级项），
   不写死在写入路径里；
2. 实际写入值会记进导出元数据（``dpx_transfer`` / ``dpx_colorimetric``），
   便于在 Resolve/Nuke 里核对；
3. 已知历史遗留：旧代码把 ``2`` 命名为 printing density，但按常见映射
   ``1 = printing density``、``2 = linear``。这里已按后者排列。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

# ---------------- 色彩空间标识 ----------------
SPACE_CINEON = "cineon"
SPACE_LOGC3 = "logc3"
SPACE_REC709 = "rec709"
SPACE_SRGB = "srgb"
SPACE_LINEAR = "linear"

#: DPX/SMPTE 268M transfer characteristic（常见实现映射，可覆盖）
DPX_TRANSFER = {
    "user_defined": 0,
    "printing_density": 1,
    "linear": 2,
    "logarithmic": 3,
    "unspecified_video": 4,
    "smpte_274": 5,
    "itu_r_709": 6,
    "itu_r_601_625": 7,
    "itu_r_601_525": 8,
    "ntsc_composite": 9,
    "pal_composite": 10,
}

#: DPX/SMPTE 268M colorimetric（常见实现映射，可覆盖）
DPX_COLORIMETRIC = {
    "user_defined": 0,
    "printing_density": 1,
    "itu_r_bt709": 2,
    "itu_r_bt601_5": 3,
    "smpte_240m": 4,
    "cie": 5,
}

#: 每个输出空间的属性
SPACE_INFO = {
    SPACE_CINEON: {
        "label": "Cineon（printing density 对数）",
        "log": True,
        "oetf": None,
        "dpx_transfer": DPX_TRANSFER["printing_density"],
        "dpx_colorimetric": DPX_COLORIMETRIC["user_defined"],
        "range_default": "full",
    },
    SPACE_LOGC3: {
        "label": "ARRI LogC3（对数）",
        "log": True,
        "oetf": None,
        "dpx_transfer": DPX_TRANSFER["logarithmic"],
        "dpx_colorimetric": DPX_COLORIMETRIC["user_defined"],
        "range_default": "full",
    },
    SPACE_REC709: {
        "label": "Rec.709（BT.709 OETF，显示域）",
        "log": False,
        "oetf": "bt709",
        "dpx_transfer": DPX_TRANSFER["itu_r_709"],
        "dpx_colorimetric": DPX_COLORIMETRIC["itu_r_bt709"],
        "range_default": "legal",
    },
    SPACE_SRGB: {
        "label": "sRGB（IEC 61966-2-1，显示域）",
        "log": False,
        "oetf": "srgb",
        "dpx_transfer": DPX_TRANSFER["user_defined"],   # DPX 无 sRGB 编码值
        "dpx_colorimetric": DPX_COLORIMETRIC["itu_r_bt709"],
        "range_default": "full",
    },
    SPACE_LINEAR: {
        "label": "线性（BT.709 基色）",
        "log": False,
        "oetf": None,
        "dpx_transfer": DPX_TRANSFER["linear"],
        "dpx_colorimetric": DPX_COLORIMETRIC["itu_r_bt709"],
        "range_default": "full",
    },
}

#: 可选的 LUT 输入空间（LUT 期望喂什么）
LUT_INPUT_SPACES = (SPACE_LOGC3, SPACE_CINEON)
#: 可选的 LUT 目标空间
LUT_TARGET_SPACES = (SPACE_REC709, SPACE_SRGB, SPACE_LINEAR)

MODE_LOG = "log"
MODE_LUT = "lut"
OUTPUT_MODES = (MODE_LOG, MODE_LUT)

RANGE_FULL = "full"
RANGE_LEGAL = "legal"
OUTPUT_RANGES = (RANGE_FULL, RANGE_LEGAL)


@dataclass(frozen=True)
class OutputTransform:
    """导出链的末端定义。

    * ``mode='log'``：直通对数，``space`` 由 ``log_space`` 决定；
    * ``mode='lut'``：先按 ``lut_input_space`` 归一化，套 LUT，
      输出即 ``lut_target_space``（显示域）。
    """

    mode: str = MODE_LOG
    #: 对数模式下的输出空间：'cineon' | 'logc3'
    log_space: str = SPACE_CINEON
    #: LUT 期望的输入空间：'logc3' | 'cineon'
    lut_input_space: str = SPACE_LOGC3
    #: LUT 的目标空间（= 最终输出空间）
    lut_target_space: str = SPACE_REC709
    #: 容器级范围：'full' | 'legal'
    range: str = RANGE_FULL
    #: DPX 字段覆盖（None = 按空间自动）
    dpx_transfer: int | None = None
    dpx_colorimetric: int | None = None
    #: LUT 文件路径（仅记录）
    lut_path: str | None = None
    note: str = ""

    # ---------- 派生 ----------
    @property
    def effective_space(self) -> str:
        """最终写入容器的空间。"""
        return self.log_space if self.mode == MODE_LOG else self.lut_target_space

    @property
    def is_display_referred(self) -> bool:
        return not SPACE_INFO[self.effective_space]["log"]

    @property
    def source_space(self) -> str:
        """喂给 LUT / 容器的**输入**空间（即管线编码出来的东西）。"""
        return self.log_space if self.mode == MODE_LOG else self.lut_input_space

    @property
    def effective_range(self) -> str:
        if self.range in OUTPUT_RANGES:
            return self.range
        return SPACE_INFO[self.effective_space]["range_default"]

    def transfer_code(self) -> int:
        if self.dpx_transfer is not None:
            return int(self.dpx_transfer)
        return int(SPACE_INFO[self.effective_space]["dpx_transfer"])

    def colorimetric_code(self) -> int:
        if self.dpx_colorimetric is not None:
            return int(self.dpx_colorimetric)
        return int(SPACE_INFO[self.effective_space]["dpx_colorimetric"])

    def label(self) -> str:
        space = SPACE_INFO[self.effective_space]["label"]
        if self.mode == MODE_LOG:
            return f"{space}（对数直通，{self.effective_range}）"
        return f"{space}（经 LUT，{self.effective_range}）"

    def sanitize(self) -> "OutputTransform":
        mode = self.mode if self.mode in OUTPUT_MODES else MODE_LOG
        log_space = self.log_space if self.log_space in (SPACE_CINEON, SPACE_LOGC3) \
            else SPACE_CINEON
        lut_in = self.lut_input_space if self.lut_input_space in LUT_INPUT_SPACES \
            else SPACE_LOGC3
        lut_out = self.lut_target_space if self.lut_target_space in LUT_TARGET_SPACES \
            else SPACE_REC709
        rng = self.range if self.range in OUTPUT_RANGES else RANGE_FULL
        return replace(self, mode=mode, log_space=log_space,
                       lut_input_space=lut_in, lut_target_space=lut_out,
                       range=rng)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "log_space": self.log_space,
            "lut_input_space": self.lut_input_space,
            "lut_target_space": self.lut_target_space,
            "range": self.range,
            "dpx_transfer": self.dpx_transfer,
            "dpx_colorimetric": self.dpx_colorimetric,
            "lut_path": self.lut_path,
            "effective_space": self.effective_space,
            "effective_range": self.effective_range,
            "is_display_referred": self.is_display_referred,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data) -> "OutputTransform":
        if not data:
            return cls()
        base = cls()
        out = {}
        for key in ("mode", "log_space", "lut_input_space", "lut_target_space",
                    "range", "lut_path", "note"):
            if data.get(key) is not None:
                out[key] = str(data[key])
        for key in ("dpx_transfer", "dpx_colorimetric"):
            if data.get(key) is not None:
                out[key] = int(data[key])
        return replace(base, **out).sanitize()


def legal_bounds(bits: int) -> tuple:
    """视频合法范围（10-bit 64..940）按位深缩放：12-bit 256..3760、
    16-bit 4096..60160。"""
    if bits < 10:
        raise ValueError("合法范围只对 10-bit 及以上定义")
    shift = bits - 10
    return 64 << shift, 940 << shift


def full_scale(bits: int) -> int:
    return (1 << bits) - 1


def scale_to_int(data, bits: int = 16, range_mode: str = RANGE_FULL):
    """[0,1] → 指定位深的整数，按 full / legal 映射（四舍五入 + 裁剪）。"""
    import numpy as np

    arr = np.clip(np.asarray(data, dtype=np.float64), 0.0, 1.0)
    if range_mode == RANGE_LEGAL:
        lo, hi = legal_bounds(bits)
        scaled = lo + arr * (hi - lo)
    else:
        scaled = arr * full_scale(bits)
    return np.rint(scaled).astype(np.uint32)


__all__ = [
    "SPACE_CINEON", "SPACE_LOGC3", "SPACE_REC709", "SPACE_SRGB", "SPACE_LINEAR",
    "SPACE_INFO", "DPX_TRANSFER", "DPX_COLORIMETRIC",
    "LUT_INPUT_SPACES", "LUT_TARGET_SPACES",
    "MODE_LOG", "MODE_LUT", "OUTPUT_MODES",
    "RANGE_FULL", "RANGE_LEGAL", "OUTPUT_RANGES",
    "OutputTransform", "legal_bounds", "full_scale", "scale_to_int",
]
