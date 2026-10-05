"""参数注册表：把「UI 控件 ↔ 管线状态」的映射集中到一处。

为什么需要它（R7 类缺陷的根治）
------------------------------
旧实现里 UI 把值写进 ``hd_softness``，而管线读的是 ``hd_clip_softness``
——「软度 ε」滑块完全无效，而且没有任何机制能发现这种错配。

本模块提供**唯一**的键空间：

* :data:`PARAM_SPECS` 描述每个参数（范围、默认、单位、是否属于调色层）；
* :func:`apply_param` / :func:`read_param` 是唯一允许读写管线的地方，
  未知键直接报错；
* :func:`look_keys` 标出属于「调色（可选·非校准）」的参数，
  UI 据此分区，默认全部恒等。

``tests/test_params.py`` 会断言：每个注册键都能被读写、每个键都能往返、
且 UI 用到的键集合与注册表完全一致。这样「控件与属性名不匹配」这类
缺陷在结构上不可能再出现。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .output import DPX_COLORIMETRIC as DPX_COLORIMETRIC_NAMES
from .output import DPX_TRANSFER as DPX_TRANSFER_NAMES

#: 参数分组（UI 标签页顺序）
GROUP_COLOR = "色彩校准"
GROUP_FILM = "非线性校正（胶片曲线）"
GROUP_LOOK = "调色（可选·非校准）"
GROUP_EXPORT = "导出"
GROUP_VIEW = "显示"
GROUP_ORDER = (GROUP_COLOR, GROUP_FILM, GROUP_VIEW, GROUP_EXPORT, GROUP_LOOK)


@dataclass(frozen=True)
class ParamSpec:
    key: str
    label: str
    group: str
    kind: str = "float"            # 'float' | 'int' | 'choice' | 'bool'
    lo: float = 0.0
    hi: float = 1.0
    default: object = 0.0
    unit: str = ""
    look: bool = False             # True = 调色层（可选·非校准）
    choices: tuple = ()
    doc: str = ""
    decimals: int = 3


PARAM_SPECS: dict[str, ParamSpec] = {}


def _spec(**kwargs) -> ParamSpec:
    spec = ParamSpec(**kwargs)
    PARAM_SPECS[spec.key] = spec
    return spec


# ---------------- 胶片特性曲线（校准） ----------------
_spec(key="hd_a", label="反差 a", group=GROUP_FILM, lo=0.2, hi=6.0,
      default=1.0, doc="1.0 = 保持原反差；>1 增大反差（S 曲线更陡）")
_spec(key="hd_x_mid", label="中点位置", group=GROUP_FILM, lo=0.0, hi=1.0,
      default=0.5, doc="曲线中点所在的归一化密度位置")
_spec(key="hd_s_toe", label="趾部软度", group=GROUP_FILM, lo=0.0, hi=1.0,
      default=0.06, decimals=4, doc="低密度端（片基/暗部）圆角半径")
_spec(key="hd_s_sh", label="肩部软度", group=GROUP_FILM, lo=0.0, hi=1.0,
      default=0.06, decimals=4, doc="高密度端（高光）圆角半径")
_spec(key="hd_d_min", label="输入密度下限", group=GROUP_FILM, lo=0.0, hi=1.5,
      default=0.18, doc="片基 + 灰雾；通常由片基采样自动给出")
_spec(key="hd_d_max", label="输入密度上限", group=GROUP_FILM, lo=0.3, hi=5.0,
      default=3.10, doc="最大密度；可用「测量密度范围」自动给出")

# ---------------- 校准（色彩） ----------------
_spec(key="icc_weight", label="色温插值", group=GROUP_COLOR, lo=0.0, hi=1.0,
      default=1.0, doc="0 = 钨丝灯（矩阵 2），1 = 日光（矩阵 1）")
_spec(key="target_density", label="目标中间调密度", group=GROUP_COLOR,
      lo=0.2, hi=1.6, default=0.70)
_spec(key="gain_r", label="R 通道增益", group=GROUP_COLOR, lo=0.125, hi=8.0,
      default=1.0, doc="印片光式通道增益（线性域），会改变逐通道密度")
_spec(key="gain_g", label="G 通道增益", group=GROUP_COLOR, lo=0.125, hi=8.0,
      default=1.0)
_spec(key="gain_b", label="B 通道增益", group=GROUP_COLOR, lo=0.125, hi=8.0,
      default=1.0)
_spec(key="clamp_cmy", label="染料密度钳零", group=GROUP_COLOR, kind="bool",
      default=True, doc="解串扰后把负染料密度钳到 0（物理上密度非负）")

# ---------------- 编码 / 导出 ----------------
_spec(key="auto_fit", label="自动匹配参考黑/白", group=GROUP_EXPORT,
      kind="bool", default=True,
      doc="把整个输入密度窗口映射到 Cineon 参考黑 95 / 参考白 685")
_spec(key="codes_per_density", label="编码密度", group=GROUP_EXPORT,
      lo=100.0, hi=1000.0, default=500.0, unit="codes/D",
      doc="关掉自动匹配时使用的 Cineon 标准编码密度")
_spec(key="output_colorspace", label="输出色彩空间", group=GROUP_EXPORT,
      kind="choice", default="cineon", choices=("cineon", "logc3"),
      doc="cineon = 10-bit 对数；logc3 = ARRI LogC3（经线性曝光）")
_spec(key="view_tone_scale", label="显示码率倍率", group=GROUP_VIEW,
      kind="float", default=1.0, lo=0.2, hi=5.0,
      doc="1.0 = 按片种 γ 自动算出；用标准 500 codes/decade 解会发灰")
_spec(key="export_format", label="导出格式", group=GROUP_EXPORT,
      kind="choice", default="tiff16",
      choices=("tiff16", "tiff32", "dpx10", "exr16", "exr32"))
_spec(key="lut_enabled", label="套用色彩还原 LUT", group=GROUP_EXPORT,
      kind="bool", default=False,
      doc="开启后输出变成 LUT 的目标色彩空间（例如 LogC3→Rec.709），不再是对数素材")
_spec(key="lut_input_space", label="LUT 输入空间", group=GROUP_EXPORT,
      kind="choice", default="logc3", choices=("logc3", "cineon"),
      doc="这个 LUT 期望喂什么（ARRI 官网的 LogC3→Rec.709 要选 logc3）")
_spec(key="lut_target_space", label="LUT 目标空间", group=GROUP_EXPORT,
      kind="choice", default="rec709", choices=("rec709", "srgb", "linear"),
      doc="LUT 输出所在的色彩空间，也是最终写入容器的空间")
_spec(key="output_range", label="输出范围", group=GROUP_EXPORT,
      kind="choice", default="full", choices=("full", "legal"),
      doc="legal = 视频合法范围（10-bit 64..940）。若 LUT 本身已输出合法范围，选 full 以免二次压缩")
_spec(key="dpx_transfer", label="DPX transfer", group=GROUP_EXPORT,
      kind="choice", default="auto",
      choices=("auto",) + tuple(DPX_TRANSFER_NAMES),
      doc="默认按输出空间自动；可在 Resolve 里核对后手动覆盖")
_spec(key="dpx_colorimetric", label="DPX colorimetric", group=GROUP_EXPORT,
      kind="choice", default="auto",
      choices=("auto",) + tuple(DPX_COLORIMETRIC_NAMES),
      doc="同上，默认按输出空间自动")

# ---------------- 显示 ----------------
_spec(key="view_exposure_ev", label="显示曝光", group=GROUP_VIEW,
      lo=-4.0, hi=4.0, default=0.0, unit="EV")
_spec(key="view_gamma", label="显示伽马", group=GROUP_VIEW,
      lo=0.2, hi=3.0, default=1.0)
_spec(key="view_mode", label="显示模式", group=GROUP_VIEW, kind="choice",
      default="video", choices=("video", "linear", "false_color"))

# ---------------- 调色（可选·非校准） ----------------
_spec(key="tone_exposure_ev", label="曝光基准", group=GROUP_LOOK, lo=-5.0,
      hi=5.0, default=0.0, unit="EV", look=True,
      doc="按 Cineon 曝光标度平移 code（1 档 = 150.5 codes）")
_spec(key="tone_contrast", label="对比度", group=GROUP_LOOK, lo=0.1, hi=4.0,
      default=1.0, look=True, doc="围绕 18% 中灰旋转")
_spec(key="tone_master_lift", label="主 Lift", group=GROUP_LOOK, lo=-0.2,
      hi=0.2, default=0.0, look=True)
_spec(key="tone_master_gamma", label="主 Gamma", group=GROUP_LOOK, lo=0.2,
      hi=3.0, default=1.0, look=True)
_spec(key="tone_master_gain", label="主 Gain", group=GROUP_LOOK, lo=0.2, hi=3.0,
      default=1.0, look=True)
for _ch, _name in (("r", "R"), ("g", "G"), ("b", "B")):
    _spec(key=f"cdl_lift_{_ch}", label=f"{_name} Lift", group=GROUP_LOOK,
          lo=-0.2, hi=0.2, default=0.0, look=True)
    _spec(key=f"cdl_gamma_{_ch}", label=f"{_name} Gamma", group=GROUP_LOOK,
          lo=0.2, hi=3.0, default=1.0, look=True)
    _spec(key=f"cdl_gain_{_ch}", label=f"{_name} Gain", group=GROUP_LOOK,
          lo=0.2, hi=3.0, default=1.0, look=True)

#: 曲线参数的键 → HDParams 字段名
HD_KEYS = {
    "hd_a": "a", "hd_x_mid": "x_mid", "hd_s_toe": "s_toe",
    "hd_s_sh": "s_sh", "hd_d_min": "d_min", "hd_d_max": "d_max",
}
#: 增益的键 → 通道下标
GAIN_KEYS = {"gain_r": 0, "gain_g": 1, "gain_b": 2}
#: CDL 的键 → (字段, 通道)
CDL_KEYS = {f"cdl_{field}_{ch}": (field, index)
            for field in ("lift", "gamma", "gain")
            for index, ch in enumerate("rgb")}


def specs_for_group(group: str) -> list:
    """返回某分组下的参数（保持注册顺序）。"""
    return [spec for spec in PARAM_SPECS.values() if spec.group == group]


def look_keys() -> frozenset:
    """属于调色层（可选·非校准）的参数键。"""
    return frozenset(k for k, s in PARAM_SPECS.items() if s.look)


def calibration_keys() -> frozenset:
    return frozenset(PARAM_SPECS) - look_keys()


def spec(key: str) -> ParamSpec:
    try:
        return PARAM_SPECS[key]
    except KeyError as exc:
        raise KeyError(f"未注册的参数键: {key!r}") from exc


def validate(key: str, value):
    """按注册表校验并夹取取值（未知键报错）。"""
    s = spec(key)
    if s.kind == "bool":
        return bool(value)
    if s.kind == "choice":
        if value not in s.choices:
            raise ValueError(f"{key} 必须是 {s.choices} 之一，收到 {value!r}")
        return value
    if s.kind == "int":
        return int(np.clip(int(value), s.lo, s.hi))
    return float(np.clip(float(value), s.lo, s.hi))


# ======================================================================
# 管线读写（唯一入口）
# ======================================================================

def apply_param(pipeline, key: str, value):
    """把单个参数写入管线（未知键报错，非法值夹取）。"""
    from .tonemap import ToneControls

    if key in HD_KEYS:
        return pipeline.set_hd_params(**{HD_KEYS[key]: validate(key, value)})
    if key in GAIN_KEYS:
        gains = np.array(pipeline.channel_gains, dtype=np.float64)
        gains[GAIN_KEYS[key]] = validate(key, value)
        return pipeline.set_channel_gains(gains)
    if key in CDL_KEYS:
        field_name, index = CDL_KEYS[key]
        tone = pipeline.tone
        current = np.array(getattr(tone, field_name), dtype=np.float64)
        current[index] = validate(key, value)
        return pipeline.set_tone_controls(
            _replace_tone(tone, {field_name: tuple(current.tolist())}))
    if key.startswith("tone_"):
        tone = pipeline.tone
        field_name = key[len("tone_"):]
        if not hasattr(tone, field_name):
            raise KeyError(f"未注册的调色参数: {key!r}")
        return pipeline.set_tone_controls(
            _replace_tone(tone, {field_name: validate(key, value)}))
    if key == "icc_weight":
        pipeline.set_icc_weight(validate(key, value))
        return pipeline.icc_weight
    if key == "target_density":
        pipeline.target_density = validate(key, value)
        return pipeline.target_density
    if key == "clamp_cmy":
        pipeline.clamp_cmy = validate(key, value)
        return pipeline.clamp_cmy
    if key == "auto_fit":
        pipeline.auto_fit = validate(key, value)
        return pipeline.auto_fit
    if key == "codes_per_density":
        pipeline.codes_per_density = validate(key, value)
        return pipeline.codes_per_density
    if key == "output_colorspace":
        pipeline.set_output_colorspace(validate(key, value))
        return pipeline.output_colorspace
    if key == "lut_enabled":
        pipeline.set_output_lut(pipeline.lut, enabled=validate(key, value),
                                path=pipeline.lut_path)
        return pipeline.lut_enabled
    if key == "lut_input_space":
        from dataclasses import replace
        pipeline.output = replace(
            pipeline.output,
            lut_input_space=validate(key, value)).sanitize()
        return pipeline.output.lut_input_space
    if key == "lut_target_space":
        from dataclasses import replace
        pipeline.output = replace(
            pipeline.output,
            lut_target_space=validate(key, value)).sanitize()
        return pipeline.output.lut_target_space
    if key == "output_range":
        from dataclasses import replace
        pipeline.output = replace(
            pipeline.output, range=validate(key, value)).sanitize()
        return pipeline.output.range
    if key in ("dpx_transfer", "dpx_colorimetric"):
        from dataclasses import replace
        table = (DPX_TRANSFER_NAMES if key == "dpx_transfer"
                 else DPX_COLORIMETRIC_NAMES)
        name = validate(key, value)
        code = None if name == "auto" else int(table[name])
        pipeline.output = replace(pipeline.output, **{key: code}).sanitize()
        return name
    if key.startswith("view_"):
        from dataclasses import replace
        pipeline.view = replace(pipeline.view,
                                **{key[len("view_"):]: validate(key, value)})
        return getattr(pipeline.view, key[len("view_"):])
    if key == "export_format":
        return validate(key, value)          # 纯 UI 状态，不进管线
    raise KeyError(f"未注册的参数键: {key!r}")


def read_param(pipeline, key: str):
    """从管线读取单个参数的当前值。"""
    if key in HD_KEYS:
        return getattr(pipeline.hd, HD_KEYS[key])
    if key in GAIN_KEYS:
        return float(np.asarray(pipeline.channel_gains)[GAIN_KEYS[key]])
    if key in CDL_KEYS:
        field_name, index = CDL_KEYS[key]
        return float(getattr(pipeline.tone, field_name)[index])
    if key.startswith("tone_"):
        field_name = key[len("tone_"):]
        if not hasattr(pipeline.tone, field_name):
            raise KeyError(f"未注册的调色参数: {key!r}")
        return float(getattr(pipeline.tone, field_name))
    if key == "icc_weight":
        return pipeline.icc_weight
    if key == "target_density":
        return pipeline.target_density
    if key == "clamp_cmy":
        return pipeline.clamp_cmy
    if key == "auto_fit":
        return pipeline.auto_fit
    if key == "codes_per_density":
        return pipeline.codes_per_density
    if key == "output_colorspace":
        return pipeline.output_colorspace
    if key == "lut_enabled":
        return pipeline.lut_enabled
    if key == "lut_input_space":
        return pipeline.output.lut_input_space
    if key == "lut_target_space":
        return pipeline.output.lut_target_space
    if key == "output_range":
        return pipeline.output.range
    if key in ("dpx_transfer", "dpx_colorimetric"):
        table = (DPX_TRANSFER_NAMES if key == "dpx_transfer"
                 else DPX_COLORIMETRIC_NAMES)
        code = getattr(pipeline.output, key)
        if code is None:
            return "auto"
        for name, value_ in table.items():
            if value_ == code:
                return name
        return "auto"
    if key.startswith("view_"):
        return getattr(pipeline.view, key[len("view_"):])
    raise KeyError(f"未注册的参数键: {key!r}")


def apply_all(pipeline, values: dict):
    """批量写入（忽略未知键会报错，便于尽早发现错配）。"""
    for key, value in values.items():
        apply_param(pipeline, key, value)
    return pipeline


def snapshot(pipeline) -> dict:
    """把注册表里所有「进管线」的参数读成一个 dict（撤销栈用）。"""
    out = {}
    for key in PARAM_SPECS:
        if key == "export_format":
            continue
        try:
            out[key] = read_param(pipeline, key)
        except KeyError:
            continue
    return out


def restore(pipeline, values: dict):
    """从 :func:`snapshot` 的快照恢复。"""
    for key, value in values.items():
        if key in PARAM_SPECS and key != "export_format":
            apply_param(pipeline, key, value)
    return pipeline


def defaults(look: bool = True) -> dict:
    """所有参数的默认值（``look=False`` 时排除调色层）。"""
    return {k: s.default for k, s in PARAM_SPECS.items()
            if look or not s.look}


def _replace_tone(tone, changes):
    from dataclasses import replace
    return replace(tone, **changes)


__all__ = [
    "ParamSpec", "PARAM_SPECS", "GROUP_ORDER", "GROUP_COLOR", "GROUP_FILM",
    "GROUP_LOOK", "GROUP_EXPORT", "GROUP_VIEW", "HD_KEYS", "GAIN_KEYS",
    "CDL_KEYS", "spec", "specs_for_group", "look_keys", "calibration_keys",
    "validate", "apply_param", "read_param", "apply_all", "snapshot",
    "restore", "defaults",
]
