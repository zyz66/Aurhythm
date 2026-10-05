"""参数存取：完整 JSON 快照、旧版本迁移、批量复制。

* :func:`save_settings` / :func:`load_settings`：整份参数（含 profile 路径、
  WB、片基采样、色卡矩阵、增益、预设、曲线、调色、输出设置）。
* :func:`migrate`：把 v4 的扁平参数（``hd_slope`` / ``hd_clip_softness`` /
  ``hd_mid`` …）迁移到 v5 结构，未知键**报告**而不是静默丢弃。
* :func:`apply_settings_to_pipeline`：把快照写进管线（批量处理复制参数用），
  profile 一律用**完整路径**（R10）。
"""

from __future__ import annotations

import json
import os

import numpy as np

from .constants import SCHEMA_VERSION
from . import params as params_mod

_V4_KEYS_V5 = frozenset({
    "icc_source", "icc_weight", "output_colorspace", "channel_gains",
    "base_val_rgb", "lut_path", "preset",
})

#: v4 → v5 的顶层键改名
_V4_KEY_MAP = {
    "preset": "preset",
    "icc_weight": "icc_weight",
    "output_colorspace": "output_colorspace",
    "channel_gains": "channel_gains",
    "base_val_rgb": "base_val_rgb",
    "lut_path": "lut_path",
}

#: 已知的 v4 键（用于「未知键报告」）
_V4_KNOWN = set(_V4_KEY_MAP) | {
    "hd_min", "hd_max", "hd_slope", "hd_mid", "hd_clip_softness", "hd_softness",
    "icc_source", "calibration_source", "color_correction_matrix", "auto_fit",
}

#: 当前（v5）全部合法键 + v4 历史键；其余一律作为「未知参数」报告
_KNOWN_KEYS = _V4_KNOWN | _V4_KEYS_V5 | {
    "schema_version", "hd", "tone", "curves", "colorchecker", "view",
    "gain_clamp", "target_density", "wb_mode", "wb_gains",
    "crosstalk_enabled", "clamp_cmy", "codes_per_density", "export_format",
    "format", "bit_depth", "description", "summary", "items",
}


def save_settings(path: str, settings: dict) -> str:
    payload = dict(settings)
    payload.setdefault("schema_version", SCHEMA_VERSION)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
    return path


def load_settings(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return migrate(data)


def migrate(data: dict, report: list | None = None) -> dict:
    """把任意历史版本的参数快照迁移到当前结构。

    ``report`` 若给定，会追加「被忽略的键」等说明，便于 UI 提示用户，
    避免旧设置文件静默失效。
    """
    version = int(data.get("schema_version", 0) or 0)
    out = dict(data)
    out["schema_version"] = SCHEMA_VERSION

    # hd 子字典：交给 HDParams.from_dict 处理旧字段
    hd = dict(out.get("hd") or {})
    for legacy in ("hd_min", "hd_max", "hd_slope", "hd_mid",
                   "hd_clip_softness", "hd_softness"):
        if legacy in out and legacy not in hd:
            hd[legacy] = out[legacy]
        out.pop(legacy, None)
    if hd:
        from .constants import HDParams
        out["hd"] = HDParams.from_dict(hd).to_dict()

    # 色卡矩阵
    if "color_correction_matrix" in out and "colorchecker" not in out:
        out["colorchecker"] = {
            "calibrated": True,
            "source": out.get("calibration_source", "imported"),
            "matrix": out.pop("color_correction_matrix"),
        }

    # 导出格式（v4 是 tiff/exr/dpx + 位深）
    if "export_format" not in out:
        fmt = out.pop("export_format_v4", None)
        if fmt is None:
            legacy_fmt = data.get("format") or "tiff"
            legacy_bits = int(data.get("bit_depth", 16) or 16)
            if legacy_fmt == "exr":
                fmt = "exr32" if legacy_bits == 32 else "exr16"
            elif legacy_fmt == "dpx":
                fmt = "dpx10"
            else:
                fmt = "tiff32" if legacy_bits == 32 else "tiff16"
        out["export_format"] = fmt

    if report is not None and version < SCHEMA_VERSION:
        report.append(f"设置文件版本 {version} 已迁移到 {SCHEMA_VERSION}")
    if report is not None:
        for key in data:
            if key not in _KNOWN_KEYS:
                report.append(f"忽略未知参数: {key}")
    return out


def apply_settings_to_pipeline(pipeline, data: dict, copy_pixels: bool = False):
    """把参数快照写进管线。

    ``copy_pixels=True`` 时连色卡矫正矩阵一起复制（批量处理时对每张图
    独立生效）；profile 使用**完整路径**重新加载（R10）。
    """
    report: list = []
    data = migrate(data, report=report)

    icc_source = data.get("icc_source")
    if icc_source and os.path.exists(icc_source):
        try:
            pipeline.load_icc_profile(icc_source)
        except Exception as exc:                        # noqa: BLE001
            report.append(f"无法重新加载 profile: {exc}")
    elif icc_source:
        report.append(f"profile 路径不存在: {icc_source}")

    if data.get("icc_weight") is not None:
        pipeline.set_icc_weight(float(data["icc_weight"]))

    if data.get("wb_gains") is not None:
        pipeline.set_wb(data.get("wb_mode", "manual"), data["wb_gains"])
    elif data.get("wb_mode"):
        pipeline.wb_mode = data["wb_mode"]

    checker = data.get("colorchecker") or {}
    if checker.get("matrix") is not None and (copy_pixels or
                                              checker.get("calibrated")):
        pipeline.set_color_matrix(np.array(checker["matrix"], dtype=np.float64))
        pipeline.calibration_source = checker.get("source", "imported")

    if data.get("base_val_rgb") is not None:
        pipeline.set_base_val(np.array(data["base_val_rgb"], dtype=np.float64))
    if data.get("gain_clamp") is not None:
        pipeline.gain_clamp = tuple(data["gain_clamp"])
    if data.get("target_density") is not None:
        pipeline.target_density = float(data["target_density"])
    if data.get("channel_gains") is not None:
        pipeline.set_channel_gains(data["channel_gains"])
    if data.get("crosstalk_enabled") is not None:
        pipeline.crosstalk_enabled = bool(data["crosstalk_enabled"])
    if data.get("clamp_cmy") is not None:
        pipeline.clamp_cmy = bool(data["clamp_cmy"])

    if data.get("preset"):
        if not pipeline.set_preset(data["preset"]):
            report.append(f"未知预设: {data['preset']}")
    if data.get("hd"):
        from .constants import HDParams
        pipeline.hd = HDParams.from_dict(data["hd"])
    if data.get("tone"):
        from .tonemap import ToneControls
        pipeline.set_tone_controls(ToneControls.from_dict(data["tone"]))
    if data.get("curves"):
        from .tonemap import CurveSet
        pipeline.set_curves(CurveSet.from_dict(data["curves"]))

    if data.get("auto_fit") is not None:
        pipeline.auto_fit = bool(data["auto_fit"])
    if data.get("codes_per_density") is not None:
        pipeline.codes_per_density = float(data["codes_per_density"])
    if data.get("output_colorspace"):
        pipeline.set_output_colorspace(data["output_colorspace"])
    if data.get("output"):
        from .output import OutputTransform
        pipeline.set_output_transform(OutputTransform.from_dict(data["output"]))

    lut_path = data.get("lut_path")
    pipeline.lut_path = lut_path
    want_lut = bool(data.get("lut_enabled")) or bool(
        (data.get("output") or {}).get("mode") == "lut")
    if want_lut:
        if lut_path and os.path.exists(lut_path):
            try:
                from .lut import CubeLUT
                lut = CubeLUT()
                lut.load(lut_path)
                pipeline.set_output_lut(
                    lut, enabled=True, path=lut_path,
                    input_space=(data.get("output") or {}).get("lut_input_space"),
                    target_space=(data.get("output") or {}).get(
                        "lut_target_space"),
                    range_mode=(data.get("output") or {}).get("range"))
            except Exception as exc:                    # noqa: BLE001
                report.append(f"无法重新加载 LUT: {exc}")
        else:
            report.append(f"LUT 路径不可用（输出将退回对数直通）: {lut_path}")
            pipeline.set_output_lut(None, enabled=False)

    return report


def copy_settings_to_pipeline(source, target, copy_pixels: bool = True) -> list:
    """把参数从一条管线复制到另一条（批量处理用）。"""
    return apply_settings_to_pipeline(target, source.to_settings(),
                                      copy_pixels=copy_pixels)


def settings_to_param_values(data: dict) -> dict:
    """把参数快照转成注册表键空间的值（UI 表单回填用）。"""
    values = {}
    hd = data.get("hd") or {}
    for key, field in params_mod.HD_KEYS.items():
        if field in hd:
            values[key] = hd[field]
    gains = data.get("channel_gains")
    if gains is not None:
        for key, index in params_mod.GAIN_KEYS.items():
            values[key] = float(np.asarray(gains)[index])
    tone = data.get("tone") or {}
    for key in params_mod.PARAM_SPECS:
        if key.startswith("tone_") and key[5:] in tone:
            values[key] = tone[key[5:]]
        if key in params_mod.CDL_KEYS:
            field_name, index = params_mod.CDL_KEYS[key]
            if tone.get(field_name) is not None:
                values[key] = float(tone[field_name][index])
    for key in ("icc_weight", "target_density", "clamp_cmy", "auto_fit",
                "codes_per_density", "output_colorspace", "export_format",
                "lut_enabled", "lut_input_space", "lut_target_space",
                "output_range"):
        if key == "lut_input_space" and data.get("output"):
            values[key] = data["output"].get("lut_input_space",
                                             values.get(key))
            continue
        if key == "lut_target_space" and data.get("output"):
            values[key] = data["output"].get("lut_target_space",
                                             values.get(key))
            continue
        if key == "output_range" and data.get("output"):
            values[key] = data["output"].get("range", values.get(key))
            continue
        if data.get(key) is not None:
            values[key] = data[key]
    return values


__all__ = [
    "save_settings", "load_settings", "migrate", "apply_settings_to_pipeline",
    "copy_settings_to_pipeline", "settings_to_param_values",
]
