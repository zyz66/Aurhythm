"""命令行接口（headless）：convert / calibrate / lut / synth / info。

退出码
------
====  ==================================================
0     全部成功
1     部分失败（有 skipped/failed 项）
2     用法错误（参数不合法）
3     不支持的输入（格式/文件）
====  ==================================================

``--json`` 输出稳定 schema 的报告，便于脚本消费。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__
from . import batch as batch_mod
from . import io_read
from . import io_write
from . import params as params_mod
from . import settings as settings_mod
from . import synth as synth_mod
from .constants import FILM_PRESETS, PRESET_ORDER
from .lut import CubeLUT, LutError
from .output import DPX_COLORIMETRIC as DPX_COLORIMETRIC_NAMES
from .output import DPX_TRANSFER as DPX_TRANSFER_NAMES
from .pipeline import ScientificFilmPipeline
from .profiles import ProfileError

EXIT_OK = 0
EXIT_PARTIAL = 1
EXIT_USAGE = 2
EXIT_UNSUPPORTED = 3


# ======================================================================
# 通用
# ======================================================================

def _load_base_settings(args) -> dict:
    """把 CLI 参数整理成管线设置快照。"""
    settings: dict = {"schema_version": 1}
    if getattr(args, "settings", None):
        settings = settings_mod.load_settings(args.settings)
    if getattr(args, "preset", None):
        if args.preset not in FILM_PRESETS:
            raise SystemExit(
                f"未知预设 {args.preset!r}；可用: {', '.join(PRESET_ORDER)}")
        preset = FILM_PRESETS[args.preset]
        settings["preset"] = args.preset
        settings["hd"] = preset.hd.to_dict()
        settings["crosstalk_enabled"] = not preset.skip
    for key, attr in (("auto_fit", "auto_fit"),
                      ("codes_per_density", "codes_per_density"),
                      ("output_colorspace", "colorspace")):
        value = getattr(args, attr, None)
        if value is not None:
            settings[key] = value
    if getattr(args, "clamp_cmy", None) is not None:
        settings["clamp_cmy"] = args.clamp_cmy
    if getattr(args, "exposure_ev", None) is not None:
        tone = dict(settings.get("tone") or {})
        tone["exposure_ev"] = args.exposure_ev
        settings["tone"] = tone
    if getattr(args, "contrast", None) is not None:
        tone = dict(settings.get("tone") or {})
        tone["contrast"] = args.contrast
        settings["tone"] = tone
    if getattr(args, "d_min", None) is not None:
        hd = dict(settings.get("hd") or {})
        hd["d_min"] = args.d_min
        settings["hd"] = hd
    if getattr(args, "d_max", None) is not None:
        hd = dict(settings.get("hd") or {})
        hd["d_max"] = args.d_max
        settings["hd"] = hd
    if getattr(args, "profile", None):
        settings["icc_source"] = os.path.abspath(args.profile)
    if getattr(args, "profile_weight", None) is not None:
        settings["icc_weight"] = args.profile_weight
    if getattr(args, "base", None):
        if args.base != "auto":
            values = [float(v) for v in args.base.split(",")]
            if len(values) != 3:
                raise SystemExit("--base 需要 'auto' 或 'r,g,b'")
            settings["base_val_rgb"] = values
    # 输出变换：--lut 与 DPX 字段覆盖是**独立**的，后者不套 LUT 也应当生效
    output = dict(settings.get("output") or {})
    if getattr(args, "lut", None):
        settings["lut_path"] = os.path.abspath(args.lut)
        settings["lut_enabled"] = not args.no_lut
        output.update({"lut_input_space": args.lut_input,
                       "lut_target_space": args.lut_target,
                       "range": args.range,
                       "mode": "log" if args.no_lut else "lut"})
    elif getattr(args, "range", None):
        output["range"] = args.range
    if getattr(args, "dpx_transfer", None):
        output["dpx_transfer"] = _code_from_name(
            DPX_TRANSFER_NAMES, args.dpx_transfer, "--dpx-transfer")
    if getattr(args, "dpx_colorimetric", None):
        output["dpx_colorimetric"] = _code_from_name(
            DPX_COLORIMETRIC_NAMES, args.dpx_colorimetric,
            "--dpx-colorimetric")
    if output:
        settings["output"] = output
    if getattr(args, "channel_gains", None):
        values = [float(v) for v in args.channel_gains.split(",")]
        if len(values) != 3:
            raise SystemExit("--channel-gains 需要 'r,g,b'")
        settings["channel_gains"] = values
    return settings


def _code_from_name(table, name, option):
    if name == "auto":
        return None
    if name not in table:
        raise SystemExit(f"{option} 取值必须是 auto 或 {', '.join(table)}")
    return int(table[name])


def _print(payload, as_json: bool, text: str):
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print(text)


# ======================================================================
# convert
# ======================================================================

def cmd_convert(args) -> int:
    outputs = []
    for path in args.input:
        if not os.path.exists(path):
            print(f"跳过不存在的文件: {path}", file=sys.stderr)
            outputs.append(None)
            continue
        outputs.append(path)
    paths = [p for p in outputs if p]
    if not paths:
        print("没有可处理的输入", file=sys.stderr)
        return EXIT_UNSUPPORTED
    try:
        settings = _load_base_settings(args)
    except ProfileError as exc:
        print(f"profile 错误: {exc}", file=sys.stderr)
        return EXIT_UNSUPPORTED
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE

    os.makedirs(args.out, exist_ok=True)

    def progress(index, total, item):
        if not args.json:
            print(f"[{index}/{total}] {item.name}: {item.status} "
                  f"{item.message}", file=sys.stderr)

    result = batch_mod.run_batch(
        paths, settings, progress=progress,
        out_dir=args.out, export_format=args.format,
        auto_base=not args.no_auto_base, neutralize=not args.no_neutralize,
        target_density=args.density_target, overwrite=args.overwrite,
    )
    summary = result.summary
    if not args.json:
        try:
            pipe = ScientificFilmPipeline()
            settings_mod.apply_settings_to_pipeline(pipe, settings)
            print(f"输出: {pipe.output.label()}", file=sys.stderr)
        except Exception:                               # noqa: BLE001
            pass
    _print(result.to_dict(), args.json,
           f"完成: {summary['ok']}/{summary['total']} 成功，"
           f"{summary['failed']} 失败，{summary['skipped']} 跳过，"
           f"用时 {summary['elapsed_s']}s")
    if args.report:
        settings_mod.save_settings(args.report, result.to_dict())
    return EXIT_OK if summary["failed"] == 0 and summary["skipped"] == 0 \
        else EXIT_PARTIAL


# ======================================================================
# calibrate
# ======================================================================

def cmd_calibrate(args) -> int:
    from . import colorchecker as cc

    try:
        image, info = io_read.load_any(args.chart)
    except io_read.ReadError as exc:
        print(f"读取失败: {exc}", file=sys.stderr)
        return EXIT_UNSUPPORTED

    if args.corners:
        values = [float(v) for v in args.corners.split(",")]
        if len(values) != 8:
            print("--corners 需要 8 个数: x1,y1,x2,y2,x3,y3,x4,y4", file=sys.stderr)
            return EXIT_USAGE
        corners = [[values[i * 2], values[i * 2 + 1]] for i in range(4)]
        reason = ""
    else:
        detection, reason = cc.detect_chart(image, explain=True)
        if detection is None:
            print(f"自动检测失败: {reason}", file=sys.stderr)
            print("请用 --corners 手动给出四角（顺序：左上, 右上, 右下, 左下）",
                  file=sys.stderr)
            return EXIT_PARTIAL
        corners = detection.corners
        if not args.json:
            print(f"检测成功（置信度 {detection.confidence:.2f}）", file=sys.stderr)

    sampled = cc.sample_patches(image, corners, margin=args.margin)
    result = cc.solve_correction(sampled, mode=args.mode, ridge=args.ridge)
    payload = {
        "source": info.to_dict(),
        "corners": [[float(x), float(y)] for x, y in corners],
        "detection_reason": reason,
        "matrix": result.matrix.tolist(),
        "offset": None if result.offset is None else result.offset.tolist(),
        "stats": result.stats,
        "table": cc.error_table(result.stats),
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"matrix": result.matrix.tolist(),
                       "offset": None if result.offset is None
                       else result.offset.tolist(),
                       "description": f"Aurhythm calibrate {os.path.basename(args.chart)}",
                       "stats": result.stats}, fh,
                      ensure_ascii=False, indent=2)
    _print(payload, args.json, cc.format_error_report(result.stats))
    return EXIT_OK


# ======================================================================
# lut
# ======================================================================

def cmd_lut(args) -> int:
    try:
        settings = _load_base_settings(args)
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE
    pipe = ScientificFilmPipeline()
    settings_mod.apply_settings_to_pipeline(pipe, settings)
    lut = CubeLUT.from_pipeline(pipe, size=args.size, include_look=args.include_look)
    lut.save(args.out)
    payload = {"output": os.path.abspath(args.out), "size": args.size,
               "include_look": bool(args.include_look),
               "preset": pipe.preset_name,
               "codes_per_unit": pipe.transfer.codes_per_unit}
    _print(payload, args.json,
           f"已写出 {args.out}（{args.size}³，"
           f"{'含调色层' if args.include_look else '仅校准变换'}）")
    return EXIT_OK


# ======================================================================
# synth
# ======================================================================

def cmd_synth(args) -> int:
    import numpy as np

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if args.kind == "chart":
        corners = None
        if args.corners:
            values = [float(v) for v in args.corners.split(",")]
            if len(values) != 8:
                print("--corners 需要 8 个数", file=sys.stderr)
                return EXIT_USAGE
            corners = [[values[i * 2], values[i * 2 + 1]] for i in range(4)]
        image = synth_mod.make_chart(height=args.height, width=args.width,
                                    corners=corners, noise=args.noise)
        note = "24 色卡（含块间黑格与背景）"
    elif args.kind == "scene":
        from . import filmsim

        result = synth_mod.make_film_negative(
            args.width, args.height, seed=args.seed, border=args.border,
            grain=args.grain)
        image = result["negative"]
        base = result["base_rgb"]
        extra = {
            "base_rgb": [round(float(v), 6) for v in base],
            "stock": result["stock"].to_dict(),
            "densitometry": filmsim.densitometry_report(result),
            "matching_preset": "测试：C-41 合成负片（橙色罩+串扰）",
        }
        note = (f"C-41 合成负片（橙色罩 + 三层染料串扰 + 逐层特性曲线）；"
                f"真实片基 {[round(float(v), 4) for v in base]}")
        # 对照正片：让用户一眼看出管线该还原成什么样
        stem = os.path.splitext(args.out)[0]
        truth_path = stem + "_truth.png"
        io_write.write_png(truth_path, filmsim.to_srgb8(result["scene"]))
        extra["truth_positive"] = truth_path
        # 扫描仪看到的原貌：线性值直接按 sRGB 编码 —— 就是那个橙色罩的样子
        from . import colorimetry as _cm
        raw_path = stem + "_rawview.png"
        io_write.write_png(raw_path, np.clip(
            _cm.srgb_encode(np.clip(result["negative"], 0.0, 1.0)) * 255.0,
            0, 255))
        extra["raw_view"] = raw_path
    else:
        image, base = synth_mod.make_negative(
            height=args.height, width=args.width, cast=tuple(args.cast),
            density_gamma=tuple(args.density_gamma), noise=args.noise)
        note = f"彩色负片，真实片基 {base.tolist()}"
        if tuple(args.density_gamma) != (1.0, 1.0, 1.0):
            note += f"；逐通道密度反差 {tuple(args.density_gamma)}（可演示中性化）"
    peak = float(np.max(image)) if image.size else 0.0
    if peak > 1.0:
        note += (f"；⚠ 最大值 {peak:.3f} > 1.0 —— 写整数格式（tiff16/dpx10）时"
                 "会被裁剪，片基/高光会被削平")
    if args.format == "tiff32":
        io_write.write_tiff(args.out, image.astype("float32"), 32)
    else:
        io_write.write_image(args.out, image.astype("float32"), args.format)
    payload = {"output": os.path.abspath(args.out), "kind": args.kind,
               "width": args.width, "height": args.height,
               "format": args.format, "note": note}
    if args.kind != "chart":
        payload["base_rgb"] = [round(float(v), 4) for v in base]
    payload.update(locals().get("extra", {}))
    # 把片种参数与配套预设一起写进 sidecar，便于复现
    if args.kind == "scene":
        io_write.write_provenance(args.out, {
            "kind": "scene", "note": note, "width": args.width,
            "height": args.height, "seed": args.seed, "border": args.border,
            **extra}, embedded=False)
    _print(payload, args.json, f"已写出 {args.out}：{note}")
    return EXIT_OK


# ======================================================================
# info
# ======================================================================

def cmd_info(args) -> int:
    try:
        payload = io_read.probe(args.input)
    except Exception as exc:                            # noqa: BLE001
        print(f"无法读取: {exc}", file=sys.stderr)
        return EXIT_UNSUPPORTED
    if args.settings:
        try:
            payload["settings"] = settings_mod.load_settings(args.settings)
        except Exception as exc:                        # noqa: BLE001
            payload["settings_error"] = str(exc)
    _print(payload, args.json,
           "\n".join(f"{k}: {v}" for k, v in payload.items()))
    return EXIT_OK


# ======================================================================
# 参数解析
# ======================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aurhythm",
        description=f"Aurhythm 胶片 Cineon/LogC3 校准器 v{__version__}（命令行）")
    parser.add_argument("--version", action="version",
                        version=f"aurhythm {__version__}")
    sub = parser.add_subparsers(dest="command")

    def add_common(p, with_io=True):
        if with_io:
            p.add_argument("--settings", help="参数快照 JSON")
        p.add_argument("--preset", choices=PRESET_ORDER, help="胶片预设")
        p.add_argument("--profile", help="相机 ICC/DCP 路径")
        p.add_argument("--profile-weight", type=float, default=None,
                       help="0=钨丝灯, 1=日光")
        p.add_argument("--d-min", type=float, default=None, help="输入密度下限")
        p.add_argument("--d-max", type=float, default=None, help="输入密度上限")
        p.add_argument("--exposure-ev", type=float, default=None,
                       help="曝光基准偏移（档）")
        p.add_argument("--contrast", type=float, default=None, help="对比度")
        p.add_argument("--auto-fit", dest="auto_fit", action="store_true",
                       default=None, help="自动匹配参考黑/白（默认开）")
        p.add_argument("--no-auto-fit", dest="auto_fit", action="store_false",
                       help="改用 Cineon 标准 500 codes/密度单位")
        p.add_argument("--codes-per-density", type=float, default=None)
        p.add_argument("--clamp-cmy", dest="clamp_cmy", action="store_true",
                       default=None)
        p.add_argument("--no-clamp-cmy", dest="clamp_cmy",
                       action="store_false")
        p.add_argument("--json", action="store_true", help="输出 JSON 报告")

    convert = sub.add_parser("convert", help="批量转换/导出")
    convert.add_argument("--in", dest="input", nargs="+", required=True,
                         help="输入文件（RAW/TIFF/PNG）或目录")
    convert.add_argument("--out", required=True, help="输出目录")
    convert.add_argument("--format", default="tiff16",
                         choices=io_write.SUPPORTED_FORMATS)
    convert.add_argument("--colorspace", default=None,
                         choices=("cineon", "logc3"))
    convert.add_argument("--base", default="auto", help="'auto' 或 'r,g,b'")
    convert.add_argument("--channel-gains", default=None, help="'r,g,b'")
    convert.add_argument("--density-target", type=float, default=None)
    convert.add_argument("--no-auto-base", action="store_true")
    convert.add_argument("--no-neutralize", action="store_true")
    convert.add_argument("--recursive", action="store_true")
    convert.add_argument("--overwrite", action="store_true")
    convert.add_argument("--report", help="把批量结果写成 JSON")
    convert.add_argument("--lut", help="色彩还原 LUT（.cube，如 LogC3→Rec.709）")
    convert.add_argument("--no-lut", action="store_true",
                         help="只记录 LUT 路径但不套用（对数直通）")
    convert.add_argument("--lut-input", default="logc3",
                         choices=("logc3", "cineon"),
                         help="这个 LUT 期望的输入空间")
    convert.add_argument("--lut-target", default="rec709",
                         choices=("rec709", "srgb", "linear"),
                         help="LUT 的目标色彩空间（= 最终输出空间）")
    convert.add_argument("--range", default="full", choices=("full", "legal"),
                         help="full = 满量程；legal = 视频合法范围(10bit 64..940)")
    convert.add_argument("--dpx-transfer", default=None,
                         help="覆盖 DPX transfer 字段（默认按输出空间自动）")
    convert.add_argument("--dpx-colorimetric", default=None,
                         help="覆盖 DPX colorimetric 字段")
    add_common(convert)
    convert.set_defaults(func=cmd_convert)

    calibrate = sub.add_parser("calibrate", help="由色卡求解矫正矩阵")
    calibrate.add_argument("--chart", required=True, help="含色卡的图像")
    calibrate.add_argument("--corners", default=None,
                           help="手动四角 x1,y1,...,x4,y4（左上,右上,右下,左下）")
    calibrate.add_argument("--mode", default="3x3",
                           choices=("3x3", "3x3+offset"))
    calibrate.add_argument("--ridge", type=float, default=1e-3)
    calibrate.add_argument("--margin", type=float, default=0.25)
    calibrate.add_argument("--out", help="写出矩阵 JSON")
    calibrate.add_argument("--json", action="store_true")
    calibrate.set_defaults(func=cmd_calibrate)

    lut = sub.add_parser("lut", help="导出校准变换 .cube")
    lut.add_argument("--out", required=True)
    lut.add_argument("--size", type=int, default=33)
    lut.add_argument("--include-look", action="store_true",
                     help="把调色层（CDL/曲线）也烘进去")
    add_common(lut, with_io=True)
    lut.set_defaults(func=cmd_lut)

    synth = sub.add_parser("synth", help="生成合成测试素材")
    synth.add_argument("--out", required=True)
    synth.add_argument("--kind", default="negative",
                       choices=synth_mod.KINDS,
                       help="scene = 有画面的 C-41 合成负片（橙色罩+串扰），"
                            "并附对照正片")
    synth.add_argument("--width", type=int, default=768)
    synth.add_argument("--height", type=int, default=512)
    synth.add_argument("--seed", type=int, default=7, help="随机种子")
    synth.add_argument("--border", type=float, default=0.025,
                       help="片基边框比例（未曝光片边，供片基采样）")
    synth.add_argument("--grain", type=float, default=None,
                       help="颗粒强度（密度域）")
    synth.add_argument("--corners", default=None)
    synth.add_argument("--cast", nargs=3, type=float,
                       default=(1.0, 1.0, 1.0), metavar=("R", "G", "B"),
                       help="乘性整体偏色（在密度域会对消，只影响片基颜色）")
    synth.add_argument("--density-gamma", nargs=3, type=float,
                       default=(1.0, 1.0, 1.0), metavar=("R", "G", "B"),
                       help="逐通道密度反差；≠(1,1,1) 才会制造需要中性化的偏色")
    synth.add_argument("--noise", type=float, default=0.0)
    synth.add_argument("--format", default="tiff16",
                       choices=io_write.SUPPORTED_FORMATS)
    synth.add_argument("--json", action="store_true")
    synth.set_defaults(func=cmd_synth)

    info = sub.add_parser("info", help="查看图像/设置文件信息")
    info.add_argument("input")
    info.add_argument("--settings", help="附加显示某个参数快照")
    info.add_argument("--json", action="store_true")
    info.set_defaults(func=cmd_info)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:            # argparse 的用法错误 → 统一退出码
        return int(exc.code) if isinstance(exc.code, int) else EXIT_USAGE
    if not getattr(args, "command", None):
        parser.print_help()
        return EXIT_USAGE
    try:
        return args.func(args)
    except (io_read.ReadError, io_write.WriteError, LutError,
            ProfileError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return EXIT_UNSUPPORTED


__all__ = ["main", "build_parser", "EXIT_OK", "EXIT_PARTIAL", "EXIT_USAGE",
           "EXIT_UNSUPPORTED"]
