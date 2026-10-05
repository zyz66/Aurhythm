"""Aurhythm 胶片 Cineon / LogC3 校准器 v5.0 —— 启动入口。

用法::

    python Aurhythm.py                        # 图形界面
    python Aurhythm.py convert --in a.dng --out out/   # 命令行
    python Aurhythm.py --help                 # 命令行帮助

v5.0 起实现拆分到 ``aurhythm/`` 包；本文件只做两件事：

1. 转发入口（无参数 = GUI，有参数 = CLI）；
2. 保持旧的模块级名字（``ScientificFilmPipeline`` / ``CubeLUT`` /
   ``FilmProcessorUI`` / ``ICCLoader`` / ``ColorCheckerCalibration`` /
   ``FILM_PRESETS`` / ``COLORCHECKER_24_D50``）可用，避免破坏已有脚本。

注意：本模块**不在顶层 import rawpy / tifffile / tkinter**。
缺少可选依赖时 GUI 仍能启动、TIFF/PNG 输入仍可用，只有真的用到
对应功能时才给出可读错误（旧版在顶层 import rawpy，缺依赖直接崩溃）。
"""

from __future__ import annotations

import sys

from aurhythm import __version__
from aurhythm.constants import (  # noqa: F401  (兼容再导出)
    CINEON, COLORCHECKER_24_NAMES, COLORCHECKER_24_SRGB, COLORCHECKER_24_SRGB8,
    COLORCHECKER_24_D50_COMPAT as COLORCHECKER_24_D50,
    FILM_PRESETS, HDParams, LOGC3, PRESET_ORDER,
)
from aurhythm.lut import CubeLUT, LutError  # noqa: F401
from aurhythm.pipeline import ScientificFilmPipeline  # noqa: F401


# ======================================================================
# 旧名字兼容层
# ======================================================================

class ICCLoader:
    """兼容旧接口：转发到 :mod:`aurhythm.profiles`。"""

    @staticmethod
    def load_dcp(filepath):
        from aurhythm.profiles import load_dcp
        return load_dcp(filepath)

    @staticmethod
    def load_icc(filepath):
        from aurhythm.profiles import load_icc
        return load_icc(filepath)

    @staticmethod
    def detect_format(filepath):
        from aurhythm.profiles import load_profile
        return load_profile(filepath)


class ColorCheckerCalibration:
    """兼容旧接口：转发到 :mod:`aurhythm.colorchecker`。"""

    @staticmethod
    def load_ccmx(filepath):
        from aurhythm.colorchecker import load_calibration_file
        return load_calibration_file(filepath)

    @staticmethod
    def load_json(filepath):
        from aurhythm.colorchecker import load_calibration_file
        return load_calibration_file(filepath)

    @staticmethod
    def detect_format(filepath):
        from aurhythm.colorchecker import load_calibration_file
        return load_calibration_file(filepath)


def FilmProcessorUI():                                  # noqa: N802
    """兼容旧的 UI 类名（真正的实现在 :class:`aurhythm.app.AurhythmApp`）。"""
    from aurhythm.app import AurhythmApp
    return AurhythmApp()


def apply_dark_theme(root=None):                        # noqa: F401
    """兼容旧接口。"""
    from aurhythm.app import apply_dark_theme as _apply
    return _apply(root)


def _report_dependencies() -> None:
    """打印可选依赖状态（缺少也不影响启动）。"""
    checks = (
        ("rawpy", "读取 RAW（NEF/CR2/ARW/DNG…）"),
        ("tifffile", "TIFF 16/32 位读写（缺失时用内置写出器）"),
        ("PIL", "预览渲染与 PNG/JPEG 输入"),
        ("imageio", "EXR 导出"),
    )
    for module, purpose in checks:
        try:
            __import__(module)
            print(f"  ✓ {module:<10} {purpose}")
        except Exception:                               # noqa: BLE001
            print(f"  ✗ {module:<10} {purpose} —— 缺失时该功能会给出提示")
    try:
        import tkinter  # noqa: F401
        print("  ✓ tkinter    图形界面")
    except Exception:                                   # noqa: BLE001
        print("  ✗ tkinter    图形界面（命令行仍可用）")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv:
        from aurhythm.cli import main as cli_main
        return cli_main(argv)
    print("=" * 64)
    print(f"Aurhythm 胶片 Cineon 校准器 v{__version__}")
    print("=" * 64)
    print("流程: 白平衡 → 相机 profile → 色卡 → 密度 → 中性化 → 解串扰")
    print("      → 胶片特性曲线（非线性校正）→ Cineon / LogC3 编码")
    print("说明: 调色（CDL/曲线）在独立标签页，默认恒等、非校准")
    print("-" * 64)
    print("可选依赖:")
    _report_dependencies()
    print("-" * 64)
    from aurhythm.app import main as gui_main
    return gui_main()


if __name__ == "__main__":
    sys.exit(main())
