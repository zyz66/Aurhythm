"""Aurhythm —— 胶片 Cineon / LogC3 科学校准管线.

包内模块的依赖边界（硬性约定，测试会强制检查）::

    constants / colorimetry / tonemap / lut / profiles / colorchecker
    pipeline / settings / params / synth
        -> 只允许 numpy + 标准库

    io_read    -> 懒加载 rawpy / tifffile / PIL
    io_write   -> 懒加载 tifffile / PIL / imageio
    app        -> 唯一允许 import tkinter 的模块

这样既保证 GUI 之外的所有数值逻辑可以在无 tkinter / 无 rawpy 的环境里
被完全自动化测试，也让 CLI 与批处理无需 GUI 依赖。
"""

__version__ = "5.0.0"
__all__ = ["__version__"]
