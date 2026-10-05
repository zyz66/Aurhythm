"""测试包：把仓库根目录放进 sys.path，保证任意 cwd 都能 import aurhythm。

用法（在 Aurhythm-main/ 目录下）::

    python -m unittest discover -s tests -t . -v

只需要 numpy（与可选的 Pillow）。没有 tkinter / rawpy / tifffile 也能全绿。
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
