"""``python -m aurhythm``：无参数进 GUI，有参数走命令行。"""

from __future__ import annotations

import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv:
        from .cli import main as cli_main
        return cli_main(argv)
    from .app import main as gui_main
    return gui_main()


if __name__ == "__main__":
    sys.exit(main())
