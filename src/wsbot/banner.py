"""The WORDSEARCH-SOLVER banner, pre-rendered (figlet "ANSI Shadow" and "Small").

`banner(width)` picks the biggest layout that fits: one line, stacked, small, or text.
WORDSEARCH is orange, SOLVER is white.
"""

from __future__ import annotations

from rich.text import Text

ORANGE = "#F97316"
WHITE = "#FFFFFF"

_BIG_WORD = r"""
██╗    ██╗ ██████╗ ██████╗ ██████╗ ███████╗███████╗ █████╗ ██████╗  ██████╗██╗  ██╗
██║    ██║██╔═══██╗██╔══██╗██╔══██╗██╔════╝██╔════╝██╔══██╗██╔══██╗██╔════╝██║  ██║
██║ █╗ ██║██║   ██║██████╔╝██║  ██║███████╗█████╗  ███████║██████╔╝██║     ███████║
██║███╗██║██║   ██║██╔══██╗██║  ██║╚════██║██╔══╝  ██╔══██║██╔══██╗██║     ██╔══██║
╚███╔███╔╝╚██████╔╝██║  ██║██████╔╝███████║███████╗██║  ██║██║  ██║╚██████╗██║  ██║
 ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝╚═════╝ ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝
""".strip("\n").splitlines()

_BIG_DASH = ["", "", "█████╗", "╚════╝", "", ""]

_BIG_SOLVER = r"""
███████╗ ██████╗ ██╗    ██╗   ██╗███████╗██████╗
██╔════╝██╔═══██╗██║    ██║   ██║██╔════╝██╔══██╗
███████╗██║   ██║██║    ██║   ██║█████╗  ██████╔╝
╚════██║██║   ██║██║    ╚██╗ ██╔╝██╔══╝  ██╔══██╗
███████║╚██████╔╝███████╗╚████╔╝ ███████╗██║  ██║
╚══════╝ ╚═════╝ ╚══════╝ ╚═══╝  ╚══════╝╚═╝  ╚═╝
""".strip("\n").splitlines()

_SMALL_WORD = r"""
__      _____  ___ ___  ___ ___   _   ___  ___ _  _
\ \    / / _ \| _ \   \/ __| __| /_\ | _ \/ __| || |
 \ \/\/ / (_) |   / |) \__ \ _| / _ \|   / (__| __ |
  \_/\_/ \___/|_|_\___/|___/___/_/ \_\_|_\\___|_||_|
""".strip("\n").splitlines()

_SMALL_SOLVER = r"""
 ___  ___  _ __   _____ ___
/ __|/ _ \| |\ \ / / __| _ \
\__ \ (_) | |_\ V /| _||   /
|___/\___/|____\_/ |___|_|_\
""".strip("\n").splitlines()


def _pad(lines: list[str]) -> list[str]:
    width = max(len(line) for line in lines)
    return [line.ljust(width) for line in lines]


def _side_by_side(parts: list[tuple[list[str], str]], gap: int = 0) -> Text:
    blocks = [(_pad(lines), style) for lines, style in parts]
    out = Text()
    for row in range(len(blocks[0][0])):
        for i, (lines, style) in enumerate(blocks):
            if i:
                out.append(" " * gap)
            out.append(lines[row], style=f"bold {style}")
        out.append("\n")
    out.rstrip()
    return out


def _stacked(top: list[str], bottom: list[str]) -> Text:
    width = max(len(line) for line in top + bottom)
    out = Text()
    for line in top:
        out.append(line.center(width).rstrip() + "\n", style=f"bold {ORANGE}")
    for line in bottom:
        out.append(line.center(width).rstrip() + "\n", style=f"bold {WHITE}")
    out.rstrip()
    return out


def banner(width: int, max_rows: int = 12) -> Text:
    """The largest banner that fits in `width` columns and `max_rows` rows."""
    one_line = [(_BIG_WORD, ORANGE), (_BIG_DASH, WHITE), (_BIG_SOLVER, WHITE)]
    if width >= 140 and max_rows >= 6:
        return _side_by_side(one_line)
    if width >= 86 and max_rows >= 12:
        return _stacked(_BIG_WORD, _BIG_SOLVER)
    if width >= 84:
        return _side_by_side([(_SMALL_WORD, ORANGE), (_SMALL_SOLVER, WHITE)], gap=2)
    if width >= 54 and max_rows >= 8:
        return _stacked(_SMALL_WORD, _SMALL_SOLVER)
    return text_banner()


def compact_banner(width: int) -> Text:
    """A short banner for the run dashboard."""
    if width >= 84:
        return _side_by_side([(_SMALL_WORD, ORANGE), (_SMALL_SOLVER, WHITE)], gap=2)
    return text_banner()


def text_banner() -> Text:
    out = Text()
    out.append("WORDSEARCH", style=f"bold {ORANGE}")
    out.append("-SOLVER", style=f"bold {WHITE}")
    return out
