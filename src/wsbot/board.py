"""Find the letter board on screen and split it into a rows x cols grid.

Detection keys on the game's own UI (the big white board panel and the black
letter glyphs on it), never on the landscape background, which changes per level.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

WHITE_MIN = 235  # board panel pixels are near-pure white
DARK_MAX = 90  # letter glyphs are near-black
MIN_PANEL_AREA = 300_000  # the board is always the biggest white panel on screen
_OPEN_KERNEL = np.ones((9, 9), np.uint8)


@dataclass
class Cell:
    row: int
    col: int
    center: tuple[int, int]  # screen coords (calibration space)
    glyph: np.ndarray  # binary crop of the letter, white-on-black


@dataclass
class Board:
    panel: tuple[int, int, int, int]  # x, y, w, h
    rows: int
    cols: int
    letter_h: float
    cells: list[Cell] = field(default_factory=list)

    def cell(self, r: int, c: int) -> Cell:
        return self.cells[r * self.cols + c]


def find_panel(img: np.ndarray, scale: float = 1.0) -> tuple[int, int, int, int] | None:
    """Bounding box of the largest near-white connected region, if it's board-sized.

    `scale`: `img` is the calibration frame shrunk by this much (the watcher looks at a
    half-size frame: ~4x less work). The box comes back in calibration pixels."""
    white = cv2.inRange(img, (WHITE_MIN,) * 3, (255,) * 3)
    # Opening erases thin white bridges (e.g. anti-aliased edges of the word-found
    # banner) that would otherwise merge the hint card and the board into one panel.
    kernel = _OPEN_KERNEL if scale == 1.0 else _kernel(scale)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, kernel)
    n, _, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=4)
    if n < 2:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, area = (int(v) for v in stats[i])
    if area < MIN_PANEL_AREA * scale * scale:
        return None
    if scale == 1.0:
        return x, y, w, h
    return round(x / scale), round(y / scale), round(w / scale), round(h / scale)


def _kernel(scale: float) -> np.ndarray:
    k = max(3, round(_OPEN_KERNEL.shape[0] * scale) | 1)
    return np.ones((k, k), np.uint8)


def _cluster(values: list[float], gap: float) -> list[float]:
    """Group sorted 1-D values into clusters split by gaps larger than `gap`; return means."""
    values = sorted(values)
    groups: list[list[float]] = [[values[0]]]
    for v in values[1:]:
        if v - groups[-1][-1] > gap:
            groups.append([v])
        else:
            groups[-1].append(v)
    return [sum(g) / len(g) for g in groups]


def _fills(centers: list[float], extent: int) -> bool:
    """True if evenly spaced rows (or cols) span the panel with matching margins.

    A toast ("You have already collected this word!") or a tutorial box over part of
    the board hides whole rows; what's left still looks like a clean smaller grid, and
    reading it as a new board fakes a level change. Hidden rows show up as a lopsided
    margin (edge rows hidden) or an irregular gap (middle rows hidden).
    """
    gaps = np.diff(centers)
    pitch = float(np.median(gaps))
    if np.any(np.abs(gaps - pitch) > 0.35 * pitch):
        return False
    before, after = centers[0], extent - centers[-1]
    return abs(before - after) <= 0.5 * pitch


def covered_below(img: np.ndarray, panel: tuple[int, int, int, int], scale: float = 1.0) -> bool:
    """True if a flat light box (a toast) sits right under the panel's bottom edge.
    `panel` is in calibration pixels; `img` is the calibration frame shrunk by `scale`.

    The toast is light grey, not board white, so it cuts the panel short and the rows
    above it pass as a whole board. Under a real board there's landscape, never a flat
    light strip.
    """
    px, py, pw, ph = (round(v * scale) for v in panel)
    y = py + ph + max(2, round(6 * scale))
    if y >= img.shape[0]:
        return False
    strip = img[y, px + pw // 5 : px + pw - pw // 5].min(axis=1).astype(np.float32)
    return float(strip.mean()) >= 200 and float(strip.std()) < 8


def find_panel_near(
    img: np.ndarray, near: tuple[int, int, int, int], pad: int = 32
) -> tuple[int, int, int, int] | None:
    """find_panel, looking only around `near` (the panel as a smaller frame saw it):
    the same box for ~1/6 of the work on a phone-sized frame."""
    x, y, w, h = near
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
    panel = find_panel(img[y0:y1, x0:x1])
    if panel is None:
        return None
    return panel[0] + x0, panel[1] + y0, panel[2], panel[3]


def read_board(img: np.ndarray, near: tuple[int, int, int, int] | None = None) -> Board | None:
    """Locate the board and its letter cells. Returns None if no clean grid is visible.
    `near`: roughly where the panel is (see find_panel_near)."""
    panel = find_panel(img) if near is None else find_panel_near(img, near)
    if panel is None:
        return None
    px, py, pw, ph = panel
    roi = img[py : py + ph, px : px + pw]
    dark = cv2.inRange(roi, (0, 0, 0), (DARK_MAX,) * 3)
    _, _, stats, _ = cv2.connectedComponentsWithStats(dark)

    blobs = []
    for x, y, w, h, area in stats[1:]:
        if area < 150 or x == 0 or y == 0 or x + w >= pw or y + h >= ph:
            continue  # specks and the panel's rounded corners
        blobs.append((int(x), int(y), int(w), int(h)))
    if len(blobs) < 9:
        return None

    med_h = float(np.median([b[3] for b in blobs]))
    # Letters share a cap height; Q's tail makes it a bit taller.
    blobs = [b for b in blobs if 0.6 * med_h <= b[3] <= 1.5 * med_h]

    col_xs = _cluster([x + w / 2 for x, _, w, _ in blobs], gap=med_h * 0.8)
    row_ys = _cluster([y + med_h / 2 for _, y, _, _ in blobs], gap=med_h * 0.8)
    rows, cols = len(row_ys), len(col_xs)
    if rows < 3 or cols < 3 or len(blobs) != rows * cols:
        return None
    if not (_fills(row_ys, ph) and _fills(col_xs, pw)) or covered_below(img, panel):
        return None

    grid: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for b in blobs:
        cx, cy = b[0] + b[2] / 2, b[1] + med_h / 2
        c = int(np.argmin([abs(cx - v) for v in col_xs]))
        r = int(np.argmin([abs(cy - v) for v in row_ys]))
        if (r, c) in grid:
            return None  # two glyphs in one cell: not a clean board
        grid[(r, c)] = b

    board = Board(panel=panel, rows=rows, cols=cols, letter_h=med_h)
    for r in range(rows):
        for c in range(cols):
            x, y, w, h = grid[(r, c)]
            center = (px + round(col_xs[c]), py + round(row_ys[r]))
            board.cells.append(Cell(r, c, center, dark[y : y + h, x : x + w].copy()))
    return board


def highlighted(img: np.ndarray, board: Board, r: int, c: int) -> bool:
    """True if a found-word pill covers this cell.

    Samples a patch just above the glyph: plain board is near-white there, a pill is
    a saturated color. The offset stays inside the pill for diagonal words too.
    """
    cx, cy = board.cell(r, c).center
    y = cy - int(board.letter_h * 0.68)
    patch = img[y - 3 : y + 4, cx - 3 : cx + 4].reshape(-1, 3)
    return int(patch.min(axis=1).mean()) < WHITE_MIN - 10
