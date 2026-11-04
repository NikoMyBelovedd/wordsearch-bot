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
# Faded letters (some levels grey out the filler letters that are in no word) are a
# flat light grey, ~205-212 on every channel: well apart from the white panel (255),
# black glyphs and the colored found-word pills.
FADED_MIN, FADED_MAX = 170, 228
# Every size below is a share of the frame, the panel or the letters, never fixed pixels,
# so the same board reads the same at any screen size.
PANEL_MIN_SHARE = 0.115  # the board is the biggest white panel: at least this share of the frame
SPECK_SHARE = 0.05  # a dark speck is under this share of a letter's height squared
RIM_SHARE = 0.075  # the grey anti-aliased rim round a glyph, as a share of its height
HIGHLIGHT_SHARE = 0.06  # the patch sampled for a pill, as a share of the letter height
COVER_GAP_SHARE = 0.005  # the strip under the panel's edge, as a share of its height
# A hint or found-word mark is a coloured pill with a white letter, where the dark
# glyph mask sees nothing (the Level 121 frame: a red hint H, the one missing letter).
PILL_SAT_MIN, PILL_VAL_MIN = 110, 90  # pill colour: saturated, not dark
PILL_FILL = 0.5  # share of the disc around an empty slot that must be pill colour
PILL_MIN_SIZE, PILL_MAX_SIZE = 1.2, 2.4  # a pill's width and height, in letter heights
LETTER_WHITE_MIN = 200  # the letter inside a pill is white
MIN_LINE = 3  # a row or column needs this many glyphs to belong to the grid


@dataclass
class Cell:
    row: int
    col: int
    center: tuple[int, int]  # screen coords (calibration space)
    glyph: np.ndarray  # binary crop of the letter, white-on-black
    faded: bool = False  # a greyed-out filler letter: in no word, nothing to read


@dataclass
class Board:
    panel: tuple[int, int, int, int]  # x, y, w, h
    rows: int
    cols: int
    letter_h: float
    cells: list[Cell] = field(default_factory=list)

    def cell(self, r: int, c: int) -> Cell:
        return self.cells[r * self.cols + c]


def find_panel(
    img: np.ndarray, scale: float = 1.0, frame_shape: tuple[int, int] | None = None
) -> tuple[int, int, int, int] | None:
    """Bounding box of the largest near-white connected region, if it's board-sized.

    `scale`: `img` is the calibration frame shrunk by this much (the watcher looks at a
    half-size frame: ~4x less work). The box comes back in calibration pixels."""
    white = cv2.inRange(img, (WHITE_MIN,) * 3, (255,) * 3)
    # Opening erases thin white bridges (e.g. anti-aliased edges of the word-found
    # banner) that would otherwise merge the hint card and the board into one panel.
    frame_h, frame_w = frame_shape or img.shape[:2]
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, _kernel(frame_w))
    found = _largest(white)
    if found is None:
        return None
    x, y, w, h, area = found
    if area < PANEL_MIN_SHARE * frame_h * frame_w:
        return None
    if scale == 1.0:
        return x, y, w, h
    return round(x / scale), round(y / scale), round(w / scale), round(h / scale)


def _largest(mask: np.ndarray) -> tuple[int, int, int, int, int] | None:
    """x, y, w, h, area of the biggest 4-connected component of a binary mask (the
    first in label order on a tie), or None if there is none. What
    connectedComponentsWithStats says, for ~1/3 of the work: its stats pass cost 5x
    the labelling, and only one component's box is needed."""
    n, labels = cv2.connectedComponents(mask, connectivity=4)
    if n < 2:
        return None
    areas = np.bincount(labels.ravel(), minlength=n)
    i = 1 + int(np.argmax(areas[1:]))
    x, y, w, h = cv2.boundingRect((labels == i).view(np.uint8))
    return x, y, w, h, int(areas[i])


def _blobs(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """x, y, w, h, area of every 8-connected component of a binary mask, in label
    order: connectedComponentsWithStats' rows for ~40% of its time. Each component
    has one outer border (findContours' top level, also inside another's hole), whose
    bounding box is the component's; its area is its label's pixel count."""
    n, labels = cv2.connectedComponents(mask, connectivity=8)
    if n < 2:
        return []
    areas = np.bincount(labels.ravel(), minlength=n)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c, (_, _, _, parent) in zip(contours, hierarchy[0], strict=True):
        if parent != -1:
            continue  # a hole's border
        x, y, w, h = cv2.boundingRect(c)
        px, py = c[0][0]
        i = int(labels[py, px])
        out.append((i, x, y, w, h, int(areas[i])))
    return [b[1:] for b in sorted(out)]


def _kernel(width: int) -> np.ndarray:
    """Opening kernel: a fixed share of the frame's width, so it scales with the screen."""
    k = max(3, round(width / 120) | 1)
    return np.ones((k, k), np.uint8)


def _lines(values: list[float], gap: float) -> list[list[int]]:
    """Indexes of `values` grouped into lines: sorted values split where a gap exceeds `gap`."""
    if not values:
        return []
    order = sorted(range(len(values)), key=values.__getitem__)
    lines = [[order[0]]]
    for i in order[1:]:
        if values[i] - values[lines[-1][-1]] > gap:
            lines.append([i])
        else:
            lines[-1].append(i)
    return lines


def _mean(values: list[float], line: list[int]) -> float:
    return sum(values[i] for i in line) / len(line)


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
    y = py + ph + max(2, round(ph * COVER_GAP_SHARE))
    if y >= img.shape[0]:
        return False
    strip = img[y, px + pw // 5 : px + pw - pw // 5].min(axis=1).astype(np.float32)
    return float(strip.mean()) >= 200 and float(strip.std()) < 8


def find_panel_near(
    img: np.ndarray, near: tuple[int, int, int, int], pad: int | None = None
) -> tuple[int, int, int, int] | None:
    """find_panel, looking only around `near` (the panel as a smaller frame saw it):
    the same box for ~1/6 of the work on a phone-sized frame."""
    x, y, w, h = near
    if pad is None:
        pad = round(0.03 * max(w, h))  # a margin, a few percent of the panel
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
    panel = find_panel(img[y0:y1, x0:x1], frame_shape=img.shape[:2])
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

    def letters(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
        # The panel's rounded corners touch its edge and are dropped, and so are specks:
        # anything under SPECK_SHARE of the median letter's area.
        found = [
            (x, y, w, h, area)
            for x, y, w, h, area in _blobs(mask)
            if x > 0 and y > 0 and x + w < pw and y + h < ph
        ]
        if not found:
            return []
        letter_h = float(np.median([b[3] for b in found]))
        min_area = SPECK_SHARE * letter_h * letter_h
        return [(int(x), int(y), int(w), int(h)) for x, y, w, h, area in found if area >= min_area]

    blobs = letters(dark)
    board = _grid(img, panel, blobs, dark, None, set())
    if board is not None or not blobs:
        return board
    # Not a full grid of black letters: maybe some are faded. Only looked for now,
    # so a normal board costs what it always did.
    grey = _faded_mask(roi, dark, float(np.median([b[3] for b in blobs])))
    faded = letters(grey)
    if not faded:
        return None
    return _grid(img, panel, blobs + faded, dark, grey, set(faded))


def _grid(
    img: np.ndarray,
    panel: tuple[int, int, int, int],
    blobs: list[tuple[int, int, int, int]],
    dark: np.ndarray,
    grey: np.ndarray | None,
    faded: set[tuple[int, int, int, int]],
) -> Board | None:
    """The board these letter boxes make, if they make a clean grid.

    An empty slot is only allowed to be a coloured pill with a white letter (a hint or a
    found-word mark, see _pill_glyph). Any other empty slot means the board is hidden or
    unreadable, so there's no board to read.
    """
    px, py, pw, ph = panel
    if len(blobs) < 9:
        return None

    med_h = float(np.median([b[3] for b in blobs]))
    # Letters share a cap height; Q's tail makes it a bit taller.
    blobs = [b for b in blobs if 0.6 * med_h <= b[3] <= 1.5 * med_h]
    blobs = _on_lattice(blobs, med_h)
    if len(blobs) < 9:
        return None

    gap = med_h * 0.8
    cxs = [x + w / 2 for x, _, w, _ in blobs]
    cys = [y + med_h / 2 for _, y, _, _ in blobs]
    col_lines, row_lines = _lines(cxs, gap), _lines(cys, gap)
    col_xs = [_mean(cxs, line) for line in col_lines]
    row_ys = [_mean(cys, line) for line in row_lines]
    rows, cols = len(row_ys), len(col_xs)
    if rows < 3 or cols < 3:
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
            center = (px + round(col_xs[c]), py + round(row_ys[r]))
            if (r, c) not in grid:
                glyph = _pill_glyph(img, center, med_h)
                if glyph is None:
                    return None  # an empty slot that is no pill: hidden or unreadable
                board.cells.append(Cell(r, c, center, glyph))
                continue
            x, y, w, h = grid[(r, c)]
            if grey is not None and (x, y, w, h) in faded:
                glyph = grey[y : y + h, x : x + w].copy()
                board.cells.append(Cell(r, c, center, glyph, faded=True))
            else:
                board.cells.append(Cell(r, c, center, dark[y : y + h, x : x + w].copy()))
    return board


def _on_lattice(
    blobs: list[tuple[int, int, int, int]], med_h: float
) -> list[tuple[int, int, int, int]]:
    """The glyphs that sit on a row and a column each holding at least MIN_LINE glyphs.

    A stray (a panel corner, a speck, a toast's text) sits on a row or column of its own,
    so it's dropped before the grid is counted, instead of breaking the count.
    """
    gap = med_h * 0.8
    xs = [x + w / 2 for x, _, w, _ in blobs]
    ys = [y + med_h / 2 for _, y, _, _ in blobs]
    keep = set(range(len(blobs)))
    for values in (xs, ys):
        keep &= {i for line in _lines(values, gap) if len(line) >= MIN_LINE for i in line}
    return [b for i, b in enumerate(blobs) if i in keep]


def _pill_glyph(img: np.ndarray, center: tuple[int, int], med_h: float) -> np.ndarray | None:
    """White-on-black glyph of the white letter in a coloured pill at `center`, or None.

    The pill is the coloured region holding the centre, a disc about 1.8 letter heights
    across. Its outer edge is filled, so the white letter (a hole in the colour) is inside
    it and the white panel round it is not. A banner or a capsule (too big), a plain slot
    (no colour at the centre) or a letter off-centre fails.
    """
    cx, cy = center
    half = round(1.3 * med_h)
    x0, y0 = max(0, cx - half), max(0, cy - half)
    win = img[y0 : cy + half, x0 : cx + half]
    lx, ly = cx - x0, cy - y0
    hsv = cv2.cvtColor(win, cv2.COLOR_BGR2HSV)
    coloured = cv2.inRange(hsv, (0, PILL_SAT_MIN, PILL_VAL_MIN), (180, 255, 255))
    contours, _ = cv2.findContours(coloured, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pill = next(
        (c for c in contours if cv2.pointPolygonTest(c, (float(lx), float(ly)), False) >= 0),
        None,
    )
    if pill is None:
        return None
    _, _, bw, bh = cv2.boundingRect(pill)
    if not all(PILL_MIN_SIZE * med_h <= s <= PILL_MAX_SIZE * med_h for s in (bw, bh)):
        return None
    filled = np.zeros(win.shape[:2], np.uint8)
    cv2.drawContours(filled, [pill], -1, 255, -1)
    disc = np.zeros(win.shape[:2], np.uint8)
    cv2.circle(disc, (lx, ly), round(0.9 * med_h), 255, -1)
    inside = disc > 0
    if ((filled > 0) & inside).sum() < PILL_FILL * inside.sum():
        return None
    white = cv2.inRange(win, (LETTER_WHITE_MIN,) * 3, (255,) * 3) & filled
    for x, y, w, h, area in _blobs(white):
        if area < SPECK_SHARE * med_h * med_h or not 0.6 * med_h <= h <= 1.5 * med_h:
            continue
        if abs(x + w / 2 - lx) <= 0.4 * med_h and abs(y + h / 2 - ly) <= 0.4 * med_h:
            return white[y : y + h, x : x + w].copy()
    return None


def _faded_mask(roi: np.ndarray, dark: np.ndarray, letter_h: float) -> np.ndarray:
    """Light-grey pixels (faded letters), minus the grey anti-aliased rim around
    every black glyph (which would otherwise read as a second glyph in its cell).
    Colored pills fall outside the box on at least one channel."""
    grey = cv2.inRange(roi, (FADED_MIN,) * 3, (FADED_MAX,) * 3)
    r = max(1, round(RIM_SHARE * letter_h))
    grey[cv2.dilate(dark, np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0] = 0
    return grey


def highlighted(img: np.ndarray, board: Board, r: int, c: int) -> bool:
    """True if a found-word pill covers this cell.

    Samples a patch just above the glyph: plain board is near-white there, a pill is
    a saturated color. The offset stays inside the pill for diagonal words too.
    """
    cx, cy = board.cell(r, c).center
    half = max(1, round(board.letter_h * HIGHLIGHT_SHARE))
    y = cy - int(board.letter_h * 0.68)
    patch = img[y - half : y + half + 1, cx - half : cx + half + 1].reshape(-1, 3)
    return int(patch.min(axis=1).mean()) < WHITE_MIN - 10
