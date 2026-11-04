"""Tutorial arrows: a dimmed screen where a white arrow points at one undimmed target.

The game shows some tutorials by dimming the whole screen, leaving one icon or button
bright and drawing a white arrow at it: "New Avatar pieces available. Tap to collect!"
(a compass at the left edge), the rank trophy, the Daily Challenge calendar. Only tapping
that target closes them, so the watcher taps it instead of clearing the screen blind.

Everything is measured on the frame: sizes are fractions of its width or height, and the
brightness tests compare with the frame's own dim level, so the same rules hold on any
iPhone or Android screen size and aspect ratio. No fixed coordinates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

WORK_W = 540  # the frame is measured at this width; the sizes below are fractions of it
DIM_MAX_RATIO = 0.5  # a dimmed screen: its median is under this share of its brightest part
ARROW_MIN_V = 150  # the arrow is white: at least this bright ...
ARROW_RELATIVE = 0.9  # ... and this share of the frame's brightest part (the arrow itself)
ARROW_MAX_S = 130  # and not coloured (the toast and arrow carry a faint tint)
TARGET_MIN_V = 120  # undimmed: the dimmed buttons read 55-95 on the tester's frames
ARROW_LONG = (0.06, 0.30)  # the arrow's length, as a share of the screen width
ARROW_ASPECT = (1.4, 4.0)  # length over thickness: a shaft with a head, not a blob
ARROW_FILL = (0.25, 0.6)  # its area over its box: a triangle head on a shaft
ARROW_THICK = 0.05  # its head is this thick (share of the width): rings and dots are not
ARROW_BINS = 24  # the arrow is cut into this many slices along its length
SHAFT_BINS = 6  # the shaft is a long plateau: the end whose outer slices are this many
SHAFT_FLAT = 0.25  # flat (their spread is this share of the head's thickness) ...
SHAFT_RANGE = (0.15, 0.5)  # ... and this share of the head's thickness; the tip is the other end
TAPER_SLACK = 0.1  # from the head's widest slice to the tip it only narrows (edge noise)
TARGET_SIDE = 0.025  # the smallest side of a target, as a share of the width
TARGET_AREA = (0.0004, 0.08)  # a target's area as a share of the frame (a toast is bigger)
CONE_COS = math.cos(math.radians(35))  # the target lies within 35 degrees of the arrow
REACH = 0.5  # and within this many frame heights of the arrow's tip
HALO = 2  # px (work scale) the arrow's anti-aliased edge can spread into the target mask
# The bottom-right booster row (the 200-coin burst hint, the lightbulb, shuffle and the video
# GET buttons) is never a tutorial's target: it is priced or an ad. Fractions of the frame,
# from its left and top edge. The Daily Challenge calendar sits at the far left, outside it.
PAID_ROW = (0.4, 0.75)


@dataclass(frozen=True)
class Target:
    center: tuple[int, int]  # frame px: where to tap
    tip: tuple[int, int]  # frame px: where the arrow points from


@dataclass(frozen=True)
class _Arrow:
    mask: np.ndarray  # work-scale bool mask of the whole arrow
    tip: tuple[float, float]  # work px
    toward: tuple[float, float]  # unit vector from the shaft to the tip


def find_target(img: np.ndarray) -> Target | None:
    """The tap point of a tutorial arrow on this frame, or None: no dimmed screen, no
    white arrow, or no undimmed target where the arrow points."""
    if img is None or img.size == 0:
        return None
    small, scale = _work(img)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    v, s = hsv[..., 2], hsv[..., 1]
    arrow = _find_arrow(v, s) if _dimmed(v) else None
    if arrow is None:
        return None
    center = _find_target(v, arrow)
    if center is None:
        return None
    return Target(center=_frame_px(center, scale), tip=_frame_px(arrow.tip, scale))


def _work(img: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    scale = WORK_W / w
    small = cv2.resize(img, (WORK_W, max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    return small, scale


def _frame_px(point: tuple[float, float], scale: float) -> tuple[int, int]:
    return round(point[0] / scale), round(point[1] / scale)


def _brightest(v: np.ndarray) -> float:
    return float(np.percentile(v[::2, ::2], 99))


def _dimmed(v: np.ndarray) -> bool:
    """Most of the screen is far darker than its brightest part (the white arrow)."""
    return float(np.median(v[::2, ::2])) <= DIM_MAX_RATIO * _brightest(v)


def _find_arrow(v: np.ndarray, s: np.ndarray) -> _Arrow | None:
    height, width = v.shape
    floor = max(ARROW_MIN_V, ARROW_RELATIVE * _brightest(v))
    lit = ((v >= floor) & (s <= ARROW_MAX_S)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(lit, connectivity=8)
    for label in range(1, count):
        x, y, w, h, _area = stats[label]
        if not _could_be_arrow(w, h, width):
            continue
        crop = labels[y : y + h, x : x + w] == label
        found = _arrow_of(crop, width)
        if found is None:
            continue
        tip, toward = found
        mask = np.zeros((height, width), bool)
        mask[y : y + h, x : x + w] = crop
        return _Arrow(mask=mask, tip=(tip[0] + x, tip[1] + y), toward=toward)
    return None


def _could_be_arrow(w: int, h: int, width: int) -> bool:
    """A cheap box test before the arrow's shape is measured (the box of a tilted arrow
    is bigger than the arrow, hence the loose upper bound)."""
    return ARROW_LONG[0] * width * 0.5 <= max(w, h) <= ARROW_LONG[1] * width * 1.5


def _arrow_of(
    mask: np.ndarray, width: int
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """(tip, toward) of a white arrow shape in a crop, or None."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    points = np.vstack(contours).reshape(-1, 2).astype(np.float32)
    box = cv2.boxPoints(cv2.minAreaRect(points))  # the smallest box, at any angle
    edges = [box[1] - box[0], box[2] - box[1]]
    long_edge = max(edges, key=lambda e: float(np.hypot(*e)))
    long = float(np.hypot(*long_edge))
    short = min(float(np.hypot(*e)) for e in edges)
    if not _arrow_shape(long, short, float(mask.sum()), width):
        return None
    return _tip_of(mask, long_edge / long, width)


def _arrow_shape(long: float, short: float, area: float, width: int) -> bool:
    if short <= 0 or not ARROW_LONG[0] * width <= long <= ARROW_LONG[1] * width:
        return False
    if not ARROW_ASPECT[0] <= long / short <= ARROW_ASPECT[1]:
        return False
    return ARROW_FILL[0] <= area / (long * short) <= ARROW_FILL[1]


def _tip_of(
    mask: np.ndarray, axis: np.ndarray, width: int
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """The shaft is a flat, thin end and the head a wider triangle that tapers to the tip.
    The thickness is measured across the axis, in slices along it."""
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    along = pts @ axis
    across = pts @ np.array([-axis[1], axis[0]])
    lo, hi = along.min(), along.max()
    if hi - lo < 1:
        return None
    slices = np.minimum(((along - lo) / (hi - lo) * ARROW_BINS).astype(int), ARROW_BINS - 1)
    thickness = np.zeros(ARROW_BINS)
    for k in range(ARROW_BINS):
        sel = across[slices == k]
        if sel.size:
            thickness[k] = sel.max() - sel.min()
    if thickness.max() < ARROW_THICK * width:
        return None
    shaft = _shaft_end(thickness)
    if shaft is None:
        return None
    tip_slice = ARROW_BINS - 1 if shaft == 0 else 0
    if not _tapers(thickness, int(np.argmax(thickness)), tip_slice):
        return None
    toward = axis if shaft == 0 else -axis
    tip = pts[slices == tip_slice].mean(axis=0)
    return (float(tip[0]), float(tip[1])), (float(toward[0]), float(toward[1]))


def _shaft_end(thickness: np.ndarray) -> int | None:
    """0 if the shaft is at the low end of the axis, 1 if at the high end, None if neither
    end is a flat shaft (or both are: a plain box, not an arrow)."""
    top = thickness.max()
    at_low = _is_shaft(thickness[:SHAFT_BINS], top)
    at_high = _is_shaft(thickness[-SHAFT_BINS:], top)
    if at_low == at_high:
        return None
    return 0 if at_low else 1


def _tapers(thickness: np.ndarray, peak: int, tip: int) -> bool:
    """From the head's widest slice to the tip the arrow only narrows. A wheel or a letter
    bulges back out along the way, so it is not an arrow."""
    path = thickness[peak : tip + 1] if tip >= peak else thickness[tip : peak + 1][::-1]
    return bool(np.all(np.diff(path) <= TAPER_SLACK * thickness.max()))


def _is_shaft(outer: np.ndarray, top: float) -> bool:
    mean = float(outer.mean())
    if mean <= 0:
        return False
    flat = float(outer.max() - outer.min()) <= SHAFT_FLAT * top
    return flat and SHAFT_RANGE[0] * top <= mean <= SHAFT_RANGE[1] * top


def _find_target(v: np.ndarray, arrow: _Arrow) -> tuple[float, float] | None:
    height, width = v.shape
    vivid = (v >= TARGET_MIN_V).astype(np.uint8)
    halo = cv2.dilate(arrow.mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=HALO)
    vivid[halo > 0] = 0  # the arrow itself is not a target
    count, labels, stats, _ = cv2.connectedComponentsWithStats(vivid, connectivity=8)
    best: tuple[float, tuple[float, float]] | None = None
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if not _target_size(w, h, area, width, height):
            continue
        cx, cy = _icon_center(labels[y : y + h, x : x + w] == label)
        center = (cx + x, cy + y)
        if _paid_row(center, width, height):
            continue
        dist = math.hypot(center[0] - arrow.tip[0], center[1] - arrow.tip[1])
        if dist == 0 or dist > REACH * height or _toward(center, arrow) < CONE_COS:
            continue
        if best is None or dist < best[0]:
            best = (dist, center)
    return None if best is None else best[1]


def _paid_row(center: tuple[float, float], width: int, height: int) -> bool:
    return center[0] >= PAID_ROW[0] * width and center[1] >= PAID_ROW[1] * height


def _target_size(w: int, h: int, area: int, width: int, height: int) -> bool:
    return (
        min(w, h) >= TARGET_SIDE * width
        and TARGET_AREA[0] <= area / (width * height) <= TARGET_AREA[1]
    )


def _toward(center: tuple[float, float], arrow: _Arrow) -> float:
    vx, vy = center[0] - arrow.tip[0], center[1] - arrow.tip[1]
    norm = math.hypot(vx, vy)
    return (vx * arrow.toward[0] + vy * arrow.toward[1]) / norm if norm else 0.0


def _icon_center(mask: np.ndarray) -> tuple[float, float]:
    """The centre of an icon's filled outline (crop px): holes (a dial, a letter) and a
    small badge on its corner don't move it."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hull = cv2.convexHull(np.vstack(contours))
    filled = np.zeros(mask.shape, np.uint8)
    cv2.fillConvexPoly(filled, hull, 1)
    inside = cv2.distanceTransform(filled, cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(np.argmax(inside)), inside.shape)
    return float(x), float(y)
