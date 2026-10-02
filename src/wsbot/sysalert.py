"""The iPhone's own pop-ups over the game: system alerts and notification banners.

A system alert ("'Watch' Would Like to Send You Notifications", Low Battery, Software
Update, Trust This Computer?...) is iOS's, not the game's, so no game template knows
it. iOS 26/27 ("Liquid Glass") draws every one the same way: a gray rounded box
centered on a dimmed screen, and 1-3 gray pill buttons in a row or stacked. This
module finds that shape (no text needed), reads the button labels with Tesseract,
and picks a button only from SAFE_LABELS: the ones that dismiss without granting,
buying, installing or changing anything. Everything else is never tapped.

A notification banner slides over the top of the screen for a few seconds. Tapping
it opens the app that sent it, so while one is up its area is a no-tap zone.

All found boxes are in the image's own pixels (x0, y0, x1, y1).
"""

from __future__ import annotations

import itertools
import os
import re
import unicodedata
from dataclasses import dataclass, field

import cv2
import numpy as np

# Work at about iOS point size whatever the phone: an iPhone SE is 375 pt wide.
NORM_W = 375
FLAT_STD = 4.0  # a pixel is "flat" if no color channel varies more than this around it
FLAT_WIN = 5
GRAY_CHROMA = 24  # iOS alert boxes and buttons are gray (max - min channel, mean)
# iOS rounds an alert's box (~30 pt radius) and makes its buttons capsules (radius half
# their height): squares this big at the corners of their bounding boxes are empty.
# The game's white board panel and its "already collected" toast (a gray pill-sized box
# over the board's bottom) have small corner radii: that pair read as an alert.
BOX_CORNER = 6
PILL_CORNER = 0.12  # x the button's height (a capsule's corner is empty to ~0.146)
CORNER_SLACK = 2  # px of a corner square that may still be set (anti-aliasing)

# Buttons a bot may tap, best first. All of them close the alert without saying yes
# to anything. Compared without case, apostrophes or punctuation (see normalize).
SAFE_LABELS = (
    "Don't Allow",
    "Ask App Not to Track",
    "Not Now",
    "Later",
    "Remind Me Later",
    "Close",
    "Cancel",
    "Dismiss",
    "OK",
)
SINGLE_ONLY = {"ok"}  # only when it's the alert's one button (else it may confirm something)
TRUST_LABELS = {"trust", "dont trust"}  # "Trust This Computer?": needs the owner's passcode


def normalize(text: str) -> str:
    """ "Don't  Allow\n" (any apostrophe) -> "dont allow": what OCR reads, compared."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub("['`\u2018\u2019]", "", text.lower())
    return " ".join(re.sub(r"[^a-z0-9% ]+", " ", text).split())


SAFE = [normalize(s) for s in SAFE_LABELS]


def _edits(a: str, b: str) -> int:
    """Levenshtein distance."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def safe_label(label: str) -> str | None:
    """The SAFE entry this OCR'd label is, or None. Exact match after normalize(); a
    label of 8+ letters may be one letter off (OCR), short ones never ("Allow")."""
    norm = normalize(label)
    if not norm:
        return None
    for safe in SAFE:
        if norm == safe or (len(safe) >= 8 and _edits(norm, safe) <= 1):
            return safe
    return None


@dataclass
class Alert:
    box: tuple[int, int, int, int]
    buttons: list[tuple[int, int, int, int]]
    dark: bool  # dark mode: light text on a dark box
    labels: list[str] = field(default_factory=list)  # OCR, one per button ("" = unread)
    title: str = ""

    def center(self, i: int) -> tuple[int, int]:
        x0, y0, x1, y1 = self.buttons[i]
        return (x0 + x1) // 2, (y0 + y1) // 2


@dataclass
class Choice:
    button: int | None  # index into Alert.buttons, None = tap nothing
    label: str  # the safe label chosen, or why nothing is tapped
    trust: bool = False


def choose(labels: list[str], title: str = "") -> Choice:
    """Which button to tap. Only SAFE_LABELS, in their order; OK only when it's the
    single button; never anything on a "Trust This Computer?" alert."""
    norms = [normalize(s) for s in labels]
    if TRUST_LABELS & set(norms) or "trust this computer" in normalize(title):
        return Choice(None, "Trust This Computer? needs a person", trust=True)
    found = {safe_label(s): i for i, s in reversed(list(enumerate(labels)))}
    for safe, label in zip(SAFE, SAFE_LABELS, strict=True):
        if safe in found and not (safe in SINGLE_ONLY and len(labels) != 1):
            return Choice(found[safe], label)
    return Choice(None, "no safe button")


# ---- finding the shapes ------------------------------------------------------------


def _flat(img: np.ndarray) -> np.ndarray:
    """1 where every color channel is nearly constant in a small window (flat fills,
    blurred glass), 0 on edges and text."""
    # cv2 throughout: numpy's float math and max over the channel axis took ~12 ms
    f = img.astype(np.float32)
    k = (FLAT_WIN, FLAT_WIN)
    mean = cv2.blur(f, k)
    var = cv2.subtract(cv2.blur(cv2.multiply(f, f), k), cv2.multiply(mean, mean))
    b, g, r = cv2.split(var)
    _, flat = cv2.threshold(
        cv2.max(cv2.max(b, g), r), FLAT_STD * FLAT_STD, 1, cv2.THRESH_BINARY_INV
    )
    return flat.astype(np.uint8)


def _filled(labels: np.ndarray, i: int, x: int, y: int, w: int, h: int) -> tuple[int, np.ndarray]:
    """Area of component i with its holes (text) filled, and that filled mask (its bbox)."""
    mask = (labels[y : y + h, x : x + w] == i).astype(np.uint8)
    pad = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    outside = pad.copy()
    cv2.floodFill(outside, None, (0, 0), 2)
    filled = (outside[1:-1, 1:-1] != 2).astype(np.uint8)
    return int(filled.sum()), filled


def _round_corners(filled: np.ndarray, k: int) -> bool:
    """The four k x k corner squares of a shape's filled mask (its bbox) are empty."""
    k = max(2, k)
    return all(
        int(c.sum()) <= CORNER_SLACK
        for c in (filled[:k, :k], filled[:k, -k:], filled[-k:, :k], filled[-k:, -k:])
    )


def _chroma(img: np.ndarray, mask: np.ndarray) -> float:
    c = img.max(axis=2).astype(np.int16) - img.min(axis=2)
    return float(c[mask > 0].mean()) if mask.any() else 255.0


@dataclass
class _Comp:
    i: int
    x: int
    y: int
    w: int
    h: int
    area: int


def _components(small: np.ndarray, min_area: int) -> tuple[np.ndarray, list[_Comp]]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(_flat(small), connectivity=4)
    comps = [
        _Comp(i, *(int(v) for v in stats[i]))
        for i in range(1, n)
        if stats[i, cv2.CC_STAT_AREA] >= min_area
    ]
    return labels, comps


def _shrink(img: np.ndarray) -> tuple[np.ndarray, float]:
    s = NORM_W / img.shape[1]
    small = cv2.resize(img, (NORM_W, round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return small, s


def find_alert(img: np.ndarray) -> Alert | None:
    """An iOS system alert's box and buttons in a BGR screen frame, or None."""
    small, s = _shrink(img)
    H, W = small.shape[:2]
    labels, comps = _components(small, 12)
    best = None
    for c in comps:
        if c.area < 0.03 * W * H:
            continue
        cx, cy = c.x + c.w / 2, c.y + c.h / 2
        if not (
            abs(cx - W / 2) < 0.04 * W
            and 0.55 * W <= c.w <= 0.95 * W
            and 90 <= c.h <= 0.75 * H
            and 0.2 * H <= cy <= 0.8 * H
            and c.x > 2
            and c.y > 2
            and c.x + c.w < W - 2
            and c.y + c.h < H - 2
        ):
            continue
        area, filled = _filled(labels, c.i, c.x, c.y, c.w, c.h)
        if area < 0.88 * c.w * c.h or not _round_corners(filled, BOX_CORNER):
            continue
        mask = labels[c.y : c.y + c.h, c.x : c.x + c.w] == c.i
        if _chroma(small[c.y : c.y + c.h, c.x : c.x + c.w], mask) > GRAY_CHROMA:
            continue
        buttons = _buttons(small, labels, comps, c, filled)
        if buttons and (best is None or c.area > best[0].area):
            best = (c, buttons)
    if best is None:
        return None
    c, buttons = best
    inside = small[c.y : c.y + c.h, c.x : c.x + c.w]
    box_color = float(np.median(inside[labels[c.y : c.y + c.h, c.x : c.x + c.w] == c.i]))

    def up(b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        return tuple(round(v / s) for v in b)  # type: ignore[return-value]

    return Alert(
        box=up((c.x, c.y, c.x + c.w, c.y + c.h)),
        buttons=[up(b) for b in buttons],
        dark=box_color < 128,
    )


def _buttons(
    small: np.ndarray, labels: np.ndarray, comps: list[_Comp], box: _Comp, filled: np.ndarray
) -> list[tuple[int, int, int, int]]:
    """The alert's pill buttons inside `box`: gray, flat, 30-64 pt tall, 1-3 in one row
    or stacked, at the bottom of the box. [] if the layout isn't an alert's."""
    pills = []
    for c in comps:
        if c is box or not (
            c.x > box.x and c.y > box.y and c.x + c.w < box.x + box.w and c.y + c.h < box.y + box.h
        ):
            continue
        if not (28 <= c.h <= 64 and c.w >= 1.4 * c.h):
            continue
        area, pill = _filled(labels, c.i, c.x, c.y, c.w, c.h)
        if area < 0.8 * c.w * c.h or area - c.area < 0.01 * c.w * c.h:
            continue  # not a filled pill, or no label inside it
        if not _round_corners(pill, int(PILL_CORNER * c.h)):
            continue  # square-ish ends: not an iOS button (the game's toast)
        mask = labels[c.y : c.y + c.h, c.x : c.x + c.w] == c.i
        if _chroma(small[c.y : c.y + c.h, c.x : c.x + c.w], mask) > GRAY_CHROMA:
            continue
        # inside the box's rounded outline, not in a corner cut-out
        if not filled[c.y - box.y + c.h // 2, max(0, c.x - box.x - 3)]:
            continue
        pills.append(c)
    if not pills or len(pills) > 3:
        return []
    pills.sort(key=lambda c: (c.y, c.x))
    h0 = pills[0].h
    if any(abs(p.h - h0) > 6 for p in pills):
        return []
    one_row = all(abs(p.y - pills[0].y) <= 6 for p in pills)
    stacked = all(p.w >= 0.6 * box.w for p in pills) and all(
        b.y >= a.y + a.h for a, b in itertools.pairwise(pills)
    )
    if not (one_row or stacked):
        return []
    bottom = max(p.y + p.h for p in pills)
    if box.y + box.h - bottom > 32 or pills[0].y < box.y + 0.2 * box.h:
        return []
    return [(p.x, p.y, p.x + p.w, p.y + p.h) for p in pills]


BANNER_TOP = 0.25  # banners are looked for (and found) only in this top share of a frame


def banner_area(img: np.ndarray) -> np.ndarray:
    """The part of a frame find_banner looks at (a view)."""
    return img[: round(img.shape[0] * BANNER_TOP)]


def find_banner(img: np.ndarray) -> tuple[int, int, int, int] | None:
    """A notification banner over the top of the screen: a flat rounded panel almost as
    wide as the screen, just inset from its sides, in the top ~quarter."""
    return find_banner_in(banner_area(img))


def find_banner_in(strip: np.ndarray) -> tuple[int, int, int, int] | None:
    """find_banner, given only the frame's banner_area."""
    top, s = _shrink(strip)
    W = top.shape[1]
    labels, comps = _components(top, 400)
    for c in comps:
        if not (
            0.85 * W <= c.w <= W - 6
            and abs(c.x + c.w / 2 - W / 2) < 0.03 * W
            and c.x >= 3
            and c.y >= 3
            and c.y + c.h < top.shape[0] - 2
            and 40 <= c.h <= 130
        ):
            continue
        area, _ = _filled(labels, c.i, c.x, c.y, c.w, c.h)
        if area < 0.85 * c.w * c.h or area - c.area < 0.02 * c.w * c.h:
            continue  # not a solid panel, or nothing (icon, text) on it
        return tuple(round(v / s) for v in (c.x, c.y, c.x + c.w, c.y + c.h))  # type: ignore[return-value]
    return None


# ---- reading the labels --------------------------------------------------------------


def _ocr_image(crop: np.ndarray, dark: bool, height: int) -> np.ndarray:
    """A crop as Tesseract likes it: black text on white, ~`height` px tall, margin."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if dark or np.median(gray) < 128:
        gray = 255 - gray
    scale = height / max(1, gray.shape[0])
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.copyMakeBorder(bw, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)


def _tesseract(img: np.ndarray, psm: int) -> str:
    import pytesseract  # pulls in PIL: only when an alert is on screen

    from .letters import TESSERACT

    # One thread: OpenMP's worker threads spin for CPU while the bot (and other bots on
    # the computer) keep it busy, and a one-line read is quicker alone anyway.
    os.environ.setdefault("OMP_THREAD_LIMIT", "1")

    pytesseract.pytesseract.tesseract_cmd = TESSERACT
    return pytesseract.image_to_string(img, config=f"--psm {psm}", timeout=10).strip()


def read_labels(img: np.ndarray, alert: Alert) -> list[str]:
    """OCR each button's label (the pill minus its round ends)."""
    out = []
    for x0, y0, x1, y1 in alert.buttons:
        h = y1 - y0
        crop = img[y0 + h // 8 : y1 - h // 8, x0 + h // 3 : x1 - h // 3]
        out.append(_tesseract(_ocr_image(crop, alert.dark, 72), 7) if crop.size else "")
    return out


def read_title(img: np.ndarray, alert: Alert) -> str:
    """The alert's text above its buttons (for the log), one line."""
    x0, y0, x1, _ = alert.box
    bottom = min(b[1] for b in alert.buttons)
    pad = (x1 - x0) // 20
    crop = img[y0 + pad // 2 : bottom - pad // 2, x0 + pad : x1 - pad]
    if crop.size == 0:
        return ""
    scale = 2.0 if crop.shape[1] < 1000 else 1.0
    text = _tesseract(_ocr_image(crop, alert.dark, round(crop.shape[0] * scale)), 6)
    return " ".join(text.split())[:120]
