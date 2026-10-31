"""Where the game's top bar is on a frame: the back arrow, the bonus-word star and the
shop cart, found by template match on their white icons.

Android frames are read in the 1080x2400 calibration space (device.py stretches every
phone to it), so the buttons are found where they are on that frame. Nothing is a fixed
point: the star's tap and the no-tap zones come from what was found. Matched on the
white icon only: the buttons are green early in the game and pink later, the icons are
the same.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

SCALES = (0.94, 0.97, 1.0, 1.03)  # the stretch to calibration size moves an icon a few %
MIN_SCORE = 0.85  # the back and the star must both match this well
PARTIAL_LOG = 0.6  # a weaker match is logged as "not sure", so a miss can be told apart
SEARCH_LEFT = 0.45  # back and star are searched in the left part of the top quarter
SEARCH_RIGHT = 0.55  # the cart in the right part (from here on)
ROW_TOLERANCE = 0.03  # the icons sit on one row: within this share of the frame's width
SEPARATION = 0.08  # the star sits this far right of the back arrow (home: the gear matched both)


@dataclass(frozen=True)
class TopBar:
    back: tuple[int, int]  # centres, calibration px
    star: tuple[int, int]
    cart: tuple[int, int] | None  # None when the cart wasn't found (zones fall back)


def icon_mask(img: np.ndarray) -> np.ndarray:
    """White, unsaturated pixels (button icons) as a 0/255 mask."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return (((hsv[..., 2] > 200) & (hsv[..., 1] < 70)) * 255).astype(np.uint8)


def _best(region: np.ndarray, icon: np.ndarray) -> tuple[float, tuple[int, int]]:
    """Best score of the icon over the scales, and its centre in the region's half-res px."""
    best = (-1.0, (0, 0))
    for scale in SCALES:
        f = 0.5 * scale  # the top quarter is searched at half size
        tmpl = icon_mask(cv2.resize(icon, None, fx=f, fy=f, interpolation=cv2.INTER_AREA))
        if tmpl.shape[0] > region.shape[0] or tmpl.shape[1] > region.shape[1]:
            continue
        res = cv2.matchTemplate(region, tmpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        if score > best[0]:
            centre = (loc[0] + tmpl.shape[1] // 2, loc[1] + tmpl.shape[0] // 2)
            best = (float(score), centre)
    return best


def locate(frame: np.ndarray, icons: dict[str, np.ndarray]) -> tuple[TopBar | None, str]:
    """The top bar on a calibration-size BGR frame, or None. Also a note of the scores for
    the log. `icons`: "back", "star" and "cart" template crops (BGR)."""
    h, w = frame.shape[:2]
    top = icon_mask(cv2.resize(frame[: h // 4], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
    cut = round(top.shape[1] * SEARCH_RIGHT)
    left, right = top[:, : round(top.shape[1] * SEARCH_LEFT)], top[:, cut:]
    sb, pb = _best(left, icons["back"])
    ss, ps = _best(left, icons["star"])
    sc, pc = _best(right, icons["cart"])
    # half-res centres -> calibration px (the cart's from the right part's own origin)
    back, star, cart = _px(pb), _px(ps), _px((pc[0] + cut, pc[1]))
    note = f"back {sb:.2f} at {back}, star {ss:.2f} at {star}, cart {sc:.2f} at {cart}"
    row = ROW_TOLERANCE * w
    if min(sb, ss) < MIN_SCORE or not (back[0] < 0.3 * w and 0.1 * w < star[0] < 0.45 * w):
        return None, note
    if abs(back[1] - star[1]) > row or star[0] - back[0] < SEPARATION * w:
        return None, note
    if sc < MIN_SCORE or abs(cart[1] - back[1]) > row or cart[0] < 0.6 * w:
        cart = None
    return TopBar(back=back, star=star, cart=cart), note


def _px(half: tuple[int, int]) -> tuple[int, int]:
    return round(half[0] * 2), round(half[1] * 2)
