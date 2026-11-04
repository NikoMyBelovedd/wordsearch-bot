"""Where the FREE hint button is on a frame: the lightbulb in the booster row, found by
template match on its orange disc.

The "Use the HINT" coach bubble's arrow points at this button, but a fixed offset from the
bubble only lands on it at one phone's size: the bubble and the row move with the frame's
height and aspect. So the tap is the disc's own centre, found on the frame the entry matched.
The template (templates/anchors/hint.png) is the disc's inner square, so the level's
background never counts; the white bulb and the FREE word make it unlike the coin-cost
burst button (which scores ~0.62 here, below MIN_SCORE).

Calibration px (1080x2400), like topbar.py. Searched in the lower half of the frame, where
the booster row sits above the ad banner.
"""

from __future__ import annotations

import cv2
import numpy as np

SCALES = (0.94, 0.97, 1.0, 1.03)  # the stretch to calibration size moves the disc a few %
MIN_SCORE = 0.85
TOP, BOTTOM = 0.5, 0.9  # share of the frame's height that is searched


def find_hint(frame: np.ndarray, icon: np.ndarray) -> tuple[int, int] | None:
    """Centre of the FREE hint disc on a calibration-size BGR frame, or None if no disc
    matches well enough (the caller then doesn't tap). `icon`: the template crop (BGR)."""
    h = frame.shape[0]
    y0 = round(TOP * h)
    region = frame[y0 : round(BOTTOM * h)]
    best = (-1.0, None)
    for scale in SCALES:
        tmpl = cv2.resize(icon, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if tmpl.shape[0] > region.shape[0] or tmpl.shape[1] > region.shape[1]:
            continue
        res = cv2.matchTemplate(region, tmpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        if score > best[0]:
            centre = (loc[0] + tmpl.shape[1] // 2, y0 + loc[1] + tmpl.shape[0] // 2)
            best = (float(score), centre)
    score, centre = best
    return centre if score >= MIN_SCORE else None
