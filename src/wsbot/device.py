"""Device layer: frames, taps, swipes, and the SAFETY no-tap zones.

Every coordinate in the bot is authored in the 1080x2400 calibration space and
scaled to the live device here. Every input holds `input_lock`, so the watcher
thread can never tap while the main thread is mid-swipe.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np
import uiautomator2 as u2

from .log import log

CALIB_W, CALIB_H = 1080, 2400

# Rectangles (x1, y1, x2, y2) in calibration space that must never receive input.
# These are the controls that spend coins or leave the level.
FORBIDDEN_ZONES: dict[str, tuple[int, int, int, int]] = {
    "back_button": (30, 140, 160, 270),
    "star_hint_button": (165, 140, 280, 270),
    "coins_and_shop": (700, 140, 1050, 270),
    "booster_button": (860, 1850, 1030, 2030),
}


class SafetyError(RuntimeError):
    """Raised when an input would land in a forbidden zone."""


class Device:
    def __init__(self, serial: str, *, dry_run: bool = False) -> None:
        self.serial = serial
        self.dry_run = dry_run
        self.input_lock = threading.Lock()
        self.d = u2.connect(serial)
        w, h = self.d.window_size()
        self.width, self.height = w, h
        self.sx, self.sy = w / CALIB_W, h / CALIB_H
        self.taps = 0
        self.swipes = 0
        self.refused = 0
        log("DEVICE", f"connected {serial} {w}x{h} dry_run={dry_run}")

    # ---- frames -------------------------------------------------------------

    def frame(self) -> np.ndarray:
        """Current screen as a BGR array, resized to calibration space if needed."""
        img = self.d.screenshot(format="opencv")
        if img.shape[1] != CALIB_W or img.shape[0] != CALIB_H:
            img = cv2.resize(img, (CALIB_W, CALIB_H), interpolation=cv2.INTER_AREA)
        return img

    def foreground(self) -> str:
        return self.d.app_current().get("package", "")

    # ---- input --------------------------------------------------------------

    def _check_safe(self, x: int, y: int, what: str) -> None:
        for name, (x1, y1, x2, y2) in FORBIDDEN_ZONES.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.refused += 1
                log("SAFETY", f"refused {what} at ({x},{y}) inside {name}")
                raise SafetyError(name)

    def _scale(self, x: float, y: float) -> tuple[int, int]:
        px = min(max(round(x * self.sx), 0), self.width - 1)
        py = min(max(round(y * self.sy), 0), self.height - 1)
        return px, py

    def tap(self, x: int, y: int, *, why: str = "") -> bool:
        try:
            self._check_safe(x, y, "tap")
        except SafetyError:
            return False
        log("TAP", f"({x},{y}) {why}".rstrip())
        if self.dry_run:
            return True
        with self.input_lock:
            self.d.click(*self._scale(x, y))
        self.taps += 1
        return True

    def swipe(
        self, start: tuple[int, int], end: tuple[int, int], duration: float, *, why: str = ""
    ) -> bool:
        try:
            self._check_safe(*start, "swipe start")
            self._check_safe(*end, "swipe end")
        except SafetyError:
            return False
        if self.dry_run:
            log("SWIPE", f"{start}->{end} {why} (dry)")
            return True
        with self.input_lock:
            sx, sy = self._scale(*start)
            ex, ey = self._scale(*end)
            self.d.swipe(sx, sy, ex, ey, duration=duration)
        self.swipes += 1
        return True
