"""iOS backend: an iPhone over USB with pymobiledevice3 (see iphone.py), iOS 27+.

The calibration space is the iPhone screen scaled by K, chosen so the game's UI comes
out the same pixel size as on the 1080x2400 Android calibration space (the SE's UI
is ~1.2x smaller than Android's at equal width). Popup templates and the board
reader's size thresholds then carry over; only fixed points and zones are per-platform.

Serial: `ios` (the only / first iPhone) or `ios:UDID`.
"""

from __future__ import annotations

import threading
from typing import ClassVar

import cv2
import numpy as np

from .device import BaseDevice, SafetyError
from .log import log

K = 1.728  # iPhone px -> calibration px
GAME = "in.playsimple.wordsearch"


def _box(x1: float, y1: float, x2: float, y2: float) -> tuple[int, int, int, int]:
    """iPhone px rectangle -> calibration space."""
    return round(x1 * K), round(y1 * K), round(x2 * K), round(y2 * K)


class IOSGameDevice(BaseDevice):
    platform = "ios"
    # iPhone SE (750x1334) positions, measured on game frames.
    zones: ClassVar[dict[str, tuple[int, int, int, int]]] = {
        "back_button": _box(25, 45, 95, 112),
        "star_bonus_jar": _box(96, 45, 165, 112),
        "coins_and_shop": _box(530, 45, 725, 112),
        "banner_ad": _box(0, 1228, 750, 1334),  # AdMob banner (INSTALL) on menu screens
        "video_2x": _box(96, 278, 210, 342),  # tournament "Get 2x" = watch a video ad
        "remove_ads": _box(658, 285, 740, 365),  # "no ADS" (level end); popup X at x~635
    }
    board_center = (round(375 * K), round(720 * K))
    above_board = (round(375 * K), round(330 * K))
    templates = "templates/ios"
    swipe_ms = 50  # ~3 frames, 6 touch samples: 17 ms registered but misfired now and then

    def __init__(self, udid: str | None = None, *, dry_run: bool = False) -> None:
        from .iphone import IPhone

        self.phone = IPhone(udid)
        self.phone.open()
        self.serial = f"ios:{self.phone.udid}"
        self.dry_run = dry_run
        self.input_lock = threading.Lock()
        self.width, self.height = self.phone.width, self.phone.height
        self.calib = (round(self.width * K), round(self.height * K))
        self.below_board_y = None
        self.taps = self.swipes = self.refused = 0
        self._seq = -1
        self._not_running = 0
        log(
            "DEVICE",
            f"connected {self.serial} {self.width}x{self.height} "
            f"calib {self.calib[0]}x{self.calib[1]} dry_run={dry_run}",
        )

    def _px(self, x: float, y: float) -> tuple[float, float]:
        return x / K, y / K

    # ---- frames -------------------------------------------------------------

    def frame(self) -> np.ndarray:
        """Latest screen frame in calibration space. The phone sends frames only when
        the screen changes, so wait briefly for a new one instead of spinning."""
        if not self.phone.alive:
            raise RuntimeError("iPhone USB session is down")
        self.phone.wait_newer(self._seq, 0.1)
        self._seq, _, img = self.phone.latest()
        return cv2.resize(img, self.calib, interpolation=cv2.INTER_LINEAR)

    def reconnect(self) -> None:
        """Frames failing: reopen the USB session (stream server + touch)."""
        try:
            self.phone.reopen()
        except Exception as exc:
            log("ERROR", f"iPhone reconnect failed: {exc!r}")

    # ---- app ----------------------------------------------------------------

    def _app(self, op: str, bundle: str) -> int:
        return self.phone.app(op, bundle)

    def foreground(self) -> str:
        """The game's bundle id while its process runs, else springboard ('' if unknown).
        iOS has no cheap "frontmost app" query; a backgrounded game is caught by the
        bot's no-board recovery instead."""
        try:
            running = self._app("pid", GAME)
        except Exception as exc:
            log("WARN", f"foreground check failed: {exc!r}")
            return ""
        # One "not running" answer came back while the game was on screen, and a
        # relaunch kills the running copy: only believe it three checks in a row.
        self._not_running = 0 if running else self._not_running + 1
        return "com.apple.springboard" if self._not_running >= 3 else GAME

    def app_start(self, package: str) -> None:
        log("APP", f"launch {package} -> pid {self._app('launch', package)}")

    def app_stop(self, package: str) -> None:
        self._app("kill", package)

    # ---- input --------------------------------------------------------------

    def tap(self, x: int, y: int, *, why: str = "", allow: str | None = None) -> bool:
        try:
            self._check_safe(x, y, "tap", allow)
        except SafetyError:
            return False
        log("TAP", f"({x},{y}) {why}".rstrip())
        if self.dry_run:
            return True
        with self.input_lock:
            self.phone.tap(*self._px(x, y))
        self.taps += 1
        return True

    def swipe(
        self, start: tuple[int, int], end: tuple[int, int], ms: int, *, why: str = ""
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
            self.phone.swipe(*self._px(*start), *self._px(*end), ms=ms)
        self.swipes += 1
        log("SWIPE", why)
        return True

    def hold(self, start: tuple[int, int], end: tuple[int, int]) -> bool:
        """Put a finger down at start and drag to end without lifting. Pair with release()."""
        try:
            self._check_safe(*start, "hold start")
            self._check_safe(*end, "hold end")
        except SafetyError:
            return False
        if self.dry_run:
            return True
        (sx, sy), (ex, ey) = self._px(*start), self._px(*end)
        # One sample per frame (17 ms): a 3-sample, 40 ms drag landed only its first cell,
        # so the liveness probe always read "IGNORED" and restarted the game forever.
        steps = 8
        path = [(sx + (ex - sx) * i / steps, sy + (ey - sy) * i / steps) for i in range(steps + 1)]
        with self.input_lock:
            self.phone.hold(path, step_ms=17)
        return True

    def release(self, end: tuple[int, int]) -> None:
        if self.dry_run:
            return
        with self.input_lock:
            self.phone.release()

    def close(self) -> None:
        self.phone.close()
