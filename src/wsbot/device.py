"""Device layer: frames, taps, swipes, and the SAFETY no-tap zones.

Every coordinate in the bot is authored in the 1080x2400 calibration space and
scaled to the live device here. Every input holds `input_lock`, so the watcher
thread can never tap while the main thread is mid-swipe.

Frames come from uiautomator2 (~115 ms). Input goes through one persistent
`adb shell` running `input swipe`/`input tap` (~50 ms per swipe vs ~330 ms via u2),
because there is no adb process to spawn per gesture.
"""

from __future__ import annotations

import subprocess
import threading

import cv2
import numpy as np
import uiautomator2 as u2

from .log import log

CALIB_W, CALIB_H = 1080, 2400

# Rectangles (x1, y1, x2, y2) in calibration space that must never receive input:
# controls that spend coins or leave the level. The star is the (free) bonus-word jar;
# only a popup entry that names it in "allow" may tap it. The booster row (ads, burst
# hint, lightbulb, shuffle) is covered by the dynamic below-board zone during levels.
FORBIDDEN_ZONES: dict[str, tuple[int, int, int, int]] = {
    "back_button": (30, 140, 160, 270),
    "star_bonus_jar": (165, 140, 280, 270),
    "coins_and_shop": (700, 140, 1050, 270),
}


class SafetyError(RuntimeError):
    """Raised when an input would land in a forbidden zone."""


class ShellInput:
    """A long-lived `adb shell` that runs input commands and waits for each to finish."""

    SENTINEL = "__wsbot_ok__"

    def __init__(self, serial: str) -> None:
        self.serial = serial
        self.proc: subprocess.Popen[str] | None = None

    def _open(self) -> subprocess.Popen[str]:
        if self.proc is None or self.proc.poll() is not None:
            self.proc = subprocess.Popen(
                ["adb", "-s", self.serial, "shell"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        return self.proc

    def run(self, cmd: str) -> None:
        for attempt in range(2):
            proc = self._open()
            try:
                assert proc.stdin and proc.stdout
                proc.stdin.write(f"{cmd}; echo {self.SENTINEL}\n")
                proc.stdin.flush()
                while True:
                    line = proc.stdout.readline()
                    if not line:
                        raise BrokenPipeError("adb shell closed")
                    if line.strip() == self.SENTINEL:
                        return
            except (BrokenPipeError, OSError) as exc:
                log("RECOVERY", f"input shell died ({exc!r}); reopening")
                self.proc = None
                if attempt:
                    raise

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()


class Device:
    def __init__(self, serial: str, *, dry_run: bool = False) -> None:
        self.serial = serial
        self.dry_run = dry_run
        self.input_lock = threading.Lock()
        self.d = u2.connect(serial)
        self.shell = ShellInput(serial)
        w, h = self.d.window_size()
        self.width, self.height = w, h
        self.sx, self.sy = w / CALIB_W, h / CALIB_H
        # While a level is being solved, everything below the board is off limits: the
        # booster row (ads, burst hint, lightbulb, shuffle) sits there and moves down on
        # taller boards. Set by the bot per level; None between levels.
        self.below_board_y: int | None = None
        self.taps = 0
        self.swipes = 0
        self.refused = 0
        log("DEVICE", f"connected {serial} {w}x{h} dry_run={dry_run}")

    def reconnect(self) -> None:
        """Rebuild both device channels after adb/uiautomator trouble."""
        log("RECOVERY", f"reconnecting to {self.serial}")
        try:
            subprocess.run(["adb", "-s", self.serial, "wait-for-device"], timeout=60)
            self.d = u2.connect(self.serial)
        except Exception as exc:
            log("ERROR", f"reconnect failed: {exc!r}")
        self.shell.close()
        self.shell.proc = None

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

    def _check_safe(self, x: int, y: int, what: str, allow: str | None = None) -> None:
        zones = dict(FORBIDDEN_ZONES)
        if self.below_board_y is not None:
            zones["below_board"] = (0, self.below_board_y, CALIB_W, CALIB_H)
        for name, (x1, y1, x2, y2) in zones.items():
            if name != allow and x1 <= x <= x2 and y1 <= y <= y2:
                self.refused += 1
                log("SAFETY", f"refused {what} at ({x},{y}) inside {name}")
                raise SafetyError(name)

    def _scale(self, x: float, y: float) -> tuple[int, int]:
        px = min(max(round(x * self.sx), 0), self.width - 1)
        py = min(max(round(y * self.sy), 0), self.height - 1)
        return px, py

    def tap(self, x: int, y: int, *, why: str = "", allow: str | None = None) -> bool:
        try:
            self._check_safe(x, y, "tap", allow)
        except SafetyError:
            return False
        log("TAP", f"({x},{y}) {why}".rstrip())
        if self.dry_run:
            return True
        with self.input_lock:
            self.shell.run("input tap {} {}".format(*self._scale(x, y)))
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
        sx, sy = self._scale(*start)
        ex, ey = self._scale(*end)
        with self.input_lock:
            self.shell.run(f"input swipe {sx} {sy} {ex} {ey} {ms}")
        self.swipes += 1
        log("SWIPE", why)
        return True
