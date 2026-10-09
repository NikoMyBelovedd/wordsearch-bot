"""Device layer: frames, taps, swipes, and the SAFETY no-tap zones.

Every coordinate in the bot is authored in the 1080x2400 calibration space and
scaled to the live device here. The game fits the screen's width, so a phone of
another shape is scaled by its width alone (calibration space 1080 wide, as tall as
its aspect makes it): the UI keeps the size the templates were cut at, and the top
bar's no-tap zones stay where its buttons are (measured from the top). Every input
holds `input_lock`, so the watcher thread can never tap while the main thread is
mid-swipe.

Frames come from uiautomator2 (~115 ms). Input goes through one persistent
`adb shell` running `input swipe`/`input tap` (~50 ms per swipe vs ~330 ms via u2),
because there is no adb process to spawn per gesture. The adb layer itself (deadlines,
screencap, apps, the input shell) is the shared `android.py` from ahq-device.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import cv2
import numpy as np

from .android import Android, DeviceError, InputTimeout, ShellInput
from .log import log
from .shot import Shot

__all__ = [
    "CALIB_H",
    "CALIB_W",
    "FORBIDDEN_ZONES",
    "BaseDevice",
    "Device",
    "DeviceError",
    "InputTimeout",
    "SafetyError",
    "ShellInput",
    "calib_size",
    "open_device",
]

CALIB_W, CALIB_H = 1080, 2400
U2_TIMEOUT_S = 2.5

# Rectangles (x1, y1, x2, y2) in calibration space that must never receive input:
# controls that spend coins or leave the level. The star is the (free) bonus-word jar;
# only a popup entry that names it in "allow" may tap it. The booster row (ads, burst
# hint, lightbulb, shuffle) is covered by the dynamic below-board zone during levels.
FORBIDDEN_ZONES: dict[str, tuple[int, int, int, int]] = {
    "back_button": (30, 140, 160, 270),
    "star_bonus_jar": (165, 140, 280, 270),
    "coins_and_shop": (700, 140, 1050, 270),
}


def calib_size(width: int, height: int) -> tuple[int, int]:
    """Calibration space for an Android screen: 1080 wide, height by its aspect (a
    1080x2400 phone is 1080x2400; 720x1280 (16:9) is 1080x1920; 1440x3120 is 1080x2340)."""
    return CALIB_W, round(height * CALIB_W / width)


class SafetyError(RuntimeError):
    """Raised when an input would land in a forbidden zone."""


class BaseDevice:
    """What the bot needs from a platform backend. Coordinates are in the backend's
    calibration space (`calib`); each backend scales them to its screen."""

    platform = "android"
    calib: tuple[int, int] = (CALIB_W, CALIB_H)
    zones: dict[str, tuple[int, int, int, int]] = FORBIDDEN_ZONES
    board_center = (540, 1325)  # first clear-tap target before any board was read
    above_board = (540, 620)  # neutral clear-tap target over the hint card
    templates = "templates"  # popups.json + popups/, relative to the repo root
    swipe_ms = 60  # fast-pass finger travel time
    dry_run = False
    below_board_y: int | None = None
    last_panel: tuple[int, int, int, int] | None = None  # board of the latest level
    refused = 0

    def set_dynamic_zone(self, name: str, rect: tuple[int, int, int, int] | None) -> None:
        """A no-tap zone that exists only while something (an ad button) is on screen."""
        dyn = self.__dict__.setdefault("dynamic_zones", {})
        if rect is None:
            dyn.pop(name, None)
        else:
            dyn[name] = rect

    def set_zones_due(self, due: Callable[[int | None], None] | None) -> None:
        """A callback that brings lazily kept no-tap zones up to date before an input
        at height y (None: any) is checked (see PopupWatcher._banner_due)."""
        self.__dict__["_zones_due"] = due

    def sync_zones(self, y: int | None = None) -> None:
        due = self.__dict__.get("_zones_due")
        if due is not None:
            due(y)

    def _check_safe(self, x: int, y: int, what: str, allow: str | None = None) -> None:
        self.sync_zones(y)
        zones = dict(self.zones)
        zones.update(self.__dict__.get("dynamic_zones", {}))
        if self.below_board_y is not None:
            zones["below_board"] = (0, self.below_board_y, *self.calib)
        for name, (x1, y1, x2, y2) in zones.items():
            if name != allow and x1 <= x <= x2 and y1 <= y <= y2:
                self.refused += 1
                log("SAFETY", f"refused {what} at ({x},{y}) inside {name}")
                raise SafetyError(name)

    def view_stale(self) -> bool:
        """The latest frame may not show the screen as it is now (iOS stream hiccup)."""
        return False

    def shot(self) -> Shot:
        """The current screen, sized on demand. Backends that number their frames
        override this so an unchanged screen costs nothing."""
        self.__dict__["_shot_seq"] = seq = self.__dict__.get("_shot_seq", 0) + 1
        return Shot(seq, self.frame(), self.calib)

    def close(self) -> None:
        pass


def open_device(serial: str, *, dry_run: bool = False) -> BaseDevice:
    """`ios` / `ios:UDID` = an iPhone over USB (pymobiledevice3); anything else = an adb serial."""
    if serial == "ios" or serial.startswith("ios:"):
        from .ios_device import IOSGameDevice

        return IOSGameDevice(serial.partition(":")[2] or None, dry_run=dry_run)
    return Device(serial, dry_run=dry_run)


class Device(BaseDevice):
    def __init__(self, serial: str, *, dry_run: bool = False) -> None:
        self.serial = serial
        self.dry_run = dry_run
        self.input_lock = threading.Lock()
        import uiautomator2 as u2  # Android only: an iPhone bot needn't load it (~20 MB)

        self.d = u2.connect(serial)
        # adb with a deadline on every call (10 s, screenshots 8 s), and the input shell.
        self.phone = Android(serial, dry_run=dry_run, timeout=10.0, shot_timeout=8.0)
        self.shell = ShellInput(serial, adb=self.phone.adb)
        self._u2_busy = threading.Event()
        self._u2_retry_at = 0.0
        w, h = self.d.window_size()
        self.width, self.height = w, h
        self.calib = calib_size(w, h)
        self.sx = self.sy = w / CALIB_W
        # Before any board was read: the screen's middle, wherever this shape puts it.
        self.board_center = (CALIB_W // 2, round(self.calib[1] * 1325 / CALIB_H))
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
            self.phone.wait_for_device(timeout=60)
            import uiautomator2 as u2

            self.d = u2.connect(self.serial)
        except Exception as exc:
            log("ERROR", f"reconnect failed: {exc!r}")
        self.shell.close()
        self.shell.proc = None

    # ---- frames -------------------------------------------------------------

    def frame(self) -> np.ndarray:
        """Current screen as a BGR array in calibration space.

        uiautomator2 is fastest (~115 ms) but silently restarts its on-device server
        when unhappy, which can block for 30 s+. So each u2 grab runs under a guard:
        past U2_TIMEOUT_S we switch to raw `screencap` (~240 ms, hard timeout) and give
        u2 time to recover in the background.
        """
        img = None
        if time.monotonic() >= self._u2_retry_at and not self._u2_busy.is_set():
            img = self._u2_frame()
        if img is None:
            img = self.phone.screencap()
        if (img.shape[1], img.shape[0]) != self.calib:
            img = cv2.resize(img, self.calib, interpolation=cv2.INTER_AREA)
        return img

    def _u2_frame(self) -> np.ndarray | None:
        box: list[np.ndarray] = []

        def grab() -> None:
            try:
                box.append(self.d.screenshot(format="opencv"))
            except Exception as exc:
                log("WARN", f"u2 screenshot failed: {exc!r}")
            finally:
                self._u2_busy.clear()

        self._u2_busy.set()
        worker = threading.Thread(target=grab, daemon=True, name="u2-frame")
        worker.start()
        worker.join(U2_TIMEOUT_S)
        if box:
            return box[0]
        if worker.is_alive():
            log("WARN", f"u2 screenshot stuck >{U2_TIMEOUT_S}s; using screencap for a while")
        self._u2_retry_at = time.monotonic() + 60
        return None

    def foreground(self) -> str:
        """Package of the resumed activity ('' if unknown)."""
        return self.phone.foreground_package()

    def app_start(self, package: str) -> None:
        self.phone.start_app(package)

    def app_stop(self, package: str) -> None:
        try:
            self.phone.stop_app(package)
        except DeviceError as exc:  # the start that follows is still worth trying
            log("WARN", f"stopping {package} failed: {exc}")

    def close(self) -> None:
        self.shell.close()

    # ---- input --------------------------------------------------------------

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

    def hold(self, start: tuple[int, int], end: tuple[int, int]) -> bool:
        """Put a finger down at start and drag to end without lifting. Pair with release()."""
        try:
            self._check_safe(*start, "hold start")
            self._check_safe(*end, "hold end")
        except SafetyError:
            return False
        if self.dry_run:
            return True
        sx, sy = self._scale(*start)
        ex, ey = self._scale(*end)
        mx, my = (sx + ex) // 2, (sy + ey) // 2
        with self.input_lock:
            self.shell.run(
                f"input motionevent DOWN {sx} {sy}; input motionevent MOVE {mx} {my}; "
                f"input motionevent MOVE {ex} {ey}"
            )
        return True

    def release(self, end: tuple[int, int]) -> None:
        if self.dry_run:
            return
        with self.input_lock:
            self.shell.run("input motionevent UP {} {}".format(*self._scale(*end)))
