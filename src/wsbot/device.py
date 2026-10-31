"""Device layer: frames, taps, swipes, and the SAFETY no-tap zones.

Every coordinate in the bot is authored in the 1080x2400 calibration space and
scaled to the live device here. Every input holds `input_lock`, so the watcher
thread can never tap while the main thread is mid-swipe.

Frames come from uiautomator2 (~115 ms). Input goes through one persistent
`adb shell` running `input swipe`/`input tap` (~50 ms per swipe vs ~330 ms via u2),
because there is no adb process to spawn per gesture.
"""

from __future__ import annotations

import os
import re
import subprocess
import threading
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from .debug import snap
from .hintbutton import find_hint
from .imgio import imread
from .log import log
from .shot import Shot
from .topbar import TopBar, locate

CALIB_W, CALIB_H = 1080, 2400
# The Android user the game is installed in (a work profile is 10, Samsung's Secure
# Folder 150). AutomationHQ sets it when the game isn't in the main user; `--android-user`
# sets it too. Without either the bot looks for the first user that has the game.
USER_ENV = "AHQ_ANDROID_USER"
U2_TIMEOUT_S = 2.5

# Rectangles (x1, y1, x2, y2) in calibration space that must never receive input:
# controls that spend coins or leave the level. The star is the (free) bonus-word jar;
# only a popup entry that names it in "allow" may tap it. The booster row (ads, burst
# hint, lightbulb, shuffle) is covered by the dynamic below-board zone during levels.
# Until the top bar is found on a frame (topbar.py) these are the fallback: a band over
# the buttons' row, generous in height (the bar sits about 138 px down on every phone).
# Once found, topbar_zones() puts the zones on the buttons themselves.
FORBIDDEN_ZONES: dict[str, tuple[int, int, int, int]] = {
    "back_button": (30, 60, 160, 240),
    "star_bonus_jar": (165, 60, 280, 240),
    "coins_and_shop": (700, 60, 1050, 240),
}
BUTTON_HALF = 65  # half-size of a top-bar button's no-tap zone (the buttons are ~96 px)
CART_SPAN = 310  # the coin counter and the cart: the zone reaches this far left of the cart
ANCHOR_RETRY_S = 2.0  # look for the top bar this often
ICON_DIR = Path(__file__).resolve().parents[2] / "templates" / "anchors"


def parse_users(listing: str) -> list[int]:
    """User ids from `pm list users` (`UserInfo{10:Work profile:1030} running`), the
    main user (0) first."""
    return sorted({int(m) for m in re.findall(r"UserInfo\{(\d+):", listing)} | {0})


BOUNDS = re.compile(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]")


def view_center(xml: str, resource_id: str) -> tuple[int, int] | None:
    """Centre (device px) of the first view with this resource id in a uiautomator dump."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return None
    for node in root.iter("node"):
        bounds = BOUNDS.fullmatch(node.get("bounds") or "")
        if node.get("resource-id") == resource_id and bounds:
            left, top, right, bottom = map(int, bounds.groups())
            return (left + right) // 2, (top + bottom) // 2
    return None


def topbar_zones(bar: TopBar | None) -> dict[str, tuple[int, int, int, int]]:
    """No-tap zones around the top bar's buttons (the fallback strip without one)."""
    if bar is None:
        return dict(FORBIDDEN_ZONES)

    def around(x: int, y: int) -> tuple[int, int, int, int]:
        return (x - BUTTON_HALF, y - BUTTON_HALF, x + BUTTON_HALF, y + BUTTON_HALF)

    zones = {"back_button": around(*bar.back), "star_bonus_jar": around(*bar.star)}
    if bar.cart is None:
        zones["coins_and_shop"] = FORBIDDEN_ZONES["coins_and_shop"]
    else:
        cx, cy = bar.cart
        zones["coins_and_shop"] = (cx - CART_SPAN, cy - BUTTON_HALF, CALIB_W, cy + BUTTON_HALF)
    return zones


def has_package(path_output: str) -> bool:
    """`pm path` printed an apk: the package is installed for that user."""
    return any(
        line.startswith("package:") and line[8:].strip() for line in path_output.splitlines()
    )


class SafetyError(RuntimeError):
    """Raised when an input would land in a forbidden zone."""


class ShellInput:
    """A long-lived `adb shell` that runs input commands and waits for each to finish."""

    SENTINEL = b"__wsbot_ok__"

    def __init__(self, serial: str) -> None:
        self.serial = serial
        self.proc: subprocess.Popen[bytes] | None = None

    def _open(self) -> subprocess.Popen[bytes]:
        if self.proc is None or self.proc.poll() is not None:
            self.proc = subprocess.Popen(
                ["adb", "-s", self.serial, "shell"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,  # bytes, not text: Windows text pipes would turn \n into \r\n
            )
        return self.proc

    def run(self, cmd: str) -> None:
        for attempt in range(2):
            proc = self._open()
            try:
                assert proc.stdin and proc.stdout
                proc.stdin.write(f"{cmd}; echo {self.SENTINEL.decode()}\n".encode())
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
    # The watcher's latest look has a level board on screen (set in _zones_due). Without a
    # watcher the zone holds as before.
    board_in_view = True
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
        if self.below_board_y is not None and self.board_in_view:
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
        self.shell = ShellInput(serial)
        self._u2_busy = threading.Event()
        self._u2_retry_at = 0.0
        self._dump_busy = threading.Event()
        self._users: dict[str, int] = {}
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
        self.zones = dict(FORBIDDEN_ZONES)  # narrowed to the top bar once it is found
        self._topbar: TopBar | None = None
        self._anchors: dict[str, tuple[int, int]] = {}  # "back", "star": calibration px
        self._anchor_icons = {n: imread(ICON_DIR / f"{n}.png") for n in ("back", "star", "cart")}
        self._hint_icon = imread(ICON_DIR / "hint.png")
        self._last_anchor_try = 0.0
        self._topbar_missed = False
        log("DEVICE", f"connected {serial} {w}x{h} dry_run={dry_run}")

    def adb(self, *args: str, timeout: float = 10.0) -> subprocess.CompletedProcess[bytes]:
        """One-shot adb command with a hard timeout (never hangs the caller)."""
        return subprocess.run(
            ["adb", "-s", self.serial, *args], capture_output=True, timeout=timeout, check=False
        )

    def reconnect(self) -> None:
        """Rebuild both device channels after adb/uiautomator trouble."""
        log("RECOVERY", f"reconnecting to {self.serial}")
        try:
            self.adb("wait-for-device", timeout=60)
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
            img = self._screencap_frame()
        if img.shape[1] != CALIB_W or img.shape[0] != CALIB_H:
            img = cv2.resize(img, (CALIB_W, CALIB_H), interpolation=cv2.INTER_AREA)
        return img

    def shot(self) -> Shot:
        """The frame, and now and then a look for the top bar on it (see _find_topbar)."""
        shot = super().shot()
        now = time.monotonic()
        if now - self._last_anchor_try >= ANCHOR_RETRY_S:
            self._last_anchor_try = now
            self._find_topbar(shot.calib)
        return shot

    def _find_topbar(self, img: np.ndarray) -> None:
        """Locate the top bar on this frame. A frame without it keeps the last one found
        (a popup can cover the bar); until one is found the strip fallback stays up."""
        try:
            bar, note = locate(img, self._anchor_icons)
        except Exception as exc:
            log("ERROR", f"top bar search failed: {exc!r}")
            return
        if bar is None:
            if not self._topbar_missed and self._topbar is None:
                self._topbar_missed = True
                log("DIAG", f"top bar not found ({note}): no taps up there until it is")
            return
        if self._topbar is None:
            log("DEVICE", f"top bar found: back {bar.back} star {bar.star} cart {bar.cart}")
            snap("android_topbar", img, every_s=0)
        self._topbar = bar
        self._anchors = {"back": bar.back, "star": bar.star}
        self.zones = topbar_zones(bar)

    def anchor(self, name: str, shot: Shot | None = None) -> tuple[int, int] | None:
        """A named tap point for popups.json "@name" tap points: the top bar's "back" or
        "star", or "hint", the FREE hint button found on `shot` (the frame the entry matched).
        None when it isn't there, so the entry doesn't tap."""
        if name == "hint":
            return None if shot is None else find_hint(shot.calib, self._hint_icon)
        return self._anchors.get(name)

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

    def _screencap_frame(self) -> np.ndarray:
        raw = self.adb("exec-out", "screencap", timeout=8).stdout
        if len(raw) < 16:
            raise RuntimeError("screencap returned no data")
        w, h = (int(v) for v in np.frombuffer(raw[:8], dtype=np.uint32))
        pixels = np.frombuffer(raw[len(raw) - w * h * 4 :], dtype=np.uint8)
        return cv2.cvtColor(pixels.reshape(h, w, 4), cv2.COLOR_RGBA2BGR)

    def foreground(self) -> str:
        """Package of the resumed activity ('' if unknown)."""
        out = self.adb("shell", "dumpsys activity activities | grep -m1 topResumedActivity")
        match = re.search(r" ([\w.]+)/", out.stdout.decode(errors="ignore"))
        return match.group(1) if match else ""

    def dump_ui(self) -> str | None:
        """The screen's view tree (uiautomator2's XML), or None if it isn't back in time.
        Under a guard thread like the frames: a stuck u2 call must not stall the watcher."""
        if self._dump_busy.is_set():
            return None
        box: list[str] = []

        def grab() -> None:
            try:
                box.append(self.d.dump_hierarchy())
            except Exception as exc:
                log("WARN", f"ui dump failed: {exc!r}")
            finally:
                self._dump_busy.clear()

        self._dump_busy.set()
        worker = threading.Thread(target=grab, daemon=True, name="u2-dump")
        worker.start()
        worker.join(U2_TIMEOUT_S)
        if worker.is_alive():
            log("WARN", f"ui dump stuck >{U2_TIMEOUT_S}s")
        return box[0] if box else None

    def find_view(self, resource_id: str) -> tuple[int, int] | None:
        """Centre of the on-screen view with this resource id, in calibration px."""
        xml = self.dump_ui()
        center = view_center(xml, resource_id) if xml else None
        if center is None:
            return None
        return round(center[0] / self.sx), round(center[1] / self.sy)

    def android_user(self, package: str) -> int:
        """The Android user `package` runs in: the one asked for, else the first
        (main user first) that has it installed, else 0. Asked once per package."""
        if package not in self._users:
            self._users[package] = self._requested_user() or self._find_user(package)
        return self._users[package]

    @staticmethod
    def _requested_user() -> int | None:
        value = os.environ.get(USER_ENV, "").strip()
        return int(value) if value.isdigit() else None

    def _find_user(self, package: str) -> int:
        try:
            users = parse_users(self.adb("shell", "pm", "list", "users").stdout.decode())
            for user in users:
                out = self.adb("shell", "pm", "path", "--user", str(user), package)
                if has_package(out.stdout.decode(errors="ignore")):
                    return user
        except (subprocess.SubprocessError, OSError):
            pass
        return 0

    def _user_args(self, package: str) -> list[str]:
        """`--user N` for another user's game; nothing for the main user."""
        user = self.android_user(package)
        return ["--user", str(user)] if user else []

    def app_start(self, package: str) -> None:
        user = self.android_user(package)
        if not user:  # the main user: what has always worked
            self.adb(
                "shell", "monkey", "-p", package, "-c", "android.intent.category.LAUNCHER", "1"
            )
            return
        # monkey only starts the main user's copy: start the launcher activity in the
        # game's user.
        resolved = self.adb(
            "shell", "cmd", "package", "resolve-activity", "--brief", "--user", str(user), package
        ).stdout.decode(errors="ignore")
        component = resolved.strip().splitlines()[-1].strip() if resolved.strip() else ""
        if not component.startswith(package + "/"):
            log("ERROR", f"can't find {package}'s main screen in Android user {user}")
            return
        self.adb("shell", "am", "start", "--user", str(user), "-n", component)

    def app_stop(self, package: str) -> None:
        self.adb("shell", "am", "force-stop", *self._user_args(package), package)

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

    def back(self, *, why: str = "") -> bool:
        """The phone's Back key (not the game's on-screen back arrow, a forbidden zone)."""
        log("TAP", f"BACK {why}".rstrip())
        if self.dry_run:
            return True
        with self.input_lock:
            self.shell.run("input keyevent KEYCODE_BACK")
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
