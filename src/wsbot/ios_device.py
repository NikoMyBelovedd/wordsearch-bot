"""iOS backend: an iPhone over USB with pymobiledevice3 (see iphone.py), iOS 27+.

The calibration space is the iPhone screen scaled by K, chosen so the game's UI comes
out the same pixel size as on the 1080x2400 Android calibration space. Popup templates
and the board reader's size thresholds then carry over; only fixed points and zones
are per-layout. Two layouts:

- "se": 16:9 phones (iPhone SE 2/3, 750x1334). The game fits the screen height; K = 1.728
  and the zones below were measured on SE frames.
- "tall": notch / Dynamic Island phones (iPhone 17 = iPhone18,3, 1206x2622, and the other
  ~19.5:9 iPhones). The game fits the width like on Android, so K starts at 1080 / width.
  The top bar sits under the safe area, which differs per model, so the zones are placed
  from where the back and cart buttons actually are (template match on live frames).
  Ad buttons on the level-end screen are
  `avoid` templates (templates/ios_tall) instead of fixed zones.

Serial: `ios` (the only / first iPhone) or `ios:UDID`.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np

from .debug import dbg, snap
from .device import BaseDevice, SafetyError
from .imgio import imread
from .log import log
from .shot import Shot

K = 1.728  # iPhone SE px -> calibration px
GAME = "in.playsimple.wordsearch"
ROOT = Path(__file__).resolve().parents[2]
TALL_ASPECT = 0.5  # width / height below this = a tall (notch) iPhone
ANCHOR_MIN_SCORE = 0.85
ANCHOR_RETRY_S = 2.0
WAIT_RETRY_S = 3.0  # retry this often while waiting for the phone's USB tunnel
_stop_waiting = threading.Event()  # set by cancel_connect()
# Tall layout, relative to the back button's center (bx, by) and the cart's (cx, cy), in
# calibration px. Measured on the SE, whose UI is the same size in calibration space.
BTN_HALF = 68  # half size of a top-bar zone (the buttons are ~105 px)
STAR_DX = 124  # star (bonus jar) center, right of the back button
ABOVE_BOARD_DY = 434  # neutral clear-tap over the hint card, below the top bar
BOARD_CENTER_DY = 1108  # first guess of the board center, below the top bar
TROPHY_DX, TROPHY_DY = 8, 394  # tournament trophy on the level-end screen
CALENDAR_DX, CALENDAR_DY = 35, 97  # daily-challenge button: board panel left, below its bottom


# ProductType -> marketing name. Apple's internal numbers run a generation ahead:
# the iPhone 17 reports "iPhone18,3".
IPHONE_NAMES = {
    "iPhone12,1": "iPhone 11",
    "iPhone12,3": "iPhone 11 Pro",
    "iPhone12,5": "iPhone 11 Pro Max",
    "iPhone12,8": "iPhone SE 2",
    "iPhone14,6": "iPhone SE 3",
    "iPhone13,1": "iPhone 12 mini",
    "iPhone13,2": "iPhone 12",
    "iPhone13,3": "iPhone 12 Pro",
    "iPhone13,4": "iPhone 12 Pro Max",
    "iPhone14,4": "iPhone 13 mini",
    "iPhone14,5": "iPhone 13",
    "iPhone14,2": "iPhone 13 Pro",
    "iPhone14,3": "iPhone 13 Pro Max",
    "iPhone14,7": "iPhone 14",
    "iPhone14,8": "iPhone 14 Plus",
    "iPhone15,2": "iPhone 14 Pro",
    "iPhone15,3": "iPhone 14 Pro Max",
    "iPhone15,4": "iPhone 15",
    "iPhone15,5": "iPhone 15 Plus",
    "iPhone16,1": "iPhone 15 Pro",
    "iPhone16,2": "iPhone 15 Pro Max",
    "iPhone17,3": "iPhone 16",
    "iPhone17,4": "iPhone 16 Plus",
    "iPhone17,1": "iPhone 16 Pro",
    "iPhone17,2": "iPhone 16 Pro Max",
    "iPhone17,5": "iPhone 16e",
    "iPhone18,3": "iPhone 17",
    "iPhone18,1": "iPhone 17 Pro",
    "iPhone18,2": "iPhone 17 Pro Max",
    "iPhone18,4": "iPhone Air",
}


def cancel_connect() -> None:
    """Abort a connect or reopen that is waiting for the phone (the UI's stop key)."""
    _stop_waiting.set()


def model_name(product_type: str) -> str:
    return IPHONE_NAMES.get(product_type, product_type or "iPhone")


def _box(x1: float, y1: float, x2: float, y2: float) -> tuple[int, int, int, int]:
    """iPhone SE px rectangle -> calibration space."""
    return round(x1 * K), round(y1 * K), round(x2 * K), round(y2 * K)


def _icon_mask(img: np.ndarray) -> np.ndarray:
    """White, unsaturated pixels (button icons) as a 0/255 mask."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return (((hsv[..., 2] > 200) & (hsv[..., 1] < 70)) * 255).astype(np.uint8)


class IOSGameDevice(BaseDevice):
    platform = "ios"
    layout = "se"
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
        self.phone.cancel = _stop_waiting  # the UI's stop key also ends a wait in reopen()
        self._open_phone()
        self.serial = f"ios:{self.phone.udid}"
        self.dry_run = dry_run
        self.input_lock = threading.Lock()
        self.width, self.height = self.phone.width, self.phone.height
        self.K = K
        self._anchors: dict[str, tuple[int, int]] = {}
        self._topbar: tuple[int, int, int, int] | None = None  # bx, by, cx, cy
        self._last_anchor_try = 0.0
        if self.width / self.height < TALL_ASPECT:
            self.layout = "tall"
            self.templates = "templates/ios_tall"
            self.K = 1080 / self.width
            self._anchor_tmpl = {
                n: imread(ROOT / self.templates / "anchors" / f"{n}.png") for n in ("back", "cart")
            }
        self.calib = (round(self.width * self.K), round(self.height * self.K))
        if self.layout == "tall":
            self._tall_zones(None)
        self.below_board_y = None
        self.taps = self.swipes = self.refused = 0
        self._seq = -1
        self._not_running = 0
        log(
            "DEVICE",
            f"connected {self.serial} {model_name(self.phone.product_type)} "
            f"({self.phone.product_type or '?'}) {self.width}x{self.height} "
            f"calib {self.calib[0]}x{self.calib[1]} layout={self.layout} K={self.K:.4f} "
            f"dry_run={dry_run}",
        )

    def _open_phone(self) -> None:
        """Open the USB session; while the phone has no tunnel (unplugged, or tunneld
        still setting it up after a replug) wait for it instead of crashing."""
        from .iphone import NoTunnel

        _stop_waiting.clear()
        waiting = False
        while True:
            try:
                self.phone.open()
                return
            except NoTunnel as exc:
                if not waiting:
                    log("DEVICE", f"waiting for the iPhone: {exc}")
                    waiting = True
            if _stop_waiting.wait(WAIT_RETRY_S):
                self.phone.close()
                raise RuntimeError("stopped while waiting for the iPhone")

    def _px(self, x: float, y: float) -> tuple[float, float]:
        return x / self.K, y / self.K

    # ---- tall layout: zones from the live top bar ---------------------------

    def _tall_zones(self, topbar: tuple[int, int, int, int] | None) -> None:
        """Zones and clear-tap points for a tall iPhone. Without a located top bar they
        are generous guesses (the Dynamic Island safe area is ~170-190 calib px)."""
        w, h = self.calib
        if topbar is None:
            bx, by, cx, cy = 105, 235, w - 100, 235
            pad_y = 130  # unsure where the bar is: cover more height
        else:
            bx, by, cx, cy = topbar
            pad_y = BTN_HALF
        self.zones = {
            "back_button": (bx - BTN_HALF, by - pad_y, bx + BTN_HALF, by + pad_y),
            "star_bonus_jar": (bx + BTN_HALF, by - pad_y, bx + STAR_DX + BTN_HALF, by + pad_y),
            "coins_and_shop": (cx - 310, cy - pad_y, w, cy + pad_y),
            "banner_ad": (0, round(h * 0.905), w, h),  # AdMob banner above the home bar
        }
        self.above_board = (w // 2, by + ABOVE_BOARD_DY)
        self.board_center = (w // 2, min(by + BOARD_CENTER_DY, h * 2 // 3))
        self._anchors = {"star": (bx + STAR_DX, by), "trophy": (bx + TROPHY_DX, by + TROPHY_DY)}

    def _locate_topbar(self, img: np.ndarray) -> None:
        """Find the back and cart buttons in a raw phone frame (top quarter). Only their
        positions: the flat round buttons score ~0.99 over a wide scale range, so they
        can't measure the UI scale (K stays width-fit, which is how the game lays out).
        Matched on the white icon (arrow, cart) only: the buttons are pink late in the
        game and green early on (level 9 on a fresh account), the icons are the same."""
        top = _icon_mask(
            cv2.resize(img[: img.shape[0] // 4], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        )
        best: dict[str, tuple[float, tuple[int, int]]] = {}
        for name, tmpl in self._anchor_tmpl.items():
            if tmpl is None:
                return
            for k in (self.K * 0.94, self.K * 0.97, self.K, self.K * 1.03):
                f = 0.5 / k
                t = _icon_mask(cv2.resize(tmpl, None, fx=f, fy=f, interpolation=cv2.INTER_AREA))
                res = cv2.matchTemplate(top, t, cv2.TM_CCOEFF_NORMED)
                _, score, _, loc = cv2.minMaxLoc(res)
                c = ((loc[0] + t.shape[1] / 2) * 2, (loc[1] + t.shape[0] / 2) * 2)
                if name not in best or score > best[name][0]:
                    best[name] = (float(score), (round(c[0]), round(c[1])))
        (sb, pb), (sc, pc) = best["back"], best["cart"]
        dbg(f"top bar try: back {sb:.3f} at {pb}, cart {sc:.3f} at {pc} (raw {img.shape})")
        snap("topbar_try_raw", img, every_s=30, note=f"back {sb:.2f} cart {sc:.2f}")
        w = img.shape[1]
        ok = (
            min(sb, sc) >= ANCHOR_MIN_SCORE
            and abs(pb[1] - pc[1]) <= 0.03 * w
            and pb[0] < w * 0.3 < w * 0.7 < pc[0]
        )
        if not ok:
            if max(sb, sc) >= 0.6:
                log("DIAG", f"top bar not sure: back {sb:.2f} at {pb}, cart {sc:.2f} at {pc}")
            return
        topbar = (
            round(pb[0] * self.K),
            round(pb[1] * self.K),
            round(pc[0] * self.K),
            round(pc[1] * self.K),
        )
        self._topbar = topbar
        self._tall_zones(topbar)
        snap("tall_topbar_found", img, every_s=0)  # for --report: the real phone screen
        log(
            "DEVICE",
            f"top bar found (back {sb:.2f}, cart {sc:.2f}): back {topbar[:2]} cart {topbar[2:]} "
            f"zones {self.zones}",
        )

    def anchor(self, name: str) -> tuple[int, int] | None:
        """A named tap point for popups.json "@name" tap points (None = unknown)."""
        if name == "calendar" and self.last_panel is not None:
            x, y, _, h = self.last_panel
            return x + CALENDAR_DX, y + h + CALENDAR_DY
        if self.layout == "se":
            return {"star": (225, 135), "trophy": (109, 530)}.get(name)
        return self._anchors.get(name) if self._topbar else None

    # ---- frames -------------------------------------------------------------

    def frame(self) -> np.ndarray:
        """Latest screen frame in calibration space."""
        return self.shot().calib

    def shot(self) -> Shot:
        """Latest screen frame at the phone's own size (resized on demand). The phone
        sends frames only when the screen changes, so wait briefly for a new one
        instead of spinning; an unchanged screen keeps its sequence number."""
        if not self.phone.alive:
            raise RuntimeError("iPhone USB session is down")
        self.phone.wait_newer(self._seq, 0.1)
        self._seq, _, img = self.phone.latest()
        if self.layout == "tall" and self._topbar is None:
            now = time.monotonic()
            if now - self._last_anchor_try > ANCHOR_RETRY_S:
                self._last_anchor_try = now
                try:
                    self._locate_topbar(img)
                except Exception as exc:
                    log("ERROR", f"top bar search failed: {exc!r}")
        return Shot(self._seq, img, self.calib)

    def peek(self, box: tuple[int, int, int, int], scale: float) -> tuple[np.ndarray, float]:
        """Part of the newest frame (calib box x0,y0,x1,y1) at `scale`, and when it
        arrived. A few ms, any thread: frames are already in memory here."""
        _, t, img = self.phone.latest()
        h, w = img.shape[:2]
        sx, sy = w / self.calib[0], h / self.calib[1]
        x0, y0, x1, y1 = box
        crop = img[round(y0 * sy) : round(y1 * sy), round(x0 * sx) : round(x1 * sx)]
        size = (round((x1 - x0) * scale), round((y1 - y0) * scale))
        return cv2.resize(crop, size, interpolation=cv2.INTER_AREA), t

    def set_quiet(self, quiet: bool) -> None:
        self.phone.set_quiet(quiet)

    def view_stale(self) -> bool:
        return self.phone.frozen

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

    def game_missing(self) -> bool:
        """The last check found the game not running (it crashes now and then): taps
        now land on the home screen and can open other apps."""
        return self._not_running > 0

    def app_start(self, package: str) -> None:
        log("APP", f"launch {package} -> pid {self._app('launch', package)}")

    def app_stop(self, package: str) -> None:
        self._app("kill", package)

    # ---- input --------------------------------------------------------------

    def tap(self, x: int, y: int, *, why: str = "", allow: str | None = None) -> bool:
        dbg(
            f"tap request calib=({x},{y}) phone={tuple(round(v) for v in self._px(x, y))} "
            f"why={why!r} allow={allow} below_board_y={self.below_board_y}"
        )
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
