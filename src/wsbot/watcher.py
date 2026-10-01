"""Popup-watcher: the mortar between the bricks.

A daemon thread that owns the screenshot pipeline. Every frame it:
  1. publishes the frame for the main thread (single capture pipeline),
  2. checks whether the board is visible (a popup dims it, a level transition removes it),
  3. template-matches every registered popup button and taps the ones it finds.

Templates are crops from *inside* buttons only, so no level background ever leaks
into them. Registry: templates/popups.json, where list order is priority order.
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .board import covered_below, find_panel, read_board
from .debug import dbg, snap
from .device import Device
from .imgio import imread, imwrite
from .log import log

SCALE = 0.5  # match on a half-res frame: ~4x faster, still plenty of detail for buttons
REFINE_PAD = 8  # half-res px around the quarter-res spot where match() scores a popup
COARSE_MIN = 10  # templates smaller than this at quarter res are searched at half res
UNKNOWN_AFTER_S = 8.0  # board hidden and nothing matched this long -> unknown overlay
APP_CHECK_EVERY_S = 5.0
ACTION_SETTLE_S = 1.0  # after tapping a popup, give it this long to disappear
# A toast last seen this recently may still be up. The watcher looks only ~2x a second,
# so without the fast eye this must span a couple of its frames.
COVER_LINGER_S = 1.0
COVER_LINGER_EYE_S = 0.3
EYE_BOX = (0.1, 0.5, 0.9, 0.95)  # screen fractions the toast shows in (any board, any phone)
EYE_EVERY_S = 0.04


@dataclass
class Popup:
    name: str
    template: np.ndarray  # half-res BGR
    threshold: float
    cooldown: float
    level_done: bool  # seeing this means the level is over
    tap: bool
    # tap here instead of the match center; "@name" = a device anchor (see ios_device.py)
    tap_point: tuple[int, int] | str | None = None
    allow: str | None = None  # forbidden zone this entry may tap (see device.py)
    confirm: int = 1  # consecutive matching frames required before acting
    blocking: bool = True  # False: tapping it doesn't make the bot wait (a toast)
    holdoff: float = 0.0  # don't tap within this long of tapping any other popup
    avoid: bool = False  # never tap: while it's visible its area is a no-tap zone (ad buttons)
    avoid_pad: tuple[int, int] = (40, 40)  # zone = template box grown by this (x, y)
    # A toast over the board: the y range (relative to the match) where it swallows swipes
    covers: tuple[int, int] | None = None
    coarse: np.ndarray | None = None  # quarter-res template for match()'s first look
    last_hit: float = 0.0
    streak: int = 0


def load_popups(folder: Path) -> list[Popup]:
    registry = folder / "popups.json"
    entries = json.loads(registry.read_text(encoding="utf-8")) if registry.exists() else []
    popups = []
    for e in entries:
        img = imread(folder / "popups" / e["file"])
        if img is None:
            log("WARN", f"popup template missing: {e['file']}")
            continue
        small = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
        quarter = cv2.resize(small, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        popups.append(
            Popup(
                name=e["name"],
                template=small,
                threshold=e.get("threshold", 0.85),
                cooldown=e.get("cooldown", 1.5),
                level_done=e.get("level_done", False),
                tap=e.get("tap", True),
                tap_point=_tap_point(e.get("tap_point")),
                allow=e.get("allow"),
                confirm=e.get("confirm", 1),
                blocking=e.get("blocking", True),
                holdoff=e.get("holdoff", 0.0),
                avoid=e.get("avoid", False),
                avoid_pad=tuple(e.get("avoid_pad", (40, 40))),
                covers=tuple(e["covers"]) if "covers" in e else None,
                coarse=quarter if min(quarter.shape[:2]) >= COARSE_MIN else None,
            )
        )
    log("WATCHER", f"loaded {len(popups)} popup templates")
    return popups


def _tap_point(value) -> tuple[int, int] | str | None:
    if value is None or isinstance(value, str):
        return value
    return tuple(value)


def coarse_frame(small_frame: np.ndarray) -> np.ndarray:
    """The half-res frame halved again, for match()'s first look (once per frame)."""
    return cv2.resize(small_frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)


def match(
    small_frame: np.ndarray, popup: Popup, coarse: np.ndarray | None = None
) -> tuple[float, tuple[int, int]]:
    """Best score and full-res center of `popup` in a half-res frame.

    With `coarse` (coarse_frame of it): find the spot at quarter res, then score it at
    half res in a small window around it. Same scores, ~4x less work: the full
    half-res search took 2.4 s a tick for 18 templates on a 2-core laptop (i5-6200U),
    so popups, toasts and level ends were seen seconds late.
    """
    th, tw = popup.template.shape[:2]
    ox = oy = 0
    if coarse is not None and popup.coarse is not None:
        res = cv2.matchTemplate(coarse, popup.coarse, cv2.TM_CCOEFF_NORMED)
        _, _, _, (qx, qy) = cv2.minMaxLoc(res)
        h, w = small_frame.shape[:2]
        ox, oy = max(0, 2 * qx - REFINE_PAD), max(0, 2 * qy - REFINE_PAD)
        x1, y1 = min(w, 2 * qx + tw + REFINE_PAD), min(h, 2 * qy + th + REFINE_PAD)
        small_frame = small_frame[oy:y1, ox:x1]
    res = cv2.matchTemplate(small_frame, popup.template, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    loc = (loc[0] + ox, loc[1] + oy)
    cx, cy = (loc[0] + tw / 2) / SCALE, (loc[1] + th / 2) / SCALE
    return float(score), (round(cx), round(cy))


class PopupWatcher(threading.Thread):
    def __init__(self, device: Device, templates: Path, package: str, diagnostics: Path) -> None:
        super().__init__(daemon=True, name="popup-watcher")
        self.device = device
        self.package = package
        self.diagnostics = diagnostics
        self.popups = load_popups(templates)
        self.stop_event = threading.Event()
        self.idle = threading.Event()  # set while the bot sleeps until its next day
        self.level_done = threading.Event()
        self.last_action = 0.0  # monotonic time the watcher last tapped a blocking popup
        self._last_tap = 0.0  # monotonic time the watcher last tapped any popup
        self.hits: Counter[str] = Counter()
        self.last_match = 0.0  # monotonic time any popup template last matched
        # The "already collected" toast: frame times it first showed (this appearance)
        # and was last seen, and the screen rows it covers. See covering().
        self.cover_since = 0.0
        self.cover_seen = 0.0
        self.cover_span = (0, 0)
        self._eye = next((p for p in self.popups if p.covers), None)
        self._eye_on = self._eye is not None and hasattr(device, "peek")
        self.board_visible = False
        # Set by the main thread while it solves a level. Then "board visible" is just
        # "the white panel is still exactly there": cheap, and unlike a full grid read
        # it isn't fooled by letters flying off after a word is found.
        self.expected_panel: tuple[int, int, int, int] | None = None
        self.frame: np.ndarray | None = None
        # The newest frames with the board up (the level's last ones, at its end)
        self.board_frames: deque[np.ndarray] = deque(maxlen=8)
        self.frame_time = 0.0
        self.fps = 0.0
        self._frame_cond = threading.Condition()
        self._hidden_since: float | None = None
        self._last_unknown_dump = 0.0
        self._last_app_check = 0.0
        self._last_score_log = 0.0

    # ---- API for the main thread -------------------------------------------

    def latest(self, newer_than: float = 0.0, timeout: float = 3.0) -> np.ndarray | None:
        """Most recent frame captured after `newer_than` (a time.monotonic() value)."""
        deadline = time.monotonic() + timeout
        with self._frame_cond:
            while self.frame_time <= newer_than:
                left = deadline - time.monotonic()
                if left <= 0:
                    return None
                self._frame_cond.wait(left)
            return self.frame

    def stop(self) -> None:
        self.stop_event.set()

    # ---- thread body --------------------------------------------------------

    def run(self) -> None:
        log("WATCHER", "started")
        if self._eye_on:
            threading.Thread(target=self._watch_cover, daemon=True, name="toast-eye").start()
        failures = 0
        while not self.stop_event.is_set():
            if self.idle.is_set():
                self.stop_event.wait(1.0)
                continue
            t0 = time.monotonic()
            try:
                self._tick()
                failures = 0
            except Exception as exc:  # one bad frame must never kill the watcher
                failures += 1
                log("ERROR", f"watcher frame failed ({failures}x): {exc!r}")
                if failures % 5 == 0:
                    self.device.reconnect()
                self.stop_event.wait(min(0.5 * failures, 10.0))
            dt = time.monotonic() - t0
            self.fps = 0.8 * self.fps + 0.2 * (1 / dt if dt > 0 else 0)
        log("WATCHER", "stopped")

    def covering(self) -> tuple[int, int] | None:
        """The y range a toast covers right now, or None. Swipes there are swallowed."""
        linger = COVER_LINGER_EYE_S if self._eye_on else COVER_LINGER_S
        if self.cover_seen and time.monotonic() - self.cover_seen <= linger:
            return self.cover_span
        return None

    def busy(self) -> bool:
        """True right after the watcher tapped something: give that popup time to go."""
        return time.monotonic() - self.last_action < ACTION_SETTLE_S

    def _tick(self) -> None:
        t0 = time.monotonic()
        frame = self.device.frame()
        now = time.monotonic()
        self.board_visible = self._board_visible(frame)
        if self.board_visible:
            self.board_frames.append(frame)
        with self._frame_cond:
            self.frame, self.frame_time = frame, now
            self._frame_cond.notify_all()

        small = cv2.resize(frame, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
        snap("calib_frame", frame, every_s=15, note=f"board_visible={self.board_visible}")
        matched = self._handle_popups(small, coarse_frame(small), now)
        if matched or self.board_visible:
            self._hidden_since = None
        else:
            self._check_unknown(frame, now)

        took = time.monotonic() - t0
        if took > 2.0:
            log("WARN", f"slow watcher tick {took:.1f}s (screenshot {now - t0:.1f}s)")

        # The game gone: confirm it within ~2 s instead of ~10 (3 checks in a row).
        every = 1.0 if getattr(self.device, "game_missing", lambda: False)() else APP_CHECK_EVERY_S
        if now - self._last_app_check > every:
            self._last_app_check = now
            self._ensure_foreground()

    def _board_visible(self, frame: np.ndarray) -> bool:
        expected = self.expected_panel
        if expected is None:
            # A real letter grid, not just a white panel: popups have big white bodies too.
            return read_board(frame) is not None
        panel = find_panel(frame)
        if panel is None:
            return False
        if all(abs(a - b) <= 12 for a, b in zip(panel, expected, strict=True)):
            return True
        # The "already collected" toast cuts the panel short. The bot shouldn't cause it
        # any more; if it shows anyway, the toast template taps it away and the burst
        # carries on instead of stopping to wait it out.
        return (
            all(abs(a - b) <= 12 for a, b in zip(panel[:3], expected[:3], strict=True))
            and panel[3] < expected[3]
            and covered_below(frame, panel)
        )

    def _handle_popups(self, small: np.ndarray, coarse: np.ndarray, now: float) -> bool:
        """Act on the highest-priority popup that is on screen. True if any matched."""
        hit = None
        scores = []
        for popup in self.popups:
            try:
                score, center = match(small, popup, coarse)
            except Exception as exc:
                log("ERROR", f"match {popup.name} failed: {exc!r}")
                continue
            scores.append((score, popup.name, center))
            if popup.covers and score >= popup.threshold and not self._eye_on:
                self._note_cover(popup, center, now)
            if popup.avoid:
                self._guard(popup, center if score >= popup.threshold else None)
                continue
            if score >= popup.threshold:
                popup.streak += 1
                if hit is None:
                    hit = (popup, score, center)
            else:
                popup.streak = 0
        if now - self._last_score_log > 1.0:
            self._last_score_log = now
            top = sorted(scores, reverse=True)[:4]
            dbg(
                f"watcher: board_visible={self.board_visible} fps={self.fps:.1f} "
                f"expected={self.expected_panel} top={[(n, round(sc, 3), c) for sc, n, c in top]}"
            )
        if hit is None:
            return False
        popup, score, center = hit
        self.last_match = now
        # The top match owns this frame even while cooling down or unconfirmed, so a
        # lower-priority button (like a close X) never jumps ahead of it.
        if popup.streak < popup.confirm or now - popup.last_hit < popup.cooldown:
            return True
        # Closing the bonus popup while its claimed coins still fly leaves the game
        # ignoring every touch until a restart. It closes itself once they land.
        if now - self._last_tap < popup.holdoff:
            return True
        popup.last_hit = now
        self.hits[popup.name] += 1
        if popup.tap or not popup.covers:  # a toast left alone is logged by _note_cover
            log("WATCHER", f"{popup.name} score={score:.2f} at {center}")
        if popup.level_done:
            self.level_done.set()
        if popup.tap:
            if popup.blocking:
                self.last_action = now
            self._last_tap = now
            target = popup.tap_point or center
            if isinstance(target, str):
                anchor = getattr(self.device, "anchor", lambda _: None)(target.lstrip("@"))
                if anchor is None:
                    log("WARN", f"{popup.name}: anchor {target} not known yet; tapping the match")
                target = anchor or center
            self.device.tap(*target, why=popup.name, allow=popup.allow)
        return True

    def _watch_cover(self) -> None:
        """iPhone: frames are free, so look for the toast ~25x a second. At the watcher's
        ~2 fps the bot learned of it up to 0.5 s late and kept swiping under it."""
        w, h = self.device.calib
        box = (
            round(EYE_BOX[0] * w),
            round(EYE_BOX[1] * h),
            round(EYE_BOX[2] * w),
            round(EYE_BOX[3] * h),
        )
        popup, (x0, y0, _, _) = self._eye, box
        while not self.stop_event.is_set():
            if self.idle.is_set():
                self.stop_event.wait(1.0)
                continue
            try:
                small, t = self.device.peek(box, SCALE)
                score, (cx, cy) = match(small, popup)
                if score >= popup.threshold:
                    self._note_cover(popup, (x0 + cx, y0 + cy), t)
            except Exception as exc:  # no frame yet, phone reconnecting
                dbg(f"toast eye: {exc!r}")
                self.stop_event.wait(1.0)
            self.stop_event.wait(EYE_EVERY_S)

    def _note_cover(self, popup: Popup, center: tuple[int, int], now: float) -> None:
        if now - self.cover_seen > 0.5:  # gone that long: this is a new one
            self.cover_since = now
            log(
                "WATCHER",
                f"{popup.name} toast over y {center[1] + popup.covers[0]}-"
                f"{center[1] + popup.covers[1]}",
            )
        self.cover_seen = now
        self.cover_span = (center[1] + popup.covers[0], center[1] + popup.covers[1])

    def _guard(self, popup: Popup, center: tuple[int, int] | None) -> None:
        """Keep a no-tap zone over an avoid-template (an ad button) while it's visible."""
        zone = None
        if center is not None:
            th, tw = popup.template.shape[:2]
            hw = tw / SCALE / 2 + popup.avoid_pad[0]
            hh = th / SCALE / 2 + popup.avoid_pad[1]
            cx, cy = center
            zone = (round(cx - hw), round(cy - hh), round(cx + hw), round(cy + hh))
            if popup.streak == 0:
                log("WATCHER", f"{popup.name} on screen: no taps in {zone}")
            popup.streak += 1
        else:
            popup.streak = 0
        self.device.set_dynamic_zone(popup.name, zone)

    def _check_unknown(self, frame: np.ndarray, now: float) -> None:
        if self._hidden_since is None:
            self._hidden_since = now
            return
        hidden = now - self._hidden_since
        if hidden > UNKNOWN_AFTER_S and now - self._last_unknown_dump > 30:
            self._last_unknown_dump = now
            path = self.diagnostics / f"unknown_popup_{time.strftime('%Y%m%d_%H%M%S')}.png"
            imwrite(path, frame)
            log("WARN", f"board hidden {hidden:.0f}s with no known popup -> saved {path.name}")

    def _ensure_foreground(self) -> None:
        pkg = self.device.foreground()
        if pkg and pkg != self.package:
            log("RECOVERY", f"foreground is {pkg}, relaunching {self.package}")
            if not self.device.dry_run:
                self.device.app_start(self.package)
