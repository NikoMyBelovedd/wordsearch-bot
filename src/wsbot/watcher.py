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
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .board import find_panel, read_board
from .device import Device
from .log import log

SCALE = 0.5  # match on a half-res frame: ~4x faster, still plenty of detail for buttons
UNKNOWN_AFTER_S = 8.0  # board hidden and nothing matched this long -> unknown overlay
APP_CHECK_EVERY_S = 5.0
ACTION_SETTLE_S = 1.0  # after tapping a popup, give it this long to disappear


@dataclass
class Popup:
    name: str
    template: np.ndarray  # half-res BGR
    threshold: float
    cooldown: float
    level_done: bool  # seeing this means the level is over
    tap: bool
    tap_point: tuple[int, int] | None = None  # tap here instead of the match center
    allow: str | None = None  # forbidden zone this entry may tap (see device.py)
    confirm: int = 1  # consecutive matching frames required before acting
    last_hit: float = 0.0
    streak: int = 0


def load_popups(folder: Path) -> list[Popup]:
    registry = folder / "popups.json"
    entries = json.loads(registry.read_text()) if registry.exists() else []
    popups = []
    for e in entries:
        img = cv2.imread(str(folder / "popups" / e["file"]))
        if img is None:
            log("WARN", f"popup template missing: {e['file']}")
            continue
        small = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
        popups.append(
            Popup(
                name=e["name"],
                template=small,
                threshold=e.get("threshold", 0.85),
                cooldown=e.get("cooldown", 1.5),
                level_done=e.get("level_done", False),
                tap=e.get("tap", True),
                tap_point=tuple(e["tap_point"]) if "tap_point" in e else None,
                allow=e.get("allow"),
                confirm=e.get("confirm", 1),
            )
        )
    log("WATCHER", f"loaded {len(popups)} popup templates")
    return popups


def match(small_frame: np.ndarray, popup: Popup) -> tuple[float, tuple[int, int]]:
    """Best score and full-res center of `popup` in a half-res frame."""
    res = cv2.matchTemplate(small_frame, popup.template, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    th, tw = popup.template.shape[:2]
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
        self.last_action = 0.0  # monotonic time the watcher last tapped a popup
        self.hits: Counter[str] = Counter()
        self.last_match = 0.0  # monotonic time any popup template last matched
        self.board_visible = False
        # Set by the main thread while it solves a level. Then "board visible" is just
        # "the white panel is still exactly there": cheap, and unlike a full grid read
        # it isn't fooled by letters flying off after a word is found.
        self.expected_panel: tuple[int, int, int, int] | None = None
        self.frame: np.ndarray | None = None
        self.frame_time = 0.0
        self.fps = 0.0
        self._frame_cond = threading.Condition()
        self._hidden_since: float | None = None
        self._last_unknown_dump = 0.0
        self._last_app_check = 0.0

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

    def busy(self) -> bool:
        """True right after the watcher tapped something: give that popup time to go."""
        return time.monotonic() - self.last_action < ACTION_SETTLE_S

    def _tick(self) -> None:
        t0 = time.monotonic()
        frame = self.device.frame()
        now = time.monotonic()
        self.board_visible = self._board_visible(frame)
        with self._frame_cond:
            self.frame, self.frame_time = frame, now
            self._frame_cond.notify_all()

        small = cv2.resize(frame, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
        matched = self._handle_popups(small, now)
        if matched or self.board_visible:
            self._hidden_since = None
        else:
            self._check_unknown(frame, now)

        took = time.monotonic() - t0
        if took > 2.0:
            log("WARN", f"slow watcher tick {took:.1f}s (screenshot {now - t0:.1f}s)")

        if now - self._last_app_check > APP_CHECK_EVERY_S:
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
        return all(abs(a - b) <= 12 for a, b in zip(panel, expected, strict=True))

    def _handle_popups(self, small: np.ndarray, now: float) -> bool:
        """Act on the highest-priority popup that is on screen. True if any matched."""
        hit = None
        for popup in self.popups:
            try:
                score, center = match(small, popup)
            except Exception as exc:
                log("ERROR", f"match {popup.name} failed: {exc!r}")
                continue
            if score >= popup.threshold:
                popup.streak += 1
                if hit is None:
                    hit = (popup, score, center)
            else:
                popup.streak = 0
        if hit is None:
            return False
        popup, score, center = hit
        self.last_match = now
        # The top match owns this frame even while cooling down or unconfirmed, so a
        # lower-priority button (like a close X) never jumps ahead of it.
        if popup.streak < popup.confirm or now - popup.last_hit < popup.cooldown:
            return True
        popup.last_hit = now
        self.hits[popup.name] += 1
        log("WATCHER", f"{popup.name} score={score:.2f} at {center}")
        if popup.level_done:
            self.level_done.set()
        if popup.tap:
            self.last_action = now
            target = popup.tap_point or center
            self.device.tap(*target, why=popup.name, allow=popup.allow)
        return True

    def _check_unknown(self, frame: np.ndarray, now: float) -> None:
        if self._hidden_since is None:
            self._hidden_since = now
            return
        hidden = now - self._hidden_since
        if hidden > UNKNOWN_AFTER_S and now - self._last_unknown_dump > 30:
            self._last_unknown_dump = now
            path = self.diagnostics / f"unknown_popup_{time.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(str(path), frame)
            log("WARN", f"board hidden {hidden:.0f}s with no known popup -> saved {path.name}")

    def _ensure_foreground(self) -> None:
        pkg = self.device.foreground()
        if pkg and pkg != self.package:
            log("RECOVERY", f"foreground is {pkg}, relaunching {self.package}")
            if not self.device.dry_run:
                self.device.app_start(self.package)
