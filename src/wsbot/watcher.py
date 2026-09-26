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

from .board import read_board
from .device import Device
from .log import log

SCALE = 0.5  # match on a half-res frame: ~4x faster, still plenty of detail for buttons
UNKNOWN_AFTER_S = 8.0  # board hidden and nothing matched this long -> unknown overlay
APP_CHECK_EVERY_S = 5.0


@dataclass
class Popup:
    name: str
    template: np.ndarray  # half-res BGR
    threshold: float
    cooldown: float
    level_done: bool  # seeing this means the level is over
    tap: bool
    last_hit: float = 0.0


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
        self.level_done = threading.Event()
        self.popup_active = threading.Event()
        self.hits: Counter[str] = Counter()
        self.last_match = 0.0  # monotonic time any popup template last matched
        self.board_visible = False
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
        while not self.stop_event.is_set():
            t0 = time.monotonic()
            try:
                self._tick()
            except Exception as exc:  # one bad frame must never kill the watcher
                log("ERROR", f"watcher frame failed: {exc!r}")
                time.sleep(0.5)
            dt = time.monotonic() - t0
            self.fps = 0.8 * self.fps + 0.2 * (1 / dt if dt > 0 else 0)
        log("WATCHER", "stopped")

    def _tick(self) -> None:
        frame = self.device.frame()
        now = time.monotonic()
        # A real letter grid, not just a white panel: popups have big white bodies too.
        self.board_visible = read_board(frame) is not None
        with self._frame_cond:
            self.frame, self.frame_time = frame, now
            self._frame_cond.notify_all()

        small = cv2.resize(frame, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
        matched = False
        for popup in self.popups:
            try:
                score, center = match(small, popup)
            except Exception as exc:
                log("ERROR", f"match {popup.name} failed: {exc!r}")
                continue
            if score < popup.threshold:
                continue
            matched = True
            self.last_match = now
            if now - popup.last_hit < popup.cooldown:
                continue
            popup.last_hit = now
            self.hits[popup.name] += 1
            log("WATCHER", f"{popup.name} score={score:.2f} at {center}")
            if popup.level_done:
                self.level_done.set()
            if popup.tap:
                self.popup_active.set()
                self.device.tap(*center, why=popup.name)
            break  # one action per frame; the next frame shows what's under it

        if matched or self.board_visible:
            self._hidden_since = None
            if not matched:
                self.popup_active.clear()
        else:
            self._check_unknown(frame, now)

        if now - self._last_app_check > APP_CHECK_EVERY_S:
            self._last_app_check = now
            self._ensure_foreground()

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
                self.device.d.app_start(self.package)
