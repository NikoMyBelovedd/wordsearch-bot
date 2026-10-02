"""Popup-watcher: the mortar between the bricks.

A daemon thread that owns the screenshot pipeline. Every frame it:
  1. publishes the frame for the main thread (single capture pipeline),
  2. checks whether the board is visible (a popup dims it, a level transition removes it),
  3. template-matches every registered popup button and taps the ones it finds.

It is the bot's biggest CPU cost, and a farm runs one bot per phone on one computer,
so it works on a half-size frame, at most MAX_FPS times a second, does nothing new for
an unchanged picture, and while the level's board is fully in view (no popup can be
over it) looks for the whole popup list only once a second.

Templates are crops from *inside* buttons only, so no level background ever leaks
into them. Registry: templates/popups.json, where list order is priority order.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .board import covered_below
from .debug import dbg, snap, snap_due
from .device import Device
from .imgio import imread, imwrite
from .log import log
from .shot import SCALE, Shot

# SCALE: match on a half-res frame: ~4x faster, still plenty of detail for buttons
REFINE_PAD = 8  # half-res px around the quarter-res spot where match() scores a popup
COARSE_MIN = 10  # templates smaller than this at quarter res are searched at half res
# Gray finds a template's spot 3x faster than color, but lost the flat, low-contrast
# ones (close_x_grey, get_reward) on pasted-popup tests: those keep the color look.
GRAY_MIN_STD = 32.0
UNKNOWN_AFTER_S = 8.0  # board hidden and nothing matched this long -> unknown overlay
APP_CHECK_EVERY_S = 5.0
ACTION_SETTLE_S = 1.0  # after tapping a popup, give it this long to disappear
# A toast last seen this recently may still be up. The watcher looks only ~2x a second,
# so without the fast eye this must span a couple of its frames.
COVER_LINGER_S = 1.0
COVER_LINGER_EYE_S = 0.3
EYE_BOX = (0.1, 0.5, 0.9, 0.95)  # screen fractions the toast shows in (any board, any phone)
EYE_EVERY_S = 0.1


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name) or default)
    except ValueError:
        return default


# Looks a second at most. The phone streams up to 60; the bot needs a few (two
# matching board reads, a button that stays put). WSBOT_FPS lowers it on weak PCs.
MAX_FPS = max(1.0, _env_float("WSBOT_FPS", 5.0))
FULL_EVERY_S = 1.0  # board fully in view: the whole popup list this often
# Otherwise (a popup, a level change) this often: the whole list was ~45% of a bot's
# CPU at 5 looks a second, and a button tapped 0.2 s later costs nothing.
SCAN_EVERY_S = 0.4
# While the bot rests between levels (paced play) nothing is urgent: look once a
# second and scan the whole list every few seconds. Rests are ~half of paced play.
REST_PERIOD_S = 1.0
REST_SCAN_S = 3.0


def board_visible_in(shot: Shot, expected: tuple[int, int, int, int] | None) -> bool:
    """The level's board is on screen (see PopupWatcher._board_visible)."""
    panel = shot.panel  # half-size look first: no panel, no board (popups, transitions)
    if panel is None:
        return False
    if expected is None:
        # A real letter grid, not just a white panel: popups have big white bodies too.
        return shot.board is not None
    if all(abs(a - b) <= 12 for a, b in zip(panel, expected, strict=True)):
        return True
    # The "already collected" toast cuts the panel short. The bot shouldn't cause it
    # any more; if it shows anyway, the toast template taps it away and the burst
    # carries on instead of stopping to wait it out.
    return (
        all(abs(a - b) <= 12 for a, b in zip(panel[:3], expected[:3], strict=True))
        and panel[3] < expected[3]
        and covered_below(shot.mini, panel, shot.mini_scale)
    )


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
    # Shows during a level (the bonus jar): the board hidden under it isn't the level ending
    mid_level: bool = False
    coarse: np.ndarray | None = None  # quarter-res template for match()'s first look
    coarse_gray: bool = True  # `coarse` is grayscale (else BGR; see GRAY_MIN_STD)
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
        # Ad buttons mark no-tap zones: never risk losing one to the faster look.
        gray = not e.get("avoid", False) and (
            cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).std() >= GRAY_MIN_STD
        )
        if gray:
            quarter = cv2.cvtColor(quarter, cv2.COLOR_BGR2GRAY)
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
                mid_level=e.get("mid_level", False),
                coarse=quarter if min(quarter.shape[:2]) >= COARSE_MIN else None,
                coarse_gray=gray,
            )
        )
    log("WATCHER", f"loaded {len(popups)} popup templates")
    return popups


def _tap_point(value) -> tuple[int, int] | str | None:
    if value is None or isinstance(value, str):
        return value
    return tuple(value)


def coarse_frame(small_frame: np.ndarray, gray: bool = True) -> np.ndarray:
    """The half-res frame halved again (grayscale for templates with coarse_gray), for
    match()'s first look. The score itself is always taken in color."""
    quarter = cv2.resize(small_frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(quarter, cv2.COLOR_BGR2GRAY) if gray else quarter


def match(
    small_frame: np.ndarray, popup: Popup, coarse: np.ndarray | None = None
) -> tuple[float, tuple[int, int]]:
    """Best score and full-res center of `popup` in a half-res frame.

    With `coarse` (coarse_frame of it, gray if popup.coarse_gray): find the spot at
    quarter res, then score it in color at half res in a small window around it. Same
    scores, ~10x less work: the full half-res search took 2.4 s a tick for 18 templates
    on a 2-core laptop (i5-6200U), so popups, toasts and level ends were seen seconds
    late. (Gray only finds the spot: gray scores run ~0.05 higher and false-matched
    next_level.)
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
        self.resting = threading.Event()  # set while the bot rests (see REST_PERIOD_S)
        self.level_done = threading.Event()
        self.last_action = 0.0  # monotonic time the watcher last tapped a blocking popup
        self._last_tap = 0.0  # monotonic time the watcher last tapped any popup
        self.hits: Counter[str] = Counter()
        self.last_match = 0.0  # monotonic time any popup template last matched
        self.mid_level_seen = 0.0  # monotonic time a mid_level popup last matched
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
        self.shot: Shot | None = None
        # The newest frames with the board up (the level's last ones, at its end)
        self.board_frames: deque[Shot] = deque(maxlen=8)
        # Popup scores of the last new picture: name -> (score, center), and which
        # popups that look covered (see _score).
        self._scores: dict[str, tuple[float, tuple[int, int]]] = {}
        self._looked: list[Popup] = []
        self._last_full = 0.0
        self.frame_time = 0.0
        self.fps = 0.0
        self._frame_cond = threading.Condition()
        self._hidden_since: float | None = None
        self._last_unknown_dump = 0.0
        self._last_app_check = 0.0
        self._last_score_log = 0.0

    # ---- API for the main thread -------------------------------------------

    @property
    def frame(self) -> np.ndarray | None:
        """The newest frame in calibration space."""
        shot = self.shot
        return shot.calib if shot is not None else None

    def latest(self, newer_than: float = 0.0, timeout: float = 3.0) -> np.ndarray | None:
        """Most recent frame captured after `newer_than` (a time.monotonic() value)."""
        shot = self.latest_shot(newer_than, timeout)
        return shot.calib if shot is not None else None

    def latest_shot(self, newer_than: float = 0.0, timeout: float = 3.0) -> Shot | None:
        """latest() as a Shot: its board read is shared with the watcher's."""
        deadline = time.monotonic() + timeout
        with self._frame_cond:
            while self.frame_time <= newer_than:
                left = deadline - time.monotonic()
                if left <= 0:
                    return None
                self._frame_cond.wait(left)
            return self.shot

    def stop(self) -> None:
        self.stop_event.set()

    # ---- thread body --------------------------------------------------------

    def run(self) -> None:
        log("WATCHER", "started")
        if self._eye_on:
            threading.Thread(target=self._watch_cover, daemon=True, name="toast-eye").start()
        failures = 0
        quiet = False
        while not self.stop_event.is_set():
            if self.idle.is_set():
                if not quiet:  # nobody looks for a while: stop decoding the phone's video
                    quiet = True
                    self._set_quiet(True)
                self.stop_event.wait(1.0)
                continue
            if quiet:
                quiet = False
                self._set_quiet(False)
            period = REST_PERIOD_S if self.resting.is_set() else 1.0 / MAX_FPS
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
            if (rest := period - (time.monotonic() - t0)) > 0:
                self.stop_event.wait(rest)
            dt = time.monotonic() - t0
            self.fps = 0.8 * self.fps + 0.2 * (1 / dt if dt > 0 else 0)
        log("WATCHER", "stopped")

    def _set_quiet(self, quiet: bool) -> None:
        set_quiet = getattr(self.device, "set_quiet", None)
        if set_quiet is not None:
            try:
                set_quiet(quiet)
            except Exception as exc:
                log("WARN", f"couldn't {'pause' if quiet else 'resume'} the screen stream: {exc!r}")

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
        shot = self.device.shot()
        now = time.monotonic()
        fresh = not shot.same_picture(self.shot)
        if fresh:
            self.board_visible = self._board_visible(shot)
            if self.board_visible:
                # Only the phone's own frame (3 MB on an SE): the sizes made of it can
                # be 4x that, and these are kept for the level's end.
                self.board_frames.append(Shot(shot.seq, shot.native, shot.calib_size))
        else:
            shot = self.shot  # the same picture: keep the sizes already made of it
        with self._frame_cond:
            self.shot, self.frame_time = shot, now
            self._frame_cond.notify_all()

        if snap_due("calib_frame", 15):
            snap("calib_frame", shot.calib, every_s=15, note=f"board_visible={self.board_visible}")
        if fresh:
            self._score(shot, now)
        matched = self._handle_popups(now)
        if matched or self.board_visible:
            self._hidden_since = None
        else:
            self._check_unknown(shot, now)

        took = time.monotonic() - t0
        if took > 2.0:
            log("WARN", f"slow watcher tick {took:.1f}s (screenshot {now - t0:.1f}s)")

        # The game gone: confirm it within ~2 s instead of ~10 (3 checks in a row).
        every = 1.0 if getattr(self.device, "game_missing", lambda: False)() else APP_CHECK_EVERY_S
        if now - self._last_app_check > every:
            self._last_app_check = now
            self._ensure_foreground()

    def _board_visible(self, shot: Shot) -> bool:
        # Set by the main thread while it solves a level. Then "board visible" is just
        # "the white panel is still exactly there": cheap, and unlike a full grid read
        # it isn't fooled by letters flying off after a word is found.
        return board_visible_in(shot, self.expected_panel)

    def _score(self, shot: Shot, now: float) -> None:
        """Match the popup templates against a new picture. While the level's board is
        fully in view nothing can be over it but the toast (popups dim or hide the
        board, so it stops being "visible" the frame they show), so then the whole list
        is checked only every FULL_EVERY_S; the rest of the time every SCAN_EVERY_S.
        In between, only the toast (unless the eye watches it)."""
        in_view = self.board_visible and self.expected_panel is not None
        every = FULL_EVERY_S if in_view else SCAN_EVERY_S
        if self.resting.is_set():
            every = max(every, REST_SCAN_S)
        full = now - self._last_full >= every
        if full:
            self._last_full = now
        looked = []
        for popup in self.popups:
            if not full and not (popup.covers and not self._eye_on):
                continue
            try:
                coarse = shot.coarse if popup.coarse_gray else shot.coarse_color
                self._scores[popup.name] = match(shot.small, popup, coarse)
            except Exception as exc:
                log("ERROR", f"match {popup.name} failed: {exc!r}")
                continue
            looked.append(popup)
        self._looked = looked

    def _handle_popups(self, now: float) -> bool:
        """Act on the highest-priority popup that is on screen. True if any matched.
        Runs every tick; an unchanged picture keeps its scores from _score."""
        hit = None
        scores = []
        for popup in self._looked:
            score, center = self._scores[popup.name]
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
        if popup.mid_level:
            self.mid_level_seen = now
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
        last = None
        while not self.stop_event.is_set():
            if self.idle.is_set() or self.resting.is_set():  # no swipes, no toasts
                self.stop_event.wait(0.5)
                continue
            try:
                small, t = self.device.peek(box, SCALE)
                if t == last:  # no new frame: the screen hasn't changed
                    self.stop_event.wait(EYE_EVERY_S)
                    continue
                last = t
                score, (cx, cy) = match(small, popup, coarse_frame(small, popup.coarse_gray))
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

    def _check_unknown(self, shot: Shot, now: float) -> None:
        if self._hidden_since is None:
            self._hidden_since = now
            return
        hidden = now - self._hidden_since
        if hidden > UNKNOWN_AFTER_S and now - self._last_unknown_dump > 30:
            self._last_unknown_dump = now
            path = self.diagnostics / f"unknown_popup_{time.strftime('%Y%m%d_%H%M%S')}.png"
            imwrite(path, shot.calib)
            log("WARN", f"board hidden {hidden:.0f}s with no known popup -> saved {path.name}")

    def _ensure_foreground(self) -> None:
        pkg = self.device.foreground()
        if pkg and pkg != self.package:
            log("RECOVERY", f"foreground is {pkg}, relaunching {self.package}")
            if not self.device.dry_run:
                self.device.app_start(self.package)
