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
from .shot import LOOKS, SCALE, Shot, look

# SCALE: match on a half-res frame: ~4x faster, still plenty of detail for buttons
REFINE_PAD = 8  # half-res px around the quarter-res spot where match() scores a popup
COARSE_MIN = 10  # templates smaller than this at quarter res are searched at half res
# Gray finds a template's spot 3x faster than color, but lost the flat, low-contrast
# ones (close_x_grey, get_reward) on pasted-popup tests: those keep the color look.
GRAY_MIN_STD = 32.0
# A full scan rescores only where the quarter-res picture changed since the last one
# (ChangeScan); video noise moves a still screen's pixels by a few levels, a change
# that matters (a popup, a dimmed board, a moving button) by far more.
CHANGE_LEVEL = 12
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


# AutomationHQ's low-power mode (AHQ_LOW_POWER=1, weak or busy computers; the file
# local/low_power in the repo does the same for tests): look and scan less often.
# Popups and toasts get noticed a little later; the bot plays the same way.
LOW_POWER = (
    os.environ.get("AHQ_LOW_POWER") == "1"
    or (Path(__file__).resolve().parents[2] / "local" / "low_power").is_file()
)

# Looks a second at most. The phone streams up to 60; the bot needs a few (two
# matching board reads, a button that stays put). WSBOT_FPS lowers it on weak PCs.
MAX_FPS = max(1.0, _env_float("WSBOT_FPS", 3.0 if LOW_POWER else 5.0))
FULL_EVERY_S = 3.0 if LOW_POWER else 2.0  # board fully in view: the whole popup list
# Otherwise (a popup, a level change) this often: the whole list was ~45% of a bot's
# CPU at 5 looks a second, and a button tapped ~0.4 s later costs ~1% of a level.
SCAN_EVERY_S = 1.2 if LOW_POWER else 0.8
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
    coarse_look: str = "gray"  # how `coarse` sees the frame (see shot.look)
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
        # "coarse" in popups.json: a cheaper look checked on pasted-popup tests (one
        # channel for a template gray loses). Otherwise gray when it has the contrast;
        # ad buttons mark no-tap zones, so they never risk a cheaper look untested.
        how = e.get("coarse")
        if how not in LOOKS:
            gray_ok = not e.get("avoid", False) and (
                cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).std() >= GRAY_MIN_STD
            )
            how = "gray" if gray_ok else "color"
        quarter = look(quarter, how)
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
                coarse_look=how,
            )
        )
    log("WATCHER", f"loaded {len(popups)} popup templates")
    return popups


def _tap_point(value) -> tuple[int, int] | str | None:
    if value is None or isinstance(value, str):
        return value
    return tuple(value)


def coarse_frame(small_frame: np.ndarray, how: str = "gray") -> np.ndarray:
    """The half-res frame halved again, as the template's `coarse_look` sees it, for
    match()'s first look. The score itself is always taken in color."""
    quarter = cv2.resize(small_frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    return look(quarter, how)


class ChangeScan:
    """The full scans' first looks, redone only where the picture changed.

    A full scan searches every template over the whole quarter-res frame: ~90% of its
    work. Between two full scans most of the screen usually stays put (a level being
    played changes a few cells; a popup sits still), and a template's score at a spot
    depends only on the pixels under it there. So each template keeps its whole score
    map from the last full scan, and only the spots whose template-sized window covers
    a changed pixel are scored again. The color score that decides a match (match())
    is taken at the best spot of that map, and taken again unless the pixels it looks
    at are exactly those it was last taken on (a still screen's video frames repeat
    unchanged areas bit for bit), so a kept score is the score of this picture.

    "Changed" tolerates video noise: a pixel counts once any channel moved more than
    CHANGE_LEVEL from `ref`, the picture the maps were scored against. `ref` follows a
    pixel only when it counts as changed, so a slow fade still adds up to a change
    instead of creeping by under the bar, and every kept score was taken on pixels
    within CHANGE_LEVEL of `ref`. A mostly changed frame is scored whole, as before.
    """

    TILE = 8  # quarter-res px: changes are tracked per tile, spots rescored per tile
    REDO_ALL = 0.5  # more than this share of a map to redo: score the whole frame

    def __init__(self) -> None:
        self.ref: np.ndarray | None = None
        self.scan = 0  # full scans so far
        self.changed: np.ndarray | None = None  # changed tiles this scan; None = all
        self.maps: dict[str, tuple[int, np.ndarray]] = {}  # name -> (scan, score map)
        # name -> (box, its half-res pixels, match() result) of its last color score:
        # the result depends on nothing else (only these pixels are kept, not frames)
        self.scores: dict[str, tuple[tuple[int, int, int, int], np.ndarray, tuple]] = {}

    def begin(self, coarse_color: np.ndarray) -> None:
        """Start a full scan of this picture (`Shot.coarse_color`)."""
        self.scan += 1
        ref = self.ref
        if ref is None or ref.shape != coarse_color.shape:
            self.ref, self.changed = coarse_color.copy(), None
            return
        # (cv2 throughout: numpy's max over the channel axis alone took ~5 ms)
        b, g, r = cv2.split(cv2.absdiff(coarse_color, ref))
        _, moved = cv2.threshold(cv2.max(cv2.max(b, g), r), CHANGE_LEVEL, 255, cv2.THRESH_BINARY)
        cv2.copyTo(coarse_color, moved, ref)
        t = self.TILE
        h, w = moved.shape
        gh, gw = -(-h // t), -(-w // t)
        moved = cv2.copyMakeBorder(moved, 0, gh * t - h, 0, gw * t - w, cv2.BORDER_CONSTANT)
        # a tile's mean is above 0 if any of its pixels moved
        tiles = cv2.resize(moved, (gw, gh), interpolation=cv2.INTER_AREA)
        self.changed = (tiles > 0).astype(np.uint8)

    def forget(self, name: str) -> None:
        """Drop a template's map (its scan failed): it is scored whole next time."""
        self.maps.pop(name, None)
        self.scores.pop(name, None)

    def kept(self, name: str, box: tuple[int, int, int, int], window: np.ndarray):
        """The match() result last taken at `box`, if `window` (the half-res pixels
        there) is exactly what it was taken on."""
        last = self.scores.get(name)
        if last is None or last[0] != box or not np.array_equal(last[1], window):
            return None
        return last[2]

    def keep(self, name: str, box: tuple[int, int, int, int], window: np.ndarray, result) -> None:
        self.scores[name] = (box, window.copy(), result)

    def spot(self, popup: Popup, coarse: np.ndarray) -> tuple[int, int]:
        """The template's best quarter-res spot in this scan's picture: cv2.minMaxLoc
        over its score map, where spots that saw no change keep their last score."""
        tmpl = popup.coarse
        th, tw = tmpl.shape[:2]
        kept = self.maps.get(popup.name)
        rects = None if kept is None or kept[0] != self.scan - 1 else self._redo(th, tw)
        if rects is None:
            res = cv2.matchTemplate(coarse, tmpl, cv2.TM_CCOEFF_NORMED)
        else:
            res = kept[1]
            for x0, y0, x1, y1 in rects:
                res[y0:y1, x0:x1] = cv2.matchTemplate(
                    coarse[y0 : y1 + th - 1, x0 : x1 + tw - 1], tmpl, cv2.TM_CCOEFF_NORMED
                )
        self.maps[popup.name] = (self.scan, res)
        return cv2.minMaxLoc(res)[3]

    def _redo(self, th: int, tw: int) -> list[tuple[int, int, int, int]] | None:
        """Spots (x0, y0, x1, y1 boxes of the score map) whose window covers a changed
        tile, or None to score the whole frame."""
        changed = self.changed
        if changed is None:
            return None
        if not changed.any():
            return []
        t = self.TILE
        # spots in tile (i, j) see pixel tiles i .. i + (t + th - 2) // t, same for x
        kh, kw = (t + th - 2) // t + 1, (t + tw - 2) // t + 1
        spots = cv2.dilate(changed, np.ones((kh, kw), np.uint8), anchor=(0, 0))
        h, w = self.ref.shape[:2]
        rh, rw = h - th + 1, w - tw + 1  # the score map's size
        n, _, stats, _ = cv2.connectedComponentsWithStats(spots, connectivity=8)
        rects, area = [], 0
        for x, y, bw, bh, _ in stats[1:n]:
            x0, y0 = x * t, y * t
            x1, y1 = min(rw, (x + bw) * t), min(rh, (y + bh) * t)
            if x0 < x1 and y0 < y1:
                rects.append((x0, y0, x1, y1))
                area += (x1 - x0) * (y1 - y0)
        return None if area > self.REDO_ALL * rh * rw else rects


def match(
    small_frame: np.ndarray,
    popup: Popup,
    coarse: np.ndarray | None = None,
    scan: ChangeScan | None = None,
) -> tuple[float, tuple[int, int]]:
    """Best score and full-res center of `popup` in a half-res frame.

    With `coarse` (coarse_frame of it, as popup.coarse_look): find the spot at
    quarter res, then score it in color at half res in a small window around it. Same
    scores, ~10x less work: the full half-res search took 2.4 s a tick for 18 templates
    on a 2-core laptop (i5-6200U), so popups, toasts and level ends were seen seconds
    late. (Gray only finds the spot: gray scores run ~0.05 higher and false-matched
    next_level.)

    `scan`: a full scan's ChangeScan (begun on this frame): it finds the spot, and
    keeps the score of a window whose pixels haven't changed since the last one.
    """
    th, tw = popup.template.shape[:2]
    ox = oy = 0
    box = None
    if coarse is not None and popup.coarse is not None:
        if scan is None:
            res = cv2.matchTemplate(coarse, popup.coarse, cv2.TM_CCOEFF_NORMED)
            qx, qy = cv2.minMaxLoc(res)[3]
        else:
            qx, qy = scan.spot(popup, coarse)
        h, w = small_frame.shape[:2]
        ox, oy = max(0, 2 * qx - REFINE_PAD), max(0, 2 * qy - REFINE_PAD)
        x1, y1 = min(w, 2 * qx + tw + REFINE_PAD), min(h, 2 * qy + th + REFINE_PAD)
        box = (ox, oy, x1, y1)
        small_frame = small_frame[oy:y1, ox:x1]
        if scan is not None and (kept := scan.kept(popup.name, box, small_frame)) is not None:
            return kept
    res = cv2.matchTemplate(small_frame, popup.template, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    loc = (loc[0] + ox, loc[1] + oy)
    cx, cy = (loc[0] + tw / 2) / SCALE, (loc[1] + th / 2) / SCALE
    result = float(score), (round(cx), round(cy))
    if scan is not None and box is not None:
        scan.keep(popup.name, box, small_frame, result)
    return result


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
        self._shot_t = 0.0  # when self.shot's picture arrived (the toast eye's clock)
        # The newest frames with the board up (the level's last ones, at its end)
        self.board_frames: deque[Shot] = deque(maxlen=8)
        # Popup scores of the last new picture: name -> (score, center), and which
        # popups that look covered (see _score).
        self._scores: dict[str, tuple[float, tuple[int, int]]] = {}
        self._looked: list[Popup] = []
        self._last_full = 0.0
        self._full_shot: Shot | None = None  # the picture the last full look was at
        self._changes = ChangeScan()
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
            self._shot_t = now
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
        # A new picture, or a full look that's due on a picture last looked at only
        # in part: a popup that went still after its last new frame was never seen
        # (a static Bonus Words popup sat 40 s until "game hung" restarted the game).
        if fresh or (self._full_shot is not shot and self._full_due(now)):
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

    def _full_due(self, now: float) -> bool:
        in_view = self.board_visible and self.expected_panel is not None
        every = FULL_EVERY_S if in_view else SCAN_EVERY_S
        if self.resting.is_set():
            every = max(every, REST_SCAN_S)
        return now - self._last_full >= every

    def _score(self, shot: Shot, now: float) -> None:
        """Match the popup templates against a new picture. While the level's board is
        fully in view nothing can be over it but the toast (popups dim or hide the
        board, so it stops being "visible" the frame they show), so then the whole list
        is checked only every FULL_EVERY_S; the rest of the time every SCAN_EVERY_S.
        In between, only the toast (unless the eye watches it)."""
        full = self._full_due(now)
        if full:
            self._last_full, self._full_shot = now, shot
            try:
                self._changes.begin(shot.coarse_color)
            except Exception as exc:  # start over: score every template whole
                log("ERROR", f"change scan failed: {exc!r}")
                self._changes = ChangeScan()
        looked = []
        for popup in self.popups:
            if not full and not (popup.covers and not self._eye_on):
                continue
            try:
                coarse = shot.coarse_as(popup.coarse_look)
                scan = self._changes if full else None
                self._scores[popup.name] = match(shot.small, popup, coarse, scan)
            except Exception as exc:
                log("ERROR", f"match {popup.name} failed: {exc!r}")
                self._changes.forget(popup.name)
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
        """iPhone: look for the toast in every new frame the watcher takes (5 a second),
        right away and only in its box. At the old watcher's ~2 fps the bot learned of
        it up to 0.5 s late and kept swiping under it. It used to grab its own frames
        10x a second: each one a full-frame conversion, ~20% of a bot's CPU."""
        w, h = self.device.calib
        box = (
            round(EYE_BOX[0] * w),
            round(EYE_BOX[1] * h),
            round(EYE_BOX[2] * w),
            round(EYE_BOX[3] * h),
        )
        popup, (x0, y0, x1, y1) = self._eye, box
        last = None
        while not self.stop_event.is_set():
            if self.idle.is_set() or self.resting.is_set():  # no swipes, no toasts
                self.stop_event.wait(0.5)
                continue
            try:
                shot, t = self.shot, self._shot_t
                if shot is None or shot is last:  # no new picture since the last look
                    self.stop_event.wait(EYE_EVERY_S)
                    continue
                last = shot
                img = shot.native
                sx, sy = img.shape[1] / w, img.shape[0] / h
                crop = img[round(y0 * sy) : round(y1 * sy), round(x0 * sx) : round(x1 * sx)]
                size = (round((x1 - x0) * SCALE), round((y1 - y0) * SCALE))
                small = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
                score, (cx, cy) = match(small, popup, coarse_frame(small, popup.coarse_look))
                if score >= popup.threshold:
                    self._note_cover(popup, (x0 + cx, y0 + cy), t)
            except Exception as exc:
                dbg(f"toast eye: {exc!r}")
                self.stop_event.wait(1.0)
            self.stop_event.wait(EYE_EVERY_S / 2)

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
