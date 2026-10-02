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

import contextlib
import json
import os
import threading
import time
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import sysalert
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

# Scaled copies of a template ("scales" in popups.json) are scored only in a window this
# many half-res px around where the template itself was found: the game's buttons pulse
# (Next Level 1.0x-1.10x), so one size alone missed them for seconds at a time.
VARIANT_PAD = 24
# Never sure we're still in the game: no board, no known game screen this long. Then the
# bot stops tapping blind (on the iPhone home screen a blind tap opened Apple's Watch app).
BLIND_TAP_WINDOW_S = 20.0
# A known screen we only wait on (level complete, loading) that stays this long counts
# as unknown: the game is stuck on it.
WAIT_STUCK_S = 90.0
# Unknown screen (not the board, nothing known) this long: relaunch the game; still
# unknown this long after that: restart it and reopen the phone connection; still
# unknown this long after that: exit with an error so AutomationHQ restarts the bot.
UNKNOWN_RELAUNCH_S = 45.0
UNKNOWN_RESTART_S = 60.0
UNKNOWN_GIVE_UP_S = 120.0
RELAUNCH_COOLDOWN_S = 30.0  # home screen seen again right after a relaunch: give it time
UNKNOWN_DUMP_MIN_S = 10.0  # unknown screens are saved once each, at most this often
UNKNOWN_SAME = 6.0  # thumbnails this close (mean abs difference) are the same screen
# The iPhone's own alerts (see sysalert.py): looked for this often while the board isn't
# in view (~10 ms a look); acted on once one has sat still this long (it fades and
# zooms in), at most every ALERT_TAP_GAP_S, and never within ALERT_HOLDOFF_S of any
# other tap; one that is still there after ALERT_MAX_TAPS taps is left to the
# unknown-screen recovery. Its labels are read again before every tap.
ALERT_EVERY_S = 0.5
ALERT_SETTLE_S = 0.6
ALERT_TAP_GAP_S = 2.5
ALERT_HOLDOFF_S = 1.5
ALERT_MAX_TAPS = 3
ALERT_SAME = 8.0  # mean abs difference of two 48x24 looks at the box: the same alert
TRUST_LOG_EVERY_S = 600.0
# Notification banners over the top of the screen: a no-tap zone while one is up (a tap
# opens the app that sent it), plus this long after it was last seen (it slides away).
BANNER_EVERY_S = 0.3
BANNER_LINGER_S = 0.8
BANNER_PAD = 16  # calibration px around the banner's box


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
    # Scaled copies (half-res), scored near where `template` was found (see VARIANT_PAD)
    variants: list[np.ndarray] = field(default_factory=list)
    group: str | None = None  # entries of one group share a cooldown (one button, many looks)
    # tap False: a known screen to wait on. tap_after: ...unless it stays this long, then tap
    tap_after: float = 0.0
    relaunch: bool = False  # not the game (the iPhone home screen): relaunch it, never tap
    system: bool = False  # the phone's own UI, not the game: no proof we're in the game
    # A whole screen of its own (level complete, home screen): never there while the
    # level's board is in view, so full scans then skip it (~2.5 ms each)
    off_board: bool = False
    # avoid zone as (left, top, right, bottom) px from the match center, instead of avoid_pad
    avoid_box: tuple[int, int, int, int] | None = None
    seen_since: float = 0.0  # when this entry started matching (0 = not matching)
    coarse: np.ndarray | None = None  # quarter-res template for match()'s first look
    coarse_look: str = "gray"  # how `coarse` sees the frame (see shot.look)
    last_hit: float = 0.0
    streak: int = 0


@dataclass
class SeenAlert:
    """An iPhone system alert on screen (see PopupWatcher._check_alert)."""

    alert: sysalert.Alert  # boxes in the phone's own px (Shot.native)
    zone: tuple[int, int, int, int]  # its box in calibration px
    look: np.ndarray  # 48x24 gray of the box: still the same alert?
    since: float  # when it showed (or was last tapped): it settles from here
    choice: sysalert.Choice | None = None  # None = labels not read (yet / since the tap)
    taps: int = 0
    last_tap: float = 0.0
    gave_up: bool = False


def _alert_look(img: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    crop = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    return cv2.resize(crop, (48, 24), interpolation=cv2.INTER_AREA).astype(np.int16)


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
        variants = [
            cv2.resize(small, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
            for s in e.get("scales", [])
            if s != 1.0
        ]
        pad = e.get("avoid_pad", (40, 40))
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
                avoid_pad=tuple(pad[:2]),
                covers=tuple(e["covers"]) if "covers" in e else None,
                mid_level=e.get("mid_level", False),
                coarse=quarter if min(quarter.shape[:2]) >= COARSE_MIN else None,
                coarse_look=how,
                variants=variants,
                group=e.get("group"),
                tap_after=e.get("tap_after", 0.0),
                relaunch=e.get("relaunch", False),
                system=e.get("system", False),
                off_board=e.get("off_board", False),
                avoid_box=tuple(e["avoid_box"]) if "avoid_box" in e else None,
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


def match_variants(
    small_frame: np.ndarray, popup: Popup, found: tuple[float, tuple[int, int]]
) -> tuple[float, tuple[int, int]]:
    """`found` (match() of the template itself), or a scaled copy's score and center if
    one scores higher in a window around that spot. A pulsing button is still found
    by the template at its spot; only its score drops (0.47 at 1.10x)."""
    best = found
    if not popup.variants:
        return best
    h, w = small_frame.shape[:2]
    cx, cy = found[1][0] * SCALE, found[1][1] * SCALE
    for tmpl in popup.variants:
        th, tw = tmpl.shape[:2]
        x0, y0 = max(0, round(cx - tw / 2 - VARIANT_PAD)), max(0, round(cy - th / 2 - VARIANT_PAD))
        x1, y1 = min(w, round(cx + tw / 2 + VARIANT_PAD)), min(h, round(cy + th / 2 + VARIANT_PAD))
        if x1 - x0 < tw or y1 - y0 < th:
            continue
        res = cv2.matchTemplate(small_frame[y0:y1, x0:x1], tmpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, (lx, ly) = cv2.minMaxLoc(res)
        if score > best[0]:
            center = (round((x0 + lx + tw / 2) / SCALE), round((y0 + ly + th / 2) / SCALE))
            best = (float(score), center)
    return best


def top_match(
    popups: list[Popup], scores: dict[str, tuple[float, tuple[int, int]]]
) -> tuple[Popup, float, tuple[int, int]] | None:
    """The entry that owns a picture: the first in list (priority) order that matches.
    Ad buttons (avoid) never do; they only mark no-tap zones."""
    for popup in popups:
        score, center = scores[popup.name]
        if not popup.avoid and score >= popup.threshold:
            return popup, score, center
    return None


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
        # Stuck on screens it doesn't know (see _escalate) and the end of the line
        self.on_fatal: Callable[[str], None] | None = None  # the bot: stop, exit non-zero
        self.fatal: str | None = None
        self.last_known = time.monotonic()  # the board or a known game screen last seen
        self.system_seen = 0.0  # the phone's own UI (home screen, an alert) last seen
        self._group_hit: dict[str, float] = {}  # group -> its last action (shared cooldown)
        self._dumped: deque[np.ndarray] = deque(maxlen=40)  # thumbnails of saved unknowns
        self._esc_step = 0
        self._esc_t = 0.0
        self._esc_relaunches: deque[float] = deque(maxlen=8)
        self._last_relaunch = 0.0
        # The iPhone's own alerts and notification banners (sysalert.py)
        self._sys_ui = getattr(device, "platform", "ios") == "ios"
        self.alert: SeenAlert | None = None
        self._alert_check = 0.0
        self._alert_shot: Shot | None = None
        self._trust_logged = -TRUST_LOG_EVERY_S
        self._no_ocr_logged = False
        self._alert_dump = -60.0
        self._banner_check = 0.0
        self._banner_shot: Shot | None = None
        self._banner_found = False
        self._banner_seen = 0.0
        self._banner_zone = False

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
                # Nobody looks: nothing is "unknown for long" (a 30 min break read as
                # "board hidden 1806s" and counted toward giving up)
                self._hidden_since, self.last_known, self._esc_step = None, time.monotonic(), 0
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

    def blind_taps_ok(self) -> bool:
        """May the bot tap blind to clear an unknown popup? Only while we know we're in
        the game: on the iPhone home screen a blind tap opened Apple's Watch app."""
        now = time.monotonic()
        return now - self.last_known < BLIND_TAP_WINDOW_S and now - self.system_seen > 3.0

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
        # Before the game's buttons: an alert's no-tap zone must be up before any tap
        alert = self._sys_ui and self._check_system_ui(shot, now)
        # Under an iPhone alert the game's own buttons still match (dimmed), but taps
        # can't reach them and they're no proof of anything: only the phone's own count.
        matched = self._handle_popups(now, system_only=self.alert is not None)
        if self.board_visible:
            self.last_known = now
        if matched or self.board_visible or alert:
            self._hidden_since = None
        else:
            self._check_unknown(shot, now)
        self._escalate(now)

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
            if popup.off_board and self.board_visible and self.expected_panel is not None:
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
        self._score_variants(shot, looked)

    def _score_variants(self, shot: Shot, looked: list[Popup]) -> None:
        for popup in looked:
            if popup.variants and popup.name in self._scores:
                self._scores[popup.name] = match_variants(
                    shot.small, popup, self._scores[popup.name]
                )

    def _handle_popups(self, now: float, *, system_only: bool = False) -> bool:
        """Act on the highest-priority popup that is on screen. True if any matched.
        Runs every tick; an unchanged picture keeps its scores from _score.
        system_only: only the phone's own screens (an iPhone alert is up)."""
        hit = None
        scores = []
        for popup in self._looked:
            if system_only and not (popup.system or popup.avoid):
                popup.streak, popup.seen_since = 0, 0.0
                continue
            score, center = self._scores[popup.name]
            scores.append((score, popup.name, center))
            if popup.covers and score >= popup.threshold and not self._eye_on:
                self._note_cover(popup, center, now)
            if popup.avoid:
                self._guard(popup, center if score >= popup.threshold else None)
                continue
            if score >= popup.threshold:
                popup.streak += 1
                popup.seen_since = popup.seen_since or now
                if hit is None:
                    hit = (popup, score, center)
            else:
                popup.streak = 0
                popup.seen_since = 0.0
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
        if popup.system:
            self.system_seen = now  # not the game: no blind taps, and it's not "known"
        elif popup.tap or popup.tap_after or now - popup.seen_since < WAIT_STUCK_S:
            self.last_known = now
        # The top match owns this frame even while cooling down or unconfirmed, so a
        # lower-priority button (like a close X) never jumps ahead of it.
        group = popup.group or popup.name
        if popup.streak < popup.confirm or now - self._group_hit.get(group, 0.0) < popup.cooldown:
            return True
        # Closing the bonus popup while its claimed coins still fly leaves the game
        # ignoring every touch until a restart. It closes itself once they land.
        if now - self._last_tap < popup.holdoff:
            return True
        popup.last_hit = self._group_hit[group] = now
        self.hits[popup.name] += 1
        if popup.relaunch:
            self._relaunch_game(f"{popup.name} on screen (score {score:.2f}): not in the game")
            return True
        tap = popup.tap or (popup.tap_after > 0 and now - popup.seen_since >= popup.tap_after)
        waiting = not popup.tap and not popup.covers
        # A screen to wait on is logged once per appearance, not every cooldown
        if tap or (waiting and popup.seen_since > self._group_hit.get(f"{group}:logged", -1.0)):
            self._group_hit[f"{group}:logged"] = now
            note = " (waiting)" if waiting and not tap else ""
            log("WATCHER", f"{popup.name} score={score:.2f} at {center}{note}")
        if popup.level_done:
            self.level_done.set()
        if tap:
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
            cx, cy = center
            if popup.avoid_box is not None:
                left, top, right, bottom = popup.avoid_box
                zone = (cx + left, cy + top, cx + right, cy + bottom)
            else:
                th, tw = popup.template.shape[:2]
                hw = tw / SCALE / 2 + popup.avoid_pad[0]
                hh = th / SCALE / 2 + popup.avoid_pad[1]
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
        if hidden <= UNKNOWN_AFTER_S or now - self._last_unknown_dump < UNKNOWN_DUMP_MIN_S:
            return
        # Once per screen: the same unknown screen every 30 s filled the folder.
        thumb = shot.thumb
        if any(np.abs(thumb - t).mean() < UNKNOWN_SAME for t in self._dumped):
            return
        self._dumped.append(thumb)
        self._last_unknown_dump = now
        path = self.diagnostics / f"unknown_popup_{time.strftime('%Y%m%d_%H%M%S')}.png"
        imwrite(path, shot.calib)
        log("WARN", f"board hidden {hidden:.0f}s with no known popup -> saved {path.name}")

    # ---- the iPhone's own alerts and banners ------------------------------------------

    def _check_system_ui(self, shot: Shot, now: float) -> bool:
        """Notification banners: a no-tap zone while one is up. System alerts: tap a safe
        button (sysalert.SAFE_LABELS) or none. True while an alert is on screen that the
        watcher is handling (so it isn't an unknown screen)."""
        try:
            self._check_banner(shot, now)
        except Exception as exc:
            log("ERROR", f"banner check failed: {exc!r}")
        try:
            return self._check_alert(shot, now)
        except Exception as exc:
            log("ERROR", f"alert check failed: {exc!r}")
            return False

    @staticmethod
    def _to_calib(shot: Shot, *xy: int) -> tuple[int, ...]:
        h, w = shot.native.shape[:2]
        sx, sy = shot.calib_size[0] / w, shot.calib_size[1] / h
        return tuple(round(v * (sx if i % 2 == 0 else sy)) for i, v in enumerate(xy))

    def _check_banner(self, shot: Shot, now: float) -> None:
        """A banner over the top of the game is waited out: swipes on the board go on,
        taps under it are refused (a tap opens the app that sent it)."""
        if shot is self._banner_shot:
            if self._banner_found:
                self._banner_seen = now  # the same picture: still there
        elif now - self._banner_check >= BANNER_EVERY_S:
            self._banner_check, self._banner_shot = now, shot
            box = sysalert.find_banner(shot.native)
            self._banner_found = box is not None
            if box is not None:
                x0, y0, x1, y1 = self._to_calib(shot, *box)
                p = BANNER_PAD
                zone = (x0 - p, y0 - p, x1 + p, y1 + p)
                if not self._banner_zone:
                    log(
                        "WATCHER",
                        f"notification banner on the iPhone: no taps in {zone} until it goes",
                    )
                self._banner_seen, self._banner_zone = now, True
                self.device.set_dynamic_zone("ios_banner", zone)
        if self._banner_zone and now - self._banner_seen > BANNER_LINGER_S:
            self._banner_zone = False
            self.device.set_dynamic_zone("ios_banner", None)

    def _check_alert(self, shot: Shot, now: float) -> bool:
        if self.board_visible:  # an alert dims the whole screen: the board can't be in view
            self._drop_alert()
            return False
        if shot is not self._alert_shot and now - self._alert_check >= ALERT_EVERY_S:
            self._alert_check, self._alert_shot = now, shot
            self._see_alert(shot, now, sysalert.find_alert(shot.native))
        seen = self.alert
        if seen is None:
            return False
        # The phone's, not the game's: no blind taps (one could hit "Allow"), and the
        # board hidden under it isn't the level ending.
        self.system_seen = self.mid_level_seen = self.last_match = now
        if seen.gave_up:
            return False
        if now - seen.since < ALERT_SETTLE_S:
            return True
        if seen.choice is None:
            seen.choice = self._read_alert(self._alert_shot or shot, seen, now)
        if seen.choice.button is None:
            return False  # nothing safe to tap: an unknown screen (saved once, recovery)
        if now - seen.last_tap < ALERT_TAP_GAP_S or now - self._last_tap < ALERT_HOLDOFF_S:
            return True
        if seen.taps >= ALERT_MAX_TAPS:
            seen.gave_up = True
            log(
                "WARN",
                f"iPhone alert still up after {seen.taps} taps on {seen.choice.label!r}; "
                "leaving it to the unknown-screen recovery",
            )
            return False
        x, y = self._to_calib(self._alert_shot or shot, *seen.alert.center(seen.choice.button))
        label = seen.choice.label
        seen.taps += 1
        seen.last_tap = seen.since = now
        seen.choice = None  # read it again before any other tap: it may be a new alert
        self.last_action = self._last_tap = now
        self.hits[f"ios_alert:{label}"] += 1
        self.device.tap(x, y, why=f"iPhone alert: {label}", allow="ios_alert")
        return True

    def _see_alert(self, shot: Shot, now: float, found: sysalert.Alert | None) -> None:
        if found is None:
            self._drop_alert()
            return
        look = _alert_look(shot.native, found.box)
        zone = self._to_calib(shot, *found.box)
        seen = self.alert
        if (
            seen is None
            or len(seen.alert.buttons) != len(found.buttons)
            or any(abs(a - b) > 12 for a, b in zip(seen.zone, zone, strict=True))
            or np.abs(seen.look - look).mean() > ALERT_SAME
        ):
            self.alert = SeenAlert(found, zone, look, now)  # a new one (or it moved): settle
        else:
            found.labels, found.title = seen.alert.labels, seen.alert.title
            seen.alert, seen.zone, seen.look = found, zone, look
        # No other tap inside it (a game button under it, a blind clear tap): only the
        # alert's own safe button, which names this zone.
        self.device.set_dynamic_zone("ios_alert", zone)

    def _drop_alert(self) -> None:
        if self.alert is not None:
            self.alert = None
            self.device.set_dynamic_zone("ios_alert", None)

    def _read_alert(self, shot: Shot, seen: SeenAlert, now: float) -> sysalert.Choice:
        """OCR the alert and choose a button (or none), with one log line."""
        from .letters import TESSERACT

        a = seen.alert
        if TESSERACT is None:
            if not self._no_ocr_logged:
                self._no_ocr_logged = True
                log(
                    "WARN",
                    "iPhone alert on screen, but Tesseract isn't installed to read it: no tap",
                )
            return sysalert.Choice(None, "Tesseract isn't installed")
        try:
            a.labels = sysalert.read_labels(shot.native, a)
            a.title = sysalert.read_title(shot.native, a)
        except Exception as exc:
            log("ERROR", f"reading the iPhone alert failed: {exc!r}")
            return sysalert.Choice(None, "unreadable")
        choice = sysalert.choose(a.labels, a.title)
        buttons = " | ".join(s or "?" for s in a.labels)
        if choice.trust and now - self._trust_logged >= TRUST_LOG_EVERY_S:
            self._trust_logged = now
            log(
                "WARN",
                'The iPhone asks "Trust This Computer?". A person must tap Trust on the phone '
                "and enter its passcode; the bot never taps it.",
            )
        what = (
            f"tapping {choice.label!r}"
            if choice.button is not None
            else f"not tapping ({choice.label}); left to the unknown-screen recovery"
        )
        log("WATCHER", f'iPhone alert "{a.title}" [{buttons}]: {what}')
        if choice.button is not None and seen.taps == 0 and now - self._alert_dump >= 60:
            # One picture per tapped alert, to check later what was read and tapped (one
            # with nothing safe is saved as an unknown screen)
            self._alert_dump = now
            name = f"ios_alert_{time.strftime('%Y%m%d_%H%M%S')}.png"
            with contextlib.suppress(Exception):
                imwrite(self.diagnostics / name, shot.calib)
        return choice

    def alert_note(self) -> str:
        """The alert on screen, for an error message ("" if none)."""
        a = self.alert.alert if self.alert is not None else None
        if a is None or not a.labels:
            return ""
        return f' (an iPhone alert it may not tap: "{a.title}" [{" | ".join(a.labels)}])'

    # ---- not in the game / stuck ------------------------------------------------

    def _relaunch_game(self, why: str, *, force: bool = False) -> None:
        """Bring the game back (iPhone: kill + launch over USB). Never a tap: on the home
        screen a tap opens whatever app is under it."""
        now = time.monotonic()
        if not force and now - self._last_relaunch < RELAUNCH_COOLDOWN_S:
            return
        self._last_relaunch = now
        log("RECOVERY", f"{why}; relaunching the game")
        if getattr(self.device, "dry_run", False):
            return
        try:
            self.device.app_start(self.package)
        except Exception as exc:
            log("ERROR", f"relaunching the game failed: {exc!r}")

    def _escalate(self, now: float) -> None:
        """Never sit on a screen it doesn't know. The bot's blind taps get the first
        BLIND_TAP_WINDOW_S; then, step by step: relaunch the game, restart it and the
        phone connection, and finally stop with an error (AutomationHQ restarts the bot)."""
        if self._esc_step and self.last_known > self._esc_t:
            log("RECOVERY", "back in the game")
            self._esc_step = 0
        if getattr(self.device, "view_stale", lambda: False)():
            return  # a frozen picture isn't the screen (iphone.py recovers the stream)
        unknown = now - self.last_known
        dry = getattr(self.device, "dry_run", False)
        if self._esc_step == 0 and unknown >= UNKNOWN_RELAUNCH_S:
            self._esc_step, self._esc_t = 1, now
            recent = [t for t in self._esc_relaunches if now - t < 1200]
            self._esc_relaunches.append(now)
            if len(recent) >= 3:  # relaunched, got back, lost again: over and over
                self._fatal(
                    f"lost the game {len(recent) + 1} times in 20 min (unknown screen); "
                    "stopping so the bot can be restarted"
                )
                return
            self._relaunch_game(f"unknown screen for {unknown:.0f}s", force=True)
        elif self._esc_step == 1 and now - self._esc_t >= UNKNOWN_RESTART_S:
            self._esc_step, self._esc_t = 2, now
            log(
                "RECOVERY",
                f"still not the game after {unknown:.0f}s; restarting it and the phone link",
            )
            if not dry:
                try:
                    self.device.reconnect()
                    self.device.app_stop(self.package)
                    time.sleep(1.0)
                    self.device.app_start(self.package)
                except Exception as exc:
                    log("ERROR", f"restarting the game failed: {exc!r}")
        elif self._esc_step == 2 and now - self._esc_t >= UNKNOWN_GIVE_UP_S:
            self._esc_step = 3
            self._fatal(
                f"stuck on a screen it doesn't know for {unknown / 60:.0f} min"
                f"{self.alert_note()}; stopping so the bot can be restarted"
            )

    def _fatal(self, message: str) -> None:
        if self.fatal is not None:
            return
        self.fatal = message
        if self.on_fatal is None:
            log("ERROR", message)
        else:
            self.on_fatal(message)

    def _ensure_foreground(self) -> None:
        pkg = self.device.foreground()
        if pkg and pkg != self.package:
            log("RECOVERY", f"foreground is {pkg}, relaunching {self.package}")
            if not self.device.dry_run:
                self.device.app_start(self.package)
