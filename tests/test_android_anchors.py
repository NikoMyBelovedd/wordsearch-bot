"""Android top bar, the bonus banner's tap and the Daily Wisdom card (1080x2400 calibration).

Seen on a fresh install (AVD, Word Search Explorer):
- "Claim bonus word reward" tapped a fixed (216, 205), but the star button sits at
  about (224, 134): the tap missed and the banner kept coming back over the board.
  The no-tap zones were fixed at the same wrong height. Now the star and back buttons are
  found on the frame and the zones and the tap come from them.
- "Daily Wisdom" had no template, so the bot could not close it and the level was lost.

Fixtures (tests/data/android/): claim_bonus_level8.jpg (level 8, the banner over the
board), daily_wisdom.jpg and daily_wisdom_2.jpg (the card over a dimmed board),
level_board.jpg (a plain level, the bar in full view).
"""

from __future__ import annotations

import collections
import threading
from pathlib import Path

import cv2
import pytest
import uiautomator2

from wsbot import watcher as watcher_mod
from wsbot.device import FORBIDDEN_ZONES, Device, topbar_zones
from wsbot.shot import Shot
from wsbot.topbar import TopBar, locate

ROOT = Path(__file__).resolve().parents[1]
ANDROID = ROOT / "tests" / "data" / "android"
ICONS = {
    n: cv2.imread(str(ROOT / "templates" / "anchors" / f"{n}.png"))
    for n in ("back", "star", "cart")
}
CALIB = (1080, 2400)
BACK, STAR, CART = (94, 138), (224, 134), (984, 132)  # the level frames' top bar
OLD_TAP = (216, 205)  # the fixed point the banner used to tap: never the star
CONTINUE = (540, 1917)  # Daily Wisdom's Continue button
BANNER = (542, 1713)  # the "Claim bonus word reward" banner, level 8
TOL = 8  # px: the same button on another phone size is within this


def load(name: str, size: tuple[int, int] | None = None):
    img = cv2.imread(str(ANDROID / name))
    assert img is not None, name
    if size is not None:  # a phone of another size: its own frame, read in calibration px
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img


def near(point, ref, tol=TOL) -> bool:
    return abs(point[0] - ref[0]) <= tol and abs(point[1] - ref[1]) <= tol


@pytest.fixture(scope="module")
def popups():
    return watcher_mod.load_popups(ROOT / "templates")


def top(popups, img, size=CALIB):
    shot = Shot(1, img, size)
    scores = {}
    for p in popups:
        found = watcher_mod.match(shot.small, p, shot.coarse_as(p.coarse_look))
        scores[p.name] = watcher_mod.match_variants(shot.small, p, found)
    return watcher_mod.top_match(popups, scores)


class FakeU2:
    def window_size(self):
        return CALIB


@pytest.fixture
def device(monkeypatch):
    monkeypatch.setattr(uiautomator2, "connect", lambda serial: FakeU2())
    return Device("FAKE", dry_run=True)  # dry run: taps are decided and logged, never sent


class FakeTaps:
    """The watcher's device for _act: the real anchor, a recorded tap."""

    def __init__(self, anchor_of: Device | None):
        self.anchor_of = anchor_of
        self.taps: list = []

    def anchor(self, name, shot=None):
        return self.anchor_of.anchor(name, shot) if self.anchor_of else None

    def tap(self, x, y, *, why="", allow=None):
        self.taps.append(((x, y), allow))
        return True


def act(popups, device, center):
    watcher = watcher_mod.PopupWatcher.__new__(watcher_mod.PopupWatcher)
    watcher.device, watcher.shot = device, None
    popup = next(p for p in popups if p.name == "tutorial_claim_bonus")
    watcher._act(popup, center)
    return device.taps


# ---- the top bar ---------------------------------------------------------------------


def test_the_top_bar_is_found_on_a_level_frame():
    bar, note = locate(load("claim_bonus_level8.jpg"), ICONS)
    assert bar is not None, note
    assert near(bar.back, BACK) and near(bar.star, STAR) and near(bar.cart, CART), bar


@pytest.mark.parametrize("size", [(720, 1600), (1440, 3200), (1080, 2340), (1440, 3088)])
def test_the_top_bar_is_found_at_other_phone_sizes(size):
    frame = Shot(1, load("claim_bonus_level8.jpg", size), CALIB).calib  # what the device reads
    bar, note = locate(frame, ICONS)
    assert bar is not None, (size, note)
    assert near(bar.back, BACK) and near(bar.star, STAR), (size, bar)


def test_the_star_is_not_at_the_old_fixed_point():
    bar, _ = locate(load("claim_bonus_level8.jpg"), ICONS)
    assert not near(bar.star, OLD_TAP, 40)


@pytest.mark.parametrize(
    "size", [(720, 1600), (1080, 2400), (1440, 3200), (1080, 2340), (1440, 3088)]
)
def test_the_home_gear_is_not_a_top_bar(size):
    # The home screen's gear matched the back arrow (0.96) and the star template (0.45) at
    # the same spot: the star must sit well right of the back arrow to count.
    frame = Shot(1, load("home_play_level.jpg", size), CALIB).calib
    bar, note = locate(frame, ICONS)
    assert bar is None, (size, note)


def test_no_top_bar_on_a_dimmed_card():
    bar, _ = locate(load("daily_wisdom.jpg"), ICONS)
    assert bar is None


def test_zones_sit_on_the_buttons_not_on_the_old_height():
    zones = topbar_zones(TopBar(back=BACK, star=STAR, cart=CART))
    x1, y1, x2, y2 = zones["star_bonus_jar"]
    assert x1 <= STAR[0] <= x2 and y1 <= STAR[1] <= y2
    x1, y1, x2, y2 = zones["back_button"]
    assert x1 <= BACK[0] <= x2 and y1 <= BACK[1] <= y2
    assert not (
        zones["back_button"][0] <= STAR[0] <= zones["back_button"][2] and y1 <= STAR[1] <= y2
    )
    x1, y1, x2, y2 = zones["coins_and_shop"]
    assert x1 <= CART[0] <= x2 and y1 <= CART[1] <= y2


def test_without_a_located_bar_the_fallback_zones_stand(device):
    assert device.anchor("star") is None
    assert device.zones == FORBIDDEN_ZONES
    assert not device.tap(*OLD_TAP)  # the fallback star zone: no tap there unless allowed


def test_a_popup_frame_keeps_the_last_bar_found(device):
    device._find_topbar(Shot(1, load("claim_bonus_level8.jpg"), CALIB).calib)
    star = device.anchor("star")
    assert near(star, STAR)
    device._find_topbar(Shot(2, load("daily_wisdom.jpg"), CALIB).calib)  # dimmed: no bar
    assert device.anchor("star") == star


def test_the_star_and_the_back_arrow_are_zoned_once_found(device):
    device._find_topbar(Shot(1, load("claim_bonus_level8.jpg"), CALIB).calib)
    assert device.tap(*STAR, allow="star_bonus_jar")
    assert not device.tap(*STAR)  # the star needs its popup's allow
    assert not device.tap(*BACK, allow="star_bonus_jar")  # the back arrow never


# ---- the banner -----------------------------------------------------------------------


def test_the_banner_entry_taps_the_star_anchor(popups):
    popup = next(p for p in popups if p.name == "tutorial_claim_bonus")
    assert popup.tap_point == "@star" and popup.allow == "star_bonus_jar"


def test_the_banner_taps_the_detected_star(popups, device):
    device._find_topbar(Shot(1, load("claim_bonus_level8.jpg"), CALIB).calib)
    taps = act(popups, FakeTaps(device), BANNER)  # the banner's centre on the level-8 frame
    assert len(taps) == 1
    (x, y), allow = taps[0]
    assert near((x, y), STAR) and allow == "star_bonus_jar"


def test_already_collected_is_a_passive_cooled_cover(popups):
    # The toast is not a button: tapping it did nothing and the bot re-fired ~20 taps.
    popup = next(p for p in popups if p.name == "already_collected")
    assert popup.tap is False and popup.covers and popup.cooldown >= 3.0


def test_the_banner_does_not_tap_blind_before_the_star_is_found(popups):
    taps = act(popups, FakeTaps(None), (540, 1700))  # the banner's own middle: no anchor
    assert taps == []


# ---- Daily Wisdom -----------------------------------------------------------------------


@pytest.mark.parametrize("name", ["daily_wisdom.jpg", "daily_wisdom_2.jpg", "daily_wisdom_3.jpg"])
def test_daily_wisdom_is_closed_by_its_continue_button(name, popups):
    popup, score, center = top(popups, load(name))
    assert popup.name == "daily_wisdom_continue", (name, score)
    assert near(center, CONTINUE, 6), center  # the match's centre, not a fixed point
    assert popup.allow == "below_board"  # the button sits below the board's panel
    for zone in FORBIDDEN_ZONES.values():
        assert not (zone[0] <= center[0] <= zone[2] and zone[1] <= center[1] <= zone[3])


def test_no_other_android_frame_matches_daily_wisdom(popups):
    entry = next(p for p in popups if p.name == "daily_wisdom_continue")
    for path in sorted(ANDROID.glob("*.jpg")):
        if "daily_wisdom" in path.name:
            continue
        shot = Shot(1, cv2.imread(str(path)), CALIB)
        found = watcher_mod.match(shot.small, entry, shot.coarse_as(entry.coarse_look))
        score = watcher_mod.match_variants(shot.small, entry, found)[0]
        assert score < entry.threshold, (path.name, round(score, 3))


# ---- an anchor that never shows drops its entry; the bot plays on ---------------------
#
# run5: the coach's "@hint" never showed on a plain level (its header matched) and the
# bot looped every 2 s on "not located yet" for 80 s. Now the look is counted: after
# ANCHOR_LOOKS looks the entry is dropped for ANCHOR_DROP_S, logged once, for any entry
# whose tap point is an anchor (word_master "@hint", tutorial_claim_bonus "@star").


class Anchored:
    """A watcher's device whose anchors come and go: `point` None = not on the frame."""

    def __init__(self, point=None):
        self.point = point
        self.taps: list = []

    def anchor(self, name, shot=None):
        return self.point

    def tap(self, x, y, *, why="", allow=None):
        self.taps.append(((x, y), allow))
        return True


def looking_watcher(popup, device, score=0.94, center=(540, 1000)):
    """A PopupWatcher with only the state _handle_popups reads, and `popup` on screen."""
    watcher = watcher_mod.PopupWatcher.__new__(watcher_mod.PopupWatcher)
    watcher.device, watcher.shot = device, None
    watcher._looked = [popup]
    watcher._scores = {popup.name: (score, center)}
    watcher.board_visible, watcher.expected_panel, watcher.fps = False, None, 0.0
    watcher.resting, watcher.idle = threading.Event(), threading.Event()
    watcher._eye_on, watcher._last_score_log = False, 0.0
    watcher._group_hit, watcher._last_tap = {}, -1e9
    watcher._tap_run, watcher._tap_run_told = None, False
    watcher.last_match, watcher.last_action, watcher.last_known = 0.0, 0.0, 0.0
    watcher.system_seen, watcher.mid_level_seen = 0.0, 0.0
    watcher.hits, watcher.level_done = collections.Counter(), threading.Event()
    return watcher


@pytest.fixture
def warnings(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(watcher_mod, "log", lambda kind, msg: seen.append(f"{kind} {msg}"))
    return seen


ANCHOR_ENTRIES = ["word_master", "tutorial_claim_bonus"]  # "@hint" and "@star"


@pytest.mark.parametrize("entry", ANCHOR_ENTRIES)
def test_an_anchor_never_on_the_frame_drops_the_entry_after_three_looks(entry, warnings):
    popups = watcher_mod.load_popups(ROOT / "templates")
    popup = next(p for p in popups if p.name == entry)
    device = Anchored(None)
    watcher = looking_watcher(popup, device)
    step = popup.cooldown + 0.5
    for k in range(watcher_mod.ANCHOR_LOOKS):
        assert watcher._handle_popups(100.0 + k * step) is True
    assert device.taps == []  # never blind
    assert popup.dropped_until == pytest.approx(100.0 + 2 * step + watcher_mod.ANCHOR_DROP_S)
    dropped = [w for w in warnings if "dropping it" in w]
    assert len(dropped) == 1 and entry in dropped[0], warnings

    # while dropped the screen is not a popup: the bot plays on, and nothing is logged again
    count = len(warnings)
    assert watcher._handle_popups(popup.dropped_until - 1.0) is False
    assert device.taps == [] and len(warnings) == count


@pytest.mark.parametrize("entry", ANCHOR_ENTRIES)
def test_a_dropped_entry_that_still_matches_gets_its_looks_again_not_a_tap(entry, warnings):
    popups = watcher_mod.load_popups(ROOT / "templates")
    popup = next(p for p in popups if p.name == entry)
    device = Anchored(None)
    watcher = looking_watcher(popup, device)
    step = popup.cooldown + 0.5
    for k in range(watcher_mod.ANCHOR_LOOKS):
        watcher._handle_popups(100.0 + k * step)
    back = popup.dropped_until
    assert watcher._handle_popups(back + 0.1) is True  # the cool-down is over: a new look
    assert popup.unlocated == 1 and device.taps == []
    assert len([w for w in warnings if "dropping it" in w]) == 1  # the same screen: no new log


@pytest.mark.parametrize("entry", ANCHOR_ENTRIES)
def test_the_anchor_found_in_time_taps_and_resets_the_count(entry, warnings):
    popups = watcher_mod.load_popups(ROOT / "templates")
    popup = next(p for p in popups if p.name == entry)
    device = Anchored(None)
    watcher = looking_watcher(popup, device)
    watcher._handle_popups(100.0)
    watcher._handle_popups(100.0 + popup.cooldown + 0.5)
    assert popup.unlocated == 2 and device.taps == []
    device.point = (222, 333)  # the anchor shows up before the third look
    watcher._handle_popups(100.0 + 2 * (popup.cooldown + 0.5))
    assert device.taps == [((222, 333), popup.allow)]
    assert popup.unlocated == 0 and popup.dropped_until == 0.0
    assert not [w for w in warnings if "dropping it" in w]
