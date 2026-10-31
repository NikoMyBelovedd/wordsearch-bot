"""Android screens a brand-new player meets early, which the bot read as unknown:
the tutorial coach bubbles (a dimmed board, a white bubble, an arrow at a button) and the
"Word Master" reward ladder over the level-complete banner. Fixtures are the customer's
1080x2400 frames (app 0.3.13) in tests/data/android/.

What each one does (tap targets are in calibration px, 1080x2400):
- coach_hint_free: "Use the HINT ..." (level 4). Taps the FREE hint button the arrow points at.
- coach_burst: "The BURST HINT ..." (level 11). Never taps the 200-coin burst button; taps the
  dimmed board above the bubble, then presses Back if it is still there.
- word_master: the reward ladder. Taps the dim backdrop above the popup (no close X), then Back.
- coach_arrow: any other dimmed coach with an arrow (its template is the burst arrow). Taps the
  backdrop only.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from wsbot import watcher as watcher_mod
from wsbot.device import FORBIDDEN_ZONES
from wsbot.shot import Shot

ROOT = Path(__file__).resolve().parents[1]
ANDROID = ROOT / "tests" / "data" / "android"
SCREENS = ROOT / "tests" / "data" / "screens"
ALERTS = ROOT / "tests" / "data" / "alerts"
CALIB = (1080, 2400)
FREE_HINT = (798, 1908)  # the FREE hint button, level 4
BURST_BUTTON = (636, 1908)  # the 200-coin burst button, level 11: must never be tapped

# fixture -> (entry that owns it, how far its tap may be from the expected point)
CASES = {
    "coach_hint_free.jpg": ("coach_hint_free", FREE_HINT),
    "coach_burst.jpg": ("coach_burst", None),
    "word_master.jpg": ("word_master", None),
}


@pytest.fixture(scope="module")
def popups():
    return watcher_mod.load_popups(ROOT / "templates")


def scores_of(popups, shot: Shot) -> dict:
    out = {}
    for p in popups:  # every entry: top_match walks the whole registry in priority order
        found = watcher_mod.match(shot.small, p, shot.coarse_as(p.coarse_look))
        out[p.name] = watcher_mod.match_variants(shot.small, p, found)
    return out


def target_of(popup, center):
    """Where the watcher taps for an entry that owns the frame (see PopupWatcher._act)."""
    dx, dy = popup.tap_offset
    return center[0] + dx, center[1] + dy


def in_zone(point, zone) -> bool:
    x, y = point
    x1, y1, x2, y2 = zone
    return x1 <= x <= x2 and y1 <= y <= y2


def load(name: str, size: tuple[int, int] | None = None) -> np.ndarray:
    img = cv2.imread(str(ANDROID / name))
    assert img is not None, name
    if size is not None:  # a phone of another size: its native frame, not the calibration one
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img


def top(popups, img, size=CALIB):
    shot = Shot(1, img, size)
    hit = watcher_mod.top_match(popups, scores_of(popups, shot))
    return hit


@pytest.mark.parametrize("name", sorted(CASES))
def test_each_android_screen_maps_to_its_entry_and_tap(name, popups):
    entry, expected = CASES[name]
    hit = top(popups, load(name))
    assert hit is not None, name
    popup, score, center = hit
    assert popup.name == entry, (name, popup.name, score)
    target = target_of(popup, center)
    if expected is not None:
        assert abs(target[0] - expected[0]) <= 40 and abs(target[1] - expected[1]) <= 40, target
        assert popup.allow == "below_board"  # the FREE button sits in the booster row
    else:
        assert target[1] < 1240, target  # above the bubble (or the popup), on the dim backdrop
    for zone in FORBIDDEN_ZONES.values():
        assert not in_zone(target, zone), (name, target, zone)
    assert target[1] < 2200  # never the bottom ad row (Word Master's Arrows banner)


def test_the_burst_coach_never_taps_the_burst_button(popups):
    hit = top(popups, load("coach_burst.jpg"))
    popup, _, center = hit
    assert popup.name == "coach_burst"
    target = target_of(popup, center)
    assert abs(target[0] - BURST_BUTTON[0]) > 150 or abs(target[1] - BURST_BUTTON[1]) > 150


@pytest.mark.parametrize("size", [(720, 1600), (1440, 3200), (1080, 2340), (1080, 2400)])
def test_the_same_screens_are_read_on_other_phone_sizes(size, popups):
    for name, (entry, _) in CASES.items():
        ref = top(popups, load(name))
        hit = top(popups, load(name, size))  # the bot still reads it in 1080x2400 calibration
        assert hit is not None and hit[0].name == entry, (name, size)
        assert abs(hit[2][0] - ref[2][0]) <= 40 and abs(hit[2][1] - ref[2][1]) <= 40, (name, size)


def test_an_unknown_dimmed_coach_gets_the_outside_tap(popups):
    """Paint over the burst bubble's keyword line so no keyword entry matches: the arrow
    fallback owns the screen and taps the dim backdrop above the bubble, never the button."""
    img = load("coach_burst.jpg")
    img[1350:1408, 330:680] = (255, 255, 250)
    hit = top(popups, img)
    assert hit is not None and hit[0].name == "coach_arrow", hit and hit[0].name
    target = target_of(hit[0], hit[2])
    assert target[1] < 1240 and not in_zone(target, FORBIDDEN_ZONES["coins_and_shop"])
    assert not hit[0].then_back  # tap-only: a Back press on a guess could leave the level


def test_no_new_entry_fires_on_the_other_screens(popups):
    """The iPhone and alert frames are not Android screens: none may match a new entry."""
    shots = sorted(SCREENS.glob("*.jpg")) + sorted(ALERTS.glob("*.jpg"))
    assert shots
    for path in shots:
        img = cv2.imread(str(path))
        for name, s in scores_of(popups, Shot(1, img, CALIB)).items():
            threshold = next(p.threshold for p in popups if p.name == name)
            assert s[0] < threshold, (path.name, name, round(s[0], 3))


class FakeDevice:
    def __init__(self):
        self.calls = []

    def tap(self, x, y, *, why="", allow=None):
        self.calls.append(("tap", (x, y), allow))
        return True

    def back(self, *, why=""):
        self.calls.append(("back",))
        return True


def test_then_back_taps_once_then_presses_back_then_stops(popups):
    popup = next(p for p in popups if p.name == "coach_burst")
    watcher = watcher_mod.PopupWatcher.__new__(watcher_mod.PopupWatcher)
    watcher.device = FakeDevice()
    for _ in range(4):
        watcher._act(popup, (502, 1379))
    assert watcher.device.calls == [("tap", (502, 1139), None), ("back",)]


def test_the_log_forgets_taps_only_after_the_entry_is_gone(popups):
    popup = next(p for p in popups if p.name == "coach_hint_free")
    watcher = watcher_mod.PopupWatcher.__new__(watcher_mod.PopupWatcher)
    popup.appearance_taps, popup.last_seen = 2, 4.9
    watcher._forget_taps(popup, 5.0)  # a flicker: still counted
    assert popup.appearance_taps == 2
    popup.last_seen = 1.0
    watcher._forget_taps(popup, 5.0)  # gone for a while: a new appearance starts over
    assert popup.appearance_taps == 0
