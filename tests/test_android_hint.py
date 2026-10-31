"""The "Use the HINT" coach (word_master) taps the FREE hint button, found on the frame.

A fixed offset from the bubble (the old tap_offset 246, 1444) only lands on the button at one
phone's size: the bubble and the booster row move with the frame's height and aspect. Now
popups.json names "@hint", and the watcher resolves it on the frame the entry matched
(Device.anchor -> hintbutton.find_hint). Fixture: tests/data/android/word_master.jpg (level 4,
1080x2400). Variants: the same screen resized to other phones (the device stretches every
frame to the 1080x2400 calibration, as Device.frame does), and padded with a notch band or a
bottom band, which changes the aspect ratio the bubble and the row sit in.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from wsbot import watcher as watcher_mod
from wsbot.device import Device
from wsbot.hintbutton import find_hint
from wsbot.shot import Shot

ROOT = Path(__file__).resolve().parents[1]
ANDROID = ROOT / "tests" / "data" / "android"
CALIB = (1080, 2400)
DISC = (783, 1911)  # centre of the FREE disc on the fixture
DISC_BOX = (711, 1841, 855, 1981)  # its bounding box (x1, y1, x2, y2) on the fixture
TOLERANCE = 6  # px, calibration space
ICON = cv2.imread(str(ROOT / "templates" / "anchors" / "hint.png"))

# name, pad above, pad below (black bands, native px), resized to (W, H) or None
VARIANTS = [
    ("fixture", 0, 0, None),
    ("720x1600", 0, 0, (720, 1600)),
    ("1080x2340", 0, 0, (1080, 2340)),
    ("1440x3088", 0, 0, (1440, 3088)),
    ("notch_top90", 90, 0, None),
    ("bottom_band150", 0, 150, None),
]


def phone_frame(pad_top: int, pad_bottom: int, size) -> np.ndarray:
    """The native frame of a phone showing the fixture screen."""
    img = cv2.imread(str(ANDROID / "word_master.jpg"))
    if pad_top or pad_bottom:
        bands = (
            np.zeros((pad_top, 1080, 3), np.uint8),
            img,
            np.zeros((pad_bottom, 1080, 3), np.uint8),
        )
        img = np.vstack(bands)
    if size is not None:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img


def calibrated(native: np.ndarray) -> np.ndarray:
    """What Device.frame hands the watcher: the native frame stretched to 1080x2400."""
    if native.shape[1] != CALIB[0] or native.shape[0] != CALIB[1]:
        return cv2.resize(native, CALIB, interpolation=cv2.INTER_AREA)
    return native


def expected_disc(pad_top: int, pad_bottom: int) -> tuple[int, int]:
    """Where the disc is on the calibration frame: its share of the native frame's size."""
    fx = DISC[0] / 1080
    fy = (DISC[1] + pad_top) / (2400 + pad_top + pad_bottom)
    return round(fx * CALIB[0]), round(fy * CALIB[1])


class TapLog:
    """A real Device, with only its input swapped for a recorder."""

    def __init__(self) -> None:
        self.device = Device.__new__(Device)
        self.device._anchors = {}
        self.device._hint_icon = ICON
        self.taps: list[tuple[tuple[int, int], str | None]] = []
        self.device.tap = lambda x, y, *, why="", allow=None: self.taps.append(((x, y), allow))


@pytest.fixture(scope="module")
def popups():
    return watcher_mod.load_popups(ROOT / "templates")


def scores_of(popups, shot: Shot) -> dict:
    out = {}
    for p in popups:
        found = watcher_mod.match(shot.small, p, shot.coarse_as(p.coarse_look))
        out[p.name] = watcher_mod.match_variants(shot.small, p, found)
    return out


def close(point, target, tolerance=TOLERANCE) -> bool:
    return abs(point[0] - target[0]) <= tolerance and abs(point[1] - target[1]) <= tolerance


@pytest.mark.parametrize("name,pad_top,pad_bottom,size", VARIANTS, ids=[v[0] for v in VARIANTS])
def test_the_disc_is_found_on_each_frame(name, pad_top, pad_bottom, size):
    frame = calibrated(phone_frame(pad_top, pad_bottom, size))
    centre = find_hint(frame, ICON)
    assert centre is not None, name
    assert close(centre, expected_disc(pad_top, pad_bottom)), (name, centre)


@pytest.mark.parametrize("name,pad_top,pad_bottom,size", VARIANTS, ids=[v[0] for v in VARIANTS])
def test_word_master_taps_the_disc_on_each_frame(name, pad_top, pad_bottom, size, popups):
    shot = Shot(1, calibrated(phone_frame(pad_top, pad_bottom, size)), CALIB)
    hit = watcher_mod.top_match(popups, scores_of(popups, shot))
    assert hit is not None and hit[0].name == "word_master", (name, hit and hit[0].name)
    popup, _, center = hit
    log = TapLog()
    watcher = watcher_mod.PopupWatcher.__new__(watcher_mod.PopupWatcher)
    watcher.device, watcher.shot = log.device, shot
    watcher._act(popup, center)
    assert len(log.taps) == 1, (name, log.taps)
    (x, y), allow = log.taps[0]
    assert close((x, y), expected_disc(pad_top, pad_bottom)), (name, (x, y))
    assert allow == "below_board"  # the disc sits in the booster row, under the board


def test_the_old_offset_from_the_bubble_misses_the_disc_on_a_taller_frame(popups):
    """Why the anchor: with a 150 px band below the screen the bubble's centre plus the old
    fixed offset lands ~80 px below the disc (its radius is ~72): a miss."""
    shot = Shot(1, calibrated(phone_frame(0, 150, None)), CALIB)
    _, _, center = watcher_mod.top_match(popups, scores_of(popups, shot))
    old = (center[0] + 246, center[1] + 1444)
    x, y = expected_disc(0, 150)
    assert abs(old[1] - y) > 72 and close(find_hint(shot.calib, ICON), (x, y)), (old, y)


def test_no_tap_when_the_disc_is_not_on_the_frame(popups):
    """The bubble is up but the FREE disc is not found: log it, tap nothing."""
    img = phone_frame(0, 0, None)
    x1, y1, x2, y2 = DISC_BOX
    img[y1:y2, x1:x2] = (40, 70, 30)  # the lake behind the disc
    shot = Shot(1, calibrated(img), CALIB)
    assert find_hint(shot.calib, ICON) is None
    hit = watcher_mod.top_match(popups, scores_of(popups, shot))
    assert hit is not None and hit[0].name == "word_master"
    log = TapLog()
    watcher = watcher_mod.PopupWatcher.__new__(watcher_mod.PopupWatcher)
    watcher.device, watcher.shot = log.device, shot
    watcher._act(hit[0], hit[2])
    assert log.taps == []


def test_a_normal_level_frame_never_taps_the_hint(popups):
    """A plain level (no coach bubble): no entry matches, so no hint tap, and the disc
    anchor finds nothing on it either. The other fixture screens: none is word_master."""
    level = Shot(1, calibrated(phone_frame(0, 0, None)), CALIB)
    level_img = cv2.imread(str(ANDROID / "level_board.jpg"))
    shot = Shot(1, level_img, CALIB)
    hit = watcher_mod.top_match(popups, scores_of(popups, shot))
    assert hit is None or hit[0].name != "word_master", hit and hit[0].name
    assert find_hint(shot.calib, ICON) is None
    for path in sorted(ANDROID.glob("*.jpg")):
        if path.name == "word_master.jpg":
            continue
        other = Shot(1, cv2.imread(str(path)), CALIB)
        hit = watcher_mod.top_match(popups, scores_of(popups, other))
        assert hit is None or hit[0].name != "word_master", (path.name, hit and hit[0].name)
    assert level.calib.shape[:2] == (CALIB[1], CALIB[0])


def test_word_master_never_backs_out():
    popups = watcher_mod.load_popups(ROOT / "templates")
    popup = next(p for p in popups if p.name == "word_master")
    assert not popup.then_back and popup.allow == "below_board"
    assert popup.tap_point == "@hint" and popup.tap_offset is None
