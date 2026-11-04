"""Tutorial arrows (tutorial.py): a dimmed screen where one white arrow points at the one
undimmed target that closes it. The tall-iPhone frames in tests/data/screens_tall/ are real
frames from a tester's 1080x2337 iPhone (the three arrows, the toasts, the boards, the
relaunch and the level-end screens). The other sizes are resizes of the avatar frame, to
check that nothing is measured in pixels."""

from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from wsbot import tutorial
from wsbot import watcher as watcher_mod
from wsbot.device import BaseDevice, SafetyError
from wsbot.ios_device import IOSGameDevice
from wsbot.shot import Shot
from wsbot.watcher import PopupWatcher

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data"
TALL = DATA / "screens_tall"
TALL_SIZE = (1080, 2337)

# fixture -> the box (frame px of 1080x2337) its tap point must fall in
ARROW_TARGETS = {
    "event_avatar_compass_arrow": (20, 460, 200, 660),  # the compass at the left edge
    "event_trophy_arrow": (30, 860, 200, 1060),  # the rank trophy, left edge
    "daily_challenge_down_arrow": (60, 1890, 210, 2030),  # the Daily Challenge calendar
}


def load(name: str) -> np.ndarray:
    img = cv2.imread(str(TALL / f"{name}.jpg"))
    assert img is not None, name
    return img


def inside(point: tuple[int, int], box: tuple[int, int, int, int]) -> bool:
    x, y = point
    x0, y0, x1, y1 = box
    return x0 <= x <= x1 and y0 <= y <= y1


@pytest.mark.parametrize("name", sorted(ARROW_TARGETS))
def test_the_arrow_points_at_its_target(name):
    target = tutorial.find_target(load(name))
    assert target is not None, name
    assert inside(target.center, ARROW_TARGETS[name]), target


@pytest.mark.parametrize("name", sorted(ARROW_TARGETS))
def test_the_arrow_is_found_on_a_darker_or_recompressed_frame(name):
    """A slightly darker screen (a stronger dim, a dimmer display) or heavier JPEG: the
    arrow's brightness and the dim are measured against the frame, not fixed."""
    img = load(name)
    darker = np.clip(img.astype(np.float32) * 0.8, 0, 255).astype(np.uint8)
    _, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 40])
    for frame in (darker, cv2.imdecode(jpeg, cv2.IMREAD_COLOR)):
        target = tutorial.find_target(frame)
        assert target is not None and inside(target.center, ARROW_TARGETS[name]), (name, target)


@pytest.mark.parametrize(
    "size",
    [(750, 1622), (1290, 2791), (1080, 2000), (1080, 2600), (750, 1334)],
    ids=["short-uniform", "pro-max-uniform", "squat", "tall", "se-aspect"],
)
def test_the_same_arrow_on_other_screen_sizes_and_shapes(size):
    """Scaled by the frame: the target is found where the same proportions put it."""
    width, height = size
    resized = cv2.resize(load("event_avatar_compass_arrow"), size, interpolation=cv2.INTER_AREA)
    target = tutorial.find_target(resized)
    assert target is not None, size
    sx, sy = width / TALL_SIZE[0], height / TALL_SIZE[1]
    x0, y0, x1, y1 = ARROW_TARGETS["event_avatar_compass_arrow"]
    scaled = (round(x0 * sx), round(y0 * sy), round(x1 * sx), round(y1 * sy))
    assert inside(target.center, scaled), (size, target)


def _other_fixtures() -> list[Path]:
    return sorted(p for p in DATA.rglob("*.jpg") if p.stem not in ARROW_TARGETS)


@pytest.mark.parametrize("path", _other_fixtures(), ids=lambda p: str(p.relative_to(DATA)))
def test_no_arrow_on_any_other_fixture(path):
    """Every other frame in tests/data (boards, toasts, relaunch, level end, the SE and
    Android screens, the coaches): no tutorial target, so the watcher never taps them."""
    img = cv2.imread(str(path))
    assert img is not None
    assert tutorial.find_target(img) is None, path


def test_the_android_burst_coach_arrow_is_not_a_target():
    """coach_burst.jpg has a real white arrow at the 200-coin burst hint: a priced button,
    never a tutorial's target (the existing coach never taps it either)."""
    img = cv2.imread(str(DATA / "android" / "coach_burst.jpg"))
    assert tutorial.find_target(img) is None


# ---- the watcher: taps the target once the board has been hidden a while -------------


class Phone(BaseDevice):
    """A fake iPhone with the real no-tap zones (BaseDevice._check_safe)."""

    platform = "ios"
    zones = IOSGameDevice.zones
    dry_run = False

    def __init__(self, shot: Shot) -> None:
        self.calib = shot.calib_size
        self.shot_ = shot
        self.taps: list[tuple[int, int, str]] = []
        self.refused: list[tuple[int, int, str]] = []

    def shot(self) -> Shot:
        return self.shot_

    def tap(self, x, y, *, why="", allow=None) -> bool:
        try:
            self._check_safe(x, y, "tap", allow)
        except SafetyError:
            self.refused.append((x, y, why))
            return False
        self.taps.append((x, y, why))
        return True

    def foreground(self) -> str:
        return "pkg"

    def app_start(self, package) -> None:
        pass

    def app_stop(self, package) -> None:
        pass

    def reconnect(self) -> None:
        pass


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(watcher_mod.time, "monotonic", c)
    real_sleep = watcher_mod.time.sleep
    main = threading.main_thread()
    monkeypatch.setattr(
        watcher_mod.time,
        "sleep",
        lambda s: None if threading.current_thread() is main else real_sleep(s),
    )
    return c


def watcher_on(name: str, tmp_path: Path, clock: Clock) -> tuple[PopupWatcher, Phone]:
    img = cv2.imread(str(TALL / f"{name}.jpg")) if (TALL / f"{name}.jpg").exists() else None
    if img is None:
        img = cv2.imread(str(DATA / "android" / f"{name}.jpg"))
    shot = Shot(1, img, (img.shape[1], img.shape[0]))
    phone = Phone(shot)
    w = PopupWatcher(phone, ROOT / "templates" / "ios_tall", "pkg", tmp_path)
    w._last_app_check = float("inf")
    w.last_known = clock()  # the game was seen a moment ago: blind taps are allowed
    return w, phone


def run(w: PopupWatcher, clock: Clock, seconds: float, step: float = 0.5) -> None:
    end = clock.t + seconds
    while clock.t < end - 1e-9:
        clock.t += step
        w.last_known = clock()  # the game stays in front: blind taps stay allowed
        w._tick()


def arrow_taps(phone: Phone) -> list[tuple[int, int, str]]:
    return [t for t in phone.taps if t[2] == "tutorial arrow target"]


def test_the_watcher_waits_then_taps_the_compass(clock, tmp_path):
    w, phone = watcher_on("event_avatar_compass_arrow", tmp_path, clock)
    run(w, clock, 1.0)
    assert arrow_taps(phone) == []  # not yet: a moment's hidden board is normal
    run(w, clock, 3.0)
    taps = arrow_taps(phone)
    assert taps, phone.taps
    x, y, _ = taps[0]
    assert inside((x, y), ARROW_TARGETS["event_avatar_compass_arrow"])


def test_the_watcher_taps_at_most_a_few_times_per_screen(clock, tmp_path):
    w, phone = watcher_on("event_avatar_compass_arrow", tmp_path, clock)
    run(w, clock, 60.0)
    taps = arrow_taps(phone)
    assert 1 <= len(taps) <= watcher_mod.TUTORIAL_MAX_TAPS, len(taps)


def test_the_watcher_never_taps_the_burst_coach_arrow(clock, tmp_path):
    """The Android burst coach: the arrow is real but the target is priced, so the arrow
    path never taps it, whatever the known coach does."""
    w, phone = watcher_on("coach_burst", tmp_path, clock)
    run(w, clock, 20.0)
    assert arrow_taps(phone) == [], phone.taps
