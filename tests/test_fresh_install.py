"""The screens a brand-new Android install shows before the first level, in order:
the notification prompt (a permission dialog in front of the game), the "Welcome! ... Yes,
I Accept" consent dialog over the dimmed board, and the game's home screen ("Play Level N",
which only the bottom tab bar's orange Home tab is read by, so any N reads the same).

Fixtures are the customer's 1080x2400 frames in tests/data/android/ (the emulator run of
v1.2.2). Taps are in calibration px (1080x2400). Never a coin, ad or store button: the
home's Play Level is the only target on the home screen, and the dialogs' own buttons."""

from __future__ import annotations

import re
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from wsbot import watcher as watcher_mod
from wsbot.device import FORBIDDEN_ZONES, Device, view_center
from wsbot.shot import Shot

ROOT = Path(__file__).resolve().parents[1]
ANDROID = ROOT / "tests" / "data" / "android"
SCREENS = ROOT / "tests" / "data" / "screens"
CALIB = (1080, 2400)
PACKAGE = "in.playsimple.wordsearch"
DENY_ID = "com.android.permissioncontroller:id/permission_deny_button"
ALLOW_ID = "com.android.permissioncontroller:id/permission_allow_button"
DENY_CENTER = (540, 1499)  # "Don't allow", the dump's bounds [133,1426][947,1573]
PLAY_LEVEL = (540, 1993)  # the home's green Play Level button (any level number)
YES_I_ACCEPT = (540, 1447)  # the consent dialog's green button

NEW_ENTRIES = ("welcome_accept", "home_play_level")

# fixture -> (the entry that owns it, where it taps; None: no game entry may read it)
CASES = {
    "welcome_accept.jpg": ("welcome_accept", YES_I_ACCEPT),
    "home_play_level.jpg": ("home_play_level", PLAY_LEVEL),
    "permission_prompt.jpg": (None, None),
}
SIZES = [(1080, 2400), (720, 1600), (1440, 3200)]


@pytest.fixture(scope="module")
def popups():
    return watcher_mod.load_popups(ROOT / "templates")


def load(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    img = cv2.imread(str(path))
    assert img is not None, path
    if size is not None:  # the phone's own size: the bot scales it to calibration itself
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img


def scores_of(popups, img: np.ndarray) -> dict:
    shot = Shot(1, img, CALIB)
    out = {}
    for p in popups:
        found = watcher_mod.match(shot.small, p, shot.coarse_as(p.coarse_look))
        out[p.name] = watcher_mod.match_variants(shot.small, p, found)
    return out


def top(popups, img: np.ndarray):
    return watcher_mod.top_match(popups, scores_of(popups, img))


def near(a, b, tol=12) -> bool:
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


@pytest.mark.parametrize("size", SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("name", sorted(CASES))
def test_fresh_install_screen_maps_to_its_entry_and_tap(name, size, popups):
    entry, expected = CASES[name]
    hit = top(popups, load(ANDROID / name, size))
    if entry is None:
        assert hit is None or hit[0].name not in NEW_ENTRIES, hit and hit[0].name
        return
    assert hit is not None, name
    popup, score, center = hit
    assert popup.name == entry, (name, popup.name, score)
    target = (
        (center[0] + popup.tap_offset[0], center[1] + popup.tap_offset[1])
        if popup.tap_offset
        else center
    )
    assert near(target, expected), (name, target, expected)
    for zone in FORBIDDEN_ZONES.values():
        x1, y1, x2, y2 = zone
        assert not (x1 <= target[0] <= x2 and y1 <= target[1] <= y2), (name, target)


def test_the_new_entries_read_no_other_android_or_screen_fixture(popups):
    """Their templates must not turn an existing screen into a tap on the consent or Play."""
    others = [p for d in (ANDROID, SCREENS) for p in sorted(d.glob("*.jpg"))]
    for path in others:
        if path.name in CASES:
            continue
        sc = scores_of(popups, load(path))
        for name in NEW_ENTRIES:
            assert sc[name][0] < 0.85, (path.name, name, sc[name][0])


def test_view_center_picks_the_deny_button_not_allow():
    xml = (ANDROID / "permission_prompt.xml").read_text(encoding="utf-8")
    assert view_center(xml, DENY_ID) == DENY_CENTER
    assert view_center(xml, ALLOW_ID) == (540, 1342)
    assert view_center(xml, "no.such:id/x") is None
    assert view_center("not xml at all", DENY_ID) is None


class FakeUI:
    """Stands in for uiautomator2: the dump of a given screen."""

    def __init__(self, xml: str) -> None:
        self.xml = xml

    def dump_hierarchy(self) -> str:
        return self.xml


def scaled_dump(size: tuple[int, int]) -> str:
    """The fixture's dump with its bounds in a phone's own px (1080x2400 -> size)."""
    fx, fy = size[0] / CALIB[0], size[1] / CALIB[1]
    xml = (ANDROID / "permission_prompt.xml").read_text(encoding="utf-8")

    def scale(m: re.Match) -> str:
        left, top, right, bottom = map(int, m.groups())
        return f"[{round(left * fx)},{round(top * fy)}][{round(right * fx)},{round(bottom * fy)}]"

    return re.sub(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", scale, xml)


@pytest.mark.parametrize("size", SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_find_view_gives_calibration_px_on_any_phone(size):
    dev = Device.__new__(Device)  # no adb / uiautomator2 needed: the dump is faked
    dev.d = FakeUI(scaled_dump(size))
    dev.sx, dev.sy = size[0] / CALIB[0], size[1] / CALIB[1]
    dev._dump_busy = threading.Event()
    point = dev.find_view(DENY_ID)
    assert point is not None and near(point, DENY_CENTER, tol=3), (size, point)


class FakePhone:
    """Just what the watcher's foreground check touches."""

    calib = CALIB
    platform = "android"
    dry_run = False

    def __init__(self, front: str, deny_on_screen: bool = True) -> None:
        self.front = front
        self.deny_on_screen = deny_on_screen
        self.taps: list[tuple[int, int]] = []
        self.launches: list[str] = []

    def foreground(self) -> str:
        return self.front

    def find_view(self, resource_id: str):
        return DENY_CENTER if self.deny_on_screen and resource_id == DENY_ID else None

    def tap(self, x, y, why="", allow=None) -> bool:
        self.taps.append((x, y))
        return True

    def app_start(self, package: str) -> None:
        self.launches.append(package)


def watcher_for(phone: FakePhone, tmp_path: Path):
    return watcher_mod.PopupWatcher(phone, ROOT / "templates", PACKAGE, tmp_path)


def test_a_permission_prompt_gets_dont_allow_and_no_relaunch(tmp_path):
    phone = FakePhone("com.google.android.permissioncontroller")
    watcher_for(phone, tmp_path)._ensure_foreground()
    assert phone.taps == [DENY_CENTER]
    assert phone.launches == []


def test_a_prompt_not_on_screen_yet_is_left_alone(tmp_path):
    phone = FakePhone("com.google.android.permissioncontroller", deny_on_screen=False)
    watcher_for(phone, tmp_path)._ensure_foreground()
    assert phone.taps == [] and phone.launches == []


def test_the_launcher_in_front_still_relaunches_the_game(tmp_path):
    phone = FakePhone("com.google.android.apps.nexuslauncher")
    watcher_for(phone, tmp_path)._ensure_foreground()
    assert phone.taps == [] and phone.launches == [PACKAGE]


def test_the_game_in_front_is_left_alone(tmp_path):
    phone = FakePhone(PACKAGE)
    watcher_for(phone, tmp_path)._ensure_foreground()
    assert phone.taps == [] and phone.launches == []


class ScriptedPhone(FakePhone):
    """A phone that shows a script of screens, one per tick: (fixture, front package)."""

    def __init__(self, script: list[tuple[str, str]]) -> None:
        super().__init__(script[0][1])
        self.script = script
        self.step = 0
        self.frames = {name: load(ANDROID / name) for name, _ in script}
        self.below_board_y = None

    def shot(self) -> Shot:
        name, front = self.script[min(self.step, len(self.script) - 1)]
        self.front = front
        return Shot(self.step + 1, self.frames[name], CALIB)

    def set_dynamic_zone(self, name, zone) -> None:
        pass

    def set_zones_due(self, due) -> None:
        pass


def test_fresh_start_walks_prompt_welcome_home_then_play(tmp_path):
    """The customer's first launch, as the watcher sees it tick by tick: the notification
    prompt is answered, the consent is accepted, the home's Play Level is tapped."""
    script = [
        ("permission_prompt.jpg", "com.google.android.permissioncontroller"),
        ("welcome_accept.jpg", PACKAGE),
        ("home_play_level.jpg", PACKAGE),
    ]
    phone = ScriptedPhone(script)
    w = watcher_for(phone, tmp_path)
    for step in range(len(script)):
        phone.step = step
        for _ in range(2):  # the second look at the same picture: a cooldown, not a tap
            w._last_app_check = w._last_full = -100.0  # due on every tick (a real tick: ~1 s)
            w._tick()
    assert near(phone.taps[0], DENY_CENTER)
    accept = [i for i, t in enumerate(phone.taps) if near(t, YES_I_ACCEPT, tol=30)]
    play = [i for i, t in enumerate(phone.taps) if near(t, PLAY_LEVEL, tol=30)]
    assert accept and play and accept[0] < play[0], phone.taps
    assert phone.launches == []  # never relaunched over the prompt or the consent
    for x, y in phone.taps:
        for zone in FORBIDDEN_ZONES.values():
            x1, y1, x2, y2 = zone
            assert not (x1 <= x <= x2 and y1 <= y <= y2), (x, y)


def test_the_home_is_left_alone_while_the_bot_rests(tmp_path):
    """Paced play: a rest or a sleep keeps the game on its home screen (no level starts)."""
    phone = ScriptedPhone([("home_play_level.jpg", PACKAGE)])
    w = watcher_for(phone, tmp_path)

    def tick() -> None:
        w._last_app_check = w._last_full = -100.0
        w._tick()

    w.resting.set()
    tick()
    tick()
    assert PLAY_LEVEL not in phone.taps
    w.resting.clear()
    w.idle.set()
    tick()
    assert PLAY_LEVEL not in phone.taps
    w.idle.clear()
    tick()
    assert any(near(t, PLAY_LEVEL) for t in phone.taps), phone.taps
