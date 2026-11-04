"""Android level-complete screens: the "SPLENDID!" banner (MAGNIFICENT and BRILLIANT are the
same screen with another word). Next Level is seen at once with or without the reward
wheel, with or without a ribbon under it (Twister, Sequence, Fun Fact, Picture Puzzle), at
its normal size and pulsed. Fixtures are a customer's 1080x2400 frames in tests/data/android/.

What must hold (tap targets are in calibration px, 1080x2400):
- the Next Level button is recognized on the first look, and the tap lands inside it;
- the tap passes the safety check: with the reward wheel the button sits below the board
  (the below-board zone), so the entry must allow it;
- a Next Level look while a level's board is in view never taps (the booster row is there).

Other sizes are the same frames resized: scale only, at the same aspect ratio, except the
1080x2340 one (a 19.5:9 phone, stretched 2.6% by the calibration). No other phone was
available, so the real layouts are the one 1080x2400 Android phone's.
"""

from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from wsbot import watcher as watcher_mod
from wsbot.device import BaseDevice, SafetyError
from wsbot.shot import Shot
from wsbot.watcher import PopupWatcher

ROOT = Path(__file__).resolve().parents[1]
ANDROID = ROOT / "tests" / "data" / "android"
CALIB = (1080, 2400)
SIZES = [CALIB, (720, 1600), (1440, 3200), (1080, 2340)]
NEXT = ("next_level", "next_level_text")

# fixture -> the Next Level button's box in calibration px (x1, y1, x2, y2), measured on it
LAYOUTS = {
    "next_level_plain": (297, 1694, 783, 1817),
    "next_level_sequence": (301, 1694, 778, 1818),  # a purple ribbon under it
    "next_level_twister_pulsed": (286, 1691, 794, 1822),  # a red ribbon, pulsed (1.05x)
    "next_level_wheel": (297, 1964, 783, 2087),  # the reward wheel above: Next Level lower
    "next_level_wheel_twister": (301, 1964, 778, 2087),
    "next_level_wheel_pulsed": (281, 1959, 799, 2092),  # a bigger pulse (1.07x), a ribbon
}
WHEEL = {"next_level_wheel", "next_level_wheel_twister", "next_level_wheel_pulsed"}
PULSED = {"next_level_twister_pulsed", "next_level_wheel_pulsed"}
BOARD = "board_level_23"  # a level in play: the board in view
OTHER_SCREENS = [
    BOARD,
    "daily_wisdom_3",  # Daily Wisdom: a green Continue button
    "coach_hint_free",
    "coach_burst",
    "word_master",
    "welcome_accept",
    "home_play_level",
    "permission_prompt",
]


def _load(name: str, size: tuple[int, int]) -> np.ndarray:
    img = cv2.imread(str(ANDROID / f"{name}.jpg"))
    assert img is not None, name
    if size != CALIB:  # a phone of another size: its native frame, the bot calibrates it
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img


def shot_of(name: str, size: tuple[int, int] = CALIB, seq: int = 1) -> Shot:
    return Shot(seq, _load(name, size), CALIB)


def scores_of(popups, shot: Shot) -> dict:
    out = {}
    for p in popups:  # every entry: top_match walks the whole registry in priority order
        found = watcher_mod.match(shot.small, p, shot.coarse_as(p.coarse_look))
        out[p.name] = watcher_mod.match_variants(shot.small, p, found)
    return out


def inside(point, box) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def below_board_y(panel: tuple[int, int, int, int]) -> int:
    """The bot's no-tap zone under a level's board (see bot.py, where the level starts)."""
    return panel[1] + panel[3] + 5


@pytest.fixture(scope="module")
def popups():
    return watcher_mod.load_popups(ROOT / "templates")


@pytest.mark.parametrize("size", SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("name", sorted(LAYOUTS))
def test_each_level_complete_layout_is_next_level_on_the_first_look(name, size, popups):
    shot = shot_of(name, size)
    scores = scores_of(popups, shot)
    hit = watcher_mod.top_match(popups, scores)
    top = {k: round(v[0], 2) for k, v in scores.items() if v[0] > 0.6}
    assert hit is not None, (name, size, top)
    popup, score, center = hit
    assert popup.name in NEXT, (name, popup.name, score)
    assert popup.level_done and not popup.avoid
    assert inside(center, LAYOUTS[name]), (name, size, center)
    if name in PULSED:  # the pulse is what the scaled copies are for
        assert popup.name == "next_level_text" and score >= popup.threshold
    # the tap must pass the safety check; below the board it needs the allow
    phone = PhoneStub(below_board=below_board_y(BOARD_PANEL))
    phone._check_safe(*center, "tap", popup.allow)  # no SafetyError
    if name in WHEEL:
        with pytest.raises(SafetyError):  # the wheel layout is below the board's zone
            phone._check_safe(*center, "tap", None)


def test_the_plain_layout_is_above_the_board_zone():
    """Without the wheel the button is above the zone: it never needed the allow."""
    phone = PhoneStub(below_board=below_board_y(BOARD_PANEL))
    x1, y1, x2, y2 = LAYOUTS["next_level_plain"]
    phone._check_safe((x1 + x2) // 2, (y1 + y2) // 2, "tap", None)


@pytest.mark.parametrize("name", OTHER_SCREENS)
def test_no_other_android_screen_scores_as_next_level(name, popups):
    shot = shot_of(name)
    scores = scores_of(popups, shot)
    for entry in NEXT:
        popup = next(p for p in popups if p.name == entry)
        assert scores[entry][0] < popup.threshold, (name, entry, round(scores[entry][0], 3))


# ---- the watcher acting on the screens ---------------------------------------------


class PhoneStub(BaseDevice):
    """The bot's device contract: taps go through the real safety check (_check_safe)."""

    def __init__(self, shots: list[Shot] | None = None, below_board: int | None = None) -> None:
        self.shots = shots or []
        self.below_board_y = below_board
        self.taps: list[tuple[int, int, str]] = []
        self.refusals: list[tuple[int, int, str]] = []

    def shot(self) -> Shot:
        return self.shots.pop(0) if len(self.shots) > 1 else self.shots[0]

    def tap(self, x, y, *, why="", allow=None) -> bool:
        try:
            self._check_safe(x, y, "tap", allow)
        except SafetyError:
            self.refusals.append((x, y, why))
            return False
        self.taps.append((x, y, why))
        return True

    def back(self, *, why="") -> bool:
        return True

    def foreground(self) -> str:
        return "pkg"


BOARD_PANEL = shot_of(BOARD).panel  # the level in play: its board's panel (calibration px)


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(watcher_mod.time, "monotonic", c)
    # a no-op sleep for the test's thread: the watcher's background loops must not spin
    real_sleep = watcher_mod.time.sleep
    main = threading.main_thread()
    monkeypatch.setattr(
        watcher_mod.time,
        "sleep",
        lambda s: None if threading.current_thread() is main else real_sleep(s),
    )
    return c


def watcher_on(shots: list[Shot], tmp_path: Path, panel) -> tuple[PopupWatcher, PhoneStub]:
    phone = PhoneStub(shots, below_board=below_board_y(panel))
    w = PopupWatcher(phone, ROOT / "templates", "pkg", tmp_path)
    w._last_app_check = float("inf")
    w.expected_panel = panel  # the bot sets this while it solves a level
    return w, phone


def run(w: PopupWatcher, clock: Clock, ticks: int) -> None:
    for _ in range(ticks):
        clock.t += 1.0
        w._tick()


@pytest.mark.parametrize("size", SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("name", sorted(LAYOUTS))
def test_level_complete_is_tapped_on_the_first_look_no_refusal(name, size, clock, tmp_path):
    panel = shot_of(BOARD, size).panel
    board = [shot_of(BOARD, size, seq=i) for i in range(1, 4)]
    end = shot_of(name, size, seq=10)
    w, phone = watcher_on([*board, end], tmp_path, panel)
    run(w, clock, 3)  # the level in view: nothing to tap
    assert phone.taps == [] and phone.refusals == []
    run(w, clock, 1)  # the first look at the level-complete screen
    assert len(phone.taps) == 1, phone.taps
    x, y, why = phone.taps[0]
    assert why in NEXT and inside((x, y), LAYOUTS[name]), (name, size, phone.taps)
    assert phone.refusals == []  # no SAFETY refusal for this known button
    assert w.level_done.is_set()  # the level is over: the bot stops swiping now
    assert w.last_match == clock.t  # seen this tick: no clear-popup tap can come first
    assert not list(tmp_path.glob("unknown_popup_*"))
    run(w, clock, 1)  # still on screen, but inside its cooldown: no second tap
    assert len(phone.taps) == 1, phone.taps


def test_a_next_level_look_during_a_level_never_taps(clock, tmp_path):
    """Paint the plain button over a level in play: the board is in view, so the entry is
    not looked at (off_board) and the booster row under it is never tapped."""
    board = _load(BOARD, CALIB)
    painted = _load("next_level_plain", CALIB)
    board[1694:1817, 297:783] = painted[1694:1817, 297:783]
    frame = Shot(1, board, CALIB)
    panel = frame.panel
    assert panel is not None
    w, phone = watcher_on([Shot(i, board, CALIB) for i in range(1, 6)], tmp_path, panel)
    run(w, clock, 5)
    assert w.board_visible
    assert phone.taps == [] and phone.refusals == []
    assert not w.level_done.is_set()


def test_the_next_level_entries_are_allowed_below_the_board_and_share_a_cooldown(popups):
    entries = {p.name: p for p in popups if p.name in NEXT}
    assert set(entries) == set(NEXT)
    for p in entries.values():
        assert p.allow == "below_board" and p.level_done and p.off_board
    assert entries["next_level"].group == entries["next_level_text"].group == "next_level"
