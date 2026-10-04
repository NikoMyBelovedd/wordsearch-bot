"""Screens the iPhone bot met on a night run (iOS 27, iPhone SE) and what it must do on
each: tap the right button, wait, relaunch the game, never tap an ad or "Allow".

tests/data/screens/*.jpg are real frames at the phone's size (750x1334)."""

from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from wsbot import watcher as watcher_mod
from wsbot.ios_device import IOSGameDevice
from wsbot.shot import Shot
from wsbot.watcher import PopupWatcher

ROOT = Path(__file__).resolve().parents[1]
SCREENS = ROOT / "tests" / "data" / "screens"
CALIB = (1296, 2305)

# screen -> (entry that owns it, what the bot does)
EXPECTED = {
    "level_complete_no_button": ("level_complete", "wait"),
    "next_level_plain": ("next_level", "tap"),
    "next_level_pulsed_mid": ("next_level_text", "tap"),
    "next_level_pulsed_big": ("next_level_text", "tap"),
    "next_level_picture_puzzle_big": ("next_level_text", "tap"),
    "next_level_get_reward": ("next_level", "tap"),
    "next_level_get_reward_twister": ("next_level_text", "tap"),
    "country_gift": ("country_gift", "tap"),
    "country_gift_squashed": ("country_gift", "tap"),
    "country_coin": ("country_coin", "wait"),
    "keep_playing": ("keep_playing", "tap"),
    "loading": ("loading", "wait"),
    "home": ("ios_home_game_icon", "relaunch"),
    "home_no_status_bar": ("ios_home_game_icon", "relaunch"),
    "app_switcher": ("ios_app_switcher", "relaunch"),
    "alert_watch_notifications": ("ios_dont_allow", "tap"),
    "claim_bonus_tutorial": ("tutorial_claim_bonus", "tap"),
    "board": (None, None),
    "board_letters_flying": (None, None),
}


def shot_of(name: str, seq: int = 1) -> Shot:
    img = cv2.imread(str(SCREENS / f"{name}.jpg"))
    assert img is not None, name
    return Shot(seq, img, CALIB)


@pytest.fixture(scope="module")
def popups():
    return watcher_mod.load_popups(ROOT / "templates" / "ios")


def scores_of(popups, shot: Shot) -> dict:
    out = {}
    for p in popups:
        found = watcher_mod.match(shot.small, p, shot.coarse_as(p.coarse_look))
        out[p.name] = watcher_mod.match_variants(shot.small, p, found)
    return out


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_each_screen_maps_to_the_right_action(name, popups):
    entry, action = EXPECTED[name]
    hit = watcher_mod.top_match(popups, scores_of(popups, shot_of(name)))
    if entry is None:
        assert hit is None, f"{name}: {hit and hit[0].name}"
        return
    assert hit is not None, name
    popup = hit[0]
    assert popup.name == entry
    got = "relaunch" if popup.relaunch else "tap" if popup.tap else "wait"
    assert got == action


def test_the_pulsing_next_level_button_is_found_at_every_size(popups):
    """Next Level pulses 1.0x-1.10x: at the wrong size the template alone scored 0.47-0.75
    and the bot sat on the level-complete screen until a blind tap."""
    for name in ("next_level_pulsed_mid", "next_level_pulsed_big", "next_level_picture_puzzle_big"):
        shot = shot_of(name)
        nlt = next(p for p in popups if p.name == "next_level_text")
        alone = watcher_mod.match(shot.small, nlt, shot.coarse_as(nlt.coarse_look))
        assert alone[0] < nlt.threshold  # what the bot saw before
        with_sizes = watcher_mod.match_variants(shot.small, nlt, alone)
        assert with_sizes[0] >= 0.88
        assert abs(with_sizes[1][0] - 650) < 12  # the button's center, not elsewhere


def test_get_reward_zone_covers_the_ad_wheel_but_not_next_level_or_clear_taps(popups):
    shot = shot_of("next_level_get_reward_twister")
    scores = scores_of(popups, shot)
    wheel = next(p for p in popups if p.name == "get_reward_wheel")
    score, (cx, cy) = scores["get_reward_wheel"]
    assert score >= wheel.threshold
    left, top, right, bottom = wheel.avoid_box
    zone = (cx + left, cy + top, cx + right, cy + bottom)

    def inside(x, y):
        return zone[0] <= x <= zone[2] and zone[1] <= y <= zone[3]

    assert inside(cx, cy)  # the Get reward button
    assert inside(648, 1600)  # the reward wheel above it
    hit = watcher_mod.top_match(popups, scores)
    assert hit is not None and not inside(*hit[2])  # Next Level stays tappable
    assert not inside(*IOSGameDevice.board_center)
    assert not inside(*IOSGameDevice.above_board)


def test_dont_allow_is_tapped_never_allow(popups):
    shot = shot_of("alert_watch_notifications")
    hit = watcher_mod.top_match(popups, scores_of(popups, shot))
    assert hit is not None and hit[0].name == "ios_dont_allow"
    x, y = hit[2]
    # "Allow" is the right-hand button (x >= ~700 at this size); Don't Allow the left one
    assert x < 560 and 1330 < y < 1430
    # and the template doesn't fit the Allow button
    dont = next(p for p in popups if p.name == "ios_dont_allow")
    right = shot.small[:, 330:]
    res = cv2.matchTemplate(right, dont.template, cv2.TM_CCOEFF_NORMED)
    assert res.max() < 0.6


def test_tall_registry_loads_with_the_same_new_screens():
    names = [p.name for p in watcher_mod.load_popups(ROOT / "templates" / "ios_tall")]
    for n in ("level_complete", "country_gift", "keep_playing", "ios_dont_allow", "loading"):
        assert n in names
    # system UI first (over everything), screens to wait on last (anything else wins)
    assert names[0] == "ios_dont_allow"
    assert names[-2:] == ["level_complete", "loading"]


# ---- the watcher acting on these screens ------------------------------------------


class Phone:
    calib = CALIB
    dry_run = False

    def __init__(self, shots: list[Shot]) -> None:
        self.shots = shots
        self.taps: list[tuple[int, int, str]] = []
        self.launches = 0
        self.stops = 0
        self.reconnects = 0
        self.zones: dict = {}

    def shot(self) -> Shot:
        return self.shots.pop(0) if len(self.shots) > 1 else self.shots[0]

    def tap(self, x, y, why="", allow=None) -> bool:
        self.taps.append((x, y, why))
        return True

    def set_dynamic_zone(self, name, zone) -> None:
        self.zones[name] = zone

    def foreground(self) -> str:
        return "pkg"

    def app_start(self, package) -> None:
        self.launches += 1

    def app_stop(self, package) -> None:
        self.stops += 1

    def reconnect(self) -> None:
        self.reconnects += 1


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(watcher_mod.time, "monotonic", c)
    # watcher_mod.time is the time module: a no-op sleep for every thread turned other
    # tests' leftover background loops (time.sleep(10)) into busy loops (tests 100x slower)
    real_sleep = watcher_mod.time.sleep
    main = threading.main_thread()
    monkeypatch.setattr(
        watcher_mod.time,
        "sleep",
        lambda s: None if threading.current_thread() is main else real_sleep(s),
    )
    return c


def watcher_on(shots: list[Shot], tmp_path: Path) -> tuple[PopupWatcher, Phone]:
    phone = Phone(shots)
    w = PopupWatcher(phone, ROOT / "templates" / "ios", "pkg", tmp_path)
    w._last_app_check = float("inf")
    return w, phone


def run(w: PopupWatcher, clock: Clock, seconds: float, step: float = 1.0) -> None:
    end = clock.t + seconds
    while clock.t < end:
        clock.t += step
        w._tick()


def test_level_complete_is_waited_on_then_next_level_tapped_once(clock, tmp_path):
    shots = [shot_of("level_complete_no_button", i) for i in range(1, 9)]
    shots += [shot_of("next_level_pulsed_big", 9)]
    shots += [shot_of("next_level_plain", i) for i in range(10, 20)]
    w, phone = watcher_on(shots, tmp_path)
    run(w, clock, 8)
    assert phone.taps == []  # waited: no blind taps, no "unknown" dumps
    assert w.level_done.is_set()
    assert not list(tmp_path.glob("unknown_popup_*"))
    assert w.blind_taps_ok()  # known screen: we're in the game
    # Tapped at its big size, then seen at its normal size by another template: one
    # group, one cooldown, no second tap (it can land on the next level's booster row)
    run(w, clock, 2)
    assert [t[2] for t in phone.taps] == ["next_level_text"]


def test_home_screen_relaunches_the_game_and_never_taps(clock, tmp_path):
    w, phone = watcher_on([shot_of("home", i) for i in range(1, 100)], tmp_path)
    run(w, clock, 10)
    assert phone.taps == []
    assert phone.launches == 1  # once, then RELAUNCH_COOLDOWN_S
    assert not w.blind_taps_ok()  # the bot's own clear taps are off too
    run(w, clock, watcher_mod.RELAUNCH_COOLDOWN_S)
    assert phone.launches == 2


def test_dont_allow_alert_is_dismissed(clock, tmp_path):
    w, phone = watcher_on([shot_of("alert_watch_notifications")], tmp_path)
    run(w, clock, 2)
    assert [t[2] for t in phone.taps] == ["ios_dont_allow"]
    assert phone.taps[0][0] < 560
    assert not w.blind_taps_ok()


def test_country_complete_taps_the_gift_and_waits_on_the_reward(clock, tmp_path):
    w, phone = watcher_on([shot_of("country_gift")], tmp_path)
    run(w, clock, 2)
    assert [t[2] for t in phone.taps] == ["country_gift"]
    x, y, _ = phone.taps[0]
    assert abs(x - 650) < 20 and 1350 < y < 1600  # on the gift box

    w, phone = watcher_on([shot_of("country_coin", i) for i in range(1, 50)], tmp_path)
    run(w, clock, 3)
    assert phone.taps == []  # Collect comes by itself
    run(w, clock, 4)
    assert [t[2] for t in phone.taps] == ["country_coin"]  # tap_after: still there


def test_unknown_screen_escalates_step_by_step(clock, tmp_path):
    """No board, nothing known: relaunch, then restart + reconnect, then give up (exit)."""
    blank = np.full((1334, 750, 3), (40, 30, 20), np.uint8)
    w, phone = watcher_on([Shot(1, blank, CALIB)], tmp_path)
    fatal = []
    w.on_fatal = fatal.append
    run(w, clock, watcher_mod.BLIND_TAP_WINDOW_S - 1)
    assert w.blind_taps_ok()
    run(w, clock, 2)
    assert not w.blind_taps_ok()  # too long without proof we're in the game
    run(w, clock, watcher_mod.UNKNOWN_RELAUNCH_S - watcher_mod.BLIND_TAP_WINDOW_S)
    assert phone.launches == 1 and phone.stops == 0
    run(w, clock, watcher_mod.UNKNOWN_RESTART_S)
    assert phone.reconnects == 1 and phone.stops == 1 and phone.launches == 2
    assert not fatal
    run(w, clock, watcher_mod.UNKNOWN_GIVE_UP_S)
    assert len(fatal) == 1 and "stuck" in fatal[0]
    assert phone.taps == []
    # the same unknown screen is saved once, not every few seconds
    assert len(list(tmp_path.glob("unknown_popup_*"))) == 1


def test_back_in_the_game_resets_the_escalation(clock, tmp_path):
    blank = np.full((1334, 750, 3), (40, 30, 20), np.uint8)
    shots = [Shot(i, blank, CALIB) for i in range(1, 50)]
    shots += [shot_of("loading", i) for i in range(50, 60)]
    shots += [Shot(i, blank, CALIB) for i in range(60, 200)]
    w, phone = watcher_on(shots, tmp_path)
    fatal = []
    w.on_fatal = fatal.append
    run(w, clock, 49)
    assert phone.launches == 1 and w._esc_step == 1
    run(w, clock, 10)  # the relaunched game shows its loading screen
    assert w._esc_step == 0
    run(w, clock, 40)
    assert phone.launches == 1  # the clock started again from the loading screen


def test_relaunch_loops_give_up(clock, tmp_path):
    """Relaunched, back in the game, lost again: over and over -> stop with an error."""
    blank = np.full((1334, 750, 3), (40, 30, 20), np.uint8)
    shots = []
    for k in range(5):
        shots += [Shot(1000 * k + i, blank, CALIB) for i in range(50)]
        shots += [shot_of("loading", 1000 * k + 500 + i) for i in range(3)]
    w, phone = watcher_on(shots, tmp_path)
    fatal = []
    w.on_fatal = fatal.append
    run(w, clock, 5 * 53)
    assert phone.launches == 3
    assert len(fatal) == 1 and "lost the game" in fatal[0]


def test_a_button_that_never_goes_away_escalates(clock, tmp_path):
    """A Bonus "Claim" tapped 317 times in 15 min, the game taking none of them: every
    tap counted as "in the game", so only AutomationHQ's stall check got it out."""
    w, phone = watcher_on([shot_of("keep_playing", i) for i in range(1, 400)], tmp_path)
    fatal = []
    w.on_fatal = fatal.append
    run(w, clock, watcher_mod.TAP_STUCK_S - 5)
    assert len(phone.taps) > 10 and phone.launches == 0
    assert not list(tmp_path.glob("stuck_tap_*"))
    run(w, clock, 10 + watcher_mod.UNKNOWN_RELAUNCH_S)
    assert phone.launches == 1  # relaunched about 45 s after the taps stopped counting
    assert len(list(tmp_path.glob("stuck_tap_keep_playing_*"))) == 1  # saved once
    run(w, clock, watcher_mod.UNKNOWN_RESTART_S + watcher_mod.UNKNOWN_GIVE_UP_S)
    assert phone.stops == 1 and len(fatal) == 1  # still there: restart, then give up
    assert len(list(tmp_path.glob("stuck_tap_*"))) == 1


def test_the_board_between_taps_is_progress(clock, tmp_path):
    """Normal play taps one button for up to ~40 s (Next Level, 20 times) and the board
    comes back: tapping on and off like that for minutes never escalates."""
    shots, seq = [], 1
    for _ in range(6):
        shots += [shot_of("keep_playing", seq + i) for i in range(40)]
        shots += [shot_of("board_readable", seq + 40 + i) for i in range(5)]
        seq += 45
    w, phone = watcher_on(shots, tmp_path)
    run(w, clock, 6 * 45)
    assert len(phone.taps) > 60
    assert phone.launches == 0 and w._esc_step == 0
    assert not list(tmp_path.glob("stuck_tap_*"))


def test_idle_resets_the_unknown_clock(clock, tmp_path):
    """A 30 min break read as "board hidden 1806s" and would count toward giving up."""
    blank = np.full((1334, 750, 3), (40, 30, 20), np.uint8)
    w, _ = watcher_on([Shot(1, blank, CALIB)], tmp_path)
    w.stop_event.wait = lambda timeout=None: None  # run()'s pauses
    run(w, clock, 10)
    clock.t += 1800
    w.idle.set()
    stopper = iter([False, True])
    w.stop_event.is_set = lambda: next(stopper)
    w.run()
    assert clock.t - w.last_known < 1 and w._hidden_since is None


def test_full_screen_entries_are_skipped_while_the_board_is_in_view(clock, tmp_path):
    w, _ = watcher_on([shot_of("board")], tmp_path)
    shot = shot_of("board")
    w.board_visible, w.expected_panel = True, (100, 700, 1000, 1000)
    w._score(shot, clock.t)
    looked = {p.name for p in w._looked}
    skipped = {p.name for p in w.popups if p.off_board}
    assert skipped >= {"level_complete", "ios_home_game_icon", "country_gift", "loading"}
    assert not looked & skipped and len(looked) == len(w.popups) - len(skipped)
    w.board_visible, w.expected_panel = False, None
    clock.t += 10
    w._score(shot, clock.t)
    assert len(w._looked) == len(w.popups)


def test_reward_wheel_zone_goes_when_the_next_board_shows(clock, tmp_path):
    """The level-complete screen's reward wheel is a no-tap zone. On the next board it
    stayed up (the wheel isn't scanned while a board is in view), so swipes on the
    board's bottom rows were refused until the level counted as stuck."""
    board = shot_of("board_readable", 50)
    shots = [shot_of("next_level_get_reward_twister", i) for i in range(1, 6)]
    shots += [shot_of("board_readable", 51 + i) for i in range(20)]
    w, phone = watcher_on(shots, tmp_path)
    run(w, clock, 4)
    assert phone.zones.get("get_reward_wheel") is not None
    run(w, clock, 2)  # the last wheel frame, then the board; the bot hasn't read it yet
    assert w.board_visible
    w.expected_panel = board.panel  # now it has: the level started
    w._zones_due(1600)  # the first swipe, before the next tick
    assert phone.zones.get("get_reward_wheel") is None
    run(w, clock, 6)
    assert phone.zones.get("get_reward_wheel") is None


def test_reward_wheel_zone_stays_while_the_wheel_is_up(clock, tmp_path):
    """Swipes still in flight when the level ends must not land on the ad wheel."""
    w, phone = watcher_on(
        [shot_of("next_level_get_reward_twister", i) for i in range(1, 20)], tmp_path
    )
    w.expected_panel = shot_of("board_readable").panel  # the bot is mid-level
    run(w, clock, 4)
    w._zones_due(1600)
    assert phone.zones.get("get_reward_wheel") is not None
