"""When a level counts as done (bot._wait_level_end): only on the watcher's level-end
screen (level_done) or a different board read identically twice. A hidden board is not
the end. A "Daily Wisdom" card covered the board with no template, the hidden board read
as the level's end after 3 s, and the bot moved on from a level it hadn't finished.

Fixtures: daily_wisdom.jpg (the card over the board: no board read), level_board.jpg (a
plain level: its grid reads from the frame)."""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import cv2
import pytest

from wsbot import bot as bot_mod
from wsbot.shot import Shot
from wsbot.watcher import PopupWatcher  # noqa: F401  (the real type the bot's watcher is)

ROOT = Path(__file__).resolve().parents[1]
ANDROID = ROOT / "tests" / "data" / "android"
CALIB = (1080, 2400)
GRID = ["ABCDEF"] * 7  # the level the bot is solving
OTHER = ["QWERTY"] * 7  # the next level, read from the screen


def frame(name: str) -> Shot:
    img = cv2.imread(str(ANDROID / name))
    assert img is not None, name
    return Shot(1, img, CALIB)


class FakeClock:
    """time.monotonic that moves 1 s per call: a 20 s wait takes ~20 looks."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        self.now += 1.0
        return self.now


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(bot_mod, "time", fake)
    return fake


class FakeWatcher:
    def __init__(self, shots: list[Shot]) -> None:
        self.level_done = threading.Event()
        self.frame_time = 0.0
        self._shots = shots
        self._i = 0

    def latest_shot(self, newer_than: float = 0.0, timeout: float = 3.0):
        shot = self._shots[min(self._i, len(self._shots) - 1)]  # the last one repeats
        self._i += 1
        self.frame_time = float(self._i)
        return shot

    def covering(self):
        return None

    mid_level_seen = 0.0


def wait(shots, reads=None, *, timeout=20.0, level_done=False, grid=GRID):
    """_wait_level_end on a stand-in bot: the frames, and the grid each board reads as."""
    watcher = FakeWatcher(shots)
    if level_done:
        watcher.level_done.set()
    fake = SimpleNamespace(
        stop_event=threading.Event(),
        watcher=watcher,
        read_grid=lambda board: next(reads) if reads is not None else GRID,
    )
    return bot_mod.Bot._wait_level_end(fake, grid, timeout)


def test_a_daily_wisdom_card_over_the_board_is_not_the_level_end():
    assert wait([frame("daily_wisdom.jpg")]) is False  # hidden for the whole 20 s


def test_a_hidden_board_is_not_the_end_even_when_the_card_closes_on_the_same_level():
    frames = [frame("daily_wisdom.jpg")] * 5 + [frame("level_board.jpg")]
    assert wait(frames, reads=iter([GRID] * 50)) is False


def test_the_watchers_level_end_screen_ends_the_level():
    assert wait([frame("daily_wisdom.jpg")], level_done=True) is True


def test_a_new_board_read_twice_ends_the_level():
    assert wait([frame("level_board.jpg")], reads=iter([OTHER] * 5)) is True


def test_one_odd_read_is_not_a_new_level():
    frames = [frame("level_board.jpg"), frame("daily_wisdom.jpg")]
    assert wait(frames, reads=iter([OTHER] * 5)) is False


def test_the_same_board_again_is_not_a_new_level():
    assert wait([frame("level_board.jpg")], reads=iter([GRID] * 50)) is False
