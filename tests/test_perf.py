"""The low-CPU frame pipeline must decide exactly what the old one did, only cheaper."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from wsbot import watcher as watcher_mod
from wsbot.board import find_panel, read_board
from wsbot.bot import DUMP_EVERY_S, Bot
from wsbot.shot import Shot
from wsbot.watcher import PopupWatcher

ROOT = Path(__file__).resolve().parents[1]
CALIB = (1296, 2305)  # iPhone SE calibration space
NATIVE = (750, 1334)


def board_frame(rows: int = 6, cols: int = 6) -> np.ndarray:
    """A calibration-space screen: a landscape, a white board panel, black letters."""
    img = np.zeros((CALIB[1], CALIB[0], 3), np.uint8)
    img[:] = (90, 140, 60)
    x0, y0, size = 120, 700, 1050
    img[y0 : y0 + size, x0 : x0 + size] = 250
    pitch = size / cols
    for r in range(rows):
        for c in range(cols):
            letter = "ABCDEFGHJKLMNOPRSTUVWXYZ"[(r * cols + c) % 24]
            cx, cy = x0 + (c + 0.5) * pitch, y0 + (r + 0.5) * pitch
            cv2.putText(
                img, letter, (int(cx - 30), int(cy + 30)), cv2.FONT_HERSHEY_SIMPLEX, 2.4, 0, 9
            )
    return img


def native_of(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, NATIVE, interpolation=cv2.INTER_AREA)


def test_shot_board_and_panel_match_the_full_size_read():
    calib = cv2.resize(native_of(board_frame()), CALIB, interpolation=cv2.INTER_LINEAR)
    old = read_board(calib)
    assert old is not None
    shot = Shot(1, native_of(board_frame()), CALIB)
    assert shot._calib is None  # nothing made yet
    assert all(abs(a - b) <= 6 for a, b in zip(shot.panel, find_panel(calib), strict=True))
    assert shot._calib is None  # the panel check alone never makes the big copy
    new = shot.board
    assert new is not None
    assert (new.panel, new.rows, new.cols) == (old.panel, old.rows, old.cols)
    assert [c.center for c in new.cells] == [c.center for c in old.cells]
    assert shot.board is new  # read once per picture


def test_shot_without_a_panel_never_reads_the_board():
    blank = np.full((NATIVE[1], NATIVE[0], 3), (90, 140, 60), np.uint8)
    shot = Shot(1, blank, CALIB)
    assert shot.panel is None and shot.board is None
    assert shot._calib is None


def test_small_is_the_old_half_size_frame():
    native = native_of(board_frame())
    calib = cv2.resize(native, CALIB, interpolation=cv2.INTER_LINEAR)
    old = cv2.resize(calib, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    assert np.array_equal(Shot(1, native, CALIB).small, old)


def test_same_picture():
    native = native_of(board_frame())
    a = Shot(1, native, CALIB)
    assert a.same_picture(Shot(1, native, CALIB))
    assert a.same_picture(Shot(2, native.copy(), CALIB))  # Android: new number, same pixels
    other = native.copy()
    other[600:700, 100:300] = 0
    assert not a.same_picture(Shot(3, other, CALIB))
    assert not a.same_picture(None)


class FakeDevice:
    calib = CALIB

    def __init__(self, shots: list[Shot]) -> None:
        self.shots = shots
        self.taps: list[tuple[int, int]] = []

    def shot(self) -> Shot:
        return self.shots.pop(0) if len(self.shots) > 1 else self.shots[0]

    def tap(self, x, y, why="", allow=None) -> None:
        self.taps.append((x, y))

    def set_dynamic_zone(self, name, zone) -> None:
        pass

    def foreground(self) -> str:
        return "pkg"


@pytest.fixture
def counted(monkeypatch):
    calls: list[str] = []
    real = watcher_mod.match

    def match(small, popup, coarse=None):
        calls.append(popup.name)
        return real(small, popup, coarse)

    monkeypatch.setattr(watcher_mod, "match", match)
    return calls


def make_watcher(shots: list[Shot], tmp_path: Path) -> PopupWatcher:
    w = PopupWatcher(FakeDevice(shots), ROOT / "templates" / "ios", "pkg", tmp_path)
    w._last_app_check = float("inf")  # no foreground checks in tests
    return w


def test_an_unchanged_picture_is_not_matched_again(counted, tmp_path):
    native = native_of(board_frame())
    w = make_watcher([Shot(1, native, CALIB)], tmp_path)
    w._tick()
    first = len(counted)
    assert first == len(w.popups)
    for _ in range(5):
        w._tick()
    assert len(counted) == first  # same picture: scores reused, nothing matched
    assert w.frame_time > 0 and w.latest(timeout=0) is not None


def test_board_in_view_checks_the_full_list_once_a_second(counted, tmp_path, monkeypatch):
    frames = []
    for i in range(6):
        img = board_frame()
        img[700 + 1060 + 20 :, :] = (20 + 30 * i, 140, 60)  # landscape changes, board stays
        frames.append(Shot(i, native_of(img), CALIB))
    w = make_watcher(frames, tmp_path)
    w.expected_panel = frames[0].board.panel
    now = [100.0]
    monkeypatch.setattr(watcher_mod.time, "monotonic", lambda: now[0])
    for _ in range(6):  # 5 frames a second: 0.0 .. 1.0 s
        w._tick()
        now[0] += 0.2
    assert w.board_visible
    # The eye watches the toast on iPhones; the fake has no peek, so the toast stays in
    # the watcher's list: full list at the first frame, only the toast until 1 s passed.
    toast = [p.name for p in w.popups if p.covers]
    assert counted.count("next_level") == 2
    assert counted.count(toast[0]) == 6


def test_dumps_of_a_kind_are_rate_limited(tmp_path, monkeypatch):
    saved = []
    fake = SimpleNamespace(
        _dumped={},
        watcher=SimpleNamespace(
            frame=np.zeros((4, 4, 3), np.uint8), hits={}, board_visible=False, fps=0
        ),
        diagnostics=tmp_path,
    )
    monkeypatch.setattr("wsbot.bot.imwrite", lambda path, img: saved.append(path.name))
    now = [1000.0]
    monkeypatch.setattr("wsbot.bot.time.monotonic", lambda: now[0])
    for r in range(5):
        Bot._dump(fake, f"unreadable_r{r}c0")
    Bot._dump(fake, "no_board")
    assert len(saved) == 2
    now[0] += DUMP_EVERY_S
    Bot._dump(fake, "unreadable_r0c1")
    assert len(saved) == 3


def test_packed_dictionary_keeps_the_old_answers(tmp_path):
    from wsbot.solver import Dictionary

    words = tmp_path / "words.txt"
    words.write_text("cat\ncar\ncart\nat\ncat\ndog\ndo-g\nzebra\n", encoding="utf-8")
    d = Dictionary(words)
    assert d.rank["CAT"] == 0  # first line of a duplicate wins
    assert d.rank["CART"] == 2 and d.rank.get("DOG") == 5
    assert "AT" not in d.rank and "DO-G" not in d.rank  # too short / not letters
    assert d.has_longer("CA") and d.has_longer("CAR") and not d.has_longer("CART")
    assert not d.has_longer("ZEBRA") and not d.has_longer("Q")
    d.add("ZEBRAS", 99)  # a learned word: found, and its prefixes lead on
    assert d.rank["ZEBRAS"] == 99 and d.has_longer("ZEBRA")
    d.add("CAT", 7)  # learning a known word re-ranks it, as the dict did
    assert d.rank["CAT"] == 7
    hits = d.solve(["CART", "XXXX", "XXXX"])
    assert [(h.word, h.start, h.end) for h in hits] == [
        ("CAR", (0, 0), (0, 2)),
        ("CART", (0, 0), (0, 3)),
    ]


def test_without_the_board_the_full_list_is_checked_every_scan_period(
    counted, tmp_path, monkeypatch
):
    frames = []
    for i in range(6):
        img = np.full((CALIB[1], CALIB[0], 3), (20 + 30 * i, 140, 60), np.uint8)
        frames.append(Shot(i, native_of(img), CALIB))
    w = make_watcher(frames, tmp_path)
    now = [100.0]
    monkeypatch.setattr(watcher_mod.time, "monotonic", lambda: now[0])
    for _ in range(6):  # 0.0 .. 1.0 s: full looks at 0.0, 0.4, 0.8
        w._tick()
        now[0] += 0.2
    assert not w.board_visible
    assert counted.count("next_level") == 3
