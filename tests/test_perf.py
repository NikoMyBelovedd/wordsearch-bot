"""The low-CPU frame pipeline must decide exactly what the old one did, only cheaper."""

from __future__ import annotations

import threading
import time
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

    def match(small, popup, *args):
        calls.append(popup.name)
        return real(small, popup, *args)

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


def test_board_in_view_checks_the_full_list_every_two_seconds(counted, tmp_path, monkeypatch):
    frames = []
    for i in range(11):
        img = board_frame()
        img[700 + 1060 + 20 :, :] = (20 + 20 * i, 140, 60)  # landscape changes, board stays
        frames.append(Shot(i, native_of(img), CALIB))
    w = make_watcher(frames, tmp_path)
    w.expected_panel = frames[0].board.panel
    now = [100.0]
    monkeypatch.setattr(watcher_mod.time, "monotonic", lambda: now[0])
    for _ in range(11):  # 5 frames a second: 0.0 .. 2.0 s
        w._tick()
        now[0] += 0.2
    assert w.board_visible
    # The eye watches the toast on iPhones; the fake has no peek, so the toast stays in
    # the watcher's list: full list at the first frame, only the toast until 1 s passed.
    toast = [p.name for p in w.popups if p.covers]
    assert counted.count("next_level") == 2
    assert counted.count(toast[0]) == 11


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
    for _ in range(6):  # 0.0 .. 1.0 s: full looks at 0.0 and 0.8
        w._tick()
        now[0] += 0.2
    assert not w.board_visible
    assert counted.count("next_level") == 2


def test_resting_scans_rarely(counted, tmp_path, monkeypatch):
    frames = []
    for i in range(16):
        img = np.full((CALIB[1], CALIB[0], 3), (20 + 10 * i, 140, 60), np.uint8)
        frames.append(Shot(i, native_of(img), CALIB))
    w = make_watcher(frames, tmp_path)
    w.resting.set()
    now = [100.0]
    monkeypatch.setattr(watcher_mod.time, "monotonic", lambda: now[0])
    for _ in range(16):  # 1 look a second while resting: full lists at 0, 3, 6 ... 15 s
        w._tick()
        now[0] += 1.0
    assert counted.count("next_level") == 6


def test_idle_watcher_quiets_the_stream_and_wakes_it(tmp_path, monkeypatch):
    native = native_of(board_frame())
    w = make_watcher([Shot(1, native, CALIB)], tmp_path)
    calls: list[bool] = []
    w.device.set_quiet = calls.append
    w._eye_on = False
    monkeypatch.setattr(watcher_mod, "MAX_FPS", 50.0)
    w.idle.set()
    w.start()
    deadline = time.monotonic() + 3
    while not calls and time.monotonic() < deadline:
        time.sleep(0.02)
    w.idle.clear()
    while len(calls) < 2 and time.monotonic() < deadline + 3:
        time.sleep(0.02)
    w.stop()
    assert calls[:2] == [True, False]


def test_deferred_module_loads_on_first_use(monkeypatch):
    import sys

    from wsbot.slim import defer

    monkeypatch.delitem(sys.modules, "json.tool", raising=False)
    defer("json.tool")
    stub = sys.modules["json.tool"]
    assert stub.Question.__name__ == "Question"  # annotation-only name, no load
    assert callable(stub.main)  # first real use loads the real module
    assert sys.modules["json.tool"] is not stub


def test_toast_eye_finds_the_toast_in_the_watchers_frames(tmp_path):
    from wsbot.imgio import imread

    img = np.full((CALIB[1], CALIB[0], 3), (90, 140, 60), np.uint8)
    toast = imread(ROOT / "templates" / "ios" / "popups" / "already_collected.png")
    th, tw = toast.shape[:2]
    x, y = 300, 1600  # inside EYE_BOX on an SE
    img[y : y + th, x : x + tw] = toast
    w = make_watcher([Shot(1, native_of(img), CALIB)], tmp_path)
    w.device.peek = lambda *a: None  # an iPhone: the eye watches the toast
    w2 = PopupWatcher(w.device, ROOT / "templates" / "ios", "pkg", tmp_path)
    assert w2._eye_on
    w2.shot, w2._shot_t = Shot(1, native_of(img), CALIB), time.monotonic()
    t = threading.Thread(target=w2._watch_cover, daemon=True)
    t.start()
    deadline = time.monotonic() + 3
    while not w2.cover_seen and time.monotonic() < deadline:
        time.sleep(0.02)
    w2.stop()
    assert w2.cover_seen
    lo, hi = w2.cover_span
    assert lo < y + th // 2 < hi


def test_a_popup_that_goes_still_still_gets_a_full_look(counted, tmp_path, monkeypatch):
    """The last new frame got only a partial look; the picture then stays put."""
    a = np.full((CALIB[1], CALIB[0], 3), (20, 140, 60), np.uint8)
    b = a.copy()
    b[100:300] = 200
    frames = [Shot(1, native_of(a), CALIB), Shot(2, native_of(b), CALIB)]
    w = make_watcher(frames, tmp_path)
    now = [100.0]
    monkeypatch.setattr(watcher_mod.time, "monotonic", lambda: now[0])
    w._tick()  # full look at frame 1
    now[0] += 0.2
    w._tick()  # frame 2 is new but no full look is due: partial
    assert counted.count("next_level") == 1
    for _ in range(5):  # frame 2 stays on screen
        now[0] += 0.2
        w._tick()
    assert counted.count("next_level") == 2  # looked at in full once it was due
    for _ in range(10):
        now[0] += 0.2
        w._tick()
    assert counted.count("next_level") == 2  # and never again for the same picture


def test_change_scan_finds_what_a_whole_search_finds():
    """Full scans rescore only where the picture changed; the spots and scores stay
    those of a whole-frame search, for a popup that shows, moves and goes."""
    from wsbot.imgio import imread

    rng = np.random.default_rng(3)
    popups = watcher_mod.load_popups(ROOT / "templates" / "ios")
    base = board_frame()
    base[:600] = rng.integers(0, 255, (600, 1, 3), dtype=np.uint8)  # some texture
    scan = watcher_mod.ChangeScan()
    img = base.copy()
    for step in range(10):
        if step in (2, 5):  # a popup shows (dimmed screen), then another spot
            img = (base * 0.5).astype(np.uint8)
            got = imread(ROOT / "templates" / "ios" / "popups" / "got_it.png")
            x, y = (400, 1500) if step == 2 else (200, 300)
            img[y : y + got.shape[0], x : x + got.shape[1]] = got
        elif step == 7:  # gone again
            img = base.copy()
        elif step >= 8:  # a still screen
            pass
        else:  # a found word lights up somewhere, the rest stays
            x, y = rng.integers(100, 900), rng.integers(700, 1300)  # not over the popup
            img[y : y + 90, x : x + 260] = rng.integers(60, 255, 3)
        shot = Shot(step, native_of(img), CALIB)
        scan.begin(shot.coarse_color)
        if step >= 8:
            assert not scan.changed.any()
        elif step not in (0, 2, 5, 7):  # a few cells changed, not the whole screen
            assert scan.changed is not None and 0 < scan.changed.sum() < scan.changed.size / 10
        for p in popups:
            coarse = shot.coarse_as(p.coarse_look)
            whole = watcher_mod.match(shot.small, p, coarse)
            patched = watcher_mod.match(shot.small, p, coarse, scan)
            if step >= 8:  # the same pixels: every score is the exact one, kept
                assert patched == whole
            if whole[0] >= p.threshold or patched[0] >= p.threshold:
                assert patched == whole, (step, p.name)
            if p.name == "got_it":
                assert (whole[0] >= p.threshold) == (step in (2, 3, 4, 5, 6))


def test_change_scan_change_mask():
    scan = watcher_mod.ChangeScan()
    a = np.full((576, 324, 3), 100, np.uint8)
    scan.begin(a)
    assert scan.changed is None  # first picture: everything is new
    b = a.copy()
    b[575, 323, 2] = 100 + watcher_mod.CHANGE_LEVEL + 1  # moved: the last tile
    b[0, 0, 0] = 100 + watcher_mod.CHANGE_LEVEL  # noise
    b[300, 100, 1] = 50
    scan.begin(b)
    assert [tuple(t) for t in np.argwhere(scan.changed)] == [(37, 12), (71, 40)]
    assert scan.ref[0, 0, 0] == 100  # noise isn't followed: a slow drift still adds up
    assert scan.ref[300, 100, 1] == 50
    c = b.copy()
    c[0, 0, 0] = 100 + 2 * watcher_mod.CHANGE_LEVEL  # crept on: now it counts
    scan.begin(c)
    assert [tuple(t) for t in np.argwhere(scan.changed)] == [(0, 0)]


def test_full_scans_rescore_a_popup_pasted_on_a_still_screen(tmp_path, monkeypatch):
    from wsbot.imgio import imread

    img = np.full((CALIB[1], CALIB[0], 3), (90, 140, 60), np.uint8)
    frames = [Shot(1, native_of(img), CALIB)]
    got = imread(ROOT / "templates" / "ios" / "popups" / "got_it.png")
    img[1200 : 1200 + got.shape[0], 500 : 500 + got.shape[1]] = got
    frames.append(Shot(2, native_of(img), CALIB))
    still = frames[1]
    w = make_watcher(frames, tmp_path)
    now = [100.0]
    monkeypatch.setattr(watcher_mod.time, "monotonic", lambda: now[0])
    w._tick()
    assert not w.device.taps
    now[0] += 1.0
    w._tick()
    got_it = next(p for p in w.popups if p.name == "got_it")
    want = watcher_mod.match(still.small, got_it, still.coarse)
    assert w._scores["got_it"] == want
    assert want[0] >= 0.85 and w.device.taps == [want[1]]


def test_change_scan_keeps_scores_only_for_the_same_pixels(monkeypatch):
    popups = watcher_mod.load_popups(ROOT / "templates" / "ios")
    img = board_frame()
    scan = watcher_mod.ChangeScan()
    calls = []
    real = cv2.matchTemplate
    monkeypatch.setattr(cv2, "matchTemplate", lambda *a: calls.append(1) or real(*a))

    def full_scan(native):
        shot = Shot(1, native, CALIB)
        scan.begin(shot.coarse_color)
        calls.clear()
        return [
            watcher_mod.match(shot.small, p, shot.coarse_as(p.coarse_look), scan) for p in popups
        ]

    first = full_scan(native_of(img))
    assert len(calls) == 2 * len(popups)  # whole maps + color scores
    assert full_scan(native_of(img)) == first and not calls  # a still screen: all kept
    assert full_scan(native_of(img)) == first and not calls  # and on, while it stays
    noisy = native_of(img).astype(np.int16) + np.random.default_rng(1).integers(
        -2, 3, (1334, 750, 3)
    )
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    shot = Shot(1, noisy, CALIB)
    again = full_scan(noisy)
    # noise: maps kept, but every color score taken again on the new pixels; where
    # nothing matches, the best of a flat map can sit elsewhere, never a decision
    assert len(calls) == len(popups)
    for p, (score, center) in zip(popups, again, strict=True):
        whole = watcher_mod.match(shot.small, p, shot.coarse_as(p.coarse_look))
        assert (score >= p.threshold) == (whole[0] >= p.threshold)
        if score >= p.threshold:
            assert (score, center) == whole


def test_component_stats_are_opencvs():
    """board._largest / _blobs: connectedComponentsWithStats' answers for less work."""
    from wsbot.board import _blobs, _kernel, _largest

    frames = [board_frame(), board_frame(9, 8)]
    frames += [
        cv2.resize(cv2.imread(str(p)), CALIB, interpolation=cv2.INTER_LINEAR)
        for p in sorted((ROOT / "tests" / "data" / "screens").glob("*.jpg"))
        if cv2.imread(str(p)).shape[1] == NATIVE[0]
    ]
    for img in frames:
        mini = cv2.resize(native_of(img), (NATIVE[0] // 2, NATIVE[1] // 2))
        white = cv2.inRange(mini, (235,) * 3, (255,) * 3)
        white = cv2.morphologyEx(white, cv2.MORPH_OPEN, _kernel(mini.shape[1] / CALIB[0]))
        n, _, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=4)
        if n < 2:
            assert _largest(white) is None
        else:
            i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            assert _largest(white) == tuple(int(v) for v in stats[i])
        dark = cv2.inRange(img, (0, 0, 0), (90, 90, 90))
        n, _, stats, _ = cv2.connectedComponentsWithStats(dark)
        assert _blobs(dark) == [tuple(int(v) for v in s) for s in stats[1:n]]
    empty = np.zeros((40, 30), np.uint8)
    assert _largest(empty) is None and _blobs(empty) == []


def test_letter_scores_are_remembered_per_glyph_until_a_letter_is_learned(tmp_path: Path):
    import shutil

    from wsbot.letters import LetterReader

    folder = tmp_path / "letters"
    shutil.copytree(ROOT / "templates" / "letters" / "ref", folder / "ref")
    reader = LetterReader(folder)
    glyph, other = reader.templates[5][1].copy(), reader.templates[9][1].copy()
    first = reader.ranked(glyph)
    assert first[0][1] == reader.templates[5][0]
    reader._mat = np.zeros_like(reader._mat)  # a recomputed list would now be all zeros
    assert reader.ranked(glyph) == first and reader.ranked(glyph.copy()) == first
    assert reader.ranked(other)[0][0] == 0.0  # not remembered: scored (by the zeros)
    # learning a letter forgets every remembered list (they're one template short)
    reader.learn("Q", glyph)
    assert reader.ranked(glyph)[0] == (pytest.approx(1.0, abs=1e-5), "Q")


def test_toast_eye_reuses_an_unchanged_box(counted, tmp_path):
    """A new picture whose toast box shows the same pixels: the last look's answer."""
    from wsbot.imgio import imread

    img = np.full((CALIB[1], CALIB[0], 3), (90, 140, 60), np.uint8)
    toast = imread(ROOT / "templates" / "ios" / "popups" / "already_collected.png")
    th, tw = toast.shape[:2]
    img[1600 : 1600 + th, 300 : 300 + tw] = toast
    w = make_watcher([Shot(1, native_of(img), CALIB)], tmp_path)
    w.device.peek = lambda *a: None
    w = PopupWatcher(w.device, ROOT / "templates" / "ios", "pkg", tmp_path)
    counted.clear()
    first = w._eye_look(Shot(1, native_of(img), CALIB), 10.0)
    assert counted == ["already_collected"] and w.cover_seen == 10.0
    top_changed = img.copy()
    top_changed[100:300] = 255  # the word list changed, not the toast's box
    again = w._eye_look(Shot(2, native_of(top_changed), CALIB), 10.2)
    assert again == first and counted == ["already_collected"] and w.cover_seen == 10.2
    moved = img.copy()
    moved[1600 : 1600 + th, 300 : 300 + tw] = 255  # the toast went
    assert w._eye_look(Shot(3, native_of(moved), CALIB), 10.4) < 0.85
    assert len(counted) == 2 and w.cover_seen == 10.2


def test_saved_screens_are_pruned_to_the_newest(tmp_path):
    import os

    from wsbot.imgio import MAX_DIAGNOSTICS, prune_pngs

    for i in range(MAX_DIAGNOSTICS + 5):
        p = tmp_path / f"unknown_popup_{i:03d}.png"
        p.write_bytes(b"x")
        os.utime(p, (1000 + i, 1000 + i))
    (tmp_path / "notes.txt").write_text("kept")
    prune_pngs(tmp_path)
    left = sorted(p.name for p in tmp_path.glob("*.png"))
    assert left == [f"unknown_popup_{i:03d}.png" for i in range(5, MAX_DIAGNOSTICS + 5)]
    assert (tmp_path / "notes.txt").exists()


def test_unknown_screen_saves_prune_the_folder(tmp_path, monkeypatch):
    """The watcher's unknown-screen saves keep diagnostics/ at MAX_DIAGNOSTICS too."""
    import os

    from wsbot.imgio import MAX_DIAGNOSTICS

    for i in range(MAX_DIAGNOSTICS + 5):
        p = tmp_path / f"old_{i:03d}.png"
        p.write_bytes(b"x")
        os.utime(p, (1000 + i, 1000 + i))
    clock = [1000.0]
    monkeypatch.setattr(watcher_mod.time, "monotonic", lambda: clock[0])
    img = np.full((CALIB[1], CALIB[0], 3), (30, 60, 200), np.uint8)  # nothing it knows
    w = make_watcher([Shot(1, native_of(img), CALIB)], tmp_path)
    for _ in range(12):
        w._tick()
        clock[0] += 1.0
    pngs = list(tmp_path.glob("*.png"))
    assert len(pngs) == MAX_DIAGNOSTICS
    assert any(p.name.startswith("unknown_popup_") for p in pngs)
    assert not (tmp_path / "old_000.png").exists()


def test_iphone_frames_convert_to_the_same_pixels():
    """The one kept single-thread converter gives what frame.to_ndarray gave: the
    software decoder's frames (yuvj420p) and a GPU decoder's (nv12), frame after frame,
    back and forth between sizes."""
    import av

    from wsbot.iphone import IPhone

    rng = np.random.default_rng(3)
    phone = SimpleNamespace(_uncollapsed=0, _published_logged=5, _bgr=None)
    kept = None
    for w, h in [(752, 1344), (64, 96), (752, 1344)]:
        img = cv2.GaussianBlur(rng.integers(0, 256, (h, w, 3), np.uint8), (0, 0), 2)
        img[h // 3 :, : w // 2] = (30, 200, 90)
        frame = av.VideoFrame.from_ndarray(img, format="bgr24")
        for fmt in ("yuvj420p", "nv12", "yuvj420p"):
            f = frame.reformat(format=fmt)
            phone.width, phone.height = w, h
            assert np.array_equal(IPhone._to_bgr(phone, f), f.to_ndarray(format="bgr24"))
            kept = kept or phone._bgr
            assert phone._bgr is kept


def _uncollapse_full(img: np.ndarray) -> np.ndarray:
    """iphone._uncollapse before its edge shortcut."""
    h, w = img.shape[:2]
    step = 8
    gray = (np.abs(img[::step, ::step].astype(np.int16) - 128) < 6).all(axis=2)
    cols = np.flatnonzero(gray.mean(axis=0) < 0.6)
    rows = np.flatnonzero(gray.mean(axis=1) < 0.6)
    if not len(cols) or not len(rows):
        return img
    cw, ch = (cols[-1] + 1) * step, (rows[-1] + 1) * step
    if 0.2 * w < cw < 0.92 * w and 0.2 * h < ch < 0.92 * h:
        return cv2.resize(img[:ch, :cw], (w, h), interpolation=cv2.INTER_LINEAR)
    return img


def test_uncollapse_shortcut_keeps_every_answer():
    """Collapsed frames (gray padding right and bottom, at many sizes), normal ones, and
    ones gray at only one edge or with gray padding too thin to count: the same picture
    back as before the shortcut."""
    from wsbot.iphone import _uncollapse

    rng = np.random.default_rng(5)
    frames = [
        cv2.imread(str(p)) for p in sorted((ROOT / "tests" / "data" / "screens").glob("*.jpg"))
    ]
    for w, h in [(750, 1334), (1206, 2622), (752, 1344)]:
        content = cv2.resize(frames[0], (w, h))
        for fx, fy in [
            (0.5, 0.5),
            (0.7, 0.85),
            (0.9, 0.95),
            (0.95, 0.9),
            (1.0, 0.6),
            (0.6, 1.0),
            (0.15, 0.5),
        ]:
            img = np.full((h, w, 3), 128, np.uint8)
            cw, ch = int(w * fx), int(h * fy)
            img[:ch, :cw] = cv2.resize(content, (cw, ch))
            frames.append(img)
            noisy = np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
            frames.append(noisy)
        gray_right = content.copy()
        gray_right[:, -40:] = 128
        gray_bottom = content.copy()
        gray_bottom[-60:] = 128
        frames += [content, gray_right, gray_bottom, np.full((h, w, 3), 128, np.uint8)]
    resized = 0
    for img in frames:
        want, got = _uncollapse_full(img), _uncollapse(img)
        assert (got is img) == (want is img)
        assert np.array_equal(got, want)
        resized += want is not img
    assert resized >= 10
