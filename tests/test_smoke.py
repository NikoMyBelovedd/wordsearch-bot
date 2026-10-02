"""Device-free checks: everything here runs on Linux, macOS and Windows CI."""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np

from wsbot.imgio import imread, imwrite
from wsbot.letters import LetterReader, _score, normalize
from wsbot.solver import Dictionary
from wsbot.watcher import SCALE, coarse_frame, load_popups, match

ROOT = Path(__file__).resolve().parents[1]


def test_solver_finds_words_in_all_directions():
    words = Dictionary(ROOT / "data" / "words.txt")
    grid = ["CATQ", "QQQQ", "GODQ", "QQQQ"]
    hits = {(h.word, h.start, h.end) for h in words.solve(grid)}
    assert ("CAT", (0, 0), (0, 2)) in hits  # left to right
    assert ("DOG", (2, 2), (2, 0)) in hits  # right to left


def test_imgio_roundtrip_non_ascii_path(tmp_path: Path):
    img = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    path = tmp_path / "José ünïcode 日本" / "frame.png"
    path.parent.mkdir()
    assert imwrite(path, img)
    assert np.array_equal(imread(path), img)
    assert imread(tmp_path / "missing.png") is None


def test_every_registered_popup_template_loads():
    for folder in (ROOT / "templates", ROOT / "templates" / "ios"):
        entries = json.loads((folder / "popups.json").read_text(encoding="utf-8"))
        assert len(load_popups(folder)) == len(entries), folder


def test_reference_letters_read_themselves(tmp_path: Path):
    ref = ROOT / "templates" / "letters" / "ref"
    (tmp_path / "ref").mkdir()
    for p in ref.glob("*.png"):
        (tmp_path / "ref" / p.name).write_bytes(p.read_bytes())
    reader = LetterReader(tmp_path)
    letters = sorted(p.stem.split("_")[0] for p in ref.glob("*.png"))
    assert len(set(letters)) == 26
    for letter in set(letters):
        glyph = imread(ref / f"{letter}.png", cv2.IMREAD_GRAYSCALE)
        if glyph is None:
            continue
        assert reader.ranked(normalize(glyph))[0][1] == letter


def test_spread_keeps_every_swipe_and_separates_overlaps():
    from wsbot.bot import all_lines, path_cells, spread

    grid = ["EKIBROTOM", "UNMTRAMHZ", "PVJTRUCKN"]
    lines = all_lines(grid)
    out = spread(lines)
    assert sorted(out) == sorted(lines)
    first = [h.word for h in out[:4]]
    assert not ({"EKIBROTOM", "MOTORBIKE"} <= set(first[:2]))
    pairs = list(pairwise(out))
    clashes = sum(bool(set(path_cells(a)) & set(path_cells(b))) for a, b in pairs)
    assert clashes < len(pairs) // 10


def test_learning_needs_a_whole_lit_run():
    from wsbot.bot import _whole_run
    from wsbot.solver import Hit

    lit = {(0, c) for c in range(3, 8)}  # LMOAR lit, MOAR is a piece of it
    assert not _whole_run(Hit(0, "MOAR", (0, 4), (0, 7)), lit)
    assert _whole_run(Hit(0, "LMOAR", (0, 3), (0, 7)), lit)
    assert _whole_run(Hit(0, "RAOML", (0, 7), (0, 3)), lit)


def test_learning_keeps_the_longest_and_skips_ambiguous_lines():
    from wsbot.bot import _drop_ambiguous
    from wsbot.solver import Hit

    netball = Hit(0, "NETBALL", (4, 0), (4, 6))
    llabten = Hit(0, "LLABTEN", (4, 6), (4, 0))
    netbal = Hit(0, "NETBAL", (4, 0), (4, 5))
    assert _drop_ambiguous([netbal, netball, llabten]) == [netball, llabten]
    itind, tindy = Hit(0, "ITIND", (0, 0), (4, 4)), Hit(0, "TINDY", (1, 1), (5, 5))
    assert _drop_ambiguous([itind, tindy]) == []


def test_coarse_popup_match_finds_the_same_spot_and_score():
    """match() with a quarter-res first look must agree with the full half-res search."""
    rng = np.random.default_rng(7)
    for p in load_popups(ROOT / "templates" / "ios"):
        frame = rng.integers(0, 255, (2305 // 2, 1296 // 2, 3), dtype=np.uint8)
        frame = cv2.GaussianBlur(frame, (9, 9), 0)  # game screens are smooth, not noise
        th, tw = p.template.shape[:2]
        y, x = 400, 600 - tw
        frame[y : y + th, x : x + tw] = p.template
        full = match(frame, p)
        fast = match(frame, p, coarse_frame(frame, p.coarse_look))
        assert full[0] > 0.99, p.name
        assert fast[1] == full[1], p.name
        assert abs(fast[0] - full[0]) < 1e-4, p.name
        assert fast[1] == (round((x + tw / 2) / SCALE), round((y + th / 2) / SCALE))


def test_letter_scores_match_one_template_at_a_time(tmp_path: Path):
    """ranked() scores every template in one matrix product; it must equal matchTemplate."""
    reader = LetterReader(ROOT / "templates" / "letters")
    rng = np.random.default_rng(3)
    for letter, tmpl in reader.templates[::40]:
        noisy = np.clip(tmpl.astype(int) + rng.integers(-40, 40, tmpl.shape), 0, 255)
        glyph = noisy.astype(np.uint8)
        best: dict[str, float] = {}
        for other, t in reader.templates:
            best[other] = max(best.get(other, -1.0), _score(glyph, t))
        fast = dict((name, s) for s, name in reader.ranked(glyph))
        assert fast.keys() == best.keys()
        assert all(abs(fast[k] - best[k]) < 1e-5 for k in best), letter
        assert reader.ranked(glyph)[0][1] == max(best, key=best.get)
    assert reader.ranked(np.zeros((64, 64), np.uint8))[0][0] == 0.0  # blank cell
