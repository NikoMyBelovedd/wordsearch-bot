"""Device-free checks: everything here runs on Linux, macOS and Windows CI."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from wsbot.imgio import imread, imwrite
from wsbot.letters import LetterReader, normalize
from wsbot.solver import Dictionary
from wsbot.watcher import load_popups

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
