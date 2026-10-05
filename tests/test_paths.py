"""Which path of a word gets swiped: a short word often also lies inside a longer one.

Level 4999 (local/wsbot.log, 2026-09-30): TAB lies at 2,4-2,6 inside BATTER (row 2 read
backwards) and, the real one, at 6,3-4,3. The fast pass fired only the first.
"""

from __future__ import annotations

import json
from pathlib import Path

from wsbot.bot import (
    RETRY_RANK,
    Bot,
    _open,
    all_lines,
    other_paths,
    path_cells,
    pick_paths,
)
from wsbot.solver import Dictionary, Hit

ROOT = Path(__file__).resolve().parents[1]
GRID = "ELZZIRD/ELDDIRG/ZRETTAB/SKFLIPG/EICBXOR/RVOAKUU/VSOTTRR/EPKQKSG".split("/")
REAL_TAB = ((6, 3), (4, 3))
BATTER_TAB = ((2, 4), (2, 6))


def _hits() -> list[Hit]:
    return Dictionary(ROOT / "data" / "words.txt").solve(GRID)


def _path(h: Hit) -> tuple[tuple[int, int], tuple[int, int]]:
    return h.start, h.end


def _batter(hits: list[Hit]) -> set[tuple[int, int]]:
    return set(path_cells(next(h for h in hits if h.word == "BATTER")))


def test_the_grid_has_both_tabs():
    tabs = {_path(h) for h in _hits() if h.word == "TAB"}
    assert tabs == {REAL_TAB, BATTER_TAB}


def test_picker_skips_the_tab_inside_batter():
    picked = pick_paths(_hits())
    assert [_path(h) for h in picked if h.word == "TAB"] == [REAL_TAB]
    assert len({h.word for h in picked}) == len(picked)  # one path per word
    assert [h.rank for h in picked] == sorted(h.rank for h in picked)  # still rank order


def test_picker_prefers_a_path_off_lit_cells():
    hits = [Hit(5, "CAT", (0, 0), (0, 2)), Hit(5, "CAT", (2, 0), (2, 2))]
    assert pick_paths(hits)[0] == hits[0]  # all else equal: grid order
    assert pick_paths(hits, {(0, 1)})[0] == hits[1]


def test_paths_pass_gets_the_real_tab_once_batter_is_lit():
    hits = _hits()
    seen: set[str] = set()
    top_left = [h for h in hits if not (h.word in seen or seen.add(h.word))]  # the old pick
    alt = other_paths(hits, top_left, _batter(hits))
    assert REAL_TAB in {_path(h) for h in alt if h.word == "TAB"}
    assert all(len(h.word) <= 4 for h in alt)


def test_paths_pass_skips_lit_paths_and_words_found_where_fired():
    hits = _hits()
    picked = pick_paths(hits)
    lit = _batter(hits)
    alt = other_paths(hits, picked, lit)
    assert BATTER_TAB not in {_path(h) for h in alt if h.word == "TAB"}  # all lit
    real = set(path_cells(Hit(0, "TAB", *REAL_TAB)))
    assert not [h for h in other_paths(hits, picked, lit | real) if h.word == "TAB"]


def _bot(swiped_paths: set) -> Bot:
    bot = Bot.__new__(Bot)
    bot.swiped_paths = swiped_paths
    return bot


def test_exhaustive_pass_goes_by_path_not_word():
    bot = _bot({("TAB", *BATTER_TAB)})
    todo = {(h.word, *_path(h)) for h in bot._new_paths(all_lines(GRID))}
    assert ("TAB", *REAL_TAB) in todo
    assert ("TAB", *BATTER_TAB) not in todo


def test_open_takes_a_common_three_letter_word_while_all_unlit():
    tab = Hit(3_650, "TAB", *REAL_TAB)
    assert tab.rank < RETRY_RANK
    assert _open(tab, set())
    assert not _open(tab, {(5, 3)})
    assert not _open(Hit(RETRY_RANK + 1, "TAB", *REAL_TAB), set())


def test_swiped_state_keeps_paths_and_loads_old_files(tmp_path: Path):
    bot = Bot.__new__(Bot)
    bot._swiped_path = tmp_path / "level_swiped.json"
    bot._swiped_path.write_text(json.dumps({"grid": GRID, "words": ["TAB"]}))
    assert bot._load_swiped() == ({"TAB"}, set(), GRID)  # before paths were kept

    bot.swiped, bot.swiped_paths, bot._swiped_grid = {"TAB"}, {("TAB", *BATTER_TAB)}, GRID
    bot._save_swiped()
    assert bot._load_swiped() == ({"TAB"}, {("TAB", *BATTER_TAB)}, GRID)
