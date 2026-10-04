"""Faded letters: some levels grey out the filler letters that are in no word
(level 9345, 2026-10-03). The bot read such a board as "no board" and looped through
taps and game restarts until it gave up."""

from __future__ import annotations

from pathlib import Path

from wsbot.board import read_board
from wsbot.bot import FADED, all_lines
from wsbot.imgio import imread

SCREENS = Path(__file__).resolve().parent / "data" / "screens"

# Level 9345: '.' = faded filler, every other cell is a letter of one of its words.
FADED_LAYOUT = [
    "#######",
    "#..##.#",
    "#.#####",
    "#######",
    "#######",
    ".######",
    ".##...#",
    ".###...",
]


def test_a_board_with_faded_letters_is_still_a_board():
    board = read_board(imread(SCREENS / "board_faded.jpg"))
    assert board is not None
    assert (board.rows, board.cols) == (8, 7)
    layout = [
        "".join("." if board.cell(r, c).faded else "#" for c in range(board.cols))
        for r in range(board.rows)
    ]
    assert layout == FADED_LAYOUT


def test_normal_boards_have_no_faded_cells():
    board = read_board(imread(SCREENS / "board_readable.jpg"))
    assert board is not None
    assert not any(c.faded for c in board.cells)


def test_screens_that_are_not_a_board_still_are_not():
    for name in ("toast_over_board.jpg", "board_letters_flying.jpg"):
        assert read_board(imread(SCREENS / name)) is None, name


def test_no_line_crosses_a_faded_cell():
    grid = ["ABCD", f"EF{FADED}H", "IJKL"]
    lines = all_lines(grid)
    assert lines
    assert all(FADED not in hit.word for hit in lines)
    assert any(hit.word == "ABCD" for hit in lines)
    # Row 1 is split by the faded cell: no 3-letter line fits in "EF" or "H".
    assert not any(hit.start[0] == hit.end[0] == 1 for hit in lines)
