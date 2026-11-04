"""A hint's red pill over a letter (Level 121, Word Search 1.2.5, Pixel 6a 1080x2400).

The hint H is white on red, so the dark-glyph mask never saw it: the grid came out one
letter short and the board was "not visible" for minutes on end.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from wsbot.board import read_board
from wsbot.imgio import imread
from wsbot.letters import LetterReader, normalize

ROOT = Path(__file__).resolve().parents[1]
HINT_FRAME = ROOT / "tests" / "data" / "screens" / "board_hint_red.png"

# Whole board as shown on the frame (the H is the red hint at row 6, col 5).
BOARD = [
    "WRELUR",
    "RCSAWR",
    "ELKXEE",
    "NQIFFM",
    "CVIABM",
    "HNZJNA",
    "KTLOBH",
]


@pytest.fixture(scope="module")
def reader() -> LetterReader:
    return LetterReader(ROOT / "templates" / "letters")


def _letter(reader: LetterReader, board, r: int, c: int) -> str:
    return reader.ranked(normalize(board.cell(r, c).glyph))[0][1]


# The native frame, and the same frame at two other phone sizes (uniform resize).
SIZES = [(1080, 2400), (720, 1600), (1440, 3200)]


def _frame_at(size: tuple[int, int]):
    frame = imread(HINT_FRAME)
    if size == (frame.shape[1], frame.shape[0]):
        return frame
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


@pytest.mark.parametrize("size", SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_the_hint_pill_is_one_of_the_grid_cells(size: tuple[int, int]):
    board = read_board(_frame_at(size))
    assert board is not None
    assert (board.rows, board.cols) == (7, 6)
    assert len(board.cells) == 42


@pytest.mark.parametrize("size", SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_the_hint_cell_reads_its_white_letter(reader: LetterReader, size: tuple[int, int]):
    board = read_board(_frame_at(size))
    assert board is not None
    for r, c in [(0, 0), (2, 4), (3, 0), (6, 0), (6, 5)]:
        assert _letter(reader, board, r, c) == BOARD[r][c], (r, c)


@pytest.mark.parametrize("size", SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_every_size_reads_the_same_board(reader: LetterReader, size: tuple[int, int]):
    board = read_board(_frame_at(size))
    assert board is not None
    rows = [
        "".join(_letter(reader, board, r, c) for c in range(board.cols)) for r in range(board.rows)
    ]
    assert rows == BOARD


def test_a_dark_stray_off_the_lattice_is_not_a_cell():
    frame = imread(HINT_FRAME)
    # A 40 px black square between columns 0 and 1, rows 0 and 1: on no row or column.
    cv2.rectangle(frame, (190, 940), (230, 980), (0, 0, 0), -1)
    board = read_board(frame)
    assert board is not None
    assert (board.rows, board.cols) == (7, 6)


def test_an_empty_plain_slot_still_means_no_board():
    frame = imread(HINT_FRAME)
    # Erase the letter at row 2, col 2 to plain white: no pill, so no board.
    cx, cy = 459, 1208
    cv2.rectangle(frame, (cx - 40, cy - 40), (cx + 40, cy + 40), (255, 255, 255), -1)
    assert read_board(frame) is None
