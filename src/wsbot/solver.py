"""Brute-force every dictionary word on the grid, in wordlist (frequency) order.

words.txt is frequency-ordered (common words first, Scrabble oddities last), so
sorting hits by line number means real puzzle words get swiped early.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MIN_LEN = 3
DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1), (0, -1), (-1, 0), (-1, -1), (-1, 1)]


@dataclass(frozen=True, order=True)
class Hit:
    rank: int
    word: str
    start: tuple[int, int]  # (row, col)
    end: tuple[int, int]


class Dictionary:
    def __init__(self, path: Path) -> None:
        self.rank: dict[str, int] = {}
        self.prefixes: set[str] = set()
        for i, line in enumerate(path.read_text(encoding="utf-8").split()):
            w = line.strip().upper()
            if len(w) < MIN_LEN or not w.isalpha() or w in self.rank:
                continue
            self.rank[w] = i
            for k in range(1, len(w)):
                self.prefixes.add(w[:k])

    def solve(self, grid: list[str]) -> list[Hit]:
        """Every dictionary word along straight lines, best-ranked first.

        A word that appears on several paths yields a hit per path: a misread letter can
        create a phantom path, and firing only one could miss the real word.
        """
        rows, cols = len(grid), len(grid[0])
        hits: list[Hit] = []
        for r in range(rows):
            for c in range(cols):
                for dr, dc in DIRECTIONS:
                    word = ""
                    rr, cc = r, c
                    while 0 <= rr < rows and 0 <= cc < cols:
                        word += grid[rr][cc]
                        if len(word) >= MIN_LEN and word in self.rank:
                            hits.append(Hit(self.rank[word], word, (r, c), (rr, cc)))
                        if word not in self.prefixes:
                            break
                        rr, cc = rr + dr, cc + dc
        return sorted(hits)
