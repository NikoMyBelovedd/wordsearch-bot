"""Brute-force every dictionary word on the grid, in wordlist (frequency) order.

words.txt is frequency-ordered (common words first, SCOWL's rarer levels last), so
sorting hits by line number means real puzzle words get swiped early.
"""

from __future__ import annotations

from array import array
from bisect import bisect_left
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
    """The wordlist: rank (line number) of every word, and which strings start a longer
    word. A farm runs one bot per phone, so it is packed: every word in one sorted
    bytes blob, found by binary search (~4 MB; the dict + prefix set it replaced took
    64 MB per bot). Words added later (learned theme words) live in a small dict."""

    def __init__(self, path: Path) -> None:
        pairs = []
        for i, line in enumerate(path.read_text(encoding="utf-8").split()):
            w = line.strip().upper()
            if len(w) >= MIN_LEN and w.isalpha():
                pairs.append((w.encode(), i))
        pairs.sort()  # by word, then line: the first of duplicates is the best rank
        words, ranks, last = [], array("I"), None
        for w, i in pairs:
            if w != last:
                words.append(w)
                ranks.append(i)
                last = w
        del pairs
        self._blob = b"\n".join(words) + b"\n"
        starts = array("I", [0])
        for w in words:
            starts.append(starts[-1] + len(w) + 1)
        self._starts = starts  # word k is _blob[starts[k] : starts[k + 1] - 1]
        self._ranks = ranks
        self._n = len(words)
        self._extra: dict[str, int] = {}
        self._extra_prefixes: set[str] = set()
        self.rank = _Ranks(self)

    def _word(self, k: int) -> bytes:
        return self._blob[self._starts[k] : self._starts[k + 1] - 1]

    def _find(self, key: bytes) -> int:
        """Index of the first packed word >= key."""
        return bisect_left(range(self._n), key, key=self._word)

    def _base_rank(self, word: str) -> int | None:
        key = word.encode()
        k = self._find(key)
        return self._ranks[k] if k < self._n and self._word(k) == key else None

    def get(self, word: str) -> int | None:
        rank = self._extra.get(word)
        return rank if rank is not None else self._base_rank(word)

    def has_longer(self, prefix: str) -> bool:
        """Some word starts with `prefix` and is longer than it."""
        if prefix in self._extra_prefixes:
            return True
        key = prefix.encode()
        k = self._find(key)
        if k < self._n and self._word(k) == key:
            k += 1
        return k < self._n and self._word(k).startswith(key)

    def _step(self, word: str, lo: int) -> tuple[int | None, bool, int]:
        """(rank of `word` or None, whether a longer word starts with it, where it sorts):
        get() and has_longer() in one search, from `lo` on."""
        key = word.encode()
        k = bisect_left(range(self._n), key, lo, key=self._word)
        rank = None
        nxt = k
        if k < self._n and self._word(k) == key:
            rank = self._ranks[k]
            nxt = k + 1
        longer = nxt < self._n and self._word(nxt).startswith(key)
        if self._extra:
            extra = self._extra.get(word)
            rank = extra if extra is not None else rank
            longer = longer or word in self._extra_prefixes
        return rank, longer, k

    def add(self, word: str, rank: int) -> None:
        self._extra[word] = rank
        for k in range(1, len(word)):
            self._extra_prefixes.add(word[:k])

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
                    lo = 0  # every word with this prefix sorts at or after the last one's
                    while 0 <= rr < rows and 0 <= cc < cols:
                        word += grid[rr][cc]
                        rank, longer, lo = self._step(word, lo)
                        if len(word) >= MIN_LEN and rank is not None:
                            hits.append(Hit(rank, word, (r, c), (rr, cc)))
                        if not longer:
                            break
                        rr, cc = rr + dr, cc + dc
        return sorted(hits)


class _Ranks:
    """Dictionary.rank: `word in rank`, `rank[word]`, `rank.get(word)` like the dict
    it used to be."""

    def __init__(self, d: Dictionary) -> None:
        self._d = d

    def __contains__(self, word: object) -> bool:
        return isinstance(word, str) and self._d.get(word) is not None

    def __getitem__(self, word: str) -> int:
        rank = self._d.get(word)
        if rank is None:
            raise KeyError(word)
        return rank

    def get(self, word: str, default: int | None = None) -> int | None:
        rank = self._d.get(word)
        return default if rank is None else rank
