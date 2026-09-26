"""Main script: read the board, fire every candidate word, wait for the next level.

READ_BOARD -> BURST -> WAIT_NEXT, forever. The popup-watcher runs beside it and
owns every screenshot; this thread only reads its frames and swipes.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from .board import Board, highlighted, read_board
from .device import Device
from .letters import LetterReader
from .log import log
from .solver import Dictionary, Hit
from .watcher import PopupWatcher

PACKAGE = "in.playsimple.wordsearch"


@dataclass
class Pacing:
    swipe_s: float = 0.20  # finger travel time per swipe
    gap_s: float = 0.15  # pause after each swipe
    refire_gap_s: float = 0.7  # slower refires land after word-found animations
    settle_s: float = 0.4  # pause after a popup clears before swiping again
    level_end_s: float = 2.5  # how long to wait for the level to end after a pass
    next_level_timeout_s: float = 20.0


@dataclass
class Stats:
    started: float = field(default_factory=time.monotonic)
    levels: int = 0
    swipes: int = 0
    level_times: list[float] = field(default_factory=list)


class Bot:
    def __init__(self, serial: str, root: Path, *, dry_run: bool = False) -> None:
        self.root = root
        self.device = Device(serial, dry_run=dry_run)
        self.diagnostics = root / "diagnostics"
        self.diagnostics.mkdir(exist_ok=True)
        self.letters = LetterReader(root / "templates" / "letters")
        self.words = Dictionary(root / "data" / "words.txt")
        self.watcher = PopupWatcher(self.device, root / "templates", PACKAGE, self.diagnostics)
        self.pacing = Pacing()
        self.stats = Stats()
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        # Live state for the UI.
        self.grid: list[str] = []
        self.hits: list[Hit] = []
        self.fired: set[str] = set()
        self.status = "idle"

    # ---- board ---------------------------------------------------------------

    def read_grid(self, board: Board) -> list[str] | None:
        rows = []
        for r in range(board.rows):
            row = ""
            for c in range(board.cols):
                letter = self.letters.read(board.cell(r, c).glyph)
                if letter is None:
                    self._dump(f"unreadable_r{r}c{c}")
                    return None
                row += letter
            rows.append(row)
        return rows

    def wait_for_board(self, *, different_from: list[str] | None = None, timeout: float = 20.0):
        """Block until two consecutive frames show the same readable board."""
        deadline = time.monotonic() + timeout
        last_time, prev = 0.0, None
        while not self.stop_event.is_set() and time.monotonic() < deadline:
            frame = self.watcher.latest(newer_than=last_time)
            if frame is None:
                continue
            last_time = self.watcher.frame_time
            if self.watcher.popup_active.is_set():
                prev = None
                continue
            board = read_board(frame)
            grid = self.read_grid(board) if board else None
            if grid is None or grid == different_from:
                prev = None
                continue
            if grid == prev:
                return board, grid
            prev = grid
        return None, None

    # ---- burst ---------------------------------------------------------------

    def _ready_to_swipe(self, grid: list[str]) -> bool:
        """Hold while a popup or pause is up. False means the level is over."""
        while not self.stop_event.is_set():
            if self.pause_event.is_set():
                time.sleep(0.2)
                continue
            if self.watcher.level_done.is_set():
                return False
            if self.watcher.board_visible and not self.watcher.popup_active.is_set():
                return True
            # Board hidden: a popup, or the level is finishing. Wait it out, then
            # make sure it's still the same board before swiping again.
            self.status = "waiting for popup"
            _, new_grid = self.wait_for_board(timeout=self.pacing.next_level_timeout_s)
            if new_grid != grid:
                return False
            time.sleep(self.pacing.settle_s)
            self.status = "solving"
        return False

    def burst(self, board: Board, grid: list[str], hits: list[Hit], gap: float) -> bool:
        """Swipe every hit in order. True if all were fired without the level ending."""
        for hit in hits:
            if not self._ready_to_swipe(grid):
                return False
            ok = self.device.swipe(
                board.cell(*hit.start).center,
                board.cell(*hit.end).center,
                self.pacing.swipe_s,
                why=hit.word,
            )
            if ok:
                self.fired.add(hit.word)
                self.stats.swipes += 1
            time.sleep(gap)
        return True

    # ---- main loop -------------------------------------------------------------

    def run(self) -> None:
        self.watcher.start()
        previous: list[str] | None = None
        level_n = 0
        try:
            while not self.stop_event.is_set():
                self.status = "reading board"
                board, grid = self.wait_for_board(different_from=previous, timeout=60)
                if board is None:
                    if self.stop_event.is_set():
                        break
                    log("WARN", "no new board within 60s; still waiting")
                    self._dump("no_board")
                    continue
                self.watcher.level_done.clear()
                level_n += 1
                t0 = time.monotonic()
                self.grid, self.fired = grid, set()
                self.hits = self.words.solve(grid)
                log(
                    "LEVEL",
                    f"#{level_n} {board.rows}x{board.cols} {'/'.join(grid)} "
                    f"-> {len(self.hits)} candidates",
                )
                self.status = "solving"

                finished = self.solve_level(board, grid)
                if not finished:
                    self._dump("level_stuck")
                    log("ERROR", "level still not done after all passes; waiting for help")
                    self._wait_level_end(grid, timeout=float("inf"))

                dt = time.monotonic() - t0
                self.stats.levels += 1
                self.stats.level_times.append(dt)
                log("LEVEL", f"#{level_n} done in {dt:.1f}s ({len(self.fired)} swipes)")
                previous = grid
        finally:
            self.watcher.stop()
            self.status = "stopped"

    def solve_level(self, board: Board, grid: list[str]) -> bool:
        """Fast blind pass, then targeted slow refires. True once the level ends."""
        passes = [
            ("fast", self.pacing.gap_s, False),
            ("refire-unfound", self.pacing.refire_gap_s, True),
            ("refire-all", self.pacing.refire_gap_s, False),
        ]
        for name, gap, only_unfound in passes:
            hits = self.hits
            if only_unfound:
                hits = self._unfound(board, grid)
                log("PASS", f"{name}: {len(hits)} of {len(self.hits)} candidates not highlighted")
            elif name != "fast":
                log("PASS", f"{name}: {len(hits)} candidates")
            if not self.burst(board, grid, hits, gap):
                return True
            self.status = "waiting for level end"
            if self._wait_level_end(grid, timeout=self.pacing.level_end_s):
                return True
        return False

    def _unfound(self, board: Board, grid: list[str]) -> list[Hit]:
        """Candidates with at least one cell not yet covered by a found-word pill."""
        frame = self.watcher.latest(newer_than=time.monotonic())
        if frame is None or read_board(frame) is None:
            return self.hits
        lit = {
            (r, c)
            for r in range(board.rows)
            for c in range(board.cols)
            if highlighted(frame, board, r, c)
        }
        out = []
        for hit in self.hits:
            (r0, c0), (r1, c1) = hit.start, hit.end
            n = max(abs(r1 - r0), abs(c1 - c0))
            dr, dc = (r1 - r0) // n, (c1 - c0) // n
            if any((r0 + i * dr, c0 + i * dc) not in lit for i in range(n + 1)):
                out.append(hit)
        return out

    def _wait_level_end(self, grid: list[str], timeout: float) -> bool:
        """True once the board is gone or shows a different grid."""
        deadline = time.monotonic() + timeout
        while not self.stop_event.is_set() and time.monotonic() < deadline:
            if self.watcher.level_done.is_set() or not self.watcher.board_visible:
                return True
            frame = self.watcher.latest(newer_than=self.watcher.frame_time, timeout=1.0)
            board = read_board(frame) if frame is not None else None
            if board and self.read_grid(board) != grid:
                return True
        return False

    def _dump(self, why: str) -> None:
        frame = self.watcher.frame
        if frame is not None:
            path = self.diagnostics / f"{why}_{time.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(str(path), frame)
            log("DIAG", f"saved {path.name}")
