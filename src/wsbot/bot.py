"""Main script: read the board, fire every candidate word, wait for the next level.

READ_BOARD -> SOLVE -> NEXT, until the goal is met. The popup-watcher runs beside it
and owns every screenshot; this thread only reads its frames and swipes.

Solving escalates through passes until the level ends:
  1. fast        every dictionary word once (best-ranked path), back to back
  2. unfound     words not yet highlighted, every path, a little slower
  3. exhaustive  every straight line of 3+ letters with an unlit cell: finds words
                 the dictionary doesn't know, so no level can block the run
  4. restart     relaunch the app and start the level over
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from .board import Board, highlighted, read_board
from .device import Device
from .goal import Goal, seconds_until_midnight
from .letters import LetterReader
from .log import log
from .solver import DIRECTIONS, MIN_LEN, Dictionary, Hit
from .watcher import PopupWatcher

PACKAGE = "in.playsimple.wordsearch"
MAX_DIAGNOSTICS = 150  # newest dumps kept; a 10-day run must not fill the disk
HIDDEN_END_S = 1.5  # board gone this long after a pass = the level is over
STALE_PREVIOUS_S = 15.0  # "finished" board still up this long = it wasn't finished


@dataclass
class Pacing:
    swipe_ms: int = 60  # finger travel time per swipe (the game takes 40 ms fine)
    gap_s: float = 0.0  # pause after each swipe; the game never blocks input on a find
    refire_swipe_ms: int = 120  # later passes go slower, in case speed caused a miss
    refire_gap_s: float = 0.1
    settle_s: float = 0.3  # pause after a popup clears before swiping again
    clear_tap_s: float = 2.5  # board hidden this long with no known popup -> clear tap
    level_end_s: float = 2.5  # how long to wait for the level to end after a pass
    no_board_restart_s: float = 150.0  # no board at all this long -> restart the app


@dataclass
class Stats:
    started: float = field(default_factory=time.monotonic)
    levels: int = 0
    swipes: int = 0
    restarts: int = 0
    level_times: deque[float] = field(default_factory=lambda: deque(maxlen=200))

    def avg_level_s(self) -> float:
        return sum(self.level_times) / len(self.level_times) if self.level_times else 0.0


class Bot:
    def __init__(self, serial: str, root: Path, goal: Goal, *, dry_run: bool = False) -> None:
        self.root = root
        self.goal = goal
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
        self.fired_cells: set[tuple[int, int]] = set()
        self.found_cells: set[tuple[int, int]] = set()
        self.status = "starting"
        self.phase = ""
        self.level_started = 0.0
        self.board_center = (540, 1325)  # middle of the board; updated each level
        self.clear_taps = 0

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
        """Block until two consecutive frames show the same readable board.

        While the board stays hidden and no known popup is being handled, tap to clear:
        the game's tutorial and bonus popups close on any click, so this handles them
        without a template for each one.
        """
        start = time.monotonic()
        deadline = start + timeout
        last_time, prev = 0.0, None
        hidden_since = last_clear = last_report = start
        stale_since: float | None = None
        why = "no frame yet"
        while not self.stop_event.is_set() and time.monotonic() < deadline:
            now = time.monotonic()
            if now - last_report >= 5:
                last_report = now
                log("WAIT", f"no board for {now - start:.0f}s: {why}")
            frame = self.watcher.latest(newer_than=last_time)
            if frame is None:
                why = "watcher produced no frame"
                continue
            last_time = self.watcher.frame_time
            board = read_board(frame)
            now = time.monotonic()
            if board is not None:
                hidden_since = now
            elif self._should_clear_tap(now, hidden_since, last_clear):
                last_clear = now
                self._clear_tap()
            if self.watcher.busy():
                prev, why = None, "watcher is handling a popup"
                continue
            grid = self.read_grid(board) if board else None
            if board is None:
                prev, why = None, "board not visible"
                continue
            if grid is None:
                prev, why = None, "letters unreadable"
                continue
            if grid == different_from:
                stale_since = stale_since or now
                if now - stale_since < STALE_PREVIOUS_S:
                    prev, why = None, "still the previous level"
                    continue
                log("WARN", "the previous level is still on screen; solving it again")
                different_from = None
            else:
                stale_since = None
            if grid == prev:
                x, y, w, h = board.panel
                self.board_center = (x + w // 2, y + h // 2)
                return board, grid
            why = (
                "waiting for a second matching read"
                if prev is None
                else f"reads disagree: {'/'.join(prev)} vs {'/'.join(grid)}"
            )
            prev = grid
        return None, None

    # Where the clear tap alternates: the board center clears most popups, but tutorial
    # boxes sit right over it and only close on a tap outside them (the hint card area).
    ABOVE_BOARD = (540, 620)

    def _clear_tap(self) -> None:
        if self.clear_taps % 2 == 0:
            self.device.tap(*self.board_center, why="clear popup (board center)")
        else:
            self.device.tap(*self.ABOVE_BOARD, why="clear popup (above board)")
        self.clear_taps += 1

    def _should_clear_tap(self, now: float, hidden_since: float, last_clear: float) -> bool:
        wait = self.pacing.clear_tap_s
        return (
            now - hidden_since >= wait
            and now - last_clear >= wait
            and now - self.watcher.last_match >= wait
        )

    # ---- burst ---------------------------------------------------------------

    def _ready_to_swipe(self, grid: list[str]) -> bool:
        """Hold while a popup or pause is up. False means the level is over."""
        while not self.stop_event.is_set():
            if self.pause_event.is_set():
                self.status = "paused"
                time.sleep(0.2)
                continue
            if self.watcher.level_done.is_set():
                return False
            if self.watcher.board_visible and not self.watcher.busy():
                self.status = "solving"
                return True
            # Board hidden: a popup, or the level is finishing. Wait it out, then
            # make sure it's still the same board before swiping again.
            self.status = "waiting for popup"
            t0 = time.monotonic()
            _, new_grid = self.wait_for_board(timeout=20)
            if new_grid != grid:
                return False
            log("PAUSE", f"board was hidden {time.monotonic() - t0:.1f}s; resuming")
            time.sleep(self.pacing.settle_s)
        return False

    def burst(self, board: Board, grid: list[str], hits: list[Hit], ms: int, gap: float) -> bool:
        """Swipe every hit in order. True if all were fired without the level ending."""
        for hit in hits:
            if not self._ready_to_swipe(grid):
                return False
            ok = self.device.swipe(
                board.cell(*hit.start).center, board.cell(*hit.end).center, ms, why=hit.word
            )
            if ok:
                self.stats.swipes += 1
                self.fired_cells.update(path_cells(hit))
            if gap:
                time.sleep(gap)
        return True

    # ---- level solving ---------------------------------------------------------

    def solve_level(self, board: Board, grid: list[str]) -> bool:
        """Escalating passes (see module docstring). True once the level ends."""
        hits = self.words.solve(grid)
        seen: set[str] = set()
        fast = [h for h in hits if not (h.word in seen or seen.add(h.word))]
        p = self.pacing
        passes = [
            ("fast", lambda: fast, p.swipe_ms, p.gap_s),
            ("unfound", lambda: self._unlit(board, hits), p.refire_swipe_ms, p.refire_gap_s),
            ("exhaustive", lambda: self._unlit(board, all_lines(grid)), p.swipe_ms, p.gap_s),
        ]
        for name, pick, ms, gap in passes:
            todo = pick()
            self.phase = f"{name} ({len(todo)})"
            if name != "fast":
                log("PASS", f"{name}: {len(todo)} swipes")
            if not self.burst(board, grid, todo, ms, gap):
                return True
            if self._wait_level_end(grid, timeout=p.level_end_s):
                return True
        return False

    def _lit_cells(self, board: Board) -> set[tuple[int, int]] | None:
        """Cells covered by a found-word pill, read from a frame where the board is up."""
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not self.stop_event.is_set():
            frame = self.watcher.latest(newer_than=time.monotonic())
            if frame is not None and self.watcher.board_visible:
                return {
                    (r, c)
                    for r in range(board.rows)
                    for c in range(board.cols)
                    if highlighted(frame, board, r, c)
                }
        return None

    def _unlit(self, board: Board, hits: list[Hit]) -> list[Hit]:
        """Hits with at least one cell not yet covered by a found-word pill."""
        lit = self._lit_cells(board)
        if lit is None:
            return hits
        self.found_cells = lit
        return [h for h in hits if not set(path_cells(h)) <= lit]

    def _wait_level_end(self, grid: list[str], timeout: float) -> bool:
        """True once the level has really ended.

        One odd frame (a banner over the board, a misread glyph) must not count, or the
        bot moves on while the level is unfinished and then waits forever for a "new"
        board. So: the watcher saw a level-end button, or the board stayed hidden for
        HIDDEN_END_S, or a different grid read identically twice in a row.
        """
        deadline = time.monotonic() + timeout
        hidden_since: float | None = None
        other: list[str] | None = None
        while not self.stop_event.is_set() and time.monotonic() < deadline:
            if self.watcher.level_done.is_set():
                return True
            frame = self.watcher.latest(newer_than=self.watcher.frame_time, timeout=1.0)
            if frame is None:
                continue
            now = time.monotonic()
            board = read_board(frame)
            if board is None:
                hidden_since = hidden_since or now
                if now - hidden_since >= HIDDEN_END_S:
                    return True
                continue
            hidden_since = None
            read = self.read_grid(board)
            if read is None or read == grid:
                other = None
                continue
            if read == other:
                return True
            other = read
        return False

    # ---- main loop -------------------------------------------------------------

    def run(self) -> None:
        self.watcher.start()
        previous: list[str] | None = None
        restarts_this_level = 0
        try:
            while not self.stop_event.is_set():
                if self.goal.finished():
                    log("GOAL", f"target reached: {self.goal.title}")
                    break
                if self.goal.quota_reached_today():
                    self._sleep_until_tomorrow()
                    continue

                self.status = "reading board"
                self.phase = ""
                board, grid = self.wait_for_board(
                    different_from=previous, timeout=self.pacing.no_board_restart_s
                )
                if board is None:
                    if self.stop_event.is_set():
                        break
                    self._dump("no_board")
                    self.restart_app("no board for a long time")
                    previous = None
                    continue

                self.watcher.level_done.clear()
                self.watcher.expected_panel = board.panel
                self.device.below_board_y = board.panel[1] + board.panel[3] + 5
                self.grid, self.fired_cells, self.found_cells = grid, set(), set()
                self.level_started = t0 = time.monotonic()
                n = self.goal.done_total + 1
                log("LEVEL", f"#{n} {board.rows}x{board.cols} {'/'.join(grid)}")
                self.status = "solving"
                finished = self.solve_level(board, grid)
                self.watcher.expected_panel = None
                self.device.below_board_y = None
                if self.stop_event.is_set():
                    break
                if not finished:
                    restarts_this_level += 1
                    self._dump("level_stuck")
                    self.restart_app(f"level stuck after every pass (try {restarts_this_level})")
                    previous = None  # the same level comes back after a restart
                    continue

                restarts_this_level = 0
                dt = time.monotonic() - t0
                self.stats.levels += 1
                self.stats.level_times.append(dt)
                self.goal.record_level()
                log("LEVEL", f"#{n} done in {dt:.1f}s · today {self.goal.done_today}")
                previous = grid
        finally:
            self.watcher.stop()
            self.device.shell.close()
            self.status = "stopped"

    def restart_app(self, why: str) -> None:
        self.stats.restarts += 1
        log("RECOVERY", f"restarting the game: {why}")
        self.status = "restarting game"
        if not self.device.dry_run:
            self.device.app_stop(PACKAGE)
            time.sleep(1.0)
            self.device.app_start(PACKAGE)
        time.sleep(4.0)

    def _sleep_until_tomorrow(self) -> None:
        wait = seconds_until_midnight() + 5
        log("GOAL", f"today's {self.goal.per_day:,} levels done; resuming in {wait / 3600:.1f}h")
        self.status = "daily target reached · resumes at midnight"
        self.watcher.idle.set()  # stop screenshotting while there's nothing to do
        self.stop_event.wait(wait)
        self.watcher.idle.clear()

    def _dump(self, why: str) -> None:
        frame = self.watcher.frame
        if frame is None:
            return
        path = self.diagnostics / f"{why}_{time.strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(str(path), frame)
        log("DIAG", f"saved {path.name}")
        dumps = sorted(self.diagnostics.glob("*.png"), key=lambda p: p.stat().st_mtime)
        for old in dumps[:-MAX_DIAGNOSTICS]:
            old.unlink(missing_ok=True)


def path_cells(hit: Hit) -> list[tuple[int, int]]:
    (r0, c0), (r1, c1) = hit.start, hit.end
    n = max(abs(r1 - r0), abs(c1 - c0))
    dr, dc = (r1 - r0) // n, (c1 - c0) // n
    return [(r0 + i * dr, c0 + i * dc) for i in range(n + 1)]


def all_lines(grid: list[str]) -> list[Hit]:
    """Every straight segment of MIN_LEN+ cells, as pseudo-hits (common word lengths first)."""
    rows, cols = len(grid), len(grid[0])
    out = []
    for r in range(rows):
        for c in range(cols):
            for dr, dc in DIRECTIONS:
                word, rr, cc = "", r, c
                while 0 <= rr < rows and 0 <= cc < cols:
                    word += grid[rr][cc]
                    if len(word) >= MIN_LEN:
                        out.append(Hit(abs(len(word) - 5), word, (r, c), (rr, cc)))
                    rr, cc = rr + dr, cc + dc
    return sorted(out)
