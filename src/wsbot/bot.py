"""Main script: read the board, fire every candidate word, wait for the next level.

READ_BOARD -> SOLVE -> NEXT, until the goal is met. The popup-watcher runs beside it
and owns every screenshot; this thread only reads its frames and swipes.

Solving escalates through passes until the level ends:
  1. fast        every dictionary word once (best-ranked path), back to back
  2. exhaustive  every straight line of 3+ letters with an unlit cell, never swiped
                 yet: finds words the dictionary doesn't know (ORANGUTAN)
  3. repeat      last resort: every path of every word not yet highlighted, even
                 ones already swiped
  4. restart     relaunch the app and start the level over

A word is swiped once per level before the repeat pass: re-swiping one the game has
already taken (a bonus word) pops an "already collected" toast over the bottom rows
that swallows the swipes under it.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from .board import Board, highlighted, read_board
from .device import open_device
from .goal import Goal, local_file, seconds_until_midnight
from .imgio import imwrite
from .letters import LetterReader
from .log import log
from .schedule import Pacer
from .solver import DIRECTIONS, MIN_LEN, Dictionary, Hit
from .watcher import PopupWatcher

PACKAGE = "in.playsimple.wordsearch"
MAX_DIAGNOSTICS = 150  # newest dumps kept; a 10-day run must not fill the disk
HIDDEN_END_S = 1.5  # board gone this long after a pass = the level is over
STALE_PREVIOUS_S = 15.0  # "finished" board still up this long = it wasn't finished
PROGRESS_CHECK_EVERY = 300  # exhaustive swipes between "is anything still being found?"


class InputBlocked(Exception):
    """The board is up but the game ignores touches (an invisible overlay)."""


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
        self.device = open_device(serial, dry_run=dry_run)
        self.diagnostics = root / "diagnostics"
        self.diagnostics.mkdir(exist_ok=True)
        self.letters = LetterReader(root / "templates" / "letters")
        self.words = Dictionary(root / "data" / "words.txt")
        templates = root / self.device.templates
        self.watcher = PopupWatcher(self.device, templates, PACKAGE, self.diagnostics)
        self.pacing = Pacing(swipe_ms=self.device.swipe_ms)
        self.pacer = Pacer(goal.schedule, goal.progress) if goal.schedule else None
        self.resting_until: float | None = None  # epoch; set while idling / on a break
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
        self.board_center = self.device.board_center  # middle of the board; updated each level
        self.clear_taps = 0
        # Words swiped on the current level, saved so a bot restart mid-level doesn't
        # re-swipe them (each re-swipe of a taken word pops a toast).
        self._swiped_path = local_file(root, serial, "level_swiped.json")
        self.swiped, self._swiped_grid = self._load_swiped()

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
            if different_from is not None and same_level(grid, different_from):
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
    # boxes sit right over it and only close on a tap outside them (the hint card area,
    # device.above_board).

    def _clear_tap(self) -> None:
        if self.clear_taps % 2 == 0:
            self.device.tap(*self.board_center, why="clear popup (board center)")
        else:
            self.device.tap(*self.device.above_board, why="clear popup (above board)")
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
            if new_grid is None:
                # Couldn't read any board: that is not evidence of a new level (an
                # unreadable glyph once ended a level early). Let the level-end
                # check below decide.
                if self._wait_level_end(grid, timeout=self.pacing.level_end_s):
                    return False
                continue
            if new_grid != grid:
                return False
            log("PAUSE", f"board was hidden {time.monotonic() - t0:.1f}s; resuming")
            time.sleep(self.pacing.settle_s)
        return False

    def burst(
        self,
        board: Board,
        grid: list[str],
        hits: list[Hit],
        ms: int,
        gap: float,
        check_every: int = 0,
    ) -> bool:
        """Swipe every hit in order. True if all were fired without the level ending.

        No refiring here: most swipes near a popup did land, and re-swiping a word the
        game already took pops a toast. A swipe a popup really ate is caught by the
        repeat pass.
        """
        for i, hit in enumerate(hits):
            if check_every and i and i % check_every == 0:
                self._check_input(board, self._lit_cells(board))
            if not self._ready_to_swipe(grid):
                return False
            ok = self.device.swipe(
                board.cell(*hit.start).center, board.cell(*hit.end).center, ms, why=hit.word
            )
            if ok:
                self.stats.swipes += 1
                self.fired_cells.update(path_cells(hit))
                self.swiped.add(hit.word)
            if gap:
                time.sleep(gap)
        return True

    # ---- level solving ---------------------------------------------------------

    def solve_level(self, board: Board, grid: list[str]) -> bool:
        """Escalating passes (see module docstring). True once the level ends."""
        hits = self.words.solve(grid)
        seen: set[str] = set()
        fast = [h for h in hits if not (h.word in seen or seen.add(h.word))]
        if self._swiped_grid is None or not same_level(grid, self._swiped_grid):
            self.swiped, self._swiped_grid = set(), grid  # kept across restarts of a level

        def new(todo: list[Hit]) -> list[Hit]:
            done = set(self.swiped)  # snapshot: repeated non-words within a pass are fine
            return [h for h in todo if h.word not in done]

        p = self.pacing
        passes = [
            ("fast", lambda: new(self._unlit(board, fast)), p.swipe_ms, p.gap_s, 0),
            (
                "exhaustive",
                lambda: new(self._unlit(board, all_lines(grid), most_unlit_first=True)),
                p.swipe_ms,
                p.gap_s,
                PROGRESS_CHECK_EVERY,
            ),
            ("repeat", lambda: self._unlit(board, hits), p.refire_swipe_ms, p.refire_gap_s, 0),
        ]
        self._checked_lit = set()
        try:
            for name, pick, ms, gap, check_every in passes:
                todo = pick()
                self.phase = f"{name} ({len(todo)})"
                if name != "fast":
                    log("PASS", f"{name}: {len(todo)} swipes")
                    self._check_input(board, self.found_cells)
                finished = not self.burst(board, grid, todo, ms, gap, check_every)
                self._save_swiped()
                if finished:
                    return True
                if self._wait_level_end(grid, timeout=p.level_end_s):
                    return True
        except InputBlocked:
            self._dump("input_blocked")
            log("WARN", "the game ignores touches on the board; restarting")
        return False

    def _check_input(self, board: Board, lit: set[tuple[int, int]] | None) -> None:
        """Raise InputBlocked if nothing new got found and a probe drag doesn't show up.

        A forced tutorial or a half-finished popup animation can leave the board looking
        normal while every touch is swallowed; without this the exhaustive pass burns
        minutes of swipes into nothing before the restart.
        """
        if lit is None or lit != self._checked_lit:
            self._checked_lit = lit if lit is not None else self._checked_lit
            return
        for _ in range(2):
            alive = self._probe_input(board, lit)
            if alive is not False:
                return
            time.sleep(1.0)
        raise InputBlocked

    def _probe_input(self, board: Board, lit: set[tuple[int, int]]) -> bool | None:
        """Drag across two unlit cells without lifting and look for the selection pill.

        Two letters are never a word, so lifting the finger afterwards is free.
        None = couldn't probe (no unlit pair, board hidden).
        """
        pair = next(
            (
                ((r, c), (r, c + 1))
                for r in range(board.rows)
                for c in range(board.cols - 1)
                if (r, c) not in lit and (r, c + 1) not in lit
            ),
            None,
        )
        if pair is None or not self.watcher.board_visible or self.watcher.busy():
            return None
        a, b = (board.cell(*cell).center for cell in pair)
        if not self.device.hold(a, b):
            return None
        try:
            if self.device.dry_run:
                return None
            # The selection can take ~0.5 s to show (iPhone), so watch frames for a while.
            alive, seen = False, False
            deadline = time.monotonic() + 2.0
            while not alive and time.monotonic() < deadline:
                frame = self.watcher.latest(newer_than=self.watcher.frame_time, timeout=0.5)
                if frame is None:
                    continue
                seen = True
                alive = all(highlighted(frame, board, *cell) for cell in pair)
            if not seen:
                return None
        finally:
            self.device.release(b)
        log("PROBE", f"drag {pair[0]}->{pair[1]}: {'registered' if alive else 'IGNORED'}")
        return alive

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

    def _unlit(self, board: Board, hits: list[Hit], *, most_unlit_first: bool = False) -> list[Hit]:
        """Hits with at least one cell not yet covered by a found-word pill.

        most_unlit_first: words the dictionary misses are mostly long theme words
        (ORANGUTAN) lying on untouched cells, so lines with the most unlit cells go first.
        """
        lit = self._lit_cells(board)
        if lit is None:
            return hits
        self.found_cells = lit
        todo = [h for h in hits if not set(path_cells(h)) <= lit]
        if most_unlit_first:
            todo.sort(key=lambda h: (-sum(c not in lit for c in path_cells(h)), -len(h.word)))
        return todo

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
            if read is None or same_level(read, grid):
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
                    g = self.goal
                    if g.session_target is not None and g.session_levels >= g.session_target:
                        log("GOAL", f"played the {g.session_target} levels asked for; stopping")
                    else:
                        log("GOAL", f"target reached: {g.title}")
                    break
                if self.goal.quota_reached_today():
                    self._sleep_until_tomorrow()
                    continue
                if self.pacer and (wait := self.pacer.wait_before_start()) > 0:
                    self._rest(wait, "waiting for today's play window")
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
                    self._forget_swiped(everything=restarts_this_level >= 3)
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
                self._pace(dt)
        finally:
            # Stop the watcher before the device closes, or its next tick sees the phone
            # gone and tries to reconnect it.
            self.watcher.stop()
            if self.watcher.is_alive():
                self.watcher.join(timeout=10)
            self.device.close()
            self.status = "stopped"

    def _pace(self, level_s: float) -> None:
        """Idle between levels / take a break, per the goal's schedule."""
        if self.pacer is None or self.goal.finished() or self.goal.quota_reached_today():
            return
        if brk := self.pacer.break_due():
            self._rest(brk, "on a break")
            self.pacer.break_taken()
        elif (idle := self.pacer.idle_after_level(level_s)) >= 1:
            self._rest(idle, "resting")

    def _rest(self, seconds: float, what: str) -> None:
        """Sit still (stoppable). Long rests stop the watcher's screenshots too."""
        until = time.time() + seconds
        self.resting_until = until
        back = time.strftime("%H:%M", time.localtime(until))
        self.status = f"{what} · back at {back}"
        if seconds >= 60:
            log("PACE", f"{what} for {seconds / 60:.0f} min, back at {back}")
            self.watcher.idle.set()
        try:
            while not self.stop_event.is_set() and (left := until - time.time()) > 0:
                self.stop_event.wait(min(1.0, left))
        finally:
            self.watcher.idle.clear()
            self.resting_until = None

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

    def _load_swiped(self) -> tuple[set[str], list[str] | None]:
        try:
            data = json.loads(self._swiped_path.read_text(encoding="utf-8"))
            return set(data["words"]), data["grid"]
        except (OSError, ValueError, KeyError):
            return set(), None

    def _forget_swiped(self, *, everything: bool) -> None:
        """A stuck level gets a fresh try: swipes fired while the game ignored input
        (a tutorial, a half-closed popup) never counted, and kept as "swiped" they were
        never fired again. Lines outside the dictionary go first (SCAVENGER was one:
        swiped once into a blocked board, then skipped on 27 restarts); from the third
        try, every word, accepting a few "already collected" toasts."""
        before = len(self.swiped)
        if everything:
            self.swiped.clear()
        else:
            self.swiped = {w for w in self.swiped if w in self.words.rank}
        self._save_swiped()
        log("RECOVERY", f"will re-swipe {before - len(self.swiped)} words on this level")

    def _save_swiped(self) -> None:
        data = {"grid": self._swiped_grid, "words": sorted(self.swiped)}
        tmp = self._swiped_path.with_suffix(".tmp")
        try:
            self._swiped_path.parent.mkdir(exist_ok=True)
            tmp.write_text(json.dumps(data), encoding="utf-8")
            tmp.replace(self._swiped_path)
        except OSError as exc:
            log("WARN", f"couldn't save swiped words: {exc!r}")

    def _dump(self, why: str) -> None:
        frame = self.watcher.frame
        if frame is None:
            return
        path = self.diagnostics / f"{why}_{time.strftime('%Y%m%d_%H%M%S')}.png"
        imwrite(path, frame)
        log("DIAG", f"saved {path.name}")
        dumps = sorted(self.diagnostics.glob("*.png"), key=lambda p: p.stat().st_mtime)
        for old in dumps[:-MAX_DIAGNOSTICS]:
            old.unlink(missing_ok=True)


def same_level(a: list[str], b: list[str]) -> bool:
    """True if two reads are the same board: most overlapping letters agree.

    A new level's letters match the old ones only by chance (~1 in 26), while a misread
    glyph or a partly hidden board keeps nearly all of them.
    """
    if a == b:
        return True
    pairs = [(x, y) for ra, rb in zip(a, b, strict=False) for x, y in zip(ra, rb, strict=False)]
    return bool(pairs) and sum(x == y for x, y in pairs) >= 0.6 * len(pairs)


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
