"""Main script: read the board, fire every candidate word, wait for the next level.

READ_BOARD -> SOLVE -> NEXT, until the goal is met. The popup-watcher runs beside it
and owns every screenshot; this thread only reads its frames and swipes.

Solving escalates through passes until the level ends:
  1. fast        every dictionary word once (best-ranked path), back to back
  2. retry       the common dictionary words still unlit, and every unlit one of 5+
                 letters, once more with a breath between swipes: the iPhone game
                 drops some back-to-back swipes, and the exhaustive pass reaches short
                 ones (MILK, OIL) only at its end and never re-fires a swiped word
                 (ICICLES, rank 42,102, waited for the repeat pass)
  3. careful     the unlit 5+ letter dictionary words once more, slowly with a long
                 breath: a long theme word the game dropped twice
  4. exhaustive  every straight line of 3+ letters with an unlit cell, never swiped
                 yet: finds words the dictionary doesn't know (ORANGUTAN)
  5. slow        lines outside the dictionary on unlit cells, once more, slowly: the
                 game drops a swipe now and then, and the exhaustive pass has one shot
  6. repeat      last resort: every path of every word not yet highlighted, even
                 ones already swiped
  7. restart     relaunch the app and start the level over

A word is swiped once per level before the repeat pass: re-swiping one the game has
already taken (a bonus word) pops an "already collected" toast over the bottom rows
that swallows the swipes under it.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .board import Board, highlighted
from .debug import dbg, snap
from .device import open_device
from .goal import Goal, local_file, seconds_until_midnight
from .imgio import imwrite
from .letters import LetterReader
from .log import log
from .schedule import Pacer
from .solver import DIRECTIONS, MIN_LEN, Dictionary, Hit
from .watcher import PopupWatcher

PACKAGE = "in.playsimple.wordsearch"
MAX_DIAGNOSTICS = 40  # newest dumps kept (~2 MB each); a 10-day run must not fill the disk
DUMP_EVERY_S = 300.0  # one dump of a kind this often: a stuck read saved one a second
# Board gone this long after a pass = the level is over. 3 s, not 1.5: the bonus
# "Claim" popup fades in over the board ~1 s before the watcher sees it, and 1.5 s
# counted that as a level end. Free: the next board takes 5+ s to come anyway.
HIDDEN_END_S = 3.0
MID_LEVEL_HOLD_S = 3.0  # a mid-level popup seen this recently explains a hidden board
STALE_PREVIOUS_S = 15.0  # "finished" board still up this long = it wasn't finished
PROGRESS_CHECK_EVERY = 300  # exhaustive swipes between "is anything still being found?"
# The game left alone this long (bot start, a break, the night) ignored every touch
# until relaunched: the first level burned its fast pass, then a probe restarted it.
IDLE_RELAUNCH_S = 300.0
HUNG_S = 40.0  # board gone and not one pixel changed this long despite clear taps: hung
RETRY_RANK = 20_000  # retry pass: dictionary words this common (MILK, OIL, CHARGER 4,635)
# ...and dictionary words this long at any rank: long theme words are often rare
# (ICICLES 42,102, FRISBEE 35,363, PINWHEEL 59,144). Only the retry pass re-fires a
# dropped dictionary word, so outside it they waited for the repeat pass, minutes
# later. Costs ~4.6 swipes a level (4+ letters would be ~14, mostly junk). Only while
# at least half their cells are unlit: a bonus word the game already took mostly lies
# on found words (TRESSED on DESSERT's row), and re-swiping it pops the toast; with
# them the retry pass of level 5378 waited out 5 toasts (27 s).
RETRY_MIN_LEN = 5
# 4-letter words too, while every cell is unlit: rare short theme words (HARP 39,783,
# LUTE, OBOE, LYRE, GONG, FIFE all > 20k) were dropped the same way; HARP, the first
# swipe of its level, fell to the exhaustive pass. ~10 such words a board at most.
RETRY_SHORT_LEN = 4
# Swipes fired this long before the toast was first seen landed under it (the eye sees
# it ~50 ms late). Wider re-fired swipes from just before it showed: taken bonus words,
# whose re-swipe popped another toast, in a cascade (the retry pass took 32 s).
TOAST_EATS_S = 0.1
# Swipes fired this long before the bot had to hold for a popup may have been eaten:
# the watcher's ~0.45 s tick plus the popup fading in.
POPUP_EATS_S = 0.75
LEARN_PASSES = ("exhaustive", "slow")
LEARN_MIN_LEN = 4
LEARNED_RANK = 1_000  # learned theme words go early in the fast pass


class InputBlocked(Exception):
    """The board is up but the game ignores touches (an invisible overlay)."""


@dataclass
class Pacing:
    swipe_ms: int = 60  # finger travel time per swipe (the game takes 40 ms fine)
    gap_s: float = 0.0  # pause after each swipe; the game never blocks input on a find
    refire_swipe_ms: int = 120  # later passes go slower, in case speed caused a miss
    refire_gap_s: float = 0.1
    retry_gap_s: float = 0.1  # retry pass: a breath between swipes lets the game take each
    careful_gap_s: float = 0.5  # careful pass: the game took 5/5 dropped words with long pauses
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
        self._learned_path = root / "local" / "learned-words.txt"
        self._load_learned()
        self.pass_fired: list[Hit] = []
        templates = root / self.device.templates
        self.watcher = PopupWatcher(self.device, templates, PACKAGE, self.diagnostics)
        self.pacing = Pacing(swipe_ms=self.device.swipe_ms)
        self.pacer = Pacer(goal.schedule, goal.progress) if goal.schedule else None
        self.resting_until: float | None = None  # epoch; set while idling / on a break
        self.stats = Stats()
        self._relaunch = True  # see IDLE_RELAUNCH_S
        self.stop_event = threading.Event()
        # Set while AutomationHQ has paused the bot (see control.py).
        self.pause_event = threading.Event()
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
        self.held_at: float | None = None  # see _ready_to_swipe
        self.maybe_eaten: set[Hit] = set()
        self._input_blocked = False
        self._blocked_grid: list[str] | None = None  # level a "touches ignored" restart was for
        # Words swiped on the current level, saved so a bot restart mid-level doesn't
        # re-swipe them (each re-swipe of a taken word pops a toast).
        self._swiped_path = local_file(root, serial, "level_swiped.json")
        self.swiped, self._swiped_grid = self._load_swiped()
        self._dumped: dict[str, float] = {}  # dump kind -> when (see _dump)

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
        still, still_since = None, start  # the game froze once with letters mid-flight
        why = "no frame yet"
        while not self.stop_event.is_set() and time.monotonic() < deadline:
            now = time.monotonic()
            if now - last_report >= 5:
                last_report = now
                log("WAIT", f"no board for {now - start:.0f}s: {why}")
            shot = self.watcher.latest_shot(newer_than=last_time)
            if shot is None:
                why = "watcher produced no frame"
                continue
            last_time = self.watcher.frame_time
            frame, board = shot.calib, shot.board
            now = time.monotonic()
            thumb = cv2.resize(frame, (27, 48), interpolation=cv2.INTER_AREA).astype(np.int16)
            if board is not None or still is None or np.abs(thumb - still).mean() > 1.0:
                still, still_since = thumb, now
            elif now - still_since >= HUNG_S:
                log("WARN", f"the screen hasn't changed for {HUNG_S:.0f}s despite taps: game hung")
                return None, None
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
        if getattr(self.device, "game_missing", lambda: False)():
            return  # on the home screen a blind tap opens apps; the watcher relaunches
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
        """Hold while a popup or pause is up. False means the level is over.

        held_at: when this call started holding for a popup (None = it didn't)."""
        self.held_at = None
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
            self.held_at = self.held_at or time.monotonic()
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

        No refiring after popups: most swipes near one did land, and re-swiping a word
        the game already took pops a toast. A swipe a popup really ate is caught by the
        repeat pass.

        The "already collected" toast is the exception, because it eats every swipe on
        the rows under it for ~1.5 s (MOTORBIKE, on the bottom row, was lost that way):
        while it's up, swipes there wait at the back of the line, and the ones fired
        there in the moment before the bot saw it are fired again, once.
        """
        todo = deque(hits)
        later: list[Hit] = []  # waiting for the toast to go
        fired: deque[tuple[float, Hit]] = deque(maxlen=40)
        refired: set[Hit] = set()
        cover_handled = self.watcher.cover_since
        n = 0
        while todo or later:
            if not todo:  # only swipes under the toast are left: wait it out
                self._wait_uncovered()
                todo.extend(later)
                later.clear()
            hit = todo.popleft()
            if not self._ready_to_swipe(grid):
                return False
            if self.held_at:
                # The popup was fading in before the watcher (~2 fps) saw it: the swipes
                # fired just before may have been eaten (PERFUME, EARRING under "claim
                # bonus"). The retry pass fires them again if they're still unlit; firing
                # them right away re-swiped taken bonus words into a cascade of toasts.
                self.maybe_eaten.update(h for t, h in fired if t >= self.held_at - POPUP_EATS_S)
            span = self.watcher.covering()
            if span and _under(board, hit, span):
                later.append(hit)
                continue
            if later and not span:
                todo.appendleft(hit)
                todo.extendleft(reversed(later))
                later.clear()
                continue
            if check_every and n and n % check_every == 0:
                self._check_input(board, self._lit_cells(board))
            n += 1
            self._hold()
            if self.stop_event.is_set():
                return False
            t = time.monotonic()
            ok = self.device.swipe(
                board.cell(*hit.start).center, board.cell(*hit.end).center, ms, why=hit.word
            )
            if ok:
                self.stats.swipes += 1
                self.fired_cells.update(path_cells(hit))
                self.swiped.add(hit.word)
                fired.append((t, hit))
                self.pass_fired.append(hit)
            since = self.watcher.cover_since
            if since != cover_handled:
                cover_handled = since
                span = self.watcher.cover_span
                eaten = [
                    h
                    for t, h in fired
                    if t >= since - TOAST_EATS_S and h not in refired and _under(board, h, span)
                ]
                if eaten:
                    refired.update(eaten)
                    later.extend(eaten)
                    log("PAUSE", f"toast over the board; will re-swipe {len(eaten)} under it")
            if gap:
                time.sleep(gap)
        return True

    def _wait_uncovered(self, timeout: float = 4.0) -> None:
        deadline = time.monotonic() + timeout
        while self.watcher.covering() and time.monotonic() < deadline:
            if self.stop_event.wait(0.05):
                return

    # ---- level solving ---------------------------------------------------------

    def solve_level(self, board: Board, grid: list[str]) -> bool:
        """Escalating passes (see module docstring). True once the level ends."""
        hits = self.words.solve(grid)
        seen: set[str] = set()
        fast = [h for h in hits if not (h.word in seen or seen.add(h.word))]
        # Words lying inside a longer one go last. The game ignores a swipe on cells it
        # is still animating from a word it just took, for about a second, and those
        # sub-words are bonus words it takes: ART fired before ARTICLE ate ARTICLE twice
        # (ADA -> CICADA, ORC -> ORCHARD, REG -> CHARGER). Not longest-first overall:
        # theme words back to back lost DRAGONFLY and WHEAT.
        fast = _subwords_last(fast)
        if self._swiped_grid is None or not same_level(grid, self._swiped_grid):
            self.swiped, self._swiped_grid = set(), grid  # kept across restarts of a level

        def new(todo: list[Hit]) -> list[Hit]:
            done = set(self.swiped)  # snapshot: repeated non-words within a pass are fine
            return [h for h in todo if h.word not in done]

        self.maybe_eaten: set[Hit] = set()  # fired just before a popup (see burst)
        common = [h for h in fast if h.rank < RETRY_RANK or len(h.word) >= RETRY_SHORT_LEN]
        long_words = [h for h in fast if len(h.word) >= RETRY_SHORT_LEN]

        def open_long(hits: list[Hit]) -> list[Hit]:
            todo = self._unlit(board, hits)
            return [h for h in todo if h.rank < RETRY_RANK or _open(h, self.found_cells)]

        def retry() -> list[Hit]:
            # Re-swiping taken bonus words pops toasts over the bottom rows: fire the
            # words there first, before the first toast can hold them up.
            first = sorted(self.maybe_eaten)
            rest = open_long([h for h in common if h not in self.maybe_eaten])
            todo = self._unlit(board, first) + rest
            low = board.rows - 2
            return sorted(_subwords_last(todo), key=lambda h: max(h.start[0], h.end[0]) < low)

        p = self.pacing
        passes = [
            ("fast", lambda: new(self._unlit(board, fast)), p.swipe_ms, p.gap_s, 0),
            # not filtered by new(): these are the swiped words the game didn't take
            ("retry", retry, p.swipe_ms, p.retry_gap_s, 0),
            # still unlit after two tries: one slow, spaced-out try before the long
            # exhaustive pass (a few swipes, a few seconds)
            (
                "careful",
                lambda: [h for h in self._unlit(board, long_words) if _open(h, self.found_cells)],
                p.refire_swipe_ms,
                p.careful_gap_s,
                0,
            ),
            (
                "exhaustive",
                lambda: new(self._unlit(board, all_lines(grid), most_unlit_first=True)),
                p.swipe_ms,
                p.gap_s,
                PROGRESS_CHECK_EVERY,
            ),
            # not filtered by new(): a theme word the dictionary lacks gets one shot in
            # the exhaustive pass, and MOTORBIKE was dropped there on 3 tries in a row
            ("slow", lambda: self._open_lines(board, grid), p.refire_swipe_ms, p.refire_gap_s, 0),
            ("repeat", lambda: self._unlit(board, hits), p.refire_swipe_ms, p.refire_gap_s, 0),
        ]
        self._checked_lit = set()
        self._input_blocked = False
        try:
            for name, pick, ms, gap, check_every in passes:
                todo = pick()
                if not todo and name in ("retry", "careful", "slow"):
                    continue
                self.phase = f"{name} ({len(todo)})"
                if name != "fast":
                    log("PASS", f"{name}: {len(todo)} swipes")
                    self._check_input(board, self.found_cells)
                lit_before = set(self.found_cells) if name in LEARN_PASSES else None
                self.pass_fired = []
                finished = not self.burst(board, grid, spread(todo), ms, gap, check_every)
                self._save_swiped()
                over = finished or self._wait_level_end(grid, timeout=p.level_end_s)
                if lit_before is not None:
                    self._learn(board, lit_before, over)
                if over:
                    return True
        except InputBlocked:
            self._dump("input_blocked")
            log("WARN", "the game ignores touches on the board; restarting")
            self._input_blocked = True
        return False

    # ---- learning theme words the wordlist lacks -------------------------------

    def _load_learned(self) -> None:
        try:
            words = self._learned_path.read_text(encoding="utf-8").split()
        except OSError:
            return
        for w in words:
            if w not in self.words.rank:
                self.words.add(w, LEARNED_RANK)
        log("WORDS", f"{len(words)} learned theme words")

    def _learn(self, board: Board, lit_before: set[tuple[int, int]], over: bool) -> None:
        """Remember the lines outside the wordlist that this pass found, so the fast pass
        gets them next time (MOTORBIKE came back on 3 days, each time costing a full
        exhaustive pass). Found = every cell of the line lit up during the pass. The pill
        doesn't show a direction, so a line and its reverse are both learned."""
        if over:
            # The level's last frames with the board up. The very last can be mid-
            # animation (letters flying off read as unlit: "48 -> 6 lit"), and lit cells
            # only grow during a level, so take the one with the most.
            lit = max(
                (
                    {
                        (r, c)
                        for r in range(board.rows)
                        for c in range(board.cols)
                        if highlighted(frame, board, r, c)
                    }
                    for frame in (shot.calib for shot in list(self.watcher.board_frames))
                ),
                key=len,
                default=None,
            )
            if lit is None:
                return
        else:
            lit = self._lit_cells(board)
            if lit is None:
                return
        fresh = lit - lit_before
        g = self.grid
        # In the main log: the cells a late pass lit spell the word it finally found,
        # which tells why the earlier passes missed it.
        log(
            "PASS",
            f"late find: lit {len(lit_before)}->{len(lit)}, new cells "
            f"{' '.join(f'{g[r][c]}{r},{c}' for r, c in sorted(fresh))}",
        )
        # A found word lights exactly its own cells, so the line must cover a whole lit
        # run: a piece of a longer find (MOAR inside a found row) has lit cells beyond an
        # end. And it must not be a known word (either way round: PMET = TEMP) plus one
        # cell another find lit (ELANDING = LANDING + the E of STEP's row).
        keep = [
            h
            for h in self.pass_fired
            if len(h.word) >= LEARN_MIN_LEN
            and not self._known_inside(h.word)
            and not self._known_outside(h)
            and set(path_cells(h)) <= lit
            and sum(c not in fresh for c in path_cells(h)) <= 2
            and _whole_run(h, fresh)
        ]
        # Up to 2 cells may have been lit before: theme words cross found ones
        # (PORCUPINE's E was BULLET's). Then the line minus that end qualifies too
        # (PORCUPIN, NETBAL inside NETBALL): keep only the longest line.
        keep = _drop_ambiguous(keep)
        words = sorted({h.word for h in keep})
        if not words or len(words) > 8:
            return
        # Most of what lit up must be these lines: a level-end flash or a misread frame
        # lights cells everywhere, and learning from that would teach junk.
        covered = {c for h in keep for c in path_cells(h)}
        if len(fresh) > 2 * len(covered):
            dbg(f"learn: skipped {words}: {len(fresh)} cells lit vs {len(covered)}")
            return
        for w in words:
            self.words.add(w, LEARNED_RANK)
        try:
            self._learned_path.parent.mkdir(exist_ok=True)
            with self._learned_path.open("a", encoding="utf-8") as f:
                f.write("".join(w + "\n" for w in words))
        except OSError as exc:
            log("WARN", f"couldn't save learned words: {exc!r}")
        log("WORDS", f"learned {', '.join(words)}")

    def _known_inside(self, word: str) -> bool:
        rank = self.words.rank
        return any(v in rank for w in (word, word[::-1]) for v in (w, w[1:], w[:-1]))

    def _known_outside(self, hit: Hit) -> bool:
        """The line one cell longer at either end is a known word (DRAGONFL: the Y of
        DRAGONFLY was lit before, by a crossing word)."""
        cells = path_cells(hit)
        (r0, c0), (r1, c1) = cells[0], cells[-1]
        dr, dc = (r1 - r0) // (len(cells) - 1), (c1 - c0) // (len(cells) - 1)
        g, rows, cols = self.grid, len(self.grid), len(self.grid[0])
        for r, c, w in (
            (r0 - dr, c0 - dc, lambda x: x + hit.word),
            (r1 + dr, c1 + dc, lambda x: hit.word + x),
        ):
            if 0 <= r < rows and 0 <= c < cols:
                longer = w(g[r][c])
                if longer in self.words.rank or longer[::-1] in self.words.rank:
                    return True
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
        # Once per level. The probe can read IGNORED on a board that takes touches (at
        # level 5382 it did so 3 restarts in a row, every time just before the
        # exhaustive pass that alone could find SCAVENGER; run by hand the same drag
        # registered). A real freeze costs a pass of swipes; a restart loop, the level.
        if self._blocked_grid is not None and same_level(self.grid, self._blocked_grid):
            log("PROBE", "still reads IGNORED after a restart on this level; carrying on")
            return
        self._blocked_grid = self.grid
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
        # A frozen iPhone picture showed every drag as IGNORED and looped restarts.
        deadline = time.monotonic() + 10.0
        while self.device.view_stale() and time.monotonic() < deadline:
            time.sleep(0.2)
        if self.device.view_stale():
            log("PROBE", "screen picture is frozen; can't tell")
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
            if not seen or (not alive and self.device.view_stale()):
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

    def _open_lines(self, board: Board, grid: list[str]) -> list[Hit]:
        """Lines of 4+ letters outside the dictionary lying on unlit cells (one lit cell
        allowed: a theme word may cross a found one), untouched ones first, longest first."""
        lit = self._lit_cells(board)
        if lit is None:
            return []
        self.found_cells = lit
        out = []
        for h in all_lines(grid):
            n_lit = sum(c in lit for c in path_cells(h))
            if len(h.word) >= 4 and n_lit <= 1 and h.word not in self.words.rank:
                out.append((n_lit, -len(h.word), h))
        return [h for *_, h in sorted(out)]

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
        # A board that went away just before the deadline still gets its HIDDEN_END_S:
        # cutting that short started the exhaustive pass on the level-complete screen.
        grace = deadline + HIDDEN_END_S + 1.0
        while not self.stop_event.is_set() and (
            time.monotonic() < deadline or (hidden_since is not None and time.monotonic() < grace)
        ):
            if self.watcher.level_done.is_set():
                return True
            shot = self.watcher.latest_shot(newer_than=self.watcher.frame_time, timeout=1.0)
            if shot is None:
                continue
            now = time.monotonic()
            board = shot.board
            if board is None and self.watcher.covering():
                # The "already collected" toast cuts the board short for ~1.5 s: that
                # counted fake "level done"s, then "the previous level is still on screen".
                hidden_since = None
                continue
            if board is None and now - self.watcher.mid_level_seen < MID_LEVEL_HOLD_S:
                # The bonus jar popup (opened by its tutorial) covers the whole board:
                # that counted a fake "level done" mid-level.
                hidden_since = None
                continue
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
        dbg(
            f"bot run: device={type(self.device).__name__} calib={self.device.calib} "
            f"templates={self.device.templates} popups={[p.name for p in self.watcher.popups]}"
        )
        self.watcher.start()
        previous: list[str] | None = None
        restarts_this_level = 0
        try:
            while not self.stop_event.is_set():
                try:
                    self._hold()
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

                    if self._relaunch:
                        self._relaunch = False
                        self.restart_app("it ignores touches after sitting idle", fresh=True)
                        previous = None
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
                    self.device.last_panel = board.panel
                    snap("level_start", self.watcher.frame, every_s=0, note=f"{board.panel} {grid}")
                    dbg(
                        f"board panel={board.panel} {board.rows}x{board.cols} "
                        f"letter_h={board.letter_h:.1f} calib={self.device.calib} "
                        f"zones={getattr(self.device, 'zones', None)}"
                    )
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
                        # Swipes into a board that ignored touches never counted: fire them
                        # again.
                        self._forget_swiped(
                            everything=restarts_this_level >= 3 or self._input_blocked
                        )
                        self._dump("level_stuck")
                        self.restart_app(
                            f"level stuck after every pass (try {restarts_this_level})"
                        )
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
                except Exception as exc:  # a 10-day run must outlive any one failure
                    if self.stop_event.is_set():
                        break
                    log("ERROR", f"{type(exc).__name__}: {exc}; restarting the game")
                    dbg(f"main loop error: {traceback.format_exc()}")
                    self.stop_event.wait(10)
                    with contextlib.suppress(Exception):
                        self.restart_app("recovering from an error")
                    previous = None
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
        self.watcher.resting.set()
        try:
            while not self.stop_event.is_set() and (left := until - time.time()) > 0:
                self._hold()
                self.stop_event.wait(min(1.0, left))
        finally:
            self.watcher.idle.clear()
            self.watcher.resting.clear()
            self.resting_until = None
            if seconds >= IDLE_RELAUNCH_S:
                self._relaunch = True

    def _hold(self) -> None:
        """While AutomationHQ has paused the bot, wait here; then carry on from here."""
        if not self.pause_event.is_set() or self.stop_event.is_set():
            return
        log("AHQ", "paused")
        status, self.status = self.status, "paused"
        self.watcher.idle.set()
        since = time.monotonic()
        try:
            while self.pause_event.is_set() and not self.stop_event.is_set():
                self.stop_event.wait(0.5)
        finally:
            self.watcher.idle.clear()
            self.status = status
        if time.monotonic() - since >= IDLE_RELAUNCH_S:
            self._relaunch = True  # the game ignores touches after sitting idle
        if not self.stop_event.is_set():
            log("AHQ", "resumed")

    def restart_app(self, why: str, *, fresh: bool = False) -> None:
        """fresh: a routine relaunch, not a recovery (not counted as a restart)."""
        if fresh:
            log("LOG", f"relaunching the game: {why}")
        else:
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
        self._relaunch = True

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
        kind = why.partition("_r")[0]  # unreadable_r3c4 -> unreadable
        now = time.monotonic()
        if now - self._dumped.get(kind, -DUMP_EVERY_S) < DUMP_EVERY_S:
            return
        self._dumped[kind] = now
        frame = self.watcher.frame
        if frame is None:
            return
        path = self.diagnostics / f"{why}_{time.strftime('%Y%m%d_%H%M%S')}.png"
        imwrite(path, frame)
        log("DIAG", f"saved {path.name}")
        dbg(
            f"dump {why}: watcher hits={dict(self.watcher.hits)} "
            f"board_visible={self.watcher.board_visible} fps={self.watcher.fps:.1f}"
        )
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


def _under(board: Board, hit: Hit, span: tuple[int, int]) -> bool:
    """True if any cell of the swipe lies in the y range a toast covers."""
    y0, y1 = span
    return any(y0 <= board.cell(*c).center[1] <= y1 for c in path_cells(hit))


def spread(hits: list[Hit], recent: int = 3, window: int = 24) -> list[Hit]:
    """Same swipes, reordered so none shares a cell with the last `recent` ones.

    Rare words sit in the wordlist alphabetically, so a word used to fire right after
    its own prefix on the same cells (ORC -> ORCHARD, BAH -> BAHT), and the exhaustive
    pass fires a line right after its reverse (EKIBROTOM -> MOTORBIKE). The game drops
    some swipes that land on cells it is still animating from the swipe before. Looks
    only `window` ahead, so the order stays close to best-first.
    """
    todo = list(hits)
    out: list[Hit] = []
    busy: deque[set[tuple[int, int]]] = deque(maxlen=recent)
    while todo:
        pick = 0
        for i, h in enumerate(todo[:window]):
            cells = set(path_cells(h))
            if not any(cells & b for b in busy):
                pick = i
                break
        h = todo.pop(pick)
        out.append(h)
        busy.append(set(path_cells(h)))
    return out


def _subwords_last(hits: list[Hit]) -> list[Hit]:
    """Same order, except hits whose cells all lie inside another hit's go to the end."""
    cells = [set(path_cells(h)) for h in hits]
    inside = [any(c < o for o in cells) for c in cells]
    return [h for h, i in zip(hits, inside, strict=True) if not i] + [
        h for h, i in zip(hits, inside, strict=True) if i
    ]


def _drop_ambiguous(hits: list[Hit]) -> list[Hit]:
    """Learning candidates minus lines inside a longer one (NETBAL in NETBALL). Two that
    only partly overlap (ITIND, TINDY around 3 new cells) can't both be the find, and
    which one is can't be told: learn neither."""
    cells = [frozenset(path_cells(h)) for h in hits]
    keep = [h for h, c in zip(hits, cells, strict=True) if not any(c < o for o in cells)]
    kept = {frozenset(path_cells(h)) for h in keep}
    if any(a != b and a & b for a in kept for b in kept):
        dbg(f"learn: skipped {sorted(h.word for h in keep)}: overlapping lines")
        return []
    return keep


def _mostly_unlit(hit: Hit, lit: set[tuple[int, int]]) -> bool:
    """At least half the hit's cells, and 2 or more, are not yet on a found word."""
    cells = path_cells(hit)
    n = sum(c not in lit for c in cells)
    return n >= 2 and 2 * n >= len(cells)


def _open(hit: Hit, lit: set[tuple[int, int]]) -> bool:
    """A rare word worth re-firing: 5+ letters mostly unlit, or 4 letters all unlit."""
    if len(hit.word) >= RETRY_MIN_LEN:
        return _mostly_unlit(hit, lit)
    return len(hit.word) == RETRY_SHORT_LEN and not any(c in lit for c in path_cells(hit))


def _whole_run(hit: Hit, lit: set[tuple[int, int]]) -> bool:
    """The cells just before the start and just past the end (along the line) are unlit."""
    cells = path_cells(hit)
    (r0, c0), (r1, c1) = cells[0], cells[-1]
    dr, dc = (r1 - r0) // (len(cells) - 1), (c1 - c0) // (len(cells) - 1)
    return (r0 - dr, c0 - dc) not in lit and (r1 + dr, c1 + dc) not in lit


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
