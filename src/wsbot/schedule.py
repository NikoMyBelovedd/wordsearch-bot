"""Human-like pacing: spread each day's levels over a play window, with breaks.

A day's plan (when the window opens and closes) is drawn once per day and saved in
the progress file, so a restart keeps the same day. Between levels the bot idles
just long enough to land the day's quota at the window's end; idles are jittered and
shrink to zero when it's behind (a restart, a slow level), so it catches up.
"""

from __future__ import annotations

import datetime as dt
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .goal import Progress, today

END_MARGIN_S = 15 * 60  # the window always closes this long before midnight
MAX_IDLE_S = 8 * 60  # never sit longer than this between two levels (breaks aside)


@dataclass
class Schedule:
    """How a day's levels are spread out. Durations in hours/minutes, as the UI shows them."""

    per_day: int = 1_400
    hours_min: float = 10.0  # the play window's length is drawn from this range each day
    hours_max: float = 16.0  # 0 = as fast as possible (no idling between levels)
    break_every_min: int = 60  # minutes of play between breaks, drawn from this range
    break_every_max: int = 120  # 0 = no breaks
    break_len_min: int = 25  # break length range, minutes
    break_len_max: int = 35
    day_start: float | None = 9.0  # hour the window opens (jittered); None = when started
    start_jitter_min: int = 45

    @property
    def paced(self) -> bool:
        return self.hours_max > 0

    @property
    def breaks(self) -> bool:
        return self.break_every_max > 0 and self.break_len_max > 0

    def summary(self) -> str:
        spread = (
            f"over {self.hours_min:g}-{self.hours_max:g} h" if self.paced else "as fast as possible"
        )
        pause = (
            f"{self.break_len_min}-{self.break_len_max} min break every "
            f"{self.break_every_min}-{self.break_every_max} min"
            if self.breaks
            else "no breaks"
        )
        return f"{self.per_day:,}/day {spread} · {pause}"


PLAY_SCHEDULE = Schedule()


def load_custom(path: Path) -> Schedule:
    try:
        return Schedule(**json.loads(path.read_text(encoding="utf-8"))["custom"])
    except (OSError, ValueError, KeyError, TypeError):
        return Schedule()


def save_custom(path: Path, schedule: Schedule) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"custom": asdict(schedule)}, indent=1), encoding="utf-8")


def _at_hour(day: dt.date, hour: float) -> float:
    return dt.datetime.combine(day, dt.time()).timestamp() + hour * 3600


class Pacer:
    """Decides how long to rest between levels and when to take breaks."""

    def __init__(self, schedule: Schedule, progress: Progress, *, rng: random.Random | None = None):
        self.s = schedule
        self.progress = progress
        self.rng = rng or random.Random()
        self.last_break_end = time.time()
        self.next_break_after = self._draw_break_gap()

    # ---- the day's window -------------------------------------------------------

    def window(self) -> tuple[float, float]:
        """(start, end) epoch seconds of today's play window, drawn once per day."""
        saved = self.progress.meta("window")
        if saved and saved.get("date") == today():
            return saved["start"], saved["end"]
        now = time.time()
        d = dt.date.today()
        midnight = _at_hour(d, 24)
        start = now
        if self.s.day_start is not None:
            jitter = self.rng.uniform(-1, 1) * self.s.start_jitter_min * 60
            start = max(now, _at_hour(d, self.s.day_start) + jitter)
        length = self.rng.uniform(self.s.hours_min, self.s.hours_max) * 3600
        end = min(start + length, midnight - END_MARGIN_S)
        end = max(end, start)  # started late: a zero-length window just means "no idling"
        self.progress.set_meta("window", {"date": today(), "start": start, "end": end})
        return start, end

    def wait_before_start(self) -> float:
        """Seconds until today's window opens (0 if it's open)."""
        if not self.s.paced:
            return 0.0
        start, _ = self.window()
        return max(0.0, start - time.time())

    # ---- between levels ----------------------------------------------------------

    def idle_after_level(self, level_s: float) -> float:
        """Seconds to rest after a level so the quota lands at the window's end."""
        if not self.s.paced:
            return 0.0
        left = self.s.per_day - self.progress.day()
        if left <= 0:
            return 0.0
        _, end = self.window()
        now = time.time()
        time_left = end - now
        if self.s.breaks:  # leave room for the breaks still to come
            mean_gap = (self.s.break_every_min + self.s.break_every_max) / 2 * 60
            mean_len = (self.s.break_len_min + self.s.break_len_max) / 2 * 60
            time_left -= int(max(0.0, time_left) // (mean_gap + mean_len)) * mean_len
        interval = time_left / left  # wall time per level to finish right on time
        idle = (interval - level_s) * self.rng.uniform(0.6, 1.4)
        return max(0.0, min(idle, MAX_IDLE_S))

    def break_due(self) -> float:
        """Seconds of break to take now (0 = keep playing)."""
        if not self.s.breaks or self.progress.day() >= self.s.per_day:
            return 0.0
        if time.time() - self.last_break_end < self.next_break_after:
            return 0.0
        return self.rng.uniform(self.s.break_len_min, self.s.break_len_max) * 60

    def break_taken(self) -> None:
        self.last_break_end = time.time()
        self.next_break_after = self._draw_break_gap()

    def next_break_at(self) -> float | None:
        if not self.s.breaks:
            return None
        return self.last_break_end + self.next_break_after

    def _draw_break_gap(self) -> float:
        return self.rng.uniform(self.s.break_every_min, max(1, self.s.break_every_max)) * 60
