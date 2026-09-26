"""Run goals: how many levels to play and how to spread them over days.

Progress lives in local/progress.json so a crash, reboot or restart picks up the
daily count and plan total exactly where they were. Days roll over at local midnight.
"""

from __future__ import annotations

import datetime as dt
import json
import threading
from dataclasses import dataclass
from pathlib import Path

PLAY_TOTAL = 14_000
PLAY_DAYS = 10
PLAY_PER_DAY = PLAY_TOTAL // PLAY_DAYS  # 1,400
CUSTOM_MIN, CUSTOM_MAX, CUSTOM_STEP = 50, 5_000, 50
CUSTOM_DEFAULT = 1_000


def today() -> str:
    return dt.date.today().isoformat()


def seconds_until_midnight() -> float:
    now = dt.datetime.now()
    midnight = dt.datetime.combine(now.date() + dt.timedelta(days=1), dt.time())
    return (midnight - now).total_seconds()


class Progress:
    """Thread-safe, crash-safe level counters persisted as JSON."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.data: dict = {"days": {}, "plans": {}, "all_time": 0}
        if path.exists():
            try:
                self.data.update(json.loads(path.read_text()))
            except (OSError, ValueError):
                pass  # a corrupt file must never stop the bot; start counting afresh

    def day(self, date: str | None = None) -> int:
        return self.data["days"].get(date or today(), 0)

    def plan(self, plan_id: str) -> int:
        return self.data["plans"].get(plan_id, 0)

    def record(self, plan_id: str) -> None:
        with self.lock:
            d = today()
            self.data["days"][d] = self.data["days"].get(d, 0) + 1
            self.data["plans"][plan_id] = self.data["plans"].get(plan_id, 0) + 1
            self.data["all_time"] = self.data.get("all_time", 0) + 1
            self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1))
        tmp.replace(self.path)  # atomic: a crash mid-write never corrupts progress


@dataclass
class Goal:
    """What to play. `per_day`/`total` of None mean unlimited."""

    plan_id: str
    title: str
    per_day: int | None
    total: int | None
    progress: Progress
    session_levels: int = 0

    @classmethod
    def play(cls, progress: Progress) -> Goal:
        return cls(
            "play14k",
            f"Play · {PLAY_TOTAL:,} levels in {PLAY_DAYS} days",
            PLAY_PER_DAY,
            PLAY_TOTAL,
            progress,
        )

    @classmethod
    def custom(cls, progress: Progress, per_day: int) -> Goal:
        per_day = max(CUSTOM_MIN, min(CUSTOM_MAX, per_day))
        return cls("custom", f"Custom · {per_day:,} levels/day", per_day, None, progress)

    @classmethod
    def single(cls, progress: Progress) -> Goal:
        return cls("single", "Single level", None, None, progress)

    # ---- queries used by the bot and the UI -----------------------------------

    @property
    def done_today(self) -> int:
        return self.progress.day()

    @property
    def done_total(self) -> int:
        return self.progress.plan(self.plan_id)

    def finished(self) -> bool:
        if self.plan_id == "single":
            return self.session_levels >= 1
        return self.total is not None and self.done_total >= self.total

    def quota_reached_today(self) -> bool:
        return self.per_day is not None and self.done_today >= self.per_day

    def record_level(self) -> None:
        self.session_levels += 1
        self.progress.record(self.plan_id)
