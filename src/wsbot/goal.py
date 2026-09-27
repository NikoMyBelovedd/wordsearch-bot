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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schedule import Schedule

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


def local_file(root: Path, serial: str, name: str) -> Path:
    """Per-platform state file: an iPhone plays its own game account, so its level
    count and swiped words must never mix with the Android ones."""
    stem, dot, ext = name.partition(".")
    if serial == "ios" or serial.startswith("ios:"):
        stem += "-" + serial.replace(":", "-")
    return root / "local" / f"{stem}{dot}{ext}"


class Progress:
    """Thread-safe, crash-safe level counters persisted as JSON."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.data: dict = {"days": {}, "plans": {}, "all_time": 0}
        if path.exists():
            try:
                self.data.update(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                pass  # a corrupt file must never stop the bot; start counting afresh

    def day(self, date: str | None = None) -> int:
        return self.data["days"].get(date or today(), 0)

    def plan(self, plan_id: str) -> int:
        return self.data["plans"].get(plan_id, 0)

    def meta(self, key: str):
        return self.data.get("meta", {}).get(key)

    def set_meta(self, key: str, value) -> None:
        with self.lock:
            self.data.setdefault("meta", {})[key] = value
            self._save()

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
        tmp.write_text(json.dumps(self.data, indent=1), encoding="utf-8")
        tmp.replace(self.path)  # atomic: a crash mid-write never corrupts progress


@dataclass
class Goal:
    """What to play. `per_day`/`total` of None mean unlimited."""

    plan_id: str
    title: str
    per_day: int | None
    total: int | None
    progress: Progress
    schedule: Schedule | None = None  # None = no pacing (single level)
    session_levels: int = 0
    session_target: int | None = None  # stop after this many levels this run (--levels)

    @classmethod
    def play(cls, progress: Progress) -> Goal:
        from .schedule import PLAY_SCHEDULE

        return cls(
            "play14k",
            f"Play · {PLAY_TOTAL:,} levels in {PLAY_DAYS} days",
            PLAY_PER_DAY,
            PLAY_TOTAL,
            progress,
            PLAY_SCHEDULE,
        )

    @classmethod
    def custom(cls, progress: Progress, schedule: Schedule) -> Goal:
        schedule.per_day = max(CUSTOM_MIN, min(CUSTOM_MAX, schedule.per_day))
        title = f"Custom · {schedule.per_day:,} levels/day"
        return cls("custom", title, schedule.per_day, None, progress, schedule)

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
        if self.session_target is not None and self.session_levels >= self.session_target:
            return True
        return self.total is not None and self.done_total >= self.total

    def quota_reached_today(self) -> bool:
        return self.per_day is not None and self.done_today >= self.per_day

    def record_level(self) -> None:
        self.session_levels += 1
        self.progress.record(self.plan_id)
