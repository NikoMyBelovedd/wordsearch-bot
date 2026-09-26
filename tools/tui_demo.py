"""Run the TUI against a fake bot: no device, no adb, nothing tapped.

uv run python tools/tui_demo.py            # interactive demo
"""

from __future__ import annotations

import random
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from wsbot.goal import Goal
from wsbot.log import log
from wsbot.tui import AdbDevice, WordsearchApp

LETTERS = "ABCDEFGHIKLMNOPRSTUW"


class FakeStats:
    def __init__(self) -> None:
        self.started = time.monotonic()
        self.levels = 0
        self.swipes = 0
        self.restarts = 0
        self.times: list[float] = []

    def avg_level_s(self) -> float:
        return sum(self.times) / len(self.times) if self.times else 0.0


class FakeBot:
    """Quacks like wsbot.bot.Bot for the UI."""

    def __init__(self, serial: str, goal: Goal, level_s: float = 1.2) -> None:
        self.goal = goal
        self.level_s = level_s
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.status, self.phase = "starting", ""
        self.grid: list[str] = []
        self.fired_cells: set[tuple[int, int]] = set()
        self.found_cells: set[tuple[int, int]] = set()
        self.stats = FakeStats()
        self.watcher = SimpleNamespace(fps=4.7, hits=Counter())
        self.device = SimpleNamespace(refused=0, taps=0, swipes=0)
        log("DEVICE", f"connected {serial} (fake)")

    def run(self) -> None:
        rng = random.Random(7)
        while not self.stop_event.is_set() and not self.goal.finished():
            if self.goal.quota_reached_today():
                self.status = "daily target reached · resumes at midnight"
                self.stop_event.wait(0.2)
                continue
            n = rng.choice([5, 6, 7, 8])
            self.grid = ["".join(rng.choice(LETTERS) for _ in range(n)) for _ in range(n)]
            self.fired_cells, self.found_cells = set(), set()
            log("LEVEL", f"#{self.goal.done_total + 1} {n}x{n} {'/'.join(self.grid)}")
            self.status, self.phase = "solving", "fast (40)"
            t0 = time.monotonic()
            for _ in range(12):
                while self.pause_event.is_set() and not self.stop_event.is_set():
                    self.status = "paused"
                    time.sleep(0.05)
                if self.stop_event.is_set():
                    return
                self.status = "solving"
                r, c = rng.randrange(n), rng.randrange(n)
                self.fired_cells.add((r, c))
                if rng.random() < 0.4:
                    self.found_cells.add((r, c))
                self.stats.swipes += 1
                log("SWIPE", "".join(rng.choice(LETTERS) for _ in range(4)))
                time.sleep(self.level_s / 12)
            self.watcher.hits["next_level"] += 1
            log("WATCHER", "next_level score=1.00 at (540, 1754)")
            self.stats.levels += 1
            self.stats.times.append(time.monotonic() - t0)
            self.goal.record_level()
            log("LEVEL", f"done in {time.monotonic() - t0:.1f}s · today {self.goal.done_today}")
        self.status = "stopped"


def fake_devices() -> list[AdbDevice]:
    return [
        AdbDevice("emulator-5554", "device", "sdk gphone16k x86 64"),
        AdbDevice("R58M123ABC", "unauthorized", ""),
        AdbDevice("192.168.1.20:5555", "offline", "Pixel 7"),
    ]


def make_app(root: Path, level_s: float = 1.2) -> WordsearchApp:
    return WordsearchApp(
        "emulator-5554",
        root,
        bot_factory=lambda serial, goal: FakeBot(serial, goal, level_s),
        device_lister=fake_devices,
    )


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="wsbot-demo-"))
    make_app(root).run()
