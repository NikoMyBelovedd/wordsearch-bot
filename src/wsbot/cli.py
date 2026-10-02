"""Command-line entrypoint: `wsbot` (TUI) or `wsbot --headless`."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SERIAL = "ios"  # the iPhone over USB; --serial picks an adb device


def _one_thread_each() -> None:
    """A farm runs one bot per phone: OpenCV's per-call worker threads (one per core,
    in every bot) only add switching. One each unless WSBOT_THREADS says otherwise."""
    import cv2

    cv2.setNumThreads(int(os.environ.get("WSBOT_THREADS") or 1))


def _memory_report() -> None:
    """local/memreport (a file) or $WSBOT_MEMREPORT: log where Python memory goes,
    2 and 10 minutes in (tracemalloc; slows the bot a little, so only on request)."""
    if not (os.environ.get("WSBOT_MEMREPORT") or (ROOT / "local" / "memreport").exists()):
        return
    import threading
    import time
    import tracemalloc

    tracemalloc.start(3)

    def report() -> None:
        from .log import log

        for wait in (120, 480):
            time.sleep(wait)
            snap = tracemalloc.take_snapshot()
            total = sum(s.size for s in snap.statistics("filename"))
            log("DIAG", f"memreport: python-tracked {total / 1e6:.0f} MB")
            for stat in snap.statistics("traceback")[:20]:
                where = " <- ".join(
                    f"{f.filename.rsplit(os.sep, 1)[-1]}:{f.lineno}" for f in stat.traceback
                )
                log("DIAG", f"memreport: {stat.size / 1e6:6.1f} MB x{stat.count} {where}")

    threading.Thread(target=report, daemon=True, name="memreport").start()


def _experiments() -> None:
    """local/stream_pace (a number in the file): WSBOT_STREAM_PACE for testing."""
    f = ROOT / "local" / "stream_pace"
    if f.exists() and not os.environ.get("WSBOT_STREAM_PACE"):
        os.environ["WSBOT_STREAM_PACE"] = f.read_text(encoding="utf-8").strip() or "10"


def main() -> None:
    _experiments()
    from .slim import slim

    slim()
    _one_thread_each()
    _memory_report()
    p = argparse.ArgumentParser(prog="wsbot", description="Word Search Explorer auto-solver")
    p.add_argument(
        "--serial",
        help="ADB serial, or ios / ios:UDID for an iPhone over USB (default: $WSBOT_SERIAL or ios)",
    )
    p.add_argument("--headless", action="store_true", help="run without the TUI")
    p.add_argument(
        "--mode", choices=["play", "custom", "single"], default="play", help="--headless goal"
    )
    p.add_argument(
        "--per-day", type=int, help="--mode custom: levels per day (default: saved setting)"
    )
    p.add_argument("--fast", action="store_true", help="no idling or breaks (testing)")
    p.add_argument("--levels", type=int, help="stop after this many levels")
    p.add_argument("--dry-run", action="store_true", help="decide everything, touch nothing")
    p.add_argument("--diagnose", action="store_true", help="score every template on this frame")
    p.add_argument("--calibrate", action="store_true", help="save an annotated overlay")
    p.add_argument("--capture", metavar="NAME", help="save a popup template from the screen")
    p.add_argument("--crop", metavar="X,Y,W,H", help="crop box for --capture (calib space)")
    p.add_argument("--level-done", action="store_true", help="--capture: marks level end")
    p.add_argument("--report", action="store_true", help="zip logs + screenshots to send back")
    p.add_argument(
        "--debug", action="store_true", help="detailed logs + snapshots for --report ($WSBOT_DEBUG)"
    )
    args = p.parse_args()
    serial = args.serial or os.environ.get("WSBOT_SERIAL") or DEFAULT_SERIAL
    if args.report:
        from .debug import make_report

        return make_report(ROOT)
    if args.debug or os.environ.get("WSBOT_DEBUG"):
        from .debug import setup

        setup(ROOT)
    if args.capture:
        return capture(serial, args.capture, args.crop, args.level_done)
    if args.diagnose or args.calibrate:
        from .diagnostics import run as diag

        return diag(serial, ROOT, overlay=args.calibrate)
    if args.headless:
        try:
            return headless(serial, args.mode, args.per_day, args.dry_run, args.fast, args.levels)
        except KeyboardInterrupt:  # stopped before the bot was up (its SIGINT handler)
            print("stopped")
            return None

    from .tui import run_tui

    run_tui(serial, ROOT, dry_run=args.dry_run)


def make_goal(mode: str, per_day: int | None, serial: str, fast: bool = False):
    from dataclasses import replace

    from .goal import Goal, Progress, local_file
    from .schedule import load_custom

    progress = Progress(local_file(ROOT, serial, "progress.json"))
    if mode == "single":
        return Goal.single(progress)
    if mode == "custom":
        schedule = load_custom(ROOT / "local" / "settings.json")
        if per_day:
            schedule.per_day = per_day
        goal = Goal.custom(progress, schedule)
    else:
        goal = Goal.play(progress)
    if fast and goal.schedule:
        goal.schedule = replace(goal.schedule, hours_min=0, hours_max=0, break_every_max=0)
    return goal


def headless(
    serial: str, mode: str, per_day: int | None, dry_run: bool, fast: bool, levels: int | None
) -> None:
    from .bot import Bot
    from .log import file_sink, set_sinks, stdout_sink

    (ROOT / "local").mkdir(exist_ok=True)
    set_sinks(stdout_sink, file_sink(ROOT / "local" / "wsbot.log"))
    goal = make_goal(mode, per_day, serial, fast)
    goal.session_target = levels
    bot = Bot(serial, ROOT, goal, dry_run=dry_run)
    signal.signal(signal.SIGINT, lambda *_: bot.stop_event.set())
    from .control import listen

    listen(bot.pause_event)
    bot.run()


def capture(serial: str, name: str, crop: str | None, level_done: bool) -> None:
    from .device import open_device
    from .imgio import imwrite

    device = open_device(serial)
    frame = device.frame()
    if not crop:
        (ROOT / "diagnostics").mkdir(exist_ok=True)
        out = ROOT / "diagnostics" / f"capture_{name}.png"
        imwrite(out, frame)
        print(f"saved full frame to {out}; pick a box INSIDE the button and rerun with --crop")
        return
    x, y, w, h = (int(v) for v in crop.split(","))
    folder = ROOT / device.templates
    imwrite(folder / "popups" / f"{name}.png", frame[y : y + h, x : x + w])
    registry = folder / "popups.json"
    entries = json.loads(registry.read_text(encoding="utf-8")) if registry.exists() else []
    entries = [e for e in entries if e["name"] != name]
    entry = {"name": name, "file": f"{name}.png", "threshold": 0.85, "cooldown": 1.5}
    if level_done:
        entry["level_done"] = True
    entries.append(entry)
    registry.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    print(f"registered popup '{name}' ({w}x{h})")


if __name__ == "__main__":
    sys.exit(main())
