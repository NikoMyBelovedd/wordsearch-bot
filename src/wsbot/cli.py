"""Command-line entrypoint: `wsbot` (TUI) or `wsbot --headless`."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SERIAL = "emulator-5554"


def main() -> None:
    p = argparse.ArgumentParser(prog="wsbot", description="Word Search Explorer auto-solver")
    p.add_argument("--serial", help="ADB serial (default: $WSBOT_SERIAL or emulator-5554)")
    p.add_argument("--headless", action="store_true", help="run without the TUI")
    p.add_argument(
        "--mode", choices=["play", "custom", "single"], default="play", help="--headless goal"
    )
    p.add_argument("--per-day", type=int, default=1000, help="--mode custom: levels per day")
    p.add_argument("--dry-run", action="store_true", help="decide everything, touch nothing")
    p.add_argument("--diagnose", action="store_true", help="score every template on this frame")
    p.add_argument("--calibrate", action="store_true", help="save an annotated overlay")
    p.add_argument("--capture", metavar="NAME", help="save a popup template from the screen")
    p.add_argument("--crop", metavar="X,Y,W,H", help="crop box for --capture (calib space)")
    p.add_argument("--level-done", action="store_true", help="--capture: marks level end")
    args = p.parse_args()
    serial = args.serial or os.environ.get("WSBOT_SERIAL") or DEFAULT_SERIAL

    if args.capture:
        return capture(serial, args.capture, args.crop, args.level_done)
    if args.diagnose or args.calibrate:
        from .diagnostics import run as diag

        return diag(serial, ROOT, overlay=args.calibrate)
    if args.headless:
        return headless(serial, args.mode, args.per_day, args.dry_run)

    from .tui import run_tui

    run_tui(serial, ROOT, dry_run=args.dry_run)


def make_goal(mode: str, per_day: int):
    from .goal import Goal, Progress

    progress = Progress(ROOT / "local" / "progress.json")
    if mode == "custom":
        return Goal.custom(progress, per_day)
    if mode == "single":
        return Goal.single(progress)
    return Goal.play(progress)


def headless(serial: str, mode: str, per_day: int, dry_run: bool) -> None:
    from .bot import Bot
    from .log import file_sink, set_sinks, stdout_sink

    (ROOT / "local").mkdir(exist_ok=True)
    set_sinks(stdout_sink, file_sink(ROOT / "local" / "wsbot.log"))
    bot = Bot(serial, ROOT, make_goal(mode, per_day), dry_run=dry_run)
    signal.signal(signal.SIGINT, lambda *_: bot.stop_event.set())
    bot.run()


def capture(serial: str, name: str, crop: str | None, level_done: bool) -> None:
    import cv2

    from .device import Device

    frame = Device(serial).frame()
    if not crop:
        out = ROOT / "diagnostics" / f"capture_{name}.png"
        cv2.imwrite(str(out), frame)
        print(f"saved full frame to {out}; pick a box INSIDE the button and rerun with --crop")
        return
    x, y, w, h = (int(v) for v in crop.split(","))
    folder = ROOT / "templates"
    cv2.imwrite(str(folder / "popups" / f"{name}.png"), frame[y : y + h, x : x + w])
    registry = folder / "popups.json"
    entries = json.loads(registry.read_text()) if registry.exists() else []
    entries = [e for e in entries if e["name"] != name]
    entry = {"name": name, "file": f"{name}.png", "threshold": 0.85, "cooldown": 1.5}
    if level_done:
        entry["level_done"] = True
    entries.append(entry)
    registry.write_text(json.dumps(entries, indent=2) + "\n")
    print(f"registered popup '{name}' ({w}x{h})")


if __name__ == "__main__":
    sys.exit(main())
