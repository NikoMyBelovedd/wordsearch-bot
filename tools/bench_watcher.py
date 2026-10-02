"""Time the popup watcher's per-frame work on saved screenshots, and record what it
decides (board visible, panel, every popup's score and spot) so a faster version can
be checked against the old one frame by frame.

    uv run python tools/bench_watcher.py <frames dir> [--templates templates/ios]
        [--native 750x1334] [--out decisions.json] [--threads 1] [--repeat 3]

Frames are calibration-space screenshots (the diagnostics/*.png the bot saves).
--native shrinks each to the phone's real size first, the way an iPhone frame arrives.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wsbot import board, watcher


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", type=Path)
    ap.add_argument("--templates", type=Path, default=Path("templates/ios"))
    ap.add_argument("--native", default="")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=3)
    a = ap.parse_args()
    cv2.setNumThreads(a.threads)

    paths = sorted(p for p in a.frames.iterdir() if p.suffix in (".png", ".jpg"))
    frames = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        calib = (img.shape[1], img.shape[0])
        native = img
        if a.native:
            w, h = (int(v) for v in a.native.split("x"))
            native = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        frames.append((p.name, calib, native))
    popups = watcher.load_popups(a.templates)
    shot = getattr(watcher, "Shot", None)

    timings: dict[str, list[float]] = {}
    decisions = {}

    def timed(key: str, fn):
        t0 = time.perf_counter()
        out = fn()
        timings.setdefault(key, []).append((time.perf_counter() - t0) * 1000)
        return out

    for _ in range(a.repeat):
        for name, calib, native in frames:
            t0 = time.perf_counter()
            if shot is not None:  # the new pipeline: lazy sizes from the native frame
                s = shot(1, native, calib)
                small = timed("small", lambda s=s: s.small)
                coarse = timed("coarse", lambda s=s: (s.coarse, s.coarse_color))
                panel = timed("find_panel", lambda s=s: s.panel)
                b = timed("read_board", lambda s=s: s.board)
                vis = b is not None
            else:
                full = timed(
                    "to_calib",
                    lambda n=native, c=calib: cv2.resize(n, c, interpolation=cv2.INTER_LINEAR),
                )
                b = timed("read_board", lambda f=full: board.read_board(f))
                vis = b is not None
                panel = timed("find_panel", lambda f=full: board.find_panel(f))
                small = timed(
                    "small",
                    lambda f=full: cv2.resize(
                        f, None, fx=watcher.SCALE, fy=watcher.SCALE, interpolation=cv2.INTER_AREA
                    ),
                )
                coarse = timed("coarse", lambda s=small: (None, watcher.coarse_frame(s)))
            scores = {}
            for p in popups:
                sc, center = timed(
                    "match",
                    lambda p=p, s=small, c=coarse: watcher.match(
                        s, p, c[0] if getattr(p, "coarse_gray", False) else c[1]
                    ),
                )
                scores[p.name] = [round(sc, 4), list(center)]
            timings.setdefault("tick", []).append((time.perf_counter() - t0) * 1000)
            decisions[name] = {
                "visible": vis,
                "panel": panel,
                "board": b and [b.panel, b.rows, b.cols, [c.center for c in b.cells]],
                "scores": scores,
            }

    n = len(frames)
    print(f"{n} frames x {a.repeat}, {len(popups)} templates, {a.threads} cv2 thread(s)")
    for key, vals in timings.items():
        per_frame = sum(vals) / (n * a.repeat)
        print(
            f"  {key:14s} {per_frame:7.2f} ms/frame (median call {statistics.median(vals):.2f} ms)"
        )
    if a.out:
        a.out.write_text(json.dumps(decisions, indent=1, default=list))


if __name__ == "__main__":
    main()
