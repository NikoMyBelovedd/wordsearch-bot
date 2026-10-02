"""Replay the popup watcher's decision on saved screenshots: which registered entry owns
each frame and what the bot would do (tap where / wait / relaunch the game / nothing =
an unknown screen), plus the no-tap zones the ad buttons on it mark.

    uv run python tools/replay_popups.py <frames dir or files...> [--templates templates/ios]
        [--native 750x1334] [--json out.json]

Frames are calibration-space screenshots (the diagnostics/*.png the bot saves);
--native first shrinks each to the phone's own size, the way an iPhone frame arrives.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wsbot import watcher
from wsbot.shot import Shot

SE_CALIB = (1296, 2305)


def decide(popups: list[watcher.Popup], shot: Shot) -> dict:
    """What the watcher would do on this picture (a full scan, the way _score and
    _handle_popups see it)."""
    scores = {}
    for p in popups:
        found = watcher.match(shot.small, p, shot.coarse_as(p.coarse_look))
        scores[p.name] = watcher.match_variants(shot.small, p, found)
    zones = {p.name: scores[p.name] for p in popups if p.avoid and scores[p.name][0] >= p.threshold}
    hit = watcher.top_match(popups, scores)
    if hit is None:
        best = max((p for p in popups if not p.avoid), key=lambda p: scores[p.name][0])
        return {
            "entry": None,
            "action": "unknown",
            "best": [best.name, round(scores[best.name][0], 3)],
            "zones": sorted(zones),
        }
    popup, score, center = hit
    if popup.relaunch:
        action = "relaunch"
    elif popup.tap:
        action = f"tap {tuple(popup.tap_point) if popup.tap_point else center}"
    elif popup.tap_after:
        action = f"wait (tap {center} after {popup.tap_after:.0f}s)"
    else:
        action = "wait"
    return {
        "entry": popup.name,
        "score": round(score, 3),
        "action": action + (" +level_done" if popup.level_done else ""),
        "zones": sorted(zones),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", type=Path, nargs="+")
    ap.add_argument("--templates", type=Path, default=Path("templates/ios"))
    ap.add_argument("--native", default="750x1334")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    cv2.setNumThreads(1)
    paths = []
    for f in a.frames:
        paths += sorted(f.glob("*.png")) + sorted(f.glob("*.jpg")) if f.is_dir() else [f]
    popups = watcher.load_popups(a.templates)
    out = {}
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        calib = (img.shape[1], img.shape[0])
        if calib[0] < 1000:  # already at the phone's size (tests/data): the SE's calibration
            calib = SE_CALIB
        native = img
        if a.native:
            w, h = (int(v) for v in a.native.split("x"))
            if (w, h) != (img.shape[1], img.shape[0]):
                native = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        d = decide(popups, Shot(1, native, calib))
        key = f"{p.parent.name}/{p.name}"
        out[key] = d
        extra = f" zones={d['zones']}" if d["zones"] else ""
        what = f"{d['entry']} {d['score']}" if d["entry"] else f"(best {d['best']})"
        print(f"{key:58} {d['action']:34} {what}{extra}")
    if a.json:
        a.json.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
