"""Popup full scans on realistic frame sequences: the whole search (match()) against the
change-driven one (ChangeScan), frame by frame, with timings and every difference.

    uv run python tools/bench_sequences.py <frames dir> [--kinds static,play,popup,fade]
        [--crf 30] [--keyint 30] [--noise 0] [--every 1]

Each saved screen (calibration space, like diagnostics/*.png) becomes a few short
sequences, one frame per full scan (0.8-3 s apart in the bot):
  static  the screen, still
  play    a level being played: found words light up, a counter ticks
  popup   a popup shows over the dimmed screen, stays, goes
  fade    a popup fades and grows in over a dimming screen
  corpus  the saved screens in order: unrelated pictures (the worst case)
Frames are round-tripped through a real HEVC encoder (libx265, via PyAV) at the
phone's size, like the iPhone's stream; --noise adds per-pixel jitter before encoding
and a small --keyint makes many frames keyframes (every pixel re-quantized).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wsbot import watcher
from wsbot.imgio import imread
from wsbot.shot import Shot

NATIVE = (750, 1334)


def hevc(frames: list[np.ndarray], crf: int, keyint: int) -> list[np.ndarray]:
    h, w = frames[0].shape[:2]
    enc = av.CodecContext.create("libx265", "w")
    enc.width, enc.height, enc.pix_fmt = w, h, "yuv420p"
    enc.time_base = Fraction(1, 60)
    params = f"keyint={keyint}:min-keyint={keyint}:bframes=0:scenecut=0:log-level=error"
    enc.options = {"preset": "ultrafast", "crf": str(crf), "x265-params": params}
    dec = av.CodecContext.create("hevc", "r")
    out = []
    packets = []
    for i, f in enumerate(frames):
        vf = av.VideoFrame.from_ndarray(f, format="bgr24").reformat(format="yuv420p")
        vf.pts = i
        packets += enc.encode(vf)
    packets += enc.encode(None)
    for p in [*packets, None]:
        out += [fr.to_ndarray(format="bgr24") for fr in dec.decode(p)]
    assert len(out) == len(frames)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", type=Path)
    ap.add_argument("--templates", type=Path, default=Path("templates/ios"))
    ap.add_argument("--kinds", default="static,play,popup,fade")
    ap.add_argument("--crf", type=int, default=30)
    ap.add_argument("--keyint", type=int, default=30)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--every", type=int, default=1, help="use every Nth saved screen")
    a = ap.parse_args()
    cv2.setNumThreads(1)
    random.seed(7)
    rng = np.random.default_rng(7)
    kinds = a.kinds.split(",")
    popups = watcher.load_popups(a.templates)
    registry = json.loads((a.templates / "popups.json").read_text(encoding="utf-8"))
    art = {e["name"]: imread(a.templates / "popups" / e["file"]) for e in registry}
    files = sorted(p for p in a.frames.iterdir() if p.suffix == ".png")[:: a.every]

    def pill(im):  # a found word lights up
        w, h = random.randint(80, 420), random.randint(60, 110)
        x, y = random.randint(60, im.shape[1] - w - 60), random.randint(600, 1900)
        over = im.copy()
        cv2.rectangle(over, (x, y), (x + w, y + h), [int(v) for v in rng.integers(60, 255, 3)], -1)
        cv2.addWeighted(over, 0.55, im, 0.45, 0, im)

    def counter(im, i):
        cv2.rectangle(im, (1000, 60), (1240, 140), (40, 40, 40), -1)
        cv2.putText(im, str(1200 + 37 * i), (1010, 125), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 255, 5)

    def paste(im, name, scale=1.0, alpha=1.0, at=None):
        t = art[name]
        if scale != 1.0:
            t = cv2.resize(t, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        th, tw = t.shape[:2]
        x, y = at or (
            random.randint(0, im.shape[1] - tw - 1),
            random.randint(0, im.shape[0] - th - 1),
        )
        im[y : y + th, x : x + tw] = cv2.addWeighted(
            t, alpha, im[y : y + th, x : x + tw], 1 - alpha, 0
        )

    def sequences(img):
        if "static" in kinds:
            yield "static", [img] * 6
        if "play" in kinds:
            im, seq = img.copy(), []
            for i in range(8):
                if i:
                    for _ in range(random.randint(1, 3)):
                        pill(im)
                    counter(im, i)
                seq.append(im.copy())
            yield "play", seq
        if "popup" in kinds:
            a1 = img.copy()
            counter(a1, 1)
            b = (img * 0.5).astype(np.uint8)
            paste(b, random.choice(popups).name)
            c = img.copy()
            pill(c)
            counter(c, 2)
            yield "popup", [img, a1, b, b, b, c, c]
        if "fade" in kinds:
            name = random.choice(popups).name
            th, tw = art[name].shape[:2]
            at = (
                random.randint(0, img.shape[1] - tw - 1),
                random.randint(0, img.shape[0] - th - 1),
            )
            seq = [img]
            for dim, sc, al in [
                (0.9, 0.85, 0.3),
                (0.7, 0.95, 0.7),
                (0.5, 1.0, 1.0),
                (0.5, 1.0, 1.0),
            ]:
                f = (img * dim).astype(np.uint8)
                off = (at[0] + round(tw * (1 - sc) / 2), at[1] + round(th * (1 - sc) / 2))
                paste(f, name, sc, al, off)
                seq.append(f)
            yield "fade", seq

    stats: dict[str, dict] = {}

    def run(kind, calib_frames, scan):
        h, w = calib_frames[0].shape[:2]
        native = [cv2.resize(f, NATIVE, interpolation=cv2.INTER_AREA) for f in calib_frames]
        if a.noise:
            native = [
                np.clip(n + rng.normal(0, a.noise, n.shape), 0, 255).astype(np.uint8)
                for n in native
            ]
        st = stats.setdefault(kind, dict.fromkeys(("n", "base", "new", "wn", "wbase", "wnew"), 0))
        st.setdefault("diffs", 0)
        st.setdefault("spots", 0)
        st.setdefault("closest", 9.0)
        for i, n in enumerate(hevc(native, a.crf, a.keyint)):
            s = Shot(1, n, (w, h))
            for p in popups:  # the frame's sizes: made once either way
                s.coarse_as(p.coarse_look)
            t0 = time.perf_counter()
            base = [watcher.match(s.small, p, s.coarse_as(p.coarse_look)) for p in popups]
            t1 = time.perf_counter()
            scan.begin(s.coarse_color)
            new = [watcher.match(s.small, p, s.coarse_as(p.coarse_look), scan) for p in popups]
            t2 = time.perf_counter()
            st["n"] += 1
            st["base"] += t1 - t0
            st["new"] += t2 - t1
            if i:
                st["wn"] += 1
                st["wbase"] += t1 - t0
                st["wnew"] += t2 - t1
            for p, (s1, c1), (s2, c2) in zip(popups, base, new, strict=True):
                hit1, hit2 = s1 >= p.threshold, s2 >= p.threshold
                if hit1 != hit2 or (hit1 and max(abs(c1[0] - c2[0]), abs(c1[1] - c2[1])) > 2):
                    st["diffs"] += 1
                    print(f"DIFF {kind} frame {i} {p.name}: {s1:.4f} {c1} -> {s2:.4f} {c2}")
                elif c1 != c2:  # no match either way, the best non-match sits elsewhere
                    st["spots"] += 1
                    st["closest"] = min(st["closest"], p.threshold - max(s1, s2))

    t_start = time.time()
    if "corpus" in kinds:
        imgs = [cv2.imread(str(f)) for f in files]
        scan = watcher.ChangeScan()
        for k in range(0, len(imgs), 30):
            run("corpus", imgs[k : k + 30], scan)
    for f in files:
        for kind, seq in sequences(cv2.imread(str(f))):
            run(kind, seq, watcher.ChangeScan())
    print(f"{len(files)} screens, crf {a.crf}, keyint {a.keyint}, noise {a.noise}")
    for kind, st in stats.items():
        wn = max(1, st["wn"])
        print(
            f"  {kind:7s} {st['n']:5d} scans: {1000 * st['base'] / st['n']:5.1f} -> "
            f"{1000 * st['new'] / st['n']:5.1f} ms "
            f"({100 * (1 - st['new'] / st['base']):4.1f}% less)"
            f" | after the first frame {1000 * st['wbase'] / wn:5.1f} -> "
            f"{1000 * st['wnew'] / wn:5.1f} ms | decision diffs {st['diffs']} | non-match "
            f"spots elsewhere {st['spots']} (closest {st['closest']:.3f} under threshold)"
        )
    print(f"{time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
