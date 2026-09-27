"""--diagnose / --calibrate: see exactly what the bot sees on the current frame."""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from .board import read_board
from .device import open_device
from .imgio import imwrite
from .letters import LetterReader
from .solver import Dictionary
from .watcher import SCALE, load_popups, match


def run(serial: str, root: Path, *, overlay: bool) -> None:
    device = open_device(serial)
    frame = device.frame()
    small = cv2.resize(frame, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)

    print("\nPOPUP TEMPLATES")
    popups = load_popups(root / device.templates)
    matches = []
    for p in popups:
        score, center = match(small, p)
        hit = score >= p.threshold
        matches.append((p, score, center, hit))
        print(
            f"  {'MATCH   ' if hit else 'mismatch'} {p.name:<28} {score:.3f} "
            f"(>= {p.threshold}) at {center}"
        )

    print("\nBOARD")
    board = read_board(frame)
    hits = []
    if board is None:
        print("  no board visible")
    else:
        reader = LetterReader(root / "templates" / "letters")
        grid = [
            "".join(reader.read(board.cell(r, c).glyph) or "?" for c in range(board.cols))
            for r in range(board.rows)
        ]
        print(f"  {board.rows}x{board.cols} panel={board.panel}")
        for row in grid:
            print("   ", " ".join(row))
        if "?" not in "".join(grid):
            hits = Dictionary(root / "data" / "words.txt").solve(grid)
            print(
                f"  {len(hits)} candidates: {' '.join(h.word for h in hits[:40])}"
                f"{' ...' if len(hits) > 40 else ''}"
            )

    if not overlay:
        return
    out = frame.copy()
    for name, (x1, y1, x2, y2) in device.zones.items():
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 4)
        cv2.putText(out, name, (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    if board:
        x, y, w, h = board.panel
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 200, 0), 4)
        for cell in board.cells:
            cv2.circle(out, cell.center, 12, (0, 200, 0), -1)
        for hit in hits[:15]:
            a, b = board.cell(*hit.start).center, board.cell(*hit.end).center
            cv2.line(out, a, b, (0, 140, 255), 6)
    for p, _score, center, hit in matches:
        if hit:
            cv2.circle(out, center, 40, (255, 0, 255), 6)
            cv2.putText(
                out,
                p.name,
                (center[0] - 60, center[1] - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 0, 255),
                3,
            )
    (root / "diagnostics").mkdir(exist_ok=True)
    path = root / "diagnostics" / f"calibrate_{time.strftime('%Y%m%d_%H%M%S')}.png"
    imwrite(path, out)
    print(f"\noverlay saved: {path}")
