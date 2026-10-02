"""cv2.imread / cv2.imwrite that also work on Windows paths with non-ASCII characters
(OpenCV's own file functions silently fail on e.g. C:\\Users\\José\\...)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, flags) if data.size else None


def imwrite(path: str | Path, img: np.ndarray) -> bool:
    ok, buf = cv2.imencode(Path(path).suffix or ".png", img)
    if ok:
        buf.tofile(str(path))
    return bool(ok)


MAX_DIAGNOSTICS = 40  # newest saved screens kept (~2 MB each); a 10-day run must not fill the disk


def prune_pngs(folder: Path, keep: int = MAX_DIAGNOSTICS) -> None:
    """Delete all but the `keep` newest *.png in `folder`. Every writer of saved
    screens calls it after a save (the bot's dumps, the watcher's unknown screens and
    alerts): on a phone that shows ads, each ad is a new unknown screen."""

    def mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except OSError:  # deleted meanwhile (another thread pruning)
            return 0.0

    for old in sorted(folder.glob("*.png"), key=mtime)[:-keep]:
        old.unlink(missing_ok=True)
