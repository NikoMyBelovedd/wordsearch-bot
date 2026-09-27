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
