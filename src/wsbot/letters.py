"""Letter recognition: template match, with Tesseract as a second vote for new glyphs.

Two template sets:
  ref/   A-Z rendered from Lato Black (the game's font), committed, always present
  game/  glyphs captured from the real board, learned when OCR and templates agree

Every glyph is normalized to a 64x64 square (aspect kept, centered), so templates
learned on a 5x5 board still match the smaller letters of a 10x10 board.

A glyph is only ever learned when two independent readers agree: Tesseract alone
once read a V as "A", and a self-learning cache must never be poisoned.
"""

from __future__ import annotations

import os
import shutil
import string
from pathlib import Path

import cv2
import numpy as np
import pytesseract

from .imgio import imread, imwrite
from .log import log

SIZE = 64
ACCEPT = 0.90  # best-letter score needed to accept without OCR
MARGIN = 0.08  # ...and how far ahead of the runner-up letter it must be
# A slightly off-shape glyph (iOS mirror frames are upscaled and softer) can sit just
# under ACCEPT while no other letter comes close; a big lead is as good as a high score.
LEAD_ACCEPT, LEAD_MARGIN = 0.85, 0.15
OCR_HEIGHT = 40  # Tesseract reads single glyphs most reliably at this size


def _find_tesseract() -> str | None:
    """$TESSERACT_CMD, then PATH, then the usual Windows install folders."""
    candidates = [os.environ.get("TESSERACT_CMD"), shutil.which("tesseract")]
    for base in (
        os.environ.get("ProgramFiles"),
        os.environ.get("ProgramFiles(x86)"),
        os.environ.get("LOCALAPPDATA"),
    ):
        if base:
            candidates.append(str(Path(base) / "Tesseract-OCR" / "tesseract.exe"))
    return next((c for c in candidates if c and Path(c).is_file()), None)


TESSERACT = _find_tesseract()
if TESSERACT:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT


def normalize(glyph: np.ndarray) -> np.ndarray:
    """Pad a binary glyph to a square (keeping its aspect) and resize to SIZE x SIZE."""
    h, w = glyph.shape
    side = max(h, w)
    square = np.zeros((side, side), np.uint8)
    y0, x0 = (side - h) // 2, (side - w) // 2
    square[y0 : y0 + h, x0 : x0 + w] = glyph
    return cv2.resize(square, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def _score(a: np.ndarray, b: np.ndarray) -> float:
    return float(cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)[0, 0])


def _unit(img: np.ndarray) -> tuple[np.ndarray, bool]:
    """The image as a zero-mean unit vector, and whether it was flat (no contrast).

    TM_CCOEFF_NORMED of two same-size images is the dot product of these vectors, so
    one matrix product scores a glyph against every template at once.
    """
    v = img.astype(np.float64).ravel()
    v -= v.mean()
    n = float(np.linalg.norm(v))
    return (v / n, False) if n > 1e-9 else (v, True)


class LetterReader:
    def __init__(self, folder: Path) -> None:
        self.game_dir = folder / "game"
        self.game_dir.mkdir(parents=True, exist_ok=True)
        self.templates: list[tuple[str, np.ndarray]] = []
        self.learned: set[str] = set()  # letters with at least one real board glyph
        # ranked() scores against these in one matrix product: one matchTemplate call
        # per template took 6.3 s a 10x9 board on a 2-core laptop (517 templates), and
        # boards are read twice per level start and after every pause.
        self._rows: list[np.ndarray] = []
        self._flat: list[bool] = []
        self._mat: np.ndarray | None = None
        for sub in ("ref", "game"):
            for p in sorted((folder / sub).glob("*.png")):
                img = imread(p, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                letter = p.stem.split("_")[0]
                self._add(letter, img)
                if sub == "game":
                    self.learned.add(letter)
        n, learned = len(self.templates), len(self.learned)
        log("LETTERS", f"loaded {n} templates, {learned} letters learned")
        if TESSERACT is None:
            log("WARNING", "Tesseract not found: new glyphs can't be learned (set TESSERACT_CMD)")

    def _add(self, letter: str, tmpl: np.ndarray) -> None:
        self.templates.append((letter, tmpl))
        row, flat = _unit(tmpl)
        self._rows.append(row)
        self._flat.append(flat)
        self._mat = None

    def ranked(self, norm: np.ndarray) -> list[tuple[float, str]]:
        """Best score per letter, highest first (same scores as matchTemplate)."""
        if self._mat is None:
            self._mat = np.stack(self._rows)
            self._flat_idx = np.flatnonzero(self._flat)
        g, flat = _unit(norm)
        scores = np.zeros(len(self._rows)) if flat else self._mat @ g
        scores[self._flat_idx] = 1.0  # what matchTemplate says for a flat template
        best: dict[str, float] = {}
        for (letter, _), s in zip(self.templates, scores.tolist(), strict=True):
            if s > best.get(letter, -1.0):
                best[letter] = s
        return sorted(((s, letter) for letter, s in best.items()), reverse=True)

    def ocr(self, glyph: np.ndarray) -> str | None:
        if TESSERACT is None:
            return None  # templates alone still read every letter already learned
        img = 255 - glyph  # Tesseract wants dark text on white, with a margin
        scale = OCR_HEIGHT / img.shape[0]
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        img = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        text = pytesseract.image_to_string(
            img, config=f"--psm 8 -c tessedit_char_whitelist={string.ascii_uppercase}"
        ).strip()
        return text if len(text) == 1 else None

    def learn(self, letter: str, norm: np.ndarray) -> None:
        n = len(list(self.game_dir.glob(f"{letter}_*.png"))) + 1
        imwrite(self.game_dir / f"{letter}_{n}.png", norm)
        self._add(letter, norm)
        self.learned.add(letter)
        log("OCR", f"learned '{letter}' from the board (variant {n})")

    def read(self, glyph: np.ndarray) -> str | None:
        norm = normalize(glyph)
        (s1, top), (s2, _) = self.ranked(norm)[:2]
        confident = (s1 >= ACCEPT and s1 - s2 >= MARGIN) or (
            s1 >= LEAD_ACCEPT and s1 - s2 >= LEAD_MARGIN
        )
        if confident and top in self.learned:
            return top
        ocr = self.ocr(glyph)
        if ocr == top:
            self.learn(top, norm)  # two readers agree: capture the real game glyph
            return top
        if confident:
            log("WARN", f"OCR said {ocr} but template is sure of {top} ({s1:.2f}); using {top}")
            return top
        log("WARN", f"unreadable glyph: template {top} ({s1:.2f} vs {s2:.2f}), OCR {ocr}")
        return None
