"""Letter recognition: template match first, Tesseract only for glyphs never seen before.

Every glyph is normalized to a 64x64 square (aspect kept, centered), so templates
learned on a 5x5 board still match the smaller letters of a 10x10 board.
"""

from __future__ import annotations

import string
from pathlib import Path

import cv2
import numpy as np
import pytesseract

from .log import log

SIZE = 64
MATCH_MIN = 0.80  # accept a template match at or above this correlation
LEARN_MARGIN = 0.90  # below this, a confident OCR read is saved as an extra template variant


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


class LetterReader:
    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.folder.mkdir(parents=True, exist_ok=True)
        self.templates: list[tuple[str, np.ndarray]] = []
        for p in sorted(self.folder.glob("*.png")):
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                self.templates.append((p.stem.split("_")[0], img))
        log("LETTERS", f"loaded {len(self.templates)} templates from {folder}")

    def best_match(self, norm: np.ndarray) -> tuple[str, float]:
        best, best_score = "?", -1.0
        for letter, tmpl in self.templates:
            s = _score(norm, tmpl)
            if s > best_score:
                best, best_score = letter, s
        return best, best_score

    def ocr(self, glyph: np.ndarray) -> str | None:
        # Tesseract wants dark text on white with a margin around it.
        img = cv2.copyMakeBorder(255 - glyph, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
        text = pytesseract.image_to_string(
            img, config=f"--psm 10 -c tessedit_char_whitelist={string.ascii_uppercase}"
        ).strip()
        return text if len(text) == 1 and text in string.ascii_uppercase else None

    def learn(self, letter: str, norm: np.ndarray) -> None:
        n = sum(1 for t, _ in self.templates if t == letter) + 1
        cv2.imwrite(str(self.folder / f"{letter}_{n}.png"), norm)
        self.templates.append((letter, norm))
        log("OCR", f"learned '{letter}' (variant {n})")

    def read(self, glyph: np.ndarray) -> str | None:
        norm = normalize(glyph)
        letter, score = self.best_match(norm)
        if score >= LEARN_MARGIN:
            return letter
        read = self.ocr(glyph)
        if read is None:
            if score >= MATCH_MIN:
                return letter
            return None
        if score >= MATCH_MIN and read != letter:
            log("WARN", f"template says {letter} ({score:.2f}) but OCR says {read}; trusting OCR")
        self.learn(read, norm)
        return read
