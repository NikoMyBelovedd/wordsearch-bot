"""Render the A-Z reference glyphs in templates/letters/ref/ (run once; output is committed).

The game's board font is effectively Lato Black, so these renders classify real
board glyphs with a wide margin and act as the second vote next to Tesseract.

    uv run --group dev python tools/render_reference.py path/to/Lato-Black.ttf
"""

import string
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from wsbot.letters import normalize

OUT = Path(__file__).resolve().parents[1] / "templates" / "letters" / "ref"


def main(font_path: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(font_path, 200)
    for letter in string.ascii_uppercase:
        img = Image.new("L", (400, 400), 0)
        ImageDraw.Draw(img).text((100, 50), letter, fill=255, font=font)
        a = np.array(img)
        ys, xs = np.where(a > 127)
        glyph = ((a[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1] > 127) * 255).astype(np.uint8)
        cv2.imwrite(str(OUT / f"{letter}.png"), normalize(glyph))
    print(f"wrote 26 reference glyphs to {OUT}")


if __name__ == "__main__":
    main(sys.argv[1])
