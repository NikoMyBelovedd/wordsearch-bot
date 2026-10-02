"""Rebuild data/words.txt from SCOWL.

    uv run python tools/build_words.py <scowl>/final

The old list (google-10000 + a Scrabble list, cut at 200,000 lines) lacked 23% of
SCOWL's size-35 words (everyday spelling: WATERFALL, SANCTUARY, WHISPERED) and most
place names (SIBERIA, OSLO, VIRGO): theme words like those fell to the exhaustive
pass, ~30 s a level. Order: the current first 10,000 lines (google-10000, unchanged),
then SCOWL levels 10..80 (English, American, British, Canadian, Australian words,
capitalised words and proper names), each level in the old list's order where it had
the word. Level 95 (and the old list's remaining Scrabble oddities, nearly all of it
95) only adds junk swipes. SCOWL's licence is in data/SCOWL-LICENSE.txt.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORDS = ROOT / "data" / "words.txt"
LEVELS = (10, 20, 35, 40, 50, 55, 60, 70, 80)
NAME = re.compile(
    r"(english|american|british|canadian|australian)-(words|upper|proper-names)\.(\d+)$"
)


def main(final: Path) -> None:
    level: dict[str, int] = {}
    for f in final.iterdir():
        m = NAME.match(f.name)
        if not m:
            continue
        for line in f.read_text(encoding="latin-1").split():
            w = line.upper()
            if len(w) >= 3 and w.isascii() and w.isalpha():
                level[w] = min(level.get(w, 999), int(m.group(3)))
    old = WORDS.read_text(encoding="utf-8").split()
    rank = {w: i for i, w in enumerate(old)}
    out = old[:10_000]
    seen = set(out)
    for lv in LEVELS:
        new = [w for w, x in level.items() if x == lv and w not in seen]
        new.sort(key=lambda w: (rank.get(w, len(old)), w))
        out += new
        seen.update(new)
    WORDS.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"{len(out)} words")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
