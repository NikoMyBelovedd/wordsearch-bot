"""wordsearch-bot: ADB + OCR auto-solver for Word Search Explorer."""

import os

__version__ = "0.1.0"

# One bot per phone on one computer: math libraries get one thread each (before numpy
# loads; OpenBLAS also commits memory per thread).
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
