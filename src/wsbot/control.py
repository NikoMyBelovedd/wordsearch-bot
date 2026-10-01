"""Pause and resume from AutomationHQ (its control protocol v1).

When AHQ_CONTROL=stdin-v1 is set, AutomationHQ writes "pause" or "resume" lines
to stdin. The bot holds at its next safe point (before a swipe, at the top of
the main loop or during a rest), prints "[AHQ] paused", and carries on from the
same spot after "resume" ("[AHQ] resumed"). Without the variable nothing reads
stdin, so the bot behaves exactly as before.
"""

from __future__ import annotations

import os
import sys
import threading
from typing import IO

PROTOCOL = "stdin-v1"


def listen(pause: threading.Event, stream: IO[str] | None = None) -> threading.Thread | None:
    """Sets/clears `pause` from control lines, on a background thread."""
    if os.environ.get("AHQ_CONTROL") != PROTOCOL:
        return None
    source = stream if stream is not None else sys.stdin

    def read() -> None:
        for line in source:
            command = line.strip().lower()
            if command == "pause":
                pause.set()
            elif command == "resume":
                pause.clear()
        # End of input: AutomationHQ went away. Never leave the bot stuck paused.
        pause.clear()

    thread = threading.Thread(target=read, name="ahq-control", daemon=True)
    thread.start()
    return thread
