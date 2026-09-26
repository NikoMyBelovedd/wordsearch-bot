"""Tagged, greppable logging: `[HH:MM:SS] [TAG] message`.

Sinks are plain callables so the TUI can subscribe without the bot knowing about it.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

Sink = Callable[[str, str, str], None]  # (timestamp, tag, message)

_sinks: list[Sink] = []
_lock = threading.Lock()


def _stdout_sink(ts: str, tag: str, msg: str) -> None:
    print(f"[{ts}] [{tag}] {msg}", flush=True)


_sinks.append(_stdout_sink)


def set_sinks(*sinks: Sink) -> None:
    """Replace every sink (the TUI swaps stdout out for its log panel)."""
    with _lock:
        _sinks[:] = sinks


def log(tag: str, msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    with _lock:
        sinks = list(_sinks)
    for sink in sinks:
        try:
            sink(ts, tag, msg)
        except Exception:
            pass
