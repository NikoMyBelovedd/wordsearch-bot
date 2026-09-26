"""Tagged, greppable logging: `[HH:MM:SS] [TAG] message`.

Sinks are plain callables so the TUI can subscribe without the bot knowing about it.
"""

from __future__ import annotations

import logging
import logging.handlers
import threading
import time
from collections.abc import Callable

Sink = Callable[[str, str, str], None]  # (timestamp, tag, message)

_sinks: list[Sink] = []
_lock = threading.Lock()


def stdout_sink(ts: str, tag: str, msg: str) -> None:
    print(f"[{ts}] [{tag}] {msg}", flush=True)


_sinks.append(stdout_sink)


def file_sink(path, max_bytes: int = 5_000_000, backups: int = 5) -> Sink:
    """Append to `path`, rotating at `max_bytes` so a multi-day run stays bounded."""
    handler = logging.handlers.RotatingFileHandler(
        path, maxBytes=max_bytes, backupCount=backups, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("wsbot.file")
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.handlers[:] = [handler]
    return lambda ts, tag, msg: logger.info(f"{time.strftime('%Y-%m-%d')} {ts} [{tag}] {msg}")


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
