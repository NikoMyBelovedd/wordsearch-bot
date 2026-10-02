"""The iPhone stream on a break: a still screen sends no frames, and pymobiledevice3's own
stall watchdog restarted the stream every ~68 s for the whole break. No phone needed: the
real watchdog runs against a fake stream server."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest
from pymobiledevice3.remote.core_device import screen_stream

import wsbot.iphone
from wsbot.iphone import IPhone


class FakeServer:
    """What the stall watchdog and IPhone's keyframe watchdog read of the server."""

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self._subscribers = {object(): SimpleNamespace(needs_key=False, needs_key_since=0.0)}
        self._active_service = object()
        self._last_good_au_t = loop.time()  # a frame just came; the screen then stays still
        self._last_restart_t = 0.0
        self._consecutive_restarts = 0
        self.restarts = 0
        self.key_asks = 0

    async def _ensure_fresh_stream(self, force: bool = False) -> None:
        self.restarts += 1
        self._last_good_au_t = asyncio.get_running_loop().time()

    def _request_recovery_idr(self, *, reason: str) -> None:
        self.key_asks += 1


@pytest.fixture
def phone(monkeypatch):
    monkeypatch.setattr(screen_stream, "_STALL_RESTART_SECS", 1.0)  # checks every 0.25 s
    monkeypatch.setattr(screen_stream, "_STALL_RESTART_COOLDOWN_SECS", 0.0)
    monkeypatch.setattr(wsbot.iphone, "WAKE_KEY_S", 0.2)
    p = IPhone("fake")
    srv = FakeServer(p._loop)
    p._srv = srv
    tasks = [
        p._call(_start(screen_stream.ScreenStreamServer._stall_watchdog(srv)), 5),
        p._call(_start(p._keyframe_watchdog(srv)), 5),
    ]
    yield p, srv
    for t in tasks:
        p._loop.call_soon_threadsafe(t.cancel)


async def _start(coro) -> asyncio.Task:
    return asyncio.create_task(coro)


def test_a_quiet_stream_is_not_restarted_for_sending_nothing(phone):
    p, srv = phone
    p.set_quiet(True)
    time.sleep(2.5)  # ten watchdog checks, 2.5x the (shortened) stall limit
    assert srv.restarts == 0

    # Waking: no frame comes (still screen, keyframe request ignored) -> one restart.
    p.set_quiet(False)
    assert srv.key_asks >= 1
    assert srv.restarts == 1


def test_a_watched_still_stream_is_still_restarted(phone):
    """The watchdog itself still works when somebody looks (the test isn't vacuous)."""
    _, srv = phone
    time.sleep(1.6)
    assert srv.restarts >= 1
