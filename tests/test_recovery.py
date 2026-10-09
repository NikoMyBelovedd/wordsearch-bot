"""One bot per phone, a stream port per bot, and giving up loudly instead of looping."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from wsbot import iphone
from wsbot.bot import Bot, Pacing
from wsbot.instance import AlreadyRunning, acquire, lock_path

ROOT = Path(__file__).resolve().parents[1]

HOLDER = """
import sys, time
sys.path.insert(0, {src!r})
from wsbot.instance import acquire
from pathlib import Path
lock = acquire(Path({root!r}), {serial!r})
print("locked", flush=True)
time.sleep(60)
"""


def hold_in_child(root: Path, serial: str) -> subprocess.Popen:
    code = HOLDER.format(src=str(ROOT / "src"), root=str(root), serial=serial)
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "locked"
    return p


def test_second_bot_on_the_same_phone_fails_fast(tmp_path):
    child = hold_in_child(tmp_path, "ios:00008110-ABC")
    try:
        t0 = time.monotonic()
        with pytest.raises(AlreadyRunning, match="already playing on this phone"):
            acquire(tmp_path, "ios:00008110-ABC")
        assert time.monotonic() - t0 < 1.0
        other = acquire(tmp_path, "ios:00008110-DEF")  # another phone: fine
        other.release()
    finally:
        child.kill()
        child.wait()
    lock = acquire(tmp_path, "ios:00008110-ABC")  # the killed bot's lock is gone
    assert lock.held
    lock.release()


def test_lock_names_are_safe_file_names(tmp_path):
    assert lock_path(tmp_path, "ios:0000-AB").name == "bot-ios-0000-AB.lock"
    assert lock_path(tmp_path, "192.168.1.5:5555").name == "bot-192.168.1.5-5555.lock"


def test_cli_exits_with_a_one_line_error_when_the_phone_is_taken():
    serial = "ios:TEST-LOCK-CLI"
    lock = acquire(ROOT, serial)
    try:
        t0 = time.monotonic()
        r = subprocess.run(
            [sys.executable, "-m", "wsbot", "--headless", "--serial", serial],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        )
        assert time.monotonic() - t0 < 30
    finally:
        lock.release()
        lock_path(ROOT, serial).unlink(missing_ok=True)
        for f in (ROOT / "local").glob("*TEST-LOCK-CLI*"):
            f.unlink()
    assert r.returncode == 3
    assert "Another copy of the bot is already playing on this phone" in r.stderr
    assert "Traceback" not in r.stderr


def test_bot_releases_its_lock_when_the_phone_fails_to_open(tmp_path, monkeypatch):
    def broken(serial, dry_run=False):
        raise RuntimeError("no phone")

    monkeypatch.setattr("wsbot.bot.open_device", broken)
    with pytest.raises(RuntimeError, match="no phone"):
        Bot("ios:XYZ", tmp_path, SimpleNamespace(schedule=None))
    acquire(tmp_path, "ios:XYZ").release()  # free again


def test_no_blind_taps_when_not_sure_we_are_in_the_game():
    taps = []
    fake = SimpleNamespace(
        device=SimpleNamespace(tap=lambda *a, **k: taps.append(a), above_board=(1, 2)),
        watcher=SimpleNamespace(blind_taps_ok=lambda: False, game_chrome=lambda shot: True),
        board_center=(3, 4),
        clear_taps=0,
    )
    Bot._clear_tap(fake, None)
    assert taps == []
    fake.watcher.blind_taps_ok = lambda: True
    Bot._clear_tap(fake, None)
    assert taps == [(3, 4)]
    # The game's top bar gone (a full-screen ad after a level): no blind tap.
    fake.watcher.game_chrome = lambda shot: False
    Bot._clear_tap(fake, None)
    assert taps == [(3, 4)]


def test_fatal_stops_the_bot():
    import threading

    fake = SimpleNamespace(fatal=None, stop_event=threading.Event())
    Bot._fatal(fake, "stream dead")
    assert fake.fatal == "stream dead" and fake.stop_event.is_set()


# ---- iPhone stream port ---------------------------------------------------------------


def test_each_bot_gets_its_own_stream_port(monkeypatch):
    monkeypatch.delenv("WSBOT_STREAM_PORT", raising=False)
    assert iphone.stream_port_setting() is None
    a, b = iphone.free_port(), iphone.free_port()
    assert a > 0 and b > 0
    monkeypatch.setenv("WSBOT_STREAM_PORT", "8090")
    assert iphone.stream_port_setting() == 8090
    monkeypatch.setenv("WSBOT_STREAM_PORT", "nonsense")
    assert iphone.stream_port_setting() is None


def test_a_taken_port_is_an_open_error_not_another_bots_phone():
    """Second bot, same port: its server can't bind, and reading the port would show the
    FIRST bot's phone. _wait_listening must say so instead of carrying on."""

    async def scenario() -> tuple[str, str | None]:
        async def handler(reader, writer):
            writer.close()

        other = await asyncio.start_server(handler, "127.0.0.1", 0)  # the other bot
        port = other.sockets[0].getsockname()[1]
        ours = asyncio.ensure_future(asyncio.start_server(handler, "127.0.0.1", port))
        fake = SimpleNamespace(port=port)
        try:
            await iphone.IPhone._wait_listening(fake, ours)
            taken = "no error"
        except iphone.IPhoneError as exc:
            taken = str(exc)
        # our own server on a free port: fine
        free = iphone.free_port()

        async def serve():
            srv = await asyncio.start_server(handler, "127.0.0.1", free)
            async with srv:
                await srv.serve_forever()

        task = asyncio.ensure_future(serve())
        fake.port = free
        ok = None
        try:
            await iphone.IPhone._wait_listening(fake, task)
        except iphone.IPhoneError as exc:
            ok = str(exc)
        task.cancel()
        other.close()
        return taken, ok

    taken, ok = asyncio.run(scenario())
    assert "can't listen on port" in taken
    assert ok is None


# ---- frozen stream escalation -----------------------------------------------------------


class FrozenServer:
    def __init__(self) -> None:
        loop = asyncio.get_running_loop()
        self._subscribers = {1: SimpleNamespace(needs_key=True, needs_key_since=loop.time())}
        self.restarts = 0

    async def _ensure_fresh_stream(self, force=False):
        self.restarts += 1
        raise RuntimeError("CoreDevice.error code 24")

    def _request_recovery_idr(self, reason=""):
        pass


def test_failing_stream_restarts_escalate_to_a_reopen(monkeypatch):
    monkeypatch.setattr(iphone, "KEY_RESTART_S", 0.1)
    monkeypatch.setattr(iphone, "KEY_RESTART_COOLDOWN_S", 0.1)

    async def scenario():
        srv = FrozenServer()
        escalated = []
        fake = SimpleNamespace(escalated=escalated, _quiet=False)
        fake._frozen_since = types.MethodType(iphone.IPhone._frozen_since, fake)
        fake._escalate = escalated.append
        await asyncio.wait_for(iphone.IPhone._keyframe_watchdog(fake, srv), 10)
        return srv.restarts, escalated

    restarts, escalated = asyncio.run(scenario())
    assert restarts == iphone.STREAM_RESTART_FAILS
    assert len(escalated) == 1 and "restarts failed" in escalated[0]


def escalator(reopen) -> SimpleNamespace:
    fake = SimpleNamespace(
        fatal=None, on_fatal=None, _escalating=True, _escalations=[], reopen=reopen
    )
    fake._given_up = lambda: False
    fake._give_up = types.MethodType(iphone.IPhone._give_up, fake)
    return fake


def test_reopen_that_fails_gives_up_with_a_clear_error():
    def reopen(give_up_after=None):
        assert give_up_after == iphone.REOPEN_GIVE_UP_S
        raise iphone.IPhoneError("not back after 180s")

    fake = escalator(reopen)
    told = []
    fake.on_fatal = told.append
    iphone.IPhone._escalate_run(fake, "picture frozen 60s")
    assert told and "screen stream is dead" in told[0] and fake.fatal == told[0]
    assert fake._escalating is False


def test_reopens_that_dont_hold_give_up():
    reopens = []
    fake = escalator(lambda give_up_after=None: reopens.append(1))
    told = []
    fake.on_fatal = told.append
    for _ in range(iphone.ESCALATION_REOPENS):
        iphone.IPhone._escalate_run(fake, "frozen")
    assert len(reopens) == iphone.ESCALATION_REOPENS and not told
    iphone.IPhone._escalate_run(fake, "frozen")
    assert len(reopens) == iphone.ESCALATION_REOPENS
    assert told and "keeps freezing" in told[0]


def test_the_first_blind_tap_after_a_mid_level_vanish_waits_for_the_level_end_screen():
    fake = SimpleNamespace(pacing=Pacing(), watcher=SimpleNamespace(last_match=0.0))
    should = lambda now, at_least=None: Bot._should_clear_tap(fake, now, 0.0, 0.0, at_least)  # noqa: E731
    assert should(2.5), "between levels: 2.5 s as before"
    assert not should(2.5, at_least=Pacing().level_end_tap_s), "mid-level: not yet"
    assert should(Pacing().level_end_tap_s, at_least=Pacing().level_end_tap_s)


def test_a_headless_stop_also_ends_a_wait_for_the_iphone(monkeypatch):
    import threading

    from wsbot import cli, ios_device

    monkeypatch.setattr(ios_device, "_stop_waiting", threading.Event())
    bot = SimpleNamespace(pause_event=threading.Event(), stop_event=threading.Event())
    bot.pause_event.set()
    cli.stop(bot, "ios")
    assert bot.stop_event.is_set() and not bot.pause_event.is_set()
    assert ios_device._stop_waiting.is_set()  # reopen() stops waiting for the phone
    android = SimpleNamespace(pause_event=threading.Event(), stop_event=threading.Event())
    monkeypatch.setattr(ios_device, "_stop_waiting", threading.Event())
    cli.stop(android, "R5CT123")
    assert android.stop_event.is_set() and not ios_device._stop_waiting.is_set()
