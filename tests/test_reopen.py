"""Reopening the iPhone session after the helper (`pymobiledevice3 remote tunneld`)
restarted: the phone's tunnel comes back at a new address and port. On the laptop the
bot's own reopen then failed for 209 s and it exited 4, while a fresh process connected
at once. A reopen now does what a fresh process does: asks tunneld again (and survives a
stale tunnel in its list), starts on a new event loop (the dead session's tasks can't
linger), and makes the frame reader leave a connection nothing will ever close.

No phone: fake tunneld listings and RSDs, a real HEVC frame from PyAV's encoder, local
sockets for the stream server."""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from fractions import Fraction
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
from pymobiledevice3.exceptions import TunneldConnectionError

from wsbot import iphone
from wsbot.iphone import IPhone, NoTunnel

UDID = "00008110-TEST"


class FakeRSD:
    """A tunnel endpoint: "stale..." addresses are gone, "blackhole..." never answer."""

    connected: ClassVar[list[tuple[str, int]]] = []

    def __init__(self, address, name=None, open_connection=None, auxiliary_metadata=None):
        self.address = tuple(address)
        self.service = SimpleNamespace(address=self.address)
        self.udid = UDID
        self.closed = False

    async def connect(self) -> None:
        host = self.address[0]
        if host.startswith("stale"):
            raise OSError(101, "Network is unreachable")
        if host.startswith("blackhole"):
            await asyncio.sleep(3600)
        FakeRSD.connected.append(self.address)

    async def close(self) -> None:
        self.closed = True


class Tunneld:
    """What `GET /` on tunneld answers, one listing per call (the last one repeats)."""

    def __init__(self, *listings) -> None:
        self.listings = list(listings)
        self.calls = 0

    async def __call__(self, *a, **k):
        self.calls += 1
        listing = self.listings.pop(0) if len(self.listings) > 1 else self.listings[0]
        if listing is None:
            raise TunneldConnectionError()
        return {
            udid: [{"tunnel-address": h, "tunnel-port": p, "interface": "utun9"} for h, p in ts]
            for udid, ts in listing.items()
        }


@pytest.fixture
def tunneld(monkeypatch):
    FakeRSD.connected = []
    monkeypatch.setattr(
        "pymobiledevice3.remote.remote_service_discovery.RemoteServiceDiscoveryService", FakeRSD
    )
    monkeypatch.setattr(iphone, "TUNNEL_CONNECT_S", 0.3)

    def install(*listings) -> Tunneld:
        fake = Tunneld(*listings)
        monkeypatch.setattr("pymobiledevice3.tunneld.api.get_tunneld_tunnels", fake)
        return fake

    return install


def rsds_of(phone) -> list:
    return asyncio.run(IPhone._tunneld_rsds(phone))


def test_every_open_asks_tunneld_again(tunneld):
    fake = tunneld({UDID: [("fd01::1", 5000)]}, {UDID: [("fd02::1", 6000)]})
    phone = SimpleNamespace(udid=UDID)
    assert [r.address for r in rsds_of(phone)] == [("fd01::1", 5000)]
    assert [r.address for r in rsds_of(phone)] == [("fd02::1", 6000)]  # the helper restarted
    assert fake.calls == 2


def test_a_stale_or_silent_tunnel_doesnt_hide_the_good_one(tunneld):
    """pymobiledevice3's get_tunneld_devices let one OSError end the whole list, and
    waited on a silent tunnel with no timeout."""
    tunneld({UDID: [("fd02::1", 6000), ("blackhole::1", 1), ("stale::1", 5000)]})
    t0 = time.monotonic()
    rsds = rsds_of(SimpleNamespace(udid=UDID))
    assert [r.address for r in rsds] == [("fd02::1", 6000)]
    assert time.monotonic() - t0 < 2


def test_only_our_phone_and_a_clear_error_when_the_helper_is_down(tunneld):
    tunneld({"OTHER": [("fd09::1", 1)], UDID: [("fd02::1", 6000)]})
    assert [r.address for r in rsds_of(SimpleNamespace(udid=UDID))] == [("fd02::1", 6000)]
    assert len(rsds_of(SimpleNamespace(udid=""))) == 2  # no UDID: every phone, as before
    tunneld(None)
    with pytest.raises(NoTunnel, match="helper"):
        rsds_of(SimpleNamespace(udid=UDID))


def test_the_tunnel_move_is_logged(monkeypatch):
    lines = []
    monkeypatch.setattr(iphone, "log", lambda tag, msg: lines.append((tag, msg)))
    phone = SimpleNamespace(_tunnel=None)
    IPhone._note_tunnel(phone, FakeRSD(("fd01::1", 5000)))
    IPhone._note_tunnel(phone, FakeRSD(("fd02::1", 6000)))
    assert lines[-1][0] == "RECOVERY" and "moved" in lines[-1][1] and "fd02::1" in lines[-1][1]


def test_app_control_uses_the_stream_servers_current_tunnel():
    """The stream server reconnects through tunneld by itself after a drop; launching or
    checking the game must follow it, not the dead RSD the session started with."""
    phone = SimpleNamespace(_srv=SimpleNamespace(_rsd="new"), _rsd="old")
    assert IPhone._live_rsd(phone) == "new"
    phone._srv = None
    assert IPhone._live_rsd(phone) == "old"


# ---- reopen() as a whole ---------------------------------------------------------------


@pytest.fixture
def phone(monkeypatch, tunneld):
    monkeypatch.setattr(iphone, "REOPEN_RETRY_S", 0.05)
    p = IPhone(UDID)
    yield p
    p._closing.set()
    p._stop.set()


def fake_open(p: IPhone, opened: list):
    """IPhone.open with the real tunnel lookup and a stand-in stream server."""

    async def session() -> None:
        rsds = await p._tunneld_rsds()
        if not rsds:
            raise NoTunnel("no tunnel yet")
        rsd = rsds[0]
        p._note_tunnel(rsd)

        async def serve() -> None:
            await asyncio.sleep(3600)

        p._srv = SimpleNamespace(_rsd=rsd)
        p._serve_task = asyncio.get_running_loop().create_task(serve())
        p._rsd = rsd
        opened.append((asyncio.get_running_loop(), rsd.address))

    return lambda: p._call(session(), 10)


def test_reopen_after_the_helper_restarted_finds_the_new_tunnel(phone, tunneld):
    tunneld(
        {UDID: [("fd01::1", 5000)]},  # first open
        None,  # helper restarting: not answering
        {},  # back, phone's tunnel not up yet
        {UDID: [("fd02::1", 6000), ("stale::1", 5000)]},  # new tunnel (+ a stale one)
    )
    opened: list = []
    phone.open = fake_open(phone, opened)
    phone.open()
    first_loop = phone._loop
    # A task of the dead session that nothing ends (like the stream server's own
    # tunneld reconnect loop, or a /stream.bin handler its teardown never reached)
    zombie = asyncio.run_coroutine_threadsafe(asyncio.sleep(3600), first_loop)

    phone.reopen(give_up_after=30)

    assert [a for _, a in opened] == [("fd01::1", 5000), ("fd02::1", 6000)]
    assert opened[-1][0] is phone._loop and phone._loop is not first_loop
    assert zombie.cancelled()
    for _ in range(50):
        if not first_loop.is_running():
            break
        time.sleep(0.02)
    assert not first_loop.is_running()
    assert phone.alive and phone._tunnel == ("fd02::1", 6000)


def test_reopen_gives_up_only_after_enough_tries(phone, monkeypatch):
    tries = []

    def never():
        tries.append(1)
        raise NoTunnel("still no tunnel")

    phone.open = never
    with pytest.raises(iphone.IPhoneError, match=f"{iphone.REOPEN_MIN_TRIES} tries"):
        phone.reopen(give_up_after=0.0)
    assert len(tries) == iphone.REOPEN_MIN_TRIES


# ---- the frame reader --------------------------------------------------------------------


def hevc_keyframe(size: int = 64) -> bytes:
    """One HEVC keyframe as /stream.bin sends it: length-prefixed NAL units."""
    import av

    enc = av.CodecContext.create("libx265", "w")
    enc.width = enc.height = size
    enc.pix_fmt = "yuv420p"
    enc.time_base = Fraction(1, 30)
    enc.options = {"x265-params": "log-level=none"}
    img = np.zeros((size, size, 3), np.uint8)
    img[:, : size // 2] = 200
    frame = av.VideoFrame.from_ndarray(img, format="bgr24").reformat(format="yuv420p")
    frame.pts = 0
    annexb = b"".join(bytes(p) for p in [*enc.encode(frame), *enc.encode(None)])
    out = bytearray()
    for nal in annexb.split(b"\x00\x00\x01"):
        nal = nal.rstrip(b"\x00")
        if nal:
            out += len(nal).to_bytes(4, "big") + nal
    return bytes(out)


class StreamServer:
    """A local /stream.bin: `frames` records after the headers, then silence (a still
    screen, or a server that's gone but never closed this connection)."""

    def __init__(self, frames: list[bytes]) -> None:
        self.frames = frames
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen()
        self.port = self.sock.getsockname()[1]
        self.connections = 0
        self.conns: list[socket.socket] = []
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            self.connections += 1
            self.conns.append(conn)
            conn.recv(4096)
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\n\r\n")
            for i, au in enumerate(self.frames):
                # 2 = a keyframe after a (re)start; then the same picture again, which
                # also pushes the first one out of the decoder
                body = (b"\x02" if i == 0 else b"\x00") + au
                conn.sendall(len(body).to_bytes(4, "big") + body)

    def close(self) -> None:
        self.sock.close()
        for c in self.conns:
            c.close()


def test_the_reader_leaves_a_dead_servers_connection_and_reads_the_new_one(monkeypatch):
    zombie = StreamServer([])  # the old session's server: never sends, never closes
    fresh = StreamServer([hevc_keyframe()] * 3)
    p = IPhone(UDID)
    p.width = p.height = 64
    p._hw = False  # this checks the reconnect, not GPU decoding (CI GPUs drop tiny frames)
    p.port = zombie.port
    p._reader = threading.Thread(target=p._read_frames, daemon=True)
    p._reader.start()
    try:
        deadline = time.monotonic() + 5
        while p._reader_sock is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert p._reader_sock is not None and zombie.connections == 1
        # The new session listens on another port, but the reader is stuck in a read
        # (what kept every reopen waiting for a first frame that never came)
        p.port = fresh.port
        time.sleep(1.5)
        assert p.frames_decoded == 0 and zombie.connections == 1
        p._drop_reader_conn()  # what reopen() does
        deadline = time.monotonic() + 10
        while p.frames_decoded == 0 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert p.frames_decoded >= 1 and fresh.connections == 1
        _, _, img = p.latest()
        assert img.shape == (64, 64, 3) and img[:, :20].mean() > 150 > img[:, 44:].mean()
    finally:
        p._stop.set()
        p._closing.set()
        p._drop_reader_conn()
        zombie.close()
        fresh.close()
