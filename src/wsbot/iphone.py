"""An iPhone over USB with pymobiledevice3 (iOS 27+): screen frames, touches, buttons, apps.

One pymobiledevice3 ScreenStreamServer (the engine behind `pymobiledevice3 ...
serve-web`) runs on our own asyncio thread and owns the phone's CoreDevice media
stream: RTCP feedback, keyframe recovery, stall restarts, keep-awake. That stream
is also what makes iOS accept injected HID reports:

- frames: we read its /stream.bin (HEVC access units) and decode them with PyAV
  in a thread; the newest frame is kept in memory (latest / wait_newer);
- touches: 58-byte mainTouchscreen reports (_ServiceID 257) through the stream
  server's own HID handles, which it reopens after every stream restart;
- Home / App Switcher: CoreDevice hardware-button events;
- apps: DVT process control (launch / kill / pid).

A browser viewer of the phone comes for free at http://127.0.0.1:<port>/ (a free port
per bot, logged at start; WSBOT_STREAM_PORT pins one).

Needs on the PC: the tunnel service (`pymobiledevice3 remote tunneld`, run as root on
Linux/macOS or from an Administrator terminal on Windows) and the Developer Disk Image mounted
(`pymobiledevice3 mounter auto-mount --tunnel ''`). On the phone: Developer Mode
and Settings > Developer > UI Automation. Coordinates are screen pixels of the
decoded frame (750x1334 on an iPhone SE, 1206x2622 on an iPhone 17).
"""

from __future__ import annotations

import asyncio
import contextlib
import http.client
import logging
import os
import socket
import sys
import threading
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

from .debug import dbg, snap
from .log import log

USB_MAX = 65535
CONTACT, RELEASE = 0xC2, 0x02
TOUCHSCREEN = 257  # mainTouchscreen _ServiceID
HOME = (0x0C, 0x40)  # consumer page, Menu = the Home button
BTN_DOWN, BTN_UP = 1, 2
START_CODE = b"\x00\x00\x00\x01"
STREAM_HEAD_MAX = 3_000_000
_HEAD_STARTED = threading.Event()  # DEVTEST: raw /stream.bin bytes kept for offline replay
DECODE_THREADS = int(os.environ.get("WSBOT_THREADS") or 1)
# Decode the phone's video on the GPU (Direct3D 11 / VideoToolbox / VA-API) when the
# computer has one, keeping frames there until the bot looks at one: ~half the decode
# CPU on an i5-6200U. "off" forces software; a device type name forces that one.
HW_DECODE = (os.environ.get("WSBOT_HWDECODE") or "auto").lower()
HW_FAIL_LIMIT = 20  # decode errors in a row on the GPU: fall back to software for good
_HW_BY_PLATFORM = {"win32": "d3d11va", "darwin": "videotoolbox", "linux": "vaapi"}


def _hw_device() -> str | None:
    if HW_DECODE == "off":
        return None
    want = _HW_BY_PLATFORM.get(sys.platform) if HW_DECODE == "auto" else HW_DECODE
    if not want:
        return None
    try:
        from av.codec.hwaccel import hwdevices_available

        return want if want in hwdevices_available() else None
    except Exception:
        return None


# No audio stream: pymobiledevice3 starts the phone's audio next to the video, like
# Xcode, as a session-liveness signal. The bot never listens, and without it the
# tunnel and Apple's USB service move ~30% fewer packets and the bot uses ~100 MB
# less memory (measured on Windows; video, touch and recovery unchanged).
# WSBOT_AUDIO=1 keeps it.
NO_AUDIO = not os.environ.get("WSBOT_AUDIO")


# EXPERIMENT (off by default): ask the phone for fewer frames. It streams ~58 fps
# while the game animates; receiving, relaying (tunneld + Apple's USB service) and
# decoding all scale with that. Its rate controller lowers the frame rate when the
# receiver reports one-way delay / jitter (an old pymobiledevice3 bug did exactly
# that by accident). WSBOT_STREAM_PACE="owrd=40,jitter=20" (ms) adds that much to
# what our RCTL feedback reports; the file local/stream_pace in the repo works too.
def _local_setting(name: str) -> str:
    """The text of local/<name> in the repo ("" if there's none): test switches."""
    f = Path(__file__).resolve().parents[2] / "local" / name
    return f.read_text().strip() if f.is_file() else ""


def _stream_pace() -> dict[str, int]:
    raw = os.environ.get("WSBOT_STREAM_PACE")
    if raw is None:
        raw = _local_setting("stream_pace")
    out: dict[str, int] = {}
    for part in raw.replace(";", ",").split(","):
        key, _, val = part.partition("=")
        if key.strip() in ("owrd", "jitter") and val.strip().isdigit():
            out[key.strip()] = int(val.strip())
    return out


def _paced_rctl(build, owrd_ms: int, jitter_ms: int):
    """Wraps ScreenStreamServer._build_rctl_packet: w4 = (arrival ms << 16) | jitter
    (24 kHz units) at bytes 24..27."""
    import struct

    def wrapped(self) -> bytes:
        pkt = bytearray(build(self))
        (w4,) = struct.unpack_from("!I", pkt, 24)
        arrival = ((w4 >> 16) + owrd_ms) & 0xFFFF
        jitter = min(0xFFFF, (w4 & 0xFFFF) + jitter_ms * 24)
        struct.pack_into("!I", pkt, 24, (arrival << 16) | jitter)
        return bytes(pkt)

    wrapped._wsbot_paced = True  # type: ignore[attr-defined]
    return wrapped


STALL_RESTART_S = 60.0  # no frames this long = a real encoder stall (pymobiledevice3: 5 s)
# After a lost packet the stream server holds every frame until the phone sends a
# keyframe. It asks once; when the phone ignores that, the picture stays frozen while
# the phone keeps streaming (so the stall watchdog never fires), and the bot acted on
# a stale screen for 20-60 s: "touches ignored" restart loops, Next Level tapped 10x.
REOPEN_RETRY_S = 5.0  # phone gone: try to reopen the session this often
KEY_RETRY_S = 1.5  # frozen this long: ask for a keyframe again (and every KEY_RETRY_S)
KEY_RESTART_S = 6.0  # still frozen: restart the stream session
KEY_RESTART_COOLDOWN_S = 20.0
WAKE_KEY_S = 4.0  # waking from quiet: wait this long for a keyframe, then restart the stream
# Stream restarts failing (CoreDevice error 24 = the phone out of file handles: a night
# with three bots on one phone froze the picture for 30 min while every restart failed)
# or a picture frozen this long: reopen the whole USB session. Reopens that don't hold
# (another freeze within ESCALATION_WINDOW_S, more than ESCALATION_REOPENS times) or
# can't reopen within REOPEN_GIVE_UP_S: stop the bot with an error, so AutomationHQ's
# auto-restart takes over instead of the bot looping silently.
STREAM_RESTART_FAILS = 3
TUNNEL_CONNECT_S = 15.0  # connecting to one tunnel listed by tunneld
# The stream server listens once its first stream start ends (it gives that 25 s)
LISTEN_WAIT_S = 30.0
FROZEN_REOPEN_S = 90.0
REOPEN_GIVE_UP_S = 180.0
REOPEN_MIN_TRIES = 3  # ...and at least this many tries (one can take ~2 min)
ESCALATION_REOPENS = 2
ESCALATION_WINDOW_S = 900.0
# Where the phone connection comes from. "userspace" = pymobiledevice3's in-process tunnel:
# no root, no separate tunneld window, and the phone's video (UDP) lands inside this
# process, so the macOS firewall can't drop it (it did: 0 RTP packets with tunneld on a
# Mac). "tunneld" = the root `pymobiledevice3 remote tunneld` service (Linux default).
TUNNEL_MODE = (
    os.environ.get("WSBOT_TUNNEL")
    or _local_setting("tunnel")  # the file local/tunnel in the repo, for tests
    or "auto"
).lower()  # auto | userspace | tunneld


def use_userspace_tunnel() -> bool:
    return TUNNEL_MODE == "userspace" or (TUNNEL_MODE == "auto" and sys.platform == "darwin")


def stream_port_setting() -> int | None:
    """WSBOT_STREAM_PORT: a fixed port for the viewer / stream server (default: a free
    one per bot, so two bots on one computer never share one: the second bot used to
    fail to bind 8090 and then read the FIRST bot's phone)."""
    try:
        port = int(os.environ.get("WSBOT_STREAM_PORT") or 0)
    except ValueError:
        return None
    return port if 0 < port < 65536 else None


def free_port() -> int:
    """A TCP port nothing listens on right now (the OS picks it)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _StreamLog(logging.Handler):
    """pymobiledevice3's stream warnings (restarts, stalls) into the bot log."""

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if record.exc_info and record.exc_info[1] is not None:
            msg += f" ({record.exc_info[1]!r})"
        if "AAC" not in msg:  # audio track noise: "AAC-ELD decode requires macOS"
            tag = "WARN" if record.levelno >= logging.WARNING else "DIAG"
            log(tag, f"iPhone stream: {msg}")


# INFO too (DEVTEST): the codec string and stream starts tell a new model's story.
_stream_logger = logging.getLogger("pymobiledevice3.remote.core_device.screen_stream")
if not any(isinstance(h, _StreamLog) for h in _stream_logger.handlers):
    _h = _StreamLog(logging.INFO)
    _stream_logger.addHandler(_h)
    if _stream_logger.getEffectiveLevel() > logging.INFO:
        _stream_logger.setLevel(logging.INFO)


class IPhoneError(RuntimeError):
    pass


class NoTunnel(IPhoneError):
    """No tunnel to the phone (unplugged, not trusted yet, or tunneld down)."""


# The CoreDevice screen stream and the USB touch it unlocks (Xcode's Device Hub) are new
# in iOS 27. Older phones have no tunnel (iOS 15-16) or no such stream (iOS 17-26), so
# without this check they waited forever for a tunnel or failed with "Developer Disk
# Image mounted?".
MIN_IOS = (27,)


class TooOld(IPhoneError):
    """The phone's iOS is older than this backend can work with."""


def ios_version(text: str | None) -> tuple[int, ...]:
    """ "26.4.1" -> (26, 4, 1); () when unknown."""
    parts = []
    for part in (text or "").split("."):
        if not part.isdigit():
            break
        parts.append(int(part))
    return tuple(parts)


def too_old(version: str | None) -> bool:
    """Below MIN_IOS. An unknown version isn't too old."""
    have = ios_version(version)
    return bool(have) and have < MIN_IOS


def check_ios(udid: str, version: str | None) -> None:
    """Raise TooOld for a phone below MIN_IOS."""
    if too_old(version):
        need = ".".join(map(str, MIN_IOS))
        raise TooOld(
            f"iPhone {udid or '?'} has iOS {version}; the bot's iPhone mode needs iOS {need} or "
            f"newer (it uses iOS {need}'s USB screen sharing and touch). Update the iPhone in "
            "Settings > General > Software Update, or play on an Android phone."
        )


def _annexb(au: bytes) -> bytes:
    """Length-prefixed NAL units (hvcC style, as /stream.bin sends them) -> Annex-B."""
    out, i = bytearray(), 0
    while i + 4 <= len(au):
        n = int.from_bytes(au[i : i + 4], "big")
        out += START_CODE + au[i + 4 : i + 4 + n]
        i += 4 + n
    return bytes(out)


def _nal_types(au: bytes) -> list[int]:
    """HEVC NAL unit types in a length-prefixed access unit (32-34 VPS/SPS/PPS, 19/20 IDR)."""
    out, i = [], 0
    while i + 5 <= len(au):
        n = int.from_bytes(au[i : i + 4], "big")
        out.append((au[i + 4] >> 1) & 0x3F)
        i += 4 + n
    return out


def _uncollapse(img: np.ndarray) -> np.ndarray:
    """Under load the encoder shrinks the picture into the top-left corner and pads
    the rest with flat gray (128); stretch the content back to full size, like the
    pymobiledevice3 viewer does."""
    h, w = img.shape[:2]
    step = 8
    # Almost every frame isn't collapsed: its last sampled column (or row) isn't padding,
    # which alone means "not collapsed" below (it makes cw >= w, or ch >= h). Checking
    # those first skips ~95% of the work (0.4 ms a frame).
    for edge in (img[::step, ((w - 1) // step) * step], img[((h - 1) // step) * step, ::step]):
        if (np.abs(edge.astype(np.int16) - 128) < 6).all(axis=1).mean() < 0.6:
            return img
    gray = (np.abs(img[::step, ::step].astype(np.int16) - 128) < 6).all(axis=2)
    cols = np.flatnonzero(gray.mean(axis=0) < 0.6)
    rows = np.flatnonzero(gray.mean(axis=1) < 0.6)
    if not len(cols) or not len(rows):
        return img
    cw, ch = (cols[-1] + 1) * step, (rows[-1] + 1) * step
    if 0.2 * w < cw < 0.92 * w and 0.2 * h < ch < 0.92 * h:
        return cv2.resize(img[:ch, :cw], (w, h), interpolation=cv2.INTER_LINEAR)
    return img


def _quiet_resets(loop: asyncio.AbstractEventLoop, context: dict) -> None:
    """Windows' proactor logs a full traceback each time the phone drops a connection
    (every stream restart): "ConnectionResetError: [WinError 10054]". Harmless; keep it
    out of the log, report everything else as usual."""
    if isinstance(context.get("exception"), ConnectionResetError):
        dbg(f"iphone loop: {context.get('message')}: connection reset")
        return
    loop.default_exception_handler(context)


class IPhone:
    tap_ms = 30  # contact time of a tap
    # Swipes: contact at the start, `steps` evenly timed samples, lift. 1.8 ms swipes
    # registered 400/400 in a test page; the game gets 50 ms (see ios_device.py).

    def __init__(self, udid: str | None = None, *, port: int | None = None):
        self.udid = udid or ""
        self._fixed_port = port or stream_port_setting()
        self.port = self._fixed_port or 0  # chosen on open (see free_port)
        # Gave up on the phone (see _escalate): why, and who to tell (the bot stops)
        self.fatal: str | None = None
        self.on_fatal = None
        self._escalating = False
        self._escalations: list[float] = []
        self.width, self.height = 750, 1334  # screen px; read from the phone on open
        self.product_type = ""  # "iPhone18,3" = iPhone 17 (see ios_device.IPHONE_NAMES)
        self._loop = self._new_loop()
        self._srv = None
        self._serve_task: asyncio.Task | None = None
        self._key_task: asyncio.Task | None = None
        self._log_task: asyncio.Task | None = None
        self._rsd = None
        self._tunnel: tuple[str, int] | None = None  # the phone's RSD address via tunneld
        self._held: tuple[int, int] | None = None
        self._us = None  # pymobiledevice3 UserspaceRsdTunnel while one is open
        # frames
        self._cond = threading.Condition()
        # The newest decoded frame (an av.VideoFrame), turned into a BGR array only when
        # someone asks (latest): the phone sends up to 60 a second, the bot looks at ~5.
        self._frame = None
        self._img: np.ndarray | None = None  # self._frame as BGR, once asked for
        # One BGR converter for every frame (see _to_bgr); used under self._cond only
        self._bgr = None
        self._quiet = False  # nobody looks: drop the phone's frames undecoded
        self._hw: str | bool | None = None  # GPU decoder: None = not tried, False = off
        self._hw_errors = 0
        self._want_key = False  # after a quiet spell: decode again from a keyframe
        self._seq = 0
        self._frame_t = 0.0
        self.stream_connected = False
        self.frames_decoded = 0
        # startup diagnostics: what the /stream.bin reader got so far
        self.stats = {
            "http": None,
            "aus": 0,
            "keys": 0,
            "bytes": 0,
            "decode_err": 0,
            "err": "",
            "size": None,
        }
        self._stop = threading.Event()
        self._closing = threading.Event()  # close() called: stop waiting for the phone
        self.cancel = threading.Event()  # the owner's stop key: stop waiting too
        self._reopen_lock = threading.Lock()
        self._opens = 0  # successful reopens, so waiting threads can tell one happened
        self._reader: threading.Thread | None = None
        self._reader_sock: socket.socket | None = None  # its /stream.bin connection
        self._head_bytes = 0
        self._aus_logged = 0
        self._published_logged = 0
        self._uncollapsed = 0
        self._touches = 0
        self._opened_t = 0.0
        threading.Thread(target=self._debug_loop, name="iphone-debug", daemon=True).start()

    # ---- DEVTEST debug ----------------------------------------------------------------

    def srv_state(self) -> dict:
        """Everything readable about pymobiledevice3's stream server right now."""
        srv = self._srv
        if srv is None:
            return {"srv": None}
        st: dict = {}
        for name in (
            "_active_service",
            "_active_session_id",
            "_display_id",
            "_sender_ip",
            "_rtcp_dest",
            "_local_ssrc",
            "_remote_ssrc",
            "_codec_string",
            "_rtp_packets_received",
            "_rtp_frames_received",
            "_rtp_highest_seq",
            "_stream_dirty",
            "_allow_rtcp_fb",
            "_ltrp_enabled",
            "_uhs",
            "_indigo",
        ):
            with contextlib.suppress(Exception):
                v = getattr(srv, name)
                st[name.lstrip("_")] = (
                    v if isinstance(v, (int, float, str, bool, type(None))) else type(v).__name__
                )
        with contextlib.suppress(Exception):
            st["init_sequence_len"] = (
                None if srv._init_sequence is None else len(srv._init_sequence)
            )
        with contextlib.suppress(Exception):
            st["stream_ready"] = srv._stream_ready.is_set()
        with contextlib.suppress(Exception):
            st["last_good_au_age"] = round(self._loop.time() - srv._last_good_au_t, 2)
        with contextlib.suppress(Exception):
            subs = list(srv._subscribers.values())
            st["subscribers"] = len(subs)
            st["subs_need_key"] = sum(1 for x in subs if x.needs_key)
        with contextlib.suppress(Exception):
            st["serve_task_done"] = self._serve_task.done() if self._serve_task else None
            if self._serve_task is not None and self._serve_task.done():
                st["serve_task_exc"] = repr(self._serve_task.exception())
        return st

    def _srv_state_of(self, srv) -> dict:
        old = self._srv
        self._srv = srv
        try:
            return self.srv_state()
        finally:
            self._srv = old

    def _debug_loop(self) -> None:
        last = (0, 0, time.monotonic())
        while True:
            time.sleep(10)
            with contextlib.suppress(Exception):
                now = time.monotonic()
                fps = (self.frames_decoded - last[0]) / (now - last[2])
                aups = (self.stats["aus"] - last[1]) / (now - last[2])
                last = (self.frames_decoded, self.stats["aus"], now)
                dbg(
                    f"iphone: decoded={self.frames_decoded} fps={fps:.1f} au/s={aups:.1f} "
                    f"stats={self.stats} connected={self.stream_connected} alive={self.alive} "
                    f"frozen={self.frozen} touches={self._touches} uncollapsed={self._uncollapsed} "
                    f"size={self.width}x{self.height} srv={self.srv_state()}"
                )

    # ---- session ------------------------------------------------------------------

    @staticmethod
    def _new_loop() -> asyncio.AbstractEventLoop:
        loop = asyncio.new_event_loop()
        # ScreenStreamServer.serve() installs Ctrl-C handlers; only the main thread may.
        loop.add_signal_handler = lambda *a, **k: None
        loop.set_exception_handler(_quiet_resets)
        threading.Thread(target=loop.run_forever, name="iphone-loop", daemon=True).start()
        return loop

    def _retire_loop(self) -> None:
        """Start the next session on a new event loop, like a fresh process would. The
        old one can hold tasks of a dead session that nothing ends: the stream server's
        own tunneld reconnect loop, its /stream.bin handlers (its teardown cancels them
        only when it runs to the end, and with a dead tunnel it may not), tunneld
        bridges. Those are cancelled (bounded) and the loop stopped."""
        old, self._loop = self._loop, self._new_loop()

        async def cancel_all() -> None:
            me = asyncio.current_task()
            tasks = [t for t in asyncio.all_tasks() if t is not me and not t.done()]
            for t in tasks:
                t.cancel()
            if tasks:
                await asyncio.wait(tasks, timeout=5)

        try:
            asyncio.run_coroutine_threadsafe(cancel_all(), old).result(10)
        except Exception as exc:
            dbg(f"old iphone loop cleanup: {exc!r}")
        old.call_soon_threadsafe(old.stop)

    def _drop_reader_conn(self) -> None:
        """Make the frame reader leave its /stream.bin connection now. It reads with no
        timeout (a still screen sends nothing for minutes), so a connection to a server
        that's gone but never closed its side would hold it, and no reopen would ever
        see a frame again; it reconnects to the new server's port by itself."""
        sock = self._reader_sock
        if sock is not None:
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
            if sys.platform == "win32":
                # Windows' shutdown doesn't wake a recv that's already waiting; closing
                # the handle does. sock.close() would wait for the reader's makefile to
                # let go, so close the handle itself (the reader then gets an OSError).
                with contextlib.suppress(OSError):
                    socket._socket.socket.close(sock)

    def _call(self, coro, timeout: float):
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return fut.result(timeout)
        except TimeoutError:
            fut.cancel()
            raise IPhoneError(f"phone call timed out after {timeout:.0f}s") from None

    def open(self, timeout: float = 60.0) -> None:
        seq0 = self._seq
        self._call(self._open(), timeout)
        if self._reader is None or not self._reader.is_alive():
            self._stop.clear()
            self._reader = threading.Thread(
                target=self._read_frames, name="iphone-frames", daemon=True
            )
            self._reader.start()
        # The first frame of THIS session (a reopen keeps the old one in memory: it
        # counted as "first frame", and the bot read a pre-unplug screen for minutes).
        t0 = time.monotonic()
        next_note = t0 + 5
        while not self._fresh(seq0) and time.monotonic() < t0 + 45:
            time.sleep(0.05)
            if time.monotonic() > next_note:
                next_note += 5
                log(
                    "DIAG",
                    f"waiting for the first frame {time.monotonic() - t0:.0f}s: {self.stats} "
                    f"server {self.srv_state()}",
                )
        if not self._fresh(seq0):
            log("ERROR", f"no frames after 45s: {self.stats} server {self.srv_state()}")
            with contextlib.suppress(Exception):
                c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
                c.request("GET", "/codec")
                r = c.getresponse()
                log("DIAG", f"/codec -> HTTP {r.status}: {r.read()[:300]!r}")
            raise IPhoneError(
                "the screen stream sends no frames (is the phone unlocked, screen on?)"
            )
        log("DIAG", f"first frame after {time.monotonic() - t0:.1f}s: {self.stats}")

    def set_quiet(self, quiet: bool) -> None:
        """Quiet: stop decoding (the bot sleeps or rests long, or AutomationHQ paused
        it); the phone keeps streaming and its frames are dropped, and the stream is
        never restarted for sending nothing (see _keyframe_watchdog). Waking waits up
        to a few seconds for a fresh keyframe, so nothing acts on the old picture, and
        restarts the stream if none comes."""
        if quiet == self._quiet:
            return
        if quiet:
            self._quiet = True
            return
        seq0 = self._seq
        self._quiet = False
        if not self._wait_key(seq0) and (srv := self._srv) is not None:
            log("RECOVERY", "iPhone stream: no keyframe after the quiet spell; restarting it")
            try:
                self._call(asyncio.wait_for(srv._ensure_fresh_stream(force=True), 10.0), 15.0)
            except Exception as exc:
                log("WARN", f"iPhone stream restart failed: {exc!r}")
            self._wait_key(seq0)

    def _wait_key(self, seq0: int) -> bool:
        """Ask for a keyframe until a frame newer than `seq0` is decoded (or WAKE_KEY_S)."""
        deadline = time.monotonic() + WAKE_KEY_S
        while self._seq == seq0 and time.monotonic() < deadline and not self._given_up():
            self._ask_key()
            with self._cond:
                self._cond.wait_for(lambda: self._seq > seq0, min(1.0, WAKE_KEY_S))
        return self._seq != seq0

    def _ask_key(self) -> None:
        srv = self._srv
        if srv is None:
            return
        with contextlib.suppress(Exception):  # rate-limited by the server itself
            self._loop.call_soon_threadsafe(lambda: srv._request_recovery_idr(reason="wsbot-wake"))

    def _fresh(self, seq0: int) -> bool:
        return self._frame is not None and self._seq > seq0

    def close(self) -> None:
        self._closing.set()
        self._stop.set()
        with contextlib.suppress(Exception):
            self._call(self._close(), 60)
        self._drop_reader_conn()

    def reopen(self, give_up_after: float | None = None) -> None:
        """Rebuild the session. If the phone is gone (the tunnel dropped for a while),
        wait for it to come back instead of raising: a raise here killed a 16 h run.
        One thread reopens at a time; the others wait for it and use its session.
        give_up_after: raise IPhoneError if it isn't back after this many seconds."""
        gen = self._opens
        with self._reopen_lock:
            if self._opens != gen and self.alive:
                return  # another thread just reopened it
            t0 = time.monotonic()
            next_note = 0.0
            tries = 0
            while True:
                log("RECOVERY", "reopening the iPhone USB session")
                with contextlib.suppress(Exception):
                    self._call(self._close(), 60)
                self._retire_loop()
                self._drop_reader_conn()
                tries += 1
                try:
                    self.open()
                    self._opens += 1
                    return
                except Exception as exc:  # NoTunnel, no frames, OSError from the tunnel
                    if self._given_up() or isinstance(exc, TooOld):
                        raise
                    waited = time.monotonic() - t0
                    if (
                        give_up_after is not None
                        and waited >= give_up_after
                        and tries >= REOPEN_MIN_TRIES
                    ):
                        raise IPhoneError(
                            f"not back after {waited:.0f}s and {tries} tries: {exc}"
                        ) from exc
                    if waited >= next_note:
                        log("WARN", f"iPhone not back yet ({waited:.0f}s): {exc}")
                        next_note = waited + 60
                deadline = time.monotonic() + REOPEN_RETRY_S
                while time.monotonic() < deadline:
                    if self._given_up():
                        raise IPhoneError("stopped while waiting for the iPhone")
                    time.sleep(0.25)

    def _given_up(self) -> bool:
        return self._closing.is_set() or self.cancel.is_set()

    @property
    def alive(self) -> bool:
        """The stream server runs (the frame reader reconnects to it on its own)."""
        return (
            self._srv is not None and self._serve_task is not None and not self._serve_task.done()
        )

    async def _open(self) -> None:
        from pymobiledevice3.remote.core_device import screen_stream
        from pymobiledevice3.remote.core_device.screen_stream import ScreenStreamServer

        # The phone sends frames only when the screen changes, so the server's 5 s
        # "no frames = stalled" watchdog restarted the stream whenever the board sat
        # still (the bot thinking, a probe) and dropped the touches sent meanwhile.
        screen_stream._STALL_RESTART_SECS = STALL_RESTART_S
        pace = _stream_pace()
        build = getattr(ScreenStreamServer, "_build_rctl_packet", None) if pace else None
        if build is not None and not getattr(build, "_wsbot_paced", False):
            ScreenStreamServer._build_rctl_packet = _paced_rctl(  # type: ignore[method-assign]
                build, pace.get("owrd", 0), pace.get("jitter", 0)
            )
            log("LOG", f"diag: stream pacing experiment on: {pace}")

        await self._close()
        t_open = time.monotonic()
        await self._check_usb_versions()
        rsds = []
        if use_userspace_tunnel():
            try:
                rsds = [await self._userspace_rsd()]
            except Exception as exc:
                if type(exc).__name__ == "NoDeviceConnectedError":
                    raise NoTunnel(
                        "no iPhone on USB: plug it into this computer with a cable, unlock it, "
                        "and tap Trust if it asks"
                    ) from exc
                log("WARN", f"built-in USB tunnel failed ({exc!r}); trying tunneld")
                dbg(f"userspace tunnel traceback: {traceback.format_exc()}")
        if not rsds:
            log("DIAG", "using the tunneld service")
            rsds = await self._tunneld_rsds()
        for r in rsds:
            props = {}
            with contextlib.suppress(Exception):
                props = {
                    k: v
                    for k, v in r.peer_info["Properties"].items()
                    if k
                    in (
                        "ProductType",
                        "OSVersion",
                        "BuildVersion",
                        "HardwareModel",
                        "DeviceClass",
                        "ProductName",
                        "CPUArchitecture",
                        "ChipID",
                        "BoardId",
                        "ModelNumber",
                        "OSInstallEnvironment",
                    )
                }
            log(
                "DIAG",
                f"tunnel device {getattr(r, 'udid', '?')} "
                f"{getattr(r, 'product_type', '?')} iOS {getattr(r, 'product_version', '?')} "
                f"{props}",
            )
            with contextlib.suppress(Exception):
                dbg(f"rsd services for {r.udid}: {sorted(r.peer_info.get('Services', {}))}")
        matching = [r for r in rsds if not self.udid or r.udid == self.udid]
        # Without a UDID, prefer a phone that's new enough over an older one.
        rsd = next(
            (r for r in matching if not too_old(getattr(r, "product_version", None))),
            matching[0] if matching else None,
        )
        for r in rsds:
            if r is not rsd:
                await r.close()
        if rsd is None:
            raise NoTunnel(
                f"no USB tunnel for {self.udid or 'an iPhone'}: is it plugged in and trusted, and "
                "is `pymobiledevice3 remote tunneld` running (as root / Administrator)?"
            )
        from pymobiledevice3.remote.core_device.device_info import DeviceInfoService

        try:
            check_ios(rsd.udid, getattr(rsd, "product_version", None))
        except TooOld:
            await rsd.close()
            raise
        with contextlib.suppress(Exception):
            self.product_type = rsd.product_type or ""
        self._note_tunnel(rsd)

        try:
            async with DeviceInfoService(rsd) as info:
                display = await info.get_display_info()
                dbg(f"display info (full): {display!r}"[:20000])
                mode = display["displays"][0]["currentMode"]["size"]
                self.width, self.height = round(mode[0]), round(mode[1])
                log("DIAG", f"iPhone display mode {mode} ({len(display['displays'])} displays)")
        except Exception as exc:
            log("WARN", f"display info failed ({exc!r}); assuming {self.width}x{self.height}")
            dbg(f"display info traceback: {traceback.format_exc()}")
        self.port = self._fixed_port or free_port()
        srv = ScreenStreamServer(rsd, bind="127.0.0.1", http_port=self.port)
        if NO_AUDIO:

            async def no_audio() -> None:
                return None

            srv._ensure_audio_stream = no_audio
        task = asyncio.create_task(srv.serve(), name="screen-stream")
        try:
            deadline = time.monotonic() + 40
            # Stream session up = the touch gate is open. Its first keyframe only comes
            # once a viewer asks (the frame reader's /stream.bin sends the request).
            while srv._active_service is None:
                if task.done():
                    task.result()
                    raise IPhoneError("screen stream server exited during startup")
                if time.monotonic() > deadline:
                    raise IPhoneError(
                        "screen stream did not start (Developer Disk Image mounted? "
                        "`pymobiledevice3 mounter auto-mount --tunnel ''`)"
                    )
                await asyncio.sleep(0.1)
            log(
                "DIAG",
                f"stream session up after {time.monotonic() - t_open:.1f}s: "
                f"{self._srv_state_of(srv)}",
            )
            await srv._ensure_hid()
            log("DIAG", f"HID handles up after {time.monotonic() - t_open:.1f}s")
            await self._wait_listening(task)
        except BaseException as exc:
            log("ERROR", f"iPhone open failed: {exc!r}")
            dbg(f"open traceback: {traceback.format_exc()}")
            task.cancel()
            with contextlib.suppress(BaseException):
                await asyncio.wait_for(task, 15)
            if self._us is not None:
                tunnel, self._us = self._us, None
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(tunnel.aclose(), 15)
            else:
                await rsd.close()
            raise
        self._srv, self._serve_task, self._rsd = srv, task, rsd
        self._opened_t = time.monotonic()
        self._key_task = asyncio.create_task(self._keyframe_watchdog(srv), name="keyframe-watchdog")
        self._log_task = asyncio.create_task(self._stream_log(), name="stream-log")
        self._held = None
        self.udid = rsd.udid
        log(
            "DEVICE",
            f"iPhone {rsd.udid}: USB stream + touch up, viewer http://127.0.0.1:{self.port}/",
        )

    async def _tunneld_rsds(self) -> list:
        """The phone's RSD through the tunneld helper, asked for afresh on every open:
        after the helper restarts, the phone's tunnel comes back at a new address and
        port (fd..::1). Each listed tunnel is tried on its own, with a timeout: one stale
        or half-made tunnel (pymobiledevice3's get_tunneld_devices lets an OSError from
        one end the whole list, and has no connect timeout) must not hide the good one."""
        from pymobiledevice3.remote.remote_service_discovery import RemoteServiceDiscoveryService
        from pymobiledevice3.tunneld.api import get_tunneld_tunnels

        try:
            tunnels = await asyncio.wait_for(get_tunneld_tunnels(), 10)
        except Exception as exc:
            raise NoTunnel(
                "the iPhone helper (`pymobiledevice3 remote tunneld`) isn't answering: is it "
                f"running? ({exc!r})"
            ) from exc
        dbg(f"tunneld lists: {tunnels}")
        rsds = []
        for udid, entries in tunnels.items():
            if self.udid and udid != self.udid:
                continue
            for entry in reversed(entries):  # the newest last
                address = (entry["tunnel-address"], entry["tunnel-port"])
                rsd = RemoteServiceDiscoveryService(
                    address,
                    name=entry.get("interface"),
                    auxiliary_metadata=entry.get("auxiliary-metadata"),
                )
                try:
                    await asyncio.wait_for(rsd.connect(), TUNNEL_CONNECT_S)
                except Exception as exc:
                    log("WARN", f"iPhone tunnel {address} doesn't answer ({exc!r}); skipping it")
                    with contextlib.suppress(Exception):
                        await asyncio.wait_for(rsd.close(), 5)
                    continue
                rsds.append(rsd)
                break  # one per phone
        return rsds

    def _note_tunnel(self, rsd) -> None:
        """Log where the phone's tunnel is, and when it moved (the helper restarted)."""
        address = getattr(getattr(rsd, "service", None), "address", None)
        if address is None:
            return
        address = tuple(address)
        if self._tunnel is not None and address != self._tunnel:
            log(
                "RECOVERY",
                f"the iPhone's tunnel moved from {self._tunnel} to {address} (rebuilt: the "
                "iPhone helper restarted, or the phone reconnected); using the new one",
            )
        else:
            log("DIAG", f"iPhone tunnel at {address}")
        self._tunnel = address

    async def _wait_listening(self, task: asyncio.Task) -> None:
        """The stream server opens its HTTP port only after the phone's stream is up.
        Make sure it is OUR server on self.port: if the port is taken its task dies,
        and reading from the port would show another bot's phone."""
        deadline = time.monotonic() + LISTEN_WAIT_S
        while time.monotonic() < deadline:
            if task.done():
                try:
                    task.result()
                    why = "it exited"
                except BaseException as exc:
                    why = repr(exc)
                raise IPhoneError(f"screen stream server can't listen on port {self.port}: {why}")
            with contextlib.suppress(OSError, TimeoutError):
                _, writer = await asyncio.wait_for(
                    asyncio.open_connection("127.0.0.1", self.port), 1.0
                )
                writer.close()
                await asyncio.sleep(0.2)  # a failed bind ends the task right away
                if not task.done():
                    return
            await asyncio.sleep(0.1)
        raise IPhoneError(f"screen stream server isn't listening on port {self.port}")

    async def _check_usb_versions(self) -> None:
        """Before any tunnel: is the phone (or, without a UDID, every iPhone on USB) new
        enough? Read over plain lockdown without pairing, so it never asks for Trust. Any
        failure here is ignored: the tunnel path reports it as before."""
        from pymobiledevice3.lockdown import create_using_usbmux
        from pymobiledevice3.usbmux import list_devices

        versions: dict[str, str] = {}
        with contextlib.suppress(Exception):
            for dev in await asyncio.wait_for(list_devices(), 10):
                if not dev.is_usb or (self.udid and dev.serial != self.udid):
                    continue
                with contextlib.suppress(Exception):
                    lockdown = await asyncio.wait_for(
                        create_using_usbmux(dev.serial, autopair=False, connection_type="USB"),
                        10,
                    )
                    try:
                        versions[dev.serial] = lockdown.all_values.get("ProductVersion") or ""
                    finally:
                        await lockdown.close()
        dbg(f"iPhones on USB by iOS version: {versions}")
        # Without a UDID the first tunnel wins, so only refuse when no phone could work.
        if versions and all(too_old(v) for v in versions.values()):
            udid, version = next(iter(versions.items()))
            check_ios(udid, version)

    async def _userspace_rsd(self):
        from pymobiledevice3.remote.userspace_tunnel import UserspaceRsdTunnel

        t0 = time.monotonic()
        tunnel = UserspaceRsdTunnel(serial=self.udid or None)
        rsd = await asyncio.wait_for(tunnel.aopen(), 45)
        self._us = tunnel
        log(
            "DIAG",
            f"built-in USB tunnel up in {time.monotonic() - t0:.1f}s "
            f"(stack addr {getattr(tunnel.tun, 'addr', '?')})",
        )
        return rsd

    async def _close(self) -> None:
        task, rsd = self._serve_task, self._rsd
        # The stream server reconnects through tunneld by itself (a dropped tunnel) and
        # then holds a newer RSD than ours: close that one too.
        rebound = getattr(self._srv, "_rsd", None)
        rebound = rebound if rebound is not None and rebound is not rsd else None
        for t in (self._key_task, self._log_task):
            if t is not None:
                t.cancel()
        self._key_task = self._log_task = None
        self._srv = self._serve_task = self._rsd = None
        if task is not None:
            task.cancel()  # serve() stops the device-side streams in its finally
            # asyncio.wait, not wait_for: on a timeout wait_for cancels serve() again,
            # in the middle of its teardown (each step there waits out a dead tunnel),
            # and its last step, closing the /stream.bin connections, never ran.
            with contextlib.suppress(BaseException):
                await asyncio.wait({task}, timeout=30)
        if self._us is not None:  # the tunnel owns its RSD; a reopen builds a fresh one
            tunnel, self._us = self._us, None
            with contextlib.suppress(Exception):
                await asyncio.wait_for(tunnel.aclose(), 15)
        elif rsd is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(rsd.close(), 5)
        if rebound is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(rebound.close(), 5)

    # ---- frozen picture ---------------------------------------------------------------

    def _frozen_since(self, srv) -> float | None:
        """Loop time since which our /stream.bin subscriber waits for a keyframe."""
        try:
            states = list(srv._subscribers.values())
        except (AttributeError, RuntimeError):
            return None
        waiting = [s.needs_key_since for s in states if s.needs_key]
        return min(waiting) if waiting else None

    @property
    def frozen(self) -> bool:
        """The latest frame may be stale: frames are held back until a keyframe."""
        srv = self._srv
        if srv is None:
            return False
        since = self._frozen_since(srv)
        return since is not None and time.monotonic() - since > 0.3

    async def _stream_log(self) -> None:
        """The phone's frame rate and data rate in the log every 10 minutes."""
        loop = asyncio.get_running_loop()
        last_t, last_aus, last_bytes = loop.time(), self.stats["aus"], self.stats["bytes"]
        while True:
            await asyncio.sleep(600)
            now, aus, nbytes = loop.time(), self.stats["aus"], self.stats["bytes"]
            log(
                "DIAG",
                f"stream: phone sent {(aus - last_aus) / (now - last_t):.1f} frames/s "
                f"({(nbytes - last_bytes) * 8 / 1000 / (now - last_t):.0f} kbps), "
                f"decoder {self._hw or 'cpu'}, {self._uncollapsed} shrunken so far",
            )
            last_t, last_aus, last_bytes = now, aus, nbytes

    async def _keyframe_watchdog(self, srv) -> None:
        loop = asyncio.get_running_loop()
        episode: float | None = None
        last_ask = last_restart = 0.0
        fails = 0  # stream restarts failed in a row (this episode)
        while True:
            await asyncio.sleep(0.25)
            now = loop.time()
            if self._quiet:
                # Nobody looks (a break, a long rest, paused), and a still screen sends
                # no frames: pymobiledevice3's stall watchdog then restarted the stream
                # every ~68 s for the whole break ("no AU progress ... restarting
                # stream"), burning CPU and USB for nothing. Hold its clock while quiet;
                # waking asks for a keyframe and restarts the stream if none comes.
                srv._last_good_au_t = max(srv._last_good_au_t, now)
                episode, fails = None, 0
                continue
            since = self._frozen_since(srv)
            if since is None:
                if episode is not None and now - episode > KEY_RETRY_S:
                    log("LOG", f"iPhone stream: picture was frozen {now - episode:.1f}s, recovered")
                episode, fails = None, 0
                continue
            episode = since if episode is None else min(episode, since)
            age = now - episode
            restarted = age > KEY_RESTART_S and now - last_restart > KEY_RESTART_COOLDOWN_S
            if restarted:
                last_restart = now
                log("RECOVERY", f"iPhone stream frozen {age:.0f}s (no keyframe); restarting it")
                try:
                    await asyncio.wait_for(srv._ensure_fresh_stream(force=True), timeout=10.0)
                except Exception as exc:
                    fails += 1
                    log("WARN", f"iPhone stream restart failed ({fails}x): {exc!r}")
            if fails >= STREAM_RESTART_FAILS or age > FROZEN_REOPEN_S:
                self._escalate(f"picture frozen {age:.0f}s, {fails} stream restarts failed")
                return  # the reopened session runs its own watchdog
            if restarted:
                continue
            if age > KEY_RETRY_S and now - last_ask > KEY_RETRY_S:
                last_ask = now
                with contextlib.suppress(Exception):
                    srv._request_recovery_idr(reason="wsbot-frozen")

    def _escalate(self, why: str) -> None:
        """The stream can't heal itself: reopen the whole USB session, from a thread of
        its own (reopen() waits on this event loop). Give up when that doesn't hold."""
        if self._escalating or self._given_up() or self.fatal:
            return
        self._escalating = True
        threading.Thread(
            target=self._escalate_run, args=(why,), name="iphone-escalate", daemon=True
        ).start()

    def _escalate_run(self, why: str) -> None:
        try:
            now = time.monotonic()
            self._escalations = [t for t in self._escalations if now - t < ESCALATION_WINDOW_S]
            if len(self._escalations) >= ESCALATION_REOPENS:
                self._give_up(
                    f"the iPhone screen stream keeps freezing ({why}; reopened "
                    f"{len(self._escalations)}x in {ESCALATION_WINDOW_S / 60:.0f} min)"
                )
                return
            self._escalations.append(now)
            log("RECOVERY", f"iPhone stream: {why}; reopening the USB connection")
            try:
                self.reopen(give_up_after=REOPEN_GIVE_UP_S)
            except Exception as exc:
                if not self._given_up():
                    self._give_up(f"the iPhone screen stream is dead ({why}); reopen failed: {exc}")
        finally:
            self._escalating = False

    def _give_up(self, message: str) -> None:
        if self.fatal is not None:
            return
        self.fatal = message
        handler = self.on_fatal
        if handler is None:
            log("ERROR", message)
        else:
            handler(message)

    # ---- frames ---------------------------------------------------------------------

    def _read_frames(self) -> None:
        while not self._stop.is_set():
            try:
                self._stream_once()
            except Exception as exc:
                self.stats["err"] = f"{type(exc).__name__}: {exc}"[:200]
                dbg(f"stream reader error: {traceback.format_exc()}")
                if not self.frames_decoded:
                    log("DIAG", f"stream reader: {type(exc).__name__}: {exc}")
                if not self._stop.is_set() and self.frames_decoded:
                    log("WARN", f"iPhone screen stream dropped ({type(exc).__name__}: {exc})")
            self.stream_connected = False
            self._stop.wait(1.0)

    def _stream_once(self) -> None:
        import av

        conn = http.client.HTTPConnection("127.0.0.1", self.port)  # a static screen is silent
        try:
            conn.connect()
            self._reader_sock = conn.sock
            conn.request("GET", "/stream.bin")
            resp = conn.getresponse()
            self.stats["http"] = resp.status
            if resp.status != 200:
                body = resp.read()[:300]
                raise ConnectionError(f"/stream.bin -> HTTP {resp.status} {body!r}")
            dbg(f"/stream.bin connected, headers {resp.getheaders()}")
            self.stream_connected = True
            codec = None
            while not self._stop.is_set():
                head = resp.read(4)
                if len(head) < 4:
                    raise ConnectionError("stream closed")
                body = resp.read(int.from_bytes(head, "big"))
                kind, au = body[0], body[1:]
                self.stats["aus"] += 1
                self.stats["bytes"] += len(body)
                self.stats["keys"] += kind != 0
                self._keep_head(head, body)
                if self._aus_logged < 60 or kind != 0:
                    self._aus_logged += 1
                    dbg(
                        f"AU #{self.stats['aus']} kind={kind} bytes={len(au)} nals={_nal_types(au)}"
                    )
                if self._quiet:
                    self._want_key = True
                    continue
                if self._want_key:  # frames skipped: only a keyframe can restart decoding
                    if kind == 1:
                        continue
                    self._want_key, codec = False, None
                if codec is None or kind == 2:  # 2 = keyframe after a restart: fresh decoder
                    dbg(f"new HEVC decoder (kind={kind})")
                    codec = self._decoder(av)
                try:
                    decoded = codec.decode(av.Packet(_annexb(au)))
                    self._hw_errors = 0
                except av.error.FFmpegError as exc:
                    self.stats["decode_err"] += 1
                    if self._hw:
                        self._hw_errors += 1
                        if self._hw_errors >= HW_FAIL_LIMIT:
                            log("WARN", f"GPU video decoding keeps failing ({exc}); using the CPU")
                            self._hw, codec, self._want_key = False, None, True
                            self._ask_key()
                            continue
                    self.stats["err"] = f"decode: {exc}"[:200]
                    if self.stats["decode_err"] <= 50:
                        dbg(
                            f"decode error #{self.stats['decode_err']} on AU kind={kind} "
                            f"nals={_nal_types(au)}: {exc!r}"
                        )
                    continue
                for frame in decoded:
                    if self.stats["size"] is None:
                        self.stats["size"] = f"{frame.width}x{frame.height} {frame.format.name}"
                        log("DIAG", f"iPhone first decoded frame {self.stats['size']}")
                        with contextlib.suppress(Exception):
                            cc = codec
                            dbg(
                                f"decoder: profile={getattr(cc, 'profile', '?')} "
                                f"pix_fmt={getattr(cc, 'pix_fmt', '?')} "
                                f"{getattr(cc, 'width', '?')}x{getattr(cc, 'height', '?')}"
                            )
                    self._publish(frame)
        finally:
            self._reader_sock = None
            conn.close()

    def _decoder(self, av):
        """A fresh HEVC decoder: on the GPU when there is one (see HW_DECODE)."""
        codec = None
        if self._hw is None:
            self._hw = _hw_device() or False
        if self._hw:
            try:
                from av.codec.hwaccel import HWAccel

                hw = HWAccel(device_type=self._hw, allow_software_fallback=True, is_hw_owned=True)
                codec = av.CodecContext.create("hevc", "r", hwaccel=hw)
            except Exception as exc:
                log("DIAG", f"GPU video decoding ({self._hw}) unavailable: {exc!r}; using the CPU")
                self._hw = False
        if codec is None:
            codec = av.CodecContext.create("hevc", "r")
        # Not "AUTO": frame-threading workers hold our packets, and freeing the old
        # decoder (a restart, shutdown) then deadlocks on the GIL.
        codec.thread_type = "SLICE"
        # One bot per phone on one computer: no decoder thread per core each.
        codec.thread_count = DECODE_THREADS
        return codec

    def _keep_head(self, head: bytes, body: bytes) -> None:
        """Raw /stream.bin bytes (same framing) for offline replay: diagnostics/debug."""
        if self._head_bytes >= STREAM_HEAD_MAX:
            return
        with contextlib.suppress(Exception):
            from .debug import _root

            if _root is not None:
                # "wb" on this process's first write: one run per file, so it fits the report
                mode = "ab" if _HEAD_STARTED.is_set() else "wb"
                _HEAD_STARTED.set()
                with open(_root / "diagnostics" / "debug" / "stream_head.bin", mode) as f:
                    f.write(head + body)
                self._head_bytes += len(head) + len(body)

    def _publish(self, frame) -> None:
        """Keep the newest decoded frame; converting it waits until it's asked for."""
        with self._cond:
            self._frame, self._img = frame, None
            self._seq += 1
            self._frame_t = time.monotonic()
            self.frames_decoded += 1
            self._cond.notify_all()

    def _to_bgr(self, frame) -> np.ndarray:
        # frame.to_ndarray(format="bgr24") set up a new converter for every frame, and
        # one that splits each picture over a thread per core: ~3.5 ms of CPU a frame
        # on 16 cores. One kept converter on this thread: ~0.4 ms, the same pixels.
        if self._bgr is None:
            from av.video.reformatter import VideoReformatter

            self._bgr = VideoReformatter()
        img = self._bgr.reformat(frame, format="bgr24", threads=1).to_ndarray()
        h, w = img.shape[:2]
        raw_shape = img.shape
        snap("phone_raw_decoded", img, every_s=60)
        if 0 <= w - self.width < 64 and 0 <= h - self.height < 64:
            # HEVC codes whole blocks: 750x1334 arrives as 752x1344, the iPhone 17's
            # 1206x2622 as 1216x2656, with black padding right and bottom that the stream
            # doesn't mark for cropping. Scaling it in instead squashed the picture 1.3%.
            img = img[: self.height, : self.width]
        before = img
        img = _uncollapse(img)
        if img is not before:
            self._uncollapsed += 1
            if self._uncollapsed <= 20 or self._uncollapsed % 100 == 0:
                dbg(f"uncollapsed a gray-padded frame (#{self._uncollapsed})")
        if self._published_logged < 5:
            self._published_logged += 1
            dbg(f"publish: decoded {raw_shape} -> {img.shape} (screen {self.width}x{self.height})")
        if img.shape[1] != self.width or img.shape[0] != self.height:
            img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return img

    def latest(self) -> tuple[int, float, np.ndarray]:
        """(sequence number, time.monotonic() it arrived, BGR frame)."""
        with self._cond:
            if self._frame is None:
                raise IPhoneError("no frame from the iPhone yet")
            if self._img is None:  # a few ms, under the lock: one conversion per frame
                self._img = self._to_bgr(self._frame)
            return self._seq, self._frame_t, self._img

    def wait_newer(self, seq: int, timeout: float) -> bool:
        """The phone sends frames only when the screen changes."""
        with self._cond:
            return self._cond.wait_for(lambda: self._seq > seq, timeout)

    # ---- input ------------------------------------------------------------------------

    def _u(self, x: float, y: float) -> tuple[int, int]:
        """Screen px -> the touchscreen's 0..65535 range."""
        w, h = self.width, self.height
        return (
            min(max(round(x * USB_MAX / (w - 1)), 0), USB_MAX),
            min(max(round(y * USB_MAX / (h - 1)), 0), USB_MAX),
        )

    async def _send(self, state: int, x: int, y: int) -> None:
        # Mid stream-restart the HID handles belong to the dying session: a touch sent
        # then is dropped, and handles reopened then stay dead. Wait for the new stream.
        deadline = time.monotonic() + 15
        while self._srv._active_service is None and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        await self._srv._ensure_hid()  # no-op unless a stream restart dropped the handles
        await self._srv._uhs.send_touchscreen(state, x, y, service_id=TOUCHSCREEN)
        self._touches += 1
        if self._touches <= 400 or self._touches % 50 == 0:
            dbg(
                f"touch #{self._touches} {'DOWN' if state == CONTACT else 'UP'} u=({x},{y}) "
                f"px=({x * (self.width - 1) / USB_MAX:.0f},{y * (self.height - 1) / USB_MAX:.0f})"
            )

    async def _down(self, x: int, y: int) -> None:
        """A new touch. One still held (a call that timed out or was cancelled after its
        touch-down) is lifted first: otherwise this one reads as dragging it here."""
        if self._held:
            await self._up()
        await self._move(x, y)

    async def _move(self, x: int, y: int) -> None:
        """The finger that is down moves here (or goes down here)."""
        self._held = (x, y)
        await self._send(CONTACT, x, y)

    async def _up(self) -> None:
        if self._held:
            x, y = self._held
            self._held = None
            await self._send(RELEASE, x, y)

    async def _lift(self) -> None:
        """_up for a finally: never hides the error that got us there, never hangs."""
        try:
            await asyncio.wait_for(self._up(), 5)
        except Exception as exc:  # the touch stays marked held: the next _down lifts it
            dbg(f"touch release failed: {exc!r}")

    async def _tap(self, x: int, y: int, hold_ms: float) -> None:
        try:
            await self._down(x, y)
            await asyncio.sleep(hold_ms / 1000)
        finally:
            await self._lift()

    async def _swipe(self, x1: int, y1: int, x2: int, y2: int, ms: float, steps: int) -> None:
        steps = max(1, steps)
        try:
            await self._down(x1, y1)
            t0 = time.perf_counter()
            for i in range(1, steps + 1):
                delay = t0 + ms / 1000 * i / steps - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
                await self._move(
                    round(x1 + (x2 - x1) * i / steps), round(y1 + (y2 - y1) * i / steps)
                )
        finally:
            await self._lift()

    async def _press(self, button: tuple[int, int], ms: float = 60) -> None:
        await self._srv._ensure_hid()
        await self._srv._indigo.send_button(*button, BTN_DOWN)
        await asyncio.sleep(ms / 1000)
        await self._srv._indigo.send_button(*button, BTN_UP)

    async def _app(self, op: str, bundle: str) -> int:
        from pymobiledevice3.services.dvt.instruments.dvt_provider import DvtProvider
        from pymobiledevice3.services.dvt.instruments.process_control import ProcessControl

        async with DvtProvider(self._live_rsd()) as dvt, ProcessControl(dvt) as pc:
            if op == "launch":
                return await pc.launch(bundle, kill_existing=True)
            pid = await pc.process_identifier_for_bundle_identifier(bundle)
            if op == "kill" and pid:
                await pc.kill(pid)
            return pid

    def _live_rsd(self):
        """The RSD the stream server uses now: after a tunnel drop it reconnects through
        tunneld by itself, and ours then points at the dead tunnel."""
        return getattr(self._srv, "_rsd", None) or self._rsd

    def _do(self, coro_fn, timeout: float = 30.0):
        """Run one input coroutine; reopen the session once if the phone went away."""
        for attempt in (1, 2):
            try:
                if not self.alive:
                    self.reopen()
                return self._call(coro_fn(), timeout)
            except IPhoneError:
                raise
            except Exception as exc:
                dbg(f"input attempt {attempt} failed: {traceback.format_exc()}")
                if attempt == 2:
                    raise IPhoneError(f"iPhone input failed: {exc!r}") from exc
                log("WARN", f"iPhone input failed ({exc!r}); reopening")
                self.reopen()

    def tap(self, x: float, y: float, hold_ms: float | None = None) -> None:
        u = self._u(x, y)
        self._do(lambda: self._tap(*u, self.tap_ms if hold_ms is None else hold_ms))

    def swipe(
        self, x1: float, y1: float, x2: float, y2: float, ms: float, steps: int | None = None
    ) -> None:
        a, b = self._u(x1, y1), self._u(x2, y2)
        n = steps or max(1, round(ms / 8))  # ~120 Hz samples, like a finger
        self._do(lambda: self._swipe(*a, *b, ms, n))

    def hold(self, points: list[tuple[float, float]], step_ms: float = 20) -> None:
        """Touch down at points[0] and drag through the rest without lifting."""
        us = [self._u(x, y) for x, y in points]

        async def go() -> None:
            await self._down(*us[0])
            for u in us[1:]:
                await asyncio.sleep(step_ms / 1000)
                await self._move(*u)

        self._do(go)

    def release(self) -> None:
        self._do(self._up)

    def home(self) -> None:
        self._do(lambda: self._press(HOME))

    def app(self, op: str, bundle: str) -> int:
        """op = launch (kills a running copy first) | kill | pid; returns the pid (0 = none)."""
        t = time.monotonic()
        try:
            pid = self._do(lambda: self._app(op, bundle), timeout=30)
        except Exception as exc:
            log("DIAG", f"app {op} {bundle} failed after {time.monotonic() - t:.1f}s: {exc!r}")
            raise
        dbg(f"app {op} {bundle} -> {pid} in {time.monotonic() - t:.2f}s")
        return pid
