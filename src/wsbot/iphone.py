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

A browser viewer of the phone comes for free at http://127.0.0.1:<port>/.

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
import sys
import threading
import time
import traceback

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
STALL_RESTART_S = 60.0  # no frames this long = a real encoder stall (pymobiledevice3: 5 s)
# After a lost packet the stream server holds every frame until the phone sends a
# keyframe. It asks once; when the phone ignores that, the picture stays frozen while
# the phone keeps streaming (so the stall watchdog never fires), and the bot acted on
# a stale screen for 20-60 s: "touches ignored" restart loops, Next Level tapped 10x.
REOPEN_RETRY_S = 5.0  # phone gone: try to reopen the session this often
KEY_RETRY_S = 1.5  # frozen this long: ask for a keyframe again (and every KEY_RETRY_S)
KEY_RESTART_S = 6.0  # still frozen: restart the stream session
KEY_RESTART_COOLDOWN_S = 20.0
# Where the phone connection comes from. "userspace" = pymobiledevice3's in-process tunnel:
# no root, no separate tunneld window, and the phone's video (UDP) lands inside this
# process, so the macOS firewall can't drop it (it did: 0 RTP packets with tunneld on a
# Mac). "tunneld" = the root `pymobiledevice3 remote tunneld` service (Linux default).
TUNNEL_MODE = os.environ.get("WSBOT_TUNNEL", "auto").lower()  # auto | userspace | tunneld


def use_userspace_tunnel() -> bool:
    return TUNNEL_MODE == "userspace" or (TUNNEL_MODE == "auto" and sys.platform == "darwin")


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
    """"26.4.1" -> (26, 4, 1); () when unknown."""
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

    def __init__(self, udid: str | None = None, *, port: int = 8090):
        self.udid = udid or ""
        self.port = port
        self.width, self.height = 750, 1334  # screen px; read from the phone on open
        self.product_type = ""  # "iPhone18,3" = iPhone 17 (see ios_device.IPHONE_NAMES)
        self._loop = asyncio.new_event_loop()
        # ScreenStreamServer.serve() installs Ctrl-C handlers; only the main thread may.
        self._loop.add_signal_handler = lambda *a, **k: None
        self._loop.set_exception_handler(_quiet_resets)
        threading.Thread(target=self._loop.run_forever, name="iphone-loop", daemon=True).start()
        self._srv = None
        self._serve_task: asyncio.Task | None = None
        self._key_task: asyncio.Task | None = None
        self._rsd = None
        self._held: tuple[int, int] | None = None
        self._us = None  # pymobiledevice3 UserspaceRsdTunnel while one is open
        # frames
        self._cond = threading.Condition()
        self._frame: np.ndarray | None = None
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

    def _call(self, coro, timeout: float):
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return fut.result(timeout)
        except TimeoutError:
            fut.cancel()
            raise IPhoneError(f"phone call timed out after {timeout:.0f}s") from None

    def open(self, timeout: float = 60.0) -> None:
        self._call(self._open(), timeout)
        if self._reader is None or not self._reader.is_alive():
            self._stop.clear()
            self._reader = threading.Thread(
                target=self._read_frames, name="iphone-frames", daemon=True
            )
            self._reader.start()
        # the first frame tells us the screen size
        t0 = time.monotonic()
        next_note = t0 + 5
        while self._frame is None and time.monotonic() < t0 + 45:
            time.sleep(0.05)
            if time.monotonic() > next_note:
                next_note += 5
                log(
                    "DIAG",
                    f"waiting for the first frame {time.monotonic() - t0:.0f}s: {self.stats} "
                    f"server {self.srv_state()}",
                )
        if self._frame is None:
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

    def close(self) -> None:
        self._closing.set()
        self._stop.set()
        with contextlib.suppress(Exception):
            self._call(self._close(), 30)

    def reopen(self) -> None:
        """Rebuild the session. If the phone is gone (the tunnel dropped for a while),
        wait for it to come back instead of raising: a raise here killed a 16 h run.
        One thread reopens at a time; the others wait for it and use its session."""
        gen = self._opens
        with self._reopen_lock:
            if self._opens != gen and self.alive:
                return  # another thread just reopened it
            t0 = time.monotonic()
            next_note = 0.0
            while True:
                log("RECOVERY", "reopening the iPhone USB session")
                with contextlib.suppress(Exception):
                    self._call(self._close(), 30)
                try:
                    self.open()
                    self._opens += 1
                    return
                except Exception as exc:  # NoTunnel, no frames, OSError from the tunnel
                    if self._given_up() or isinstance(exc, TooOld):
                        raise
                    waited = time.monotonic() - t0
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
        from pymobiledevice3.tunneld.api import get_tunneld_devices

        # The phone sends frames only when the screen changes, so the server's 5 s
        # "no frames = stalled" watchdog restarted the stream whenever the board sat
        # still (the bot thinking, a probe) and dropped the touches sent meanwhile.
        screen_stream._STALL_RESTART_SECS = STALL_RESTART_S

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
            rsds = await get_tunneld_devices()
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
        srv = ScreenStreamServer(rsd, bind="127.0.0.1", http_port=self.port)
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
        self._held = None
        self.udid = rsd.udid
        log(
            "DEVICE",
            f"iPhone {rsd.udid}: USB stream + touch up, viewer http://127.0.0.1:{self.port}/",
        )

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
        if self._key_task is not None:
            self._key_task.cancel()
            self._key_task = None
        self._srv = self._serve_task = self._rsd = None
        if task is not None:
            task.cancel()  # serve() stops the device-side streams in its finally
            with contextlib.suppress(BaseException):
                await asyncio.wait_for(task, 20)
        if self._us is not None:  # the tunnel owns its RSD; a reopen builds a fresh one
            tunnel, self._us = self._us, None
            with contextlib.suppress(Exception):
                await asyncio.wait_for(tunnel.aclose(), 15)
        elif rsd is not None:
            with contextlib.suppress(Exception):
                await rsd.close()

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

    async def _keyframe_watchdog(self, srv) -> None:
        loop = asyncio.get_running_loop()
        episode: float | None = None
        last_ask = last_restart = 0.0
        while True:
            await asyncio.sleep(0.25)
            now = loop.time()
            since = self._frozen_since(srv)
            if since is None:
                if episode is not None and now - episode > KEY_RETRY_S:
                    log("LOG", f"iPhone stream: picture was frozen {now - episode:.1f}s, recovered")
                episode = None
                continue
            episode = since if episode is None else min(episode, since)
            age = now - episode
            if age > KEY_RESTART_S and now - last_restart > KEY_RESTART_COOLDOWN_S:
                last_restart = now
                log("RECOVERY", f"iPhone stream frozen {age:.0f}s (no keyframe); restarting it")
                try:
                    await asyncio.wait_for(srv._ensure_fresh_stream(force=True), timeout=10.0)
                except Exception as exc:
                    log("WARN", f"iPhone stream restart failed: {exc!r}")
                continue
            if age > KEY_RETRY_S and now - last_ask > KEY_RETRY_S:
                last_ask = now
                with contextlib.suppress(Exception):
                    srv._request_recovery_idr(reason="wsbot-frozen")

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
                if codec is None or kind == 2:  # 2 = keyframe after a restart: fresh decoder
                    dbg(f"new HEVC decoder (kind={kind})")
                    codec = av.CodecContext.create("hevc", "r")
                    # Not "AUTO": frame-threading workers hold our packets, and freeing
                    # the old decoder (a restart, shutdown) then deadlocks on the GIL.
                    codec.thread_type = "SLICE"
                try:
                    decoded = codec.decode(av.Packet(_annexb(au)))
                except av.error.FFmpegError as exc:
                    self.stats["decode_err"] += 1
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
                    self._publish(frame.to_ndarray(format="bgr24"))
        finally:
            conn.close()

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

    def _publish(self, img: np.ndarray) -> None:
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
        with self._cond:
            self._frame = img
            self._seq += 1
            self._frame_t = time.monotonic()
            self.frames_decoded += 1
            self._cond.notify_all()

    def latest(self) -> tuple[int, float, np.ndarray]:
        """(sequence number, time.monotonic() it arrived, BGR frame)."""
        with self._cond:
            if self._frame is None:
                raise IPhoneError("no frame from the iPhone yet")
            return self._seq, self._frame_t, self._frame

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
        self._held = (x, y)
        await self._send(CONTACT, x, y)

    async def _up(self) -> None:
        if self._held:
            x, y = self._held
            self._held = None
            await self._send(RELEASE, x, y)

    async def _tap(self, x: int, y: int, hold_ms: float) -> None:
        await self._down(x, y)
        await asyncio.sleep(hold_ms / 1000)
        await self._up()

    async def _swipe(self, x1: int, y1: int, x2: int, y2: int, ms: float, steps: int) -> None:
        steps = max(1, steps)
        await self._down(x1, y1)
        t0 = time.perf_counter()
        for i in range(1, steps + 1):
            delay = t0 + ms / 1000 * i / steps - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            await self._down(round(x1 + (x2 - x1) * i / steps), round(y1 + (y2 - y1) * i / steps))
        await self._up()

    async def _press(self, button: tuple[int, int], ms: float = 60) -> None:
        await self._srv._ensure_hid()
        await self._srv._indigo.send_button(*button, BTN_DOWN)
        await asyncio.sleep(ms / 1000)
        await self._srv._indigo.send_button(*button, BTN_UP)

    async def _app(self, op: str, bundle: str) -> int:
        from pymobiledevice3.services.dvt.instruments.dvt_provider import DvtProvider
        from pymobiledevice3.services.dvt.instruments.process_control import ProcessControl

        async with DvtProvider(self._rsd) as dvt, ProcessControl(dvt) as pc:
            if op == "launch":
                return await pc.launch(bundle, kill_existing=True)
            pid = await pc.process_identifier_for_bundle_identifier(bundle)
            if op == "kill" and pid:
                await pc.kill(pid)
            return pid

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
                await self._down(*u)

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
