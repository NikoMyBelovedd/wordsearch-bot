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
decoded frame (750x1334 on an iPhone SE).
"""

from __future__ import annotations

import asyncio
import contextlib
import http.client
import threading
import time

import cv2
import numpy as np

from .log import log

USB_MAX = 65535
CONTACT, RELEASE = 0xC2, 0x02
TOUCHSCREEN = 257  # mainTouchscreen _ServiceID
HOME = (0x0C, 0x40)  # consumer page, Menu = the Home button
BTN_DOWN, BTN_UP = 1, 2
START_CODE = b"\x00\x00\x00\x01"
STALL_RESTART_S = 60.0  # no frames this long = a real encoder stall (pymobiledevice3: 5 s)


class IPhoneError(RuntimeError):
    pass


def _annexb(au: bytes) -> bytes:
    """Length-prefixed NAL units (hvcC style, as /stream.bin sends them) -> Annex-B."""
    out, i = bytearray(), 0
    while i + 4 <= len(au):
        n = int.from_bytes(au[i : i + 4], "big")
        out += START_CODE + au[i + 4 : i + 4 + n]
        i += 4 + n
    return bytes(out)


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


class IPhone:
    tap_ms = 30  # contact time of a tap
    # Swipes: contact at the start, `steps` evenly timed samples, lift. 1.8 ms swipes
    # registered 400/400 in a test page; the game gets 50 ms (see ios_device.py).

    def __init__(self, udid: str | None = None, *, port: int = 8090):
        self.udid = udid or ""
        self.port = port
        self.width, self.height = 750, 1334  # screen px; read from the phone on open
        self._loop = asyncio.new_event_loop()
        # ScreenStreamServer.serve() installs Ctrl-C handlers; only the main thread may.
        self._loop.add_signal_handler = lambda *a, **k: None
        threading.Thread(target=self._loop.run_forever, name="iphone-loop", daemon=True).start()
        self._srv = None
        self._serve_task: asyncio.Task | None = None
        self._rsd = None
        self._held: tuple[int, int] | None = None
        # frames
        self._cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._seq = 0
        self._frame_t = 0.0
        self.stream_connected = False
        self.frames_decoded = 0
        self._stop = threading.Event()
        self._reader: threading.Thread | None = None

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
        deadline = time.monotonic() + 15
        while self._frame is None and time.monotonic() < deadline:
            time.sleep(0.05)
        if self._frame is None:
            raise IPhoneError("the screen stream sends no frames (phone locked?)")

    def close(self) -> None:
        self._stop.set()
        with contextlib.suppress(Exception):
            self._call(self._close(), 30)

    def reopen(self) -> None:
        log("RECOVERY", "reopening the iPhone USB session")
        with contextlib.suppress(Exception):
            self._call(self._close(), 30)
        self.open()

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
        rsds = await get_tunneld_devices()
        rsd = next((r for r in rsds if not self.udid or r.udid == self.udid), None)
        for r in rsds:
            if r is not rsd:
                await r.close()
        if rsd is None:
            raise IPhoneError(
                f"no USB tunnel for {self.udid or 'an iPhone'}: is it plugged in and trusted, and "
                "is `pymobiledevice3 remote tunneld` running (as root / Administrator)?"
            )
        from pymobiledevice3.remote.core_device.device_info import DeviceInfoService

        with contextlib.suppress(Exception):
            async with DeviceInfoService(rsd) as info:
                mode = (await info.get_display_info())["displays"][0]["currentMode"]["size"]
                self.width, self.height = round(mode[0]), round(mode[1])
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
            await srv._ensure_hid()
        except BaseException:
            task.cancel()
            with contextlib.suppress(BaseException):
                await asyncio.wait_for(task, 15)
            await rsd.close()
            raise
        self._srv, self._serve_task, self._rsd = srv, task, rsd
        self._held = None
        self.udid = rsd.udid
        log(
            "DEVICE",
            f"iPhone {rsd.udid}: USB stream + touch up, viewer http://127.0.0.1:{self.port}/",
        )

    async def _close(self) -> None:
        task, rsd = self._serve_task, self._rsd
        self._srv = self._serve_task = self._rsd = None
        if task is not None:
            task.cancel()  # serve() stops the device-side streams in its finally
            with contextlib.suppress(BaseException):
                await asyncio.wait_for(task, 20)
        if rsd is not None:
            with contextlib.suppress(Exception):
                await rsd.close()

    # ---- frames ---------------------------------------------------------------------

    def _read_frames(self) -> None:
        while not self._stop.is_set():
            try:
                self._stream_once()
            except Exception as exc:
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
            if resp.status != 200:
                raise ConnectionError(f"/stream.bin -> HTTP {resp.status}")
            self.stream_connected = True
            codec = None
            while not self._stop.is_set():
                head = resp.read(4)
                if len(head) < 4:
                    raise ConnectionError("stream closed")
                body = resp.read(int.from_bytes(head, "big"))
                kind, au = body[0], body[1:]
                if codec is None or kind == 2:  # 2 = keyframe after a restart: fresh decoder
                    codec = av.CodecContext.create("hevc", "r")
                    # Not "AUTO": frame-threading workers hold our packets, and freeing
                    # the old decoder (a restart, shutdown) then deadlocks on the GIL.
                    codec.thread_type = "SLICE"
                try:
                    decoded = codec.decode(av.Packet(_annexb(au)))
                except av.error.FFmpegError:
                    continue
                for frame in decoded:
                    self._publish(frame.to_ndarray(format="bgr24"))
        finally:
            conn.close()

    def _publish(self, img: np.ndarray) -> None:
        h, w = img.shape[:2]
        if 0 <= w - self.width < 16 and 0 <= h - self.height < 16:
            # HEVC codes whole 16 px blocks: 750x1334 arrives as 752x1344 with black
            # padding right and bottom that the stream doesn't mark for cropping.
            img = img[: self.height, : self.width]
        img = _uncollapse(img)
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
        return self._do(lambda: self._app(op, bundle), timeout=30)
