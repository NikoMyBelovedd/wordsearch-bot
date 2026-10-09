"""(Shared from ahq-device: change it there, never in a bot; its sync.py copies it in.)
An Android phone over adb: every adb call a game bot makes, in one tested place.

Every command is a fixed argv (never a shell string built from outside input), and every
call has a deadline: a stalled adb server or USB link raises DeviceError instead of
freezing the bot. A long-lived input shell (ShellInput) gets the same promise: past its
deadline it is killed and the next input opens a fresh one.

What the bot decides (which game, when to restart it, what a screen means) stays in the
bot. The bot that vendors this file provides `log.py` with `log(tag, msg)` next to it.
"""

from __future__ import annotations

import queue
import re
import shutil
import subprocess
import threading
import time
import zlib
from collections.abc import Callable

import cv2
import numpy as np

from .log import log

TIMEOUT_S = 20.0  # every adb call, unless the caller says otherwise

# Raw pixels gzipped on the phone: skips the phone's slow PNG encoder (about half the
# time per shot) at about a PNG's size over USB, which matters with many phones on one
# hub. A fixed command: adb runs it in the phone's shell, nothing from outside goes in.
RAW_GZ = ("exec-out", "screencap | gzip -1")
PNG = ("exec-out", "screencap", "-p")
RGBA_8888 = 1
# Fixed commands too. The focused window line only: all of `dumpsys window` is big.
FOCUS = ("shell", "dumpsys window | grep mCurrentFocus || true")
LAUNCHER = "android.intent.category.LAUNCHER"


class DeviceError(RuntimeError):
    """The phone didn't do what was asked: adb failed, timed out, or sent nothing usable."""


class InputTimeout(DeviceError, OSError):
    """The input shell didn't answer in time (a wedged adb shell or USB link)."""


def adb_path() -> str:
    path = shutil.which("adb")
    if not path:
        raise DeviceError("adb not found")
    return path


def decode_raw(data: bytes) -> np.ndarray:
    """`screencap` without -p: a header (width, height, format, and on newer Android a
    colour space) then RGBA rows. Returns BGR like cv2.imread."""
    if len(data) < 12:
        raise ValueError("raw screenshot too short")
    w, h, fmt = np.frombuffer(data[:12], "<u4")
    pixels = int(w) * int(h) * 4
    header = len(data) - pixels
    if fmt != RGBA_8888 or header not in (12, 16):
        raise ValueError(f"raw screenshot: format {fmt}, {len(data)} bytes for {w}x{h}")
    img = np.frombuffer(data, np.uint8, count=pixels, offset=header).reshape(int(h), int(w), 4)
    return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)


def parse_wm_size(out: str) -> tuple[int, int] | None:
    """(width, height) the screen really has, from `wm size` (an override wins)."""
    found = None
    for line in out.splitlines():
        m = re.search(r"(Physical|Override) size: (\d+)x(\d+)", line)
        if m and (found is None or m.group(1) == "Override"):
            found = (int(m.group(2)), int(m.group(3)))
    return found


def parse_awake(dumpsys_power: str) -> bool | None:
    """Is the screen on, from `dumpsys power`? None if it doesn't say."""
    m = re.search(r"mWakefulness=(\w+)", dumpsys_power)
    if m:
        return m.group(1) == "Awake"
    m = re.search(r"Display Power: state=(\w+)", dumpsys_power)
    if m:
        return m.group(1) == "ON"
    return None


def resumed_line(dumpsys_activities: str) -> str:
    """The resumed-activity line of `dumpsys activity activities` ('' if none):
    topResumedActivity on Android 10+, mResumedActivity before."""
    for line in dumpsys_activities.splitlines():
        if "topResumedActivity" in line or "mResumedActivity" in line:
            return line.strip()
    return ""


def package_of(line: str) -> str:
    """The package in a `package/activity` line ('' if there is none)."""
    m = re.search(r"([\w.]+)/", line)
    return m.group(1) if m else ""


class Android:
    """One phone by its adb serial. In a dry run nothing is sent that changes the phone
    (taps, keys, starting or stopping apps); reading it still works."""

    UI_DUMP_FILE = "/sdcard/ahq-ui.xml"  # where `uiautomator dump` writes on the phone

    def __init__(
        self,
        serial: str,
        *,
        dry_run: bool = False,
        timeout: float = TIMEOUT_S,
        shot_timeout: float | None = None,
    ) -> None:
        self.serial = serial
        self.dry_run = dry_run
        self.timeout = timeout
        self.shot_timeout = timeout if shot_timeout is None else shot_timeout
        self.adb = adb_path()
        self.raw = True  # off for good if RAW_GZ fails before it ever worked
        self.raw_works = False
        self.screen_size: tuple[int, int] | None = None  # the screen's real (width, height)
        self.size_retry = 0.0
        self.clock: Callable[[], float] = time.monotonic

    # ---- adb -----------------------------------------------------------------

    def run(self, *args: str, timeout: float | None = None) -> bytes:
        """`adb -s <serial> <args>`: its output, or DeviceError when adb fails, says
        nothing usable, or takes longer than the deadline."""
        try:
            out = subprocess.run(
                [self.adb, "-s", self.serial, *args],
                capture_output=True,
                timeout=self.timeout if timeout is None else timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            raise DeviceError(f"adb {args[0]} timed out") from e
        except OSError as e:  # adb itself gone (uninstalled, a path that moved)
            raise DeviceError(f"adb {args[0]}: {e}") from e
        if out.returncode != 0:
            msg = out.stderr.decode(errors="replace").strip() or f"exit {out.returncode}"
            raise DeviceError(f"adb {args[0]}: {msg}")
        return out.stdout

    def shell(self, *args: str, timeout: float | None = None) -> str:
        return self.run("shell", *args, timeout=timeout).decode(errors="replace")

    # ---- the screen ----------------------------------------------------------

    def screenshot(self) -> np.ndarray:
        """The screen as BGR. Bots with a faster source (a video stream) override this
        and fall back to `screencap`."""
        return self.screencap()

    def screencap(self) -> np.ndarray:
        """A screenshot straight from the phone: raw pixels gzipped on the phone, or a
        PNG when that doesn't work."""
        if self.raw:
            try:
                img = decode_raw(zlib.decompress(self.run(*RAW_GZ, timeout=self.shot_timeout), 31))
                self.raw_works = True
                return img
            except (DeviceError, zlib.error, ValueError):
                # Never worked on this phone (no gzip, odd pixel format): PNG from now on.
                # Worked before: a one-off hiccup, PNG just this time.
                if not self.raw_works:
                    self.raw = False
        png = self.run(*PNG, timeout=self.shot_timeout)
        # Empty when the phone drops off USB mid-shot (OpenCV asserts on an empty buffer).
        img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR) if png else None
        if img is None:
            raise DeviceError("screenshot could not be read")
        return img

    def real_size(self) -> tuple[int, int] | None:
        """The screen's (width, height), asked once (`wm size`); None until it answers
        (asked again a minute later)."""
        if self.screen_size is None and self.clock() >= self.size_retry:
            try:
                self.screen_size = parse_wm_size(self.shell("wm", "size", timeout=10))
            except DeviceError:
                pass
            if self.screen_size is None:
                self.size_retry = self.clock() + 60
        return self.screen_size

    # Back and starting an app don't wake the screen, so a screen that timed out (a long
    # pause, a lost connection, a reboot) leaves a bot recovering from "unknown screen"
    # for good unless something wakes it.
    def awake(self) -> bool | None:
        """Is the screen on? None if Android doesn't say."""
        return parse_awake(self.shell("dumpsys", "power"))

    def wake(self) -> bool:
        """Turn the screen on (and dismiss a keyguard without a PIN) if it is off; an
        awake screen (or one Android says nothing about) is left alone. True if woken."""
        if self.dry_run or self.awake() is not False:
            return False
        self.shell("input", "keyevent", "KEYCODE_WAKEUP")
        self.shell("wm", "dismiss-keyguard")
        return True

    def stay_awake(self) -> None:
        """Keep the screen on while the phone is on USB power (Android's own switch)."""
        if not self.dry_run:
            self.shell("svc", "power", "stayon", "usb")

    def screen(self, on: bool) -> None:
        """Turn the screen on or off, whatever it is now (a rest with the screen dark)."""
        if not self.dry_run:
            self.shell("input", "keyevent", "KEYCODE_WAKEUP" if on else "KEYCODE_SLEEP")

    # ---- input (one adb call each; see ShellInput for a persistent shell) ----------

    def tap(self, x: float, y: float) -> None:
        if not self.dry_run:
            self.shell("input", "tap", str(int(x)), str(int(y)))

    def swipe(self, x0: float, y0: float, x1: float, y1: float, ms: int = 300) -> None:
        if not self.dry_run:
            self.shell("input", "swipe", *(str(int(v)) for v in (x0, y0, x1, y1, ms)))

    def key(self, code: str) -> None:
        """A key event by name, e.g. KEYCODE_BACK."""
        if not self.dry_run:
            self.shell("input", "keyevent", code)

    def back(self) -> None:
        self.key("KEYCODE_BACK")

    # ---- what's on top -------------------------------------------------------

    def foreground(self) -> str:
        """The resumed-activity line ("... com.game/.Main ...") or '' if Android has none."""
        return resumed_line(self.shell("dumpsys", "activity", "activities"))

    def foreground_package(self) -> str:
        """The package on top ('' if unknown)."""
        return package_of(self.foreground())

    def focus(self) -> str:
        """The focused-window line of `dumpsys window` (system boxes like "isn't
        responding" show up here, not as an activity)."""
        return self.run(*FOCUS).decode(errors="replace")

    def ui_dump(self, timeout: float = 30) -> str:
        """The view tree on screen (`uiautomator dump`), as XML."""
        cmd = f"uiautomator dump {self.UI_DUMP_FILE} >/dev/null && cat {self.UI_DUMP_FILE}"
        return self.run("shell", cmd, timeout=timeout).decode(errors="replace")

    def resolve_activity(self, package: str) -> str:
        """`cmd package resolve-activity --brief <package>`: its last line is the
        "package/activity" its launcher icon opens (the app's main activity)."""
        return self.shell("cmd", "package", "resolve-activity", "--brief", package)

    # ---- apps. Never clear an app's data or reinstall it: progress lives in it. ----

    def start_app(self, package: str) -> None:
        """Open the app the way its launcher icon does (brings it back if it's running)."""
        if not self.dry_run:
            self.shell("monkey", "-p", package, "-c", LAUNCHER, "1")

    def stop_app(self, package: str) -> None:
        if not self.dry_run:
            self.shell("am", "force-stop", package)

    def restart_app(
        self, package: str, gap: float = 2.0, sleep: Callable[[float], None] = time.sleep
    ) -> None:
        """Stop the app, wait `gap` s, open it again. A stop that fails (adb hiccup) is
        logged and the start still tried: a game left closed is worse."""
        try:
            self.stop_app(package)
        except DeviceError as e:
            log("WARN", f"stopping {package} failed ({e}); starting it anyway")
        sleep(gap)
        self.start_app(package)

    def mute_vibration(self, package: str) -> bool:
        """Stop this app (only this app) from vibrating the phone: a farm of buzzing
        phones drives people mad. Android's per-app switch. True if it worked."""
        if self.dry_run:
            return False
        try:
            self.shell("cmd", "appops", "set", package, "VIBRATE", "ignore")
            return True
        except DeviceError:
            return False

    def battery(self) -> str:
        return self.shell("dumpsys", "battery")

    def wait_for_device(self, timeout: float = 60) -> None:
        self.run("wait-for-device", timeout=timeout)


class ShellInput:
    """A long-lived `adb shell` that runs input commands and waits for each to finish
    (~50 ms a swipe against ~300 ms for a fresh adb process).

    Replies are read on a helper thread, so a wedged shell (adb server or USB stalled
    without closing the pipe) can't freeze the caller: past REPLY_TIMEOUT_S the shell is
    killed and InputTimeout raised; the next input opens a fresh one. A shell that died
    is reopened once and the command sent again."""

    SENTINEL = b"__ahq_ok__"
    REPLY_TIMEOUT_S = 10.0

    def __init__(self, serial: str, adb: str = "adb") -> None:
        self.serial = serial
        self.adb = adb
        self.proc: subprocess.Popen[bytes] | None = None
        self.lines: queue.Queue[bytes | None] = queue.Queue()

    def _spawn(self) -> subprocess.Popen[bytes]:
        return subprocess.Popen(
            [self.adb, "-s", self.serial, "shell"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,  # bytes, not text: Windows text pipes would turn \n into \r\n
        )

    def _open(self) -> subprocess.Popen[bytes]:
        if self.proc is None or self.proc.poll() is not None:
            self.proc = self._spawn()
            self.lines = lines = queue.Queue()
            stdout = self.proc.stdout

            def read() -> None:
                try:
                    assert stdout
                    for line in iter(stdout.readline, b""):
                        lines.put(line)
                except (OSError, ValueError):
                    pass
                lines.put(None)  # closed

            threading.Thread(target=read, daemon=True, name="input-shell").start()
        return self.proc

    def run(self, cmd: str) -> None:
        """Run one fixed input command (e.g. "input tap 10 20") and wait for it."""
        for attempt in range(2):
            proc = self._open()
            lines = self.lines
            try:
                assert proc.stdin
                proc.stdin.write(f"{cmd}; echo {self.SENTINEL.decode()}\n".encode())
                proc.stdin.flush()
                deadline = time.monotonic() + self.REPLY_TIMEOUT_S
                while True:
                    try:
                        line = lines.get(timeout=max(0.0, deadline - time.monotonic()))
                    except queue.Empty:
                        log(
                            "RECOVERY",
                            f"input shell silent for {self.REPLY_TIMEOUT_S:.0f}s; killed",
                        )
                        self.kill()
                        raise InputTimeout(f"no reply to {cmd.split()[1]!r}") from None
                    if line is None:
                        raise BrokenPipeError("adb shell closed")
                    if line.strip() == self.SENTINEL:
                        return
            except InputTimeout:
                raise  # not sent again: it may have landed late
            except (BrokenPipeError, OSError) as exc:
                log("RECOVERY", f"input shell died ({exc!r}); reopening")
                self.proc = None
                if attempt:
                    raise

    def kill(self) -> None:
        proc, self.proc = self.proc, None
        if proc and proc.poll() is None:
            proc.kill()

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
