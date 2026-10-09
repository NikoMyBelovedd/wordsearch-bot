"""The Android device layer: a wedged adb shell can't freeze the bot; screen scaling."""

from __future__ import annotations

import subprocess
import sys
import time

import pytest

from wsbot import device

# Stand-ins for `adb shell`: one answers every command, one never says anything.
ECHO = (
    "import sys\n"
    "for line in sys.stdin:\n"
    "    sys.stdout.write(line.rsplit('echo ', 1)[1]); sys.stdout.flush()\n"
)
SILENT = "import sys, time\nsys.stdin.readline()\ntime.sleep(60)\n"


def shell(code: str) -> device.ShellInput:
    s = device.ShellInput("fake")
    s._spawn = lambda: subprocess.Popen(  # type: ignore[method-assign]
        [sys.executable, "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0
    )
    return s


def test_commands_wait_for_their_reply():
    s = shell(ECHO)
    s.run("input tap 1 2")
    s.run("input swipe 1 2 3 4 60")
    s.close()


def test_a_shell_that_never_answers_is_killed_and_raises(monkeypatch):
    monkeypatch.setattr(device.ShellInput, "REPLY_TIMEOUT_S", 0.3)
    s = shell(SILENT)
    s._open()
    proc = s.proc
    start = time.monotonic()
    with pytest.raises(device.InputTimeout):
        s.run("input tap 1 2")
    assert time.monotonic() - start < 3
    assert s.proc is None and proc.wait(5) is not None  # killed
    s._spawn = shell(ECHO)._spawn  # the next input opens a fresh shell
    s.run("input tap 1 2")
    s.close()


def test_android_screens_scale_by_width_and_keep_their_shape():
    # The game fits the width: templates and the top bar's zones only line up when the
    # frame isn't stretched to 1080x2400.
    assert device.calib_size(1080, 2400) == (1080, 2400)  # the calibration phone
    assert device.calib_size(720, 1280) == (1080, 1920)  # 16:9
    assert device.calib_size(1440, 3120) == (1080, 2340)  # 19.5:9


def test_taps_scale_the_same_both_ways(monkeypatch):
    d = device.Device.__new__(device.Device)
    d.width, d.height = 720, 1280
    d.calib = device.calib_size(720, 1280)
    d.sx = d.sy = 720 / device.CALIB_W
    assert d._scale(540, 960) == (360, 640)  # the middle of the screen
    assert d._scale(216, 205) == (144, 137)  # the star: as far down as it is across
