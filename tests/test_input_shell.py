"""The Android input shell: a wedged adb shell can't freeze the bot."""

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
