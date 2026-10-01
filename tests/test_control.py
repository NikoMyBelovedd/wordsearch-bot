import io
import threading

from wsbot import control


def test_off_unless_asked(monkeypatch):
    monkeypatch.delenv("AHQ_CONTROL", raising=False)
    assert control.listen(threading.Event(), io.StringIO("pause\n")) is None


def test_pause_and_resume_lines(monkeypatch):
    monkeypatch.setenv("AHQ_CONTROL", "stdin-v1")
    pause = threading.Event()
    seen = []

    class Lines(io.StringIO):
        def __iter__(self):
            for line in ["pause\n", "noise\n"]:
                yield line
                seen.append(pause.is_set())
            yield "RESUME\n"
            seen.append(pause.is_set())
            yield "pause\n"
            seen.append(pause.is_set())

    thread = control.listen(pause, Lines())
    thread.join(timeout=5)
    assert seen == [True, True, False, True]
    # End of input never leaves the bot paused.
    assert not pause.is_set()
