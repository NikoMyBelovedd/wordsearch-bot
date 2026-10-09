"""--debug: one file handler for debug.log (two rotated each other's files)."""

from __future__ import annotations

import logging
import logging.handlers

from wsbot import debug


def test_one_handler_writes_debug_log(tmp_path, monkeypatch):
    monkeypatch.setattr(debug, "_root", None)
    monkeypatch.setattr("faulthandler.enable", lambda *a, **k: None)
    monkeypatch.setattr("sys.excepthook", None)
    monkeypatch.setattr("threading.excepthook", None)
    monkeypatch.setattr("wsbot.log.add_sticky_sink", lambda sink: None)
    libs = [logging.getLogger(n) for n in debug.LIB_LOGGERS]
    before = {lg.name: list(lg.handlers) for lg in libs}
    try:
        debug.setup(tmp_path)
        handlers = {
            h
            for lg in [debug._file_log, *libs]
            for h in lg.handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)
        }
        assert len(handlers) == 1  # ours and the libraries' lines share it
        debug.dbg("hello")
        logging.getLogger("pymobiledevice3.x").debug("from a library")
        (handler,) = handlers
        handler.flush()
        text = (tmp_path / "local" / "debug.log").read_text(encoding="utf-8")
        assert "[DBG]" in text and "hello" in text
        assert "[LIB DEBUG]" in text and "pymobiledevice3.x: from a library" in text
    finally:
        for lg in libs:
            lg.handlers[:] = before[lg.name]
        debug._file_log.handlers[:] = []
        for h in handlers:
            h.close()
