"""Debug capture (`wsbot --debug`): everything a remote tester's run can tell us, on disk.

- local/debug.log: every wsbot log line plus chatty `dbg()` detail and pymobiledevice3's
  own DEBUG logging (tunnel, RSD, stream, HID), rotating 3 x 15 MB.
- local/sysinfo.txt: OS, Python, package + FFmpeg versions, HEVC decoder availability.
- local/threads.log: every thread's stack, rewritten each minute (for hangs).
- local/crash.log: faulthandler output (native crashes).
- diagnostics/debug/: rate-limited JPEG snapshots (raw phone frames, calib frames,
  every new board), capped so the report zip stays Discord-sized.
- uncaught exceptions in any thread: full traceback into debug.log and the log.

`wsbot --report` zips all of it (and works without --debug: then it's the normal logs).
"""

from __future__ import annotations

import contextlib
import faulthandler
import logging
import logging.handlers
import platform
import sys
import threading
import time
import traceback
import zipfile
import zlib
from pathlib import Path

import cv2
import numpy as np

from .log import log

_root: Path | None = None
_file_log = logging.getLogger("wsbot.debug")
_last_snap: dict[str, float] = {}
_snap_count: dict[str, int] = {}
_lock = threading.Lock()
SNAP_CAP = 60  # per snapshot kind; oldest files are overwritten round-robin
REPORT_BUDGET = 14_000_000  # bytes: under Discord's 15 MB upload limit with room to spare
LOG_SHARE = 0.45  # at most this much of the budget per big log (the tail is kept)


LIB_LOGGERS = ("pymobiledevice3", "asyncio", "av", "urllib3")


class _Format(logging.Formatter):
    """Our lines as they are (they carry their own time and tag); libraries' lines with
    time, level, thread and logger name."""

    LIB = logging.Formatter(
        "%(asctime)s [LIB %(levelname)s] [%(threadName)s] %(name)s: %(message)s"
    )

    def format(self, record: logging.LogRecord) -> str:
        if record.name == _file_log.name:
            return record.getMessage()
        return self.LIB.format(record)


def dbg(msg: str) -> None:
    """Detail for debug.log only (not the UI log)."""
    _file_log.debug(
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} [DBG] [{threading.current_thread().name}] {msg}"
    )


def _mirror(ts: str, tag: str, msg: str) -> None:
    _file_log.debug(f"{time.strftime('%Y-%m-%d')} {ts} [{tag}] {msg}")


def setup(root: Path) -> None:
    """Idempotent. Call once at startup (TUI, headless, diagnose)."""
    global _root
    if _root is not None:
        return
    _root = root
    local = root / "local"
    local.mkdir(exist_ok=True)
    (root / "diagnostics" / "debug").mkdir(parents=True, exist_ok=True)

    # One handler for the file: two on the same debug.log rotated each other's files
    # (on Windows the rename failed, so the log grew without limit). pymobiledevice3 (and
    # asyncio) log at DEBUG into it too, with their own format.
    handler = logging.handlers.RotatingFileHandler(
        local / "debug.log", maxBytes=15_000_000, backupCount=2, encoding="utf-8"
    )
    handler.setFormatter(_Format())
    handler.setLevel(logging.DEBUG)
    _file_log.handlers[:] = [handler]
    _file_log.setLevel(logging.DEBUG)
    _file_log.propagate = False
    for name in LIB_LOGGERS:
        lg = logging.getLogger(name)
        lg.setLevel(logging.DEBUG)
        lg.addHandler(handler)

    from .log import add_sticky_sink

    add_sticky_sink(_mirror)

    crash = open(local / "crash.log", "a", encoding="utf-8")
    crash.write(f"\n==== start {time.strftime('%Y-%m-%d %H:%M:%S')} ====\n")
    crash.flush()
    faulthandler.enable(crash, all_threads=True)

    def excepthook(exc_type, exc, tb):
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        dbg(f"UNCAUGHT (main): {text}")
        log("ERROR", f"uncaught {exc_type.__name__}: {exc} (traceback in local/debug.log)")
        sys.__excepthook__(exc_type, exc, tb)

    def thread_hook(args):
        text = "".join(
            traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)
        )
        name = args.thread.name if args.thread else "?"
        dbg(f"UNCAUGHT in thread {name}: {text}")
        log("ERROR", f"thread {name} died: {args.exc_type.__name__}: {args.exc_value}")

    sys.excepthook = excepthook
    threading.excepthook = thread_hook

    def stacks() -> None:
        while True:
            time.sleep(60)
            with contextlib.suppress(Exception):
                frames = sys._current_frames()
                names = {t.ident: t.name for t in threading.enumerate()}
                out = [f"==== {time.strftime('%Y-%m-%d %H:%M:%S')} ({len(frames)} threads) ===="]
                for ident, frame in frames.items():
                    out.append(f"\n--- {names.get(ident, ident)} ---")
                    out.extend(traceback.format_stack(frame))
                (local / "threads.log").write_text("".join(out), encoding="utf-8")

    threading.Thread(target=stacks, name="debug-stacks", daemon=True).start()
    write_sysinfo()
    dbg(f"debug capture on; argv={sys.argv}")


def write_sysinfo(root: Path | None = None) -> None:
    root = root or _root
    if root is None:
        return
    lines = [
        f"time: {time.strftime('%Y-%m-%d %H:%M:%S %z')}",
        f"platform: {platform.platform()} | machine {platform.machine()} | {platform.processor()}",
        f"mac_ver: {platform.mac_ver()} | win32_ver: {platform.win32_ver()}",
        f"python: {sys.version} ({sys.executable})",
        f"cwd root: {root}",
    ]
    from importlib import metadata

    for pkg in (
        "wsbot",
        "pymobiledevice3",
        "av",
        "opencv-python-headless",
        "numpy",
        "textual",
        "uiautomator2",
        "pytesseract",
        "cryptography",
        "construct",
    ):
        try:
            lines.append(f"pkg {pkg}: {metadata.version(pkg)}")
        except Exception as exc:
            lines.append(f"pkg {pkg}: ? ({exc!r})")
    try:
        import av

        lines.append(f"av library versions: {getattr(av, 'library_versions', '?')}")
        lines.append(f"ffmpeg version: {getattr(av, 'ffmpeg_version_info', '?')}")
        for name in ("hevc", "hevc_videotoolbox", "h264"):
            try:
                c = av.codec.Codec(name, "r")
                lines.append(f"decoder {name}: ok ({c.long_name})")
            except Exception as exc:
                lines.append(f"decoder {name}: MISSING ({exc!r})")
    except Exception as exc:
        lines.append(f"av: import failed {exc!r}")
    lines.append(f"cv2: {cv2.__version__} threads={cv2.getNumThreads()}")
    lines.append(f"numpy: {np.__version__}")
    with contextlib.suppress(Exception):
        import shutil

        for tool in ("tesseract", "adb", "idevice_id", "uv"):
            lines.append(f"which {tool}: {shutil.which(tool)}")
    with contextlib.suppress(Exception):
        import json
        import urllib.request

        with urllib.request.urlopen("http://127.0.0.1:49151/", timeout=3) as r:
            raw = r.read()
        lines.append(f"tunneld: {json.dumps(json.loads(raw or b'{}'), default=str)[:3000]}")
    (root / "local").mkdir(exist_ok=True)
    (root / "local" / "sysinfo.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        dbg(f"sysinfo {line}")


def snap_due(kind: str, every_s: float) -> bool:
    """Whether snap(kind, ..., every_s) would save now: check before making the image."""
    if _root is None:
        return False
    return time.monotonic() - _last_snap.get(kind, -1e9) >= every_s


def snap(kind: str, img: np.ndarray | None, every_s: float = 20.0, note: str = "") -> None:
    """Save a JPEG of `img` as diagnostics/debug/<kind>_<n>.jpg at most every `every_s`."""
    if _root is None or img is None:
        return
    now = time.monotonic()
    with _lock:
        if now - _last_snap.get(kind, -1e9) < every_s:
            return
        _last_snap[kind] = now
        n = _snap_count.get(kind, 0)
        _snap_count[kind] = n + 1
    path = _root / "diagnostics" / "debug" / f"{kind}_{n % SNAP_CAP:03d}.jpg"
    try:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            path.write_bytes(buf.tobytes())
            dbg(f"snap {path.name} {img.shape[1]}x{img.shape[0]} #{n} {note}")
    except Exception as exc:
        dbg(f"snap {kind} failed: {exc!r}")


def _deflated(data: bytes) -> int:
    return len(zlib.compress(data, 6)) + 120  # + zip headers


def _tail(data: bytes, limit: int) -> bytes:
    """The newest part of a log whose compressed size fits `limit`."""
    size = _deflated(data)
    while size > limit and len(data) > 1000:
        keep = int(len(data) * limit / size * 0.95)
        data = data[-keep:]
        data = b"[... older lines cut to fit the report ...]\n" + data[data.find(b"\n") + 1 :]
        size = _deflated(data)
    return data


def report_files(root: Path) -> list[tuple[Path, bool]]:
    """(file, is_text) in the order they matter: text logs first, then the newest images."""
    local, diag = root / "local", root / "diagnostics"
    text = [
        *sorted(local.glob("*.txt")),
        *sorted(local.glob("*.json")),
        *sorted(local.glob("*.log")),
        *sorted(local.glob("*.log.*")),
        *sorted((diag / "devtest").glob("*.txt")),
        *sorted((diag / "devtest").glob("*.json")),
    ]
    head = diag / "debug" / "stream_head.bin"
    images = sorted(
        [p for d in (diag, diag / "debug", diag / "devtest") for p in d.glob("*.[jp][pn]g")],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    devtest = [p for p in images if p.parent.name == "devtest"]
    rest = [p for p in images if p.parent.name != "devtest"]
    out = [(p, True) for p in text]
    out += [(p, False) for p in devtest]
    out += [(head, False)] if head.exists() else []
    return out + [(p, False) for p in rest]


def make_report(root: Path, budget: int = REPORT_BUDGET, out: Path | None = None) -> Path:
    """wsbot-report.zip: logs, system info, the raw stream head and the newest screenshots,
    packed by their real compressed size so it stays under `budget` bytes."""
    with contextlib.suppress(Exception):
        write_sysinfo(root)
    out = out or root / "wsbot-report.zip"
    used, added, skipped, cut = 0, 0, 0, []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path, is_text in report_files(root):
            try:
                data = path.read_bytes()
            except OSError:
                continue
            if is_text:
                room = min(budget - used, int(budget * LOG_SHARE))
                if _deflated(data) > room:
                    data = _tail(data, room)
                    cut.append(path.name)
                size = _deflated(data)
            else:
                size = len(data) + 120 + len(str(path))
            if used + size > budget:
                skipped += 1
                continue
            z.writestr(str(path.relative_to(root)), data)
            used += size
            added += 1
    note = f"; cut to the newest part: {', '.join(cut)}" if cut else ""
    note += f"; left out {skipped} older files to stay small" if skipped else ""
    print(f"saved {out} ({added} files, {out.stat().st_size / 1e6:.1f} MB{note}): send this file")
    return out
