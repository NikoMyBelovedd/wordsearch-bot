"""The terminal UI: pick a device, pick a goal, watch the bot play.

Screens: DeviceScreen -> ModeScreen -> RunScreen. The bot runs in a background
thread; its log lines land in a thread-safe queue that the UI drains on a timer, so
bot threads never touch widgets directly.
"""

from __future__ import annotations

import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Center, Horizontal
from textual.screen import Screen
from textual.theme import Theme
from textual.widgets import Footer, RichLog, Static

from .banner import ORANGE, WHITE, banner, compact_banner
from .goal import (
    CUSTOM_DEFAULT,
    CUSTOM_MAX,
    CUSTOM_MIN,
    CUSTOM_STEP,
    PLAY_DAYS,
    PLAY_PER_DAY,
    PLAY_TOTAL,
    Goal,
    Progress,
)
from .log import file_sink, set_sinks, stdout_sink

DIM = "#8a8a93"
POINTER = " \u203a "  # the row cursor
TIMES = "\u00d7"
MINUS = "\u2212"
CURSOR_BG = "#3b2210"
TAG_STYLES = {
    "LEVEL": f"bold {ORANGE}",
    "GOAL": f"bold {ORANGE}",
    "PASS": "#fdba74",
    "WATCHER": "#67e8f9",
    "TAP": DIM,
    "SWIPE": "#5c5c66",
    "WARN": "bold #facc15",
    "ERROR": "bold #f87171",
    "SAFETY": "bold #f87171",
    "RECOVERY": "bold #f87171",
    "OCR": "#a5b4fc",
}

THEME = Theme(
    name="wsbot",
    primary=ORANGE,
    secondary=WHITE,
    accent=ORANGE,
    foreground="#f5f5f5",
    background="#0d0d10",
    surface="#141418",
    panel="#1b1b21",
    dark=True,
)


# ---- devices ---------------------------------------------------------------------


@dataclass
class AdbDevice:
    serial: str
    state: str  # "device", "unauthorized", "offline", ...
    model: str

    @property
    def usable(self) -> bool:
        return self.state == "device"


def list_adb_devices() -> list[AdbDevice]:
    """Parse `adb devices -l`. Raises RuntimeError with a friendly message on failure."""
    try:
        out = subprocess.run(
            ["adb", "devices", "-l"], capture_output=True, text=True, timeout=8, check=False
        ).stdout
    except FileNotFoundError as exc:
        raise RuntimeError("adb not found: install Android platform-tools") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("adb did not answer within 8s") from exc
    devices = []
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 2 or line.startswith("*"):
            continue
        props = dict(p.split(":", 1) for p in parts[2:] if ":" in p)
        model = props.get("model", props.get("product", "")).replace("_", " ")
        devices.append(AdbDevice(parts[0], parts[1], model))
    return devices


# ---- small render helpers ----------------------------------------------------------


def progress_bar(done: int, total: int, width: int = 24) -> Text:
    frac = 0.0 if total <= 0 else max(0.0, min(1.0, done / total))
    filled = round(frac * width)
    bar = Text()
    bar.append("━" * filled, style=f"bold {ORANGE}")
    bar.append("━" * (width - filled), style="#2e2e36")
    bar.append(f" {frac * 100:5.1f}%", style=DIM)
    return bar


def fmt_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h >= 24:
        return f"{h // 24}d {h % 24}h"
    if h:
        return f"{h}h {m:02d}m"
    return f"{m}m {s:02d}s" if m else f"{s}s"


def log_line(ts: str, tag: str, msg: str) -> Text:
    line = Text()
    line.append(f"{ts} ", style="#5c5c66")
    line.append(f"{tag:<8} ", style=TAG_STYLES.get(tag, DIM))
    line.append(msg, style="#e5e5e5" if tag in ("LEVEL", "GOAL") else "#b4b4bd")
    return line


# ---- screen 1: device picker -------------------------------------------------------


class DeviceScreen(Screen):
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up", "move(-1)", "Up", show=False),
        Binding("down", "move(1)", "Down", show=False),
        Binding("enter", "choose", "Select"),
        Binding("r", "refresh", "Refresh"),
        Binding("q", "app.quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.devices: list[AdbDevice] = []
        self.error = ""
        self.cursor = 0

    def compose(self) -> ComposeResult:
        with Center():
            yield Static(id="banner")
        yield Static("Your Word Search Explorer autopilot", id="tagline")
        with Center():
            yield Static(id="devices", classes="panel")
        yield Static(id="hint")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#devices").border_title = "SELECT A DEVICE"
        self.action_refresh()

    def on_resize(self) -> None:
        self.query_one("#banner", Static).update(banner(self.size.width, self.size.height - 16))
        if self.is_mounted:
            self.render_devices()

    def action_refresh(self) -> None:
        try:
            self.devices, self.error = self.app.device_lister(), ""
        except RuntimeError as exc:
            self.devices, self.error = [], str(exc)
        wanted = self.app.serial
        usable = [i for i, d in enumerate(self.devices) if d.usable]
        preferred = [i for i, d in enumerate(self.devices) if d.serial == wanted and d.usable]
        self.cursor = (preferred or usable or [0])[0]
        self.render_devices()

    def action_move(self, step: int) -> None:
        if self.devices:
            self.cursor = (self.cursor + step) % len(self.devices)
            self.render_devices()

    def action_choose(self) -> None:
        if not self.devices:
            return
        device = self.devices[self.cursor]
        if not device.usable:
            self.notify(f"{device.serial} is {device.state}", severity="warning")
            return
        self.app.serial = device.serial
        self.app.push_screen(ModeScreen())

    def render_devices(self) -> None:
        body = Text()
        if self.error:
            body.append(f"  {self.error}\n", style="bold #f87171")
        elif not self.devices:
            body.append("  No devices found.\n\n", style="bold #facc15")
            body.append(
                "  Start an emulator or plug in a phone with USB debugging on,\n", style=DIM
            )
            body.append("  then press ", style=DIM)
            body.append("r", style=f"bold {ORANGE}")
            body.append(" to refresh.", style=DIM)
        inner = self.query_one("#devices").content_size.width or 74
        model_w = max(0, inner - 3 - 22 - 16)
        for i, d in enumerate(self.devices):
            here = i == self.cursor
            bg = f" on {CURSOR_BG}" if here else ""
            body.append(POINTER if here else "   ", style=f"bold {ORANGE}{bg}")
            body.append(f"{d.serial[:21]:<22}", style=f"bold {WHITE if d.usable else DIM}{bg}")
            model = d.model[: max(0, model_w - 1)].ljust(model_w)
            body.append(model, style=f"{'#d4d4d8' if d.usable else DIM}{bg}")
            state = {
                "device": ("● ready", "#4ade80"),
                "unauthorized": ("● unauthorized", "#facc15"),
                "offline": ("● offline", "#f87171"),
            }.get(d.state, (f"● {d.state}", "#f87171"))
            body.append(f"{state[0]:<16}", style=f"{state[1]}{bg}")
            body.append("\n")
        body.rstrip()
        self.query_one("#devices", Static).update(body)
        hint = Text(justify="center")
        if any(d.state == "unauthorized" for d in self.devices):
            hint.append("accept the USB debugging prompt on the phone, then press r\n\n", "#facc15")
        hint.append("↑/↓ choose   ", style=DIM)
        hint.append("enter", style=f"bold {ORANGE}")
        hint.append(" select   ", style=DIM)
        hint.append("r", style=f"bold {ORANGE}")
        hint.append(" refresh", style=DIM)
        self.query_one("#hint", Static).update(hint)


# ---- screen 2: goal picker -----------------------------------------------------------

MODES = ("play", "custom", "single")


class ModeScreen(Screen):
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up", "move(-1)", "Up", show=False),
        Binding("down", "move(1)", "Down", show=False),
        Binding("left", "adjust(-1)", MINUS, show=False),
        Binding("right", "adjust(1)", "+", show=False),
        Binding("shift+left", "adjust(-10)", f"{MINUS}10", show=False),
        Binding("shift+right", "adjust(10)", "+10", show=False),
        Binding("enter", "choose", "Select / Play"),
        Binding("escape", "back", "Devices"),
        Binding("q", "app.quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.cursor = 0  # rows 0..2 are modes, 3 is the PLAY button
        self.mode = "play"
        self.per_day = CUSTOM_DEFAULT

    def compose(self) -> ComposeResult:
        with Center():
            yield Static(id="banner")
        yield Static(id="progress-line")
        with Center():
            yield Static(id="modes", classes="panel")
        with Center():
            yield Static(id="play")
        yield Static(id="hint")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#modes").border_title = "CHOOSE A GOAL"
        self.render_all()

    def on_screen_resume(self) -> None:
        self.render_all()  # progress changed while the run screen was up

    def on_resize(self) -> None:
        self.query_one("#banner", Static).update(banner(self.size.width, self.size.height - 22))

    def action_move(self, step: int) -> None:
        self.cursor = max(0, min(3, self.cursor + step))
        self.render_all()

    def action_adjust(self, steps: int) -> None:
        if self.cursor != 1:
            return
        self.mode = "custom"
        self.per_day = max(CUSTOM_MIN, min(CUSTOM_MAX, self.per_day + steps * CUSTOM_STEP))
        self.render_all()

    def action_choose(self) -> None:
        if self.cursor < 3:
            self.mode = MODES[self.cursor]
            self.cursor = 3
            self.render_all()
            return
        progress = self.app.progress()
        goal = {
            "play": lambda: Goal.play(progress),
            "custom": lambda: Goal.custom(progress, self.per_day),
            "single": lambda: Goal.single(progress),
        }[self.mode]()
        self.app.push_screen(RunScreen(goal))

    def action_back(self) -> None:
        self.app.pop_screen()

    def render_all(self) -> None:
        progress = self.app.progress()
        line = Text(justify="center")
        line.append("device ", style=DIM)
        line.append(self.app.serial, style=f"bold {WHITE}")
        line.append("   ·   today ", style=DIM)
        line.append(f"{progress.day():,}", style=f"bold {ORANGE}")
        line.append(" levels   ·   14k plan ", style=DIM)
        line.append(f"{progress.plan('play14k'):,} / {PLAY_TOTAL:,}", style=f"bold {ORANGE}")
        self.query_one("#progress-line", Static).update(line)

        rows = [
            ("PLAY", self._play_detail()),
            ("CUSTOM TARGET", self._custom_detail()),
            ("SINGLE LEVEL", Text("play one level, then stop", style=DIM)),
        ]
        body = Text()
        for i, (name, detail) in enumerate(rows):
            here = i == self.cursor
            bg = f" on {CURSOR_BG}" if here else ""
            chosen = MODES[i] == self.mode
            body.append(POINTER if here else "   ", style=f"bold {ORANGE}{bg}")
            body.append("● " if chosen else "○ ", style=f"bold {ORANGE if chosen else DIM}{bg}")
            body.append(f"{name:<16}", style=f"bold {WHITE}{bg}")
            detail.stylize(bg.strip())
            body.append_text(detail)
            body.append(" " * max(0, 54 - len(detail.plain)), style=bg.strip())
            if i < len(rows) - 1:
                body.append("\n\n")
        self.query_one("#modes", Static).update(body)

        on_play = self.cursor == 3
        button = Text(justify="center")
        label = f"   ▶  PLAY  ·  {self._goal_summary()}   "
        style = f"bold #0d0d10 on {ORANGE}" if on_play else f"bold {ORANGE} on #1b1b21"
        button.append(label, style=style)
        self.query_one("#play", Static).update(button)

        hint = Text(f"↑/↓ move   ←/→ adjust target (shift {TIMES}10)   ", style=DIM)
        hint.append("enter", style=f"bold {ORANGE}")
        hint.append(" select · play   ", style=DIM)
        hint.append("esc", style=f"bold {ORANGE}")
        hint.append(" devices", style=DIM)
        self.query_one("#hint", Static).update(hint)

    def _play_detail(self) -> Text:
        t = Text()
        t.append(f"{PLAY_TOTAL:,}", style=f"bold {ORANGE}")
        t.append(f" levels in {PLAY_DAYS} days · ", style="#d4d4d8")
        t.append(f"{PLAY_PER_DAY:,}", style=f"bold {ORANGE}")
        t.append(" / day", style="#d4d4d8")
        return t

    def _custom_detail(self) -> Text:
        active = self.cursor == 1
        arrow = f"bold {ORANGE}" if active else DIM
        t = Text()
        t.append("◀ ", style=arrow if self.per_day > CUSTOM_MIN else "#2e2e36")
        t.append(f"{self.per_day:>5,}", style=f"bold {WHITE}")
        t.append(" ▶", style=arrow if self.per_day < CUSTOM_MAX else "#2e2e36")
        t.append(" levels / day", style="#d4d4d8")
        t.append(f"   max {CUSTOM_MAX:,}", style=DIM)
        return t

    def _goal_summary(self) -> str:
        if self.mode == "play":
            return f"{PLAY_TOTAL:,} levels · {PLAY_DAYS} days"
        if self.mode == "custom":
            return f"{self.per_day:,} levels / day"
        return "one level"


# ---- screen 3: live dashboard ------------------------------------------------------


class RunScreen(Screen):
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("p", "pause", "Pause / resume"),
        Binding("q", "stop", "Stop"),
        Binding("escape", "stop", "Stop", show=False),
    ]

    def __init__(self, goal: Goal) -> None:
        super().__init__()
        self.goal = goal
        self.bot: Any = None
        self.thread: threading.Thread | None = None
        self.error = ""
        self.stopping = False
        self.started = time.monotonic()

    def compose(self) -> ComposeResult:
        with Center():
            yield Static(id="mini-banner")
        yield Static(id="status")
        with Horizontal(id="middle"):
            yield Static(id="board", classes="panel")
            yield Static(id="stats", classes="panel")
        yield RichLog(id="log", classes="panel", max_lines=500, wrap=False, markup=False)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#board").border_title = "BOARD"
        self.query_one("#stats").border_title = self.goal.title.upper()
        self.query_one("#log").border_title = "LOG"
        self.thread = threading.Thread(target=self._run_bot, daemon=True, name="bot")
        self.thread.start()
        self.set_interval(0.5, self.refresh_view)
        self.refresh_view()

    def on_resize(self) -> None:
        self.query_one("#mini-banner", Static).update(compact_banner(self.size.width))

    def _run_bot(self) -> None:
        try:
            self.bot = self.app.bot_factory(self.app.serial, self.goal)
            self.bot.run()
        except Exception as exc:  # show it; the UI must outlive any bot failure
            self.error = f"{type(exc).__name__}: {exc}"
            self.app.sink(time.strftime("%H:%M:%S"), "ERROR", f"bot crashed: {self.error}")

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def action_pause(self) -> None:
        if self.bot is None:
            return
        if self.bot.pause_event.is_set():
            self.bot.pause_event.clear()
            self.notify("Resumed")
        else:
            self.bot.pause_event.set()
            self.notify("Paused: finishing the current swipe")
        self.refresh_view()

    def action_stop(self) -> None:
        if not self.running:
            self.app.pop_screen()
            return
        self.stopping = True
        if self.bot is not None:
            self.bot.pause_event.clear()
            self.bot.stop_event.set()
        self.refresh_view()

    def refresh_view(self) -> None:
        log = self.query_one("#log", RichLog)
        for entry in self.app.drain_log():
            log.write(log_line(*entry))
        if self.stopping and not self.running:
            self.app.pop_screen()
            return
        self.query_one("#status", Static).update(self._status())
        self.query_one("#board", Static).update(self._board())
        self.query_one("#stats", Static).update(self._stats())

    def _status(self) -> Text:
        bot = self.bot
        t = Text(justify="center")
        if self.error:
            t.append("  ✖ ", style="bold #f87171")
            t.append(self.error, style="#f87171")
            t.append("   press q to go back", style=DIM)
            return t
        if bot is None:
            t.append("  ◌ connecting to ", style=DIM)
            t.append(self.app.serial, style=f"bold {WHITE}")
            return t
        if self.stopping:
            dot, text, style = "◌", "stopping…", "#facc15"
        elif not self.running:
            dot, text, style = "✔", "finished · press q to go back", "#4ade80"
        elif bot.pause_event.is_set():
            dot, text, style = "❚❚", "paused · press p to resume", "#facc15"
        else:
            dot, text, style = "●", bot.status, "#4ade80"
        t.append(f"  {dot} ", style=f"bold {style}")
        t.append(text, style=f"bold {WHITE}")
        if bot.phase and self.running and not self.stopping and not bot.pause_event.is_set():
            t.append(f"  ·  {bot.phase}", style=ORANGE)
        return t

    def _board(self) -> Text:
        bot = self.bot
        grid = list(getattr(bot, "grid", []) or [])
        if not grid:
            return Text("\n\nwaiting for a board…", style=DIM, justify="center")
        found = set(getattr(bot, "found_cells", ()))
        fired = set(getattr(bot, "fired_cells", ()))
        t = Text(justify="center")
        for r, row in enumerate(grid):
            for c, ch in enumerate(row):
                if (r, c) in found:
                    style = f"bold #0d0d10 on {ORANGE}"
                elif (r, c) in fired:
                    style = "bold #fdba74"
                else:
                    style = f"bold {WHITE}"
                t.append(f" {ch} ", style=style)
                t.append(" ")
            t.append("\n")
        t.append(f"\n{len(grid)} {TIMES} {len(grid[0])}", style=DIM)
        return t

    def _stats(self) -> Table:
        goal, bot = self.goal, self.bot
        table = Table.grid(padding=(0, 2))
        table.add_column(style=f"bold {ORANGE}", no_wrap=True)
        table.add_column(no_wrap=True)

        today = Text(f"{goal.done_today:,}", style=f"bold {WHITE}")
        if goal.per_day:
            today.append(f" / {goal.per_day:,}  ", style=DIM)
            today.append_text(progress_bar(goal.done_today, goal.per_day))
        table.add_row("TODAY", today)
        if goal.total:
            plan = Text(f"{goal.done_total:,}", style=f"bold {WHITE}")
            plan.append(f" / {goal.total:,}  ", style=DIM)
            plan.append_text(progress_bar(goal.done_total, goal.total))
            table.add_row("PLAN", plan)

        stats = getattr(bot, "stats", None)
        levels = stats.levels if stats else 0
        elapsed = time.monotonic() - (stats.started if stats else self.started)
        avg = stats.avg_level_s() if stats else 0.0
        per_hour = levels / (elapsed / 3600) if elapsed > 60 and levels else 0.0
        session = Text(f"{levels:,} levels", style=f"bold {WHITE}")
        session.append(f"  ·  {fmt_duration(elapsed)}", style=DIM)
        table.add_row("SESSION", session)
        speed = Text(f"{avg:.1f}s", style=f"bold {WHITE}")
        speed.append(" per level  ·  ", style=DIM)
        speed.append(f"{per_hour:,.0f}", style=f"bold {WHITE}")
        speed.append(" / hour", style=DIM)
        table.add_row("SPEED", speed)

        if goal.per_day:
            left = max(0, goal.per_day - goal.done_today)
            if left == 0:
                eta = Text("today's target done ✔", style="bold #4ade80")
            elif avg:
                eta = Text(f"{fmt_duration(left * avg)}", style=f"bold {WHITE}")
                eta.append(f" to today's target ({left:,} left)", style=DIM)
            else:
                eta = Text("measuring…", style=DIM)
            table.add_row("ETA", eta)

        swipes = stats.swipes if stats else 0
        restarts = stats.restarts if stats else 0
        refused = getattr(getattr(bot, "device", None), "refused", 0)
        input_row = Text(f"{swipes:,}", style=f"bold {WHITE}")
        input_row.append(" swipes  ·  ", style=DIM)
        input_row.append(f"{restarts}", style=f"bold {WHITE if not restarts else '#facc15'}")
        input_row.append(" restarts", style=DIM)
        table.add_row("INPUT", input_row)
        safety = Text(f"{refused}", style=f"bold {'#4ade80' if not refused else '#f87171'}")
        safety.append(" taps refused (coins & hints are never touched)", style=DIM)
        table.add_row("SAFETY", safety)

        watcher = getattr(bot, "watcher", None)
        if watcher is not None:
            w = Text(f"{watcher.fps:.1f}", style=f"bold {WHITE}")
            w.append(" fps", style=DIM)
            for name, count in watcher.hits.most_common(4):
                w.append(f"  ·  {name} ", style=DIM)
                w.append(f"{count}", style=f"bold {WHITE}")
            table.add_row("WATCHER", w)
        return table


# ---- app -----------------------------------------------------------------------------

CSS = f"""
Screen {{
    background: #0d0d10;
    color: #f5f5f5;
    align: center top;
}}
#banner {{
    width: auto;
    height: auto;
    margin: 1 0 0 0;
}}
#tagline, #progress-line, #hint, #status {{
    text-align: center;
}}
#tagline, #progress-line {{
    width: 100%;
    content-align: center middle;
    color: {DIM};
    margin: 1 0 1 0;
}}
.panel {{
    border: round {ORANGE};
    border-title-color: {ORANGE};
    border-title-style: bold;
    background: #141418;
    padding: 1 2;
}}
#devices {{
    width: 80;
    max-width: 100%;
    height: auto;
}}
#modes {{
    width: 86;
    max-width: 100%;
    height: auto;
}}
#play {{
    width: 86;
    max-width: 100%;
    height: 3;
    content-align: center middle;
    margin: 1 0 0 0;
}}
#hint {{
    width: 100%;
    content-align: center middle;
    margin: 1 0 0 0;
}}
#mini-banner {{
    width: auto;
    height: auto;
    margin: 1 0 0 0;
}}
#status {{
    width: 100%;
    height: 1;
    margin: 1 0 0 0;
}}
#middle {{
    height: auto;
    max-height: 60%;
    margin: 1 1 0 1;
}}
#board {{
    width: 1fr;
    min-width: 36;
    height: auto;
    content-align: center middle;
}}
#stats {{
    width: 2fr;
    height: auto;
}}
#log {{
    height: 1fr;
    margin: 0 1;
    padding: 0 1;
    scrollbar-color: {ORANGE};
}}
Footer {{
    background: #141418;
}}
"""


BotFactory = Callable[[str, Goal], Any]


class WordsearchApp(App):
    TITLE = "WORDSEARCH-SOLVER"
    CSS = CSS

    def __init__(
        self,
        serial: str,
        root: Path,
        *,
        dry_run: bool = False,
        bot_factory: BotFactory | None = None,
        device_lister: Callable[[], list[AdbDevice]] = list_adb_devices,
    ) -> None:
        super().__init__()
        self.serial = serial
        self.root = root
        self.dry_run = dry_run
        self.device_lister = device_lister
        self.bot_factory = bot_factory or self._real_bot
        self._log: deque[tuple[str, str, str]] = deque(maxlen=2000)
        self._log_lock = threading.Lock()

    def _real_bot(self, serial: str, goal: Goal) -> Any:
        from .bot import Bot

        return Bot(serial, self.root, goal, dry_run=self.dry_run)

    def progress(self) -> Progress:
        return Progress(self.root / "local" / "progress.json")

    def sink(self, ts: str, tag: str, msg: str) -> None:
        """Log sink, safe to call from any thread."""
        with self._log_lock:
            self._log.append((ts, tag, msg))

    def drain_log(self) -> list[tuple[str, str, str]]:
        with self._log_lock:
            entries = list(self._log)
            self._log.clear()
        return entries

    def on_mount(self) -> None:
        self.register_theme(THEME)
        self.theme = "wsbot"
        (self.root / "local").mkdir(exist_ok=True)
        set_sinks(self.sink, file_sink(self.root / "local" / "wsbot.log"))
        self.push_screen(DeviceScreen())

    def on_unmount(self) -> None:
        for screen in self.screen_stack:
            if isinstance(screen, RunScreen) and screen.bot is not None:
                screen.bot.stop_event.set()
                if screen.thread:
                    screen.thread.join(timeout=5)


def run_tui(serial: str, root: Path, *, dry_run: bool = False) -> None:
    try:
        WordsearchApp(serial, root, dry_run=dry_run).run()
    finally:
        set_sinks(stdout_sink)
