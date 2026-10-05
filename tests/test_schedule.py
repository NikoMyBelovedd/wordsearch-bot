"""Breaks: the play clock restarts after any rest as long as a break."""

from __future__ import annotations

import random
import threading
import time
from types import SimpleNamespace

from wsbot.bot import Bot
from wsbot.goal import Progress
from wsbot.schedule import Pacer, Schedule


def pacer(tmp_path, **kw) -> Pacer:
    return Pacer(Schedule(**kw), Progress(tmp_path / "p.json"), rng=random.Random(1))


def test_break_due_after_a_stretch_of_play(tmp_path):
    p = pacer(tmp_path)
    assert p.break_due() == 0
    p.last_break_end -= 3 * 3600  # three hours of play
    assert 25 * 60 <= p.break_due() <= 35 * 60


def test_the_night_counts_as_a_break(tmp_path):
    p = pacer(tmp_path)
    p.last_break_end -= 10 * 3600  # bot started yesterday evening, then waited all night
    p.rested(9 * 3600)
    assert p.break_due() == 0  # no break right after the first level of the morning


def test_short_rests_dont_reset_the_clock(tmp_path):
    p = pacer(tmp_path)
    p.last_break_end -= 3 * 3600
    p.rested(5 * 60)  # an idle between levels is part of play
    assert p.break_due() > 0


def bot_shell(p: Pacer) -> Bot:
    b = Bot.__new__(Bot)
    b.pacer = p
    b.stop_event = threading.Event()
    b.pause_event = threading.Event()
    b.watcher = SimpleNamespace(idle=threading.Event(), resting=threading.Event())
    b._relaunch = False
    b.resting_until = None
    b.status = ""
    return b


def test_long_rest_and_long_pause_restart_the_clock(tmp_path, monkeypatch):
    p = pacer(tmp_path, break_len_min=0)  # any rest is "as long as a break"
    b = bot_shell(p)
    p.last_break_end -= 3 * 3600
    b._rest(0.05, "waiting for today's play window")
    assert p.break_due() == 0

    p.last_break_end -= 3 * 3600
    b.pause_event.set()
    threading.Timer(0.1, b.pause_event.clear).start()
    b._hold()
    assert p.break_due() == 0


def test_sleep_until_tomorrow_restarts_the_clock(tmp_path, monkeypatch):
    p = pacer(tmp_path)
    b = bot_shell(p)
    b.goal = SimpleNamespace(per_day=1400)
    monkeypatch.setattr("wsbot.bot.seconds_until_midnight", lambda: -4.99)
    p.last_break_end = time.time() - 10 * 3600
    b._sleep_until_tomorrow()
    assert p.break_due() == 0


def test_catchup_day_ends_at_midnight(tmp_path, monkeypatch):
    from wsbot.bot import _arm_catchup
    from wsbot.goal import Goal

    goal = Goal.custom(Progress(tmp_path / "p.json"), Schedule(per_day=500))
    (tmp_path / "local").mkdir()
    (tmp_path / "local" / "catchup-once").write_text("300\n", encoding="utf-8")
    monkeypatch.delenv("WSBOT_CATCHUP", raising=False)
    _arm_catchup(goal, tmp_path)
    assert goal.per_day == 800 and goal.schedule.per_day == 800
    assert not goal.schedule.breaks
    assert not (tmp_path / "local" / "catchup-once").exists()  # this launch only
    assert not goal.end_catchup_if_new_day()
    monkeypatch.setattr("wsbot.goal.today", lambda: "2999-01-01")
    assert goal.end_catchup_if_new_day()
    assert goal.per_day == 500 and goal.schedule.breaks
