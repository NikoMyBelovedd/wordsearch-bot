"""iPhone touches are always lifted: a call that times out after its touch-down must not
leave the finger down, or the next touch reads as dragging it across the screen."""

from __future__ import annotations

import asyncio

import pytest

from wsbot.iphone import CONTACT, IPhone, IPhoneError


@pytest.fixture
def phone():
    p = IPhone("fake")
    sent: list[tuple[str, int, int]] = []
    hang = {"next": False}

    async def send(state: int, x: int, y: int) -> None:
        if hang["next"]:
            hang["next"] = False
            await asyncio.sleep(60)  # the touch-down never comes back (a tunnel hiccup)
        sent.append(("down" if state == CONTACT else "up", x, y))

    p._send = send
    return p, sent, hang


def test_a_tap_that_times_out_still_lifts_its_finger(phone):
    p, sent, hang = phone
    hang["next"] = True
    with pytest.raises(IPhoneError):
        p._call(p._tap(10, 20, 30), 0.3)
    p._call(p._tap(500, 600, 1), 5)
    assert sent == [("up", 10, 20), ("down", 500, 600), ("up", 500, 600)]
    assert sent[0][0] == "up"  # the stuck finger was lifted before the new touch
    assert p._held is None


def test_a_new_touch_lifts_one_still_held(phone):
    p, sent, _ = phone
    p._call(p._down(1, 2), 5)  # e.g. a hold() whose release() never came
    p._call(p._tap(3, 4, 1), 5)
    assert sent == [("down", 1, 2), ("up", 1, 2), ("down", 3, 4), ("up", 3, 4)]


def test_a_swipe_moves_one_finger_and_lifts_it(phone):
    p, sent, _ = phone
    p._call(p._swipe(0, 0, 30, 0, 1, 3), 5)
    assert sent == [
        ("down", 0, 0),
        ("down", 10, 0),
        ("down", 20, 0),
        ("down", 30, 0),
        ("up", 30, 0),
    ]
