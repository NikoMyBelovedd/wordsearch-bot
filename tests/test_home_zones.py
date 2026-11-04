"""The booster-row no-tap zone (below_board_y) holds only while a level's board is on screen.

run4: the level's zone outlived the level; the home screen's Play Level tap at (540,1993)
was refused ~70 times in 5 minutes. The zone now follows the watcher's latest look.
"""

from __future__ import annotations

import pytest

from wsbot import watcher as watcher_mod
from wsbot.device import BaseDevice, SafetyError

HOME_PLAY_LEVEL = (540, 1993)  # the home screen's tap (run4)


def device_with_watcher(board_visible: bool) -> BaseDevice:
    device = BaseDevice()
    device.below_board_y = 1950  # left over from the last level
    watcher = watcher_mod.PopupWatcher.__new__(watcher_mod.PopupWatcher)
    watcher.device = device
    watcher.board_visible = board_visible
    watcher.expected_panel = (0, 0, 1, 1) if board_visible else None
    watcher.popups, watcher._sys_ui = [], False
    device.set_zones_due(watcher._zones_due)
    return device


def test_the_stale_zone_does_not_refuse_the_home_screen_tap():
    device = device_with_watcher(board_visible=False)
    device._check_safe(*HOME_PLAY_LEVEL, "tap")  # no SafetyError


def test_the_zone_holds_while_the_board_is_on_screen():
    device = device_with_watcher(board_visible=True)
    with pytest.raises(SafetyError):
        device._check_safe(*HOME_PLAY_LEVEL, "tap")


def test_without_a_watcher_the_zone_still_holds():
    device = BaseDevice()
    device.below_board_y = 1950
    with pytest.raises(SafetyError):
        device._check_safe(*HOME_PLAY_LEVEL, "tap")
