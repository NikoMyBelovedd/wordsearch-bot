"""The Android device layer: screen scaling. The input shell (a wedged adb shell can't
freeze the bot) is tested in ahq-device, where android.py comes from."""

from __future__ import annotations

from wsbot import device


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


class FakePhone:
    """Stands in for android.Android (tested in ahq-device)."""

    def __init__(self, stop_fails: bool = False) -> None:
        self.stop_fails = stop_fails
        self.calls: list[tuple[str, str]] = []

    def foreground_package(self) -> str:
        return "com.game"

    def stop_app(self, package: str) -> None:
        self.calls.append(("stop", package))
        if self.stop_fails:
            raise device.DeviceError("adb shell timed out")

    def start_app(self, package: str) -> None:
        self.calls.append(("start", package))


def test_the_game_is_started_even_when_stopping_it_failed():
    d = device.Device.__new__(device.Device)
    d.phone = FakePhone(stop_fails=True)
    assert d.foreground() == "com.game"
    d.app_stop("com.game")  # logged, not raised: the start that follows still runs
    d.app_start("com.game")
    assert d.phone.calls == [("stop", "com.game"), ("start", "com.game")]
