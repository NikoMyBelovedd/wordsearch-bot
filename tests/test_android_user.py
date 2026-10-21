"""The game in a work profile (user 10) or Secure Folder (user 150): found, started and
stopped in its own Android user. Fake adb; not tried on a real phone."""

import subprocess

from wsbot import cli, device
from wsbot.device import Device

PACKAGE = "in.playsimple.wordsearch"
USERS = "Users:\n\tUserInfo{0:Owner:c13} running\n\tUserInfo{10:Work profile:1030} running\n"
RESOLVED = f"priority=0 preferredOrder=0 match=0x108000\n{PACKAGE}/.MainActivity\n"


def phone(answers):
    """A Device without a phone: `adb` answers from a table of argv -> output."""
    d = Device.__new__(Device)
    d._users, d.calls = {}, []

    def adb(*args, timeout=10.0):
        d.calls.append(args)
        out = answers.get(args, "")
        return subprocess.CompletedProcess(args, 0, out.encode(), b"")

    d.adb = adb
    return d


def path(user):
    return ("shell", "pm", "path", "--user", str(user), PACKAGE)


LISTING = ("shell", "pm", "list", "users")
APK = f"package:/data/app/{PACKAGE}/base.apk\n"


def test_users_are_read_main_first():
    assert device.parse_users(USERS) == [0, 10]
    assert device.parse_users("") == [0]


def test_finds_the_game_only_in_the_work_profile(monkeypatch):
    monkeypatch.delenv(device.USER_ENV, raising=False)
    d = phone({LISTING: USERS, path(10): APK})
    assert d.android_user(PACKAGE) == 10
    d.calls.clear()
    assert d.android_user(PACKAGE) == 10  # asked once
    assert d.calls == []


def test_the_main_user_wins_and_the_environment_skips_looking(monkeypatch):
    monkeypatch.delenv(device.USER_ENV, raising=False)
    d = phone({LISTING: USERS, path(0): APK, path(10): APK})
    assert d.android_user(PACKAGE) == 0
    assert path(10) not in d.calls
    monkeypatch.setenv(device.USER_ENV, "150")
    d = phone({})
    assert d.android_user(PACKAGE) == 150
    assert d.calls == []


def test_the_main_user_keeps_its_commands(monkeypatch):
    monkeypatch.delenv(device.USER_ENV, raising=False)
    d = phone({LISTING: USERS, path(0): APK})
    d.app_start(PACKAGE)
    d.app_stop(PACKAGE)
    assert d.calls[-2:] == [
        ("shell", "monkey", "-p", PACKAGE, "-c", "android.intent.category.LAUNCHER", "1"),
        ("shell", "am", "force-stop", PACKAGE),
    ]


def test_the_work_profile_game_starts_and_stops_in_its_user(monkeypatch):
    monkeypatch.setenv(device.USER_ENV, "10")
    resolve = ("shell", "cmd", "package", "resolve-activity", "--brief", "--user", "10", PACKAGE)
    d = phone({resolve: RESOLVED})
    d.app_start(PACKAGE)
    d.app_stop(PACKAGE)
    assert d.calls == [
        resolve,
        ("shell", "am", "start", "--user", "10", "-n", f"{PACKAGE}/.MainActivity"),
        ("shell", "am", "force-stop", "--user", "10", PACKAGE),
    ]


def test_no_main_screen_in_that_user_starts_nothing(monkeypatch):
    monkeypatch.setenv(device.USER_ENV, "10")
    resolve = ("shell", "cmd", "package", "resolve-activity", "--brief", "--user", "10", PACKAGE)
    d = phone({resolve: "No activity found"})
    d.app_start(PACKAGE)
    assert d.calls == [resolve]


def test_the_foreground_package_is_read_when_the_line_says_u10():
    d = phone(
        {
            (
                "shell",
                "dumpsys activity activities | grep -m1 topResumedActivity",
            ): f"  topResumedActivity=ActivityRecord{{7 u10 {PACKAGE}/.MainActivity t69}}"
        }
    )
    assert d.foreground() == PACKAGE


def test_the_cli_and_the_device_use_the_same_variable():
    assert cli.ANDROID_USER_ENV == device.USER_ENV
