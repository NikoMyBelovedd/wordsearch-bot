"""The iPhone's own pop-ups: system alerts (tap only a safe button, never Allow / Install /
Trust...) and notification banners (wait them out, never tap them).

Real frames: tests/data/screens/alert_watch_notifications*.jpg (iPhone SE, iOS 27).
Synthetic ones: tests/data/alerts/ (tools/make_alert_frames.py renders iOS 26/27 alerts
and banners onto real game frames, at the SE size and an iPhone 17's)."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from wsbot import sysalert
from wsbot import watcher as watcher_mod
from wsbot.device import BaseDevice, SafetyError
from wsbot.ios_device import IOSGameDevice
from wsbot.letters import TESSERACT
from wsbot.shot import Shot
from wsbot.watcher import PopupWatcher, board_visible_in

ROOT = Path(__file__).resolve().parents[1]
SCREENS = ROOT / "tests" / "data" / "screens"
ALERTS = ROOT / "tests" / "data" / "alerts"
TRUTH = json.loads((ALERTS / "truth.json").read_text(encoding="utf-8"))
SE_CALIB = (1296, 2305)
TALL_CALIB = (1080, round(2622 * 1080 / 1206))
ocr = pytest.mark.skipif(TESSERACT is None, reason="Tesseract isn't installed")

ALERT_FRAMES = sorted(n for n, t in TRUTH.items() if "banner" not in t)
BANNER_FRAMES = sorted(n for n, t in TRUTH.items() if "banner" in t)
REAL_ALERTS = ["alert_watch_notifications", "alert_watch_notifications_2"]
GAME_SCREENS = sorted(p.stem for p in SCREENS.glob("*.jpg") if not p.stem.startswith("alert_watch"))


def frame(name: str) -> np.ndarray:
    path = ALERTS / f"{name}.jpg"
    img = cv2.imread(str(path if path.exists() else SCREENS / f"{name}.jpg"))
    assert img is not None, name
    return img


def calib_of(img: np.ndarray) -> tuple[int, int]:
    return SE_CALIB if img.shape[1] == 750 else TALL_CALIB


# ---- choosing a button -----------------------------------------------------------------


def test_safe_buttons_in_priority_order():
    assert sysalert.choose(["Don\u2019t Allow", "Allow"]).button == 0
    assert sysalert.choose(["Allow", "Don't Allow"]).button == 1
    assert sysalert.choose(["Close", "Low Power Mode"]).label == "Close"
    assert sysalert.choose(["Install Now", "Later"]).label == "Later"
    assert sysalert.choose(["Install Tonight", "Remind Me Later", "Details"]).button == 1
    assert sysalert.choose(["Ask App Not to Track", "Allow"]).button == 0
    assert sysalert.choose(["Not Now", "Settings"]).button == 0
    assert sysalert.choose(["Later", "Remind Me Later"]).label == "Later"
    assert sysalert.choose(["Cancel", "Turn Off"]).label == "Cancel"


def test_ok_only_when_it_is_the_only_button():
    assert sysalert.choose(["OK"]).button == 0
    assert sysalert.choose(["Settings", "OK"]).button is None
    assert sysalert.choose(["OK", "Buy"]).button is None


@pytest.mark.parametrize(
    "labels",
    [
        ["Allow"],
        ["Allow Once", "Allow While Using App"],
        ["Install"],
        ["Install Now"],
        ["Update"],
        ["Buy"],
        ["Call"],
        ["Settings"],
        ["Turn On"],
        ["Continue"],
        ["Low Power Mode"],
        ["Details"],
        ["Open"],
        ["Delete App", "Keep App"],
        ["", ""],
        [],
    ],
)
def test_unsafe_buttons_are_never_chosen(labels):
    assert sysalert.choose(labels).button is None


def test_trust_this_computer_is_never_tapped():
    for labels in (["Trust", "Don't Trust"], ["Don\u2019t Trust", "Trust"]):
        choice = sysalert.choose(labels)
        assert choice.button is None and choice.trust
    # even if OCR made a safe-looking word out of something on it
    choice = sysalert.choose(["Cancel"], title="Trust This Computer?")
    assert choice.button is None and choice.trust


def test_ocr_slips_are_forgiven_only_on_long_labels():
    assert sysalert.safe_label("Remind Me Latr") == "remind me later"
    assert sysalert.safe_label("Dont Allow") == "dont allow"
    assert sysalert.safe_label("Don't  Allow.") == "dont allow"
    assert sysalert.safe_label("Allow") is None  # 5 edits from "Don't Allow"
    assert sysalert.safe_label("Clone") is None  # short labels must match exactly
    assert sysalert.safe_label("OKAY") is None
    assert sysalert.safe_label("Install Now") is None  # not "Not Now"


# ---- finding alerts and banners --------------------------------------------------------


@pytest.mark.parametrize("name", REAL_ALERTS + ALERT_FRAMES)
def test_alerts_are_found_with_their_buttons(name):
    img = frame(name)
    alert = sysalert.find_alert(img)
    assert alert is not None, name
    want = TRUTH[name]["centers"] if name in TRUTH else [(227, 802), (523, 802)]
    assert len(alert.buttons) == len(want)
    for i, (x, y) in enumerate(want):
        cx, cy = alert.center(i)
        assert abs(cx - x) < 0.02 * img.shape[1] and abs(cy - y) < 0.01 * img.shape[0]
    assert sysalert.find_banner(img) is None


@pytest.mark.parametrize("name", GAME_SCREENS + BANNER_FRAMES)
def test_game_screens_are_not_alerts(name):
    """Every saved game screen (popups, level complete, home screen, app switcher...)."""
    assert sysalert.find_alert(frame(name)) is None


def test_the_already_collected_toast_over_the_board_is_not_an_alert():
    """Real iPhone, 2026-10-02: the game's "You have already collected this word!" toast
    (a gray box at the board's bottom) inside the white board panel read as a one-button
    alert. iOS rounds an alert box's corners and makes its buttons capsules; the panel and
    the toast have small corner radii."""
    img = frame("toast_over_board")
    assert sysalert.find_alert(img) is None
    # and the toast is the game's own: its template still sees it
    shot = Shot(1, img, SE_CALIB)
    toast = next(p for p in watcher_mod.load_popups(ROOT / "templates" / "ios") if p.covers)
    assert watcher_mod.match(shot.small, toast, shot.coarse_as(toast.coarse_look))[0] >= 0.85


def _tutorial_box_at_bottom(img: np.ndarray, scale: float, bottom: int) -> np.ndarray:
    """claim_bonus_tutorial with the tutorial box shrunk and moved to the bottom of the
    popup's white body (painted over the Claim button), where an alert's buttons sit."""
    out = img.copy()
    box = out[882:1014, 100:650].copy()
    out[870:1066, 104:646] = (255, 255, 240)
    small = cv2.resize(box, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = small.shape[:2]
    out[bottom - h : bottom, 375 - w // 2 : 375 - w // 2 + w] = small
    return out


def test_the_claim_bonus_tutorial_box_is_not_an_alert(monkeypatch):
    """Real iPhone (the laptop, alerts-1), 2026-10-02 11:57: the "Claim bonus word reward"
    tutorial box over the Bonus Words popup read as a one-button alert ([Claim bonus word
    reward], no safe button): the popup's white body passed as the alert box, the
    tutorial box as its button. The rounded-corner rules (see the toast test) stop it:
    the body's top corners are square (its colored header) and the tutorial box has
    small corner radii, not capsule ends. That frame wasn't saved; this is the same
    screen, and versions of it with the tutorial box at the body's bottom, which are
    alert-shaped but for their corners (the rules without corners fire on them)."""
    img = frame("claim_bonus_tutorial")
    assert sysalert.find_alert(img) is None
    near = [
        _tutorial_box_at_bottom(img, sc, bottom)
        for sc, bottom in [(0.8, 1040), (0.85, 1055), (0.9, 1055)]
    ]
    assert all(sysalert.find_alert(f) is None for f in near)
    monkeypatch.setattr(sysalert, "_round_corners", lambda filled, k: True)
    assert all(sysalert.find_alert(f) is not None for f in near)


@pytest.mark.parametrize("name", GAME_SCREENS + REAL_ALERTS)
def test_game_screens_have_no_banner(name):
    assert sysalert.find_banner(frame(name)) is None


@pytest.mark.parametrize("name", BANNER_FRAMES)
def test_banners_are_found(name):
    img = frame(name)
    box = sysalert.find_banner(img)
    assert box is not None
    want = TRUTH[name]["banner"]
    assert all(abs(a - b) <= 0.02 * img.shape[1] for a, b in zip(box, want, strict=True))


@pytest.mark.parametrize("name", [n for n in BANNER_FRAMES if n.startswith("se_")])
def test_a_banner_over_the_top_doesnt_hide_the_board(name):
    """A banner covers the top bar and the top of the hint card: the board under it is
    read the same, so the level goes on (and nothing taps the banner)."""
    clean = Shot(1, frame("board_readable"), SE_CALIB)
    covered = Shot(2, frame(name), SE_CALIB)
    assert covered.panel == clean.panel
    assert board_visible_in(covered, clean.panel) and board_visible_in(covered, None)
    assert covered.board.rows == clean.board.rows and covered.board.cols == clean.board.cols


def test_the_alert_detector_is_cheap():
    """It runs about twice a second while no board is in view, on 2-core laptops too."""
    import time

    img = frame("next_level_plain")
    t0 = time.perf_counter()
    for _ in range(10):
        sysalert.find_alert(img)
        sysalert.find_banner(img)
    assert (time.perf_counter() - t0) / 10 < 0.05


# ---- reading them ----------------------------------------------------------------------


@ocr
@pytest.mark.parametrize("name", REAL_ALERTS + ALERT_FRAMES)
def test_labels_are_read_and_only_a_safe_one_is_chosen(name):
    img = frame(name)
    alert = sysalert.find_alert(img)
    labels = sysalert.read_labels(img, alert)
    title = sysalert.read_title(img, alert)
    want = TRUTH.get(name, {"buttons": ["Don't Allow", "Allow"], "tap": "Don't Allow"})
    assert [sysalert.normalize(s) for s in labels] == [
        sysalert.normalize(s) for s in want["buttons"]
    ]
    choice = sysalert.choose(labels, title)
    if want["tap"] is None:
        assert choice.button is None
    else:
        assert choice.label == want["tap"]
        assert labels[choice.button].replace("\u2019", "'") == want["tap"]
    if "trust" in name:
        assert choice.trust and "Trust This Computer" in title


# ---- the watcher acting on them --------------------------------------------------------


class Phone(BaseDevice):
    """A fake iPhone with the real no-tap zones (BaseDevice._check_safe)."""

    platform = "ios"
    zones = IOSGameDevice.zones
    dry_run = False

    def __init__(self, shots: list[Shot]) -> None:
        self.calib = shots[0].calib_size
        self.shots = shots
        self.taps: list[tuple[int, int, str, str | None]] = []
        self.refused_taps: list[tuple[int, int, str]] = []
        self.launches = self.stops = self.reconnects = 0

    def shot(self) -> Shot:
        return self.shots.pop(0) if len(self.shots) > 1 else self.shots[0]

    def tap(self, x, y, *, why="", allow=None) -> bool:
        try:
            self._check_safe(x, y, "tap", allow)
        except SafetyError:
            self.refused_taps.append((x, y, why))
            return False
        self.taps.append((x, y, why, allow))
        return True

    def foreground(self) -> str:
        return "pkg"

    def app_start(self, package) -> None:
        self.launches += 1

    def app_stop(self, package) -> None:
        self.stops += 1

    def reconnect(self) -> None:
        self.reconnects += 1


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(watcher_mod.time, "monotonic", c)
    # watcher_mod.time is the time module: a no-op sleep for every thread turned other
    # tests' leftover background loops (time.sleep(10)) into busy loops (tests 100x slower)
    real_sleep = watcher_mod.time.sleep
    main = threading.main_thread()
    monkeypatch.setattr(
        watcher_mod.time,
        "sleep",
        lambda s: None if threading.current_thread() is main else real_sleep(s),
    )
    return c


@pytest.fixture
def logged(monkeypatch):
    lines: list[tuple[str, str]] = []
    real = watcher_mod.log

    def log(tag, msg):
        lines.append((tag, msg))
        real(tag, msg)

    monkeypatch.setattr(watcher_mod, "log", log)
    return lines


def shots_of(name: str, n: int = 1, start: int = 1) -> list[Shot]:
    img = frame(name)
    return [Shot(start + i, img, calib_of(img)) for i in range(n)]


def watcher_on(shots: list[Shot], tmp_path: Path) -> tuple[PopupWatcher, Phone]:
    phone = Phone(shots)
    tall = shots[0].calib_size != SE_CALIB
    w = PopupWatcher(phone, ROOT / "templates" / ("ios_tall" if tall else "ios"), "pkg", tmp_path)
    w._last_app_check = float("inf")
    return w, phone


def run(w: PopupWatcher, clock: Clock, seconds: float, step: float = 0.5) -> None:
    end = clock.t + seconds
    while clock.t < end - 1e-9:
        clock.t += step
        w._tick()


def native_to_calib(img: np.ndarray, x: float, y: float) -> tuple[float, float]:
    cw, ch = calib_of(img)
    return x * cw / img.shape[1], y * ch / img.shape[0]


def test_the_toast_never_puts_up_an_alert_zone(clock, tmp_path, logged):
    """Off the board (no level being solved) the toast frame is no alert: no no-tap zone,
    no "iPhone alert" line, no OCR."""
    w, phone = watcher_on(shots_of("toast_over_board", 6), tmp_path)
    run(w, clock, 3.0)
    assert w.alert is None
    assert "ios_alert" not in phone.__dict__.get("dynamic_zones", {})
    assert not [m for _, m in logged if m.startswith("iPhone alert")]


@ocr
@pytest.mark.parametrize("name", [n for n in ALERT_FRAMES if TRUTH[n]["tap"]])
def test_a_safe_alert_gets_one_tap_on_its_safe_button(name, clock, tmp_path):
    w, phone = watcher_on(shots_of(name), tmp_path)
    run(w, clock, 2.0)
    assert len(phone.taps) == 1, phone.taps
    x, y, why, allow = phone.taps[0]
    want = native_to_calib(frame(name), *TRUTH[name]["tap_center"])
    assert abs(x - want[0]) < 25 and abs(y - want[1]) < 25
    assert why == f"iPhone alert: {TRUTH[name]['tap']}" and allow == "ios_alert"
    assert not w.blind_taps_ok()  # the bot's blind taps could land on "Allow"
    assert not list(tmp_path.glob("unknown_popup_*"))  # handled, not unknown


@ocr
@pytest.mark.parametrize("name", [n for n in ALERT_FRAMES if not TRUTH[n]["tap"]])
def test_an_alert_with_no_safe_button_is_never_tapped(name, clock, tmp_path, logged):
    """Settings/OK, Update, Trust: no tap, ever. The picture is saved once and the
    unknown-screen recovery takes over (relaunch, restart, then a loud exit)."""
    w, phone = watcher_on(shots_of(name), tmp_path)
    fatal: list[str] = []
    w.on_fatal = fatal.append
    run(w, clock, 300, step=1.0)
    assert phone.taps == [] and phone.refused_taps == []
    assert phone.launches >= 1  # the escalation ran
    assert len(fatal) == 1 and "iPhone alert it may not tap" in fatal[0]
    assert len(list(tmp_path.glob("unknown_popup_*"))) == 1
    reads = [m for t, m in logged if m.startswith("iPhone alert")]
    assert reads and "not tapping" in reads[0]
    if "trust" in name:
        warned = [m for t, m in logged if "Trust This Computer" in m and t == "WARN"]
        assert len(warned) == 1 and "passcode" in warned[0]


@ocr
def test_an_alert_that_stays_is_tapped_at_most_three_times(clock, tmp_path):
    w, phone = watcher_on(shots_of("se_low_battery_light"), tmp_path)
    run(w, clock, 60)
    assert [t[2] for t in phone.taps] == ["iPhone alert: Close"] * watcher_mod.ALERT_MAX_TAPS


@ocr
def test_a_new_alert_in_the_same_place_is_read_again(clock, tmp_path):
    """After a tap the next alert may sit in the same spot with other buttons: the
    labels are read again before any other tap."""
    shots = shots_of("se_notifications_dark", 4) + shots_of("se_trust_light", 40, start=100)
    w, phone = watcher_on(shots, tmp_path)
    run(w, clock, 15)
    assert [t[2] for t in phone.taps] == ["iPhone alert: Don't Allow"]


@ocr
def test_the_real_watch_alert_only_ever_gets_dont_allow(clock, tmp_path):
    """The Don't Allow template and the alert reader both know this one: every tap
    either makes is on Don't Allow (left), never Allow."""
    for name in REAL_ALERTS:
        w, phone = watcher_on(shots_of(name), tmp_path)
        run(w, clock, 10)
        assert phone.taps and all(x < 560 and 1300 < y < 1460 for x, y, *_ in phone.taps)


def test_no_tap_lands_inside_an_alert_but_its_own(clock, tmp_path):
    w, phone = watcher_on(shots_of("se_settings_ok_dark"), tmp_path)
    run(w, clock, 1.0)
    x0, y0, x1, y1 = w.alert.zone
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    assert not phone.tap(cx, cy, why="clear popup (board center)")  # a blind clear tap
    assert phone.tap(cx, cy, why="its own button", allow="ios_alert")
    assert phone.tap(20, 300, why="outside it")


def test_a_banner_is_waited_out_never_tapped(clock, tmp_path):
    """The level goes on under a banner; taps under it are refused while it's up and
    a moment after (it slides away), then allowed again."""
    shots = shots_of("board_readable", 2) + shots_of("se_banner_light", 10, start=10)
    shots += shots_of("board_readable", 20, start=100)
    w, phone = watcher_on(shots, tmp_path)
    w.expected_panel = Shot(0, frame("board_readable"), SE_CALIB).panel
    star = (225, 135)  # the tutorial taps the bonus jar here (top bar)
    run(w, clock, 1.0)
    assert phone.tap(*star, allow="star_bonus_jar")
    run(w, clock, 1.0)
    assert w.board_visible  # still playing
    assert not phone.tap(*star, allow="star_bonus_jar")
    assert phone.tap(648, 1244)  # the board itself is fine
    run(w, clock, 5.0)
    assert w.board_visible
    run(w, clock, 4.0)
    assert phone.tap(*star, allow="star_bonus_jar")
    assert not list(tmp_path.glob("unknown_popup_*"))
    assert phone.launches == 0


def test_banner_checks_wait_for_an_input_near_the_top(clock, tmp_path, monkeypatch):
    """Mid-level the bot only swipes on the board, which no banner zone can reach: the
    frames due a banner check are kept (a few) but not looked at. The first tap near
    the top looks at what's due and sees the zone the old every-0.3 s check had."""
    looked = []
    real = sysalert.find_banner_in
    monkeypatch.setattr(sysalert, "find_banner_in", lambda *a: looked.append(1) or real(*a))
    shots = shots_of("board_readable", 2) + shots_of("se_banner_light", 10, start=10)
    shots += [Shot(100 + i, frame("board_readable").copy(), SE_CALIB) for i in range(40)]
    w, phone = watcher_on(shots, tmp_path)
    w.expected_panel = Shot(0, frame("board_readable"), SE_CALIB).panel
    for _ in range(12):  # 2 frames of the board, then 10 with the banner over it
        run(w, clock, 0.2, step=0.2)
        assert phone.tap(648, 1244)  # on the board: never looks
    assert looked == [] and len(w._banner_looks) <= 5
    assert not phone.tap(225, 135, allow="star_bonus_jar")  # the banner is up
    assert 1 <= len(looked) <= 5
    run(w, clock, 0.6, step=0.2)  # the banner went 0.6 s ago: its zone lingers
    assert not phone.tap(225, 135, allow="star_bonus_jar")
    run(w, clock, 0.4, step=0.2)  # ...for BANNER_LINGER_S
    assert phone.tap(225, 135, allow="star_bonus_jar")
    n = len(looked)
    run(w, clock, 5.0, step=0.2)
    assert len(looked) == n and len(w._banner_looks) <= 5


def test_android_has_no_iphone_alert_checks(tmp_path):
    shots = shots_of("se_notifications_dark")
    phone = Phone(shots)
    phone.platform = "android"
    w = PopupWatcher(phone, ROOT / "templates" / "ios", "pkg", tmp_path)
    assert not w._sys_ui


@pytest.mark.parametrize("name", BANNER_FRAMES + REAL_ALERTS + ["board_readable"])
def test_the_half_frame_shortcuts_are_the_same_resize(name):
    """find_alert / find_banner_in reuse Shot.mini on an SE frame: same answers."""
    img = frame(name)
    shot = Shot(1, img, calib_of(img))
    assert sysalert.find_alert(img, shot.mini) == sysalert.find_alert(img)
    half = sysalert.half_banner_area(img, shot.mini)
    if img.shape[1] != 750:
        assert half is None
        return
    want, _ = sysalert._shrink(sysalert.banner_area(img))
    assert np.array_equal(half, want)
    assert sysalert.find_banner_in(half, img.shape[1]) == sysalert.find_banner(img)
