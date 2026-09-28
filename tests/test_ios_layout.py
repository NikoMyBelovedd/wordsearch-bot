"""iOS layouts without a phone: a fake IPhone stands in for the USB session."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

import wsbot.iphone
from wsbot import debug
from wsbot.imgio import imread
from wsbot.ios_device import IOSGameDevice, _box, model_name
from wsbot.iphone import _nal_types
from wsbot.watcher import _tap_point, load_popups

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / "data"


class FakePhone:
    def __init__(self, size: tuple[int, int], product_type: str):
        self.width, self.height = size
        self.product_type = product_type

    def __call__(self, udid=None):  # stands in for the IPhone class
        self.udid = udid or "FAKE"
        return self

    def open(self) -> None:
        pass


def device(monkeypatch, size, product_type) -> IOSGameDevice:
    monkeypatch.setattr(wsbot.iphone, "IPhone", FakePhone(size, product_type))
    return IOSGameDevice()


def test_se_layout_is_unchanged(monkeypatch):
    dev = device(monkeypatch, (750, 1334), "iPhone14,6")
    assert dev.layout == "se" and dev.templates == "templates/ios"
    assert dev.calib == (1296, 2305)
    assert dev.zones["back_button"] == _box(25, 45, 95, 112)
    assert dev.anchor("star") == (225, 135) and dev.anchor("trophy") == (109, 530)


def test_iphone17_is_tall_and_width_fit(monkeypatch):
    dev = device(monkeypatch, (1206, 2622), "iPhone18,3")
    assert dev.layout == "tall" and dev.templates == "templates/ios_tall"
    assert dev.calib[0] == 1080
    assert dev.anchor("star") is None  # unknown until the top bar is found
    for name in ("back_button", "star_bonus_jar", "coins_and_shop", "banner_ad"):
        assert name in dev.zones  # safe guesses before the top bar is located


def test_iphone17_top_bar_found_on_a_real_frame(monkeypatch):
    dev = device(monkeypatch, (1206, 2622), "iPhone18,3")
    top = imread(DATA / "iphone17_top.jpg")
    frame = np.zeros((2622, 1206, 3), np.uint8)
    frame[: top.shape[0]] = top
    dev._locate_topbar(frame)
    assert dev._topbar is not None
    bx, by, cx, cy = dev._topbar
    assert abs(bx - 93) < 12 and abs(by - 213) < 12 and abs(cx - 979) < 12 and abs(cy - 213) < 12
    x1, y1, x2, y2 = dev.zones["coins_and_shop"]
    assert x1 < cx < x2 and y1 < cy < y2
    assert dev.anchor("star") is not None


def test_model_names():
    assert model_name("iPhone18,3") == "iPhone 17"
    assert model_name("iPhone14,6") == "iPhone SE 3"
    assert model_name("iPhone99,9") == "iPhone99,9"
    assert model_name("") == "iPhone"


def test_tall_popups_load_and_anchor_tap_points():
    folder = ROOT / "templates" / "ios_tall"
    entries = json.loads((folder / "popups.json").read_text(encoding="utf-8"))
    assert len(load_popups(folder)) == len(entries)
    assert _tap_point("@star") == "@star"
    assert _tap_point([1, 2]) == (1, 2)
    assert _tap_point(None) is None


def test_nal_types():
    vps = (4).to_bytes(4, "big") + bytes([32 << 1, 1, 0, 0])
    idr = (3).to_bytes(4, "big") + bytes([19 << 1, 1, 0])
    assert _nal_types(vps + idr) == [32, 19]


@pytest.mark.parametrize("budget", [60_000, 2_000_000])
def test_report_stays_under_budget(tmp_path: Path, budget: int):
    rng = np.random.default_rng(0)
    (tmp_path / "local").mkdir()
    (tmp_path / "diagnostics" / "debug").mkdir(parents=True)
    words = np.array([b"LEVEL ", b"TAP ", b"(123,456) ", b"swipe ", b"ok\n"])
    log = b"".join(rng.choice(words, 400_000))
    (tmp_path / "local" / "wsbot.log").write_bytes(log)
    for i in range(8):  # incompressible "screenshots"
        (tmp_path / "diagnostics" / "debug" / f"calib_frame_{i:03d}.jpg").write_bytes(
            rng.bytes(300_000)
        )
    out = debug.make_report(tmp_path, budget=budget, out=tmp_path / "r.zip")
    assert out.stat().st_size <= budget
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        assert "local/wsbot.log" in names
        assert z.read("local/wsbot.log").endswith(log[-2000:])  # the newest lines survive
