"""iPhones older than the iOS 27 backend get a clear error. No phone needed: pymobiledevice3's
usbmux, lockdown and tunneld lookups are replaced with fakes."""

from __future__ import annotations

import pytest
from pymobiledevice3 import lockdown, usbmux
from pymobiledevice3.tunneld import api as tunneld_api

import wsbot.iphone
from wsbot.iphone import IPhone, NoTunnel, TooOld, check_ios, ios_version, too_old

OLD, NEW = "00008020-OLD", "00008110-NEW"


def test_versions():
    assert ios_version("26.4.1") == (26, 4, 1)
    assert ios_version("27.0") == (27, 0)
    assert ios_version("") == ios_version(None) == ()
    assert too_old("16.7.10") and too_old("17.4") and too_old("26.5")
    assert not too_old("27.0") and not too_old("28.1")
    assert not too_old(None) and not too_old("unknown")  # unknown: let the tunnel path decide
    with pytest.raises(TooOld, match=r"has iOS 15\.8\.3; .* needs iOS 27 or newer"):
        check_ios(OLD, "15.8.3")
    check_ios(NEW, "27.0")


class FakeMuxDevice:
    def __init__(self, serial: str, usb: bool = True):
        self.serial = serial
        self.is_usb = usb


class FakeLockdown:
    def __init__(self, version: str):
        self.all_values = {"ProductVersion": version}
        self.closed = False

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def usb(monkeypatch):
    """{udid: iOS version} of the phones on USB; None = lockdown fails for it."""
    phones: dict[str, str | None] = {}
    opened: list[FakeLockdown] = []

    async def list_devices(usbmux_address=None):
        return [FakeMuxDevice(udid) for udid in phones]

    async def create_using_usbmux(serial=None, autopair=True, connection_type=None, **_):
        assert autopair is False, "never asks the phone for Trust"
        if phones[serial] is None:
            raise ConnectionError("lockdown failed")
        opened.append(FakeLockdown(phones[serial]))
        return opened[-1]

    monkeypatch.setattr(usbmux, "list_devices", list_devices)
    monkeypatch.setattr(lockdown, "create_using_usbmux", create_using_usbmux)
    yield phones
    assert all(lock.closed for lock in opened)


def preflight(udid: str | None = None) -> None:
    phone = IPhone(udid)
    phone._call(phone._check_usb_versions(), 10)


def test_an_old_iphone_is_refused_before_any_tunnel(usb):
    usb[OLD] = "16.7.10"
    with pytest.raises(TooOld, match=r"16\.7\.10"):
        preflight(OLD)
    with pytest.raises(TooOld):
        preflight()  # the only iPhone plugged in


def test_new_or_unknown_iphones_go_ahead_as_before(usb):
    preflight()  # nothing plugged in: the tunnel path reports it
    usb[NEW] = "27.0"
    preflight(NEW)
    usb[OLD] = "16.7.10"
    preflight()  # without a UDID a new-enough phone is still there
    preflight(NEW)
    usb[OLD] = None  # can't read it: no guessing
    preflight(OLD)


class FakeRsd:
    def __init__(self, udid: str, version: str):
        self.udid, self.product_version, self.product_type = udid, version, "iPhone12,8"
        self.peer_info = {"Properties": {}, "Services": {}}
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def test_a_tunnel_to_an_ios_17_to_26_iphone_is_refused_with_its_version(usb, monkeypatch):
    monkeypatch.setattr(wsbot.iphone, "TUNNEL_MODE", "tunneld")
    rsds = [FakeRsd(OLD, "26.5")]

    async def get_tunneld_devices(*_, **__):
        return rsds

    monkeypatch.setattr(tunneld_api, "get_tunneld_devices", get_tunneld_devices)
    phone = IPhone(OLD)
    with pytest.raises(TooOld, match=r"26\.5"):
        phone._call(phone._open(), 10)
    assert rsds[0].closed

    # Without a UDID a new-enough phone is picked over an older one.
    new = FakeRsd(NEW, "27.0")
    rsds[:] = [FakeRsd(OLD, "26.5"), new]
    phone = IPhone()

    class StopHere:
        def __init__(self, rsd, **_):
            raise NoTunnel(f"stop here with {rsd.udid}: the screen stream is out of scope")

    monkeypatch.setattr(
        "pymobiledevice3.remote.core_device.screen_stream.ScreenStreamServer", StopHere
    )
    with pytest.raises(NoTunnel, match=f"stop here with {NEW}"):
        phone._call(phone._open(), 10)
    assert phone.product_type == "iPhone12,8" and not new.closed
