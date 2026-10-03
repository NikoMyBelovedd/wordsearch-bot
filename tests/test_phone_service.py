"""AutomationHQ's iPhone service: the bot uses it when AutomationHQ says so (AHQ_IPHONE)
and falls back to its own USB session when the service can't serve it. A stand-in
`ahq_iphone.client` module, so no AutomationHQ and no phone are needed."""

from __future__ import annotations

import sys
import threading
import types

import pytest

from wsbot import ios_device, iphone


class ClientError(RuntimeError):
    pass


class ClientNoTunnel(ClientError):
    pass


class ClientUnsupported(ClientError):
    pass


class ClientUnreachable(ClientError):
    pass


class ServicePhone:
    """Fails open() with the queued errors, then works."""

    errors: list[BaseException] = []  # noqa: RUF012

    def __init__(self, udid=None):
        self.udid = udid or ""
        self.cancel = threading.Event()
        self.opened = self.closed = 0

    def open(self):
        if ServicePhone.errors:
            raise ServicePhone.errors.pop(0)
        self.opened += 1

    def close(self):
        self.closed += 1


class OwnPhone:
    made: list[OwnPhone] = []  # noqa: RUF012

    def __init__(self, udid=None):
        self.udid = udid
        self.opened = 0
        OwnPhone.made.append(self)

    def open(self):
        self.opened += 1

    def close(self):
        pass


@pytest.fixture
def service(monkeypatch):
    client = types.ModuleType("ahq_iphone.client")
    client.Phone = ServicePhone
    client.available = lambda: True
    client.NoTunnel = ClientNoTunnel
    client.Unsupported = ClientUnsupported
    client.Unreachable = ClientUnreachable
    pkg = types.ModuleType("ahq_iphone")
    pkg.client = client
    monkeypatch.setitem(sys.modules, "ahq_iphone", pkg)
    monkeypatch.setitem(sys.modules, "ahq_iphone.client", client)
    monkeypatch.setenv("AHQ_IPHONE", "127.0.0.1:1")
    monkeypatch.setenv("AHQ_IPHONE_TOKEN", "x" * 32)
    monkeypatch.delenv("WSBOT_OWN_IPHONE", raising=False)
    monkeypatch.setattr(iphone, "IPhone", OwnPhone)
    monkeypatch.setattr(ios_device, "WAIT_RETRY_S", 0.01)
    ServicePhone.errors = []
    OwnPhone.made = []
    return client


def device_with(phone) -> ios_device.IOSGameDevice:
    dev = object.__new__(ios_device.IOSGameDevice)
    dev.phone = phone
    return dev


def test_uses_the_service_when_automationhq_runs_it(service):
    assert isinstance(ios_device._new_phone("U"), ServicePhone)


def test_own_session_without_automationhq(service, monkeypatch):
    monkeypatch.delenv("AHQ_IPHONE")
    assert isinstance(ios_device._new_phone("U"), OwnPhone)


def test_own_session_when_asked(service, monkeypatch):
    monkeypatch.setenv("WSBOT_OWN_IPHONE", "1")
    assert isinstance(ios_device._new_phone("U"), OwnPhone)


def test_own_session_when_the_client_is_missing(service, monkeypatch):
    monkeypatch.setitem(sys.modules, "ahq_iphone.client", None)  # import fails
    assert isinstance(ios_device._new_phone("U"), OwnPhone)


def test_waits_while_the_service_has_no_phone(service):
    ServicePhone.errors = [ClientNoTunnel("unplugged"), ClientNoTunnel("unplugged")]
    phone = ServicePhone("U")
    dev = device_with(phone)
    dev._open_phone("U")
    assert dev.phone is phone and phone.opened == 1
    assert not OwnPhone.made


@pytest.mark.parametrize("error", [ClientUnsupported("pmd3 9"), ClientUnreachable("refused")])
def test_falls_back_to_its_own_session(service, error):
    ServicePhone.errors = [error]
    phone = ServicePhone("U")
    dev = device_with(phone)
    dev._open_phone("U")
    assert phone.closed == 1
    assert isinstance(dev.phone, OwnPhone) and dev.phone.opened == 1
    assert dev.phone.cancel is ios_device._stop_waiting


def test_other_service_errors_are_not_swallowed(service):
    ServicePhone.errors = [ClientError("no frames after 45 s")]
    with pytest.raises(ClientError):
        device_with(ServicePhone("U"))._open_phone("U")
    assert not OwnPhone.made
