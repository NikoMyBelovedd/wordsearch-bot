"""The compiled copy's self-check: Ed25519 (RFC 8032 test vectors) and the license rules."""

from __future__ import annotations

import base64
import hashlib
import json

import pytest

from wsbot import protect

# RFC 8032, 7.1 TEST 2.
PUB = bytes.fromhex("3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c")
MSG = bytes.fromhex("72")
SIG = bytes.fromhex(
    "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da"
    "085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00"
)


def test_rfc8032_vector_verifies_and_tampering_fails():
    assert protect.verify(PUB, MSG, SIG)
    assert not protect.verify(PUB, b"\x73", SIG)
    assert not protect.verify(PUB, MSG, SIG[:-1] + b"\x01")
    assert not protect.verify(PUB, MSG, b"short")


def test_source_runs_skip_the_check(tmp_path):
    assert "__compiled__" not in vars(protect)
    assert protect.check(tmp_path) is None


@pytest.fixture
def compiled(monkeypatch):
    monkeypatch.setitem(vars(protect), "__compiled__", True)


def test_a_compiled_copy_needs_a_license(tmp_path, compiled):
    assert "missing" in protect.check(tmp_path)


def test_a_license_signed_by_someone_else_is_refused(tmp_path, compiled):
    payload = json.dumps({"script": "wordsearch-bot", "platform": protect._platform(), "files": {}})
    (tmp_path / "license.json").write_text(
        json.dumps({"payload": payload, "sig": base64.b64encode(SIG).decode()})
    )
    assert "isn't signed" in protect.check(tmp_path)


def test_a_changed_code_file_is_refused(tmp_path, compiled, monkeypatch):
    code = tmp_path / "src" / "wsbot.so"
    code.parent.mkdir()
    code.write_bytes(b"compiled")
    payload = json.dumps(
        {
            "script": "wordsearch-bot",
            "platform": protect._platform(),
            "files": {"src/wsbot.so": hashlib.sha256(b"compiled").hexdigest()},
        }
    )
    (tmp_path / "license.json").write_text(json.dumps({"payload": payload, "sig": "AA=="}))
    monkeypatch.setattr(protect, "verify", lambda *_: True)  # signature checked above
    assert protect.check(tmp_path) is None
    code.write_bytes(b"patched")
    assert "changed" in protect.check(tmp_path)
