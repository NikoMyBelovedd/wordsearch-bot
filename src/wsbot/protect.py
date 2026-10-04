"""A compiled copy checks itself before it plays.

AutomationHQ ships this bot as compiled code with a license next to it
(`license.json`, signed by AutomationHQ's server): which script, version and
platform it is, the SHA-256 of each code file, and a tag for the account that
downloaded it. A compiled copy without a valid license, or with a code file
that changed, refuses to start. Running from source (development) skips this.

Ed25519 verification is done here (RFC 8032), so it needs no extra package.
"""

from __future__ import annotations

import base64
import hashlib
import json
import sys
from pathlib import Path

# AutomationHQ's license-signing key (raw Ed25519, base64).
PUBLIC_KEY = "dFi3yz0xeKLpoAqpCScpeHSHaSVaPMuYSNjRYbMXqo4="
SCRIPT = "wordsearch-bot"
EXIT_UNLICENSED = 5

_P = 2**255 - 19
_L = 2**252 + 27742317777372353535851937790883648493
_D = -121665 * pow(121666, _P - 2, _P) % _P
_I = pow(2, (_P - 1) // 4, _P)


def _xrecover(y: int) -> int:
    xx = (y * y - 1) * pow(_D * y * y + 1, _P - 2, _P)
    x = pow(xx, (_P + 3) // 8, _P)
    if (x * x - xx) % _P != 0:
        x = x * _I % _P
    if x % 2 != 0:
        x = _P - x
    return x


_BY = 4 * pow(5, _P - 2, _P) % _P
_B = (_xrecover(_BY), _BY, 1, _xrecover(_BY) * _BY % _P)


def _add(p: tuple, q: tuple) -> tuple:
    x1, y1, z1, t1 = p
    x2, y2, z2, t2 = q
    a = (y1 - x1) * (y2 - x2) % _P
    b = (y1 + x1) * (y2 + x2) % _P
    c = t1 * 2 * _D * t2 % _P
    d = z1 * 2 * z2 % _P
    e, f, g, h = b - a, d - c, d + c, b + a
    return (e * f % _P, g * h % _P, f * g % _P, e * h % _P)


def _mul(s: int, p: tuple) -> tuple:
    q = (0, 1, 1, 0)
    while s > 0:
        if s & 1:
            q = _add(q, p)
        p = _add(p, p)
        s >>= 1
    return q


def _equal(p: tuple, q: tuple) -> bool:
    return (p[0] * q[2] - q[0] * p[2]) % _P == 0 and (p[1] * q[2] - q[1] * p[2]) % _P == 0


def _decode(b: bytes) -> tuple | None:
    if len(b) != 32:
        return None
    y = int.from_bytes(b, "little") & ((1 << 255) - 1)
    if y >= _P:
        return None
    x = _xrecover(y)
    if (x & 1) != (b[31] >> 7):
        x = _P - x
    point = (x, y, 1, x * y % _P)
    # On the curve?
    if (-x * x + y * y - 1 - _D * x * x * y * y) % _P != 0:
        return None
    return point


def verify(public: bytes, message: bytes, signature: bytes) -> bool:
    """Ed25519 signature check (RFC 8032, verification only)."""
    if len(signature) != 64:
        return False
    a = _decode(public)
    r = _decode(signature[:32])
    if a is None or r is None:
        return False
    s = int.from_bytes(signature[32:], "little")
    if s >= _L:
        return False
    h = int.from_bytes(hashlib.sha512(signature[:32] + public + message).digest(), "little") % _L
    return _equal(_mul(s, _B), _add(r, _mul(h, a)))


def _platform() -> str:
    import platform as p

    machine = p.machine().lower()
    if sys.platform == "win32":
        return "windows-x86_64"
    if sys.platform == "darwin":
        return "macos-arm64" if machine in ("arm64", "aarch64") else "macos-x86_64"
    return "linux-x86_64"


def check(root: Path) -> str | None:
    """Why this copy may not run, or None if it may (or runs from source)."""
    if "__compiled__" not in globals():
        return None  # running from source: a developer's copy
    try:
        license = json.loads((root / "license.json").read_text(encoding="utf-8"))
        payload, sig = license["payload"], base64.b64decode(license["sig"])
    except (OSError, ValueError, KeyError, TypeError):
        return "its license is missing or unreadable"
    if not verify(base64.b64decode(PUBLIC_KEY), payload.encode("utf-8"), sig):
        return "its license isn't signed by AutomationHQ"
    try:
        terms = json.loads(payload)
    except ValueError:
        return "its license is unreadable"
    if terms.get("script") != SCRIPT or terms.get("platform") != _platform():
        return "its license is for another script or computer"
    files = terms.get("files") or {}
    if not files:
        return "its license lists no code"
    for rel, want in files.items():
        path = (root / rel).resolve()
        if root.resolve() not in path.parents:
            return "its license lists an odd file"
        try:
            got = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            return f"{rel} is missing"
        if got != want:
            return f"{rel} was changed"
    return None


def require(root: Path) -> None:
    """Stops here, with one line saying why, when this copy may not run."""
    why = check(root)
    if why:
        print(f"This copy of the bot can't run: {why}. Reinstall it from AutomationHQ.")
        raise SystemExit(EXIT_UNLICENSED)
