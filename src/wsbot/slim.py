"""Keep a farm bot small: one runs per phone on one computer.

pymobiledevice3 imports questionary (and with it prompt_toolkit: ~150 modules, ~12 MB)
when it loads, only for interactive prompts a headless bot never shows. A stand-in
module takes its place and loads the real one the moment anything uses it.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types


class _Deferred(types.ModuleType):
    def __getattr__(self, attr: str):
        if attr == "Question" and "_real" not in self.__dict__:
            # pymobiledevice3.utils names it in a type annotation at import time
            return type("Question", (), {})
        real = self.__dict__.get("_real")
        if real is None:
            del sys.modules[self.__name__]
            real = importlib.import_module(self.__name__)
            self.__dict__["_real"] = real
        return getattr(real, attr)


def defer(name: str) -> None:
    if name in sys.modules or importlib.util.find_spec(name) is None:
        return
    sys.modules[name] = _Deferred(name)


def slim() -> None:
    defer("questionary")
