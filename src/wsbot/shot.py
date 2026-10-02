"""One screen frame, resized only when something asks for a size.

A phone's frame arrives at its own size (750x1334 on an iPhone SE). The bot reads
letters and popups in calibration space (1296x2305 there). Most frames only need the
board panel checked, which a cheap half-native copy does; the calibration copy (and
the half-size one popups are matched on) is made only for the frames that need it,
once each, however many threads ask.
"""

from __future__ import annotations

import cv2
import numpy as np

from .board import Board, find_panel, read_board

SCALE = 0.5  # the watcher's working size (popups, panel)


class Shot:
    __slots__ = (
        "_board",
        "_calib",
        "_coarse",
        "_coarse_color",
        "_mini",
        "_panel",
        "_small",
        "calib_size",
        "native",
        "seq",
    )

    def __init__(self, seq: int, native: np.ndarray, calib_size: tuple[int, int]) -> None:
        self.seq = seq  # the device's frame number: same number = same picture
        self.native = native
        self.calib_size = calib_size
        self._calib: np.ndarray | None = None
        self._small: np.ndarray | None = None
        self._coarse: np.ndarray | None = None
        self._coarse_color: np.ndarray | None = None
        self._mini: np.ndarray | None = None
        self._panel: tuple[int, int, int, int] | bool | None = False  # False = not looked
        self._board: Board | bool | None = False

    # Two threads may ask at once; both compute the same thing and one copy wins.
    @property
    def calib(self) -> np.ndarray:
        if self._calib is None:
            w, h = self.calib_size
            img = self.native
            if img.shape[1] != w or img.shape[0] != h:
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            self._calib = img
        return self._calib

    @property
    def small(self) -> np.ndarray:
        """The calibration frame at SCALE. Made from `calib` like before: upscale plus
        an exact 2x shrink is faster than one odd-ratio shrink, and scores don't move."""
        if self._small is None:
            self._small = cv2.resize(
                self.calib, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA
            )
        return self._small

    @property
    def coarse(self) -> np.ndarray:
        """`small` halved again, in grayscale: the popup matcher's first look."""
        if self._coarse is None:
            self._coarse = cv2.cvtColor(self.coarse_color, cv2.COLOR_BGR2GRAY)
        return self._coarse

    @property
    def coarse_color(self) -> np.ndarray:
        """`small` halved again, in color (for the low-contrast templates)."""
        if self._coarse_color is None:
            self._coarse_color = cv2.resize(
                self.small, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA
            )
        return self._coarse_color

    @property
    def mini(self) -> np.ndarray:
        """The native frame halved: an exact 2x shrink is ~20x cheaper than `small`'s
        odd ratio, and plenty to find the board panel (a few calibration pixels off)."""
        if self._mini is None:
            h, w = self.native.shape[:2]
            self._mini = cv2.resize(self.native, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        return self._mini

    @property
    def mini_scale(self) -> float:
        return self.mini.shape[1] / self.calib_size[0]

    @property
    def panel(self) -> tuple[int, int, int, int] | None:
        """The white board panel (calibration pixels), found on `mini`."""
        if self._panel is False:
            self._panel = find_panel(self.mini, self.mini_scale)
        return self._panel  # type: ignore[return-value]

    @property
    def board(self) -> Board | None:
        """The letter board (read_board), read once per picture: the watcher and the
        bot's main thread both ask for the same frames."""
        if self._board is False:
            panel = self.panel
            self._board = None if panel is None else read_board(self.calib, near=panel)
        return self._board  # type: ignore[return-value]

    def same_picture(self, other: Shot | None) -> bool:
        """True if `other` shows exactly this picture (a static screen)."""
        if other is None:
            return False
        if other.seq == self.seq and other.native is self.native:
            return True
        # `mini`: the cheap copy the panel check makes anyway (`coarse` needs the big
        # calibration copy, which most frames never otherwise get)
        a, b = self.mini, other.mini
        return a.shape == b.shape and np.array_equal(a, b)
