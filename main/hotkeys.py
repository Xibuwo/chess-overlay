"""Global hotkeys by *polling* key state (GetAsyncKeyState).

Nothing is hooked or registered system-wide, so the key keeps working normally in
other programs; we only read whether it is currently down. Combos like
"CTRL+SHIFT+INSERT" are supported. Modifiers that are not part of the binding must
be up, so plain INSERT does not fire on Ctrl+Insert.
"""
from __future__ import annotations

import sys
from typing import Callable, Optional

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

IS_WINDOWS = sys.platform == "win32"
if IS_WINDOWS:
    import ctypes

    _gaks = ctypes.WinDLL("user32", use_last_error=True).GetAsyncKeyState
    _gaks.argtypes = [ctypes.c_int]
    _gaks.restype = ctypes.c_short

ACTIONS = ("menu_key", "overlay_key", "reset_key")
ACTION_LABELS = {"menu_key": "Open / close menu", "overlay_key": "Show / hide overlay",
                 "reset_key": "Reset game tracking"}

MODS = {"CTRL": 0x11, "SHIFT": 0x10, "ALT": 0x12}
_MOD_ORDER = ("CTRL", "SHIFT", "ALT")


def _build() -> dict[str, int]:
    d = {"INSERT": 0x2D, "DELETE": 0x2E, "HOME": 0x24, "END": 0x23, "PAGEUP": 0x21, "PAGEDOWN": 0x22,
         "UP": 0x26, "DOWN": 0x28, "LEFT": 0x25, "RIGHT": 0x27, "TAB": 0x09, "SPACE": 0x20,
         "PAUSE": 0x13, "SCROLLLOCK": 0x91, "BACKTICK": 0xC0}
    for i in range(24):
        d[f"F{i + 1}"] = 0x70 + i
    for i in range(10):
        d[str(i)] = 0x30 + i
        d[f"NUM{i}"] = 0x60 + i
    for i in range(26):
        d[chr(65 + i)] = 0x41 + i
    return d


VK = _build()
VK_REV = {v: k for k, v in VK.items()}


def parse(combo: str) -> Optional[tuple[tuple[str, ...], int]]:
    """'ctrl+insert' -> (('CTRL',), 0x2D); None if empty/invalid."""
    parts = [p.strip().upper() for p in (combo or "").split("+") if p.strip()]
    if not parts or parts[-1] not in VK:
        return None
    mods = tuple(m for m in parts[:-1] if m in MODS)
    if len(mods) != len(parts) - 1:
        return None
    return mods, VK[parts[-1]]


def format_combo(mods, key_name: str) -> str:
    return "+".join([m for m in _MOD_ORDER if m in mods] + [key_name])


def _down(vk: int) -> bool:
    return IS_WINDOWS and bool(_gaks(vk) & 0x8000)


def combo_down(combo: str) -> bool:
    p = parse(combo)
    if p is None or not _down(p[1]):
        return False
    mods, _ = p
    return all(_down(MODS[m]) == (m in mods) for m in MODS)


class HotkeyManager(QObject):
    triggered = pyqtSignal(str)  # one of ACTIONS

    def __init__(self, state, interval_ms: int = 25):
        super().__init__()
        self.state = state
        self._prev = {a: False for a in ACTIONS}
        self._capture_cb: Optional[Callable] = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(interval_ms)

    # ---- rebinding ---------------------------------------------------------
    def capture(self, cb: Callable[[Optional[str]], None]) -> None:
        """Next key press (with held modifiers) is passed to cb as e.g. 'CTRL+F8'.
        cb('') = cleared (Backspace), cb(None) = cancelled (Esc)."""
        self._capture_cb = cb

    def _scan_capture(self) -> None:
        if _down(0x1B):
            cb, self._capture_cb = self._capture_cb, None
            cb(None)
            return
        if _down(0x08):
            cb, self._capture_cb = self._capture_cb, None
            cb("")
            return
        for vk, name in VK_REV.items():
            if _down(vk):
                mods = [m for m in MODS if _down(MODS[m])]
                cb, self._capture_cb = self._capture_cb, None
                cb(format_combo(mods, name))
                return

    # ---- polling -------------------------------------------------------------
    def _tick(self) -> None:
        if not IS_WINDOWS:
            return
        s = self.state.settings
        now = {a: bool(getattr(s, a, "")) and combo_down(getattr(s, a)) for a in ACTIONS}
        if self._capture_cb is not None:
            self._scan_capture()
        elif s.hotkeys_enabled:
            for a in ACTIONS:
                if now[a] and not self._prev[a]:
                    self.triggered.emit(a)
        self._prev = now
