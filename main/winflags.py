"""Native Win32 window flags used by the overlay.

We only change window *styles* of our own windows. Nothing here touches another
process: no injection, no hooks, no synthetic input.
"""
from __future__ import annotations

import sys

IS_WINDOWS = sys.platform == "win32"

GWL_EXSTYLE = -20
WS_EX_TOPMOST = 0x00000008
WS_EX_TRANSPARENT = 0x00000020  # mouse events fall through to the window below
WS_EX_TOOLWINDOW = 0x00000080  # no taskbar / Alt-Tab entry
WS_EX_LAYERED = 0x00080000  # required for per-pixel alpha + click-through
WS_EX_NOACTIVATE = 0x08000000  # never takes focus from the game

WDA_NONE = 0x00
WDA_EXCLUDEFROMCAPTURE = 0x11  # Windows 10 2004+ / Windows 11

HWND_TOPMOST = -1
SWP_NOSIZE, SWP_NOMOVE, SWP_NOACTIVATE = 0x0001, 0x0002, 0x0010

if IS_WINDOWS:
    import ctypes
    from ctypes import wintypes

    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _get = _user32.GetWindowLongPtrW
    _get.argtypes = [wintypes.HWND, ctypes.c_int]
    _get.restype = ctypes.c_ssize_t
    _set = _user32.SetWindowLongPtrW
    _set.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
    _set.restype = ctypes.c_ssize_t
    _pos = _user32.SetWindowPos
    _pos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                     ctypes.c_int, ctypes.c_int, ctypes.c_uint]
    _pos.restype = wintypes.BOOL
    _aff = _user32.SetWindowDisplayAffinity
    _aff.argtypes = [wintypes.HWND, wintypes.DWORD]
    _aff.restype = wintypes.BOOL


def make_click_through(hwnd: int) -> None:
    """WS_EX_LAYERED | WS_EX_TRANSPARENT (+ no-activate, tool window)."""
    if not IS_WINDOWS:
        return
    style = _get(hwnd, GWL_EXSTYLE)
    style |= WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW
    _set(hwnd, GWL_EXSTYLE, style)


def keep_topmost(hwnd: int) -> None:
    if not IS_WINDOWS:
        return
    _pos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE)


def exclude_from_capture(hwnd: int, enable: bool = True) -> bool:
    """Hide our own window from BitBlt/DXGI screen capture (so mss/bettercam
    never see our arrows)."""
    if not IS_WINDOWS:
        return False
    return bool(_aff(hwnd, WDA_EXCLUDEFROMCAPTURE if enable else WDA_NONE))
