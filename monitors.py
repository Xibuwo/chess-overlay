"""Monitor helpers: mss monitors (physical px) <-> Qt screens (logical px)."""
from __future__ import annotations

import mss


def list_monitors() -> list[dict]:
    with mss.mss() as sct:
        return [dict(m) for m in sct.monitors[1:]]


def qt_screen_for(mon: dict):
    from PyQt6.QtGui import QGuiApplication

    best, best_d = None, 1e18
    for s in QGuiApplication.screens():
        g, r = s.geometry(), s.devicePixelRatio()
        d = abs(g.width() * r - mon["width"]) + abs(g.height() * r - mon["height"])
        d += 0.001 * (abs(g.x() * r - mon["left"]) + abs(g.y() * r - mon["top"]))
        if d < best_d:
            best, best_d = s, d
    return best or QGuiApplication.primaryScreen()
