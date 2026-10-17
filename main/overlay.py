"""Transparent, click-through, always-on-top overlay that only *draws*.

Native flags WS_EX_LAYERED | WS_EX_TRANSPARENT (plus NOACTIVATE / TOOLWINDOW) are
set on our own window. The window never receives input and never sends any.
"""
from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QWidget

import winflags
from models import FrameResult
from state import AppState

GREEN = QColor(46, 204, 64, 200)
BLUE = QColor(52, 152, 255, 200)
RED = QColor(255, 59, 48, 200)
ORANGE_TINT = QColor(255, 140, 0, 95)


class OverlayWindow(QWidget):
    def __init__(self, state: AppState, screen):
        flags = (Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint |
                 Qt.WindowType.Tool | Qt.WindowType.WindowTransparentForInput |
                 Qt.WindowType.WindowDoesNotAcceptFocus)
        super().__init__(None, flags)
        self.state = state
        self._result: FrameResult | None = None
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.move_to_screen(screen)
        self._topmost_timer = QTimer(self)
        self._topmost_timer.timeout.connect(self._reassert)
        self._topmost_timer.start(1500)

    # ---- placement / native flags ----------------------------------------
    def move_to_screen(self, screen):
        self._screen = screen
        self.dpr = float(screen.devicePixelRatio())
        g = screen.geometry()
        # 1px short of exact fullscreen: avoids DWM treating us as a fullscreen window
        self.setGeometry(g.x(), g.y(), g.width(), g.height() - 1)

    def showEvent(self, e):
        super().showEvent(e)
        self._apply_native()

    def _apply_native(self):
        hwnd = int(self.winId())
        winflags.make_click_through(hwnd)
        winflags.exclude_from_capture(hwnd, self.state.settings.exclude_from_capture)
        winflags.keep_topmost(hwnd)

    def _reassert(self):
        if self.isVisible():
            winflags.keep_topmost(int(self.winId()))

    def refresh_native(self):
        if self.isVisible():
            self._apply_native()

    # ---- data -------------------------------------------------------------
    def set_result(self, result: FrameResult):
        self._result = result
        self.update()

    # ---- painting helpers -------------------------------------------------
    def _p(self, xy) -> QPointF:
        return QPointF(float(xy[0]) / self.dpr, float(xy[1]) / self.dpr)

    def _poly(self, pts) -> QPolygonF:
        return QPolygonF([self._p(p) for p in pts])

    @staticmethod
    def _arrow(a: QPointF, b: QPointF, width: float) -> QPolygonF:
        dx, dy = b.x() - a.x(), b.y() - a.y()
        length = math.hypot(dx, dy)
        if length < 1:
            return QPolygonF()
        ux, uy = dx / length, dy / length
        nx, ny = -uy, ux
        head_len = min(width * 2.6, length * 0.6)
        head_w = width * 2.0
        # start slightly off the square centre, end slightly before the target centre
        a2 = QPointF(a.x() + ux * width * 0.6, a.y() + uy * width * 0.6)
        tip = QPointF(b.x() - ux * width * 0.4, b.y() - uy * width * 0.4)
        base = QPointF(tip.x() - ux * head_len, tip.y() - uy * head_len)
        hw = width / 2
        return QPolygonF([
            QPointF(a2.x() + nx * hw, a2.y() + ny * hw),
            QPointF(base.x() + nx * hw, base.y() + ny * hw),
            QPointF(base.x() + nx * head_w, base.y() + ny * head_w),
            tip,
            QPointF(base.x() - nx * head_w, base.y() - ny * head_w),
            QPointF(base.x() - nx * hw, base.y() - ny * hw),
            QPointF(a2.x() - nx * hw, a2.y() - ny * hw),
        ])

    @staticmethod
    def _sq_size(poly: np.ndarray, dpr: float) -> float:
        edges = [np.linalg.norm(poly[i] - poly[(i + 1) % 4]) for i in range(4)]
        return float(np.mean(edges)) / dpr

    def _draw_arrow(self, p: QPainter, r: FrameResult, move, color: QColor):
        g, fl = r.geometry, r.flipped
        a = self._p(g.square_center(move.from_square, fl))
        b = self._p(g.square_center(move.to_square, fl))
        sq = self._sq_size(g.square_polygon(move.from_square, fl), self.dpr)
        poly = self._arrow(a, b, max(4.0, sq * 0.17))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(color)
        p.drawPolygon(poly)

    def _draw_tag(self, p: QPainter, anchor: QPointF, emoji: str, text: str, size: float, center=False):
        ef = QFont("Segoe UI Emoji")
        ef.setPixelSize(max(10, int(size * 0.42)))
        p.setFont(ef)
        fm = QFontMetricsF(ef)
        ew = fm.horizontalAdvance(emoji)
        x = anchor.x() - (ew / 2 if center else 0)
        p.setPen(QPen(QColor(255, 255, 255, 255)))
        p.drawText(QPointF(x, anchor.y() + fm.ascent()), emoji)
        if text:
            tf = QFont("Segoe UI")
            tf.setPixelSize(max(9, int(size * 0.2)))
            tf.setBold(True)
            p.setFont(tf)
            tm = QFontMetricsF(tf)
            tw, th = tm.horizontalAdvance(text) + 8, tm.height() + 2
            rect = QRectF(x, anchor.y() + fm.height(), tw, th)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(20, 20, 20, 170))
            p.drawRoundedRect(rect, 4, 4)
            p.setPen(QColor(255, 255, 255, 255))
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    # ---- paint ------------------------------------------------------------
    def paintEvent(self, _):
        s = self.state.settings
        r = self._result
        if not s.overlay_visible or r is None or r.board is None or r.geometry is None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        p.setOpacity(float(s.overlay_opacity))
        g, fl = r.geometry, r.flipped
        t = r.tactics

        # 1) contested squares: orange tint
        if s.show_contested and t:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(ORANGE_TINT)
            for c in t.contested:
                p.drawPolygon(self._poly(g.square_polygon(c.square, fl)))

        # 2) arrows
        if r.my_turn:
            if s.show_second and r.second:
                self._draw_arrow(p, r, r.second, BLUE)
            if s.show_best and r.best:
                self._draw_arrow(p, r, r.best, GREEN)
        if s.show_threat and r.threat:
            self._draw_arrow(p, r, r.threat, RED)

        # 3) fire markers + attacker/defender text
        if s.show_contested and t:
            for c in t.contested:
                poly = g.square_polygon(c.square, fl)
                sq = self._sq_size(poly, self.dpr)
                tl = self._p(poly[0])
                sign = "+" if c.diff > 0 else ""
                self._draw_tag(p, QPointF(tl.x() + 2, tl.y() + 1), "\U0001F525",
                               f"A{c.attackers} D{c.defenders} ({sign}{c.diff})", sq)

        # 4) hung / vulnerable droplets
        if s.show_hung and t:
            for w in t.weak:
                poly = g.square_polygon(w.square, fl)
                sq = self._sq_size(poly, self.dpr)
                tr = self._p(poly[1])
                emoji = "\U0001FA78" if w.kind == "hung" else "\U0001F4A7"  # 🩸 hung, 💧 vulnerable
                self._draw_tag(p, QPointF(tr.x() - sq * 0.42, tr.y() - sq * 0.15), emoji, f"-{w.net}", sq)
        p.end()
