"""Four-corner calibration picker.

The user clicks four corners on *this* window (our own dim layer). Nothing is
clicked in the game: we only read where the user clicked.
"""
from __future__ import annotations

from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QWidget

PROMPTS = [
    "1/4  Click the TOP-LEFT corner of the 8x8 playing area",
    "2/4  Click the TOP-RIGHT corner",
    "3/4  Click the BOTTOM-RIGHT corner",
    "4/4  Click the BOTTOM-LEFT corner",
]


class CalibrationPicker(QWidget):
    finished = pyqtSignal(list)  # 4 x (x, y) in monitor-local physical pixels
    cancelled = pyqtSignal()

    def __init__(self, screen):
        super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint |
                         Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.dpr = float(screen.devicePixelRatio())
        self.setGeometry(screen.geometry())
        self._pts: list[QPointF] = []
        self._cursor = QPointF()
        self.setMouseTracking(True)

    def mouseMoveEvent(self, e):
        self._cursor = e.position()
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.RightButton and self._pts:
            self._pts.pop()
        elif e.button() == Qt.MouseButton.LeftButton:
            self._pts.append(e.position())
            if len(self._pts) == 4:
                out = [(pt.x() * self.dpr, pt.y() * self.dpr) for pt in self._pts]
                self.close()
                self.finished.emit(out)
                return
        self.update()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape:
            self.close()
            self.cancelled.emit()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(0, 0, 0, 70))
        pen = QPen(QColor(0, 255, 140), 2)
        p.setPen(pen)
        if len(self._pts) > 1:
            p.drawPolyline(QPolygonF(self._pts))
        if len(self._pts) == 3:
            p.drawLine(self._pts[2], self._pts[0])
        for i, pt in enumerate(self._pts):
            p.setBrush(QColor(0, 255, 140))
            p.drawEllipse(pt, 5, 5)
            p.drawText(QPointF(pt.x() + 8, pt.y() - 8), str(i + 1))
        f = QFont("Segoe UI", 16)
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor(255, 255, 255))
        msg = PROMPTS[min(len(self._pts), 3)] + "     (right-click = undo, Esc = cancel)"
        p.fillRect(20, 20, self.width() - 40, 44, QColor(0, 0, 0, 190))
        p.drawText(32, 50, msg)
