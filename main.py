"""Chess overlay launcher (Windows 11).

Run:  python main.py
"""
from __future__ import annotations

import logging
import sys

from PyQt6.QtCore import QObject, QPoint, Qt, QTimer
from PyQt6.QtGui import QAction, QColor, QFont, QGuiApplication, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from calibration import CalibrationPicker
from gui import SettingsWindow
from hotkeys import HotkeyManager
from logsetup import STATUS_LOGGER, get_logger, setup_logging
from monitors import list_monitors, qt_screen_for
from overlay import OverlayWindow
from state import AppState
from supervision import RestartPolicy
from worker import PipelineWorker

log = get_logger("main")


class Supervisor(QObject):
    """Restarts the PipelineWorker if its thread ends without anyone asking it to (the same idea as
    EngineManager.ensure(): notice the dead helper, relaunch it, carry on). Frame errors are already
    absorbed inside the worker loop; this is the safety net for anything that still kills the thread.
    A crash loop (RestartPolicy: too many restarts in a short time) is reported and left stopped."""

    def __init__(self, state, on_result, on_status, interval_ms: int = 2000, policy: RestartPolicy | None = None):
        super().__init__()
        self.state, self._on_result, self._on_status = state, on_result, on_status
        self.policy = policy or RestartPolicy()
        self.worker: PipelineWorker | None = None
        self._given_up = False
        self._closing = False
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self.check)

    def start(self):
        self._spawn()
        self._timer.start()

    def _spawn(self):
        w = PipelineWorker(self.state)
        w.result_ready.connect(self._on_result)
        w.status.connect(self._on_status)
        self.worker = w
        w.start()

    def check(self):
        w = self.worker
        if self._closing or w is None or self._given_up:
            return
        if w.isRunning():
            self.policy.note_healthy()
            return
        if w.stop_requested:
            return
        ok, why = self.policy.allow()
        if not ok:
            if why == "waiting":
                return
            self._given_up = True
            msg = f"Pipeline stopped for good ({why}). See the log file; restart the app."
            log.error(msg)
            logging.getLogger(STATUS_LOGGER).error(msg)
            return
        self.policy.record_restart()
        log.error("pipeline thread ended unexpectedly - restarting it")
        logging.getLogger(STATUS_LOGGER).warning("Pipeline restarted after a crash.")
        w.deleteLater()
        self._spawn()

    def stop(self):
        self._closing = True
        self._timer.stop()
        if self.worker is not None:
            self.worker.stop()


def _safe_pos(pos) -> tuple[int, int]:
    """Saved menu position, or the default if that spot is no longer on any screen."""
    try:
        x, y = int(pos[0]), int(pos[1])
        if QGuiApplication.screenAt(QPoint(x, y)) is not None:
            return x, y
    except Exception:
        pass
    return 60, 60


def _tray_icon() -> QIcon:
    pm = QPixmap(32, 32)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(59, 110, 245))
    p.drawEllipse(2, 2, 28, 28)
    f = QFont("Segoe UI", 14)
    f.setBold(True)
    p.setFont(f)
    p.setPen(QColor(255, 255, 255))
    p.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, "C")
    p.end()
    return QIcon(pm)


def main() -> int:
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)  # menu can be hidden; quit from its X button or the tray
    state = AppState()
    setup_logging(state.settings.log_level)
    log.info("Chess Overlay starting")
    hotkeys = HotkeyManager(state)

    mons = list_monitors()

    def monitor(idx: int) -> dict:
        return mons[min(max(idx, 1), len(mons)) - 1]

    overlay = OverlayWindow(state, qt_screen_for(monitor(state.settings.monitor_index)))
    overlay.show()
    state.listeners.append(overlay.update)

    gui = SettingsWindow(state, hotkeys)
    gui.move(*_safe_pos(state.settings.menu_pos))
    if not state.settings.start_menu_hidden:
        gui.show()

    def on_hotkey(action: str):
        if action == "menu_key":
            gui.toggle_menu()
        elif action == "overlay_key":
            gui.toggle_overlay()
        elif action == "reset_key":
            state.request("reset")
            gui.set_status("Game tracking reset.")

    hotkeys.triggered.connect(on_hotkey)

    tray = None
    if QSystemTrayIcon.isSystemTrayAvailable():
        tray = QSystemTrayIcon(_tray_icon(), app)
        tray.setToolTip("Chess Overlay")
        menu = QMenu()
        a_menu = QAction("Open / close menu", menu)
        a_menu.triggered.connect(gui.toggle_menu)
        a_over = QAction("Show / hide overlay", menu)
        a_over.triggered.connect(gui.toggle_overlay)
        a_quit = QAction("Quit", menu)
        a_quit.triggered.connect(app.quit)
        for a in (a_menu, a_over, a_quit):
            menu.addAction(a)
        tray.setContextMenu(menu)
        tray.activated.connect(lambda reason: gui.toggle_menu()
                               if reason == QSystemTrayIcon.ActivationReason.Trigger else None)
        tray._menu_ref = menu  # keep alive
        tray.setVisible(state.settings.show_tray)
        gui.tray_changed.connect(tray.setVisible)

    supervisor = Supervisor(state, overlay.set_result, gui.set_status)

    def on_monitor(idx: int):
        overlay.move_to_screen(qt_screen_for(monitor(idx)))
        overlay.refresh_native()

    gui.monitor_changed.connect(on_monitor)

    picker_ref = {}

    def start_calibration():
        idx = state.settings.monitor_index
        screen = qt_screen_for(monitor(idx))
        gui.hide()
        gui.locked = True  # hotkey must not reopen the menu mid-calibration
        picker = CalibrationPicker(screen)
        picker_ref["p"] = picker

        def done(corners):
            state.set_corners(corners, idx)
            gui.locked = False
            gui.show()
            gui.set_status("Calibrated. Put the game in its starting position and press 'Learn pieces'.")

        def cancelled():
            gui.locked = False
            gui.show()

        picker.finished.connect(done)
        picker.cancelled.connect(cancelled)
        picker.show()
        picker.activateWindow()
        picker.setFocus()

    gui.calibrate_requested.connect(start_calibration)

    supervisor.start()
    app.aboutToQuit.connect(supervisor.stop)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
