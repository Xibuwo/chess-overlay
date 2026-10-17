"""Draggable, always-on-top settings window."""
from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QGuiApplication, QPainter, QPen
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QFrame, QGridLayout,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, QSizeGrip, QSlider,
                             QSpinBox, QVBoxLayout, QWidget)

import strength
import winflags
from config import (ENGINE_KINDS, KIND_LC0, KIND_MAIA3, KIND_MAIA_LC0, KIND_STOCKFISH, SF_MODES)
from monitors import list_monitors
from hotkeys import ACTION_LABELS, ACTIONS
from state import AppState
from theme import Theme, ThemeWindow, ThemedWidget, make_style

# settings that only change painting: no need to re-run the engine
QUIET = {"overlay_visible", "overlay_opacity", "show_contested", "show_hung", "show_best"}

BASE_W = 360                 # design width of the menu in px; everything scales relative to it
MIN_SCALE, MAX_SCALE = 0.85, 2.0


class ResizeGrip(QSizeGrip):
    """Bottom-right drag handle (the window is frameless, so it has no native resize border)."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setFixedSize(18, 18)
        self.setCursor(Qt.CursorShape.SizeFDiagCursor)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setPen(QPen(QColor(150, 156, 180), 1.4))
        w, h = self.width(), self.height()
        for i in (4, 8, 12):
            p.drawLine(w - 2, h - 2 - i, w - 2 - i, h - 2)
        p.end()


class TitleBar(QFrame):
    def __init__(self, text: str, on_close, on_min):
        super().__init__()
        self.setObjectName("title")
        self.setFixedHeight(32)  # the title bar never grows with the window (only with the UI scale)
        self._btns = []
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 4, 4, 4)
        t = QLabel(text)
        t.setObjectName("titleText")
        lay.addWidget(t, 1)
        for sym, cb in (("-", on_min), ("x", on_close)):
            b = QPushButton(sym)
            b.setObjectName("flat")
            b.setFixedSize(24, 22)
            b.clicked.connect(cb)
            lay.addWidget(b)
            self._btns.append(b)

    def set_scale(self, k: float):
        self.setFixedHeight(int(round(32 * k)))
        for b in self._btns:
            b.setFixedSize(int(round(24 * k)), int(round(22 * k)))

    def mousePressEvent(self, e):  # drag the whole window by its title bar
        if e.button() == Qt.MouseButton.LeftButton:
            handle = self.window().windowHandle()
            if handle is not None:
                handle.startSystemMove()


class SettingsWindow(ThemedWidget):
    calibrate_requested = pyqtSignal()
    monitor_changed = pyqtSignal(int)
    tray_changed = pyqtSignal(bool)

    def __init__(self, state: AppState, hotkeys=None):
        super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint |
                         Qt.WindowType.Tool)
        self.state = state
        self.hotkeys = hotkeys
        self.locked = False  # set while the calibration picker is open
        self._building = True
        s = state.settings
        # resizable: drag the grip in the bottom-right corner (or Ctrl + mouse wheel). The whole UI
        # scales with the width; the height always follows the content.
        self._scale = 1.0
        self.setMinimumWidth(int(BASE_W * MIN_SCALE))
        self.setMaximumWidth(int(BASE_W * MAX_SCALE))
        self.resize(int(max(BASE_W * MIN_SCALE, min(BASE_W * MAX_SCALE, getattr(s, "menu_width", BASE_W)))), 100)
        self.theme = Theme.from_dict(s.theme)
        self.setStyleSheet(make_style(self.theme, 1.0))
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)  # hotkey won't steal focus from the game
        self.setWindowOpacity(float(s.menu_opacity))
        self._pos_timer = QTimer(self)
        self._pos_timer.setSingleShot(True)
        self._pos_timer.timeout.connect(self._save_pos)
        self._scale_timer = QTimer(self)
        self._scale_timer.setSingleShot(True)
        self._scale_timer.timeout.connect(self._apply_scale)
        self._theme_timer = QTimer(self)
        self._theme_timer.setSingleShot(True)
        self._theme_timer.timeout.connect(lambda: self.theme_win.restyle(self.theme, self._scale))
        self._scaled_btns = []   # (button, design width) for fixed-width buttons

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.title_bar = TitleBar("Chess Overlay", QApplication.quit, self.hide)
        root.addWidget(self.title_bar)
        body = QWidget()
        root.addWidget(body)
        g = QGridLayout(body)
        g.setContentsMargins(12, 10, 12, 12)
        g.setVerticalSpacing(6)
        row = 0

        # ---- engine --------------------------------------------------------
        g.addWidget(QLabel("Engine"), row, 0)
        self.engine_box = QComboBox()
        self.engine_box.addItems(ENGINE_KINDS)
        self.engine_box.setCurrentText(s.engine_kind)
        self.engine_box.currentTextChanged.connect(self._on_engine)
        g.addWidget(self.engine_box, row, 1, 1, 2)
        row += 1

        self.sf_label = QLabel("Stockfish mode")
        self.sf_box = QComboBox()
        self.sf_box.addItems(SF_MODES)
        self.sf_box.setCurrentText(s.sf_mode)
        self.sf_box.currentTextChanged.connect(self._on_sf_mode)
        g.addWidget(self.sf_label, row, 0)
        g.addWidget(self.sf_box, row, 1, 1, 2)
        row += 1

        self.cmd_label = QLabel("Command")
        self.cmd_edit = QLineEdit()
        self.cmd_edit.editingFinished.connect(self._on_cmd)
        b1 = QPushButton("...")
        b1.setObjectName("flat")
        b1.setFixedWidth(30)
        self._scaled_btns.append((b1, 30))
        b1.clicked.connect(self._browse_cmd)
        g.addWidget(self.cmd_label, row, 0)
        g.addWidget(self.cmd_edit, row, 1)
        g.addWidget(b1, row, 2)
        row += 1

        self.w_label = QLabel("Weights")
        self.w_edit = QLineEdit()
        self.w_edit.editingFinished.connect(self._on_weights)
        self.w_btn = QPushButton("...")
        self.w_btn.setObjectName("flat")
        self.w_btn.setFixedWidth(30)
        self._scaled_btns.append((self.w_btn, 30))
        self.w_btn.clicked.connect(self._browse_weights)
        g.addWidget(self.w_label, row, 0)
        g.addWidget(self.w_edit, row, 1)
        g.addWidget(self.w_btn, row, 2)
        row += 1

        # ---- dynamic ELO / strength slider --------------------------------
        self.elo_title = QLabel("ELO / Strength")
        g.addWidget(self.elo_title, row, 0, 1, 3)
        row += 1
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.slider.sliderReleased.connect(self._commit_elo)
        g.addWidget(self.slider, row, 0, 1, 3)
        row += 1
        self.elo_desc = QLabel()
        self.elo_desc.setObjectName("hint")
        self.elo_desc.setWordWrap(True)
        g.addWidget(self.elo_desc, row, 0, 1, 3)
        row += 1

        # ---- Maia 3 opponent level ---------------------------------------
        self.chk_same = QCheckBox("Opponent ELO = my ELO")
        self.chk_same.setChecked(s.oppo_same_as_elo)
        self.chk_same.toggled.connect(self._on_same)
        g.addWidget(self.chk_same, row, 0, 1, 3)
        row += 1
        self.oppo_label = QLabel()
        self.oppo_label.setObjectName("hint")
        g.addWidget(self.oppo_label, row, 0, 1, 3)
        row += 1
        self.oppo_slider = QSlider(Qt.Orientation.Horizontal)
        self.oppo_slider.valueChanged.connect(self._on_oppo_slider)
        g.addWidget(self.oppo_slider, row, 0, 1, 3)
        row += 1

        # ---- search / resources -------------------------------------------
        self.nodes_label = QLabel("Nodes (0 = default)")
        self.nodes_spin = QSpinBox()
        self.nodes_spin.setRange(0, 100000)
        self.nodes_spin.setValue(int(s.engine_nodes))
        self.nodes_spin.valueChanged.connect(lambda v: self._set(engine_nodes=v))
        g.addWidget(self.nodes_label, row, 0)
        g.addWidget(self.nodes_spin, row, 1, 1, 2)
        row += 1
        self.mt_label = QLabel("Move time")
        self.mt_spin = QSpinBox()
        self.mt_spin.setRange(20, 10000)
        self.mt_spin.setSingleStep(50)
        self.mt_spin.setSuffix(" ms")
        self.mt_spin.setValue(int(s.movetime_ms))
        self.mt_spin.valueChanged.connect(lambda v: self._set(movetime_ms=v))
        g.addWidget(self.mt_label, row, 0)
        g.addWidget(self.mt_spin, row, 1, 1, 2)
        row += 1
        self.th_label = QLabel("Threads / Hash")
        self.th_spin = QSpinBox()
        self.th_spin.setRange(1, 64)
        self.th_spin.setValue(int(s.sf_threads))
        self.th_spin.valueChanged.connect(lambda v: self._set(sf_threads=v))
        self.hash_spin = QSpinBox()
        self.hash_spin.setRange(1, 16384)
        self.hash_spin.setSuffix(" MB")
        self.hash_spin.setValue(int(s.sf_hash_mb))
        self.hash_spin.valueChanged.connect(lambda v: self._set(sf_hash_mb=v))
        g.addWidget(self.th_label, row, 0)
        g.addWidget(self.th_spin, row, 1)
        g.addWidget(self.hash_spin, row, 2)
        row += 1
        g.addWidget(QLabel("Extra UCI"), row, 0)
        self.extra_edit = QLineEdit(s.extra_uci)
        self.extra_edit.setPlaceholderText("Name=Value; Name2=Value2")
        self.extra_edit.editingFinished.connect(lambda: self._set(extra_uci=self.extra_edit.text().strip()))
        g.addWidget(self.extra_edit, row, 1, 1, 2)
        row += 1

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        g.addWidget(line, row, 0, 1, 3)
        row += 1

        # ---- game ----------------------------------------------------------
        self.chk_black = QCheckBox("I play Black")
        self.chk_black.setChecked(s.play_as_black)
        self.chk_black.toggled.connect(lambda v: self.state.update(play_as_black=v))
        self.chk_flip = QCheckBox("Board flipped (Black at bottom)")
        self.chk_flip.setChecked(s.board_flipped)
        self.chk_flip.toggled.connect(lambda v: self.state.update(board_flipped=v))
        g.addWidget(self.chk_black, row, 0, 1, 3)
        row += 1
        g.addWidget(self.chk_flip, row, 0, 1, 3)
        row += 1

        # ---- layers --------------------------------------------------------
        self._layer_checks = {}
        layers = [("show_best", "Best move (green)"), ("show_second", "2nd move (blue)"),
                  ("show_threat", "Opp. threat (red)"), ("show_contested", "Contested 🔥"),
                  ("show_hung", "Hung 🩸 / weak 💧"), ("overlay_visible", "Overlay on")]
        for i, (key, text) in enumerate(layers):
            cb = QCheckBox(text)
            cb.setChecked(getattr(s, key))
            cb.toggled.connect(lambda v, k=key: self.state.update(_quiet=(k in QUIET), **{k: v}))
            g.addWidget(cb, row + i // 2, (i % 2) * 2, 1, 2 if i % 2 == 0 else 1)
            self._layer_checks[key] = cb
        row += 3

        g.addWidget(QLabel("Opacity"), row, 0)
        self.op_slider = QSlider(Qt.Orientation.Horizontal)
        self.op_slider.setRange(20, 100)
        self.op_slider.setValue(int(s.overlay_opacity * 100))
        self.op_slider.valueChanged.connect(lambda v: self.state.update(_quiet=True, overlay_opacity=v / 100.0))
        g.addWidget(self.op_slider, row, 1, 1, 2)
        row += 1

        # ---- capture -------------------------------------------------------
        g.addWidget(QLabel("Monitor"), row, 0)
        self.mon_box = QComboBox()
        for i, m in enumerate(list_monitors(), start=1):
            self.mon_box.addItem(f"{i}: {m['width']}x{m['height']}", i)
        idx = self.mon_box.findData(s.monitor_index)
        self.mon_box.setCurrentIndex(max(0, idx))
        self.mon_box.currentIndexChanged.connect(self._on_monitor)
        g.addWidget(self.mon_box, row, 1, 1, 2)
        row += 1

        g.addWidget(QLabel("Capture"), row, 0)
        self.backend_box = QComboBox()
        self.backend_box.addItems(["mss", "bettercam"])
        self.backend_box.setCurrentText(s.capture_backend)
        self.backend_box.currentTextChanged.connect(lambda v: self.state.update(capture_backend=v))
        g.addWidget(self.backend_box, row, 1)
        self.hz = QSpinBox()
        self.hz.setRange(1, 20)
        self.hz.setSuffix(" fps")
        self.hz.setValue(int(s.scan_hz))
        self.hz.valueChanged.connect(lambda v: self.state.update(scan_hz=float(v)))
        g.addWidget(self.hz, row, 2)
        row += 1

        g.addWidget(QLabel("Vision backend"), row, 0)
        self.vision_box = QComboBox()
        self.vision_box.addItems(["classical", "yolo"])
        self.vision_box.setCurrentText(s.vision_backend if s.vision_backend in ("classical", "yolo") else "classical")
        self.vision_box.setToolTip("'yolo' needs ultralytics and weights set in settings.json; "
                                   "without them the classical reader is used.")
        self.vision_box.currentTextChanged.connect(lambda v: self.state.update(vision_backend=v))
        g.addWidget(self.vision_box, row, 1, 1, 2)
        row += 1

        g.addWidget(QLabel("Status log level"), row, 0)
        self.status_box = QComboBox()
        self.status_box.addItems(["INFO", "WARNING", "ERROR"])
        self.status_box.setCurrentText(s.status_level if s.status_level in ("INFO", "WARNING", "ERROR") else "INFO")
        self.status_box.currentTextChanged.connect(lambda v: self.state.update(_quiet=True, status_level=v))
        g.addWidget(self.status_box, row, 1, 1, 2)
        row += 1

        self.chk_auto = QCheckBox("Auto-detect board (2D + 3D, no clicking)")
        self.chk_auto.setToolTip("Finds the chessboard on screen by itself, follows it when the camera moves, "
                                 "and reads the game from the starting position - no corner clicks and no "
                                 "'Learn pieces'. Needs to see the starting position once at the beginning of "
                                 "a game (or press 'Reset game' on one).")
        self.chk_auto.setChecked(s.auto_detect)
        self.chk_auto.toggled.connect(self._on_auto)
        g.addWidget(self.chk_auto, row, 0, 1, 3)
        row += 1

        self.chk_3d = QCheckBox("3D mode (follow moves)")
        self.chk_3d.setToolTip("For 3D boards: keeps a 2D model of the game and only watches which squares "
                               "change. Start it with the board in the starting position + 'Reset game'.")
        self.chk_3d.setChecked(s.follow_moves)
        self.chk_3d.toggled.connect(self._on_3d)
        g.addWidget(self.chk_3d, row, 0, 1, 2)
        self.thr_spin = QSpinBox()
        self.thr_spin.setRange(5, 120)
        self.thr_spin.setPrefix("contrast ")
        self.thr_spin.setToolTip("Lower = more sensitive to pieces that blend into the square. "
                                 "Press 'Learn pieces' again after changing this.")
        self.thr_spin.setValue(int(s.fg_threshold))
        self.thr_spin.valueChanged.connect(self._on_thr)
        g.addWidget(self.thr_spin, row, 2)
        row += 1

        # ---- menu & hotkeys (collapsible) ---------------------------------
        self.adv_btn = QPushButton("Menu && hotkeys  \u25b8")
        self.adv_btn.setObjectName("flat")
        self.adv_btn.clicked.connect(self._toggle_adv)
        g.addWidget(self.adv_btn, row, 0, 1, 2)
        self.theme_btn = QPushButton("Theme  \u25b8")
        self.theme_btn.setObjectName("flat")
        self.theme_btn.setToolTip("Colours, font, softness and a background image for this menu.")
        self.theme_btn.clicked.connect(self._toggle_theme)
        g.addWidget(self.theme_btn, row, 2)
        row += 1
        # separate always-on-top window, so opening/closing it never resizes the main menu
        self._adv_open = False
        self.adv = ThemedWidget(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint |
                           Qt.WindowType.Tool)
        self.adv.set_theme(self.theme)
        self.adv.setStyleSheet(make_style(self.theme, 1.0))
        self.adv.setFixedWidth(340)
        self.adv.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.adv.setWindowOpacity(float(s.menu_opacity))
        adv_root = QVBoxLayout(self.adv)
        adv_root.setContentsMargins(0, 0, 0, 0)
        adv_root.setSpacing(0)
        self.adv_title = TitleBar("Menu & hotkeys", self._close_adv, self._close_adv)
        adv_root.addWidget(self.adv_title)
        adv_body = QWidget()
        adv_root.addWidget(adv_body)
        ag = QGridLayout(adv_body)
        ag.setContentsMargins(12, 10, 12, 12)
        ag.setVerticalSpacing(6)
        arow = 0
        self.chk_hk = QCheckBox("Hotkeys enabled")
        self.chk_hk.setChecked(s.hotkeys_enabled)
        self.chk_hk.toggled.connect(lambda v: self.state.update(_quiet=True, hotkeys_enabled=v))
        ag.addWidget(self.chk_hk, arow, 0, 1, 3)
        arow += 1
        self._key_btns = {}
        for act in ACTIONS:
            ag.addWidget(QLabel(ACTION_LABELS[act]), arow, 0, 1, 1)
            btn = QPushButton(getattr(s, act) or "(none)")
            btn.setObjectName("flat")
            btn.clicked.connect(lambda _=False, a=act: self._rebind(a))
            ag.addWidget(btn, arow, 1, 1, 2)
            self._key_btns[act] = btn
            arow += 1
        hint = QLabel("Click a key box, then press the new key (Ctrl/Shift/Alt combos work). "
                      "Esc = cancel, Backspace = unbind. Keys are only read, never blocked.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        ag.addWidget(hint, arow, 0, 1, 3)
        arow += 1
        ag.addWidget(QLabel("Menu opacity"), arow, 0)
        mo = QSlider(Qt.Orientation.Horizontal)
        mo.setRange(30, 100)
        mo.setValue(int(s.menu_opacity * 100))
        mo.valueChanged.connect(self._on_menu_opacity)
        ag.addWidget(mo, arow, 1, 1, 2)
        arow += 1
        self.chk_hidden = QCheckBox("Start with menu hidden")
        self.chk_hidden.setChecked(s.start_menu_hidden)
        self.chk_hidden.toggled.connect(lambda v: self.state.update(_quiet=True, start_menu_hidden=v))
        ag.addWidget(self.chk_hidden, arow, 0, 1, 3)
        arow += 1
        self.chk_tray = QCheckBox("Tray icon (click = open/close menu)")
        self.chk_tray.setChecked(s.show_tray)
        self.chk_tray.toggled.connect(self._on_tray)
        ag.addWidget(self.chk_tray, arow, 0, 1, 3)
        arow += 1
        self.chk_excl = QCheckBox("Exclude overlay from screen capture")
        self.chk_excl.setChecked(s.exclude_from_capture)
        self.chk_excl.toggled.connect(self._on_exclude)
        ag.addWidget(self.chk_excl, arow, 0, 1, 3)
        arow += 1
        b_pos = QPushButton("Reset menu position")
        b_pos.setObjectName("flat")
        b_pos.clicked.connect(lambda: self.move(60, 60))
        ag.addWidget(b_pos, arow, 0, 1, 3)

        # ---- actions -------------------------------------------------------
        self.btn_run = QPushButton("Stop detection")
        self.btn_run.setObjectName("flat")
        self.btn_run.clicked.connect(self._toggle_detection)
        g.addWidget(self.btn_run, row, 0, 1, 3)
        row += 1
        b_cal = QPushButton("Calibrate board (4 corners)")
        b_cal.clicked.connect(self.calibrate_requested.emit)
        g.addWidget(b_cal, row, 0, 1, 3)
        row += 1
        b_learn = QPushButton("Learn pieces (starting position)")
        b_learn.clicked.connect(lambda: self.state.request("learn"))
        g.addWidget(b_learn, row, 0, 1, 2)
        b_reset = QPushButton("Reset game")
        b_reset.setObjectName("flat")
        b_reset.clicked.connect(lambda: self.state.request("reset"))
        g.addWidget(b_reset, row, 2)
        row += 1

        self.status = QLabel("Starting...")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        g.addWidget(self.status, row, 0, 1, 3)

        self._theme_open = False
        self.theme_win = ThemeWindow(self.theme)
        self.theme_win.changed.connect(self._on_theme)
        self.adv.bg_key = "adv"
        self.theme_win.size_widgets = {"main": self, "adv": self.adv, "theme": self.theme_win}
        self._grip = ResizeGrip(self)
        self._building = False
        self._refresh_engine_fields()
        self._apply_scale()

    # ---- public ------------------------------------------------------------
    def set_status(self, text: str):
        self.status.setText(text)

    # ---- resizing ----------------------------------------------------------
    def _wanted_scale(self) -> float:
        return max(MIN_SCALE, min(MAX_SCALE, self.width() / BASE_W))

    def _apply_scale(self):
        """Re-style everything for the current width, then size the height to the content."""
        k = self._wanted_scale()
        self._scale = k
        style = make_style(self.theme, k)
        self.setStyleSheet(style)
        self.adv.setStyleSheet(style)
        self._theme_timer.start(350)   # editor restyles once the user pauses (not mid slider-drag)
        self.adv.setFixedWidth(int(round(340 * k)))
        for t in (self.title_bar, self.adv_title):
            t.set_scale(k)
        for b, w in self._scaled_btns:
            b.setFixedWidth(int(round(w * k)))
        self._fit()
        if self._adv_open and self.adv.isVisible():
            self._show_adv()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        g = getattr(self, "_grip", None)
        if g is not None:
            g.move(self.width() - g.width(), self.height() - g.height())
            g.raise_()
        if self._building:
            return
        if abs(self._wanted_scale() - self._scale) >= 0.03:
            self._scale_timer.start(30)          # restyle after the drag settles a moment (cheap debounce)
        self._pos_timer.start(400)                # also persists the width

    def wheelEvent(self, e):
        """Ctrl + wheel over the menu = bigger / smaller."""
        if e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            step = 24 if e.angleDelta().y() > 0 else -24
            self.resize(max(self.minimumWidth(), min(self.maximumWidth(), self.width() + step)), self.height())
            e.accept()
        else:
            super().wheelEvent(e)

    def showEvent(self, e):
        super().showEvent(e)
        hwnd = int(self.winId())
        # keep this window out of our own screen capture too
        winflags.exclude_from_capture(hwnd, self.state.settings.exclude_from_capture)
        winflags.keep_topmost(hwnd)
        if self._adv_open:
            self._show_adv()
        if self._theme_open:
            self._show_theme()
        self._fit()

    # ---- engine-dependent UI ----------------------------------------------
    def _refresh_engine_fields(self):
        s = self.state.settings
        kind = s.engine_kind
        self._building = True
        self.sf_label.setVisible(kind == KIND_STOCKFISH)
        self.sf_box.setVisible(kind == KIND_STOCKFISH)

        hints = {KIND_STOCKFISH: "Stockfish exe", KIND_MAIA3: "maia3-uci command",
                 KIND_MAIA_LC0: "lc0.exe", KIND_LC0: "lc0.exe"}
        self.cmd_label.setText(hints[kind])
        self.cmd_edit.setText(s.engine_cmd.get(kind, ""))

        has_w = kind in (KIND_MAIA_LC0, KIND_LC0)
        for w in (self.w_label, self.w_edit, self.w_btn):
            w.setVisible(has_w)
        self.w_label.setText("Maia folder" if kind == KIND_MAIA_LC0 else "Weights file")
        self.w_edit.setText(s.weights_path.get(kind, ""))

        lo, hi = strength.elo_slider_range(kind, s.sf_mode)
        elo = strength.clamp(int(s.elo), lo, hi)
        self.slider.setRange(lo // 25, hi // 25)
        self.slider.setValue(elo // 25)
        titles = {KIND_STOCKFISH: "ELO / Strength  (-> Skill Level / Depth)",
                  KIND_MAIA3: "Target ELO  (sent to Maia 3, nodes=1)",
                  KIND_MAIA_LC0: "Target ELO  (picks Maia net, nodes=1)",
                  KIND_LC0: "Strength  (-> node limit)"}
        self.elo_title.setText(titles[kind])

        # Maia 3 opponent level
        is_m3 = kind == KIND_MAIA3
        for w in (self.chk_same, self.oppo_label, self.oppo_slider):
            w.setVisible(is_m3)
        self.chk_same.setChecked(s.oppo_same_as_elo)
        self.oppo_slider.setRange(lo // 25, hi // 25)
        oppo = strength.clamp(int(s.oppo_elo), lo, hi)
        self.oppo_slider.setValue(oppo // 25)
        self.oppo_slider.setEnabled(not s.oppo_same_as_elo)
        self.oppo_label.setText(f"Opponent ELO: {oppo}" + ("  (linked to mine)" if s.oppo_same_as_elo else ""))
        # engine-specific search knobs
        is_sf = kind == KIND_STOCKFISH
        has_nodes = kind in (KIND_MAIA3, KIND_MAIA_LC0, KIND_LC0)
        for w in (self.nodes_label, self.nodes_spin):
            w.setVisible(has_nodes)
        for w in (self.mt_label, self.mt_spin):
            w.setVisible(is_sf and s.sf_mode != "Depth")
        for w in (self.th_label, self.th_spin, self.hash_spin):
            w.setVisible(is_sf)
        self._building = False
        if self.isVisible():
            self._fit()
        if elo != s.elo:
            self.state.update(elo=elo)
        self._update_elo_label(elo)

    def _update_elo_label(self, elo: int):
        s = self.state.settings
        self.elo_desc.setText(strength.describe(s.engine_kind, s.sf_mode, elo,
                                                s.weights_path.get(KIND_MAIA_LC0, "")))

    # ---- handlers ------------------------------------------------------------
    # ---- menu / hotkey helpers -----------------------------------------------
    def toggle_menu(self):
        if self.locked:
            return
        if self.isVisible():
            self.hide()
        else:
            self.show()
            self.raise_()

    def toggle_overlay(self):
        cb = self._layer_checks["overlay_visible"]
        cb.setChecked(not cb.isChecked())
        self.set_status("Overlay " + ("on" if cb.isChecked() else "off"))

    def moveEvent(self, e):
        super().moveEvent(e)
        if self._adv_open and self.adv.isVisible():
            self._place_adv()
        self._place_theme()
        if not self._building:
            self._pos_timer.start(400)

    def _save_pos(self):
        p = self.pos()
        self.state.update(_quiet=True, menu_pos=[p.x(), p.y()], menu_width=int(self.width()))

    # ---- theme -------------------------------------------------------------
    def _on_theme(self, t):
        self.theme = Theme.from_dict(t.to_dict())
        self.state.update(_quiet=True, theme=self.theme.to_dict())
        self.set_theme(self.theme)
        self.adv.set_theme(self.theme)
        self._apply_scale()
        self._place_theme()

    def _toggle_theme(self):
        self._theme_open = not self._theme_open
        if self._theme_open:
            self._show_theme()
        else:
            self.theme_win.hide()
        self.theme_btn.setText("Theme  \u25c2" if self._theme_open else "Theme  \u25b8")

    def _show_theme(self):
        self.theme_win.restyle(self.theme, self._scale)
        self._place_theme()
        self.theme_win.show()
        self.theme_win.raise_()
        winflags.exclude_from_capture(int(self.theme_win.winId()), self.state.settings.exclude_from_capture)
        winflags.keep_topmost(int(self.theme_win.winId()))

    def _place_theme(self):
        if self._theme_open and self.theme_win.isVisible():
            extra = (self.adv.width() + 6) if (self._adv_open and self.adv.isVisible()) else 0
            self.theme_win.dock_beside(self, extra)

    def _toggle_adv(self):
        if self._adv_open:
            self._close_adv()
        else:
            self._adv_open = True
            self._show_adv()

    def _close_adv(self):
        self._adv_open = False
        self.adv.hide()
        self.adv_btn.setText("Menu && hotkeys  \u25b8")

    def _show_adv(self):
        self.adv_btn.setText("Menu && hotkeys  \u25c2")
        lay = self.adv.layout()
        lay.activate()
        self.adv.setFixedHeight(lay.totalHeightForWidth(self.adv.width()))
        self._place_adv()
        self.adv.show()
        self.adv.raise_()
        winflags.exclude_from_capture(int(self.adv.winId()), self.state.settings.exclude_from_capture)
        winflags.keep_topmost(int(self.adv.winId()))

    def _place_adv(self):
        """Dock the panel beside the main menu (right side, or left if there is no room)."""
        scr = QGuiApplication.screenAt(self.pos()) or QGuiApplication.primaryScreen()
        area = scr.availableGeometry()
        x = self.x() + self.width() + 6
        if x + self.adv.width() > area.right():
            x = self.x() - self.adv.width() - 6
        y = min(self.y(), max(area.top(), area.bottom() - self.adv.height()))
        self.adv.move(max(area.left(), x), y)

    def hideEvent(self, e):
        super().hideEvent(e)
        self.adv.hide()  # follows the main menu; _adv_open is kept so it comes back with it
        self.theme_win.hide()

    def _fit(self):
        """Size the window to exactly its content. Word-wrapped labels make sizeHint()
        overestimate the height, so ask the layout for its height at our fixed width."""
        lay = self.layout()
        self.setMinimumHeight(0)  # release the old fixed height first, or it feeds back into the measurement
        self.setMaximumHeight(16777215)
        lay.invalidate()
        lay.activate()
        w = self.width()
        h = lay.totalHeightForWidth(w) if lay.hasHeightForWidth() else lay.totalSizeHint().height()
        self.setFixedHeight(max(h, lay.totalMinimumSize().height()))

    def _on_auto(self, v: bool):
        self.state.update(auto_detect=v)
        self.state.request("reset")
        self.set_status("Auto mode on: looking for a board. It needs to see the starting position once." if v else
                        "Auto mode off: calibrate the four corners, then 'Learn pieces'.")

    def _on_3d(self, v: bool):
        self.state.update(follow_moves=v)
        self.state.request("reset")
        self.set_status("3D mode on: with the board in its starting position, press 'Reset game'." if v else
                        "3D mode off: press 'Learn pieces' in the starting position.")

    def _on_thr(self, v: int):
        self.state.update(fg_threshold=int(v))
        self.set_status("Contrast changed: press 'Learn pieces' again (starting position).")

    def _toggle_detection(self):
        on = not self.state.detecting
        self.state.set_detecting(on)
        self.btn_run.setText("Stop detection" if on else "Start detection")
        self.set_status("Detection started." if on else "Detection stopped (screen capture + engine paused).")

    def toggle_detection(self):
        self._toggle_detection()

    def _on_menu_opacity(self, v: int):
        self.setWindowOpacity(v / 100.0)
        self.adv.setWindowOpacity(v / 100.0)
        self.state.update(_quiet=True, menu_opacity=v / 100.0)

    def _on_tray(self, v: bool):
        self.state.update(_quiet=True, show_tray=v)
        self.tray_changed.emit(v)

    def _on_exclude(self, v: bool):
        self.state.update(exclude_from_capture=v)
        hwnd = int(self.winId())
        winflags.exclude_from_capture(hwnd, v)
        winflags.exclude_from_capture(int(self.adv.winId()), v)
        self.set_status("Overlay window re-applies this when the monitor changes or the app restarts.")

    def _rebind(self, act: str):
        if self.hotkeys is None:
            return
        btn = self._key_btns[act]
        btn.setText("press a key...")

        def done(combo):
            s = self.state.settings
            if combo is None:
                pass
            elif combo == "" and act == "menu_key":
                self.set_status("The menu key can't be unbound (use another key instead).")
            elif combo and any(combo == getattr(s, o) for o in ACTIONS if o != act):
                self.set_status(f"{combo} is already used by another action.")
            else:
                self.state.update(_quiet=True, **{act: combo})
                self.set_status(f"{ACTION_LABELS[act]}: {combo or 'unbound'}")
            btn.setText(getattr(self.state.settings, act) or "(none)")

        self.hotkeys.capture(done)

    def _set(self, **kw):
        if not self._building:
            self.state.update(**kw)

    def _on_same(self, v: bool):
        if self._building:
            return
        self.state.update(oppo_same_as_elo=v)
        self._refresh_engine_fields()

    def _on_oppo_slider(self, v: int):
        if self._building:
            return
        self.oppo_label.setText(f"Opponent ELO: {v * 25}")
        if not self.oppo_slider.isSliderDown():
            self.state.update(oppo_elo=v * 25)

    def _on_engine(self, kind: str):
        if self._building:
            return
        self.state.update(engine_kind=kind)
        self._refresh_engine_fields()

    def _on_sf_mode(self, mode: str):
        if self._building:
            return
        self.state.update(sf_mode=mode)
        self._refresh_engine_fields()

    def _on_cmd(self):
        if self._building:
            return
        kind = self.state.settings.engine_kind
        cmds = dict(self.state.settings.engine_cmd)
        cmds[kind] = self.cmd_edit.text().strip()
        self.state.update(engine_cmd=cmds)

    def _on_weights(self):
        if self._building:
            return
        kind = self.state.settings.engine_kind
        w = dict(self.state.settings.weights_path)
        w[kind] = self.w_edit.text().strip()
        self.state.update(weights_path=w)
        self._update_elo_label(self.slider.value() * 25)

    def _browse_cmd(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select engine executable", "", "Executables (*.exe);;All (*)")
        if path:
            self.cmd_edit.setText(f'"{path}"' if " " in path else path)
            self._on_cmd()

    def _browse_weights(self):
        kind = self.state.settings.engine_kind
        if kind == KIND_MAIA_LC0:
            path = QFileDialog.getExistingDirectory(self, "Folder with maia-1100.pb.gz ... maia-1900.pb.gz")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select Lc0 weights", "", "Weights (*.pb *.pb.gz *.gz);;All (*)")
        if path:
            self.w_edit.setText(path)
            self._on_weights()

    def _on_slider_changed(self, v: int):
        if self._building:
            return
        self._update_elo_label(v * 25)
        if not self.slider.isSliderDown():
            self._commit_elo()

    def _commit_elo(self):
        if self._building:
            return
        self.state.update(elo=self.slider.value() * 25)

    def _on_monitor(self, _):
        if self._building:
            return
        i = self.mon_box.currentData()
        self.state.update(monitor_index=int(i))
        self.monitor_changed.emit(int(i))