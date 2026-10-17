"""Menu theming: colours, font, corner softness, optional background image (with opacity).

Everything about how the menu *looks* lives here, so gui.py only has to
  * give its windows a `theme` and inherit `ThemedWidget` (paints colour + image),
  * call `make_style(theme, scale)` for the stylesheet,
  * open the `ThemeWindow` editor and react to its `changed` signal.

Theme settings are stored in settings.json under "theme" (see Theme.to_dict).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields

from PyQt6.QtCore import QPoint, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QGuiApplication, QPainter, QPixmap
from PyQt6.QtWidgets import (QColorDialog, QComboBox, QFileDialog, QFontComboBox, QFrame,
                             QGridLayout, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget)

IMAGE_MODES = ["Fill", "Fit", "Stretch", "Tile", "Center"]
IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.webp *.gif);;All files (*)"

COLOR_FIELDS = [               # (attribute, label)
    ("bg", "Background"),
    ("panel", "Boxes / buttons (flat)"),
    ("text", "Text"),
    ("muted", "Hint text"),
    ("accent", "Accent / buttons"),
    ("ok", "Status text"),
    ("border", "Borders"),
    ("title", "Title bar"),
]


@dataclass
class Theme:
    # colours (#rrggbb)
    bg: str = "#1e1f26"
    panel: str = "#2b2d38"
    text: str = "#e6e6e6"
    muted: str = "#9aa0b5"
    accent: str = "#3b6ef5"
    ok: str = "#7ee2a8"
    border: str = "#3d4050"
    title: str = "#2a2c36"
    # type + shape
    font_family: str = "Segoe UI"
    font_px: int = 12              # base font size at 100 % scale
    radius: int = 4                # corner softness in px
    panel_opacity: int = 100       # % opacity of boxes/buttons (lower = glassy over an image)
    # background image
    image: str = ""                # local file path, "" = none
    image_opacity: int = 35        # %
    image_mode: str = "Fill"
    # one image for every menu, or one each (main menu / hotkeys panel / theme window)
    image_shared: bool = True
    adv_image: str = ""
    adv_image_opacity: int = 35
    adv_image_mode: str = "Fill"
    theme_image: str = ""
    theme_image_opacity: int = 35
    theme_image_mode: str = "Fill"

    def image_for(self, key: str):
        """(path, opacity %, mode) used by window `key` ('main' | 'adv' | 'theme')."""
        if self.image_shared or key == "main":
            return self.image, self.image_opacity, self.image_mode
        pre = key + "_"
        return getattr(self, pre + "image"), getattr(self, pre + "image_opacity"), getattr(self, pre + "image_mode")

    @classmethod
    def from_dict(cls, d) -> "Theme":
        t = cls()
        if not isinstance(d, dict):
            return t
        for f in fields(cls):
            if f.name not in d:
                continue
            v = d[f.name]
            try:
                if f.type in ("int", int):
                    v = int(v)
                elif f.type in ("bool", bool):
                    v = bool(v)
                elif f.type in ("str", str):
                    v = str(v)
                    if f.name in dict(COLOR_FIELDS) and not QColor.isValidColor(v):
                        continue
            except (TypeError, ValueError):
                continue
            setattr(t, f.name, v)
        t.font_px = max(8, min(24, t.font_px))
        t.radius = max(0, min(20, t.radius))
        t.panel_opacity = max(20, min(100, t.panel_opacity))
        for pre in ("", "adv_", "theme_"):
            setattr(t, pre + "image_opacity", max(0, min(100, getattr(t, pre + "image_opacity"))))
            if getattr(t, pre + "image_mode") not in IMAGE_MODES:
                setattr(t, pre + "image_mode", "Fill")
        return t

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS: dict[str, Theme] = {
    "Dark (default)": Theme(),
    "Midnight": Theme(bg="#0d1117", panel="#161b22", text="#e6edf3", muted="#8b949e", accent="#2f81f7",
                      ok="#56d364", border="#30363d", title="#010409"),
    "Light": Theme(bg="#f4f5f8", panel="#ffffff", text="#1f2430", muted="#6b7280", accent="#3b6ef5",
                   ok="#1a8f4e", border="#cfd3dc", title="#e3e6ee"),
    "Soft pastel": Theme(bg="#f3ecf7", panel="#fbf7fd", text="#3d3350", muted="#8a7fa0", accent="#a78bdb",
                         ok="#4fa88a", border="#dccdea", title="#e6d9f1", radius=10),
    "Ocean": Theme(bg="#0f2a3a", panel="#17384c", text="#e3f4fb", muted="#8fb8c9", accent="#1fb6c9",
                   ok="#7de3c0", border="#2a5670", title="#0a1f2c", radius=8),
    "Forest": Theme(bg="#18241c", panel="#223329", text="#e4efe6", muted="#92ae99", accent="#4caf6a",
                    ok="#9be38a", border="#34503e", title="#101a14", radius=6),
    "Rose": Theme(bg="#2a1a22", panel="#3a2430", text="#f7e8ee", muted="#c49aab", accent="#e0527e",
                  ok="#8fe3b0", border="#5a3547", title="#1d1017", radius=10),
    "High contrast": Theme(bg="#000000", panel="#101010", text="#ffffff", muted="#d0d0d0", accent="#ffd400",
                           ok="#00ff7f", border="#ffffff", title="#000000", radius=2),
}


# ---------------------------------------------------------------------------
# colour helpers + stylesheet
# ---------------------------------------------------------------------------
def shade(hex_color: str, amount: float) -> str:
    """amount > 0 lightens, < 0 darkens (-1..1)."""
    c = QColor(hex_color)
    r, g, b = c.red(), c.green(), c.blue()
    if amount >= 0:
        r, g, b = (int(v + (255 - v) * amount) for v in (r, g, b))
    else:
        r, g, b = (int(v * (1 + amount)) for v in (r, g, b))
    return QColor(r, g, b).name()


def _rgba(hex_color: str, opacity_pct: int) -> str:
    c = QColor(hex_color)
    return f"rgba({c.red()},{c.green()},{c.blue()},{int(round(255 * opacity_pct / 100.0))})"


def readable_on(hex_color: str) -> str:
    c = QColor(hex_color)
    lum = 0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()
    return "#101010" if lum > 150 else "#ffffff"


def make_style(t: Theme, k: float = 1.0) -> str:
    """Qt stylesheet for theme t at UI scale k (1.0 = design size). Plain widgets are transparent so
    the colour/image painted by ThemedWidget shows through."""
    def px(v: float) -> int:
        return max(1, int(round(v * k)))

    fs = t.font_px
    rad = px(t.radius)
    box = _rgba(t.panel, t.panel_opacity)
    box_hover = _rgba(shade(t.panel, 0.12), t.panel_opacity)
    btn_text = readable_on(t.accent)
    fam = t.font_family.replace("'", "")
    return f"""
QWidget {{ background:transparent; color:{t.text}; font-family:'{fam}'; font-size:{px(fs)}px; }}
QToolTip {{ background:{t.panel}; color:{t.text}; border:1px solid {t.border}; }}
QFrame#title {{ background:{_rgba(t.title, 92)}; border-bottom:1px solid {t.border}; }}
QLabel#titleText {{ font-weight:bold; font-size:{px(fs + 1)}px; background:transparent; }}
QLabel#hint {{ color:{t.muted}; }}
QLabel#status {{ color:{t.ok}; }}
QLineEdit, QComboBox, QSpinBox, QFontComboBox {{ background:{box}; border:1px solid {t.border}; border-radius:{rad}px; padding:{px(3)}px; }}
QComboBox QAbstractItemView {{ background:{t.panel}; color:{t.text}; selection-background-color:{t.accent}; selection-color:{btn_text}; }}
QPushButton {{ background:{t.accent}; color:{btn_text}; border:none; border-radius:{rad}px; padding:{px(6)}px {px(10)}px; }}
QPushButton:hover {{ background:{shade(t.accent, 0.15)}; }}
QPushButton#flat {{ background:{box}; color:{t.text}; border:1px solid {t.border}; }}
QPushButton#flat:hover {{ background:{box_hover}; }}
QSlider::groove:horizontal {{ height:{px(5)}px; background:{t.border}; border-radius:{max(1, rad // 2)}px; }}
QSlider::handle:horizontal {{ background:{t.accent}; width:{px(14)}px; margin:-{px(5)}px 0; border-radius:{px(7)}px; }}
QCheckBox::indicator {{ width:{px(14)}px; height:{px(14)}px; border:1px solid {t.border}; border-radius:{max(1, rad // 2)}px; background:{box}; }}
QCheckBox::indicator:checked {{ background:{t.accent}; }}
"""


# ---------------------------------------------------------------------------
# painting: background colour + optional image
# ---------------------------------------------------------------------------
_pix_cache: dict = {}


def _load_pixmap(path: str) -> QPixmap | None:
    if not path:
        return None
    pm = _pix_cache.get(path)
    if pm is None:
        pm = QPixmap(path)
        _pix_cache.clear()           # keep at most one source image in memory
        _pix_cache[path] = pm
    return None if pm.isNull() else pm


def paint_background(w: QWidget, t: Theme, key: str = "main") -> None:
    p = QPainter(w)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
    p.fillRect(w.rect(), QColor(t.bg))
    path, opacity, m = t.image_for(key)
    pm = _load_pixmap(path)
    if pm is not None and opacity > 0:
        p.setOpacity(opacity / 100.0)
        W, H = w.width(), w.height()
        if m == "Tile":
            p.drawTiledPixmap(w.rect(), pm)
        elif m == "Stretch":
            p.drawPixmap(QRectF(0, 0, W, H), pm, QRectF(pm.rect()))
        elif m == "Center":
            p.drawPixmap(QPoint((W - pm.width()) // 2, (H - pm.height()) // 2), pm)
        else:                                            # Fill (cover) / Fit (contain)
            s = max(W / pm.width(), H / pm.height()) if m == "Fill" else min(W / pm.width(), H / pm.height())
            dw, dh = pm.width() * s, pm.height() * s
            p.drawPixmap(QRectF((W - dw) / 2, (H - dh) / 2, dw, dh), pm, QRectF(pm.rect()))
        p.setOpacity(1.0)
    p.setPen(QColor(t.border))
    p.drawRect(w.rect().adjusted(0, 0, -1, -1))
    p.end()


class ThemedWidget(QWidget):
    """A top-level window that paints the theme's colour + image behind its (transparent) children."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.theme = Theme()
        self.bg_key = "main"          # which window's image to draw: 'main' | 'adv' | 'theme'

    def set_theme(self, t: Theme) -> None:
        self.theme = t
        self.update()

    def paintEvent(self, e):
        paint_background(self, self.theme, self.bg_key)


# Recommended image size: the menu can be scaled up to 2x, so 2x its design size keeps it sharp.
# (design width, typical height at 100 %) -> ratio. The image is cropped/fitted per "Image fit".
TARGETS = [("main", "Main menu"), ("adv", "Hotkeys panel"), ("theme", "Theme window")]
IMG_KEYS = ("image", "image_opacity", "image_mode")


# ---------------------------------------------------------------------------
# the editor window
# ---------------------------------------------------------------------------
class _Swatch(QPushButton):
    def __init__(self, color: str):
        super().__init__("")
        self.setObjectName("flat")
        self.setFixedWidth(64)
        self.set_color(color)

    def set_color(self, color: str):
        c = QColor(color)
        self.setStyleSheet(f"QPushButton {{ background:{c.name()}; border:1px solid #888; border-radius:3px; "
                           f"color:{readable_on(c.name())}; }}")
        self.setText(c.name())


class ThemeWindow(ThemedWidget):
    """Small always-on-top editor docked beside the menu. Emits `changed(Theme)` live."""
    changed = pyqtSignal(object)

    def __init__(self, theme: Theme):
        super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint |
                         Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self._t = theme
        self._busy = True
        self._k = 1.0
        self.theme = theme
        self.bg_key = "theme"
        self.size_widgets: dict = {}    # key -> window, filled by gui.py (for the size recommendation)
        self._target = "main"

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.bar = QFrame()
        self.bar.setObjectName("title")
        bl = QHBoxLayout(self.bar)
        bl.setContentsMargins(10, 4, 4, 4)
        title = QLabel("Theme")
        title.setObjectName("titleText")
        bl.addWidget(title, 1)
        self.close_btn = QPushButton("x")
        self.close_btn.setObjectName("flat")
        self.close_btn.clicked.connect(self.hide)
        bl.addWidget(self.close_btn)
        self.bar.mousePressEvent = self._drag
        root.addWidget(self.bar)

        body = QWidget()
        root.addWidget(body)
        g = QGridLayout(body)
        g.setContentsMargins(12, 10, 12, 12)
        g.setVerticalSpacing(6)
        row = 0

        g.addWidget(QLabel("Preset"), row, 0)
        self.preset = QComboBox()
        self.preset.addItem("Custom")
        self.preset.addItems(list(PRESETS))
        self.preset.activated.connect(self._pick_preset)
        g.addWidget(self.preset, row, 1, 1, 2)
        row += 1

        self.swatches: dict[str, _Swatch] = {}
        for attr, label in COLOR_FIELDS:
            g.addWidget(QLabel(label), row, 0, 1, 2)
            sw = _Swatch(getattr(theme, attr))
            sw.clicked.connect(lambda _=False, a=attr: self._pick_color(a))
            g.addWidget(sw, row, 2)
            self.swatches[attr] = sw
            row += 1

        g.addWidget(QLabel("Font"), row, 0)
        self.font_box = QFontComboBox()
        self.font_box.setCurrentFont(QFont(theme.font_family))
        self.font_box.currentFontChanged.connect(lambda f: self._set(font_family=f.family()))
        g.addWidget(self.font_box, row, 1, 1, 2)
        row += 1

        self.font_slider = self._slider(g, row, "Font size", 8, 24, theme.font_px, "font_px", " px")
        row += 1
        self.radius_slider = self._slider(g, row, "Softness", 0, 20, theme.radius, "radius", " px")
        row += 1
        self.panel_slider = self._slider(g, row, "Box opacity", 20, 100, theme.panel_opacity,
                                         "panel_opacity", " %")
        row += 1

        sep = QLabel("Background image")
        sep.setObjectName("hint")
        g.addWidget(sep, row, 0, 1, 3)
        row += 1
        self.shared_chk = QPushButton()
        self.shared_chk.setCheckable(True)
        self.shared_chk.setChecked(theme.image_shared)
        self.shared_chk.setObjectName("flat")
        self.shared_chk.toggled.connect(self._on_shared)
        g.addWidget(self.shared_chk, row, 0, 1, 3)
        row += 1
        self.target_label = QLabel("Edit image for")
        self.target_box = QComboBox()
        for _k, name in TARGETS:
            self.target_box.addItem(name, _k)
        self.target_box.currentIndexChanged.connect(self._on_target)
        g.addWidget(self.target_label, row, 0)
        g.addWidget(self.target_box, row, 1, 1, 2)
        row += 1
        self.img_label = QLabel(self._short(theme.image))
        self.img_label.setObjectName("hint")
        g.addWidget(self.img_label, row, 0, 1, 3)
        row += 1
        b_img = QPushButton("Choose image...")
        b_img.setObjectName("flat")
        b_img.clicked.connect(self._browse_image)
        b_clr = QPushButton("Remove")
        b_clr.setObjectName("flat")
        b_clr.clicked.connect(lambda: self._set(image=""))
        g.addWidget(b_img, row, 0, 1, 2)
        g.addWidget(b_clr, row, 2)
        row += 1
        self.imgop_slider = self._slider(g, row, "Image opacity", 0, 100, theme.image_opacity,
                                         "image_opacity", " %")
        row += 1
        g.addWidget(QLabel("Image fit"), row, 0)
        self.mode_box = QComboBox()
        self.mode_box.addItems(IMAGE_MODES)
        self.mode_box.setCurrentText(theme.image_mode)
        self.mode_box.currentTextChanged.connect(lambda v: self._set(image_mode=v))
        g.addWidget(self.mode_box, row, 1, 1, 2)
        row += 1
        self.size_hint = QLabel("")
        self.size_hint.setObjectName("hint")
        self.size_hint.setWordWrap(True)
        g.addWidget(self.size_hint, row, 0, 1, 3)
        row += 1

        b_reset = QPushButton("Reset")
        b_reset.setObjectName("flat")
        b_reset.clicked.connect(lambda: self.apply_theme(Theme(), emit=True))
        b_exp = QPushButton("Export...")
        b_exp.setObjectName("flat")
        b_exp.clicked.connect(self._export)
        b_imp = QPushButton("Import...")
        b_imp.setObjectName("flat")
        b_imp.clicked.connect(self._import)
        g.addWidget(b_reset, row, 0)
        g.addWidget(b_exp, row, 1)
        g.addWidget(b_imp, row, 2)

        self.setFixedWidth(330)
        self._busy = False
        self._refresh_image_controls()

    # ---- helpers ---------------------------------------------------------------
    def _slider(self, g, row, label, lo, hi, val, attr, suffix):
        g.addWidget(QLabel(label), row, 0)
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(lo, hi)
        s.setValue(val)
        v = QLabel(f"{val}{suffix}")
        v.setMinimumWidth(48)
        v.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        def on(x):
            v.setText(f"{x}{suffix}")
            self._set(**{attr: int(x)})

        s.valueChanged.connect(on)
        g.addWidget(s, row, 1)
        g.addWidget(v, row, 2)
        s._val = v
        s._suffix = suffix
        return s

    @staticmethod
    def _short(path: str) -> str:
        if not path:
            return "(none)"
        return path if len(path) <= 40 else "..." + path[-37:]

    def _drag(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self.windowHandle() is not None:
            self.windowHandle().startSystemMove()

    def _set(self, **kw):
        if self._busy:
            return
        pre = self._img_prefix()
        for k, v in kw.items():
            setattr(self._t, (pre + k) if k in IMG_KEYS else k, v)
        if "image" in kw:
            self.img_label.setText(self._short(kw["image"]))
        self.preset.setCurrentIndex(0)
        self.theme = self._t
        self.changed.emit(self._t)

    def _pick_color(self, attr):
        cur = QColor(getattr(self._t, attr))
        c = QColorDialog.getColor(cur, self, dict(COLOR_FIELDS)[attr])
        if c.isValid():
            self.swatches[attr].set_color(c.name())
            self._set(**{attr: c.name()})

    # ---- per-window images ----------------------------------------------------
    def _img_prefix(self) -> str:
        return "" if (self._t.image_shared or self._target == "main") else self._target + "_"

    def _on_shared(self, v: bool):
        self.shared_chk.setText("One image for all menus: ON" if v else "One image for all menus: OFF (separate images)")
        if self._busy:
            return
        self._t.image_shared = v
        self._refresh_image_controls()
        self.preset.setCurrentIndex(0)
        self.theme = self._t
        self.changed.emit(self._t)

    def _on_target(self, _i):
        if self._busy:
            return
        self._target = self.target_box.currentData() or "main"
        self._refresh_image_controls()

    def _refresh_image_controls(self):
        """Show the image / opacity / fit of the window being edited + the recommended size."""
        was = self._busy
        self._busy = True
        t = self._t
        shared = t.image_shared
        self.shared_chk.setChecked(shared)
        self.shared_chk.setText("One image for all menus: ON" if shared else
                                "One image for all menus: OFF (separate images)")
        self.target_label.setVisible(not shared)
        self.target_box.setVisible(not shared)
        key = "main" if shared else self._target
        path, op, mode = t.image_for(key)
        self.img_label.setText(self._short(path))
        self.imgop_slider.setValue(op)
        self.imgop_slider._val.setText(f"{op}{self.imgop_slider._suffix}")
        self.mode_box.setCurrentText(mode)
        self._update_size_hint(key, shared)
        self._busy = was

    def _update_size_hint(self, key: str, shared: bool):
        """Recommended image size = 2x the window's current size (the menu scales up to 2x)."""
        def rec(w):
            return f"{w.width() * 2} x {w.height() * 2} px"

        widgets = self.size_widgets
        if not widgets:
            return
        if shared:
            lines = [f"{name}: {rec(widgets[k])}" for k, name in TARGETS if k in widgets]
            tip = ("Recommended image size (2x, stays sharp when you enlarge the menu):\n" + "\n".join(lines) +
                   "\nOne tall image (about 9:16, e.g. 1080 x 1920) works for all three with 'Fill'.")
        else:
            w = widgets.get(key)
            name = dict(TARGETS)[key]
            tip = f"{name}: recommended {rec(w)} (ratio {w.width()}:{w.height()})." if w else ""
        self.size_hint.setText(tip)

    def _pick_preset(self, idx):
        if idx <= 0:
            return
        name = self.preset.itemText(idx)
        t = Theme.from_dict(PRESETS[name].to_dict())
        # a preset changes the look, not the user's font choice or background image
        for f in ("font_family", "image", "image_opacity", "image_mode", "image_shared", "adv_image",
                  "adv_image_opacity", "adv_image_mode", "theme_image", "theme_image_opacity",
                  "theme_image_mode"):
            setattr(t, f, getattr(self._t, f))
        self.apply_theme(t, emit=True, keep_preset=idx)

    def _browse_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Background image", self._t.image_for("main" if self._t.image_shared else self._target)[0], IMAGE_FILTER)
        if path:
            self._set(image=path)

    def _export(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export theme", "chess_overlay_theme.json",
                                              "Theme (*.json)")
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(self._t.to_dict(), f, indent=2)
            except OSError:
                pass

    def _import(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import theme", "", "Theme (*.json)")
        if path:
            try:
                with open(path, encoding="utf-8") as f:
                    self.apply_theme(Theme.from_dict(json.load(f)), emit=True)
            except (OSError, ValueError):
                pass

    # ---- public ----------------------------------------------------------------
    def apply_theme(self, t: Theme, emit: bool = False, keep_preset: int = 0):
        """Load a whole theme into the controls (and optionally announce it)."""
        self._busy = True
        self._t = t
        self.theme = t
        for attr, sw in self.swatches.items():
            sw.set_color(getattr(t, attr))
        self.font_box.setCurrentFont(QFont(t.font_family))
        for s, val in ((self.font_slider, t.font_px), (self.radius_slider, t.radius),
                       (self.panel_slider, t.panel_opacity)):
            s.setValue(val)
            s._val.setText(f"{val}{s._suffix}")
        self.preset.setCurrentIndex(keep_preset)
        self._busy = False
        self._refresh_image_controls()
        if emit:
            self.changed.emit(t)

    def showEvent(self, e):
        super().showEvent(e)
        self._refresh_image_controls()

    def restyle(self, t: Theme, k: float):
        self.theme = t
        self.setStyleSheet(make_style(t, k))
        self.close_btn.setFixedSize(int(round(24 * k)), int(round(22 * k)))
        self.bar.setFixedHeight(int(round(32 * k)))
        self.setFixedWidth(int(round(330 * k)))
        lay = self.layout()
        lay.activate()
        self.setFixedHeight(lay.totalHeightForWidth(self.width()) if lay.hasHeightForWidth()
                            else lay.totalSizeHint().height())
        self.update()

    def dock_beside(self, main: QWidget, extra_right: int = 0):
        """Place next to the main menu: right side if there is room (after any other docked panel)."""
        scr = QGuiApplication.screenAt(main.pos()) or QGuiApplication.primaryScreen()
        area = scr.availableGeometry()
        x = main.x() + main.width() + 6 + extra_right
        if x + self.width() > area.right():
            x = main.x() - self.width() - 6
        y = min(main.y(), max(area.top(), area.bottom() - self.height()))
        self.move(max(area.left(), x), y)