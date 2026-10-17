"""Settings model + JSON persistence."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

APP_DIR = Path(os.environ.get("APPDATA", str(Path.home()))) / "ChessOverlay"
SETTINGS_FILE = APP_DIR / "settings.json"
TEMPLATES_FILE = APP_DIR / "templates.npz"

log = logging.getLogger("chessoverlay.config")   # plain stdlib: logsetup imports this module

KIND_STOCKFISH = "Stockfish"
KIND_MAIA3 = "Maia 3"
KIND_MAIA_LC0 = "Maia (Lc0 + weights)"
KIND_LC0 = "Lc0"
ENGINE_KINDS = [KIND_STOCKFISH, KIND_MAIA3, KIND_MAIA_LC0, KIND_LC0]

SF_SKILL = "Skill Level"
SF_DEPTH = "Depth"
SF_UCI_ELO = "UCI_Elo"
SF_MODES = [SF_SKILL, SF_DEPTH, SF_UCI_ELO]


def _default_cmds() -> dict:
    return {
        KIND_STOCKFISH: "stockfish",
        KIND_MAIA3: "maia3-uci --model 5m",
        KIND_MAIA_LC0: "lc0",
        KIND_LC0: "lc0",
    }


@dataclass
class Settings:
    # engine
    engine_kind: str = KIND_STOCKFISH
    engine_cmd: dict = field(default_factory=_default_cmds)
    # Maia (Lc0): folder with maia-1100.pb.gz ... maia-1900.pb.gz
    # Lc0: path to a single weights file
    weights_path: dict = field(default_factory=lambda: {KIND_MAIA_LC0: "", KIND_LC0: ""})
    elo: int = 1500
    sf_mode: str = SF_SKILL
    # Maia 3: separate opponent rating (OppoElo). When oppo_same_as_elo is True the
    # single `Elo` option is used (sets both players).
    oppo_elo: int = 1500
    oppo_same_as_elo: bool = True
    # search / resource knobs
    engine_nodes: int = 0      # 0 = engine default (Maia: 1 node, Lc0: from slider)
    movetime_ms: int = 250     # Stockfish time per analysis (skill / UCI_Elo modes)
    sf_threads: int = 1
    sf_hash_mb: int = 64
    extra_uci: str = ""        # free-form UCI options: "Name=Value; Name2=Value2"

    # game
    play_as_black: bool = False
    board_flipped: bool = False  # True when Black is at the bottom of the screen

    # capture
    monitor_index: int = 1  # 1-based, same numbering as mss
    capture_backend: str = "mss"  # "mss" or "bettercam"
    scan_hz: float = 4.0
    fg_threshold: int = 34  # piece-vs-square contrast threshold for recognition
    auto_detect: bool = True    # find the board by itself (2D and 3D, no corner clicking / templates)
    follow_moves: bool = False  # 3D mode: track the game by watching which squares change
    crop_top_extend: float = 0.0  # >0 for tall 3D pieces (fraction of a cell)

    # optional YOLO vision (needs `pip install ultralytics`; see yolo.py). "classical" = no extra deps.
    vision_backend: str = "classical"      # "classical" | "yolo"
    yolo_piece_weights: str = ""           # detection model run on the warped board (12 piece classes)
    yolo_corner_weights: str = ""          # pose (4 corner keypoints) or segmentation (board mask) model
    yolo_conf: float = 0.25
    yolo_device: str = ""                  # "" = auto, "cpu", "cuda:0", ...
    yolo_piece_imgsz: int = 640
    yolo_corner_imgsz: int = 960
    yolo_corner_every: int = 5             # corner model every N frames; the lattice tracks in between

    # logging (file: %APPDATA%\ChessOverlay\logs) and which messages reach the menu's status line
    log_level: str = "INFO"                # DEBUG | INFO | WARNING | ERROR
    status_level: str = "INFO"

    # menu / hotkeys (e.g. "INSERT", "F8", "CTRL+SHIFT+H"; empty = unbound)
    hotkeys_enabled: bool = True
    menu_key: str = "INSERT"
    overlay_key: str = "HOME"
    reset_key: str = ""
    menu_opacity: float = 0.97
    menu_pos: list = field(default_factory=lambda: [60, 60])
    theme: dict = field(default_factory=dict)   # see theme.Theme (colours, font, image ...)
    menu_width: int = 360      # menu window width (drag the corner grip or Ctrl+wheel); UI scales with it
    start_menu_hidden: bool = False
    show_tray: bool = True

    # overlay layers
    overlay_visible: bool = True
    overlay_opacity: float = 0.85
    show_best: bool = True
    show_second: bool = True
    show_threat: bool = True
    show_contested: bool = True
    show_hung: bool = True
    # Keeps the overlay out of our own screen capture (otherwise our arrows
    # would be read back as part of the board).
    exclude_from_capture: bool = True

    # calibration (monitor-local physical pixels, TL, TR, BR, BL as seen on screen)
    corners: list | None = None
    corners_monitor: int = 1

    @classmethod
    def load(cls) -> "Settings":
        s = cls()
        try:
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return s
        except Exception:
            log.warning("could not read %s, using defaults", SETTINGS_FILE, exc_info=True)
            return s
        known = {f.name for f in fields(cls)}
        for k, v in data.items():
            if k not in known:
                continue
            if isinstance(getattr(s, k), dict) and isinstance(v, dict):
                merged = dict(getattr(s, k))
                merged.update(v)
                setattr(s, k, merged)
            else:
                setattr(s, k, v)
        return s

    def save(self) -> None:
        try:
            APP_DIR.mkdir(parents=True, exist_ok=True)
            SETTINGS_FILE.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        except Exception:
            log.warning("could not save %s", SETTINGS_FILE, exc_info=True)