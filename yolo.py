"""Optional YOLO vision backend (Settings.vision_backend = "yolo").

Nothing here imports torch / ultralytics at import time: `classical` runs with zero extra
dependencies. The heavy import happens lazily, the first time a model is actually needed and only
when weights are configured and present. If anything is missing (package, weights, a failing
inference) the callers fall back to the classical path: every public function here either returns
a result or None / False, and logs the reason once.

Two optional models:

* piece model  - object detection on the *warped* top-down board (640x640 from BoardGeometry.warp).
                 12 classes (white/black x P N B R Q K). Class names are parsed leniently
                 ("white-pawn", "wp", "w_knight", "P", "k" ...), see `label_from_name`.
* corner model - pose (>= 4 keypoints = the outer corners of the 8x8 area) or segmentation
                 (board mask) on the full frame. See `detect_corners`.

Detections become the (64, 13) cost matrix that BoardTracker already consumes
(0 = empty, 1..6 white PNBRQK, 7..12 black): ~0 for what the model sees, 1.0 for what it does not.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass

import cv2
import numpy as np

from geometry import BoardGeometry
from logsetup import get_logger

log = get_logger("yolo")

N_LABELS = 13
ANCHOR = 0.35          # a piece stands this far (in squares) above the bottom of its box, at most
BACKENDS = ("classical", "yolo")


@dataclass
class YoloConfig:
    backend: str = "classical"
    piece_weights: str = ""
    corner_weights: str = ""
    conf: float = 0.25
    device: str = ""            # "" = ultralytics default (cuda if available), "cpu", "cuda:0" ...
    piece_imgsz: int = 640
    corner_imgsz: int = 960
    corner_every: int = 5       # run the corner model every N frames (BoardLocator tracks in between)


_cfg = YoloConfig()
_models: dict = {}              # weights path -> loaded model (None = failed to load)
_warned: set = set()
_lock = threading.Lock()


def configure(cfg: YoloConfig) -> None:
    global _cfg
    if cfg.backend not in BACKENDS:
        log.warning("unknown vision_backend %r, using classical", cfg.backend)
        cfg = YoloConfig(**{**cfg.__dict__, "backend": "classical"})
    if cfg != _cfg:
        log.info("vision backend: %s (pieces=%s corners=%s)", cfg.backend, cfg.piece_weights or "-",
                 cfg.corner_weights or "-")
    _cfg = cfg


def config_from_settings(s) -> YoloConfig:
    return YoloConfig(backend=str(getattr(s, "vision_backend", "classical")),
                      piece_weights=str(getattr(s, "yolo_piece_weights", "") or ""),
                      corner_weights=str(getattr(s, "yolo_corner_weights", "") or ""),
                      conf=float(getattr(s, "yolo_conf", 0.25)),
                      device=str(getattr(s, "yolo_device", "") or ""),
                      piece_imgsz=int(getattr(s, "yolo_piece_imgsz", 640)),
                      corner_imgsz=int(getattr(s, "yolo_corner_imgsz", 960)),
                      corner_every=max(1, int(getattr(s, "yolo_corner_every", 5))))


def get_config() -> YoloConfig:
    return _cfg


def _warn_once(key, msg, *args, exc_info=False):
    if key not in _warned:
        _warned.add(key)
        log.warning(msg, *args, exc_info=exc_info)


def _load(path: str):
    """Model for `path` (lazy import of ultralytics), or None. Failures are remembered."""
    if not path:
        return None
    if path in _models:
        return _models[path]
    if not os.path.isfile(path):
        _warn_once(("missing", path), "YOLO weights not found: %s (falling back to classical)", path)
        return None
    with _lock:
        if path in _models:
            return _models[path]
        try:
            from ultralytics import YOLO          # lazy: torch is only needed when this runs
            model = YOLO(path)
            log.info("loaded YOLO weights %s", path)
        except Exception:
            _warn_once(("load", path), "could not load YOLO weights %s (falling back to classical)", path,
                       exc_info=True)
            model = None
        _models[path] = model
        return model


def piece_backend_active() -> bool:
    return _cfg.backend == "yolo" and _load(_cfg.piece_weights) is not None


def corner_backend_active() -> bool:
    return _cfg.backend == "yolo" and _load(_cfg.corner_weights) is not None


# ---------------------------------------------------------------------------
# class names -> labels
# ---------------------------------------------------------------------------
_PIECE_WORDS = {"pawn": 1, "knight": 2, "bishop": 3, "rook": 4, "queen": 5, "king": 6}
_PIECE_LETTERS = {"p": 1, "n": 2, "b": 3, "r": 4, "q": 5, "k": 6}


def label_from_name(name: str) -> int | None:
    """'white-pawn' / 'Black_Knight' / 'wp' / 'bK' / 'P' / 'k' ... -> 1..12 (None if not a piece)."""
    raw = str(name).strip()
    low = raw.lower().replace("-", " ").replace("_", " ").replace(".", " ")
    toks = low.split()
    color = piece = None
    for t in toks:
        if t in ("white", "w"):
            color = 0
        elif t in ("black", "b") and len(toks) > 1:
            color = 1
        elif t in _PIECE_WORDS:
            piece = _PIECE_WORDS[t]
    if piece is not None and color is not None:
        return piece + 6 * color
    if len(toks) == 1:
        t = toks[0]
        if len(t) == 2 and t[0] in "wb" and t[1] in _PIECE_LETTERS:          # 'wp', 'bk'
            return _PIECE_LETTERS[t[1]] + (6 if t[0] == "b" else 0)
        if len(t) > 2:                                                       # 'whitepawn', 'blackking'
            for cw, off in (("white", 0), ("black", 6)):
                if t.startswith(cw) and t[len(cw):] in _PIECE_WORDS:
                    return _PIECE_WORDS[t[len(cw):]] + off
        if len(raw) == 1 and raw.lower() in _PIECE_LETTERS:                  # FEN letter: case = colour
            return _PIECE_LETTERS[raw.lower()] + (0 if raw.isupper() else 6)
    return None


# ---------------------------------------------------------------------------
# detections -> costs  (pure numpy: testable without any model)
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    label: int          # 1..12
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float


def square_confidences(dets, cell: float, flipped: bool = False) -> np.ndarray:
    """(64, 13): best confidence per (square, label). A piece belongs to the square its base is in:
    the horizontal centre of its box, and a point 'ANCHOR' squares above the box bottom (the middle
    of a short box, so flat 2D pieces land in the middle of their square while tall 3D pieces,
    whose box reaches up into the squares behind them, still count for the square they stand on)."""
    conf = np.zeros((64, N_LABELS), np.float32)
    for d in dets:
        h = d.y2 - d.y1
        ax = 0.5 * (d.x1 + d.x2)
        ay = d.y2 - min(0.5 * h, ANCHOR * cell)
        col, row = int(ax // cell), int(ay // cell)
        if not (0 <= row < 8 and 0 <= col < 8) or not (1 <= d.label <= 12):
            continue
        sq = BoardGeometry.square_of(row, col, flipped)
        conf[sq, d.label] = max(conf[sq, d.label], d.conf)
    return conf


def piece_costs(conf: np.ndarray) -> np.ndarray:
    """(64, 13) cost matrix (BoardTracker contract): label cost = 1 - confidence, empty = how sure
    the model is that *something* stands there. An unobserved square costs 0 as empty, 1 otherwise."""
    occ = conf.max(axis=1)
    D = 1.0 - conf
    D[:, 0] = occ
    return np.clip(D, 0.0, 1.0).astype(np.float32)


def colour_costs(conf: np.ndarray) -> np.ndarray:
    """(64, 3) empty / white / black class costs (SquareReader.class_costs format: best class ~0)."""
    cw, cb = conf[:, 1:7].max(axis=1), conf[:, 7:13].max(axis=1)
    c = np.stack([np.maximum(cw, cb), 1.0 - cw, 1.0 - cb], axis=1)
    c = c - c.min(axis=1, keepdims=True)
    return np.clip(c, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------
def _np(x) -> np.ndarray:
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def _predict(model, img: np.ndarray, imgsz: int):
    kw = {"conf": _cfg.conf, "imgsz": imgsz, "verbose": False}
    if _cfg.device:
        kw["device"] = _cfg.device
    return model.predict(img, **kw)[0]


_label_cache: dict = {}


def _labels_for(names) -> dict:
    key = id(names)
    hit = _label_cache.get(key)
    if hit is None or hit[0] is not names:
        mapping = {}
        items = names.items() if isinstance(names, dict) else enumerate(names)
        for idx, nm in items:
            lab = label_from_name(nm)
            if lab is not None:
                mapping[int(idx)] = lab
        _label_cache[key] = hit = (names, mapping)
    return hit[1]


def detections_from_result(result) -> list[Detection]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    mapping = _labels_for(result.names)
    xyxy, conf, cls = _np(boxes.xyxy), _np(boxes.conf), _np(boxes.cls).astype(int)
    out = []
    for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
        lab = mapping.get(int(k))
        if lab is not None:
            out.append(Detection(lab, float(c), float(x1), float(y1), float(x2), float(y2)))
    return out


def board_confidences(warped: np.ndarray, flipped: bool = False) -> np.ndarray | None:
    """(64, 13) confidences from the piece model on a warped board, or None -> use classical."""
    model = _load(_cfg.piece_weights) if _cfg.backend == "yolo" else None
    if model is None:
        return None
    try:
        res = _predict(model, warped, _cfg.piece_imgsz)
        return square_confidences(detections_from_result(res), warped.shape[1] / 8.0, flipped)
    except Exception:
        _warn_once(("infer", _cfg.piece_weights), "YOLO piece inference failed, using classical",
                   exc_info=True)
        return None


def piece_cost_matrix(warped: np.ndarray, flipped: bool = False) -> np.ndarray | None:
    """Drop-in for PieceClassifier.cost_matrix: (64, 13) or None (caller falls back)."""
    conf = board_confidences(warped, flipped)
    return None if conf is None else piece_costs(conf)


def colour_class_costs(warped: np.ndarray, flipped: bool = False) -> np.ndarray | None:
    """Drop-in for SquareReader.class_costs: (64, 3) or None (caller falls back)."""
    conf = board_confidences(warped, flipped)
    return None if conf is None else colour_costs(conf)


def _quad_from_polygon(poly: np.ndarray) -> np.ndarray | None:
    poly = np.asarray(poly, np.float32).reshape(-1, 2)
    if len(poly) < 4:
        return None
    hull = cv2.convexHull(poly)
    eps = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, eps, True).reshape(-1, 2)
    if len(approx) == 4:
        return approx.astype(np.float32)
    return cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)


def detect_corners(frame: np.ndarray) -> np.ndarray | None:
    """4x2 float32 outer corners of the playing area (any order) from the corner model, or None.
    Pose models: first 4 keypoints of the most confident detection. Segmentation models: the
    board mask reduced to a quadrilateral."""
    model = _load(_cfg.corner_weights) if _cfg.backend == "yolo" else None
    if model is None:
        return None
    try:
        r = _predict(model, frame, _cfg.corner_imgsz)
        best = 0
        boxes = getattr(r, "boxes", None)
        if boxes is not None and len(boxes) > 1:
            best = int(np.argmax(_np(boxes.conf)))
        quad = None
        kp = getattr(r, "keypoints", None)
        if kp is not None and len(kp.xy) > best and len(_np(kp.xy)[best]) >= 4:
            quad = _np(kp.xy)[best][:4].astype(np.float32)
        elif getattr(r, "masks", None) is not None and len(r.masks.xy) > best:
            quad = _quad_from_polygon(_np(r.masks.xy[best]))
        if quad is None or not np.all(np.isfinite(quad)):
            return None
        h, w = frame.shape[:2]
        if cv2.contourArea(cv2.convexHull(quad)) < 0.002 * w * h:       # a speck is not a board
            return None
        return quad
    except Exception:
        _warn_once(("corners", _cfg.corner_weights), "YOLO corner inference failed, using classical",
                   exc_info=True)
        return None
