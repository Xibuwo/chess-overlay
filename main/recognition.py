"""Piece recognition on the flattened board.

Skin-agnostic template matching: the user presses "Learn pieces" while the game
is in the starting position, and we store a foreground-mask template for each
piece. Every square estimates its own background colour from its border ring,
so light/dark squares, gradients and move-highlights don't matter.

Output is a cost matrix D[square, label] (lower = better match) which the
BoardTracker uses to pick the most plausible *legal* position.
"""
from __future__ import annotations

from pathlib import Path

import chess
import cv2
import numpy as np

import yolo
from geometry import BoardGeometry
from logsetup import get_logger

log = get_logger("recognition")

S = 32          # feature resolution per square
RING = 0.12     # border ring (fraction) used to estimate the square background
N_LABELS = 13   # 0 = empty, 1..6 white P N B R Q K, 7..12 black
_K3 = np.ones((3, 3), np.uint8)


def piece_label(piece: chess.Piece) -> int:
    return piece.piece_type + (6 if piece.color == chess.BLACK else 0)


def label_to_piece(label: int) -> chess.Piece:
    if label > 6:
        return chess.Piece(label - 6, chess.BLACK)
    return chess.Piece(label, chess.WHITE)


def board_labels(board: chess.Board) -> np.ndarray:
    out = np.zeros(64, dtype=np.int64)
    for sq, p in board.piece_map().items():
        out[sq] = piece_label(p)
    return out


def _square_feature(warped, row, col, cell, thr, top_extend) -> np.ndarray:
    x0, x1 = int(round(col * cell)), int(round((col + 1) * cell))
    y0 = max(0, int(round((row - top_extend) * cell)))
    y1 = int(round((row + 1) * cell))
    crop = warped[y0:y1, x0:x1]
    if crop.size == 0:
        return np.zeros(S * S, np.float32)
    crop = cv2.resize(crop, (S, S), interpolation=cv2.INTER_AREA)
    rw = max(2, int(S * RING))
    use_top = top_extend <= 0
    f32 = crop.astype(np.float32)
    parts = [f32[:, :rw].reshape(-1, 3), f32[:, -rw:].reshape(-1, 3), f32[-rw:, :].reshape(-1, 3)]
    if use_top:
        parts.append(f32[:rw, :].reshape(-1, 3))
    bg = np.median(np.concatenate(parts), axis=0)
    diff = np.abs(f32 - bg).max(axis=2)
    mask = (diff > thr).astype(np.uint8)
    mask[:, :rw] = 0
    mask[:, -rw:] = 0
    mask[-rw:, :] = 0
    if use_top:
        mask[:rw, :] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _K3)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return (mask * (0.25 + 0.75 * gray)).ravel().astype(np.float32)


class PieceClassifier:
    def __init__(self):
        self.templates: np.ndarray | None = None   # (N, S*S)
        self.tmpl_labels: np.ndarray | None = None  # (N,)
        self._scale: float | None = None

    @property
    def ready(self) -> bool:
        return self.templates is not None and len(self.templates) > 0

    @property
    def scale(self) -> float:
        """Average 'energy' of a piece template. Costs are divided by this so that
        one wrongly-explained piece costs ~1 regardless of piece size or skin; the
        tracker's thresholds are expressed in these units."""
        if self._scale is None:
            self._scale = max(1e-3, float(np.abs(self.templates).mean())) if self.ready else 1.0
        return self._scale

    def _features(self, warped, cell, flipped, thr, top_extend) -> np.ndarray:
        feats = np.zeros((64, S * S), np.float32)
        for sq in range(64):
            r, c = BoardGeometry.cell_of(sq, flipped)
            feats[sq] = _square_feature(warped, r, c, cell, thr, top_extend)
        return feats

    # ---- learning ---------------------------------------------------------
    def learn(self, warped, cell, flipped, thr=34, top_extend=0.0) -> tuple[bool, str]:
        feats = self._features(warped, cell, flipped, thr, top_extend)
        start = chess.Board()
        # sanity: the four middle ranks must look empty
        mid = [sq for sq in range(64) if start.piece_at(sq) is None]
        if top_extend > 0:  # 3D mode: a square's crop reaches into the square above it on screen
            cell_sq = {BoardGeometry.cell_of(sq, flipped): sq for sq in range(64)}
            mid = [sq for sq in mid
                   if start.piece_at(cell_sq.get((BoardGeometry.cell_of(sq, flipped)[0] - 1,
                                                  BoardGeometry.cell_of(sq, flipped)[1]), -1)) is None]
        empty_level = float(feats[mid].mean())
        if empty_level > 0.06:
            return False, ("Empty squares don't look empty - re-calibrate the corners, and make "
                           "sure the board is in the starting position with the right orientation.")
        tmpl, labels = [], []
        for sq, piece in start.piece_map().items():
            f = feats[sq].reshape(S, S)
            if float((f > 0).mean()) < 0.03:
                return False, f"Could not see the {piece.symbol()} on {chess.square_name(sq)}."
            for dx, dy in ((0, 0), (2, 0), (-2, 0), (0, 2), (0, -2)):
                tmpl.append(np.roll(np.roll(f, dx, axis=1), dy, axis=0).ravel())
                labels.append(piece_label(piece))
        self.templates = np.asarray(tmpl, np.float32)
        self.tmpl_labels = np.asarray(labels, np.int64)
        self._scale = None
        return True, f"Learned {len(start.piece_map())} pieces."

    # ---- inference --------------------------------------------------------
    def cost_matrix(self, warped, cell, flipped, thr=34, top_extend=0.0) -> np.ndarray:
        if yolo.piece_backend_active():            # optional YOLO backend; None = fall back to classical
            D = yolo.piece_cost_matrix(warped, flipped)
            if D is not None:
                return D
        feats = self._features(warped, cell, flipped, thr, top_extend)
        D = np.full((64, N_LABELS), 1.0, np.float32)
        D[:, 0] = feats.mean(axis=1)  # distance to an all-background square
        for i in range(len(self.templates)):
            d = np.abs(feats - self.templates[i]).mean(axis=1)
            lab = int(self.tmpl_labels[i])
            np.minimum(D[:, lab], d, out=D[:, lab])
        return D / self.scale

    # ---- persistence ------------------------------------------------------
    def save(self, path: Path) -> None:
        if not self.ready:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, templates=self.templates, labels=self.tmpl_labels)
        except Exception:
            log.warning("could not save piece templates to %s", path, exc_info=True)

    def load(self, path: Path) -> None:
        try:
            z = np.load(path)
            self.templates = z["templates"].astype(np.float32)
            self.tmpl_labels = z["labels"].astype(np.int64)
            self._scale = None
        except FileNotFoundError:
            self.templates = self.tmpl_labels = None      # first run: nothing learned yet
        except Exception:
            log.warning("could not load piece templates from %s", path, exc_info=True)
            self.templates = self.tmpl_labels = None