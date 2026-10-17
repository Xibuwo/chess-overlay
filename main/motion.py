"""'Follow moves' tracking for 3D / skinned boards.

Instead of recognising every piece (hard when pieces are tall 3D models that overlap
the squares around them), keep a plain 2D model of the game (a python-chess Board) and
only watch *which squares changed*. Each square is described by a "footprint" feature
(edges + contrast against its own background, lower part of the square only, so a tall
piece's head leaning into the square above is ignored). After the picture has settled,
the legal move (or two) that best explains the changed squares is played on the model.

Assumes the game starts from the standard position: press "Reset game" (or Learn
pieces) with the board in its starting position. A return to the starting look is
detected automatically as a new game.
"""
from __future__ import annotations

import chess
import cv2
import numpy as np

from geometry import BoardGeometry

F = 12  # per-square feature resolution
_OCC = np.array([chess.Board().piece_at(sq) is not None for sq in range(64)])


def compute_feats(warped: np.ndarray, cell: float, flipped: bool) -> np.ndarray:
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy) / 255.0
    out = np.zeros((64, 2 * F * F), np.float32)
    for sq in range(64):
        r, c = BoardGeometry.cell_of(sq, flipped)
        x0, x1 = int(round((c + 0.10) * cell)), int(round((c + 0.90) * cell))
        y0, y1 = int(round((r + 0.30) * cell)), int(round((r + 0.94) * cell))
        if x1 <= x0 or y1 <= y0:
            continue
        m = cv2.resize(mag[y0:y1, x0:x1], (F, F), interpolation=cv2.INTER_AREA)
        g = cv2.resize(gray[y0:y1, x0:x1], (F, F), interpolation=cv2.INTER_AREA)
        # the square's own background: its left/right side strips (a piece sits in the middle)
        sx0, sx1 = int(round(c * cell)), int(round((c + 1) * cell))
        w = max(1, int(round(0.10 * cell)))
        strips = np.concatenate([gray[y0:y1, sx0:sx0 + w].ravel(), gray[y0:y1, sx1 - w:sx1].ravel()])
        bg = float(np.median(strips)) if strips.size else 0.0
        dev = np.clip(np.abs(g - bg) / 255.0, 0.0, 1.0)
        out[sq] = np.concatenate([m.ravel(), dev.ravel()])
    return out


def _weights(board: chess.Board, m: chess.Move) -> dict[int, float]:
    """squares that must change for move m, with how strongly they are expected to."""
    w = {m.from_square: 1.0}
    ep = board.is_en_passant(m)
    w[m.to_square] = 0.5 if (board.is_capture(m) and not ep) else 1.0  # piece swap = weak change
    if ep:
        w[chess.square(chess.square_file(m.to_square), chess.square_rank(m.from_square))] = 1.0
    elif board.is_castling(m):
        rank = chess.square_rank(m.from_square)
        if chess.square_file(m.to_square) > chess.square_file(m.from_square):
            w[chess.square(7, rank)] = 1.0
            w[chess.square(5, rank)] = 1.0
        else:
            w[chess.square(0, rank)] = 1.0
            w[chess.square(3, rank)] = 1.0
    return w


class MotionTracker:
    CONFIRM = 2          # identical readings in a row before a move is accepted
    SETTLE = 0.20        # max frame-to-frame change (normalised) to count as "picture is still"
    CHANGED = 0.30       # a square counts as changed above this
    LOST_FRAMES = 10

    def __init__(self):
        self.reset()

    def reset(self):
        self.board = chess.Board()
        self.ref = None
        self.start_ref = None
        self.prev = None
        self.scale = 1.0
        self._pend = None
        self._pend_n = 0
        self._lost = 0
        self.note = ""

    # ------------------------------------------------------------------
    def _candidates(self, d: np.ndarray, plies: int):
        dc = np.minimum(d, 1.0)
        total = float(dc.sum())
        best = None   # (net, need, moves)
        second = -1e9

        def consider(moves, w):
            nonlocal best, second
            strong = [sq for sq, v in w.items() if v >= 1.0]
            if any(d[sq] <= self.CHANGED for sq in strong):
                return
            if any(d[mv.from_square] <= self.CHANGED for mv in moves):
                return
            expl = sum(v * dc[sq] for sq, v in w.items())
            unexp = total - sum(dc[sq] for sq in w)
            net = expl - 0.8 * unexp
            need = 0.6 * sum(w.values())
            if best is None or net > best[0]:
                if best is not None:
                    second = max(second, best[0])
                best = (net, need, moves)
            else:
                second = max(second, net)

        def moves_of(b):
            for m in b.legal_moves:
                if m.promotion and m.promotion != chess.QUEEN:
                    continue  # promotions are assumed to be queens
                yield m

        for m1 in moves_of(self.board):
            w1 = _weights(self.board, m1)
            if plies == 1:
                consider([m1], w1)
                continue
            b2 = self.board.copy(stack=False)
            b2.push(m1)
            for m2 in moves_of(b2):
                w = dict(w1)
                for sq, v in _weights(b2, m2).items():
                    w[sq] = max(w.get(sq, 0.0), v)
                consider([m1, m2], w)
        return best, second

    # ------------------------------------------------------------------
    def update(self, feats: np.ndarray) -> bool:
        """feats: compute_feats() of the current frame. Returns True when the model changed."""
        if self.ref is None:  # first frame after a reset: the screen shows the starting position
            diff = float(feats[_OCC].mean() - feats[~_OCC].mean())
            self.scale = max(diff, 0.02)
            self.ref = feats.copy()
            self.start_ref = feats.copy()
            self.prev = feats
            self.note = "Following moves from the starting position."
            return True

        step = np.abs(feats - self.prev).mean(axis=1) / self.scale
        self.prev = feats
        if step.max() > self.SETTLE:   # pieces still moving / animation
            self._pend, self._pend_n = None, 0
            return False

        d = np.abs(feats - self.ref).mean(axis=1) / self.scale
        if d.max() < self.CHANGED:
            self._pend, self._pend_n, self._lost = None, 0, 0
            return False

        if self.board.move_stack:   # back at the starting look -> new game
            ds = np.abs(feats - self.start_ref).mean(axis=1) / self.scale
            if ds.max() < self.CHANGED:
                self.board = chess.Board()
                self.ref = feats.copy()
                self._pend, self._pend_n, self._lost = None, 0, 0
                self.note = "New game detected."
                return True

        best, second = self._candidates(d, 1)
        if best is None or best[0] < best[1]:
            best, second = self._candidates(d, 2)
        if best is None or best[0] < best[1] or best[0] - second < 0.15:
            self._pend, self._pend_n = None, 0
            if d.max() > 0.6:
                self._lost += 1
                if self._lost >= self.LOST_FRAMES:
                    self.note = "Lost track - press 'Reset game' with the board at the starting position."
            return False

        self._lost = 0
        key = tuple(m.uci() for m in best[2])
        self._pend_n = self._pend_n + 1 if key == self._pend else 1
        self._pend = key
        if self._pend_n < self.CONFIRM:
            return False
        for m in best[2]:
            self.board.push(m)
        self.ref = feats.copy()
        self._pend, self._pend_n = None, 0
        self.note = ""
        return True
