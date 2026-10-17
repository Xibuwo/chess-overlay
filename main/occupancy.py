"""Skin-agnostic square reader: empty / white piece / black piece.

Works for flat 2D boards and for tall 3D pieces, without piece templates and without
any per-game setup. It never has to know *which* piece is on a square - the legal-move
tracker (tracker.py) works that out from the starting position - it only has to tell
"empty", "white piece" and "black piece" apart, and a legal move changes exactly those
labels on exactly the right squares.

How it copes with the hard cases (a black piece on a dark square, a white piece on a
light square, highlighted squares, dots/outlines that mark legal moves):

* every square is compared with *its own* background (the left/right strips of the
  square), so lighting gradients, move highlights and outlines don't matter;
* instead of fixed thresholds it keeps a small memory of labelled examples (k-nearest
  neighbours on a handful of colour/contrast/edge numbers). It is seeded from the
  starting position, where every piece's colour and square colour is known, and keeps
  learning from positions the tracker has confirmed.
"""
from __future__ import annotations

import chess
import cv2
import numpy as np

import yolo
from geometry import BoardGeometry

N_FEAT = 13
EMPTY, WHITE, BLACK = 0, 1, 2
MAX_PER_CLASS = 360


def _lab(warped: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    return lab


def extract(warped: np.ndarray, cell: float, flipped: bool) -> np.ndarray:
    """(64, N_FEAT) features, indexed by chess square."""
    lab = _lab(warped)
    L = lab[..., 0]
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx * gx + gy * gy) / 8.0
    out = np.zeros((64, N_FEAT), np.float32)

    def rng(x0, f0, f1):
        return int(round(x0 + f0 * cell)), int(round(x0 + f1 * cell))

    # all squares share the same box sizes (cell is an integer in practice), so gather the
    # boxes into (64, h, w) stacks and let numpy do the percentiles for every square at once
    cells = [BoardGeometry.cell_of(sq, flipped) for sq in range(64)]

    def stack(img, fx0, fx1, fy0, fy1):
        return np.stack([img[slice(*rng(r * cell, fy0, fy1)), slice(*rng(c * cell, fx0, fx1))]
                         for r, c in cells])

    try:
        sl = stack(lab, 0.09, 0.20, 0.15, 0.80)
        sr = stack(lab, 0.80, 0.91, 0.15, 0.80)
        core = stack(lab, 0.25, 0.75, 0.20, 0.78)
        edg = stack(edge, 0.25, 0.75, 0.20, 0.78)
    except ValueError:      # ragged boxes (non-integer cell): slow, exact path
        return _extract_slow(lab, edge, cell, flipped)
    strips = np.concatenate([sl.reshape(64, -1, 3), sr.reshape(64, -1, 3)], axis=1)
    bg = np.median(strips, axis=1)                                   # (64, 3)
    core = core.reshape(64, -1, 3)
    dev = np.linalg.norm(core - bg[:, None, :], axis=2)              # (64, n)
    cl = core[..., 0]
    q = np.percentile(dev, [50, 75, 90], axis=1)
    cq = np.percentile(cl, [15, 85], axis=1)
    e = edg.reshape(64, -1)
    eq = np.percentile(e, 90, axis=1)
    mask = dev > 7.0
    for sq in range(64):
        m = mask[sq]
        if int(m.sum()) > 15:
            ml, pl = float(np.median(cl[sq][m])), float(np.percentile(cl[sq][m], 85))
        else:
            ml = pl = float(bg[sq, 0])
        out[sq] = (q[0, sq], q[1, sq], q[2, sq], m.mean(), ml, pl, bg[sq, 0], ml - bg[sq, 0],
                   pl - bg[sq, 0], cq[0, sq], cq[1, sq], float(e[sq].mean()), eq[sq])
    return out


def _extract_slow(lab, edge, cell, flipped):
    out = np.zeros((64, N_FEAT), np.float32)
    for sq in range(64):
        r, c = BoardGeometry.cell_of(sq, flipped)
        x0, y0 = c * cell, r * cell

        def box(fx0, fx1, fy0, fy1):
            return (slice(int(round(y0 + fy0 * cell)), int(round(y0 + fy1 * cell))),
                    slice(int(round(x0 + fx0 * cell)), int(round(x0 + fx1 * cell))))

        ys, xs = box(0.09, 0.20, 0.15, 0.80)
        ys2, xs2 = box(0.80, 0.91, 0.15, 0.80)
        strips = np.concatenate([lab[ys, xs].reshape(-1, 3), lab[ys2, xs2].reshape(-1, 3)])
        bg = np.median(strips, axis=0)
        cy, cx = box(0.25, 0.75, 0.20, 0.78)
        core = lab[cy, cx]
        dev = np.linalg.norm(core - bg, axis=2)
        cl = core[..., 0]
        m = dev > 7.0
        if int(m.sum()) > 15:
            ml, pl = float(np.median(cl[m])), float(np.percentile(cl[m], 85))
        else:
            ml = pl = float(bg[0])
        out[sq] = (np.percentile(dev, 50), np.percentile(dev, 75), np.percentile(dev, 90), m.mean(),
                   ml, pl, bg[0], ml - bg[0], pl - bg[0],
                   np.percentile(cl, 15), np.percentile(cl, 85),
                   float(edge[cy, cx].mean()), float(np.percentile(edge[cy, cx], 90)))
    return out


def highlight_squares(warped: np.ndarray, cell: float, thr: float = 10.0) -> frozenset:
    """Squares that carry a yellow/gold 'last move' tint (Outlast Trials, chess.com, lichess,
    most chess apps highlight the from- and to-square of the last move this way).

    Each square's own background (its left and right strips, where a piece rarely stands) is
    compared with the typical background of the same square colour, so the board's own tint
    cancels out. A real highlight tints *both* strips almost completely; a gold piece base or a
    neighbouring piece leaning in only touches part of one. Returns squares (python-chess
    numbering, white at the bottom of `warped`), or an empty set when nothing - or too much -
    is highlighted."""
    lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 1:] -= 128.0
    B = lab[..., 2]
    med = np.zeros(64, np.float32)
    strips = []
    for sq in range(64):
        r, c = BoardGeometry.cell_of(sq, False)
        y0, y1 = int(round((r + 0.10) * cell)), int(round((r + 0.95) * cell))
        a = B[y0:y1, int(round((c + 0.05) * cell)):int(round((c + 0.18) * cell))].ravel()
        b = B[y0:y1, int(round((c + 0.82) * cell)):int(round((c + 0.95) * cell))].ravel()
        strips.append((a, b))
        med[sq] = np.median(np.concatenate([a, b]))
    par = np.array([(chess.square_file(q) + chess.square_rank(q)) & 1 for q in range(64)])
    out = []
    for g in (0, 1):
        idx = np.nonzero(par == g)[0]
        base = float(np.median(med[idx]))
        for q in idx:
            a, b = strips[q]
            fa, fb = float((a - base >= 0.7 * thr).mean()), float((b - base >= 0.7 * thr).mean())
            if min(fa, fb) >= 0.75 and med[q] - base >= thr:
                out.append(int(q))
    return frozenset(out) if 1 <= len(out) <= 4 else frozenset()


def occupied_guess(f: np.ndarray) -> np.ndarray:
    """Model-free 'is something standing here' per square (used only to spot a starting position)."""
    return (f[:, 3] >= 0.30) | (f[:, 1] >= 6.0)


def start_check(f: np.ndarray, flipped_cells: bool = False):
    """Does this (64-square, *screen-oriented*, flipped=False) reading look like a starting
    position? Returns (ok, white_at_bottom) where rows 0-1 are the far side.

    Uses squares indexed by chess square for the unflipped view: ranks 1-2 are the
    bottom of the screen."""
    occ = occupied_guess(f)
    sq = np.arange(64)
    rank = sq >> 3
    near = occ[rank <= 1]           # screen bottom (ranks 1-2)
    far = occ[rank >= 6]
    mid = occ[(rank >= 2) & (rank <= 5)]
    if near.mean() < 0.85 or far.mean() < 0.85 or mid.mean() > 0.08:
        return False, None
    lum_near = float(np.median(f[(rank <= 1) & occ, 4]))
    lum_far = float(np.median(f[(rank >= 6) & occ, 4]))
    if abs(lum_near - lum_far) < 6.0:
        return False, None
    # each side must be consistent: most pieces on the same side of the midpoint
    mid_l = 0.5 * (lum_near + lum_far)
    n_ok = float(((f[(rank <= 1) & occ, 4] > mid_l) == (lum_near > lum_far)).mean())
    f_ok = float(((f[(rank >= 6) & occ, 4] > mid_l) == (lum_far > lum_near)).mean())
    if n_ok < 0.75 or f_ok < 0.75:
        return False, None
    return True, lum_near > lum_far


def start_labels_classes() -> np.ndarray:
    out = np.zeros(64, np.int64)
    for s, p in chess.Board().piece_map().items():
        out[s] = WHITE if p.color == chess.WHITE else BLACK
    return out


class SquareReader:
    def __init__(self):
        self.reset()

    def reset(self):
        self.X = [np.zeros((0, N_FEAT), np.float32) for _ in range(3)]
        self.sd = np.ones(N_FEAT, np.float32)
        self.tau = 1.3
        self.ready = False
        self._rng = np.random.default_rng(1)
        self._yolo_f = None        # features object the cached YOLO class costs belong to
        self._yolo_c = None

    # ---- learning -------------------------------------------------------
    def learn_start(self, f: np.ndarray) -> None:
        """f: features of the starting position (white at the bottom, flipped=False)."""
        cls = start_labels_classes()
        self.sd = f.std(axis=0) + 1e-3
        self.sd = np.maximum(self.sd, 0.15 * np.abs(f).mean(axis=0) + 1e-3)
        self.X = [np.zeros((0, N_FEAT), np.float32) for _ in range(3)]
        for k in range(3):
            self.X[k] = (f[cls == k] / self.sd).astype(np.float32)
        self.ready = True

    def _dist(self, z: np.ndarray, k: int) -> np.ndarray:
        X = self.X[k]
        if len(X) == 0:
            return np.full(len(z), 9.0, np.float32)
        d = np.sqrt(((np.clip(z[:, None, :] - X[None, :, :], -8, 8)) ** 2).sum(axis=2))
        n = min(3, d.shape[1])
        return np.sort(d, axis=1)[:, :n].mean(axis=1)

    def class_costs(self, f: np.ndarray) -> np.ndarray:
        """(64, 3) cost of reading each square as empty / white / black; best class ~0, others up to 1."""
        if self._yolo_c is not None and f is self._yolo_f:     # same frame the YOLO backend just read
            return self._yolo_c
        z = (f / self.sd).astype(np.float32)
        d = np.stack([self._dist(z, k) for k in range(3)], axis=1)
        c = (d - d.min(axis=1, keepdims=True)) / self.tau
        return np.clip(c, 0.0, 1.0).astype(np.float32)

    def cost_matrix(self, f: np.ndarray, warped: np.ndarray | None = None) -> np.ndarray:
        """(64, 13) cost matrix in the format BoardTracker expects (label 0 empty, 1-6 white, 7-12 black).

        `warped` (the flattened board the features came from) is only used by the optional YOLO
        backend; without it, or without usable weights, the classical reader below runs unchanged."""
        self._yolo_f = self._yolo_c = None
        if warped is not None and yolo.piece_backend_active():
            c = yolo.colour_class_costs(warped, False)
            if c is not None:
                self._yolo_f, self._yolo_c = f, c
        c = self.class_costs(f)
        D = np.empty((64, 13), np.float32)
        D[:, 0] = c[:, EMPTY]
        D[:, 1:7] = c[:, WHITE:WHITE + 1]
        D[:, 7:13] = c[:, BLACK:BLACK + 1]
        return D

    def read(self, f: np.ndarray) -> np.ndarray:
        return self.class_costs(f).argmin(axis=1)

    def adapt(self, f: np.ndarray, board: chess.Board) -> int:
        """Learn from a frame whose position the tracker is sure about. Returns #samples added."""
        if not self.ready or (self._yolo_c is not None and f is self._yolo_f):
            return 0                # YOLO read this frame: the classical memory has nothing to learn
        cls = np.zeros(64, np.int64)
        for s, p in board.piece_map().items():
            cls[s] = WHITE if p.color == chess.WHITE else BLACK
        z = (f / self.sd).astype(np.float32)
        added = 0
        for k in range(3):
            sel = np.nonzero(cls == k)[0]
            if len(sel) == 0:
                continue
            dk = self._dist(z[sel], k)          # how well the sample is already covered by its own class
            new = sel[dk > 0.45]
            if len(new) == 0:
                continue
            X = np.concatenate([self.X[k], z[new]])
            if len(X) > MAX_PER_CLASS:
                keep = np.sort(self._rng.choice(len(X), MAX_PER_CLASS, replace=False))
                keep = np.union1d(keep, np.arange(min(32, len(X))))   # never forget the seed examples
                X = X[keep]
            self.X[k] = X
            added += len(new)
        return added
