"""Automatic board finder: works for flat 2D boards and tilted 3D boards.

No templates and no clicking. It looks for the *checkerboard lattice*: every
junction where four squares meet is a saddle point of the image (dark/light/dark/
light around it). Those junctions are found with a Hessian detector, grown into a
perspective-correct lattice (so pieces that hide some junctions don't matter), and
the 8x8 window of the lattice that really looks like a chessboard is chosen.

`BoardLocator` runs it frame after frame and follows the board when the camera
moves (first-person games such as The Outlast Trials move the board every frame).
All coordinates in/out are pixels of the frame that was passed in.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

import yolo
from logsetup import get_logger

log = get_logger("autodetect")

WORK_W = 1280            # saddle detection happens at (at most) this width
CELL_MIN, CELL_MAX = 9.0, 150.0   # plausible square size in WORK_W pixels
MIN_INLIERS = 24         # lattice junctions needed (out of 81) to believe a board


@dataclass
class BoardFit:
    corners: np.ndarray      # 4x2, TL, TR, BR, BL as seen on screen, frame pixels
    score: float             # 0..1
    inliers: int             # lattice junctions confirmed by the image (max 81)
    H: np.ndarray            # board lattice (0..8, 0..8) -> frame pixels
    cell_px: float           # approximate square size in frame pixels
    pol: int = 0             # checker polarity of lattice cell (0,0): +1 light, -1 dark, 0 unknown


# --------------------------------------------------------------------------
# junction (saddle) detector
# --------------------------------------------------------------------------
def find_saddles(gray: np.ndarray, sigma: float = 2.5, rel_thr: float = 0.06):
    """gray: float32 0..1. Returns (N,2) float32 junction positions (x, y), strongest first."""
    b = cv2.GaussianBlur(gray, (0, 0), sigma)
    ixx = cv2.Sobel(b, cv2.CV_32F, 2, 0, ksize=3)
    iyy = cv2.Sobel(b, cv2.CV_32F, 0, 2, ksize=3)
    ixy = cv2.Sobel(b, cv2.CV_32F, 1, 1, ksize=3)
    r = ixy * ixy - ixx * iyy            # > 0 at saddles
    r[r < 0] = 0
    ref = float(np.percentile(r[r > 0], 99.7)) if np.any(r > 0) else 0.0
    if ref <= 0:
        return np.zeros((0, 2), np.float32)
    k = max(5, int(round(3.2 * sigma)) | 1)
    dil = cv2.dilate(r, np.ones((k, k), np.uint8))
    pk = (r == dil) & (r > rel_thr * ref)
    ys, xs = np.nonzero(pk)
    if len(xs) == 0:
        return np.zeros((0, 2), np.float32)
    order = np.argsort(-r[ys, xs])[:1200]
    return np.stack([xs[order], ys[order]], axis=1).astype(np.float32)


# --------------------------------------------------------------------------
# lattice growing
# --------------------------------------------------------------------------
def _apply_h(H, ij):
    ij = np.asarray(ij, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(ij, H).reshape(-1, 2)


def _fit_h(ij, xy, thr):
    ij = np.asarray(ij, np.float32)
    xy = np.asarray(xy, np.float32)
    if len(ij) < 4:
        return None, None
    H, mask = cv2.findHomography(ij, xy, cv2.RANSAC, thr, maxIters=500, confidence=0.995)
    if H is None or mask is None:
        return None, None
    return H.astype(np.float64), mask.ravel().astype(bool)


def _local_cell(H, i, j):
    p = _apply_h(H, [(i, j), (i + 1, j), (i, j + 1)])
    return 0.5 * (np.linalg.norm(p[1] - p[0]) + np.linalg.norm(p[2] - p[0]))


def _snap(pts, pred, tol):
    """for each predicted point: index of nearest detected junction within tol (else -1)."""
    if len(pts) == 0:
        return np.full(len(pred), -1), np.full(len(pred), 1e9)
    d = np.linalg.norm(pts[None, :, :] - pred[:, None, :], axis=2)
    idx = d.argmin(axis=1)
    dm = d[np.arange(len(pred)), idx]
    return np.where(dm <= tol, idx, -1), dm


def _grow(pts, seed_ij, seed_idx, max_iter=14, span=10):
    """Grow a lattice from a seed. Returns (H, {(i,j): idx}) in *seed-relative* lattice coords."""
    known = {tuple(ij): int(ix) for ij, ix in zip(seed_ij, seed_idx)}
    H = None
    for _ in range(max_iter):
        ij = np.array(list(known.keys()), np.float64)
        xy = pts[list(known.values())].astype(np.float64)
        H2, mask = _fit_h(ij, xy, thr=0.18 * max(8.0, _local_cell_guess(H, ij, xy)))
        if H2 is None:
            break
        H = H2
        if not mask.all():   # drop outliers
            keys = list(known.keys())
            known = {keys[k]: known[keys[k]] for k in range(len(keys)) if mask[k]}
            ij = np.array(list(known.keys()), np.float64)
        i0, j0 = ij.min(axis=0)
        i1, j1 = ij.max(axis=0)
        cand = [(i, j) for i in range(int(i0) - 1, int(i1) + 2) for j in range(int(j0) - 1, int(j1) + 2)
                if (i, j) not in known and abs(i) <= span and abs(j) <= span]
        if not cand:
            break
        pred = _apply_h(H, cand)
        cell = _local_cell(H, float(np.mean(ij[:, 0])), float(np.mean(ij[:, 1])))
        idx, _ = _snap(pts, pred, 0.22 * cell)
        used = set(known.values())
        added = 0
        for c, ix in zip(cand, idx):
            if ix >= 0 and int(ix) not in used:
                known[c] = int(ix)
                used.add(int(ix))
                added += 1
        if added == 0:
            break
    known = _prune(known)
    return H, known


def _prune(known):
    """Keep the dense core of a grown lattice: drop points with < 2 lattice neighbours and
    keep only the largest 4-connected blob (growth can leak onto clutter around the board)."""
    known = dict(known)
    for _ in range(4):
        bad = [k for k in known
               if sum(((k[0] + a, k[1] + b) in known) for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1))) < 2]
        if not bad:
            break
        for k in bad:
            del known[k]
    seen, best = set(), []
    for k in known:
        if k in seen:
            continue
        comp, stack = [], [k]
        seen.add(k)
        while stack:
            c = stack.pop()
            comp.append(c)
            for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (c[0] + a, c[1] + b)
                if n in known and n not in seen:
                    seen.add(n)
                    stack.append(n)
        if len(comp) > len(best):
            best = comp
    return {k: known[k] for k in best}


def _local_cell_guess(H, ij, xy):
    if H is not None:
        return _local_cell(H, float(np.mean(ij[:, 0])), float(np.mean(ij[:, 1])))
    d = np.linalg.norm(xy[1:] - xy[0], axis=1)
    return float(np.median(d)) if len(d) else 20.0


def _seeds(pts, max_seeds=400):
    """3x3 patches of junctions: each is (ij list, idx list). Cheap geometric pre-filter
    (a unit cell must have no junction at its centre or edge midpoints, otherwise it is a
    coarser or diagonal sub-lattice, not the square grid)."""
    n = len(pts)
    if n < 9:
        return []
    D = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    out = []
    K = 7
    for p in range(min(n, 500)):
        order = np.argsort(D[p])[1:K + 1]
        order = [q for q in order if CELL_MIN <= D[p, q] <= CELL_MAX]
        for a in range(len(order)):
            for b in range(a + 1, len(order)):
                q, r = order[a], order[b]
                u, v = pts[q] - pts[p], pts[r] - pts[p]
                lu, lv = np.linalg.norm(u), np.linalg.norm(v)
                if lu < 1e-3 or lv < 1e-3 or max(lu, lv) / min(lu, lv) > 1.7:
                    continue
                cosang = abs(float(u @ v) / (lu * lv))
                if cosang > 0.6:   # need roughly >= 53 degrees between the two directions
                    continue
                if np.sign(u[0] * v[1] - u[1] * v[0]) < 0:
                    u, v = v, u
                    q, r = r, q
                tol = 0.22 * 0.5 * (lu + lv)
                # junction pattern of the 3x3 patch under an affine guess
                ij, idx, ok = [], [], True
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        pred = pts[p] + di * u + dj * v
                        d = np.linalg.norm(pts - pred, axis=1)
                        m = int(d.argmin())
                        if d[m] > tol:
                            ok = False
                            break
                        ij.append((di, dj))
                        idx.append(m)
                    if not ok:
                        break
                if not ok or len(set(idx)) < 9:
                    continue
                # reject sub-lattices: there must be no junction at the centre of a cell
                cen = pts[p] + 0.5 * (u + v)
                if np.linalg.norm(pts - cen, axis=1).min() < 0.2 * 0.5 * (lu + lv):
                    continue
                out.append((ij, idx))
                if len(out) >= max_seeds:
                    return out
    return out


# --------------------------------------------------------------------------
# choosing the 8x8 window of the lattice
# --------------------------------------------------------------------------
def _cell_means(gray, H, i_range, j_range, res=14):
    """Mean grey of the central part of every lattice cell, via a rectified patch."""
    i0, i1 = i_range
    j0, j1 = j_range
    ni, nj = i1 - i0, j1 - j0
    S = np.array([[res, 0, -i0 * res], [0, res, -j0 * res], [0, 0, 1]], np.float64)  # lattice -> patch px
    # patch(u,v) = gray(H * S^-1 * (u,v))
    M = H @ np.linalg.inv(S)
    patch = cv2.warpPerspective(gray, M, (ni * res, nj * res), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    # WARP_INVERSE_MAP => M maps destination -> source, which is what we built
    cells = patch.reshape(nj, res, ni, res)
    # Colour of a cell = its four corner patches, not its centre: pieces stand in the centre,
    # the corners show the square itself, so a board full of pieces still reads as a checkerboard.
    lo, hi = slice(int(res * 0.14), int(res * 0.36)), slice(int(res * 0.64), int(res * 0.86))
    m = (cells[:, lo, :, lo].mean(axis=(1, 3)) + cells[:, lo, :, hi].mean(axis=(1, 3)) +
         cells[:, hi, :, lo].mean(axis=(1, 3)) + cells[:, hi, :, hi].mean(axis=(1, 3))) / 4.0
    return m, patch   # shape (nj, ni): [j, i]


def _best_window(gray, H, known_ij, shape):
    """Pick (i0, j0) so that lattice cells i0..i0+7, j0..j0+7 are the chessboard."""
    h, w = shape[:2]
    ij = np.array(list(known_ij), np.float64)
    imin, jmin = ij.min(axis=0)
    imax, jmax = ij.max(axis=0)
    i_lo, i_hi = int(imin) - 2, int(imax) + 2
    j_lo, j_hi = int(jmin) - 2, int(jmax) + 2
    if i_hi - i_lo < 8 or j_hi - j_lo < 8:
        return None
    means, _ = _cell_means(gray, H, (i_lo, i_hi), (j_lo, j_hi))
    ni, nj = i_hi - i_lo, j_hi - j_lo
    ii, jj = np.meshgrid(np.arange(ni), np.arange(nj))
    sign = np.where(((ii + jj) & 1) == 0, 1.0, -1.0)
    # which cells are inside the image at all?
    corners = np.array([[x, y] for x in range(i_lo, i_hi + 1) for y in range(j_lo, j_hi + 1)], np.float64)
    cp = _apply_h(H, corners).reshape(ni + 1, nj + 1, 2)
    inside = ((cp[..., 0] > -2) & (cp[..., 0] < w + 2) & (cp[..., 1] > -2) & (cp[..., 1] < h + 2))
    cell_in = inside[:-1, :-1] & inside[1:, :-1] & inside[:-1, 1:] & inside[1:, 1:]   # [i, j]
    cell_in = cell_in.T                                                               # [j, i]
    # junction support: lattice junctions confirmed per cell corner
    support = np.zeros((nj + 1, ni + 1), np.float32)
    for (i, j) in known_ij:
        if i_lo <= i <= i_hi and j_lo <= j <= j_hi:
            support[j - j_lo, i - i_lo] = 1.0
    best, best_s = None, -1e9
    for j0 in range(0, nj - 7):
        for i0 in range(0, ni - 7):
            win = means[j0:j0 + 8, i0:i0 + 8]
            if not cell_in[j0:j0 + 8, i0:i0 + 8].all():
                continue
            sg = sign[j0:j0 + 8, i0:i0 + 8]
            m = win.mean()
            c = (win - m) * sg
            # robust checker score: fraction of cells agreeing with the dominant polarity
            pol = 1.0 if c.sum() >= 0 else -1.0
            agree = float(((c * pol) > 0.02).mean())
            mag = float(np.clip(c * pol, -0.3, 0.3).mean())
            sup = float(support[j0:j0 + 9, i0:i0 + 9].mean())   # all 81 lattice junctions
            if sup < 0.3:
                continue
            s = 1.0 * agree + 3.0 * mag + 2.5 * sup
            if s > best_s:
                best_s, best = s, (i0 + i_lo, j0 + j_lo, agree, sup)
    return best + (best_s,) if best else None


def _polarity(gray, H) -> int:
    """+1 / -1: is lattice cell (0,0) lighter / darker than its neighbours? 0 when the picture
    does not look like a clean checkerboard under H. A lattice that has slid by an odd number of
    cells flips this sign, which is how tracking drift is caught."""
    means, _ = _cell_means(gray, H, (0, 8), (0, 8))
    ii, jj = np.meshgrid(np.arange(8), np.arange(8))
    sign = np.where(((ii + jj) & 1) == 0, 1.0, -1.0)
    c = (means - means.mean()) * sign
    mag = float(np.clip(c, -0.3, 0.3).mean())
    if abs(mag) < 0.04:
        return 0
    return 1 if mag > 0 else -1


def _order_corners(c):
    """4 corners (any order) -> TL, TR, BR, BL as seen on screen (clockwise, start top-left)."""
    c = np.asarray(c, np.float32).reshape(4, 2)
    cen = c.mean(axis=0)
    ang = np.arctan2(c[:, 1] - cen[1], c[:, 0] - cen[0])
    c = c[np.argsort(ang)]            # clockwise on screen (y down): increasing angle
    # start with the corner closest to the top-left (angle nearest -135 deg)
    k = int(np.argmin(np.abs(np.arctan2(c[:, 1] - cen[1], c[:, 0] - cen[0]) + 0.75 * np.pi)))
    return np.roll(c, -k, axis=0)


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
def _prepare(frame):
    h, w = frame.shape[:2]
    scale = min(1.0, WORK_W / float(w))
    small = cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA) \
        if scale < 1.0 else frame
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return gray, scale


def _finish(gray, scale, H, known_ij, pts):
    win = _best_window(gray, H, known_ij.keys(), gray.shape)
    if win is None:
        return None
    i0, j0, agree, sup, s = win
    if agree < 0.62:
        return None
    # re-express H in board lattice coords (0..8) and re-fit using every confirmed junction in the window
    ij = np.array([(i - i0, j - j0) for (i, j) in known_ij.keys()], np.float64)
    xy = np.array([pts[k] for k in known_ij.values()], np.float64)
    keep = (ij[:, 0] >= 0) & (ij[:, 0] <= 8) & (ij[:, 1] >= 0) & (ij[:, 1] <= 8)
    Hb = H @ np.array([[1, 0, i0], [0, 1, j0], [0, 0, 1]], np.float64)
    if keep.sum() >= 8:
        H2, mask = _fit_h(ij[keep], xy[keep], thr=2.0)
        if H2 is not None:
            Hb = H2
    corners = _apply_h(Hb, [(0, 0), (8, 0), (8, 8), (0, 8)])
    cell = _local_cell(Hb, 4, 4)
    inl = int(keep.sum())
    # inside the frame?
    return corners, Hb, inl, agree, sup, cell


def detect_board(frame: np.ndarray, sigmas=(2.5, 1.6, 4.0)) -> BoardFit | None:
    """Find the chessboard in a BGR frame. Returns None when there is no convincing board."""
    gray, scale = _prepare(frame)
    for sg in sigmas:
        pts = find_saddles(gray, sg)
        if len(pts) < 20:
            continue
        best = None
        done = set()
        for seed_ij, seed_idx in _seeds(pts):
            key = frozenset(seed_idx)
            if key <= done:
                continue
            H, known = _grow(pts, seed_ij, seed_idx)
            if H is None or len(known) < MIN_INLIERS:
                done |= set(seed_idx)
                continue
            if best is None or len(known) > len(best[1]):
                best = (H, known)
            done |= set(known.values())
            if len(known) >= 60:
                break
        if best is None:
            continue
        res = _finish(gray, scale, best[0], best[1], pts)
        if res is None:
            continue
        corners, Hb, inl, agree, sup, cell = res
        if inl < 16:
            continue
        s = float(min(1.0, 0.45 * agree + 0.55 * min(1.0, inl / 49.0)))
        corners = corners / scale
        Hf = np.diag([1.0 / scale, 1.0 / scale, 1.0]) @ Hb
        return BoardFit(_order_corners(corners), s, inl, Hf, cell / scale, _polarity(gray, Hb))
    return None


def fit_from_corners(frame: np.ndarray, corners: np.ndarray, score: float = 0.9,
                     ordered: bool = False) -> BoardFit:
    """BoardFit from the 4 outer corners of the 8x8 area (e.g. from the YOLO corner model).
    `ordered=True` keeps the given corner order (already TL, TR, BR, BL-like and aligned)."""
    c = np.asarray(corners, np.float32).reshape(4, 2)
    if not ordered:
        c = _order_corners(c)
    lattice = np.float32([[0, 0], [8, 0], [8, 8], [0, 8]])
    H = cv2.getPerspectiveTransform(lattice, c).astype(np.float64)
    gray, scale = _prepare(frame)
    pol = _polarity(gray, np.diag([scale, scale, 1.0]) @ H)
    return BoardFit(c, score, 81, H, _local_cell(H, 4, 4), pol)


class BoardLocator:
    """Follows the board from frame to frame. Cheap when the board barely moves;
    falls back to a full search when it is lost (or every `recheck` frames)."""

    def __init__(self, recheck: int = 40):
        self.recheck = recheck
        self.reset()

    def reset(self):
        self.fit: BoardFit | None = None
        self._n = 0
        self._miss = 0
        self._weak = 0
        self._yn = 0
        self._ymiss = 0

    def _track(self, gray, scale, fit: BoardFit) -> BoardFit | None:
        pts = find_saddles(gray, max(1.4, min(4.0, 0.045 * fit.cell_px * scale)))
        if len(pts) < 12:
            return None
        Hs = np.diag([scale, scale, 1.0]) @ fit.H
        lat = np.array([(i, j) for i in range(9) for j in range(9)], np.float64)
        cell = fit.cell_px * scale
        for tol in (0.35, 0.22):
            pred = _apply_h(Hs, lat)
            idx, _ = _snap(pts, pred, tol * cell)
            ok = idx >= 0
            if ok.sum() < MIN_INLIERS:
                return None
            H2, mask = _fit_h(lat[ok], pts[idx[ok]].astype(np.float64), thr=0.12 * cell)
            if H2 is None or mask.sum() < MIN_INLIERS:
                return None
            Hs = H2
        corners = _apply_h(Hs, [(0, 0), (8, 0), (8, 8), (0, 8)]) / scale
        c_ok = _local_cell(Hs, 4, 4)
        # sanity: still a convex, non-degenerate quad of about the same size
        if c_ok < 0.5 * cell or c_ok > 2.0 * cell:
            return None
        pol = _polarity(gray, Hs)
        if fit.pol and pol and pol != fit.pol:
            return None            # the lattice slid by an odd number of cells: do not trust it
        Hf = np.diag([1.0 / scale, 1.0 / scale, 1.0]) @ Hs
        return BoardFit(corners.astype(np.float32), min(1.0, mask.sum() / 49.0), int(mask.sum()), Hf,
                        c_ok / scale, fit.pol or pol)

    def _align_corners(self, c: np.ndarray) -> np.ndarray:
        """Cyclic order of corners that matches the previous fit (see _align)."""
        if self.fit is None:
            return _order_corners(c)
        prev = self.fit.corners
        best = min(range(4), key=lambda k: float(np.linalg.norm(np.roll(c, -k, axis=0) - prev)))
        # same cyclic order but either winding: also try the mirrored order
        cm = c[::-1]
        bm = min(range(4), key=lambda k: float(np.linalg.norm(np.roll(cm, -k, axis=0) - prev)))
        d1 = float(np.linalg.norm(np.roll(c, -best, axis=0) - prev))
        d2 = float(np.linalg.norm(np.roll(cm, -bm, axis=0) - prev))
        return (np.roll(c, -best, axis=0) if d1 <= d2 else np.roll(cm, -bm, axis=0)).astype(np.float32)

    def _align(self, f: BoardFit) -> BoardFit:
        """Keep the corner order identical from frame to frame. A tracked lattice has an arbitrary
        orientation and a fresh detection uses screen order; without this the 'which way is White'
        decision made at the start would silently rotate mid-game."""
        f.corners = self._align_corners(f.corners)
        return f

    def _update_yolo(self, frame: np.ndarray) -> BoardFit | None:
        """Corner-model path. Returns None when the model found nothing and there is no recent fit,
        so update() falls back to the classical lattice search."""
        every = max(1, yolo.get_config().corner_every)
        run = self.fit is None or self._yn % every == 0 or self._ymiss > 0
        self._yn += 1
        if run:
            corners = yolo.detect_corners(frame)
            if corners is not None:
                if self.fit is None:
                    # first detection: no winding to stay consistent with yet, so sort the corners
                    # (TL, TR, BR, BL) from their positions
                    f = fit_from_corners(frame, corners, ordered=False)
                else:
                    # stable winding exists: keep the previous fit's corner order
                    c = self._align_corners(corners)
                    cell = max(1.0, self.fit.cell_px)
                    if float(np.abs(c - self.fit.corners).max()) < 0.25 * cell:
                        c = (self.fit.corners + 0.5 * (c - self.fit.corners)).astype(np.float32)  # calm jitter
                    f = fit_from_corners(frame, c, ordered=True)
                if self.fit is None:
                    log.info("board found by corner model (cell %.0f px)", f.cell_px)
                self.fit, self._miss, self._ymiss = f, 0, 0
                return f
            self._ymiss += 1
            if self.fit is not None and self._ymiss < 3:
                return self.fit
            log.debug("corner model found no board; using the classical search")
            return None
        # between model runs: refine on the checker lattice if there is one, else hold the fit
        gray, scale = _prepare(frame)
        f = self._track(gray, scale, self.fit)
        if f is not None:
            self.fit = f = self._align(f)
        return self.fit

    def update(self, frame: np.ndarray) -> BoardFit | None:
        if yolo.corner_backend_active():
            f = self._update_yolo(frame)
            if f is not None:
                return f
        self._n += 1
        if self.fit is not None:
            gray, scale = _prepare(frame)
            f = self._track(gray, scale, self.fit)
            if f is not None:
                f = self._align(f)
                self._miss = 0
                self._weak = self._weak + 1 if f.inliers < 45 else 0
                if self._weak >= 25:
                    # tracking has been poorly supported for a while: look again from scratch, but
                    # only trust the fresh search if it is clearly better (a hand or sleeve over the
                    # board makes a from-scratch search pick a window that is off by a row)
                    self._weak = 0
                    d = detect_board(frame)
                    if d is not None and d.inliers > f.inliers + 10:
                        f = self._align(d)
                self.fit = f
                return f
            self._miss += 1
            if self._miss < 3:
                return self.fit   # brief hiccup (hand over the board): keep the last good one
        f = detect_board(frame)
        if f is not None:
            f = self._align(f)
            if self.fit is None:
                log.info("board found (%d lattice points, cell %.0f px)", f.inliers, f.cell_px)
            self.fit, self._miss, self._weak = f, 0, 0
        else:
            self._miss += 1
            if self._miss >= 3 and self.fit is not None:
                log.info("board lost")
                self.fit = None
        return self.fit
