"""Hands-free mode: find the board, work out its orientation, follow the game.

    frame -> BoardLocator (checker lattice, follows camera movement)
          -> 4 possible rotations tried only while looking for a starting position
          -> SquareReader (empty / white / black per square, learns from the game)
          -> BoardTracker (legal moves that explain what changed)

No calibration clicks, no 'Learn pieces'. It needs to see the starting position once
at the beginning of a game (or after you press 'Reset game' on one); from there it
follows the game even while the camera moves, hands cover the board, or pieces slide.
Rotation is handled by re-ordering the board's corners, so a board seen from Black's
side or from the side of the table is just a different corner order.
"""
from __future__ import annotations

from dataclasses import dataclass

import chess
import numpy as np

from autodetect import BoardLocator
from geometry import BoardGeometry
from resync import find_resync
from occupancy import SquareReader, extract, highlight_squares, start_check
from tracker import BoardTracker


@dataclass
class AutoStep:
    geometry: BoardGeometry | None = None
    board: chess.Board | None = None
    changed: bool = False
    status: str = ""
    moved: bool = False        # board geometry moved noticeably (camera / window moved)


def _tracker() -> BoardTracker:
    # costs are 0..1 per square (see SquareReader): a real move changes >= 2 squares
    return BoardTracker(confirm=2, margin=0.8, escalate_gap=3.0, resync_gap=1e9, can_sync=False)


class AutoGame:
    # learning from the screen (SquareReader.adapt) is only done when the picture agrees with the tracked
    # position: low cost, calm for a few frames, no sign that we are behind the game
    ADAPT_OBS = 2.5
    ADAPT_STABLE = 0
    ADAPT_MISM = 3
    ADAPT_NEEDS_HINT_OK = True

    def __init__(self):
        self.locator = BoardLocator()
        self.reader = SquareReader()
        self.tracker = _tracker()
        self.rot = 0
        self._start_k: int | None = None
        self._start_n = 0
        self._lost = 0
        self._n = 0
        self._last_corners: np.ndarray | None = None
        self._mism = 0            # consecutive frames where the tracked position disagrees with the screen
        self._last_resync = -99
        self._resync_cand = None
        self._behind = 0
        self._stable, self._stable_fen = 0, None
        self._cand_n = -99

    def reset_game(self):
        self.reader.reset()
        self.tracker = _tracker()
        self._start_k, self._start_n, self._lost = None, 0, 0
        self._mism, self._last_resync, self._resync_cand, self._behind = 0, -99, None, 0

    def full_reset(self):
        self.locator.reset()
        self.reset_game()

    # ------------------------------------------------------------------
    def _rotated(self, corners: np.ndarray, k: int) -> BoardGeometry:
        return BoardGeometry(np.roll(corners, -k, axis=0))

    def _find_start(self, frame, corners, force: bool):
        """Is a starting position on the board, and which of the 4 corner orders puts White at the
        bottom? Returns (k, features) or None."""
        found = None
        for k in range(4):
            g = self._rotated(corners, k)
            f = extract(g.warp(frame), g.cell, False)
            ok, white_bottom = start_check(f)
            if not ok or not white_bottom:
                continue
            if not f[chess.A1, 6] < f[chess.B1, 6]:    # a1 must be a dark square
                continue
            found = (k, f)
            break
        return found

    def _obs(self, f: np.ndarray, board: chess.Board) -> float:
        """How badly the tracked position explains these square features (0 = perfectly)."""
        cls = np.zeros(64, np.int64)
        for sq, p in board.piece_map().items():
            cls[sq] = 1 if p.color == chess.WHITE else 2
        return float(self.reader.class_costs(f)[np.arange(64), cls].sum())

    def step(self, frame: np.ndarray, prefer_white: bool = True, force_start: bool = False) -> AutoStep:
        self._n += 1
        out = AutoStep()
        fit = self.locator.update(frame)
        if fit is None:
            out.status = "Looking for a chessboard on screen..."
            out.board = self.tracker.board
            return out
        corners = fit.corners
        if self._last_corners is not None:
            out.moved = float(np.abs(corners - self._last_corners).max()) > 1.5
        self._last_corners = corners.copy()

        if not self.reader.ready:
            hit = self._find_start(frame, corners, force_start)
            if hit is None:
                self._start_n = 0
                out.geometry = self._rotated(corners, self.rot)
                out.status = "Board found. Waiting for a starting position (or press 'Reset game' on one)."
                return out
            k, f = hit
            self._start_n = self._start_n + 1 if k == self._start_k else 1
            self._start_k = k
            out.geometry = self._rotated(corners, k)
            if self._start_n < 2:
                return out
            self.rot = k
            self.reader.learn_start(f)
            self.tracker = _tracker()
            self._lost = 0

        g = self._rotated(corners, self.rot)
        out.geometry = g
        warped = g.warp(frame)
        f = extract(warped, g.cell, False)
        D = self.reader.cost_matrix(f, warped=warped)
        hint = highlight_squares(warped, g.cell)
        out.changed = self.tracker.update(D, prefer_white_to_move=prefer_white, hint=hint)
        out.board = self.tracker.board

        board = self.tracker.board
        if board is None:
            out.status = "Reading the starting position..."
            return out

        # how well does the tracked position explain this frame?
        cls = np.zeros(64, np.int64)
        for s, p in board.piece_map().items():
            cls[s] = 1 if p.color == chess.WHITE else 2
        cc = self.reader.class_costs(f)
        obs = float(cc[np.arange(64), cls].sum())
        if obs > 6.0:
            self._lost += 1
        elif obs < 3.0:
            self._lost = 0
        # learn from the frame only when the game's own highlight agrees with the tracked last move
        # (otherwise a wrongly tracked position would teach the reader wrong colours)
        last = board.move_stack[-1] if board.move_stack else None
        hint_ok = (not hint) or (last is not None and {last.from_square, last.to_square} <= set(hint))
        fen_now = board.board_fen()
        self._stable = self._stable + 1 if fen_now == self._stable_fen else 0
        self._stable_fen = fen_now
        if obs < self.ADAPT_OBS and (hint_ok or not self.ADAPT_NEEDS_HINT_OK) and self._stable >= self.ADAPT_STABLE \
                and self._mism < self.ADAPT_MISM and self.tracker._pending is None \
                and self._n % 4 == 0:
            self.reader.adapt(f, board)

        # Missed some moves (hand over the board, camera lurch, paused frames)? Find the legal moves
        # that explain the screen now - no starting position needed.
        # Two ways to notice we are behind the game:
        #  * the game highlights a move that is not our last move (works even if the picture is messy);
        #  * no highlight at all, but the tracked position keeps disagreeing with the screen.
        # A piece merely held in the air or a hand over the board must NOT trigger it: then the
        # highlight still agrees with our last move, so we stay put.
        two = len(hint) >= 2
        agree = (not hint) or (last is not None and {last.from_square, last.to_square} <= set(hint))
        self._behind = self._behind + 1 if (two and not agree) else 0
        self._mism = self._mism + 1 if obs > 1.5 else 0
        by_hint = self._behind >= 6
        by_look = self._mism >= 8
        if (by_hint or by_look) and self._n - self._last_resync >= 4:
            self._last_resync = self._n
            if by_hint:
                res = find_resync(board, cc, hint, accept=3.0, min_gain=-0.5, final_in_hint=True)
            else:
                res = find_resync(board, cc, hint, accept=1.6, min_gain=max(1.5, 0.5 * obs),
                                  final_in_hint=bool(hint))
            if res is not None:
                nb, plies, rew = res
                key = (nb.board_fen(), nb.turn)
                if key == self._resync_cand and self._n - self._cand_n <= 60:   # same answer twice: believe it
                    self.tracker.set_board(nb)
                    board = nb
                    out.board, out.changed = nb, True
                    self._lost = self._mism = self._behind = 0
                    self._resync_cand = None
                    out.status = f"Re-synced (+{plies}, -{rew} moves)."
                    return out
                if key != self._resync_cand:
                    self._resync_cand, self._cand_n = key, self._n

        if self._lost >= 8:
            # lost the game (new game with the other colour, or the picture changed a lot):
            # if a starting position is on screen, start over from it
            hit = self._find_start(frame, corners, False)
            if hit is not None:
                self.reset_game()
                out.status = "New game detected."
            else:
                out.status = "Lost track - trying to re-sync (or show a starting position / press 'Reset game')."
        return out
