"""Follows the game by explaining each observed frame with *legal moves*.

Instead of trusting a free per-square classification (which breaks on tall 3D
pieces, animations and highlights), every frame is scored against the tracked
position, all its 1-ply successors (and 2-ply when needed). The candidate with
the lowest total template cost wins, with debouncing.
"""
from __future__ import annotations

import chess
import numpy as np

from recognition import board_labels, label_to_piece

_IDX = np.arange(64)
START_LABELS = board_labels(chess.Board())


def labels_after(board: chess.Board, labels: np.ndarray, move: chess.Move) -> np.ndarray:
    l = labels.copy()
    piece = board.piece_at(move.from_square)
    if piece is None:
        return l
    if move.promotion:
        l[move.to_square] = move.promotion + (6 if piece.color == chess.BLACK else 0)
    else:
        l[move.to_square] = l[move.from_square]
    l[move.from_square] = 0
    if board.is_en_passant(move):
        l[chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))] = 0
    elif board.is_castling(move):
        rank = chess.square_rank(move.from_square)
        if chess.square_file(move.to_square) > chess.square_file(move.from_square):
            rf, rt = chess.square(7, rank), chess.square(5, rank)
        else:
            rf, rt = chess.square(0, rank), chess.square(3, rank)
        l[rt] = l[rf]
        l[rf] = 0
    return l


def build_board(labels: np.ndarray, prefer_white_to_move: bool) -> chess.Board | None:
    if np.array_equal(labels, START_LABELS):
        return chess.Board()
    b = chess.Board(None)
    for sq, lab in enumerate(labels):
        if lab:
            b.set_piece_at(sq, label_to_piece(int(lab)))
    b.castling_rights = chess.BB_CORNERS
    b.castling_rights = b.clean_castling_rights()
    for turn in (prefer_white_to_move, not prefer_white_to_move):
        b.turn = turn
        if b.status() == chess.STATUS_VALID:
            return b
    return None


HINT_BONUS = 2.0   # cost credit for a move whose from/to squares are the highlighted ones


def _move_sq(m: chess.Move) -> frozenset:
    return frozenset((m.from_square, m.to_square))


class BoardTracker:
    # Costs come normalised (see PieceClassifier.scale): ~1.0 = one misexplained piece.
    def __init__(self, confirm=2, margin=0.30, escalate_gap=0.8, resync_gap=2.0, resync_frames=8,
                 start_gap=1.2, ema=0.4, can_sync=True):
        # can_sync=False: the caller already proved a starting position is on screen (AutoGame),
        # so begin from it directly and never try to re-read the whole board from scratch.
        self.can_sync = can_sync
        self.start_gap = start_gap  # how far the start position may be from the best free reading
        self.ema = ema              # weight of the newest frame while looking for the first position
        self.confirm = confirm
        self.margin = margin
        self.escalate_gap = escalate_gap
        self.resync_gap = resync_gap
        self.resync_frames = resync_frames
        self.reset()

    def reset(self):
        self.board: chess.Board | None = None
        self.turn_uncertain = False
        self._pending = None
        self._pending_n = 0
        self._sync_key = None
        self._sync_n = 0
        self._bad = 0
        self._smooth = None   # smoothed cost matrix used while no position is known yet
        self._start_n = 0

    def set_board(self, board: chess.Board) -> None:
        """Adopt a position found by resync.find_resync (keeps its move history)."""
        self.board = board
        self.turn_uncertain = False
        self._pending, self._pending_n = None, 0
        self._bad = 0
        self._start_n = 0
        self._smooth = None

    @staticmethod
    def _score(D, labels) -> float:
        return float(D[_IDX, labels].sum())

    # ------------------------------------------------------------------
    def update(self, D: np.ndarray, prefer_white_to_move: bool = True, hint=None) -> bool:
        """Returns True when the tracked position changed.

        hint: optional set of squares the game itself highlights (the last move's from/to
        squares, see occupancy.highlight_squares). It is only a tie-breaker: a legal move
        whose from/to squares are exactly the highlighted ones gets a cost credit, so a move is
        found even when tall pieces make the per-square reading unreliable. Pieces that were
        merely picked up (one highlighted square) never count."""
        if self.board is None:
            if not self.can_sync:
                self.board = chess.Board()
                self.turn_uncertain = False
                return True
            return self._first_position(D, prefer_white_to_move)
        free = D.argmin(axis=1)
        hint = frozenset(hint) if hint else frozenset()
        last = _move_sq(self.board.move_stack[-1]) if self.board.move_stack else frozenset()
        fresh = hint if not last <= hint else hint - last      # squares not explained by the known last move
        hint_ok = len(fresh) >= 2

        def credit(moves) -> float:
            return HINT_BONUS if hint_ok and _move_sq(moves[-1]) <= fresh else 0.0

        cur = board_labels(self.board)
        cur_score = self._score(D, cur)
        free_score = float(D.min(axis=1).sum())

        bases = [self.board]
        if self.turn_uncertain:
            alt = self.board.copy(stack=False)
            alt.turn = not alt.turn
            bases.append(alt)

        # candidate = (kind, base, moves, raw_score, effective_score); effective = raw - hint credit
        best = ("keep", self.board, [], cur_score, cur_score)

        def offer(kind, base, moves, raw):
            nonlocal best
            eff = raw - (credit(moves) if moves else 0.0)
            if eff < best[4]:
                best = (kind, base, moves, raw, eff)

        start_score = self._score(D, START_LABELS)
        if self.board.move_stack and start_score + self.margin * 2 < cur_score:
            offer("reset", None, [], start_score)

        for base in bases:
            try:
                for m in base.legal_moves:
                    offer("move", base, [m], self._score(D, labels_after(base, cur, m)))
            except Exception:
                continue

        # one move doesn't explain what we see (or the game highlights a move we don't have): 2 plies
        gap_escalate = best[3] - free_score > self.escalate_gap
        if gap_escalate or (hint_ok and best[0] != "move"):
            for base in bases:
                try:
                    for m1 in base.legal_moves:
                        l1 = labels_after(base, cur, m1)
                        b2 = base.copy(stack=False)
                        b2.push(m1)
                        for m2 in b2.legal_moves:
                            if not gap_escalate and not _move_sq(m2) <= fresh:
                                continue   # hint-only search: the last ply must be the highlighted move
                            offer("move", base, [m1, m2], self._score(D, labels_after(b2, l1, m2)))
                except Exception:
                    continue

        kind, base, moves, score, eff = best
        changed = False
        need = self.margin * max(1, len(moves)) if kind != "reset" else self.margin * 2
        if kind != "keep" and (cur_score - eff) >= need:
            key = (kind, tuple(m.uci() for m in moves))
            self._pending_n = self._pending_n + 1 if key == self._pending else 1
            self._pending = key
            if self._pending_n >= self.confirm:
                if kind == "reset":
                    self.board = chess.Board()
                    self.turn_uncertain = False
                else:
                    self.board = base
                    for m in moves:
                        self.board.push(m)
                    self.turn_uncertain = False
                changed = True
                self._pending, self._pending_n = None, 0
        else:
            self._pending, self._pending_n = None, 0

        # lost track (e.g. we missed several moves): re-read the board from scratch
        if self.can_sync and score - free_score > self.resync_gap:
            self._bad += 1
            if self._bad >= self.resync_frames:
                self._bad = 0
                changed = self._sync(free, prefer_white_to_move, force=True) or changed
        else:
            self._bad = 0
        return changed

    # ------------------------------------------------------------------
    def resync_search(self, D: np.ndarray, hint=None, max_depth: int = 8, beam: int = 40, accept: float = 2.5):
        """Find the legal continuation (up to max_depth plies, i.e. that many *missed* moves) of the
        tracked game that best explains this frame. Used after the tracker lost the thread (a hand
        covered the board for a while, the camera jumped, several moves were played unseen).
        Returns a list of moves, or None when nothing explains the frame well enough."""
        if self.board is None:
            return None
        hint = frozenset(hint) if hint else frozenset()
        base_labels = board_labels(self.board)
        start = self._score(D, base_labels)
        level = [(self.board.copy(stack=False), base_labels, [])]
        best = None          # (effective score, raw score, moves)
        for depth in range(1, max_depth + 1):
            cand = []
            for b, l, mv in level:
                for m in b.legal_moves:
                    l2 = labels_after(b, l, m)
                    cand.append((self._score(D, l2), b, l2, mv, m))
            if not cand:
                break
            cand.sort(key=lambda c: c[0])
            nxt, seen = [], set()
            for sc, b, l2, mv, m in cand[: beam * 4]:
                b2 = b.copy(stack=False)
                b2.push(m)
                key = (b2.board_fen(), b2.turn)
                if key in seen:
                    continue
                seen.add(key)
                moves = mv + [m]
                eff = sc + 0.15 * depth - (HINT_BONUS if len(hint) >= 2 and _move_sq(m) <= hint else 0.0)
                if best is None or eff < best[0]:
                    best = (eff, sc, moves)
                nxt.append((b2, l2, moves))
                if len(nxt) >= beam:
                    break
            level = nxt
        if best is None or best[1] > accept or best[0] > start - 1.0:
            return None
        return best[2]

    # ------------------------------------------------------------------
    def _first_position(self, D: np.ndarray, prefer_white: bool) -> bool:
        """Hook onto a game. Tall 3D pieces make the raw per-square reading jitter from
        frame to frame, so (1) average the costs over a few frames, and (2) if the board is
        close to the starting position, accept it even when a few squares read wrongly."""
        self._smooth = D.copy() if self._smooth is None else (1 - self.ema) * self._smooth + self.ema * D
        E = self._smooth
        free_score = float(E.min(axis=1).sum())
        if self._score(E, START_LABELS) - free_score < self.start_gap:
            self._start_n += 1
            if self._start_n >= self.confirm:
                self.board = chess.Board()
                self.turn_uncertain = False
                self._start_n = 0
                self._smooth = None
                return True
            return False
        self._start_n = 0
        return self._sync(E.argmin(axis=1), prefer_white, force=False)

    def _sync(self, free: np.ndarray, prefer_white: bool, force: bool) -> bool:
        if not free.any():
            return False
        key = free.tobytes()
        self._sync_n = self._sync_n + 1 if key == self._sync_key else 1
        self._sync_key = key
        if self._sync_n < self.confirm + 1:
            return False
        board = build_board(free, prefer_white)
        if board is None:
            return False
        if force and self.board is not None and board.board_fen() == self.board.board_fen():
            return False
        self.board = board
        self.turn_uncertain = not np.array_equal(free, START_LABELS)
        self._pending, self._pending_n = None, 0
        return True