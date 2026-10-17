"""Get back on track mid-game without needing the starting position.

When the tracked position stops matching the screen (a few moves were missed because a hand
covered the board, the camera lurched, the game was paused ...) we search for the legal move
sequence, starting from the last position we trusted, that explains what is on screen now.

The screen only tells us "empty / white / black" per square (see occupancy.SquareReader), so this is a
beam search over *colour maps*: every legal move changes at most four squares, so the cost of a
candidate position is updated incrementally. The game's own last-move highlight (if there is one)
is used as a tie-breaker. We also try rewinding a few plies first, in case the last accepted
move(s) were wrong.
"""
from __future__ import annotations

import time

import chess
import numpy as np

EMPTY, WHITE, BLACK = 0, 1, 2
HINT_CREDIT = 1.5      # cost credit when the final move's squares are the highlighted ones
PLY_PENALTY = 0.12     # prefer the shortest explanation
REWIND_PENALTY = 0.4   # prefer not to undo moves we already accepted


def classes_of(board: chess.Board) -> np.ndarray:
    cls = np.zeros(64, np.int8)
    for sq, p in board.piece_map().items():
        cls[sq] = WHITE if p.color == chess.WHITE else BLACK
    return cls


def _changes(board: chess.Board, move: chess.Move):
    """[(square, new_class)] for the colour map after `move`."""
    piece = board.piece_at(move.from_square)
    col = WHITE if piece.color == chess.WHITE else BLACK
    ch = [(move.from_square, EMPTY), (move.to_square, col)]
    if board.is_en_passant(move):
        ch.append((chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square)), EMPTY))
    elif board.is_castling(move):
        rank = chess.square_rank(move.from_square)
        if chess.square_file(move.to_square) > chess.square_file(move.from_square):
            rf, rt = chess.square(7, rank), chess.square(5, rank)
        else:
            rf, rt = chess.square(0, rank), chess.square(3, rank)
        ch += [(rf, EMPTY), (rt, col)]
    return ch


def find_resync(board: chess.Board, class_costs: np.ndarray, hint=frozenset(), max_plies: int = 6,
                width: int = 120, rewind: int = 3, time_budget: float = 1.5, accept: float = 3.0,
                min_gain: float = 3.0, final_in_hint: bool = False):
    """board: last trusted position (its move_stack is used for rewinding).
    class_costs: (64, 3) from SquareReader.class_costs - cost of reading each square as empty/white/black.
    final_in_hint: only accept sequences whose last move is the highlighted one (use when the game
    highlights moves: it makes a hand or a held piece unable to fake a 'missed move').
    Returns (new_board, plies_added, rewound) or None."""
    t0 = time.perf_counter()
    cc = np.asarray(class_costs, np.float32)
    idx = np.arange(64)
    hint = frozenset(hint)
    root_cost = float(cc[idx, classes_of(board)].sum())
    best = None   # (eff, board)

    roots = []
    b = board.copy()
    for r in range(rewind + 1):
        roots.append((b.copy(), r))
        if not b.move_stack:
            break
        b.pop()

    for root, rew in roots:
        cls0 = classes_of(root)
        cost0 = float(cc[idx, cls0].sum())
        beam = [(cost0, root, cls0, ())]
        if rew > 0:
            # "the last move(s) we accepted were wrong": just undo them, add nothing
            last = root.move_stack[-1] if root.move_stack else None
            eff0 = cost0 + REWIND_PENALTY * rew
            ok_final = last is not None and hint and len(hint) >= 2 and {last.from_square, last.to_square} <= hint
            if ok_final:
                eff0 -= HINT_CREDIT
            if not final_in_hint or ok_final:
                if best is None or eff0 < best[0]:
                    best = (eff0, root, (), cost0, rew)
        for depth in range(1, max_plies + 1):
            if time.perf_counter() - t0 > time_budget:
                break
            cand = []   # (cost, beam_index, move, changes)
            for bi, (cost, bd, cls, _mv) in enumerate(beam):
                for m in bd.legal_moves:
                    ch = _changes(bd, m)
                    c = cost
                    for sq, new in ch:
                        c += float(cc[sq, new] - cc[sq, cls[sq]])
                    cand.append((c, bi, m, ch))
            if not cand:
                break
            cand.sort(key=lambda x: x[0])
            nxt, seen = [], set()
            for c, bi, m, ch in cand:
                if len(nxt) >= width:
                    break
                pb = beam[bi][1]
                nb = pb.copy(stack=False)
                nb.push(m)
                key = (nb.board_fen(), nb.turn)
                if key in seen:
                    continue
                seen.add(key)
                ncls = beam[bi][2].copy()
                for sq, new in ch:
                    ncls[sq] = new
                nxt.append((c, nb, ncls, beam[bi][3] + (m,)))
            beam = nxt
            for c, nb, _ncls, mv in beam[:8]:
                if final_in_hint and not (hint and len(hint) >= 2 and
                                          {mv[-1].from_square, mv[-1].to_square} <= hint):
                    continue
                eff = c + PLY_PENALTY * depth + REWIND_PENALTY * rew
                if hint and len(hint) >= 2 and mv and {mv[-1].from_square, mv[-1].to_square} <= hint:
                    eff -= HINT_CREDIT
                if best is None or eff < best[0]:
                    best = (eff, root, mv, c, rew)
            # nothing explains the picture yet and the best candidate is already excellent: stop early
            if beam and beam[0][0] < 0.8:
                break

    if best is None:
        return None
    eff, root, mv, raw, rew = best
    if not mv and rew == 0:
        return None
    if raw > accept or root_cost - raw < min_gain:
        return None
    nb = root.copy()
    for m in mv:
        nb.push(m)
    return nb, len(mv), rew
