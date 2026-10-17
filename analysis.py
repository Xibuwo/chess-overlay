"""Static tactical analysis (no engine): contested squares and hung/vulnerable pieces."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import chess

VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}
_ORDER = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)


@dataclass
class Contested:
    square: int
    piece: chess.Piece
    attackers: int
    defenders: int

    @property
    def diff(self) -> int:
        return self.defenders - self.attackers


@dataclass
class Weak:
    square: int
    piece: chess.Piece
    kind: str  # "hung" (no defender) or "vulnerable" (loses material in the exchange)
    net: int   # material the attacker wins with best exchange


@dataclass
class TacticalInfo:
    contested: list[Contested] = field(default_factory=list)
    weak: list[Weak] = field(default_factory=list)


def _least_valuable_attacker(b: chess.Board, color: bool, square: int) -> Optional[int]:
    mask = b.attackers_mask(color, square)
    if not mask:
        return None
    for pt in _ORDER:
        m = mask & b.pieces_mask(pt, color)
        if m:
            return chess.lsb(m)
    return None


def static_exchange(board: chess.Board, square: int, attacker_color: bool) -> int:
    """Material `attacker_color` nets by starting a capture sequence on `square`
    (>= 0; 0 means capturing doesn't pay). X-rays are handled by re-evaluating
    attackers after each removal."""
    target = board.piece_at(square)
    if target is None:
        return 0
    b = board.copy(stack=False)
    gains: list[int] = []
    side = attacker_color
    on_square = VALUE[target.piece_type]
    while len(gains) < 32:
        frm = _least_valuable_attacker(b, side, square)
        if frm is None:
            break
        gains.append(on_square)
        mover = b.piece_at(frm)
        on_square = VALUE[mover.piece_type]
        b.remove_piece_at(frm)
        b.set_piece_at(square, mover)
        side = not side
    score = 0
    for g in reversed(gains):
        score = max(0, g - score)
    return score


def analyse_tactics(board: chess.Board) -> TacticalInfo:
    info = TacticalInfo()
    for sq, piece in board.piece_map().items():
        enemy = not piece.color
        att = len(board.attackers(enemy, sq))
        if att == 0:
            continue
        dfn = len(board.attackers(piece.color, sq))
        info.contested.append(Contested(sq, piece, att, dfn))
        if piece.piece_type == chess.KING:
            continue
        net = static_exchange(board, sq, enemy)
        if net > 0:
            info.weak.append(Weak(sq, piece, "hung" if dfn == 0 else "vulnerable", net))
    return info
