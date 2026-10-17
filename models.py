from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import chess

from analysis import TacticalInfo
from geometry import BoardGeometry


@dataclass
class FrameResult:
    board: Optional[chess.Board] = None
    geometry: Optional[BoardGeometry] = None
    flipped: bool = False
    my_turn: bool = True
    best: Optional[chess.Move] = None
    second: Optional[chess.Move] = None
    threat: Optional[chess.Move] = None
    tactics: Optional[TacticalInfo] = None
    eval_text: str = ""
    status: str = ""
    pending: bool = False    # True while the engine is still thinking about this position (async engine)
