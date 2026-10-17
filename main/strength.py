"""Pure functions that turn the ELO/Strength slider into engine-specific settings."""
from __future__ import annotations

import re
from pathlib import Path

from config import (KIND_LC0, KIND_MAIA3, KIND_MAIA_LC0, KIND_STOCKFISH, SF_DEPTH,
                    SF_SKILL, SF_UCI_ELO)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _t(elo: float) -> float:
    """800..3200 -> 0..1"""
    return clamp((elo - 800) / 2400.0, 0.0, 1.0)


def elo_slider_range(kind: str, sf_mode: str = SF_SKILL) -> tuple[int, int]:
    """The slider range adapts to what the selected engine can actually do."""
    if kind == KIND_STOCKFISH:
        return (1350, 3150) if sf_mode == SF_UCI_ELO else (800, 3200)
    if kind == KIND_MAIA3:
        return (800, 2800)  # clamped again to the engine's own Elo option at runtime
    if kind == KIND_MAIA_LC0:
        return (1100, 1900)  # the nine released Maia nets
    return (800, 3200)


def skill_from_elo(elo: float) -> int:
    return int(round(_t(elo) * 20))  # Stockfish "Skill Level" 0..20


def depth_from_elo(elo: float) -> int:
    return max(1, int(round(1 + _t(elo) * 17)))  # depth 1..18


def nodes_from_elo(elo: float) -> int:
    return max(1, int(round(10 ** (_t(elo) * 3.3))))  # ~1..2000 nodes


_MAIA_RE = re.compile(r"maia-(\d{3,4})", re.I)


def maia_weights(folder: str) -> dict[int, Path]:
    out: dict[int, Path] = {}
    if not folder:
        return out
    p = Path(folder)
    if p.is_file():
        p = p.parent
    if not p.is_dir():
        return out
    for f in p.iterdir():
        m = _MAIA_RE.search(f.name)
        if m and f.is_file():
            out[int(m.group(1))] = f
    return out


def nearest_maia_weights(folder: str, elo: int):
    found = maia_weights(folder)
    if not found:
        return None
    best = min(found, key=lambda e: abs(e - elo))
    return best, found[best]


def describe(kind: str, sf_mode: str, elo: int, weights_folder: str = "") -> str:
    if kind == KIND_STOCKFISH:
        if sf_mode == SF_SKILL:
            return f"ELO {elo}  ->  Stockfish Skill Level {skill_from_elo(elo)}/20"
        if sf_mode == SF_DEPTH:
            return f"ELO {elo}  ->  Stockfish depth limit {depth_from_elo(elo)}"
        return f"ELO {elo}  ->  UCI_LimitStrength + UCI_Elo {elo}"
    if kind == KIND_MAIA3:
        return f"ELO {elo} sent directly to the engine (Elo option) - nodes=1 locked"
    if kind == KIND_MAIA_LC0:
        hit = nearest_maia_weights(weights_folder, elo)
        name = hit[1].name if hit else "no maia-*.pb.gz found"
        return f"ELO {elo}  ->  {name} - nodes=1 locked"
    return f"Strength {elo}  ->  Lc0 node limit {nodes_from_elo(elo)}"
