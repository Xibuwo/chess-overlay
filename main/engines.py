"""UCI engine manager: Stockfish / Maia 3 / Maia (Lc0 weights) / Lc0.

The strength slider is translated differently for each engine (see strength.py):
  * Stockfish      -> Skill Level, depth limit, or UCI_LimitStrength+UCI_Elo
  * Maia 3         -> ELO passed directly through the engine's `Elo` option, nodes=1
  * Maia (Lc0)     -> nearest maia-XXXX weights file, nodes=1
  * Lc0            -> node limit
"""
from __future__ import annotations

import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

import chess
import chess.engine

import strength as st
from logsetup import get_logger
from config import (KIND_LC0, KIND_MAIA3, KIND_MAIA_LC0, KIND_STOCKFISH, SF_DEPTH,
                    SF_SKILL, SF_UCI_ELO, Settings)


log = get_logger("engines")


@dataclass
class EngineResult:
    best: Optional[chess.Move] = None
    second: Optional[chess.Move] = None
    threat: Optional[chess.Move] = None  # our turn: opponent's best reply if we passed; their turn: predicted move
    eval_text: str = ""
    note: str = ""


def parse_extra_options(text: str) -> dict:
    """'Threads=2; Ponder=false; Name With Spaces=5' -> {'Threads': 2, 'Ponder': False, ...}"""
    out: dict = {}
    for chunk in (text or "").replace("\n", ";").split(";"):
        if "=" not in chunk:
            continue
        k, v = chunk.split("=", 1)
        k, v = k.strip(), v.strip()
        if not k:
            continue
        low = v.lower()
        if low in ("true", "false"):
            out[k] = low == "true"
        else:
            try:
                out[k] = int(v)
            except ValueError:
                out[k] = v
    return out


def split_command(cmd: str) -> list[str]:
    parts = shlex.split(cmd.strip(), posix=False)
    return [p[1:-1] if len(p) >= 2 and p[0] == p[-1] and p[0] in "\"'" else p for p in parts]


class EngineManager:
    def __init__(self):
        self.engine: chess.engine.SimpleEngine | None = None
        self._launch_key = None
        self._applied: dict | None = None
        self.error = ""

    # ---- lifecycle --------------------------------------------------------
    def close(self):
        eng, self.engine = self.engine, None
        if eng is not None:
            try:
                eng.quit()
            except Exception:
                log.warning("engine did not quit cleanly, closing the transport", exc_info=True)
                try:
                    eng.close()
                except Exception:
                    log.warning("closing the engine transport failed", exc_info=True)
        self._launch_key = None
        self._applied = None

    def kill(self):
        """Hard stop, safe to call from another thread while analyse() is blocked: terminates the
        engine process so the blocked call raises instead of hanging (used when stopping)."""
        eng = self.engine
        if eng is not None:
            try:
                eng.close()
            except Exception:
                log.warning("killing the engine failed", exc_info=True)

    def _build_command(self, s: Settings) -> list[str]:
        kind = s.engine_kind
        cmd = split_command(s.engine_cmd.get(kind, ""))
        if not cmd:
            raise ValueError("No engine command set.")
        if kind == KIND_MAIA_LC0:
            hit = st.nearest_maia_weights(s.weights_path.get(kind, ""), s.elo)
            if not hit:
                raise ValueError("No maia-XXXX.pb.gz weights found in the weights folder.")
            cmd.append(f"--weights={hit[1]}")
        elif kind == KIND_LC0 and s.weights_path.get(kind):
            cmd.append(f"--weights={s.weights_path[kind]}")
        return cmd

    def ensure(self, s: Settings) -> bool:
        try:
            cmd = self._build_command(s)
        except ValueError as e:
            self.error = str(e)
            log.warning("engine not configured: %s", e)
            self.close()
            return False
        key = tuple(cmd)
        if self.engine is not None and key == self._launch_key:
            return True
        self.close()
        try:
            kw = {"creationflags": 0x08000000} if sys.platform == "win32" else {}  # CREATE_NO_WINDOW
            self.engine = chess.engine.SimpleEngine.popen_uci(cmd, timeout=60, **kw)
            self._launch_key = key
            self.error = ""
            log.info("engine started: %s", " ".join(cmd))
            return True
        except Exception as e:  # missing exe, bad weights, ...
            log.error("could not start engine %s", cmd, exc_info=True)
            self.error = f"Could not start engine: {e}"
            self.engine = None
            return False

    # ---- strength mapping -------------------------------------------------
    def _plan(self, s: Settings, as_opponent: bool = False) -> tuple[dict, chess.engine.Limit, str]:
        """as_opponent=True: we are predicting the *other* side's move, so for Maia 3
        the roles of SelfElo / OppoElo are swapped."""
        opts: dict = {}
        kind, elo = s.engine_kind, int(s.elo)
        secs = max(0.02, int(s.movetime_ms) / 1000.0)
        nodes_override = int(s.engine_nodes) if int(s.engine_nodes) > 0 else None
        if kind == KIND_STOCKFISH:
            base = {"Threads": int(s.sf_threads), "Hash": int(s.sf_hash_mb)}
            if s.sf_mode == SF_DEPTH:
                d = st.depth_from_elo(elo)
                opts = {**base, "UCI_LimitStrength": False, "Skill Level": 20}
                return opts, chess.engine.Limit(depth=d), f"SF depth {d}"
            if s.sf_mode == SF_UCI_ELO:
                opts = {**base, "UCI_LimitStrength": True, "UCI_Elo": elo}
                return opts, chess.engine.Limit(time=secs), f"SF UCI_Elo {elo}"
            k = st.skill_from_elo(elo)
            opts = {**base, "UCI_LimitStrength": False, "Skill Level": k}
            return opts, chess.engine.Limit(time=secs), f"SF skill {k}"
        if kind == KIND_MAIA3:
            n = nodes_override or 1
            if s.oppo_same_as_elo:
                opts = {"Elo": elo}
                note = f"Maia 3 @ {elo} (both sides), nodes={n}"
            else:
                me, op = elo, int(s.oppo_elo)
                selfe, oppe = (op, me) if as_opponent else (me, op)
                opts = {"SelfElo": selfe, "OppoElo": oppe}
                note = f"Maia 3 self {selfe} / opp {oppe}, nodes={n}"
            return opts, chess.engine.Limit(nodes=n), note
        if kind == KIND_MAIA_LC0:
            n = nodes_override or 1
            return {}, chess.engine.Limit(nodes=n), f"Maia (nearest net to {elo}), nodes={n}"
        n = nodes_override or st.nodes_from_elo(elo)
        return {}, chess.engine.Limit(nodes=n), f"Lc0 nodes={n}"

    def _apply(self, opts: dict, s: Settings | None = None):
        if self.engine is None:
            return
        opts = dict(opts)
        if s is not None:
            opts.update(parse_extra_options(s.extra_uci))
        if "SelfElo" in opts and "SelfElo" not in self.engine.options and "Elo" in self.engine.options:
            opts["Elo"] = opts.pop("SelfElo")  # older Maia 3 builds only expose Elo
            opts.pop("OppoElo", None)
        if not opts or opts == self._applied:
            return
        safe = {}
        for name, val in opts.items():
            spec = self.engine.options.get(name)
            if spec is None or spec.is_managed():
                continue
            if isinstance(val, int) and not isinstance(val, bool):
                if spec.min is not None:
                    val = max(spec.min, val)
                if spec.max is not None:
                    val = min(spec.max, val)
            safe[name] = val
        if safe:
            self.engine.configure(safe)
        self._applied = dict(opts)

    # ---- analysis ---------------------------------------------------------
    def analyse(self, board: chess.Board, s: Settings, my_turn: bool) -> EngineResult:
        res = EngineResult()
        if not self.ensure(s):
            res.note = self.error
            return res
        try:
            opts, limit, note = self._plan(s)
            self._apply(opts, s)
            res.note = note
            if my_turn:
                multipv = 2 if (s.show_second and "MultiPV" in self.engine.options) else 1
                infos = self.engine.analyse(board, limit, multipv=multipv)
                if isinstance(infos, dict):
                    infos = [infos]
                moves = [i["pv"][0] for i in infos if i.get("pv")]
                if not moves:
                    moves = [self.engine.play(board, limit).move]
                res.best = moves[0]
                res.second = moves[1] if len(moves) > 1 and moves[1] != moves[0] else None
                score = infos[0].get("score") if infos else None
                if score is not None:
                    pov = score.pov(board.turn)
                    if pov.is_mate():
                        res.eval_text = f"#{pov.mate()}"
                    else:
                        res.eval_text = f"{pov.score() / 100:+.2f}"
                # threat: what would the opponent do if we passed?
                if s.show_threat and not board.is_check():
                    b2 = board.copy(stack=False)
                    b2.push(chess.Move.null())
                    if not b2.is_game_over():
                        o_opts, o_limit, _ = self._plan(s, as_opponent=True)
                        self._apply(o_opts, s)
                        res.threat = self.engine.play(b2, o_limit).move
            else:
                if s.show_threat and not board.is_game_over():
                    o_opts, o_limit, _ = self._plan(s, as_opponent=True)
                    self._apply(o_opts, s)
                    res.threat = self.engine.play(board, o_limit).move  # predicted opponent move
        except (chess.engine.EngineError, chess.engine.EngineTerminatedError, TimeoutError, OSError) as e:
            log.warning("engine failed while analysing %s", board.fen(), exc_info=True)
            self.error = f"Engine error: {e}"
            res.note = self.error
            self.close()
        return res
