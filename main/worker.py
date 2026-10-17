"""Background pipeline: capture -> flatten -> recognise -> track -> (async) engine -> overlay.

Threads
  * this QThread  - frames, vision, move tracking, tactics. Never waits for the engine.
  * EngineService - one engine thread with a "latest position wins" mailbox (engine_service.py).
                    Its result comes back through `_on_engine`, which emits the final FrameResult.

A new position is emitted at once with `pending=True` (board, tactics, geometry; the previous
position's arrows are cleared) and again with `pending=False` when the engine answers. If the board
moves on before the engine is done, the old answer is dropped.

Resilience: one bad frame (capture hiccup, vision bug, ...) is logged, reported as "Pipeline error"
and retried after a growing pause - it never ends the thread. main.py's supervisor restarts the
thread should it die anyway.
"""
from __future__ import annotations

import dataclasses
import logging
import threading
import time

import chess
from PyQt6.QtCore import QThread, pyqtSignal

import yolo
from analysis import analyse_tactics
from autogame import AutoGame
from capture import ScreenGrabber
from config import TEMPLATES_FILE
from engine_service import EngineRequest, EngineResponse, EngineService
from logsetup import STATUS_LOGGER, CallbackHandler, get_logger, parse_level
from models import FrameResult
from motion import MotionTracker, compute_feats
from state import AppState
from supervision import Backoff
from tracker import BoardTracker

log = get_logger("worker")
status_log = logging.getLogger(STATUS_LOGGER)
status_log.setLevel(logging.DEBUG)       # the GUI handler (Settings.status_level) is the filter, not the logger

RESET_AFTER_FAILURES = 10       # consecutive failed iterations before the trackers are rebuilt too


class _Ctx:
    """Mutable per-run state of the frame loop."""

    def __init__(self):
        self.grabber: ScreenGrabber | None = None
        self.grab_key = None
        self.tracker = BoardTracker()
        self.motion = MotionTracker()
        self.auto = AutoGame()
        self.was_auto = None
        self.last_fen = None
        self.last_version = -1
        self.no_board_since = None
        self.cleared = True
        self.paused = False

    def close_grabber(self):
        g, self.grabber, self.grab_key = self.grabber, None, None
        if g is not None:
            try:
                g.close()
            except Exception:
                log.warning("closing the screen grabber failed", exc_info=True)


class PipelineWorker(QThread):
    result_ready = pyqtSignal(object)  # FrameResult
    status = pyqtSignal(str)

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self._stop_evt = threading.Event()
        self._lock = threading.RLock()          # guards _cur / _wanted (pipeline + engine threads)
        self._cur: FrameResult | None = None    # the result currently on screen (pending or final)
        self._wanted: tuple | None = None       # (fen, my_turn, version) the engine owes us
        self._engine: EngineService | None = None
        self._status_handler: CallbackHandler | None = None
        self._last_status = ""

    # ---- control -----------------------------------------------------------------
    @property
    def stop_requested(self) -> bool:
        return self._stop_evt.is_set()

    def stop(self):
        self._stop_evt.set()
        eng = self._engine
        if eng is not None:
            eng.cancel()
        self.wait(8000)

    # ---- status line (GUI) ---------------------------------------------------------
    def _say(self, text: str, level: int = logging.INFO):
        """Status for the menu: goes through the 'status' logger so the log file always gets it and
        the GUI only gets what passes Settings.status_level. Repeats are suppressed."""
        if text != self._last_status:
            self._last_status = text
            status_log.log(level, text)

    # ---- engine results (engine thread) ------------------------------------------------
    def _on_engine(self, resp: EngineResponse):
        with self._lock:
            if resp.key != self._wanted or self._cur is None:
                log.debug("ignoring engine answer for an old position")
                return
            r = resp.result
            final = dataclasses.replace(self._cur, best=r.best, second=r.second, threat=r.threat,
                                        eval_text=r.eval_text, status=r.note or "", pending=False)
            self._cur, self._wanted = final, None
        self.result_ready.emit(final)
        side = "your move" if final.my_turn else "opponent to move"
        self._say(f"{side} | {final.status} {final.eval_text}".strip())
        if resp.error:
            self._say(resp.error, logging.WARNING)

    def _forget_position(self, ctx: _Ctx):
        ctx.last_fen = None
        with self._lock:
            self._cur = self._wanted = None
        if self._engine is not None:
            self._engine.cancel()

    # ---- one iteration of the frame loop -----------------------------------------------------
    def _tick(self, c: _Ctx):
        state = self.state
        t0 = time.perf_counter()
        if not state.detecting:
            if not c.paused:  # entering pause: free the screen grabber + engine, clear overlay
                c.paused = True
                self._engine.release()
                c.close_grabber()
                with self._lock:
                    self._cur = self._wanted = None
                self.result_ready.emit(FrameResult(status="Stopped"))
                c.cleared = True
                self._say("Detection stopped.")
            state.pop_flags()  # drop stale learn/reset requests
            self._stop_evt.wait(0.1)
            return
        if c.paused:
            c.paused = False
            c.last_fen = None
            self._say("Detection started...")
        s, geom, version = state.snapshot()
        flags = state.pop_flags()
        period = 1.0 / max(0.5, float(s.scan_hz))
        self._status_handler.setLevel(parse_level(s.status_level))
        yolo.configure(yolo.config_from_settings(s))

        try:
            key = (s.capture_backend, s.monitor_index)
            if c.grabber is None or key != c.grab_key:
                c.close_grabber()
                c.grabber = ScreenGrabber(s.capture_backend, s.monitor_index)
                c.grab_key = key
        except Exception as e:
            log.error("capture backend could not be opened", exc_info=True)
            self._say(f"Capture error: {e}", logging.ERROR)
            self._stop_evt.wait(1.0)
            return

        if s.auto_detect != c.was_auto:          # mode switched: start clean
            c.was_auto = s.auto_detect
            c.auto.full_reset()
            c.tracker.reset()
            c.motion.reset()
            self._forget_position(c)
        if "reset" in flags or ("learn" in flags and (s.follow_moves or s.auto_detect)):
            c.tracker.reset()
            c.motion.reset()
            c.auto.reset_game()
            self._forget_position(c)

        my_color = chess.BLACK if s.play_as_black else chess.WHITE
        flipped = s.board_flipped
        moved = False
        changed = False

        if s.auto_detect:
            # hands-free: find the board (2D or 3D), follow it when the camera moves, read the game
            frame = c.grabber.grab()
            if frame is None:
                self._stop_evt.wait(period)
                return
            st = c.auto.step(frame, prefer_white=(my_color == chess.WHITE))
            geom, board, changed, moved, flipped = st.geometry, st.board, st.changed, st.moved, False
            if geom is None:
                board = None
            if st.status:
                self._say(st.status)
        else:
            if geom is None:
                self._say("Calibrate the board: click its four corners.")
                self._stop_evt.wait(0.25)
                return

            frame = c.grabber.grab()
            if frame is None:
                self._stop_evt.wait(period)
                return
            warped = geom.warp(frame)

            if "learn" in flags and not s.follow_moves and yolo.piece_backend_active():
                self._say("Learn pieces is not needed with the YOLO backend.")
            elif "learn" in flags and not s.follow_moves:
                ok, msg = state.classifier.learn(warped, geom.cell, s.board_flipped,
                                                s.fg_threshold, s.crop_top_extend)
                if ok:
                    state.classifier.save(TEMPLATES_FILE)
                    c.tracker.reset()
                    c.last_fen = None
                self._say(msg, logging.INFO if ok else logging.WARNING)

            if s.follow_moves:
                # 3D mode: 2D model of the game, advanced by watching which squares change
                changed = c.motion.update(compute_feats(warped, geom.cell, s.board_flipped))
                board = c.motion.board
                if c.motion.note:
                    self._say(c.motion.note)
            else:
                if not state.classifier.ready and not yolo.piece_backend_active():
                    self._say("Set up the starting position, then press 'Learn pieces'.")
                    self._stop_evt.wait(0.25)
                    return
                D = state.classifier.cost_matrix(warped, geom.cell, s.board_flipped,
                                                 s.fg_threshold, s.crop_top_extend)
                changed = c.tracker.update(D, prefer_white_to_move=(my_color == chess.WHITE))
                board = c.tracker.board

        if board is None:
            c.no_board_since = c.no_board_since or time.time()
            if not c.cleared and time.time() - c.no_board_since > 3.0:
                with self._lock:
                    self._cur = self._wanted = None
                self._engine.cancel()
                self.result_ready.emit(FrameResult(status="No board"))
                c.cleared = True
            if not s.auto_detect:
                self._say("Looking for a valid position...")
        else:
            c.no_board_since = None
            fen = board.fen()
            if changed or fen != c.last_fen or version != c.last_version:
                c.last_fen, c.last_version = fen, version
                my_turn = board.turn == my_color
                side = "your move" if my_turn else "opponent to move"
                res = FrameResult(board=board.copy(stack=False), geometry=geom, flipped=flipped, my_turn=my_turn)
                res.tactics = analyse_tactics(board)
                ask = None
                if board.is_game_over():
                    res.status = "Game over"
                else:
                    res.pending = True
                    ask = (fen, my_turn, version)
                with self._lock:
                    prev = self._cur
                    if (ask and prev is not None and prev.board is not None and prev.board.fen() == fen
                            and prev.my_turn == my_turn):
                        # same position, new request (settings changed): keep the arrows until the answer
                        res.best, res.second, res.threat = prev.best, prev.second, prev.threat
                        res.eval_text, res.status = prev.eval_text, prev.status
                    self._cur, self._wanted = res, ask
                self.result_ready.emit(res)          # before submit(): the final answer must come after
                c.cleared = False
                if ask:
                    self._engine.submit(EngineRequest(ask, board.copy(stack=False), s, my_turn))
                    self._say(f"{side} | thinking...")
                else:
                    self._engine.cancel()
                    self._say(f"{side} | {res.status}")
            elif moved and geom is not None:
                # same position, but the camera / window moved: redraw at the new place
                with self._lock:
                    out = None
                    if self._cur is not None:
                        self._cur = out = dataclasses.replace(self._cur, geometry=geom, flipped=flipped)
                if out is not None:
                    self.result_ready.emit(out)

        dt = time.perf_counter() - t0
        if dt < period:
            self._stop_evt.wait(period - dt)

    # ---- thread body -------------------------------------------------------------------------
    def run(self):
        ctx = _Ctx()
        handler = CallbackHandler(self.status.emit, logging.INFO)
        status_log.addHandler(handler)
        self._status_handler = handler
        self._last_status = ""
        self._engine = EngineService(self._on_engine)
        self._engine.start()
        backoff = Backoff(first=0.5, cap=5.0)
        log.info("pipeline started")
        try:
            while not self._stop_evt.is_set():
                try:
                    self._tick(ctx)
                    backoff.reset()
                except Exception as e:
                    log.exception("pipeline iteration failed (%d in a row)", backoff.failures + 1)
                    self._say(f"Pipeline error: {e}", logging.ERROR)
                    ctx.close_grabber()                      # most likely culprit; reopened next time
                    if backoff.failures + 1 >= RESET_AFTER_FAILURES:
                        log.error("pipeline keeps failing: rebuilding trackers")
                        ctx.tracker, ctx.motion, ctx.auto = BoardTracker(), MotionTracker(), AutoGame()
                        self._forget_position(ctx)
                        backoff.reset()
                    self._stop_evt.wait(backoff.next())
        finally:
            log.info("pipeline stopping")
            try:
                self._engine.stop(timeout=3.0)
            except Exception:
                log.warning("engine service did not stop cleanly", exc_info=True)
            ctx.close_grabber()
            status_log.removeHandler(handler)
