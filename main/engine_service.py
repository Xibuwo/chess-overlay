"""Engine on its own thread with a "latest position wins" mailbox.

The pipeline thread never waits for the engine: it `submit()`s a request and carries on reading
frames. The mailbox holds at most ONE pending request - a newer submit silently replaces an older
one that has not started (it is stale: the board has moved on). A request that is already running
cannot be un-thought, but if a newer one arrived meanwhile its result is dropped instead of
delivered. Results come back through `on_result(EngineResponse)`, called on the engine thread.

Errors: a failing analysis is logged, delivered as a response with `error` set (the manager has
already closed the broken engine and relaunches it on the next request - the existing
auto-restart pattern in engines.py), and the thread keeps serving.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable

import chess

from config import Settings
from engines import EngineManager, EngineResult
from logsetup import get_logger

log = get_logger("engine_service")


@dataclass
class EngineRequest:
    key: tuple                  # (fen, my_turn, version) - identity of the question
    board: chess.Board
    settings: Settings
    my_turn: bool


@dataclass
class EngineResponse:
    key: tuple
    result: EngineResult
    error: str = ""
    elapsed: float = 0.0


class EngineService:
    def __init__(self, on_result: Callable[[EngineResponse], None], manager_factory=EngineManager,
                 name: str = "engine"):
        self._on_result = on_result
        self._mgr = manager_factory()
        self._cv = threading.Condition()
        self._latest: EngineRequest | None = None
        self._gen = 0                 # bumped by cancel(): in-flight results of older gens are dropped
        self._stop = False
        self._release = False
        self._thread = threading.Thread(target=self._loop, name=name, daemon=True)
        self.submitted = self.dropped_stale = self.delivered = 0

    # ---- control (any thread) --------------------------------------------------
    def start(self) -> None:
        self._thread.start()

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def submit(self, req: EngineRequest) -> None:
        with self._cv:
            if self._latest is not None:
                self.dropped_stale += 1
                log.debug("dropping stale request %s", self._latest.key[0])
            self._latest = req
            self.submitted += 1
            self._cv.notify()

    def cancel(self) -> None:
        """Forget the pending request and discard the result of the one being computed."""
        with self._cv:
            self._latest = None
            self._gen += 1
            self._cv.notify()

    def release(self) -> None:
        """cancel() + close the engine process once idle (detection paused)."""
        with self._cv:
            self._latest = None
            self._gen += 1
            self._release = True
            self._cv.notify()

    def stop(self, timeout: float = 5.0) -> None:
        with self._cv:
            self._stop = True
            self._latest = None
            self._gen += 1
            self._cv.notify_all()
        if not self._thread.is_alive():
            self._mgr.close()
            return
        self._thread.join(timeout)
        if self._thread.is_alive():
            log.warning("engine thread still busy after %.1fs: killing the engine process", timeout)
            self._mgr.kill()
            self._thread.join(2.0)
            if self._thread.is_alive():
                log.error("engine thread did not stop")

    # ---- the thread ------------------------------------------------------------
    def _loop(self) -> None:
        try:
            while True:
                with self._cv:
                    while self._latest is None and not self._stop and not self._release:
                        self._cv.wait()
                    if self._stop:
                        break
                    req, self._latest = self._latest, None
                    gen = self._gen
                    do_release = self._release and req is None
                    if req is None:
                        self._release = False
                if req is None:
                    if do_release:
                        self._mgr.close()
                    continue
                self._run(req, gen)
        except Exception:
            log.critical("engine thread crashed", exc_info=True)
        finally:
            try:
                self._mgr.close()
            except Exception:
                log.warning("closing the engine at shutdown failed", exc_info=True)

    def _run(self, req: EngineRequest, gen: int) -> None:
        t0 = time.perf_counter()
        error = ""
        try:
            result = self._mgr.analyse(req.board, req.settings, req.my_turn)
            error = self._mgr.error or ""
        except Exception as e:              # analyse() handles the usual engine failures itself
            log.error("analysis crashed for %s", req.key[0], exc_info=True)
            self._mgr.close()
            result, error = EngineResult(note=f"Engine error: {e}"), f"Engine error: {e}"
        elapsed = time.perf_counter() - t0
        with self._cv:
            stale = self._stop or gen != self._gen or self._latest is not None
        if stale:
            self.dropped_stale += 1
            log.debug("discarding stale result for %s (%.0f ms)", req.key[0], elapsed * 1000)
            return
        if error:
            log.warning("engine error: %s", error)
        self.delivered += 1
        try:
            self._on_result(EngineResponse(req.key, result, error, elapsed))
        except Exception:
            log.error("result callback failed", exc_info=True)
