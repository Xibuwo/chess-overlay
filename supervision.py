"""Qt-free restart / back-off policies used by worker.py and main.py (so they can be unit-tested)."""
from __future__ import annotations

import time
from collections import deque


class Backoff:
    """Growing delay for a loop that keeps failing: 0.5 s, 1 s, 2 s ... capped; reset on success."""

    def __init__(self, first: float = 0.5, cap: float = 5.0, factor: float = 2.0):
        self.first, self.cap, self.factor = first, cap, factor
        self.failures = 0
        self._next = first

    def next(self) -> float:
        self.failures += 1
        d = self._next
        self._next = min(self.cap, self._next * self.factor)
        return d

    def reset(self) -> None:
        self.failures = 0
        self._next = self.first


class RestartPolicy:
    """Decides whether a dead worker thread may be restarted: at most `max_restarts` within
    `window` seconds (a crash loop is reported and left stopped instead of spinning), and always
    after a short, growing delay."""

    def __init__(self, max_restarts: int = 5, window: float = 120.0, clock=time.monotonic):
        self.max_restarts, self.window, self._clock = max_restarts, window, clock
        self._times: deque = deque()
        self.backoff = Backoff(first=0.5, cap=10.0)
        self._not_before = 0.0

    def allow(self) -> tuple[bool, str]:
        now = self._clock()
        while self._times and now - self._times[0] > self.window:
            self._times.popleft()
        if len(self._times) >= self.max_restarts:
            return False, f"{self.max_restarts} restarts in {int(self.window)} s"
        if now < self._not_before:
            return False, "waiting"
        return True, ""

    def record_restart(self) -> None:
        now = self._clock()
        self._times.append(now)
        self._not_before = now + self.backoff.next()

    def note_healthy(self) -> None:
        """Call while the worker has been running fine: forgives old crashes."""
        if self._times and self._clock() - self._times[-1] > self.window:
            self._times.clear()
            self.backoff.reset()
