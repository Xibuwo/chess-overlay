"""Thread-safe shared state between the GUI thread and the pipeline worker."""
from __future__ import annotations

import copy
import threading

from config import TEMPLATES_FILE, Settings
from geometry import BoardGeometry
from recognition import PieceClassifier


class AppState:
    def __init__(self):
        self.lock = threading.RLock()
        self.settings = Settings.load()
        self.geometry: BoardGeometry | None = None
        self.classifier = PieceClassifier()
        self.classifier.load(TEMPLATES_FILE)
        self.version = 0
        self.detecting = True  # runtime only: False = worker paused (no capture, no engine)
        self._flags: set[str] = set()
        self.listeners: list = []  # called on the thread that changed a setting (GUI)
        if self.settings.corners:
            try:
                self.geometry = BoardGeometry(self.settings.corners)
            except Exception:
                self.geometry = None

    def update(self, _quiet: bool = False, **kw):
        """_quiet=True: UI-only change, do not bump `version` (which makes the worker
        re-run the engine)."""
        with self.lock:
            for k, v in kw.items():
                setattr(self.settings, k, v)
            if not _quiet:
                self.version += 1
        self.settings.save()
        for cb in list(self.listeners):
            cb()

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.settings), self.geometry, self.version

    def set_corners(self, corners, monitor_index: int):
        with self.lock:
            self.geometry = BoardGeometry(corners)
            self.settings.corners = [[float(x), float(y)] for x, y in corners]
            self.settings.corners_monitor = monitor_index
            self.version += 1
            self._flags.add("reset")
        self.settings.save()
        for cb in list(self.listeners):
            cb()

    def set_detecting(self, on: bool):
        with self.lock:
            self.detecting = bool(on)
            if on:
                self._flags.add("reset")  # resume with a clean tracker
                self.version += 1
        for cb in list(self.listeners):
            cb()

    def request(self, flag: str):
        with self.lock:
            self._flags.add(flag)

    def pop_flags(self) -> set[str]:
        with self.lock:
            f, self._flags = self._flags, set()
            return f