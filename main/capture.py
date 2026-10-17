"""Read-only screen capture (mss, or bettercam if installed).

Create the grabber on the thread that uses it: mss keeps per-thread Win32 handles.
"""
from __future__ import annotations

import cv2
import numpy as np


class ScreenGrabber:
    def __init__(self, backend: str = "mss", monitor_index: int = 1):
        self.backend = backend
        self.monitor_index = monitor_index
        self._sct = None
        self._cam = None
        self._last: np.ndarray | None = None
        self.mon: dict = {}
        self._open()

    def _open(self):
        import mss

        self._sct = mss.mss()
        mons = self._sct.monitors
        idx = min(max(1, self.monitor_index), len(mons) - 1)
        self.mon = dict(mons[idx])
        if self.backend == "bettercam":
            try:
                import bettercam

                self._cam = bettercam.create(output_idx=idx - 1, output_color="BGR")
            except Exception:
                self._cam = None
                self.backend = "mss"  # graceful fallback

    def grab(self) -> np.ndarray | None:
        """Returns a BGR frame of the whole monitor, or None."""
        if self._cam is not None:
            frame = self._cam.grab()
            if frame is not None:
                self._last = frame
            return self._last
        shot = self._sct.grab(self.mon)
        return cv2.cvtColor(np.asarray(shot), cv2.COLOR_BGRA2BGR)

    def close(self):
        try:
            if self._cam is not None:
                self._cam.release()
            if self._sct is not None:
                self._sct.close()
        except Exception:
            pass
