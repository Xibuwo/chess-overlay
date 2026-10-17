"""Perspective calibration: angled/3D board  ->  flat 2D grid (and back)."""
from __future__ import annotations

import chess
import cv2
import numpy as np


class BoardGeometry:
    """corners: 4 points in *monitor-local physical pixels*, clicked in the order
    top-left, top-right, bottom-right, bottom-left as seen on screen (outer corners
    of the 8x8 playing area)."""

    def __init__(self, corners, size: int = 640):
        self.corners = np.array(corners, dtype=np.float32).reshape(4, 2)
        self.size = int(size)
        self.cell = self.size / 8.0
        dst = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
        self.M = cv2.getPerspectiveTransform(self.corners, dst)      # screen -> flat
        self.Minv = cv2.getPerspectiveTransform(dst, self.corners)   # flat -> screen

    # ---- image side -------------------------------------------------------
    def warp(self, frame: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(frame, self.M, (self.size, self.size), flags=cv2.INTER_LINEAR)

    # ---- square <-> grid cell --------------------------------------------
    @staticmethod
    def cell_of(square: int, flipped: bool) -> tuple[int, int]:
        f, r = chess.square_file(square), chess.square_rank(square)
        return (r, 7 - f) if flipped else (7 - r, f)  # (row, col), row 0 = top of screen

    @staticmethod
    def square_of(row: int, col: int, flipped: bool) -> int:
        if flipped:
            return chess.square(7 - col, row)
        return chess.square(col, 7 - row)

    # ---- overlay side -----------------------------------------------------
    def to_screen(self, pts) -> np.ndarray:
        arr = np.float32(pts).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(arr, self.Minv).reshape(-1, 2)

    def square_polygon(self, square: int, flipped: bool) -> np.ndarray:
        r, c = self.cell_of(square, flipped)
        k = self.cell
        quad = [(c * k, r * k), ((c + 1) * k, r * k), ((c + 1) * k, (r + 1) * k), (c * k, (r + 1) * k)]
        return self.to_screen(quad)

    def square_center(self, square: int, flipped: bool) -> np.ndarray:
        r, c = self.cell_of(square, flipped)
        k = self.cell
        return self.to_screen([((c + 0.5) * k, (r + 0.5) * k)])[0]
