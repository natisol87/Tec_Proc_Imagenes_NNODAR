"""
Face boxes and the interface every face detector follows (FaceDetector).
The detectors themselves come with step 1.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


Point = tuple[int, int]


@dataclass(frozen=True)
class FaceBox:
    """Axis-aligned bounding box of a face, in pixel coordinates, with optional landmarks."""

    x: int
    y: int
    width: int
    height: int
    eyes: tuple[Point, Point] | None = None   # centers of both eyes (None if the cascade missed one)
    mouth: tuple[Point, Point] | None = None  # corners of the mouth (estimated in step 4 if None)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2, self.y + self.height / 2


class FaceDetector(ABC):
    """Finds faces in an RGB image."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[FaceBox]:
        """Every face found, sorted by area (largest first)."""

    @staticmethod
    def _sorted(faces: list[FaceBox]) -> list[FaceBox]:
        return sorted(faces, key=lambda f: f.area, reverse=True)
