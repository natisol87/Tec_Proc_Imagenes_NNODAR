"""
Face detection with YuNet: a small pretrained CNN (unit 1, §6.3, detectors with deep learning)
in an .onnx file, executed with cv2.FaceDetectorYN. With the rectangle of the face it returns
the centers of the eyes: step 1 uses them to make the eyes horizontal and step 4 to locate the
eye shadow.

The steps use the FaceDetector interface, so the tests can give them a detector with fixed faces.
The model file is not part of the core: the code that creates the detector gives the path.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np


Point = tuple[int, int]


@dataclass(frozen=True)
class FaceBox:
    """Rectangle of a face, in pixel coordinates, with the centers of the eyes if they are known."""

    x: int
    y: int
    width: int
    height: int
    eyes: tuple[Point, Point] | None = None   # centers of the two eyes (None if they are not known)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2, self.y + self.height / 2


class FaceDetector(ABC):
    """Detects faces in an RGB image."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[FaceBox]:
        """All the faces detected, sorted by area (the biggest first)."""

    @staticmethod
    def _sorted(faces: list[FaceBox]) -> list[FaceBox]:
        return sorted(faces, key=lambda f: f.area, reverse=True)


class YuNetFaceDetector(FaceDetector):
    def __init__(self, model_path: str, score_threshold: float = 0.8, nms_threshold: float = 0.3) -> None:
        # The input size here is temporary; detect() defines the real size for each image.
        self.model = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), score_threshold, nms_threshold)

    def detect(self, image: np.ndarray) -> list[FaceBox]:
        h, w = image.shape[:2]
        self.model.setInputSize((w, h))
        _, rows = self.model.detect(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # the model uses BGR
        if rows is None:
            return []
        faces = []
        for row in rows:  # x, y, w, h, right eye, left eye, nose, right corner of the mouth, left corner of the mouth, score
            # (only the rectangle and the eyes are used)
            x, y, bw, bh = (int(round(v)) for v in row[:4])
            x, y = max(x, 0), max(y, 0)

            def point(i):  # the point (x, y) in columns i and i + 1 of the row, as integers
                return int(round(row[i])), int(round(row[i + 1]))

            faces.append(FaceBox(x, y, min(bw, w - x), min(bh, h - y),
                                 eyes=(point(4), point(6))))
        return self._sorted(faces)
