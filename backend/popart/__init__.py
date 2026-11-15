"""
popart: turns a portrait into a Warhol-style silkscreen, one step at a time.

It does NOT import FastAPI nor anything from `app`, so it can be used from the
API, from the experiments, from the tests or from any Python program.
"""

from popart.detection import FaceBox, FaceDetector
from popart.exceptions import InvalidImage, InvalidParameters, MissingInput, NoFaceDetected, PopArtError
from popart.imageio import Photo, count_colors, encode_png, gray_to_labels, labels_to_gray, read_gray, read_photo
from popart.steps import STEPS, ZONE_COUNT, ZONE_NAMES, Step, Work

__all__ = [
    "STEPS", "ZONE_COUNT", "ZONE_NAMES", "FaceBox", "FaceDetector", "InvalidImage", "InvalidParameters",
    "MissingInput", "NoFaceDetected", "Photo", "PopArtError", "Step", "Work", "count_colors", "encode_png",
    "gray_to_labels", "labels_to_gray", "read_gray", "read_photo",
]
