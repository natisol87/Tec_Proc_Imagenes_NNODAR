"""
popart: changes a portrait into a silkscreen in the style of Warhol, one step at a time.

It does NOT import FastAPI or anything from `app`, so the API, the experiments,
the tests or any other Python program can use it.
"""

from popart.detection import FaceBox, FaceDetector, YuNetFaceDetector
from popart.exceptions import InvalidImage, InvalidParameters, MissingInput, NoFaceDetected, PopArtError
from popart.imageio import Photo, count_colors, encode_png, gray_to_labels, labels_to_gray, read_gray, read_photo
from popart.parsing import FaceParser, FaceParts, NetworkFaceParser
from popart.palettes import DEFAULT_PALETTES, PALETTES, Palette
from popart.pipeline import Models, PopArtMaker, PopArtOptions, Trace
from popart.steps import STEPS, ZONE_COUNT, ZONE_NAMES, Background, Compose, Frame, Ink, Step, Work, Zones

__all__ = [
    "DEFAULT_PALETTES", "PALETTES", "STEPS", "ZONE_COUNT", "ZONE_NAMES",
    "Background", "Compose", "FaceParser", "FaceParts", "NetworkFaceParser", "FaceBox", "FaceDetector", "Frame", "Ink",
    "InvalidImage", "InvalidParameters", "MissingInput", "Models", "NoFaceDetected", "Palette", "Photo",
    "PopArtError", "PopArtMaker", "PopArtOptions", "Step", "Trace", "Work", "YuNetFaceDetector", "Zones",
    "count_colors", "encode_png", "gray_to_labels", "labels_to_gray", "read_gray", "read_photo",
]
