"""
Creates the steps from a set of factors and executes all of them, one after the other.

The web application executes the steps ONE AT A TIME (the user adjusts and saves each one);
this module executes the complete sequence, for the experiments and the tests. Both use the same Step classes.
"""

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

from popart.detection import FaceDetector, YuNetFaceDetector
from popart.palettes import DEFAULT_PALETTES
from popart.parsing import FaceParser, NetworkFaceParser
from popart.segmentation import ModelSegmenter, PersonModel, Segmenter
from popart.steps import Background, Compose, Frame, Ink, Step, Work, Zones


class Models:
    """
    Loads the three networks only once (YuNet, PP-HumanSeg and face parsing) and gives them to
    the steps that need them.
    """

    def __init__(self, yunet_path: Path, segmentation_path: Path, parsing_path: Path) -> None:
        self.yunet_path = yunet_path
        self.segmentation_path = segmentation_path
        self.parsing_path = parsing_path

    @cached_property
    def _yunet(self) -> YuNetFaceDetector:
        return YuNetFaceDetector(str(self.yunet_path))

    @cached_property
    def _person_model(self) -> PersonModel:
        return PersonModel(str(self.segmentation_path))

    @cached_property
    def _face_parser(self) -> NetworkFaceParser:
        return NetworkFaceParser(str(self.parsing_path))

    def parser(self) -> FaceParser:
        return self._face_parser

    def detector(self) -> FaceDetector:
        return self._yunet

    def segmenter(self) -> Segmenter:
        return ModelSegmenter(self._person_model)


@dataclass(frozen=True)
class PopArtOptions:
    """All the factors of all the steps, with the initial values of the web application."""

    # 1 · Frame
    size: int = 512
    margin: float = 0.9
    offset: float = 0.25
    sharpen: float = 0.5
    face: int = 0
    straighten: bool = True
    # 2 · Ink (grayscale)
    smoothing: int = 4
    contrast: float = 2.0
    shadows: float = 1.6
    # 3 · Background
    smoothness: int = 3
    edge: int = 0
    # 4 · Zones
    hair: float = 0.0
    lip_boost: float = 0.0
    brows_as_hair: bool = True
    smooth: int = 4
    eyes: bool = True
    eye_size: int = 6
    lips: bool = True
    lip_widen: int = 1
    # 5 · Compose
    palettes: tuple[str, ...] = DEFAULT_PALETTES
    rows: int = 2
    cols: int = 2
    outline: bool = True


@dataclass(frozen=True)
class StepResult:
    step: Step
    work: Work           # all the results until this step
    preview: np.ndarray  # the result of this step, as an RGB image


@dataclass
class Trace:
    results: list[StepResult] = field(default_factory=list)

    @property
    def final(self) -> np.ndarray:
        return self.results[-1].work.final


class PopArtMaker:
    def __init__(self, models: Models) -> None:
        self.models = models

    def steps(self, o: PopArtOptions) -> list[Step]:
        return [
            Frame(self.models.detector(), o.margin, o.offset, o.size, o.sharpen, o.face, o.straighten),
            Ink(o.smoothing, o.contrast, o.shadows),
            Background(self.models.segmenter(), o.smoothness, o.edge),
            Zones(self.models.parser(), o.hair, o.lip_boost, o.brows_as_hair, o.smooth, o.eyes, o.eye_size, o.lips,
                  o.lip_widen),
            Compose(list(o.palettes), o.rows, o.cols, o.outline),
        ]

    def run(self, photo: np.ndarray, options: PopArtOptions = PopArtOptions()) -> Trace:
        work, trace = Work(photo=photo), Trace()
        for step in self.steps(options):
            work = step.run(work)
            trace.results.append(StepResult(step, work, step.preview(work)))
        return trace
