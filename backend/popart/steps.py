"""
The steps that turn a portrait into a Warhol-style silkscreen.

The order follows how a silkscreen is made: first the black screen (the photo
in grayscale, the "ink"), then the background is cleaned, and finally the
parts of the portrait (skin, hair, clothes, lips, eye shadow) are defined and
printed in flat colors under the ink.

Every step is a class with the same interface (`Step`). Its FACTORS (the
parameters the user tunes) are given to the constructor, which validates them.
`apply(work)` receives the work done so far and returns a NEW `Work` with the
fields the step produces; nothing is modified in place. That is what lets the
frontend run one step at a time, keep ("save") each result, and go back to an
earlier step to try other factors without starting over.

Conventions:
  - Images are RGB uint8 arrays (h, w, 3); masks, zone maps and the ink layer
    are (h, w) uint8.
  - A step only reads the fields listed in `requires` (and `optional`) and only
    writes the ones in `produces`.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import ClassVar

import numpy as np

from popart.detection import FaceBox
from popart.exceptions import MissingInput


@dataclass(frozen=True)
class Work:
    """Everything produced so far. Each step fills some of these fields."""

    photo: np.ndarray | None = None   # the input photo, RGB, any size
    image: np.ndarray | None = None   # the square tile being worked on, RGB
    face: FaceBox | None = None       # face box in tile coordinates
    faces: tuple[FaceBox, ...] | None = None  # every face found in the photo, left to right (photo coordinates)
    edges: np.ndarray | None = None   # ink density: 0 = no ink, 255 = full black
    mask: np.ndarray | None = None    # 255 = person, 0 = background
    labels: np.ndarray | None = None  # zone of each pixel: HAIR, SKIN, CLOTHES, LIPS or EYES
    final: np.ndarray | None = None   # the finished grid, RGB

    def update(self, **fields) -> "Work":
        return replace(self, **fields)


class Step(ABC):
    number: ClassVar[int]
    name: ClassVar[str]       # id used by the API: "frame"
    label: ClassVar[str]      # name shown in the frontend: "Encuadre"
    requires: ClassVar[tuple[str, ...]]
    optional: ClassVar[tuple[str, ...]] = ()
    produces: ClassVar[tuple[str, ...]]

    def run(self, work: Work) -> Work:
        """Checks that the inputs exist and applies the step."""
        for field in self.requires:
            if getattr(work, field) is None:
                raise MissingInput(self.label, field)
        return self.apply(work)

    @abstractmethod
    def apply(self, work: Work) -> Work:
        """Returns a NEW Work with the fields in `produces` filled."""

    @abstractmethod
    def preview(self, work: Work) -> np.ndarray:
        """An RGB image that shows what this step did (to display it)."""


STEPS: tuple[type[Step], ...] = ()
