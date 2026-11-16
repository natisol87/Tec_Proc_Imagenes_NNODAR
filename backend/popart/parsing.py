"""
Face parsing: which pixels of the person are skin, hair, clothes and lips.

A pretrained FACE PARSING network (BiSeNet with a ResNet-18 base, a segmenter of the family of
unit 1, §6.3) executed with cv2.dnn. It assigns to each pixel one of 19 parts of a head (skin,
nose, eyes, eyebrows, lips, hair, neck, clothes...). It does not depend on the colors, so it also
works with warm light at night, lips with little color or hair of the same color as the skin.

The steps use the FaceParser interface, so the tests can give them fixed zones.

The network does not decide everything: step 4 decides the pixels without a zone with the
position of the face (hair above the chin, clothes below).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np

from popart.detection import FaceBox

Point = tuple[int, int]


@dataclass(frozen=True)
class Tuning:
    """Factors of step 4 that move the borders decided by the network."""

    hair: float = 0.0           # + more hair, − more skin (added to the score of "hair")
    lips: float = 0.0           # + more lips (added to the scores of the two lips)
    brows_as_hair: bool = True  # eyebrows with the color of the hair, or as skin


@dataclass(frozen=True)
class FaceParts:
    """The parts of the person as 0/255 masks."""

    skin: np.ndarray
    lips: np.ndarray
    hair: np.ndarray
    clothes: np.ndarray


class FaceParser(ABC):
    @abstractmethod
    def parse(self, image: np.ndarray, person: np.ndarray, face: FaceBox, tuning: Tuning = Tuning()) -> FaceParts: ...


def mask_of(condition: np.ndarray) -> np.ndarray:
    return np.where(condition, 255, 0).astype(np.uint8)


class NetworkFaceParser(FaceParser):
    """BiSeNet face parsing (classes of CelebAMask-HQ). RGB image → one of 19 labels for each pixel."""

    INPUT_SIZE = (512, 512)
    MEAN = np.array([0.485, 0.456, 0.406], np.float32)  # the normalization used in the training
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    # The 19 classes of the model, in groups that are the zones of a portrait of Warhol. The eyes, nose,
    # ears and mouth (teeth) are painted as skin: their details come from the ink of step 2.
    SKIN_PARTS = (1, 4, 5, 6, 7, 8, 9, 10, 11, 14)  # skin, eyes, glasses, ears, earring, nose, mouth, neck
    BROWS = (2, 3)                                  # eyebrows: hair or skin, as the user selects
    LIP_PARTS = (12, 13)                            # upper lip, lower lip
    HAIR_PARTS = (17, 18)                           # hair, hat
    CLOTHES_PARTS = (15, 16)                        # necklace, cloth
    HAIR = 17

    def __init__(self, model_path: str) -> None:
        self.net = cv2.dnn.readNet(str(model_path))

    def labels(self, image: np.ndarray, tuning: Tuning = Tuning()) -> np.ndarray:
        """
        The label (0..18) of each pixel, with the size of the image. Each pixel gets the class with
        the highest score; the tuning adds a small value to the score of hair or lips before the
        selection. This moves the border between two zones, like changing a threshold (unit 5).
        """
        h, w = image.shape[:2]
        x = cv2.resize(image, self.INPUT_SIZE, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
        x = (x - self.MEAN) / self.STD
        self.net.setInput(x.transpose(2, 0, 1)[None])  # HWC → 1×C×H×W, the format of the network
        scores = self.net.forward()[0]                 # 19 × 512 × 512: one score for each class
        scores[self.HAIR] += tuning.hair
        for lip in self.LIP_PARTS:
            scores[lip] += tuning.lips
        labels = scores.argmax(axis=0).astype(np.uint8)
        # Labels are categories: when they are enlarged, there must not be intermediate values (INTER_NEAREST, unit 3).
        return cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)

    def parse(self, image: np.ndarray, person: np.ndarray, face: FaceBox, tuning: Tuning = Tuning()) -> FaceParts:
        labels = self.labels(image, tuning)
        hair_parts = self.HAIR_PARTS + (self.BROWS if tuning.brows_as_hair else ())
        skin_parts = self.SKIN_PARTS + (() if tuning.brows_as_hair else self.BROWS)
        hair = np.isin(labels, hair_parts) & person
        # A "hat" or "hair" below the chin is usually a collar or a helmet: step 4 decides it,
        # and it paints the part below the chin as clothes. Long hair on the shoulders is kept.
        chin = min(face.y + int(face.height * 1.05), labels.shape[0])
        hair[chin:] &= labels[chin:] == 17
        # Below the chin only "skin" and "neck" are valid: the network can classify a brilliant collar
        # or a helmet as "glasses" or "earring", and these classes are only possible on the face.
        skin = np.isin(labels, skin_parts) & person
        skin[chin:] &= np.isin(labels[chin:], (1, 14))
        return FaceParts(skin=self.connected_to_face(mask_of(skin), face),
                         lips=mask_of(np.isin(labels, self.LIP_PARTS) & person),
                         hair=mask_of(hair),
                         clothes=mask_of(np.isin(labels, self.CLOTHES_PARTS) & person))

    @staticmethod
    def connected_to_face(skin: np.ndarray, face: FaceBox) -> np.ndarray:
        """Connected components (unit 1, §4): only the skin connected to the face (face, neck, ears)."""
        count, components, stats, _ = cv2.connectedComponentsWithStats(skin)
        x0, y0 = max(face.x, 0), max(face.y, 0)
        inside = components[y0:face.y + face.height, x0:face.x + face.width]
        keep = [i for i in range(1, count) if (inside == i).any()]
        return mask_of(np.isin(components, keep))
