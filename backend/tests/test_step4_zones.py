"""Step 4 · Zones.    pytest tests/test_step4_zones.py -v"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from helpers import portrait
from popart import FaceBox, FaceParser, FaceParts, InvalidParameters, MissingInput, NetworkFaceParser, Work, Zones
from popart.parsing import Tuning
from popart.steps import CLOTHES, EYES, HAIR, LIPS, SKIN, ZONE_COUNT

MODEL = Path(__file__).resolve().parent.parent / "models" / "face_parsing_bisenet_resnet18.onnx"


class FixedParser(FaceParser):
    """Returns the parts received in the constructor, to test how step 4 uses them."""

    def __init__(self, parts: FaceParts) -> None:
        self.parts = parts

    def parse(self, image, person, face, tuning=None):
        return self.parts


def portrait_parts(size: int = 128) -> FaceParts:
    """The parts of the synthetic portrait of helpers.portrait(), as the network would give them."""
    skin, lips, hair, clothes = (np.zeros((size, size), np.uint8) for _ in range(4))
    cv2.ellipse(hair, (64, 40), (34, 32), 0, 0, 360, 255, -1)
    cv2.ellipse(skin, (64, 50), (22, 27), 0, 0, 360, 255, -1)
    hair[skin > 0] = 0
    cv2.ellipse(lips, (64, 66), (8, 3), 0, 0, 360, 255, -1)
    skin[lips > 0] = 0
    cv2.rectangle(clothes, (24, 84), (104, 127), 255, -1)
    return FaceParts(skin, lips, hair, clothes)


def zones(**kwargs) -> np.ndarray:
    return Zones(FixedParser(portrait_parts()), **kwargs).run(portrait()).labels


def test_finds_skin_hair_clothes_and_lips():
    labels = zones(eyes=False)
    assert labels[50, 64] == SKIN
    assert labels[12, 64] == HAIR
    assert labels[110, 64] == CLOTHES
    assert labels[66, 64] == LIPS


def test_eye_shadow_sits_above_the_eyes():
    labels = zones()
    assert (labels == EYES).any()
    rows = np.nonzero(labels == EYES)[0]
    assert rows.max() < 44  # the eyes are at y = 44: the shadow is above them


def test_lips_and_eyes_can_be_turned_off():
    labels = zones(lips=False, eyes=False)
    assert not (labels == LIPS).any() and not (labels == EYES).any()


def test_labels_are_valid_zones():
    assert zones().max() < ZONE_COUNT


def test_without_landmarks_uses_face_proportions():
    work = portrait()
    work = work.update(face=FaceBox(work.face.x, work.face.y, work.face.width, work.face.height))
    labels = Zones(FixedParser(portrait_parts())).run(work).labels
    assert (labels == EYES).any()


def test_same_input_same_result():
    assert np.array_equal(zones(), zones())


def test_needs_the_mask():
    with pytest.raises(MissingInput):
        Zones(FixedParser(portrait_parts())).run(Work(image=portrait().image))


@pytest.mark.parametrize("kwargs", [{"eye_size": 0}, {"lip_widen": 9}, {"hair": 2}, {"lip_boost": -1},
                                    {"smooth": 9}])
def test_factors_out_of_range(kwargs):
    with pytest.raises(InvalidParameters):
        Zones(FixedParser(portrait_parts()), **kwargs)


def test_widen_makes_the_lips_bigger():
    base = (zones(eyes=False, lip_widen=0) == LIPS).sum()
    assert (zones(eyes=False, lip_widen=3) == LIPS).sum() > base


def test_eye_shadow_follows_the_eye_and_grows_with_its_size():
    small = (zones(lips=False, eye_size=2) == EYES).sum()
    big = (zones(lips=False, eye_size=6) == EYES).sum()
    assert 0 < small < big


def test_zones_follow_what_the_parser_says():
    work = portrait()
    h, w = work.mask.shape
    skin, lips, hair, clothes = (np.zeros((h, w), np.uint8) for _ in range(4))
    skin[30:70, 44:84] = 255
    lips[64:68, 58:70] = 255
    hair[100:120, 30:50] = 255     # long hair on the sweater
    clothes[10:20, 60:70] = 255    # for example a hat that the parser classifies as clothes, above the chin
    labels = Zones(FixedParser(FaceParts(skin, lips, hair, clothes)), eyes=False, lip_widen=0).run(work).labels
    assert labels[50, 64] == SKIN and labels[66, 64] == LIPS
    assert labels[110, 40] == HAIR and labels[15, 65] == CLOTHES


def test_network_parser_returns_masks_of_the_image_size():
    work = portrait()
    parts = NetworkFaceParser(str(MODEL)).parse(work.image, work.mask > 0, work.face)
    for mask in (parts.skin, parts.lips, parts.hair, parts.clothes):
        assert mask.shape == work.mask.shape and set(np.unique(mask)) <= {0, 255}


def test_tuning_moves_the_border_of_the_network():
    """A value added to the score of hair (or lips) before the selection gives more hair (or lips)."""
    parser = NetworkFaceParser(str(MODEL))
    image = portrait().image
    plain = parser.labels(image)
    assert (parser.labels(image, Tuning(hair=1.5)) == 17).sum() >= (plain == 17).sum()
    assert (parser.labels(image, Tuning(lips=1.5)) >= 12).sum() >= (plain >= 12).sum()


def test_smoothing_removes_tiny_islands_and_can_be_turned_off():
    work = portrait()
    h, w = work.mask.shape
    skin = np.zeros((h, w), np.uint8)
    skin[30:70, 44:84] = 255
    skin[50, 64] = 0                     # an island of one pixel in the center of the face
    hair = np.zeros((h, w), np.uint8)
    hair[50, 64] = 255
    nothing = np.zeros((h, w), np.uint8)
    parser = FixedParser(FaceParts(skin, nothing, hair, nothing))
    rough = Zones(parser, smooth=0, eyes=False, lips=False).run(work).labels
    smooth = Zones(parser, smooth=8, eyes=False, lips=False).run(work).labels
    assert rough[50, 64] == HAIR and smooth[50, 64] == SKIN
