"""Step 3 · Background.    pytest tests/test_step3_background.py -v"""

import numpy as np
import pytest

from helpers import disc_mask, figure_image
from popart import Background, FaceBox, InvalidParameters, Work
from pathlib import Path

from popart.segmentation import ModelSegmenter, PersonModel, Segmenter, clean_mask

MODEL = Path(__file__).resolve().parent.parent / "models" / "human_segmentation_pphumanseg_2023mar.onnx"

FACE = FaceBox(52, 52, 24, 24)  # the center of the dark disc of figure_image


class FixedSegmenter(Segmenter):
    def __init__(self, mask):
        self.mask = mask

    def segment(self, image, face):
        return self.mask.copy()


def test_mask_is_binary_and_same_size():
    work = Background(FixedSegmenter(disc_mask())).run(Work(image=figure_image()))
    assert work.mask.shape == (128, 128)
    assert set(np.unique(work.mask)) <= {0, 255}


def test_specks_of_background_are_removed():
    raw = disc_mask()
    raw[5:8, 5:8] = 255  # a small island far from the person
    clean = clean_mask(raw, smoothness=0, edge=0)
    assert clean[6, 6] == 0
    assert clean[64, 64] == 255


def test_holes_inside_the_person_are_filled():
    raw = disc_mask()
    raw[60:68, 60:68] = 0
    assert clean_mask(raw, smoothness=0, edge=0)[64, 64] == 255


def test_edge_grows_and_shrinks_the_silhouette():
    raw = disc_mask()
    grown = clean_mask(raw, smoothness=0, edge=4)
    shrunk = clean_mask(raw, smoothness=0, edge=-4)
    assert (grown > 0).sum() > (raw > 0).sum() > (shrunk > 0).sum()


def test_network_segmenter_returns_a_mask():
    mask = ModelSegmenter(PersonModel(str(MODEL))).segment(figure_image(), FACE)
    assert mask.shape == (128, 128)
    assert set(np.unique(mask)) <= {0, 255}


def test_network_mask_does_not_cover_the_whole_image():
    """Old error: watershed marks the border of the image with -1; it must not be 'person'."""
    mask = Background(ModelSegmenter(PersonModel(str(MODEL)))).run(Work(image=figure_image(), face=FACE)).mask
    assert mask[2, 2] == 0 and mask[64, 64] == 255
    assert (mask > 0).mean() < 0.9


@pytest.mark.parametrize("kwargs", [{"smoothness": 11}, {"edge": -11}])
def test_factors_out_of_range(kwargs):
    with pytest.raises(InvalidParameters):
        Background(FixedSegmenter(disc_mask()), **kwargs)
