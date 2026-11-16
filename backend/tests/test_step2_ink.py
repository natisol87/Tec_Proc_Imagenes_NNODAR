"""Step 2 · Ink (grayscale).    pytest tests/test_step2_ink.py -v"""

import numpy as np
import pytest

from helpers import figure_image, gradient_image
from popart import Ink, InvalidParameters, MissingInput, Work


def ink(image=None, **kwargs) -> np.ndarray:
    return Ink(**kwargs).run(Work(image=figure_image() if image is None else image)).edges


def test_same_size_as_the_image():
    assert ink().shape == (128, 128)


def test_more_ink_on_dark_areas():
    image, density = figure_image(), ink()
    dark = image[:, :, 0] < 60    # dark disc
    light = image[:, :, 0] > 220  # light background
    assert density[dark].mean() > density[light].mean()


def test_more_shadows_means_less_ink():
    """On a gradient (with all the middle tones) a bigger exponent makes the middle tones lighter."""
    soft = ink(gradient_image(), shadows=1.0, contrast=0, smoothing=0)
    hard = ink(gradient_image(), shadows=3.0, contrast=0, smoothing=0)
    assert hard.sum() < soft.sum()


def test_darkest_tones_get_full_ink():
    assert ink().max() == 255


def test_ink_table_is_a_negative_with_a_power_curve():
    table = Ink(shadows=2.0).ink_table()
    assert table[0] == 255 and table[255] == 0  # black = maximum ink, white = no ink
    assert table[128] < 255 - 128               # middle tones get less ink than with a simple negative


def test_needs_the_image():
    with pytest.raises(MissingInput):
        Ink().run(Work())


@pytest.mark.parametrize("kwargs", [{"shadows": 0.5}, {"smoothing": 11}, {"contrast": -1}])
def test_factors_out_of_range(kwargs):
    with pytest.raises(InvalidParameters):
        Ink(**kwargs)
