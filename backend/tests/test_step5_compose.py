"""Step 5 · Color and grid, and the palettes.    pytest tests/test_step5_compose.py -v"""

import numpy as np
import pytest

from helpers import colors_of, work_for_compose
from popart import PALETTES, Compose, InvalidParameters
from popart.palettes import delta_e
from popart.steps import EYES

SIZE = 128


def test_grid_size():
    final = Compose(["marilyn"], rows=2, cols=3).run(work_for_compose()).final
    assert final.shape == (2 * SIZE, 3 * SIZE, 3)


def test_background_is_flat_with_the_palette_background():
    work = work_for_compose().update(edges=None)
    tile = Compose(["turquoise"], 1, 1, outline=False).run(work).final
    assert colors_of(tile[work.mask == 0]) == {tuple(PALETTES["turquoise"].background_rgb)}


def test_zones_are_painted_with_the_zone_colors():
    tile = Compose(["violet"], 1, 1, outline=False).run(work_for_compose().update(edges=None)).final
    palette = PALETTES["violet"]
    allowed = {tuple(palette.background_rgb)} | {tuple(c) for c in palette.zone_colors}
    assert colors_of(tile) <= allowed


def test_ink_is_printed_by_multiplication():
    """Density 128 makes each color 50% darker: the color is still visible."""
    work = work_for_compose()
    work = work.update(edges=np.full_like(work.mask, 128))
    without = Compose(["marilyn"], 1, 1, outline=False).run(work.update(edges=None)).final.astype(int)
    with_ink = Compose(["marilyn"], 1, 1, outline=False).run(work).final.astype(int)
    inked = (work.mask > 0) & (work.labels != EYES)  # the eye shadow is a solid color (next test)
    expected = np.rint(without * (1 - 128 / 255))
    assert np.abs(with_ink[inked] - expected[inked]).max() <= 1


def test_no_ink_on_the_eye_shadow():
    work = work_for_compose().update(edges=np.full((SIZE, SIZE), 255, np.uint8))
    tile = Compose(["marilyn"], 1, 1, outline=False).run(work).final
    shadow = (work.mask > 0) & (work.labels == EYES)
    assert colors_of(tile[shadow]) == {tuple(PALETTES["marilyn"].zone_colors[EYES])}


def test_no_ink_on_the_background():
    work = work_for_compose().update(edges=np.full((SIZE, SIZE), 255, np.uint8))
    tile = Compose(["marilyn"], 1, 1, outline=False).run(work).final
    assert colors_of(tile[work.mask == 0]) == {tuple(PALETTES["marilyn"].background_rgb)}


def test_a_thicker_outline_covers_more_pixels():
    work = work_for_compose(512).update(edges=None)  # the widths are calibrated for portraits of 512 px
    thin = (Compose(["marilyn"], 1, 1, outline=1).ink_layer(work) > 0).sum()
    thick = (Compose(["marilyn"], 1, 1, outline=5).ink_layer(work) > 0).sum()
    none = (Compose(["marilyn"], 1, 1, outline=0).ink_layer(work) > 0).sum()
    assert none == 0 < thin < thick


def test_opacity_mixes_the_color_with_the_photo():
    """Opacity 0.5: each painted pixel is the average of the color of its zone and the photo (unit 4)."""
    work = work_for_compose().update(edges=None)
    photo = np.full((SIZE, SIZE, 3), 100, np.uint8)
    flat = Compose(["marilyn"], 1, 1, outline=0).run(work).final.astype(int)
    mixed = Compose(["marilyn"], 1, 1, outline=0, opacity=0.5).run(work.update(image=photo)).final.astype(int)
    person = work.mask > 0
    assert np.abs(mixed[person] - (flat[person] + 100) / 2).max() <= 1
    assert np.array_equal(mixed[~person], flat[~person])  # the background stays flat


def test_outline_is_dilation_minus_erosion():
    import cv2
    work = work_for_compose().update(edges=None)
    density = Compose(["marilyn"], 1, 1, outline=True).ink_layer(work)
    kernel = np.ones((3, 3), np.uint8)
    expected = cv2.subtract(cv2.dilate(work.mask, kernel), cv2.erode(work.mask, kernel))
    assert np.array_equal(density, expected)


def test_custom_colors_replace_the_palette_colors():
    from popart.steps import EYES, LIPS, SKIN
    work = work_for_compose()
    custom = [{"skin": "#123456", "lips": "#abcdef", "eyes": "#00ff00"}]
    tile = Compose(["marilyn"], 1, 1, outline=False, custom=custom).run(work.update(edges=None)).final
    person = work.mask > 0
    assert tuple(tile[(work.labels == SKIN) & person][0]) == (0x12, 0x34, 0x56)
    assert tuple(tile[(work.labels == LIPS) & person][0]) == (0xAB, 0xCD, 0xEF)
    assert tuple(tile[(work.labels == EYES) & person][0]) == (0, 255, 0)


def test_custom_colors_are_per_tile_and_do_not_change_the_palette():
    compose = Compose(["marilyn", "marilyn"], 1, 2, custom=[{"skin": "#000000"}, {}])
    assert tuple(compose.tile_colors(0, "marilyn")[1]) == (0, 0, 0)
    assert tuple(compose.tile_colors(1, "marilyn")[1]) == tuple(PALETTES["marilyn"].zone_colors[1])


@pytest.mark.parametrize("custom", [[{"skin": "rojo"}], [{"clothes": "#000000"}]])
def test_invalid_custom_colors(custom):
    with pytest.raises(InvalidParameters):
        Compose(["marilyn"], custom=custom)


def test_hair_color_can_be_chosen():
    from popart.steps import HAIR
    compose = Compose(["marilyn"], custom=[{"hair": "#1b1464"}])
    assert tuple(compose.tile_colors(0, "marilyn")[HAIR]) == (0x1B, 0x14, 0x64)


def test_swatches_are_valid_and_unique():
    from popart.palettes import SWATCHES
    colors = [c for c, _ in SWATCHES]
    assert len(colors) == len(set(colors)) >= 20
    Compose(["marilyn"], custom=[{"skin": c} for c in colors])  # all are accepted


def test_palettes_cycle_over_the_tiles():
    assert Compose(["marilyn", "red"], rows=2, cols=2).tile_palettes() == ["marilyn", "red", "marilyn", "red"]


def test_unknown_palette():
    with pytest.raises(InvalidParameters):
        Compose(["sepia"])


# ---------------------------------------------------------------------------
# Palettes: the contrast rules of Warhol
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(PALETTES))
def test_background_contrasts_with_the_big_zones(name):
    palette = PALETTES[name]
    hair, skin, clothes, _, _ = palette.zone_colors
    for zone in (hair, skin, clothes):
        assert delta_e(palette.background_rgb, zone) >= 45, f"{name}: background too close to {zone}"


@pytest.mark.parametrize("name", list(PALETTES))
def test_zones_contrast_with_each_other(name):
    hair, skin, clothes, lips, eyes = PALETTES[name].zone_colors
    assert delta_e(skin, hair) >= 35
    assert delta_e(skin, clothes) >= 35
    assert delta_e(hair, clothes) >= 30
    assert delta_e(lips, skin) >= 35
    assert delta_e(eyes, skin) >= 35
