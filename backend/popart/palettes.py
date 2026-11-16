"""
The palettes of Warhol.

In the silkscreens of Marilyn and Liz each area is printed with a flat ink that
CONTRASTS strongly with its neighbors: an intense pink background with a turquoise face,
yellow hair with an orange background. Here a palette is a background color plus
one color for each zone of the portrait: hair, skin, clothes, lips and eye
shadow. The tests verify the contrast with the CIE76 color distance (ΔE in the Lab
space, where distance ≈ difference that the eye perceives):

  1. the background is very different from hair, skin and clothes (ΔE ≥ 45);
  2. the skin is different from hair and clothes (ΔE ≥ 35), the hair from the clothes (ΔE ≥ 30),
     and the lips and eye shadow from the skin around them (ΔE ≥ 35).
"""

from dataclasses import dataclass

import cv2
import numpy as np


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    return tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))


def to_lab(colors: np.ndarray) -> np.ndarray:
    """(n, 3) uint8 RGB → (n, 3) float Lab with L in 0..100 (the float convention of OpenCV)."""
    rgb = colors.reshape(1, -1, 3).astype(np.float32) / 255
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab).reshape(-1, 3)


def delta_e(color_a: np.ndarray, color_b: np.ndarray) -> float:
    """CIE76 distance between two RGB colors."""
    lab = to_lab(np.array([color_a, color_b], dtype=np.uint8))
    return float(np.linalg.norm(lab[0] - lab[1]))


@dataclass(frozen=True)
class Palette:
    name: str                    # identifier for the API: "marilyn"
    label: str                   # name visible in the frontend: "Marilyn rosa"
    background: str              # "#rrggbb"
    zones: tuple[str, ...]       # "#rrggbb" for hair, skin, clothes, lips and eye shadow

    @property
    def background_rgb(self) -> np.ndarray:
        return np.array(hex_to_rgb(self.background), dtype=np.uint8)

    @property
    def zone_colors(self) -> np.ndarray:
        """Hair, skin, clothes, lips and eye shadow as a (5, 3) uint8 RGB array (a LUT)."""
        return np.array([hex_to_rgb(c) for c in self.zones], dtype=np.uint8)


PALETTES: dict[str, Palette] = {p.name: p for p in [
    #        name         label             background   hair       skin       clothes    lips       eye shadow
    Palette("marilyn", "Marilyn rosa", "#ff2d95", ("#ffe600", "#ffd6c9", "#00b4d8", "#c8102e", "#00b4d8")),
    Palette("turquoise", "Turquesa", "#00b3b8", ("#ffe600", "#ff9ec4", "#3a0050", "#e8112d", "#2bd9ff")),
    Palette("orange", "Naranja", "#ff7a00", ("#1b1464", "#7fdbff", "#c2007a", "#e4002b", "#ff1493")),
    Palette("yellow", "Amarillo", "#ffe100", ("#c2007a", "#ffb3a7", "#0077ff", "#e4002b", "#00b4d8")),
    Palette("lime", "Verde lima", "#8be04e", ("#4a0e6b", "#ff9ec4", "#0050ff", "#e4002b", "#00c2ff")),
    Palette("blue", "Azul eléctrico", "#1e3cff", ("#ff8c00", "#ffd1dc", "#00c853", "#e8112d", "#33ccff")),
    Palette("violet", "Violeta", "#8a2be2", ("#ffea00", "#9df2c8", "#ff5f00", "#ff1493", "#2b6bff")),
    Palette("red", "Rojo", "#e4002b", ("#1b1464", "#ffe08a", "#00c4a7", "#8a0a3a", "#1e90ff")),
]}

DEFAULT_PALETTES = ("marilyn", "turquoise", "orange", "violet", "yellow", "blue", "lime", "red")

# Flat inks that the user can select for a zone (the color selectors of step 5). They are the colors
# of the palettes plus some other classic inks of pop art; a fixed set keeps the style of a silkscreen.
SWATCHES: tuple[tuple[str, str], ...] = (
    ("#ff2d95", "Rosa chicle"), ("#ff9ec4", "Rosa claro"), ("#ffd6c9", "Durazno"), ("#ffb3a7", "Salmón"),
    ("#e4002b", "Rojo"), ("#c8102e", "Rojo carmín"), ("#8a0a3a", "Bordó"), ("#ff1493", "Fucsia"),
    ("#c2007a", "Magenta"), ("#ff5f00", "Naranja"), ("#ff8c00", "Naranja claro"), ("#ffe600", "Amarillo"),
    ("#ffe08a", "Amarillo pálido"), ("#8be04e", "Verde lima"), ("#00c853", "Verde"), ("#9df2c8", "Verde agua"),
    ("#00b3b8", "Turquesa"), ("#00b4d8", "Celeste turquesa"), ("#7fdbff", "Celeste"), ("#2bd9ff", "Cian"),
    ("#0077ff", "Azul"), ("#1e3cff", "Azul eléctrico"), ("#1b1464", "Azul marino"), ("#8a2be2", "Violeta"),
    ("#4a0e6b", "Morado"), ("#3a0050", "Berenjena"), ("#ffffff", "Blanco"), ("#111111", "Negro"),
)
