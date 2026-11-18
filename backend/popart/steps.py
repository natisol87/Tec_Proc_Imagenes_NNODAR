"""
The steps that change a portrait into a silkscreen in the style of Warhol.

    photo → 1 Frame → 2 Ink → 3 Background → 4 Zones → 5 Compose

The order is the same as in a real silkscreen: first the black screen (the
photo in grayscale, the "ink"), then the background is cleaned, and at the
end the parts of the portrait (skin, hair, clothes, lips, eye shadow) are
defined and printed in flat colors under the ink.

All the steps are classes with the same interface (`Step`). The FACTORS of a
step (the parameters that the user changes) go to the constructor, which validates them.
`apply(work)` receives the work done until now and returns a NEW `Work` with the
fields that the step produces; it does not modify the original. Because of this,
the frontend can run one step at a time, keep ("save") each result, and return to a
previous step to test other factors without starting again from the beginning.

Rules:
  - Images are RGB uint8 arrays (h, w, 3); masks, zone maps and the ink layer
    are (h, w) uint8.
  - A step only reads the fields in `requires` (and `optional`) and only
    writes the fields in `produces`.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import ClassVar

import cv2
import numpy as np

from popart.detection import FaceBox, FaceDetector
from popart.exceptions import InvalidParameters, MissingInput, NoFaceDetected
from popart.palettes import PALETTES, hex_to_rgb
from popart.parsing import FaceParser, Tuning
from popart.segmentation import Segmenter, clean_mask

TILE_SIZE = 512                  # default size of the side of the portrait, in pixels
TILE_SIZES = (384, 512, 768, 1024)


def scaled(pixels: float, image: np.ndarray) -> int:
    """
    The factors in pixels (smoothing, edge, eye shadow...) are calibrated for a portrait of 512 px.
    For a bigger or smaller portrait they are scaled, so the result is the same at any size.
    """
    return int(round(pixels * image.shape[1] / TILE_SIZE))
PREVIEW_BACKGROUND = (255, 230, 0)  # flat yellow background behind the person in the previews

# The zones of the portrait, in the same order as Palette.zones.
HAIR, SKIN, CLOTHES, LIPS, EYES = 0, 1, 2, 3, 4
ZONE_COUNT = 5
ZONE_NAMES = ("pelo", "piel", "ropa", "labios", "sombra de ojos")


@dataclass(frozen=True)
class Work:
    """All the results until now. Each step completes some of these fields."""

    photo: np.ndarray | None = None   # the original photo, RGB, any size
    image: np.ndarray | None = None   # the square portrait in process, RGB
    face: FaceBox | None = None       # rectangle of the face, in coordinates of the portrait
    faces: tuple[FaceBox, ...] | None = None  # all the faces in the photo, from left to right (coordinates of the photo)
    edges: np.ndarray | None = None   # ink density: 0 = no ink, 255 = completely black
    mask: np.ndarray | None = None    # 255 = person, 0 = background
    labels: np.ndarray | None = None  # zone of each pixel: HAIR, SKIN, CLOTHES, LIPS or EYES
    final: np.ndarray | None = None   # the final grid, RGB

    def update(self, **fields) -> "Work":
        return replace(self, **fields)


def check_range(label: str, value: float, low: float, high: float) -> None:
    if not low <= value <= high:
        raise InvalidParameters(f"«{label}» tiene que estar entre {low} y {high} (llegó {value}).")


def check_choice(label: str, value: str, options) -> None:
    if value not in options:
        raise InvalidParameters(f"«{label}» tiene que ser una de: {', '.join(options)} (llegó «{value}»).")


def print_ink(colors: np.ndarray, density: np.ndarray) -> np.ndarray:
    """
    Prints the ink over the colors with a MULTIPLICATION (unit 4, arithmetic operations):
    each color is multiplied by (1 − density/255). Density 0 does not change the color,
    255 changes it to black, and the values in the middle make it darker but the color is
    still visible, as with a transparent ink.
    """
    factor = 1 - density.astype(np.float32)[:, :, None] / 255
    return np.rint(colors.astype(np.float32) * factor).astype(np.uint8)


def silhouette_outline(mask: np.ndarray, width: int = 1) -> np.ndarray:
    """
    Morphological gradient (unit 2, §4): dilation minus erosion. The result is only the border of the mask,
    and a bigger kernel (`width`) gives a thicker border.
    """
    size = 2 * max(scaled(width, mask), 1) + 1
    return cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((size, size), np.uint8))


class Step(ABC):
    number: ClassVar[int]
    name: ClassVar[str]       # identifier for the API: "frame"
    label: ClassVar[str]      # name visible in the frontend: "Encuadre"
    requires: ClassVar[tuple[str, ...]]
    optional: ClassVar[tuple[str, ...]] = ()
    produces: ClassVar[tuple[str, ...]]

    def run(self, work: Work) -> Work:
        """Verifies that the necessary inputs exist and applies the step."""
        for field in self.requires:
            if getattr(work, field) is None:
                raise MissingInput(self.label, field)
        return self.apply(work)

    @abstractmethod
    def apply(self, work: Work) -> Work:
        """Returns a NEW Work with the fields of `produces` completed."""

    @abstractmethod
    def preview(self, work: Work) -> np.ndarray:
        """An RGB image that shows the result of this step (for the frontend)."""


# ===========================================================================
# 1 · Frame: detect the face and cut a square portrait around it
# ===========================================================================

class Frame(Step):
    number, name, label = 1, "frame", "Encuadre"
    requires, produces = ("photo",), ("image", "face", "faces")

    def __init__(self, detector: FaceDetector, margin: float = 0.9, offset: float = 0.25,
                 size: int = TILE_SIZE, sharpen: float = 0.5, face: int = 0, straighten: bool = True) -> None:
        check_range("Margen alrededor de la cara", margin, 0.3, 2.0)
        check_range("Desplazamiento vertical", offset, -0.5, 0.5)
        check_range("Nitidez", sharpen, 0.0, 2.0)
        check_range("Tamaño", size, 32, max(TILE_SIZES))  # the API only offers the values of TILE_SIZES
        check_range("Cara", face, 0, 20)
        self.face = face          # 0 = the biggest face; 1, 2, ... = the faces from left to right
        self.straighten = straighten  # rotate the photo so the eyes are horizontal
        self.detector = detector
        self.margin = margin      # space around the face, in face sizes, on each side
        self.offset = offset      # moves the cut down (+) to include the shoulders, in face heights
        self.size = size          # side of the square portrait, in pixels
        self.sharpen = sharpen    # intensity of the unsharp mask (0 = no change)

    def crop_box(self, shape: tuple[int, int], face: FaceBox) -> tuple[int, int, int]:
        """Square (x, y, side) around the face, limited to the borders of the image."""
        h, w = shape
        side = min(int(max(face.width, face.height) * (1 + 2 * self.margin)), w, h)
        cx, cy = face.center
        cy += face.height * self.offset
        x = int(np.clip(cx - side / 2, 0, w - side))
        y = int(np.clip(cy - side / 2, 0, h - side))
        return x, y, side

    def apply(self, work: Work) -> Work:
        photo = work.photo
        faces = self.detector.detect(photo)
        if not faces:
            raise NoFaceDetected("No se detectó ninguna cara. Probá con un retrato de frente y bien iluminado.")
        # Numbered from left to right, as the user sees them; by default, the biggest one.
        in_order = tuple(sorted(faces, key=lambda f: f.x))
        if self.face > len(in_order):
            count = len(in_order)
            raise InvalidParameters(f"La foto tiene {count} cara{'s' if count > 1 else ''}: elegí un número "
                                    f"entre 1 y {count}, o la más grande.")
        face = faces[0] if self.face == 0 else in_order[self.face - 1]
        if self.straighten:
            photo, face = self.level_eyes(photo, face)
        x, y, side = self.crop_box(photo.shape[:2], face)
        crop = photo[y:y + side, x:x + side]
        # INTER_AREA calculates an average when the image is reduced (no aliasing); INTER_CUBIC gives a smoother result when the image is enlarged.
        interpolation = cv2.INTER_AREA if side > self.size else cv2.INTER_CUBIC
        tile = cv2.resize(crop, (self.size, self.size), interpolation=interpolation)
        if self.sharpen > 0:
            tile = self.unsharp_mask(tile)
        s = self.size / side

        def move(p):  # a point of the photo → the same point in the portrait
            return int((p[0] - x) * s), int((p[1] - y) * s)

        tile_face = FaceBox(int((face.x - x) * s), int((face.y - y) * s),
                            int(face.width * s), int(face.height * s),
                            eyes=tuple(map(move, face.eyes)) if face.eyes else None)
        return work.update(image=tile, face=tile_face, faces=in_order)

    @staticmethod
    def tilt(face: FaceBox) -> float:
        """Angle of the line between the eyes, in degrees (0 = horizontal; + = the right eye is lower)."""
        if not face.eyes:
            return 0.0
        (x1, y1), (x2, y2) = sorted(face.eyes)  # first the eye on the left of the photo
        return float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

    def level_eyes(self, photo: np.ndarray, face: FaceBox) -> tuple[np.ndarray, FaceBox]:
        """
        ROTATION around the center of the face (unit 1, §2 and unit 3: getRotationMatrix2D +
        warpAffine), so the line between the eyes is horizontal, as in the frontal portraits of Warhol.
        Very small angles (< 1°) are not corrected, and very big angles (> 45°) are probably an error.
        """
        angle = self.tilt(face)
        if not 1.0 <= abs(angle) <= 45.0:
            return photo, face
        h, w = photo.shape[:2]
        rotation = cv2.getRotationMatrix2D(face.center, angle, 1.0)
        # BORDER_REPLICATE: if the rotation leaves an empty corner, it is filled with the
        # nearest pixels and not with black, because step 3 could interpret a black corner as a shape.
        level = cv2.warpAffine(photo, rotation, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        def turn(p):  # a point of the photo → the same point after the rotation
            return tuple(int(round(v)) for v in rotation @ (p[0], p[1], 1.0))

        # The rectangle keeps its center (it is the center of the rotation) and its size.
        return level, FaceBox(face.x, face.y, face.width, face.height,
                              eyes=tuple(map(turn, face.eyes)) if face.eyes else None)

    def unsharp_mask(self, image: np.ndarray) -> np.ndarray:
        """
        UNSHARP MASK (unit 5, §1.3): image + intensity × (image − blurred image). The difference contains
        only the fine details, so adding it to the image makes the edges more defined, with less
        noise than a sharpening kernel. It is useful when the cut was enlarged.
        """
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.5 * image.shape[1] / TILE_SIZE)
        return cv2.addWeighted(image, 1 + self.sharpen, blurred, -self.sharpen, 0)

    def preview(self, work: Work) -> np.ndarray:
        out = work.image.copy()
        if work.face is not None:
            f = work.face
            cv2.rectangle(out, (f.x, f.y), (f.x + f.width, f.y + f.height), (255, 20, 147), 3)
            for point in f.eyes or ():
                cv2.circle(out, point, 4, (255, 230, 0), cv2.FILLED)
        return out


# ===========================================================================
# 2 · Ink: the photo in grayscale, the black screen of the silkscreen
# ===========================================================================

class Ink(Step):
    """
    Creates the INK DENSITY layer: 0 = no ink, 255 = completely black. It is the photo in grayscale,
    as in the silkscreens of Warhol, where the black screen was a photo with high contrast, printed
    over areas of color. Step 5 prints it over the flat colors with a multiplication: a density
    of 128 makes the color below 50% darker.
    """

    number, name, label = 2, "ink", "Escala de grises"
    requires, produces = ("image",), ("edges",)

    def __init__(self, smoothing: int = 4, contrast: float = 2.0, shadows: float = 1.6) -> None:
        check_range("Suavizado", smoothing, 0, 10)
        check_range("Contraste", contrast, 0, 5)
        check_range("Solo sombras", shadows, 1.0, 4.0)
        self.smoothing, self.contrast, self.shadows = smoothing, contrast, shadows

    def grayscale(self, image: np.ndarray) -> np.ndarray:
        """The photo in grayscale, smoothed and with more local contrast."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if self.smoothing > 0:
            # BILATERAL filter (unit 1, §2 and unit 5, §1.2): it calculates the average of each pixel only with
            # the neighbor pixels of a similar tone. The skin is smoothed, but the edges of the eyes, nose
            # and mouth stay defined, and the ink is cleaner. `smoothing` defines the distance (space) and
            # the difference of tone (color) allowed for the neighbor pixels.
            radius = scaled(self.smoothing, gray)
            gray = cv2.bilateralFilter(gray, 2 * radius + 1, sigmaColor=12 * self.smoothing, sigmaSpace=3 * radius)
        if self.contrast > 0:
            # CLAHE: local histogram equalization with a limit, so the noise does not increase.
            gray = cv2.createCLAHE(clipLimit=self.contrast, tileGridSize=(8, 8)).apply(gray)
        return gray

    def ink_table(self) -> np.ndarray:
        """
        LUT of 256 values (unit 7, §4.1, gamma correction with a LUT): gray level → ink density.
        It is the NEGATIVE (light = no ink) with an exponent: with shadows > 1 the middle tones are
        lighter and only the shadows keep the ink. This gives the high contrast of a silkscreen.
        """
        levels = np.arange(256) / 255
        return np.rint((1 - levels) ** self.shadows * 255).astype(np.uint8)

    def apply(self, work: Work) -> Work:
        # Contrast stretching to the complete range 0..255 (cv2.normalize with NORM_MINMAX, unit 5).
        gray = cv2.normalize(self.grayscale(work.image), None, 0, 255, cv2.NORM_MINMAX)
        return work.update(edges=cv2.LUT(gray, self.ink_table()))

    def preview(self, work: Work) -> np.ndarray:
        """The ink printed on white paper."""
        return print_ink(np.full((*work.edges.shape, 3), 255, np.uint8), work.edges)


# ===========================================================================
# 3 · Background: separate the person and make the background flat
# ===========================================================================

class Background(Step):
    number, name, label = 3, "background", "Fondo"
    requires, optional, produces = ("image",), ("face", "edges"), ("mask",)

    def __init__(self, segmenter: Segmenter, smoothness: int = 3, edge: int = 0) -> None:
        check_range("Suavidad del contorno", smoothness, 0, 10)
        check_range("Borde", edge, -10, 10)
        self.segmenter, self.smoothness, self.edge = segmenter, smoothness, edge

    def apply(self, work: Work) -> Work:
        # The color photo is used to detect the person: color separates the person and the background
        # much better than gray.
        raw = self.segmenter.segment(work.image, work.face)
        image = work.image
        return work.update(mask=clean_mask(raw, scaled(self.smoothness, image), scaled(self.edge, image)))

    def preview(self, work: Work) -> np.ndarray:
        """The ink of step 2 (or the photo, if step 2 was not done) cut out over a flat background."""
        person = work.mask > 0
        if work.edges is not None:
            figure = print_ink(np.full_like(work.image, 255), work.edges)
        else:
            figure = work.image
        out = np.empty_like(work.image)
        out[:] = PREVIEW_BACKGROUND
        out[person] = figure[person]
        return out


# ===========================================================================
# 4 · Zones: the parts of the portrait, each one with one flat color
# ===========================================================================

ZONE_PREVIEW = np.array([(70, 70, 70), (235, 195, 175), (120, 130, 210), (210, 30, 40), (40, 190, 210)],
                        np.uint8)


class Zones(Step):
    """
    Divides the person into skin, hair, clothes, lips and eye shadow, as the color screens of a
    portrait of Warhol. All the details come from the ink of step 2; each zone is only one flat color.
    """

    number, name, label = 4, "zones", "Zonas"
    requires, optional, produces = ("image", "mask"), ("face", "edges"), ("labels",)

    def __init__(self, parser: FaceParser, hair: float = 0.0, lip_boost: float = 0.0,
                 smooth: int = 4, eyes: bool = True, eye_size: int = 6,
                 lips: bool = True, lip_widen: int = 1, hat_as_hair: bool = False) -> None:
        check_range("Límite piel / pelo", hair, -1.5, 1.5)
        check_range("Sensibilidad de labios", lip_boost, 0.0, 1.5)
        check_range("Suavizar zonas", smooth, 0, 8)
        check_range("Tamaño de la sombra de ojos", eye_size, 1, 15)
        check_range("Engrosar labios", lip_widen, 0, 6)
        self.parser = parser
        self.tuning = Tuning(hair, lip_boost, hat_as_hair)
        self.smooth = smooth  # radius of the median filter that eliminates very small islands in the zone map
        self.eyes, self.eye_size = eyes, eye_size
        self.lips, self.lip_widen = lips, lip_widen

    def apply(self, work: Work) -> Work:
        person = work.mask > 0
        h, w = person.shape
        face = work.face or FaceBox(w // 3, h // 5, w // 3, h // 3)  # no face: a face in the center is assumed
        parts = self.parser.parse(work.image, person, face, self.tuning)

        # Pixels without a zone from the parser: hair if they are above the chin AND near the head;
        # all the rest (below the chin, or on the sides: shoulders, high collars) is clothes.
        chin = min(face.y + int(face.height * 1.05), h)
        x0 = max(face.x - int(face.width * 0.45), 0)
        x1 = min(face.x + face.width + int(face.width * 0.45), w)
        labels = np.full((h, w), CLOTHES, np.uint8)
        labels[:chin, x0:x1] = HAIR
        labels[parts.clothes > 0] = CLOTHES
        labels[parts.hair > 0] = HAIR
        labels[parts.skin > 0] = SKIN
        # Median filter on the zone map: the very small islands take the zone of their neighbors. It is
        # applied BEFORE the eyes and lips, because it would eliminate thin details, for example thin lips.
        if self.smooth > 0:
            labels = cv2.medianBlur(labels, 2 * scaled(self.smooth, labels) + 1)
        if self.eyes:
            labels[(self._eye_shadow(work.image, face, parts.skin) > 0) & (labels == SKIN)] = EYES
        if self.lips:
            lips = parts.lips
            if self.lip_widen > 0:  # a dilation makes thin lips more visible
                size = 2 * scaled(self.lip_widen, lips) + 1
                lips = cv2.dilate(lips, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
            labels[(lips > 0) & person] = LIPS
        labels[~person] = HAIR  # this value is not important: later the background is painted with a flat color
        return work.update(labels=labels)

    @staticmethod
    def eye_centers(face: FaceBox) -> tuple[tuple[int, int], tuple[int, int]]:
        """The eyes of the face detector; if they are not known, the typical proportions of a face."""
        if face.eyes:
            return face.eyes
        y = face.y + face.height * 40 // 100
        return (face.x + face.width * 31 // 100, y), (face.x + face.width * 69 // 100, y)

    def _eye_shadow(self, image: np.ndarray, face: FaceBox, skin: np.ndarray) -> np.ndarray:
        """
        A flat band of color on the upper eyelid that follows the curve of each eye:

        1. Around each eye, the threshold of Otsu (unit 5) separates the dark part (eyelashes, iris)
           from the skin, only inside the skin of the network (so the hair is not taken). Connected
           components (unit 1, §4): only the darkest region is kept, so the fold of the eyelid, a
           wrinkle or the eyebrow is not taken as the eye.
        2. An OPENING (unit 2) cuts the thin bridges between the eye and the hair or the fold.
        3. In each column, the first dark row is the upper edge of the eye (the eyelashes). A parabola
           fitted to those rows makes the edge a smooth curve; the columns far from it are ignored.
        4. From that edge, a band goes up: it is higher in the middle and thinner at the corners
           (half an ellipse), and its height grows with `eye_size`.
        Step 5 does not print ink over it, so it is a solid color, as the eyelids of the Marilyn of Warhol.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        (x1, y1), (x2, y2) = self.eye_centers(face)
        gap = max(int(np.hypot(x2 - x1, y2 - y1)), 10)  # distance between the eyes: the scale of the face
        height = gap * 8 // 100 + max(scaled(self.eye_size, image), 1)
        shadow = np.zeros((h, w), np.uint8)
        for ex, ey in ((x1, y1), (x2, y2)):
            dx, up, down = max(gap * 30 // 100, 4), max(gap * 15 // 100, 3), max(gap * 12 // 100, 3)
            bx0, bx1 = max(ex - dx, 0), min(ex + dx, w)
            by0, by1 = max(ey - up, 0), min(ey + down, h)
            if bx1 - bx0 < 4 or by1 - by0 < 4:
                continue
            _, dark = cv2.threshold(gray[by0:by1, bx0:bx1], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            dark &= skin[by0:by1, bx0:bx1]  # only inside the skin of the network (the eyes are skin): not the hair
            # OPENING (unit 2): cuts the thin bridges between the eye and the hair or the fold of the eyelid.
            k = 2 * max(gap // 60, 1) + 1
            dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
            count, components = cv2.connectedComponents(dark)
            if count < 2:
                continue
            # The eye is the DARKEST region (eyelashes and iris are darker than wrinkles, the fold of the
            # eyelid or the eyebrow), among the regions that are not too small.
            box = gray[by0:by1, bx0:bx1]
            sizes = np.bincount(components.ravel())[1:]
            candidates = [i + 1 for i in range(count - 1) if sizes[i] >= 0.2 * sizes.max()]
            darkest = min(candidates, key=lambda i: box[components == i].mean())
            eye = components == darkest
            columns = np.nonzero(eye.any(axis=0))[0]
            if len(columns) < 3:
                continue
            top = eye[:, columns].argmax(axis=0)  # first dark row of each column
            # The edge of an eyelid is a smooth curve: a parabola fitted by least squares to those rows.
            # The columns far from the first curve (loose eyelashes, hair) are ignored in a second fit.
            curve = np.polyfit(columns, top, 2)
            error = np.abs(np.polyval(curve, columns) - top)
            good = error <= max(2 * np.median(error), 1)
            if good.sum() >= 3:
                curve = np.polyfit(columns[good], top[good], 2)
                columns = np.arange(columns[good][0], columns[good][-1] + 1)  # every column, without gaps
            top = np.rint(np.polyval(curve, columns)).astype(int) + by0
            half = (columns[-1] - columns[0]) / 2
            middle = (columns[-1] + columns[0]) / 2
            for col, edge in zip(columns + bx0, top):
                band = int(height * np.sqrt(max(0.0, 1 - ((col - bx0 - middle) / (half + 1)) ** 2)))
                shadow[max(edge - band, 0):edge, col] = 255
        return shadow

    def preview(self, work: Work) -> np.ndarray:
        """Each zone in a neutral color, with the ink over it if step 2 was done."""
        out = ZONE_PREVIEW[work.labels]
        if work.edges is not None:
            out = print_ink(out, np.where(work.labels == EYES, 0, work.edges).astype(np.uint8))
        out[work.mask == 0] = 255
        return out


# ===========================================================================
# 5 · Compose: paint each portrait with a palette, print the ink and create the grid
# ===========================================================================

class Compose(Step):
    number, name, label = 5, "compose", "Color y grilla"
    requires, optional, produces = ("labels", "mask"), ("edges", "image"), ("final",)

    CUSTOM_ZONES = {"hair": HAIR, "skin": SKIN, "lips": LIPS, "eyes": EYES}  # zones where the user can change the color

    MAX_SIDE = 2048  # maximum side of the final grid, in pixels

    def __init__(self, palettes: list[str], rows: int = 2, cols: int = 2, outline: int = 1,
                 custom: list[dict[str, str]] | None = None, opacity: float = 1.0) -> None:
        check_range("Filas", rows, 1, 3)
        check_range("Columnas", cols, 1, 3)
        check_range("Grosor del contorno", outline, 0, 10)
        check_range("Opacidad de las zonas", opacity, 0.0, 1.0)
        self.opacity = opacity  # 1 = flat colors; less = the photo is visible under the colors
        if not palettes:
            raise InvalidParameters("Elegí al menos una paleta.")
        for name in palettes:
            check_choice("Paleta", name, list(PALETTES))
        self.custom = [dict(c or {}) for c in (custom or [])]  # portrait i → {"skin": "#rrggbb", ...}
        for colors in self.custom:
            for zone, color in colors.items():
                check_choice("Zona", zone, list(self.CUSTOM_ZONES))
                if not (isinstance(color, str) and len(color) == 7 and color[0] == "#"
                        and all(ch in "0123456789abcdefABCDEF" for ch in color[1:])):
                    raise InvalidParameters(f"«{color}» no es un color válido (formato #rrggbb).")
        self.palettes, self.rows, self.cols, self.outline = list(palettes), rows, cols, outline

    def tile_palettes(self) -> list[str]:
        """One palette for each portrait; when the palettes end, they start again from the first."""
        return [self.palettes[i % len(self.palettes)] for i in range(self.rows * self.cols)]

    def ink_layer(self, work: Work) -> np.ndarray:
        """The ink of step 2, only on the person, plus the black outline of the person."""
        density = np.zeros_like(work.mask)
        if work.edges is not None:
            density = np.where(work.mask > 0, work.edges, 0).astype(np.uint8)  # flat background: no ink
        density[work.labels == EYES] = 0  # the eye shadow is a solid color, without ink over it
        if self.outline > 0:  # thickness of the outline, in pixels (0 = no outline)
            density = np.maximum(density, silhouette_outline(work.mask, self.outline))
        return density

    def tile_colors(self, index: int, palette_name: str) -> np.ndarray:
        """The LUT of portrait `index`: the colors of the palette, replaced by the colors that the user selected."""
        lut = PALETTES[palette_name].zone_colors  # a new array: changes here do not modify the palette
        chosen = self.custom[index] if index < len(self.custom) else {}
        for zone, color in chosen.items():
            lut[self.CUSTOM_ZONES[zone]] = hex_to_rgb(color)
        return lut

    def paint(self, work: Work, palette_name: str, ink: np.ndarray, index: int = 0) -> np.ndarray:
        palette = PALETTES[palette_name]
        # LUT: row i = color of zone i. With the zone map as index, all the pixels are painted at the same time.
        tile = self.tile_colors(index, palette_name)[work.labels]
        if self.opacity < 1 and work.image is not None:
            # WEIGHTED MIX (unit 4, cv2.addWeighted): opacity × color + (1 − opacity) × photo. The zones are
            # transparent and the colors of the photo are visible under them.
            tile = cv2.addWeighted(tile, self.opacity, work.image, 1 - self.opacity, 0)
        tile[work.mask == 0] = palette.background_rgb  # flat background
        return print_ink(tile, ink)

    def apply(self, work: Work) -> Work:
        ink = self.ink_layer(work)
        tiles = [self.paint(work, name, ink, i) for i, name in enumerate(self.tile_palettes())]
        # The width of the complete grid is MAX_SIDE px or less (big portraits in a 3×3 grid would be
        # too heavy to send to the browser): the portraits are reduced with INTER_AREA (unit 3).
        side = tiles[0].shape[0]
        fit = min(1.0, self.MAX_SIDE / (side * max(self.rows, self.cols)))
        if fit < 1.0:
            small = int(side * fit)
            tiles = [cv2.resize(tile, (small, small), interpolation=cv2.INTER_AREA) for tile in tiles]
        rows = [np.hstack(tiles[r * self.cols:(r + 1) * self.cols]) for r in range(self.rows)]
        return work.update(final=np.vstack(rows))

    def preview(self, work: Work) -> np.ndarray:
        return work.final


STEPS: tuple[type[Step], ...] = (Frame, Ink, Background, Zones, Compose)
