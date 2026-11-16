"""
The steps that change a portrait into a silkscreen in the style of Warhol.

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
from popart.segmentation import Segmenter, clean_mask

TILE_SIZE = 512                  # default size of the side of the portrait, in pixels
TILE_SIZES = (384, 512, 768, 1024)


def scaled(pixels: float, image: np.ndarray) -> int:
    """
    The factors in pixels (smoothing, edge, eye shadow...) are calibrated for a portrait of 512 px.
    For a bigger or smaller portrait they are scaled, so the result is the same at any size.
    """
    return int(round(pixels * image.shape[1] / TILE_SIZE))


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


# 1 · Frame: detect the face and cut a square portrait around it

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

# 2 · Ink: the photo in grayscale, the black screen of the silkscreen

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
            # BILATERAL filter: it calculates the average of each pixel only with
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

# 3 · Background: separate the person and make the background flat

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
    
STEPS: tuple[type[Step], ...] = (Frame,Ink, Background)
