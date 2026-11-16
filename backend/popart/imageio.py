"""
Reading and writing of images.

All the core works with numpy arrays in RGB order (the same order as the
palettes): photos are (height, width, 3) uint8 and masks are (height, width)
uint8 with 0 / 255. Important: the OpenCV functions that read and write files
use BGR. Here the files are opened with Pillow, so this problem does not exist;
the few OpenCV functions that need BGR (the face model) convert the colors explicitly.
"""

from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from popart.exceptions import InvalidImage

# When the photos are opened, they are reduced to this maximum size: the portraits are
# 512 px, so a photo of 12 megapixels would only make the process slower.
MAX_SIDE = 1600


@dataclass(frozen=True)
class Photo:
    """A photo ready for the process."""

    format: str        # "PNG", "JPEG", "WEBP", "BMP"… detected from the content
    image: np.ndarray  # RGB uint8, in the correct position according to its EXIF orientation

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def height(self) -> int:
        return self.image.shape[0]


def read_photo(content: bytes, max_side: int = MAX_SIDE) -> Photo:
    """
    Opens the bytes of a photo with Pillow.

    - The format is detected from the content, not from the name of the file.
    - The EXIF orientation is applied (phones save the pixels rotated).
    - Transparency is replaced with white and any color mode is converted to RGB.
    - If a side is longer than `max_side`, the photo is reduced with the same proportions.
    """
    try:
        with Image.open(BytesIO(content)) as img:
            img.load()
            fmt = img.format or ""
            rgb = _flatten(ImageOps.exif_transpose(img))
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise InvalidImage("El archivo no es una imagen válida o está dañado.") from exc
    rgb.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return Photo(format=fmt, image=np.array(rgb, dtype=np.uint8))


def read_gray(content: bytes) -> np.ndarray:
    """Opens an image with one channel (a mask, a zone map, an ink layer) as a 2D uint8 array."""
    try:
        with Image.open(BytesIO(content)) as img:
            return np.array(img.convert("L"), dtype=np.uint8)
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise InvalidImage("Una de las capas enviadas no es una imagen válida.") from exc


def _flatten(img: Image.Image) -> Image.Image:
    """Any mode → RGB, with transparency replaced with white."""
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return img.convert("RGB")


def encode_png(image: np.ndarray) -> bytes:
    """Codifies an RGB (h, w, 3) or gray (h, w) array as PNG, in memory."""
    mode = "L" if image.ndim == 2 else "RGB"
    buf = BytesIO()
    Image.fromarray(image, mode).save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def encode_jpeg(image: np.ndarray, quality: int = 90) -> bytes:
    """Codifies an RGB array as JPEG (much smaller than PNG for photos)."""
    buf = BytesIO()
    Image.fromarray(image, "RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def labels_to_gray(labels: np.ndarray, levels: int) -> np.ndarray:
    """Zone map 0..levels-1 → gray 0..255 with the same distance between levels, to send it as a PNG."""
    return np.rint(labels.astype(float) * 255 / (levels - 1)).astype(np.uint8)


def gray_to_labels(gray: np.ndarray, levels: int) -> np.ndarray:
    """The inverse of labels_to_gray."""
    return np.rint(gray.astype(float) * (levels - 1) / 255).astype(np.uint8)


def count_colors(image: np.ndarray) -> int:
    """Number of different colors in the image."""
    if image.ndim == 2:
        return int(np.unique(image).size)
    rgb = image.reshape(-1, 3).astype(np.uint32)
    return int(np.unique((rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]).size)
