"""
Auxiliary functions for the tests: synthetic images, so the tests do not need real photos.
"""

from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from popart import Work

SIZE = 128


def random_image(height: int = SIZE, width: int = SIZE, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)


def gradient_image(size: int = SIZE) -> np.ndarray:
    """Black on the left, white on the right."""
    row = np.linspace(0, 255, size).astype(np.uint8)
    gray = np.tile(row, (size, 1))
    return np.dstack([gray] * 3)


def figure_image(size: int = SIZE) -> np.ndarray:
    """A dark disc (the 'person') with a light ring inside it, on a light background."""
    image = np.full((size, size, 3), 230, np.uint8)
    cv2.circle(image, (size // 2, size // 2), size // 3, (40, 40, 40), -1)
    cv2.circle(image, (size // 2, size // 2), size // 6, (200, 120, 90), -1)
    return image


def disc_mask(size: int = SIZE) -> np.ndarray:
    mask = np.zeros((size, size), np.uint8)
    cv2.circle(mask, (size // 2, size // 2), size // 3, 255, -1)
    return mask


def work_for_compose(size: int = SIZE) -> Work:
    """Five vertical bands (one for each zone) inside a disc, and a horizontal line of ink."""
    from popart import ZONE_COUNT
    labels = (np.arange(size)[None, :].repeat(size, 0) * ZONE_COUNT // size).astype(np.uint8)
    mask = disc_mask(size)
    edges = np.zeros((size, size), np.uint8)
    edges[size // 2, :] = 255
    return Work(labels=labels, mask=mask, edges=edges)


def portrait(size: int = SIZE) -> Work:
    """A synthetic portrait: dark hair, a face with the color of skin and red lips, a blue sweater."""
    image = np.full((size, size, 3), 240, np.uint8)
    cv2.ellipse(image, (64, 40), (34, 32), 0, 0, 360, (60, 40, 30), -1)     # hair
    cv2.ellipse(image, (64, 50), (22, 27), 0, 0, 360, (225, 180, 150), -1)  # face
    cv2.ellipse(image, (64, 66), (8, 3), 0, 0, 360, (200, 30, 50), -1)      # lips
    cv2.ellipse(image, (55, 44), (4, 2), 0, 0, 360, (30, 20, 20), -1)       # eyes
    cv2.ellipse(image, (73, 44), (4, 2), 0, 0, 360, (30, 20, 20), -1)
    cv2.rectangle(image, (24, 84), (104, 127), (40, 60, 180), -1)           # sweater
    mask = np.zeros((size, size), np.uint8)
    mask[(image != 240).any(axis=2)] = 255
    from popart import FaceBox
    face = FaceBox(44, 26, 40, 46, eyes=((55, 44), (73, 44)))
    return Work(image=image, mask=mask, face=face)


def png(image: np.ndarray) -> bytes:
    buf = BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    return buf.getvalue()


def colors_of(image: np.ndarray) -> set[tuple[int, int, int]]:
    return {tuple(c) for c in np.unique(image.reshape(-1, 3), axis=0).tolist()}
