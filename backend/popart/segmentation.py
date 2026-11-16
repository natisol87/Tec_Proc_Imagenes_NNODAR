"""
Separation of the person and the background.

The portraits of Warhol have a completely flat background, so the result of
this step is very important for the final effect.

A pretrained network for person segmentation gives the probability of "person" for each pixel. Its map has low resolution (192×192),
so only the very probable parts are used as markers, and WATERSHED WITH MARKERS decides the strip around the border, following the real borders of color in the photo.

After that, a cleaning with contours: keep the
biggest contour, draw it filled (this also fills the holes), smooth the border with a
median filter and, if necessary, make it a few pixels bigger or smaller.
"""

from abc import ABC, abstractmethod

import cv2
import numpy as np

from popart.detection import FaceBox

# Marker values for cv2.watershed: 0 = unknown, the other values identify regions.
UNKNOWN, BACKGROUND, PERSON = 0, 1, 2


class Segmenter(ABC):
    """Returns a mask: 255 = person, 0 = background."""

    @abstractmethod
    def segment(self, image: np.ndarray, face: FaceBox | None) -> np.ndarray: ...


def ellipse_kernel(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def run_watershed(image: np.ndarray, markers: np.ndarray) -> np.ndarray:
    """Extends the regions of the markers over the image. Returns a 0/255 mask of the PERSON region."""
    if not (markers == PERSON).any():
        return np.zeros(image.shape[:2], np.uint8)
    markers = markers.astype(np.int32)  # cv2.watershed modifies the int32 markers directly
    cv2.watershed(image, markers)
    # The pixels on the division lines get -1, and OpenCV also marks all the border of the image
    # with -1. They go to the background: if they were person, there would be a frame around
    # the photo and, after filling it, the mask would cover the complete image.
    return np.where(markers == PERSON, 255, 0).astype(np.uint8)


class PersonModel:
    """PP-HumanSeg: RGB image → probability of person for each pixel (float 0..1)."""

    INPUT_SIZE = (192, 192)

    def __init__(self, model_path: str) -> None:
        self.net = cv2.dnn.readNet(str(model_path))

    def probability(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        # The same normalization used in the training of the model: [0, 255] → [-1, 1].
        blob = cv2.dnn.blobFromImage(image.astype(np.float32) / 255.0, size=self.INPUT_SIZE)
        self.net.setInput((blob - 0.5) / 0.5)
        output = self.net.forward()[0, 1]  # channel 1 = person (it is already a softmax probability)
        return cv2.resize(output, (w, h), interpolation=cv2.INTER_LINEAR)


class ModelSegmenter(Segmenter):
    def __init__(self, model: PersonModel, band: int = 12) -> None:
        self.model = model
        self.band = band  # half of the width, in pixels, of the uncertain strip around the border of the network

    def markers(self, probability: np.ndarray, face: FaceBox | None) -> np.ndarray:
        person = (probability > 0.5).astype(np.uint8)
        kernel = ellipse_kernel(max(self.band * probability.shape[1] // 512, 1))  # scaled to the size of the image
        sure_person = cv2.erode(person, kernel)   # the internal part of the person
        near_person = cv2.dilate(person, kernel)  # the person plus a strip around it
        markers = np.full(probability.shape, UNKNOWN, np.uint8)
        markers[near_person == 0] = BACKGROUND
        markers[sure_person > 0] = PERSON
        if face is not None:  # the center of the face is always person
            dx, dy = face.width // 4, face.height // 4
            markers[face.y + dy:face.y + face.height - dy, face.x + dx:face.x + face.width - dx] = PERSON
        return markers

    def segment(self, image: np.ndarray, face: FaceBox | None) -> np.ndarray:
        return run_watershed(image, self.markers(self.model.probability(image), face))


def clean_mask(mask: np.ndarray, smoothness: int, edge: int) -> np.ndarray:
    """
    Changes a raw mask into a clean silhouette.

    smoothness (0..10): radius of the smoothing; bigger = more rounded contour, fewer points.
    edge (-10..10): pixels to make the silhouette bigger (+) or smaller (−).
    """
    # 1. Keep the biggest external contour (the person) and draw it FILLED:
    #    small spots of background disappear and the holes inside the person are covered.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(mask)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(clean, [largest], -1, 255, thickness=cv2.FILLED)

    # 2. Smooth the border: a median filter on a binary mask rounds the corners and eliminates points.
    if smoothness > 0:
        clean = cv2.medianBlur(clean, 2 * (2 * smoothness) + 1)

    # 3. Make the silhouette bigger or smaller.
    if edge > 0:
        clean = cv2.dilate(clean, ellipse_kernel(edge))
    elif edge < 0:
        clean = cv2.erode(clean, ellipse_kernel(-edge))
    return np.where(clean > 127, 255, 0).astype(np.uint8)
