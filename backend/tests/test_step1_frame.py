"""Step 1 · Frame.    pytest tests/test_step1_frame.py -v"""

import numpy as np
import pytest

from helpers import random_image
from popart import FaceBox, FaceDetector, Frame, YuNetFaceDetector, InvalidParameters, MissingInput, NoFaceDetected, Work


class FixedDetector(FaceDetector):
    def __init__(self, faces):
        self.faces = faces

    def detect(self, image):
        return self.faces


def test_output_is_square_of_the_requested_size():
    detector = FixedDetector([FaceBox(100, 80, 60, 70)])
    work = Frame(detector, size=128).run(Work(photo=random_image(400, 300)))
    assert work.image.shape == (128, 128, 3)


def test_face_box_is_translated_to_tile_coordinates():
    detector = FixedDetector([FaceBox(100, 100, 50, 50)])
    work = Frame(detector, margin=0.5, offset=0.0, size=100).run(Work(photo=random_image(400, 400)))
    # side = 50 * (1 + 2 * 0.5) = 100, with the face in the center: the face occupies the central half.
    assert work.face == FaceBox(25, 25, 50, 50)


def test_no_face_raises():
    with pytest.raises(NoFaceDetected):
        Frame(FixedDetector([])).run(Work(photo=random_image()))


def test_crop_stays_inside_the_photo():
    detector = FixedDetector([FaceBox(0, 0, 40, 40)])  # face in the corner
    x, y, side = Frame(detector).crop_box((200, 300), FaceBox(0, 0, 40, 40))
    assert x >= 0 and y >= 0 and x + side <= 300 and y + side <= 200


def test_input_is_not_modified():
    photo = random_image()
    copy = photo.copy()
    Frame(FixedDetector([FaceBox(100, 80, 60, 70)]), size=64).run(Work(photo=photo))
    assert np.array_equal(photo, copy)


def test_missing_photo():
    with pytest.raises(MissingInput):
        Frame(None).run(Work())


@pytest.mark.parametrize("margin", [0.1, 3.0])
def test_margin_out_of_range(margin):
    with pytest.raises(InvalidParameters):
        Frame(None, margin=margin)


def test_yunet_finds_no_face_in_a_flat_image():
    from pathlib import Path
    model = Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet_2023mar.onnx"
    assert YuNetFaceDetector(str(model)).detect(np.full((200, 200, 3), 128, np.uint8)) == []


def test_sharpening_adds_contrast_to_the_edges():
    """Unsharp mask (unit 5): the variance of the Laplacian, a measure of sharpness, increases."""
    import cv2
    photo = cv2.GaussianBlur(random_image(400, 400), (0, 0), 3)
    detector = FixedDetector([FaceBox(150, 150, 100, 100)])
    soft = Frame(detector, size=256, sharpen=0).run(Work(photo=photo)).image
    sharp = Frame(detector, size=256, sharpen=1.5).run(Work(photo=photo)).image

    def sharpness(im):
        return cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()

    assert sharpness(sharp) > sharpness(soft)


def test_pixel_factors_scale_with_the_size():
    from popart.steps import scaled
    assert scaled(6, np.zeros((512, 512))) == 6
    assert scaled(6, np.zeros((1024, 1024))) == 12


def test_the_user_can_choose_the_face():
    """Faces are numbered from left to right; 0 means the biggest one."""
    big_right, small_left = FaceBox(250, 100, 80, 80), FaceBox(40, 120, 50, 50)
    detector = FixedDetector([big_right, small_left])  # a detector returns the biggest first
    photo = random_image(300, 400)
    largest = Frame(detector, size=64).run(Work(photo=photo))
    first = Frame(detector, size=64, face=1).run(Work(photo=photo))
    second = Frame(detector, size=64, face=2).run(Work(photo=photo))
    assert largest.faces == (small_left, big_right)            # from left to right
    assert np.array_equal(largest.image, second.image)         # the biggest is the one on the right
    assert not np.array_equal(largest.image, first.image)      # face 1 cuts around the other face


def test_choosing_a_face_that_does_not_exist():
    detector = FixedDetector([FaceBox(40, 40, 50, 50)])
    with pytest.raises(InvalidParameters):
        Frame(detector, face=2).run(Work(photo=random_image(300, 300)))


def test_straightening_levels_the_eyes():
    """Rotation around the face (unit 3): after the rotation, the two eyes are at the same height."""
    tilted = FaceBox(150, 150, 100, 100, eyes=((170, 180), (230, 200)))  # right eye 20 px lower
    detector = FixedDetector([tilted])
    work = Frame(detector, size=200).run(Work(photo=random_image(400, 400)))
    (_, y1), (_, y2) = work.face.eyes
    assert abs(y1 - y2) <= 1
    still = Frame(detector, size=200, straighten=False).run(Work(photo=random_image(400, 400)))
    assert abs(still.face.eyes[0][1] - still.face.eyes[1][1]) > 5


def test_tilt_of_the_eyes():
    assert Frame.tilt(FaceBox(0, 0, 10, 10, eyes=((0, 0), (10, 10)))) == pytest.approx(45)
    assert Frame.tilt(FaceBox(0, 0, 10, 10)) == 0  # without eyes there is no rotation
