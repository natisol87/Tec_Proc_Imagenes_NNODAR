"""
Exceptions of the core.

The core only raises these exceptions: it does not know HTTP. The code that
uses the core decides how to show them (the API with a status code, a command
line program with a message in stderr). The messages are in Spanish because the
user reads them directly.
"""


class PopArtError(Exception):
    """Base class of all the errors of the core."""


class InvalidImage(PopArtError):
    """It is not possible to open the bytes as an image (empty, damaged, or not an image)."""


class NoFaceDetected(PopArtError):
    """The face detector did not find any face in the photo."""


class InvalidParameters(PopArtError):
    """A factor of a step is outside its range or has a value that does not exist."""


class MissingInput(PopArtError):
    """A step needs the result of a previous step, and that result is missing."""

    def __init__(self, step_label: str, field: str) -> None:
        super().__init__(f"El paso «{step_label}» necesita «{field}»: primero guardá el paso anterior.")
        self.field = field
