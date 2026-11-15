"""
Domain exceptions of the core.

The core only raises these exceptions: it does not know about HTTP. Whoever
uses it decides how to report them (the API with a status code, the command
line with a message on stderr). Messages are in Spanish because they reach
the user as they are.
"""


class PopArtError(Exception):
    """Base class of every error of the core."""


class InvalidImage(PopArtError):
    """The bytes cannot be opened as an image (empty, corrupt, not an image)."""


class NoFaceDetected(PopArtError):
    """The face detector did not find any face in the photo."""


class InvalidParameters(PopArtError):
    """A factor of a step is out of its range or has an unknown value."""


class MissingInput(PopArtError):
    """A step needs the result of a previous step that was not provided."""

    def __init__(self, step_label: str, field: str) -> None:
        super().__init__(f"El paso «{step_label}» necesita «{field}»: primero guardá el paso anterior.")
        self.field = field
