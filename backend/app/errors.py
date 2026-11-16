"""
Conversion of exceptions to HTTP responses.

All the errors of the API have the same format:

    {"detail": {"code": "NO_FACE", "message": "..."}}

The endpoints do not need try/except: the exceptions go up and these
handlers create the response.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from popart import InvalidParameters, MissingInput, NoFaceDetected, PopArtError


# Exceptions of the core → (HTTP status, code). Subclasses use the entry of their parent class.
CORE_ERRORS: dict[type[PopArtError], tuple[int, str]] = {
    InvalidParameters: (400, "INVALID_PARAMETERS"),
    MissingInput: (400, "MISSING_INPUT"),
    NoFaceDetected: (422, "NO_FACE"),
    PopArtError: (400, "POPART_ERROR"),
}


def error_response(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"detail": {"code": code, "message": message}})


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(PopArtError)
    async def handle_core_error(request: Request, exc: PopArtError) -> JSONResponse:
        for exc_type in type(exc).__mro__:
            if exc_type in CORE_ERRORS:
                status, code = CORE_ERRORS[exc_type]
                return error_response(status, code, str(exc))
        return error_response(400, "POPART_ERROR", str(exc))  # not expected: PopArtError is always in CORE_ERRORS
