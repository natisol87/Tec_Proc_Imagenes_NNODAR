"""
Translation of exceptions to HTTP responses.

Every error leaves the API with the same format:

    {"detail": {"code": "MISSING_INPUT", "message": "..."}}

Endpoints do not need try/except: they let the exceptions go up and these
handlers answer.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from popart import MissingInput, PopArtError


# Core exceptions → (HTTP status, code). Subclasses inherit the entry of their parent.
CORE_ERRORS: dict[type[PopArtError], tuple[int, str]] = {
    MissingInput: (400, "MISSING_INPUT"),
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
        return error_response(400, "POPART_ERROR", str(exc))  # pragma: no cover
