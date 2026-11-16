"""
Conversion of exceptions to HTTP responses.

All the errors of the API have the same format:

    {"detail": {"code": "NO_FACE", "message": "..."}}

The endpoints do not need try/except: the exceptions go up and these
handlers create the response.
"""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from popart import InvalidImage, InvalidParameters, MissingInput, NoFaceDetected, PopArtError


class ApiError(Exception):
    """An error of the API (not of the core), with its status and code."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


# Exceptions of the core → (HTTP status, code). Subclasses use the entry of their parent class.
CORE_ERRORS: dict[type[PopArtError], tuple[int, str]] = {
    InvalidImage: (400, "INVALID_FILE"),
    InvalidParameters: (400, "INVALID_PARAMETERS"),
    MissingInput: (400, "MISSING_INPUT"),
    NoFaceDetected: (422, "NO_FACE"),
    PopArtError: (400, "POPART_ERROR"),
}


def error_response(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"detail": {"code": code, "message": message}})


# Types of Pydantic errors → message in Spanish (ctx has the limits, for example {"le": 5}).
SPANISH_MESSAGES = {
    "less_than_equal": "tiene que ser como máximo {le}",
    "greater_than_equal": "tiene que ser como mínimo {ge}",
    "literal_error": "tiene que ser una de: {expected}",
    "int_parsing": "tiene que ser un número entero",
    "int_from_float": "tiene que ser un número entero",
    "float_parsing": "tiene que ser un número",
    "bool_parsing": "tiene que ser verdadero o falso",
    "list_type": "tiene que ser una lista",
    "missing": "falta este dato",
}


def describe_validation(errors: list[dict]) -> str:
    """Pydantic errors → one clear sentence in Spanish."""
    parts = []
    for error in errors:
        field = ".".join(str(p) for p in error.get("loc", ()) if p not in ("body", "query", "form"))
        template = SPANISH_MESSAGES.get(error.get("type", ""))
        try:
            message = template.format(**error.get("ctx", {})) if template else "valor inválido"
        except KeyError:
            message = "valor inválido"
        parts.append(f"«{field or 'dato'}» {message}")
    return "Parámetros inválidos: " + "; ".join(parts) + "."


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def handle_api_error(request: Request, exc: ApiError) -> JSONResponse:
        return error_response(exc.status, exc.code, exc.message)

    @app.exception_handler(PopArtError)
    async def handle_core_error(request: Request, exc: PopArtError) -> JSONResponse:
        for exc_type in type(exc).__mro__:
            if exc_type in CORE_ERRORS:
                status, code = CORE_ERRORS[exc_type]
                return error_response(status, code, str(exc))
        return error_response(400, "POPART_ERROR", str(exc))  # not expected: PopArtError is always in CORE_ERRORS

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        return error_response(400, "INVALID_PARAMETERS", describe_validation(exc.errors()))

    @app.exception_handler(ValidationError)
    async def handle_model_validation(request: Request, exc: ValidationError) -> JSONResponse:
        return error_response(400, "INVALID_PARAMETERS", describe_validation(exc.errors()))
