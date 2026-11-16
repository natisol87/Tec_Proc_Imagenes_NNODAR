"""
Reading and validation of photos: uploaded by the user or downloaded from a link.

The validation order is:

1. bigger than the limit                → 413 FILE_TOO_LARGE
2. it is not possible to open it          → 400 INVALID_FILE (InvalidImage, from popart)
3. format not in config.ALLOWED_FORMATS  → 400 UNSUPPORTED_FORMAT
"""

import urllib.error
import urllib.request
from urllib.parse import urlparse

from fastapi import UploadFile

from app.config import ALLOWED_FORMATS, DOWNLOAD_TIMEOUT_SECONDS, MAX_DOWNLOAD_BYTES, MAX_UPLOAD_BYTES
from app.errors import ApiError
from popart import Photo, read_photo


async def read_bytes(file: UploadFile, limit: int = MAX_UPLOAD_BYTES) -> bytes:
    # Reads a maximum of one byte more than the limit, so a very big file is not loaded completely.
    content = await file.read(limit + 1)
    if len(content) > limit:
        raise ApiError(413, "FILE_TOO_LARGE", f"La imagen supera el máximo de {limit // (1024 * 1024)} MB.")
    return content


def check_photo(content: bytes) -> Photo:
    photo = read_photo(content)
    if photo.format not in ALLOWED_FORMATS:
        allowed = ", ".join(sorted(ALLOWED_FORMATS))
        raise ApiError(400, "UNSUPPORTED_FORMAT",
                       f"El formato {photo.format or 'desconocido'} no está soportado. Usá {allowed}.")
    return photo


async def read_upload(file: UploadFile) -> Photo:
    return check_photo(await read_bytes(file))


def download_photo(url: str) -> Photo:
    """Downloads an image from the web (necessary because browsers do not allow reading images of other sites)."""
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ApiError(400, "INVALID_URL", "El enlace tiene que empezar con http:// o https://.")
    request = urllib.request.Request(url.strip(), headers={"User-Agent": "Mozilla/5.0 (Pastiche pop art)"})
    try:
        with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            content = response.read(MAX_DOWNLOAD_BYTES + 1)
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        raise ApiError(400, "DOWNLOAD_FAILED", "No se pudo descargar la imagen de ese enlace.") from exc
    if len(content) > MAX_DOWNLOAD_BYTES:
        raise ApiError(413, "FILE_TOO_LARGE",
                       f"La imagen del enlace supera {MAX_DOWNLOAD_BYTES // (1024 * 1024)} MB.")
    return check_photo(content)
