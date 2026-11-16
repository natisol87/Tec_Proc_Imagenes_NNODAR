import base64

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool

from app.schemas import ErrorOut, PhotoOut, UrlIn
from app.uploads import download_photo
from popart.imageio import encode_jpeg

router = APIRouter(prefix="/api", tags=["photo"])


@router.post(
    "/photo/from-url",
    response_model=PhotoOut,
    summary="Download a photo from a web address",
    responses={400: {"model": ErrorOut}, 413: {"model": ErrorOut}},
)
async def photo_from_url(body: UrlIn):
    photo = await run_in_threadpool(download_photo, body.url)
    data = base64.b64encode(encode_jpeg(photo.image)).decode()
    return PhotoOut(image="data:image/jpeg;base64," + data, width=photo.width, height=photo.height)
