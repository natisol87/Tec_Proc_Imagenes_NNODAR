"""
One endpoint for each step, so the frontend can apply, adjust and save each one.

The server does NOT keep information between requests (it runs on Vercel, where each
request can go to a different machine). The frontend saves the layers that each
step returns and sends the layers that the next step needs (`requires`):

    frame       photo                          → image, face
    ink         image                          → edges   (grayscale ink density)
    background  image (+ face, edges)          → mask
    zones       image, mask (+ face, edges)    → labels  (skin, hair, clothes, lips, eye shadow)
    compose     labels, mask (+ edges)         → final
"""

import base64
import json
import time
from typing import Annotated

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.config import FACE_PARSING_MODEL, SEGMENTATION_MODEL, YUNET_MODEL
from app.errors import ApiError
from app.schemas import (DESCRIPTIONS, STEP_PARAMS, ErrorOut, FaceOut, FaceRegionOut, FactorOut, StepInfoOut, StepName,
                         StepOut)
from app.uploads import read_bytes, read_upload
from popart import (STEPS, ZONE_COUNT, FaceBox, Models, Work, count_colors, encode_png, gray_to_labels,
                    labels_to_gray, read_gray, read_photo)

router = APIRouter(prefix="/api", tags=["steps"])
models = Models(YUNET_MODEL, SEGMENTATION_MODEL, FACE_PARSING_MODEL)
STEP_CLASSES = {cls.name: cls for cls in STEPS}
SCALARS = ("face", "faces")  # they are sent as values, not as images

ERRORS = {400: {"model": ErrorOut}, 413: {"model": ErrorOut}, 422: {"model": ErrorOut}}
OptionalFile = Annotated[UploadFile | None, File()]


@router.get("/steps", response_model=list[StepInfoOut], summary="The steps and their factors")
def list_steps():
    infos = []
    for cls in STEPS:
        schema = STEP_PARAMS[cls.name].model_json_schema()
        factors = [FactorOut(name=name, schema=prop) for name, prop in schema["properties"].items()]
        infos.append(StepInfoOut(number=cls.number, name=cls.name, label=cls.label,
                                 description=DESCRIPTIONS[cls.name], requires=list(cls.requires),
                                 optional=list(cls.optional), produces=list(cls.produces), factors=factors))
    return infos


@router.post("/steps/{name}", response_model=StepOut, summary="Apply one step", responses=ERRORS)
async def apply_step(
    name: StepName,
    params: Annotated[str, Form(description="Factors of the step, as JSON")] = "{}",
    face: Annotated[str | None, Form(description="Face box from step 1, as JSON")] = None,
    photo: OptionalFile = None,
    image: OptionalFile = None,
    mask: OptionalFile = None,
    labels: OptionalFile = None,
    edges: OptionalFile = None,
):
    try:
        factors = STEP_PARAMS[name].model_validate(json.loads(params or "{}"))
    except json.JSONDecodeError as exc:
        raise ApiError(400, "INVALID_PARAMETERS", "Los factores no son un JSON válido.") from exc
    step = factors.to_step(models)

    work = await build_work(photo, image, mask, labels, edges, face)
    started = time.perf_counter()
    work = await run_in_threadpool(step.run, work)
    preview = step.preview(work)
    elapsed = int((time.perf_counter() - started) * 1000)

    # The final grid IS the preview of step 5: it is sent only once, as a layer (PNG, without loss). The other
    # previews are only for the screen and are sent as JPEG, much smaller (the maximum response in Vercel is 4.5 MB).
    same = next((f for f in step.produces if f not in SCALARS and getattr(work, f) is preview), None)
    return StepOut(
        step=step.name, label=step.label, preview=None if same else jpeg_data_url(preview), preview_layer=same,
        layers={field: data_url(layer_image(work, field)) for field in step.produces if field not in SCALARS},
        face=face_out(work.face) if "face" in step.produces else None,
        faces=faces_out(work.faces, work.photo.shape) if work.faces is not None else None,
        width=preview.shape[1], height=preview.shape[0], colors=count_colors(preview), elapsed_ms=elapsed,
    )


# ===========================================================================
# Auxiliary functions: layers that enter and exit
# ===========================================================================

async def build_work(photo, image, mask, labels, edges, face) -> Work:
    fields: dict = {}
    if photo is not None:
        fields["photo"] = (await read_upload(photo)).image
    if image is not None:
        fields["image"] = read_photo(await read_bytes(image), max_side=4096).image
    if mask is not None:
        fields["mask"] = np.where(read_gray(await read_bytes(mask)) > 127, 255, 0).astype(np.uint8)
    if edges is not None:
        fields["edges"] = read_gray(await read_bytes(edges))  # ink density 0..255, not binary
    if labels is not None:
        fields["labels"] = gray_to_labels(read_gray(await read_bytes(labels)), ZONE_COUNT)
    if face:
        try:
            known = ("x", "y", "width", "height", "eyes")
            values = {k: v for k, v in json.loads(face).items() if k in known}
            if values.get("eyes"):  # JSON lists → tuples of points
                values["eyes"] = tuple(tuple(int(v) for v in point) for point in values["eyes"])
            fields["face"] = FaceBox(**values)
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError) as exc:
            raise ApiError(400, "INVALID_PARAMETERS", "La caja de la cara no es válida.") from exc

    sizes = {f: v.shape[:2] for f, v in fields.items() if isinstance(v, np.ndarray) and f != "photo"}
    if len(set(sizes.values())) > 1:
        raise ApiError(400, "SIZE_MISMATCH", "Las capas enviadas no tienen el mismo tamaño.")
    return Work(**fields)


def layer_image(work: Work, field: str) -> np.ndarray:
    if field == "labels":
        return labels_to_gray(work.labels, ZONE_COUNT)
    return getattr(work, field)


def data_url(image: np.ndarray) -> str:
    return "data:image/png;base64," + base64.b64encode(encode_png(image)).decode()


def faces_out(faces, shape) -> list[FaceRegionOut]:
    h, w = shape[:2]
    return [FaceRegionOut(x=f.x / w, y=f.y / h, width=f.width / w, height=f.height / h) for f in faces]


def jpeg_data_url(image: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode()


def face_out(face: FaceBox | None) -> FaceOut | None:
    if face is None:
        return None
    return FaceOut(x=face.x, y=face.y, width=face.width, height=face.height,
                   eyes=[list(p) for p in face.eyes] if face.eyes else None)
