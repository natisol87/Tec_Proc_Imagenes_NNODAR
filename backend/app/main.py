from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import FRONTEND_DIR
from app.errors import register_exception_handlers

app = FastAPI(
    title="Pastiche — Pop Art",
    description="API that turns a portrait into a Warhol-style silkscreen, one step at a time.",
    version="2.0.0",
)


@app.get("/api/health", tags=["system"], summary="Server status")
def health():
    return {"status": "ok"}


# Domain exceptions (core and API) → {"detail": {"code", "message"}} responses.
register_exception_handlers(app)


# Frontend (must be mounted last because it captures "/").
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
