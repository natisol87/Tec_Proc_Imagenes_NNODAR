from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import FRONTEND_DIR
from app.errors import register_exception_handlers
from app.routers import palettes, photo, steps

app = FastAPI(
    title="Pastiche — Pop Art",
    description="API that turns a portrait into a Warhol-style silkscreen, one step at a time.",
    version="2.0.0",
)


@app.get("/api/health", tags=["system"], summary="Server status")
def health():
    return {"status": "ok"}


# Exceptions (of the core and of the API) → responses {"detail": {"code", "message"}}.
register_exception_handlers(app)

app.include_router(palettes.router)
app.include_router(photo.router)
app.include_router(steps.router)

# Frontend (it must be the last one, because it receives all the paths that start with "/").
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
