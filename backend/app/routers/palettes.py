from fastapi import APIRouter

from app.schemas import PaletteOut, SwatchOut
from popart import PALETTES, Palette
from popart.palettes import SWATCHES

router = APIRouter(prefix="/api", tags=["palettes"])


def palette_out(palette: Palette) -> PaletteOut:
    return PaletteOut(name=palette.name, label=palette.label, background=palette.background,
                      zones=list(palette.zones))


@router.get("/palettes", response_model=list[PaletteOut], summary="List the Warhol palettes")
def list_palettes():
    return [palette_out(p) for p in PALETTES.values()]


@router.get("/colors", response_model=list[SwatchOut], summary="Inks the user can pick for a zone")
def list_colors():
    return [SwatchOut(hex=color, label=label) for color, label in SWATCHES]
