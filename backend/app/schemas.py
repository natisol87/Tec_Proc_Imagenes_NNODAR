"""
Input and output schemas of the API: the contract with the frontend and the tests.

The factors of each step are declared only ONCE, here, with their range, initial
value and label in Spanish. GET /api/steps publishes them, and the frontend creates
its sliders, selectors and checkboxes from that list, so a new factor here
appears automatically in the interface.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from popart import DEFAULT_PALETTES, PALETTES, Background, Compose, Frame, Ink, Models, Step, Zones

PaletteName = Literal[tuple(PALETTES)]


def factor(default, title: str, description: str, **extra):
    """A Field with the additional keys that the frontend uses (`step`, `options`, `unit`)."""
    limits = {k: extra.pop(k) for k in ("ge", "le") if k in extra}
    return Field(default, title=title, description=description, json_schema_extra=extra or None, **limits)


# ===========================================================================
# Factors of each step
# ===========================================================================

class FrameIn(BaseModel):
    size: Literal[384, 512, 768, 1024] = factor(512, "Tamaño", "Lado del retrato en píxeles. Más grande, más "
                                                "detalle, pero cada paso tarda más.",
                                                options={"384": "384 px", "512": "512 px", "768": "768 px",
                                                         "1024": "1024 px"})
    margin: float = factor(0.9, "Margen alrededor de la cara",
                           "Cuánto espacio dejar alrededor de la cara, en tamaños de cara.",
                           ge=0.3, le=2.0, step=0.05)
    offset: float = factor(0.25, "Desplazamiento vertical",
                           "Corre el encuadre hacia abajo (+) para incluir hombros, o hacia arriba (−).",
                           ge=-0.5, le=0.5, step=0.05)
    straighten: bool = factor(True, "Enderezar la cara", "Rota la foto para que los ojos queden a la misma "
                              "altura, como en los retratos de Warhol.")
    face: int = factor(0, "Cara", "Si la foto tiene varias caras, elegí cuál convertir.", ge=0, le=20, kind="face")
    sharpen: float = factor(0.5, "Nitidez", "Realza los detalles con una máscara de desenfoque. "
                            "Útil si la foto es chica y hubo que agrandarla. 0 = sin cambios.",
                            ge=0.0, le=2.0, step=0.1)

    def to_step(self, models: Models) -> Step:
        return Frame(models.detector(), self.margin, self.offset, self.size, self.sharpen, self.face, self.straighten)


class InkIn(BaseModel):
    smoothing: int = factor(4, "Suavizado", "Filtro bilateral: alisa la piel pero deja nítidos los bordes de "
                            "ojos, nariz y boca. 0 = apagado.", ge=0, le=10, step=1)
    contrast: float = factor(2.0, "Contraste", "Contraste local (CLAHE). 0 = apagado.", ge=0, le=5, step=0.1)
    shadows: float = factor(1.6, "Solo sombras", "Más alto = solo las zonas oscuras llevan tinta.",
                            ge=1.0, le=4.0, step=0.1)

    def to_step(self, models: Models) -> Step:
        return Ink(self.smoothing, self.contrast, self.shadows)


class BackgroundIn(BaseModel):
    smoothness: int = factor(3, "Suavidad del contorno", "Redondea la silueta y elimina picos.",
                             ge=0, le=10, step=1)
    edge: int = factor(0, "Borde", "Agranda (+) o achica (−) la silueta, en píxeles.", ge=-10, le=10, step=1,
                       unit="px")

    def to_step(self, models: Models) -> Step:
        return Background(models.segmenter(), self.smoothness, self.edge)


class ZonesIn(BaseModel):
    hair: float = factor(0.0, "Límite piel / pelo", "Corre el borde entre la piel y el pelo: negativo, "
                         "más piel; positivo, más pelo.", ge=-1.5, le=1.5, step=0.25)
    brows_as_hair: bool = factor(True, "Cejas como pelo", "Pinta las cejas del color del pelo. "
                                 "Apagado, quedan como piel y solo se ve la tinta.")
    hat_as_hair: bool = factor(False, "Sombrero como pelo", "Pinta el sombrero o la gorra del color del pelo. "
                               "Apagado, toma el color de la ropa.")
    smooth: int = factor(4, "Suavizar zonas", "Redondea los bordes entre zonas y quita manchas chicas.",
                         ge=0, le=8, step=1)
    eyes: bool = factor(True, "Sombra de ojos", "Pinta los párpados de su propio color.")
    eye_size: int = factor(6, "Tamaño de la sombra", "Cuánto se extiende la sombra sobre el párpado.",
                           ge=1, le=15, step=1, unit="px")
    lips: bool = factor(True, "Labios", "Pinta los labios de su propio color.")
    lip_boost: float = factor(0.0, "Sensibilidad de labios", "Súbila si los labios son pálidos y se "
                              "marcan solo en parte.", ge=0.0, le=1.5, step=0.25)
    lip_widen: int = factor(1, "Engrosar labios", "Agranda los labios (útil si son finos). 0 = tal cual.",
                            ge=0, le=6, step=1, unit="px")

    def to_step(self, models: Models) -> Step:
        return Zones(models.parser(), self.hair, self.lip_boost, self.brows_as_hair, self.smooth,
                     self.eyes, self.eye_size, self.lips, self.lip_widen, self.hat_as_hair)


HexColor = Annotated[str, Field(pattern=r"^#[0-9a-fA-F]{6}$")]


class ZoneColorsIn(BaseModel):
    """Colors selected by the user for one portrait; if a color is missing, the color of the palette is used."""

    hair: HexColor | None = None
    skin: HexColor | None = None
    lips: HexColor | None = None
    eyes: HexColor | None = None


class ComposeIn(BaseModel):
    rows: int = factor(2, "Filas", "Filas de la grilla.", ge=1, le=3, step=1)
    cols: int = factor(2, "Columnas", "Columnas de la grilla.", ge=1, le=3, step=1)
    palettes: list[PaletteName] = factor(list(DEFAULT_PALETTES), "Paletas",
                                         "Paleta de cada cuadro, en orden de lectura.", kind="palettes")
    custom: list[ZoneColorsIn] = factor([], "Colores elegidos",
                                        "Pelo, piel, labios y sombra de ojos de cada cuadro, si se cambiaron.",
                                        kind="custom")
    outline: bool = factor(True, "Contorno de la silueta", "Dibuja el borde de la figura en negro.")

    def to_step(self, models: Models) -> Step:
        custom = [c.model_dump(exclude_none=True) for c in self.custom]
        return Compose(list(self.palettes), self.rows, self.cols, self.outline, custom)


STEP_PARAMS: dict[str, type[BaseModel]] = {
    "frame": FrameIn, "ink": InkIn, "background": BackgroundIn, "zones": ZonesIn, "compose": ComposeIn,
}

StepName = Literal[tuple(STEP_PARAMS)]

DESCRIPTIONS = {
    "frame": "Busca la cara y recorta un retrato cuadrado alrededor de ella.",
    "ink": "Pasa la foto a escala de grises: es la tinta negra que después se imprime, transparente, "
           "encima de los colores.",
    "background": "Separa a la persona del fondo para que el fondo quede liso, como en Warhol.",
    "zones": "Define las zonas que llevan un color plano cada una: piel, pelo, ropa, labios y sombra de ojos.",
    "compose": "Pinta cada zona con una paleta contrastante, imprime la tinta encima y arma la grilla.",
}


# ===========================================================================
# Outputs
# ===========================================================================

class FactorOut(BaseModel):
    name: str
    schema_: dict = Field(alias="schema", description="JSON Schema of the factor (type, range, default, title…)")

    model_config = {"populate_by_name": True}


class StepInfoOut(BaseModel):
    number: int
    name: str
    label: str
    description: str
    requires: list[str] = Field(description="Layers the step needs from previous steps")
    optional: list[str]
    produces: list[str] = Field(description="Layers the step returns")
    factors: list[FactorOut]


class PaletteOut(BaseModel):
    name: str
    label: str
    background: str
    zones: list[str] = Field(description="#rrggbb for hair, skin, clothes, lips and eye shadow")


class SwatchOut(BaseModel):
    hex: str = Field(description="#rrggbb")
    label: str = Field(description="Name shown to the user")


class FaceOut(BaseModel):
    x: int
    y: int
    width: int
    height: int
    eyes: list[list[int]] | None = Field(None, description="[[x, y], [x, y]] of the two eyes (if they are known)")


class FaceRegionOut(BaseModel):
    """A face in the ORIGINAL photo, as fractions of its width and height (0..1), to draw it in the browser."""

    x: float
    y: float
    width: float
    height: float


class StepOut(BaseModel):
    step: str
    label: str
    preview: str | None = Field(None, description="JPEG data URL that shows what the step did (None if it is a layer)")
    preview_layer: str | None = Field(None, description="Name of the layer that is also the preview (step 5: final)")
    layers: dict[str, str] = Field(description="Produced layers as PNG data URLs (image, mask, labels, edges, final)")
    face: FaceOut | None = None
    faces: list[FaceRegionOut] | None = Field(None, description="Step 1: every face found, from left to right")
    width: int
    height: int
    colors: int = Field(description="Number of different colors in the preview")
    elapsed_ms: int


class PhotoOut(BaseModel):
    image: str = Field(description="JPEG data URL of the downloaded photo")
    width: int
    height: int


class UrlIn(BaseModel):
    url: str = Field(description="Address of an image on the web (http or https)")


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorOut(BaseModel):
    detail: ErrorDetail
