# Pastiche — Retratos pop art estilo Warhol

Trabajo Final Integrador de *Técnicas de Procesamiento Digital de Imágenes* (2026).

La aplicación convierte un retrato en una serigrafía al estilo de Andy Warhol.
Cada paso se ajusta con sus factores y se guarda antes de pasar al siguiente.

## Estado

| Paso | Qué hace | Técnicas |
|---|---|---|
| 1 · Encuadre | Detecta las caras (si hay varias, se elige cuál), endereza la cara y recorta un retrato cuadrado con más nitidez | YuNet (red preentrenada: cara y ojos), rotación afín, interpolación, máscara de desenfoque |
| 2 · Escala de grises | La foto en grises que se imprime como tinta sobre los colores | Filtro bilateral, CLAHE, normalización de contraste, negativo con curva gamma aplicada con LUT |
| 3 · Fondo | Separa a la persona para dejar el fondo liso | PP-HumanSeg (red preentrenada) como marcadores + watershed, mediana, dilatación/erosión |
| 4 · Zonas | Define piel, pelo, ropa, labios y sombra de ojos | Face parsing (BiSeNet), componentes conexas, mediana, dilatación; sombra de ojos con umbral de Otsu, apertura, componentes conexas y una parábola ajustada al borde del ojo |
| 5 · Color y grilla | Pinta cada zona con una paleta, imprime la tinta y arma la grilla | LUT, máscaras, mezcla ponderada (opacidad), multiplicación de imágenes, gradiente morfológico (contorno) |

## Estructura

```
backend/
  popart/            Core: no conoce FastAPI, se usa desde la API o los tests
    steps.py         Los 5 pasos (una clase por paso, misma interfaz Step)
    pipeline.py      Arma los pasos a partir de los factores y los corre todos
    detection.py     Detector de rostros YuNet (cara y ojos)
    segmentation.py  Persona / fondo: red PP-HumanSeg + watershed y limpieza con contornos
    parsing.py       Partes de la cara: red de face parsing (BiSeNet)
    palettes.py      Paletas Warhol (fondo + color de cada zona) y contraste en Lab
    imageio.py       Lectura y escritura de imágenes (Pillow, RGB)
    exceptions.py    Errores del dominio
  models/            Redes preentrenadas (.onnx de YuNet, PP-HumanSeg y face parsing)
  app/               API (FastAPI): config, errores, esquemas, subida de fotos y routers
  tests/             Un archivo de tests por paso, más la API
frontend/            index.html: página provisoria 
```

## Cómo correrlo

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate           # Linux / macOS: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8765
```

- Aplicación: http://localhost:8765
- Documentación de la API: http://localhost:8765/docs

## Tests

```bash
cd backend
pytest tests/test_step1_frame.py -v
pytest tests/test_step2_ink.py -v
pytest tests/test_step3_background.py -v
pytest tests/test_step4_zones.py -v
pytest tests/test_step5_compose.py -v
pytest                                      # todo
```

## API

| Método | Ruta | Para qué |
|---|---|---|
| GET | `/api/steps` | Los pasos y sus factores (rango, valor por defecto, nombre en español) |
| GET | `/api/palettes` | Las paletas |
| GET | `/api/colors` | Las tintas para elegir el color de cada zona |
| POST | `/api/steps/{paso}` | Aplica un paso: recibe los factores y las capas que necesita, devuelve las que produce |
| POST | `/api/photo/from-url` | Descarga una imagen de un enlace |

El servidor no guarda estado: el navegador guarda el resultado de cada paso (también en IndexedDB,
así el trabajo sobrevive a una recarga) y envía las capas que necesita el paso siguiente.

## Modelos preentrenados

La app usa tres redes preentrenadas, todas ejecutadas
con `cv2.dnn`:

- `face_detection_yunet_2023mar.onnx`: detector de rostros YuNet, con los centros de los ojos
  ([opencv_zoo](https://github.com/opencv/opencv_zoo), Apache 2.0).
- `human_segmentation_pphumanseg_2023mar.onnx`: segmentación de personas PP-HumanSeg
  ([opencv_zoo](https://github.com/opencv/opencv_zoo), Apache 2.0).
- `face_parsing_bisenet_resnet18.onnx`: segmentación de las partes de la cara, BiSeNet
  ([yakhyo/face-parsing](https://github.com/yakhyo/face-parsing), código MIT, ver
  `face_parsing_LICENSE.txt`; entrenada con CelebAMask-HQ, de uso no comercial).

`requirements.txt` fija la versión de OpenCV con la que se probó la aplicación (4.10).
