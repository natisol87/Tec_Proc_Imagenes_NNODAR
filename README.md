# Pastiche — Retratos pop art estilo Warhol

Trabajo Final Integrador de *Técnicas de Procesamiento Digital de Imágenes* (2026).

La aplicación convierte un retrato en una serigrafía al estilo de Andy Warhol.
Cada paso se ajusta con sus factores y se guarda antes de pasar al siguiente.

## Estado

Estructura del proyecto lista: núcleo `popart` (Work, Step), lectura de imágenes y API con `/api/health`.

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
pytest                                      # todo
```
