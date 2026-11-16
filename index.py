"""
Entry point for Vercel.

Vercel looks for a FastAPI `app` in index.py, in the root of the project. The real
application is in backend/app/main.py (the same structure as the TPs of the course), so
this file only adds `backend/` to the import path and exports the application again.

Local execution:  cd backend && uvicorn app.main:app --reload
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

from app.main import app  # Vercel looks for the variable `app` in this file
