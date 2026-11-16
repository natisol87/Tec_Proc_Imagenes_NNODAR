"""
Configuration of the application.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # backend/
FRONTEND_DIR = BASE_DIR.parent / "frontend"
MODELS_DIR = BASE_DIR / "models"

YUNET_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SEGMENTATION_MODEL = MODELS_DIR / "human_segmentation_pphumanseg_2023mar.onnx"
FACE_PARSING_MODEL = MODELS_DIR / "face_parsing_bisenet_resnet18.onnx"

# Accepted photo formats (detected from the content, not from the extension).
ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP", "BMP", "GIF", "MPO"}

# Vercel does not accept requests bigger than 4.5 MB; the frontend reduces the photos before sending them.
MAX_UPLOAD_BYTES = 4 * 1024 * 1024

# Photos downloaded from a link (the server downloads them, so the limit can be bigger).
MAX_DOWNLOAD_BYTES = 15 * 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 10
