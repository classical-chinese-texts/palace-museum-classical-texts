"""Application configuration."""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # proofread-app/
REPO_DIR = BASE_DIR.parent  # palace-museum-classical-texts/
TEXTS_DIR = REPO_DIR / "texts"
PIPELINE_DIR = REPO_DIR / "scripts" / "pipeline"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Database
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite+aiosqlite:///{BASE_DIR / 'proofread.db'}"
)
# Railway PostgreSQL compat
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgresql://") and "+asyncpg" not in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

# OCR
CONFIDENCE_THRESHOLD = 0.7  # below this = low confidence
PADDLE_MODEL = os.getenv("PADDLE_MODEL", "hybrid")
KRAKEN_ENV = Path(os.getenv(
    "KRAKEN_ENV",
    "/Users/yui/linebot_migrate/palace_museum_texts/kraken_env"
))
CHAT_MODELS_DIR = Path(os.getenv(
    "CHAT_MODELS_DIR",
    "/Users/yui/linebot_migrate/palace_museum_texts/CHAT_models/models"
))
KRAKEN_SEG_MODEL = "chat_seg.mlmodel"
KRAKEN_REC_MODEL = "chat_rec.mlmodel"

# Image
MAX_IMAGE_SIZE = 4096  # max dimension for display
THUMBNAIL_SIZE = 200

# Character templates
TEMPLATE_DIR = UPLOAD_DIR / "templates"
TEMPLATE_DIR.mkdir(exist_ok=True)
TEMPLATE_SIZE = 64  # normalized template image size (64x64)
