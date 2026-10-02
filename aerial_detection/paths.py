"""Repository paths shared by launchers, training, and applications."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
TRAINING_DIR = PROJECT_ROOT / "visdrone_training"


def model_path(value=None):
    """Resolve an explicit path or AERIAL_MODEL_PATH relative to this repository."""
    value = value or os.environ.get("AERIAL_MODEL_PATH") or "models/best.pt"
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path
