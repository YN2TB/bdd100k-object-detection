"""Repository defaults and explicit CLI path handling (no import-time writes)."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ARCHIVE = DATA_DIR / "archives/archive.zip"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
RUNS_DIR = PROJECT_ROOT / "runs"
TRAIN_DIR = RUNS_DIR / "train"
SMOKE_DIR = RUNS_DIR / "smoke"
PREDICTIONS_DIR = RUNS_DIR / "predictions"
EVALUATION_DIR = RUNS_DIR / "evaluation"
VERIFICATION_DIR = RUNS_DIR / "verification"
CACHE_DIR = RUNS_DIR / "cache"
DATA_CONFIG = PROJECT_ROOT / "configs/bdd_source.yaml"


def resolve_output(value: str | Path | None, default: Path) -> Path:
    """Keep explicit relative paths relative to the caller; defaults are absolute."""
    return Path(value if value is not None else default).expanduser().resolve()


def resolve_weights(value: str | Path) -> Path:
    """Bare model names use weights/; explicit paths retain caller semantics."""
    raw = str(value)
    path = Path(raw).expanduser()
    if not path.is_absolute() and "/" not in raw and "\\" not in raw:
        path = WEIGHTS_DIR / path
    return path.resolve()


def prepare_runtime() -> None:
    """Keep project runtime caches local without editing personal settings."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    for variable, directory in (
        ("YOLO_CONFIG_DIR", CACHE_DIR / "ultralytics"),
        ("MPLCONFIGDIR", CACHE_DIR / "matplotlib"),
        ("TORCH_HOME", WEIGHTS_DIR / "torch"),
    ):
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(directory)


def prepare_ultralytics() -> None:
    """Initialize project settings before model/AMP construction."""
    prepare_runtime()
    import ultralytics.utils as utils

    utils.SETTINGS.update({"weights_dir": str(WEIGHTS_DIR), "runs_dir": str(RUNS_DIR),
                           "datasets_dir": str(DATA_DIR / "downloads")})
    # Ultralytics snapshots this setting at import; AMP reads the snapshot.
    utils.WEIGHTS_DIR = WEIGHTS_DIR
