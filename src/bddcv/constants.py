"""Shared constants. Import these everywhere; never re-declare class order locally.

The class index order defined here is the contract between the YOLO label
files, the COCO annotation file, and every results table. Changing it
invalidates already-converted labels and trained checkpoints.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_LABELS_DIR = DATA_DIR / "labels_raw"
ARCHIVE = PROJECT_ROOT / "archive.zip"

# Paths inside archive.zip
ZIP_IMAGE_PREFIX = "bdd100k/bdd100k/images/100k"

RAW_LABEL_FILES = {
    "train": RAW_LABELS_DIR / "bdd100k_labels_images_train.json",
    "val": RAW_LABELS_DIR / "bdd100k_labels_images_val.json",
}

# The legacy bdd100k_labels_images_*.json release uses these category strings.
# The newer det_20 release renames three of them; keep this mapping so results
# are reported under the conventional names regardless of which release is read.
LEGACY_TO_CANONICAL = {
    "person": "pedestrian",
    "bike": "bicycle",
    "motor": "motorcycle",
}

# Fixed class index order. Do not reorder.
DET_CLASSES = [
    "pedestrian",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign",
]
CLASS_TO_IDX = {name: i for i, name in enumerate(DET_CLASSES)}

# Non-detection annotation categories carried in the same labels[] array.
# These have poly2d geometry instead of box2d and must be dropped.
NON_DET_CATEGORIES = {"drivable area", "lane"}

# Source domain for this project.
SOURCE_TIMEOFDAY = "daytime"
SOURCE_WEATHER = "clear"


def canonical_category(raw: str) -> str:
    """Map a raw label category onto the canonical class name."""
    return LEGACY_TO_CANONICAL.get(raw, raw)
