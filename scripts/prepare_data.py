"""Prepare all publicly labelled BDD100K images for train/val/test.

The official train and validation labels are combined and split deterministically
into 70% train, 20% validation, and 10% test. Images are symlinked, not copied,
so the generated view does not duplicate the full dataset.
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bddcv.constants import ARCHIVE, DATA_DIR, RAW_LABELS_DIR, RAW_LABEL_FILES  # noqa: E402
from bddcv.labels import stream_records  # noqa: E402

DATASET_DIR = DATA_DIR / "source_full"
DEFAULT_IMAGES_ROOT = (
    DATA_DIR / "kagglehub/datasets/solesensei/solesensei_bdd100k/versions/2/"
    "bdd100k/bdd100k/images/100k"
)
SEED = 0
TRAIN_FRACTION = 0.70
TEST_FRACTION = 0.10


def ensure_raw_labels(archive: Path) -> None:
    """Extract missing raw label JSONs with a bounded, atomic copy."""
    missing = {split: path for split, path in RAW_LABEL_FILES.items() if not path.exists()}
    if not missing:
        return
    if not archive.exists():
        raise SystemExit(
            "raw labels are missing and no archive was found; pass "
            "--archive /path/to/bdd100k.zip"
        )

    RAW_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        members = zf.namelist()
        for target in missing.values():
            matches = [name for name in members if Path(name).name == target.name]
            if len(matches) != 1:
                raise SystemExit(f"expected one {target.name} in archive, found {len(matches)}")
            temporary = target.with_suffix(target.suffix + ".tmp")
            try:
                with zf.open(matches[0]) as source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1 << 20)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, target)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise


def build_manifests() -> dict[str, list[str]]:
    """Return the fixed full-data split from raw labelled records."""
    names = []
    for raw_split in ("train", "val"):
        names.extend(record["name"] for record in stream_records(RAW_LABEL_FILES[raw_split]))
    if len(names) != len(set(names)):
        raise SystemExit("official labels contain duplicate filenames")

    random.Random(SEED).shuffle(names)
    train_size = int(len(names) * TRAIN_FRACTION)
    test_size = int(len(names) * TEST_FRACTION)
    return {
        "train": sorted(names[:train_size]),
        "val": sorted(names[train_size:-test_size]),
        "test": sorted(names[-test_size:]),
    }


def reject_manifest_drift(manifests: dict[str, list[str]]) -> None:
    """Reject changed protected manifests before touching dataset links."""
    drifted = []
    for split, names in manifests.items():
        path = DATASET_DIR / f"{split}_images.txt"
        expected = "\n".join(names) + "\n"
        if path.exists() and path.read_text(encoding="utf-8") != expected:
            drifted.append(path.name)
    if drifted:
        raise SystemExit(
            "regenerated manifest(s) differ from the fixed seed-0 version: "
            f"{', '.join(drifted)}. No dataset links were changed."
        )


def image_index(directory: Path) -> dict[str, Path]:
    """Index a possibly nested official image split by unique basename."""
    if not directory.is_dir():
        raise SystemExit(f"image source directory not found: {directory}")
    index: dict[str, Path] = {}
    duplicates = []
    for path in directory.rglob("*.jpg"):
        if path.name in index:
            duplicates.append(path.name)
        else:
            index[path.name] = path.resolve()
    if duplicates:
        raise SystemExit(f"duplicate image basenames in {directory}: {duplicates[:3]}")
    return index


def link_split(split: str, names: list[str], index: dict[str, Path]) -> tuple[int, int]:
    """Create a manifest-exact symlink view for one logical split."""
    missing = [name for name in names if name not in index]
    if missing:
        raise SystemExit(f"{split} is missing {len(missing)} source images, e.g. {missing[:3]}")

    destination = DATASET_DIR / "images" / split
    destination.mkdir(parents=True, exist_ok=True)
    wanted = set(names)
    unexpected = [path.name for path in destination.iterdir() if path.name not in wanted]
    if unexpected:
        raise SystemExit(
            f"{split} contains {len(unexpected)} files outside its manifest, "
            f"e.g. {unexpected[:3]}"
        )

    created = existing = 0
    for name in names:
        target = destination / name
        if target.is_file():
            existing += 1
            continue
        if target.is_symlink():
            raise SystemExit(f"broken dataset symlink: {target}")
        target.symlink_to(index[name])
        created += 1
    return created, existing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-root", type=Path, default=DEFAULT_IMAGES_ROOT)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    args = parser.parse_args()

    ensure_raw_labels(args.archive)
    manifests = build_manifests()
    reject_manifest_drift(manifests)

    train_index = image_index(args.images_root / "train")
    official_val_index = image_index(args.images_root / "val")
    overlap = set(train_index) & set(official_val_index)
    if overlap:
        raise SystemExit(f"official image splits overlap, e.g. {sorted(overlap)[:3]}")
    full_index = {**train_index, **official_val_index}
    indexes = {split: full_index for split in manifests}

    # Validate all source coverage before writing manifests or links.
    for split, names in manifests.items():
        missing = [name for name in names if name not in indexes[split]]
        if missing:
            raise SystemExit(
                f"{split} is missing {len(missing)} source images, e.g. {missing[:3]}"
            )

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    for split, names in manifests.items():
        manifest = DATASET_DIR / f"{split}_images.txt"
        if not manifest.exists():
            manifest.write_text("\n".join(names) + "\n", encoding="utf-8")
        created, existing = link_split(split, names, indexes[split])
        print(f"{split}: {len(names):,} images ({created:,} linked, {existing:,} existing)")


if __name__ == "__main__":
    main()
