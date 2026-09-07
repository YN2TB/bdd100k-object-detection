"""Select the source-domain subset and extract only those images.

The full daytime+clear pool is 12,454 train / 1,764 val images. That is small
enough to use in its entirety, so no subsampling seed is involved: the subset
is fully determined by the attribute filter and is therefore reproducible by
construction.

Only the selected images are pulled out of archive.zip (~1.2 GB) rather than
the whole 8 GB archive.

Self-contained: the label JSONs are extracted from the archive automatically if
they are not already present, so a fresh clone plus a BDD100K archive is enough
to rebuild the subset. Pass --archive if yours is not at ./archive.zip.
"""
from __future__ import annotations

import argparse
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bddcv.constants import (  # noqa: E402
    ARCHIVE,
    DATA_DIR,
    RAW_LABELS_DIR,
    RAW_LABEL_FILES,
    SOURCE_TIMEOFDAY,
    SOURCE_WEATHER,
    ZIP_IMAGE_PREFIX,
)
from bddcv.labels import image_attributes, stream_records  # noqa: E402

SUBSET_DIR = DATA_DIR / "source_daytime_clear"
IMAGES_DIR = SUBSET_DIR / "images"


def ensure_raw_labels(archive: Path) -> None:
    """Extract the two label JSONs from the archive if they are missing.

    Located by filename rather than by full path: BDD100K repacks nest the
    labels directory differently, and hardcoding one layout makes the pipeline
    fail on an otherwise valid copy of the dataset.
    """
    missing = {s: p for s, p in RAW_LABEL_FILES.items() if not p.exists()}
    if not missing:
        return

    if not archive.exists():
        raise SystemExit(
            f"label JSONs missing ({', '.join(p.name for p in missing.values())}) "
            f"and no archive at {archive}. Pass --archive /path/to/bdd100k.zip"
        )

    RAW_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"extracting {len(missing)} label file(s) from {archive.name}", flush=True)

    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        for split, target in missing.items():
            matches = [n for n in names if n.endswith(f"/{target.name}") or n == target.name]
            if not matches:
                found = [n for n in names if n.endswith(".json")][:10]
                raise SystemExit(
                    f"{target.name} not found in {archive.name}.\n"
                    f"JSON files present: {found or '(none)'}\n"
                    "This archive may be a different BDD100K release; the legacy "
                    "'bdd100k_labels_images_*.json' schema is required."
                )
            with zf.open(matches[0]) as src, open(target, "wb") as out:
                out.write(src.read())
            print(f"  {matches[0]} -> {target}", flush=True)


def build_manifest(split: str) -> list[str]:
    names = []
    for rec in stream_records(RAW_LABEL_FILES[split]):
        tod, weather = image_attributes(rec)
        if tod == SOURCE_TIMEOFDAY and weather == SOURCE_WEATHER:
            names.append(rec["name"])
    names.sort()  # deterministic order, independent of file order
    return names


def build_member_index(zf: zipfile.ZipFile, split: str) -> dict[str, str]:
    """Map image basename -> archive member, scoped to one split's subtree.

    This repack does not store train images flat: they are scattered across
    train/trainA, train/trainB, train/testA, train/testB and train/ itself.
    Scoping the index to the split subtree keeps the lookup unambiguous
    without depending on that internal foldering.
    """
    prefix = f"{ZIP_IMAGE_PREFIX}/{split}/"
    index = {}
    for member in zf.namelist():
        if member.startswith(prefix) and member.endswith(".jpg"):
            index[member.rsplit("/", 1)[-1]] = member
    return index


def extract(archive: Path, split: str, names: list[str]) -> tuple[int, int]:
    dest = IMAGES_DIR / split
    dest.mkdir(parents=True, exist_ok=True)

    extracted = skipped = 0
    t0 = time.time()
    with zipfile.ZipFile(archive) as zf:
        index = build_member_index(zf, split)
        print(f"  archive holds {len(index):,} {split} images", flush=True)

        missing = [n for n in names if n not in index]
        if missing:
            raise SystemExit(
                f"{len(missing)} images listed in labels are absent from the archive, "
                f"e.g. {missing[:3]}"
            )

        for i, name in enumerate(names, 1):
            target = dest / name
            if target.exists() and target.stat().st_size > 0:
                skipped += 1
            else:
                with zf.open(index[name]) as src, open(target, "wb") as out:
                    out.write(src.read())
                extracted += 1
            if i % 2000 == 0:
                rate = i / max(time.time() - t0, 1e-6)
                print(f"  {split}: {i}/{len(names)}  ({rate:.0f} img/s)", flush=True)
    return extracted, skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, default=ARCHIVE,
                    help=f"BDD100K zip (default: {ARCHIVE})")
    args = ap.parse_args()

    ensure_raw_labels(args.archive)

    SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    drifted = []
    for split in ("val", "train"):
        print(f"\n=== {split} ===", flush=True)
        names = build_manifest(split)
        manifest = SUBSET_DIR / f"{split}_images.txt"
        content = "\n".join(names) + "\n"

        # The manifests are tracked in git as the reproducibility record. If a
        # regenerated one differs, this archive selects a different subset and
        # any model trained on it is not comparable with the others.
        if manifest.exists() and manifest.read_text(encoding="utf-8") != content:
            drifted.append(manifest.name)

        manifest.write_text(content, encoding="utf-8")
        print(f"  manifest: {len(names):,} images -> {manifest.name}", flush=True)

        extracted, skipped = extract(args.archive, split, names)
        print(f"  extracted {extracted:,}, already present {skipped:,}")

    if drifted:
        print(
            f"\n*** WARNING: regenerated manifest(s) differ from the committed "
            f"version: {', '.join(drifted)}\n"
            "*** This archive selects a different image subset. A model trained "
            "on it is NOT comparable with models trained elsewhere.\n"
            "*** Run `git diff data/source_daytime_clear/` and resolve before training."
        )
    print("\ndone")


if __name__ == "__main__":
    main()
