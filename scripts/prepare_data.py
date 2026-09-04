"""Select the source-domain subset and extract only those images.

The full daytime+clear pool is 12,454 train / 1,764 val images. That is small
enough to use in its entirety, so no subsampling seed is involved: the subset
is fully determined by the attribute filter and is therefore reproducible by
construction.

Only the selected images are pulled out of archive.zip (~1.2 GB) rather than
the whole 8 GB archive.
"""
from __future__ import annotations

import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bddcv.constants import (  # noqa: E402
    ARCHIVE,
    DATA_DIR,
    RAW_LABEL_FILES,
    SOURCE_TIMEOFDAY,
    SOURCE_WEATHER,
    ZIP_IMAGE_PREFIX,
)
from bddcv.labels import image_attributes, stream_records  # noqa: E402

SUBSET_DIR = DATA_DIR / "source_daytime_clear"
IMAGES_DIR = SUBSET_DIR / "images"


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


def extract(split: str, names: list[str]) -> tuple[int, int]:
    dest = IMAGES_DIR / split
    dest.mkdir(parents=True, exist_ok=True)

    extracted = skipped = 0
    t0 = time.time()
    with zipfile.ZipFile(ARCHIVE) as zf:
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
    SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("val", "train"):
        print(f"\n=== {split} ===", flush=True)
        names = build_manifest(split)
        manifest = SUBSET_DIR / f"{split}_images.txt"
        manifest.write_text("\n".join(names) + "\n", encoding="utf-8")
        print(f"  manifest: {len(names):,} images -> {manifest.name}", flush=True)

        extracted, skipped = extract(split, names)
        print(f"  extracted {extracted:,}, already present {skipped:,}")

    print("\ndone")


if __name__ == "__main__":
    main()
