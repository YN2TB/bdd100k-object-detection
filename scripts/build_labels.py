"""Convert the filtered subset into YOLO txt and COCO json from one source pass.

Both target formats are written in the same loop from the same filtered
records, so the two models cannot silently end up training on different data.

Category id convention:
  YOLO class index  = i           (0-based, order fixed in constants.DET_CLASSES)
  COCO category_id  = i + 1       (1-based; pycocotools convention)
Prediction converters must add 1 when going YOLO -> COCO.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bddcv.constants import (  # noqa: E402
    DATA_DIR,
    DET_CLASSES,
    RAW_LABEL_FILES,
)
from bddcv.labels import detection_boxes, stream_records  # noqa: E402

SUBSET_DIR = DATA_DIR / "source_daytime_clear"
IMAGES_DIR = SUBSET_DIR / "images"
LABELS_DIR = SUBSET_DIR / "labels"
ANN_DIR = SUBSET_DIR / "annotations"

MIN_SIDE = 2.0  # drop boxes thinner than this after clamping


def build(split: str) -> dict:
    manifest = (SUBSET_DIR / f"{split}_images.txt").read_text(encoding="utf-8").split()
    wanted = set(manifest)
    img_dir = IMAGES_DIR / split
    lbl_dir = LABELS_DIR / split
    lbl_dir.mkdir(parents=True, exist_ok=True)

    coco_images, coco_anns = [], []
    dims = Counter()
    per_class = Counter()
    n_clamped = n_dropped = 0
    ann_id = 1

    for img_id, rec in enumerate(
        (r for r in stream_records(RAW_LABEL_FILES[split]) if r["name"] in wanted), 1
    ):
        name = rec["name"]
        path = img_dir / name
        with Image.open(path) as im:
            w, h = im.size
        dims[(w, h)] += 1

        coco_images.append({"id": img_id, "file_name": name, "width": w, "height": h})
        lines = []

        for idx, x1, y1, x2, y2 in detection_boxes(rec):
            cx1, cy1 = max(0.0, min(x1, w)), max(0.0, min(y1, h))
            cx2, cy2 = max(0.0, min(x2, w)), max(0.0, min(y2, h))
            if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
                n_clamped += 1
            bw, bh = cx2 - cx1, cy2 - cy1
            if bw < MIN_SIDE or bh < MIN_SIDE:
                n_dropped += 1
                continue

            per_class[DET_CLASSES[idx]] += 1
            lines.append(
                f"{idx} {((cx1 + cx2) / 2) / w:.6f} {((cy1 + cy2) / 2) / h:.6f} "
                f"{bw / w:.6f} {bh / h:.6f}"
            )
            coco_anns.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": idx + 1,
                "bbox": [round(cx1, 2), round(cy1, 2), round(bw, 2), round(bh, 2)],
                "area": round(bw * bh, 2),
                "iscrowd": 0,
            })
            ann_id += 1

        (lbl_dir / f"{Path(name).stem}.txt").write_text(
            ("\n".join(lines) + "\n") if lines else "", encoding="utf-8"
        )

    ANN_DIR.mkdir(parents=True, exist_ok=True)
    coco = {
        "images": coco_images,
        "annotations": coco_anns,
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(DET_CLASSES)],
    }
    (ANN_DIR / f"instances_{split}.json").write_text(json.dumps(coco), encoding="utf-8")

    return {
        "split": split,
        "images": len(coco_images),
        "boxes": len(coco_anns),
        "clamped": n_clamped,
        "dropped": n_dropped,
        "dims": dims,
        "per_class": per_class,
    }


def main() -> None:
    stats = {}
    for split in ("val", "train"):
        print(f"=== {split} ===", flush=True)
        s = build(split)
        stats[split] = s
        print(f"  images {s['images']:,}  boxes {s['boxes']:,}"
              f"  clamped {s['clamped']:,}  dropped {s['dropped']:,}")
        print(f"  dimensions: {dict(s['dims'])}")

    print(f"\n{'class':<16}{'train':>10}{'val':>10}")
    print("-" * 36)
    for c in DET_CLASSES:
        print(f"{c:<16}{stats['train']['per_class'].get(c, 0):>10,}"
              f"{stats['val']['per_class'].get(c, 0):>10,}")


if __name__ == "__main__":
    main()
