"""Two independent checks on the converted labels.

1. Numeric: decode every YOLO box back to pixels and compare against the COCO
   box for the same annotation. Catches normalisation, xywh/xyxy and axis-flip
   bugs across the whole dataset, not just a sample.
2. Visual: render a montage so a human can confirm boxes land on real objects.
   A conversion can be self-consistent and still be uniformly wrong.
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bddcv.constants import DATA_DIR, DET_CLASSES  # noqa: E402

SUBSET_DIR = DATA_DIR / "source_daytime_clear"
TOL = 1.0  # pixels; YOLO txt is written at 6dp so round-trip error is sub-pixel

PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
]


def numeric_check(split: str) -> None:
    coco = json.loads((SUBSET_DIR / "annotations" / f"instances_{split}.json").read_text())
    by_image = defaultdict(list)
    for a in coco["annotations"]:
        by_image[a["image_id"]].append(a)
    meta = {im["id"]: im for im in coco["images"]}

    checked = worst = 0
    mismatch = []
    for img_id, anns in by_image.items():
        im = meta[img_id]
        w, h = im["width"], im["height"]
        txt = SUBSET_DIR / "labels" / split / f"{Path(im['file_name']).stem}.txt"
        yolo = []
        for line in txt.read_text().split("\n"):
            if not line.strip():
                continue
            c, xc, yc, bw, bh = line.split()
            c, xc, yc, bw, bh = int(c), float(xc), float(yc), float(bw), float(bh)
            yolo.append((c, (xc - bw / 2) * w, (yc - bh / 2) * h, bw * w, bh * h))

        if len(yolo) != len(anns):
            mismatch.append(f"{im['file_name']}: {len(yolo)} yolo vs {len(anns)} coco")
            continue
        for (c, x, y, bw, bh), a in zip(yolo, anns):
            if c + 1 != a["category_id"]:
                mismatch.append(f"{im['file_name']}: class {c + 1} vs {a['category_id']}")
                break
            d = max(abs(x - a["bbox"][0]), abs(y - a["bbox"][1]),
                    abs(bw - a["bbox"][2]), abs(bh - a["bbox"][3]))
            worst = max(worst, d)
            checked += 1

    status = "OK" if not mismatch and worst <= TOL else "FAIL"
    print(f"  [{status}] {split}: {checked:,} boxes cross-checked, "
          f"max deviation {worst:.3f}px, {len(mismatch)} mismatches")
    for m in mismatch[:5]:
        print(f"      {m}")


def montage(split: str, n: int = 6, seed: int = 0) -> Path:
    coco = json.loads((SUBSET_DIR / "annotations" / f"instances_{split}.json").read_text())
    by_image = defaultdict(list)
    for a in coco["annotations"]:
        by_image[a["image_id"]].append(a)
    # prefer busy images - more chances to spot a systematic offset
    candidates = sorted(by_image, key=lambda i: -len(by_image[i]))[:200]
    random.Random(seed).shuffle(candidates)
    picks = candidates[:n]
    meta = {im["id"]: im for im in coco["images"]}

    cols, scale = 2, 0.5
    tw, th = int(1280 * scale), int(720 * scale)
    rows = (n + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * th), "black")

    for k, img_id in enumerate(picks):
        im_meta = meta[img_id]
        img = Image.open(SUBSET_DIR / "images" / split / im_meta["file_name"]).convert("RGB")
        d = ImageDraw.Draw(img)
        for a in by_image[img_id]:
            x, y, bw, bh = a["bbox"]
            cid = a["category_id"] - 1
            d.rectangle([x, y, x + bw, y + bh], outline=PALETTE[cid], width=3)
            d.text((x + 2, max(0, y - 11)), DET_CLASSES[cid], fill=PALETTE[cid])
        img = img.resize((tw, th))
        sheet.paste(img, ((k % cols) * tw, (k // cols) * th))

    out = SUBSET_DIR / f"verify_{split}.png"
    sheet.save(out)
    return out


if __name__ == "__main__":
    print("numeric YOLO <-> COCO cross-check:")
    for s in ("val", "train"):
        numeric_check(s)
    p = montage("val")
    print(f"\nmontage -> {p}")
