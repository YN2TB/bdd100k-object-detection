"""Census of the BDD100K label release.

Reports image counts per (timeofday x weather) cell and the class-instance
histogram for the source domain, so subset sizes are known before any
training decision is made.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bddcv.constants import (  # noqa: E402
    DET_CLASSES,
    RAW_LABEL_FILES,
    SOURCE_TIMEOFDAY,
    SOURCE_WEATHER,
    DATA_DIR,
)
from bddcv.labels import (  # noqa: E402
    detection_boxes,
    image_attributes,
    stream_records,
)


def scan(split: str) -> dict:
    path = RAW_LABEL_FILES[split]
    cells = Counter()
    raw_categories = Counter()
    source_class_instances = Counter()
    source_images = 0
    total = 0
    boxless_source = 0

    for rec in stream_records(path):
        total += 1
        tod, weather = image_attributes(rec)
        cells[(tod, weather)] += 1
        for label in rec.get("labels") or []:
            if label.get("category"):
                raw_categories[label["category"]] += 1
        if tod == SOURCE_TIMEOFDAY and weather == SOURCE_WEATHER:
            source_images += 1
            boxes = detection_boxes(rec)
            if not boxes:
                boxless_source += 1
            for idx, *_ in boxes:
                source_class_instances[DET_CLASSES[idx]] += 1
        if total % 20000 == 0:
            print(f"  ...{total} records", flush=True)

    return {
        "split": split,
        "total_images": total,
        "cells": {f"{t}|{w}": n for (t, w), n in sorted(cells.items())},
        "raw_categories": dict(raw_categories.most_common()),
        "source_images": source_images,
        "source_images_without_boxes": boxless_source,
        "source_class_instances": dict(source_class_instances.most_common()),
    }


def print_matrix(title: str, cells: dict) -> None:
    tods, weathers = [], []
    for key in cells:
        t, w = key.split("|")
        if t not in tods:
            tods.append(t)
        if w not in weathers:
            weathers.append(w)
    tods.sort()
    weathers.sort()
    width = max(len(w) for w in weathers) + 2
    print(f"\n{title}")
    header = "  " + "timeofday \ weather".ljust(22) + "".join(w.rjust(width) for w in weathers)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for t in tods:
        row = "  " + t.ljust(22)
        for w in weathers:
            row += str(cells.get(f"{t}|{w}", 0)).rjust(width)
        print(row)


def main() -> None:
    results = {}
    for split in ("val", "train"):
        print(f"\n=== scanning {split} ===", flush=True)
        results[split] = scan(split)

    for split, res in results.items():
        print_matrix(f"[{split}] images per condition (total {res['total_images']})", res["cells"])

    print("\n=== raw category strings present (train) ===")
    for cat, n in results["train"]["raw_categories"].items():
        print(f"  {cat:<20} {n:>10,}")

    print(f"\n=== source domain: {SOURCE_TIMEOFDAY} + {SOURCE_WEATHER} ===")
    for split, res in results.items():
        print(
            f"  {split:<6} images={res['source_images']:>7,}"
            f"   without boxes={res['source_images_without_boxes']:>6,}"
        )
    print("\n  class instances (train source domain):")
    tr = results["train"]["source_class_instances"]
    total_inst = sum(tr.values()) or 1
    for cls in DET_CLASSES:
        n = tr.get(cls, 0)
        print(f"    {cls:<16} {n:>10,}  {100 * n / total_inst:>6.2f}%")

    out = DATA_DIR / "label_census.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwritten -> {out}")


if __name__ == "__main__":
    main()
