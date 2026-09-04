"""Streaming reader for the legacy BDD100K label release.

bdd100k_labels_images_train.json is ~1.45 GB. json.load() on it needs roughly
8-12 GB of RAM, which does not fit alongside a training process. These helpers
walk the top-level array one record at a time with a bounded buffer instead.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .constants import (
    CLASS_TO_IDX,
    NON_DET_CATEGORIES,
    canonical_category,
)

_CHUNK = 8 << 20  # 8 MiB
_TRIM_AT = 32 << 20  # compact the buffer once consumed bytes exceed this


def stream_records(path: Path, chunk_size: int = _CHUNK) -> Iterator[dict]:
    """Yield each object in a top-level JSON array without loading the file.

    Memory stays bounded by chunk_size plus the largest single record.
    """
    decoder = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as fh:
        buf = fh.read(chunk_size)
        if not buf:
            return
        i = 0
        while i < len(buf) and buf[i].isspace():
            i += 1
        if i < len(buf) and buf[i] == "[":
            i += 1

        while True:
            # Advance past separators, refilling if the buffer runs dry.
            while True:
                while i < len(buf) and (buf[i].isspace() or buf[i] == ","):
                    i += 1
                if i < len(buf):
                    break
                more = fh.read(chunk_size)
                if not more:
                    return
                buf, i = more, 0

            if buf[i] == "]":
                return

            # Decode one record, extending the buffer until it is complete.
            while True:
                try:
                    record, end = decoder.raw_decode(buf, i)
                    break
                except ValueError:
                    more = fh.read(chunk_size)
                    if not more:
                        raise ValueError(f"truncated JSON array in {path}")
                    buf = buf[i:] + more
                    i = 0

            yield record
            i = end
            if i > _TRIM_AT:
                buf, i = buf[i:], 0


def image_attributes(record: dict) -> tuple[str, str]:
    """Return (timeofday, weather) for a record, defaulting to 'undefined'."""
    attrs = record.get("attributes") or {}
    return (
        attrs.get("timeofday", "undefined"),
        attrs.get("weather", "undefined"),
    )


def detection_boxes(record: dict) -> list[tuple[int, float, float, float, float]]:
    """Extract (class_idx, x1, y1, x2, y2) for the 10 detection classes.

    Drops poly2d-only annotations (drivable area, lane) and any category
    outside the detection set.
    """
    out = []
    for label in record.get("labels") or []:
        raw = label.get("category")
        if raw is None or raw in NON_DET_CATEGORIES:
            continue
        box = label.get("box2d")
        if not box:
            continue
        idx = CLASS_TO_IDX.get(canonical_category(raw))
        if idx is None:
            continue
        out.append(
            (idx, float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"]))
        )
    return out
