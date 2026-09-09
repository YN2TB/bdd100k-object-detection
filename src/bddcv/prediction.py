"""Common COCO prediction records and backend-neutral converters."""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .constants import DET_CLASSES


class PredictionFormatError(ValueError):
    """Raised when a backend output cannot be represented as COCO detections."""


@dataclass(frozen=True)
class CocoDetection:
    """One standard COCO bbox detection (empty images emit no records)."""

    image_id: int
    category_id: int
    bbox: tuple[float, float, float, float]
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": int(self.image_id),
            "category_id": int(self.category_id),
            "bbox": [round(float(value), 2) for value in self.bbox],
            "score": round(float(self.score), 5),
        }


def image_ids_by_filename(gt_json: str | Path) -> dict[str, int]:
    """Map ground-truth file names to IDs; ordering is never assumed."""
    path = Path(gt_json)
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(item["file_name"]): int(item["id"]) for item in data.get("images", [])}


image_id_map = image_ids_by_filename


def validated_prediction_images(directory: str | Path, ids: Mapping[str, int]) -> list[Path]:
    """Reject missing or extra validation images before producing detections."""
    root = Path(directory)
    images = sorted(root.glob("*.jpg"))
    actual = {image.name for image in images if image.is_file()}
    expected = set(ids)
    if actual != expected or not expected:
        raise PredictionFormatError(
            f"prediction image coverage mismatch: missing={sorted(expected - actual)[:3]}, "
            f"extra={sorted(actual - expected)[:3]}"
        )
    return images


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (str, bytes)):
        return [value]
    if not isinstance(value, (list, tuple)):
        try:
            return list(value)
        except TypeError:
            return [value]
    return list(value)


def _box_values(box: Any) -> tuple[float, float, float, float]:
    values = _as_list(box)
    if len(values) != 4:
        raise PredictionFormatError(f"bbox must have four values, got {box!r}")
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def detections_from_xyxy(
    image_id: int,
    boxes: Iterable[Sequence[float]] | Any,
    scores: Iterable[float] | Any,
    labels: Iterable[int] | Any,
    *,
    labels_are_coco: bool = False,
) -> list[dict[str, Any]]:
    """Convert xyxy outputs from any backend to standard COCO dictionaries."""
    box_values = _as_list(boxes)
    score_values = _as_list(scores)
    label_values = _as_list(labels)
    if not (len(box_values) == len(score_values) == len(label_values)):
        raise PredictionFormatError(
            f"detection fields have different lengths: boxes={len(box_values)}, "
            f"scores={len(score_values)}, labels={len(label_values)}"
        )
    output: list[dict[str, Any]] = []
    for box, score, label in zip(box_values, score_values, label_values):
        x1, y1, x2, y2 = _box_values(box)
        value = float(score)
        category = int(label) if labels_are_coco else int(label) + 1
        if not all(math.isfinite(item) for item in (x1, y1, x2, y2, value)):
            raise PredictionFormatError("non-finite bbox or score")
        if category < 1 or category > len(DET_CLASSES):
            raise PredictionFormatError(f"category id out of range: {category}")
        output.append(CocoDetection(
            int(image_id), category, (x1, y1, x2 - x1, y2 - y1), value
        ).to_dict())
    return output


def _get(result: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(result, Mapping) and name in result:
            return result[name]
        if hasattr(result, name):
            return getattr(result, name)
    return default


def _filename_from_result(result: Any) -> str | None:
    value = _get(result, "file_name", "filename", "path", "image_path", "name")
    if value is None:
        return None
    return Path(str(value)).name


def ultralytics_result_to_coco(
    result: Any,
    id_by_filename: Mapping[str, int],
) -> list[dict[str, Any]]:
    """Convert one YOLO or RT-DETR Ultralytics result, including empty output."""
    name = _filename_from_result(result)
    if name is None or name not in id_by_filename:
        raise PredictionFormatError(f"result filename is not in ground truth: {name!r}")
    boxes_obj = _get(result, "boxes")
    if boxes_obj is None:
        return []
    boxes = _get(boxes_obj, "xyxy", default=[])
    scores = _get(boxes_obj, "conf", "confidence", default=[])
    labels = _get(boxes_obj, "cls", "class_id", "classes", default=[])
    if boxes is None or len(_as_list(boxes)) == 0:
        return []
    return detections_from_xyxy(id_by_filename[name], boxes, scores, labels)


def _rfdetr_result_parts(result: Any) -> tuple[str | None, Any, Any, Any]:
    """Extract common supervision/RF-DETR fields from a result object."""
    if isinstance(result, (list, tuple)) and len(result) == 1:
        result = result[0]
    name = _filename_from_result(result)
    detections = _get(result, "detections", "predictions")
    if detections is not None and detections is not result:
        nested_name, nested_boxes, nested_scores, nested_labels = _rfdetr_result_parts(detections)
        name = name or nested_name
        return name, nested_boxes, nested_scores, nested_labels
    boxes = _get(result, "xyxy", "boxes", "bbox", default=[])
    scores = _get(result, "confidence", "conf", "scores", "score", default=[])
    labels = _get(result, "class_id", "class_ids", "labels", "classes", "class_ids", default=[])
    # RF-DETR's Detections may expose a numpy scalar for singleton fields.
    if boxes is None:
        boxes = []
    if scores is None:
        scores = []
    if labels is None:
        labels = []
    return name, boxes, scores, labels


def rfdetr_result_to_coco(
    result: Any,
    id_by_filename: Mapping[str, int],
    *,
    labels_are_coco: bool = False,
) -> list[dict[str, Any]]:
    """Convert an RF-DETR result while accepting its common output variants."""
    name, boxes, scores, labels = _rfdetr_result_parts(result)
    if name is None or name not in id_by_filename:
        raise PredictionFormatError(f"RF-DETR result filename is not in ground truth: {name!r}")
    if len(_as_list(boxes)) == 0:
        return []
    return detections_from_xyxy(
        id_by_filename[name], boxes, scores, labels, labels_are_coco=labels_are_coco
    )


def rfdetr_results_to_coco(
    results: Iterable[Any],
    id_by_filename: Mapping[str, int],
    *,
    labels_are_coco: bool = False,
) -> list[dict[str, Any]]:
    """Convert an iterable of RF-DETR results, preserving empty images."""
    output: list[dict[str, Any]] = []
    for result in results:
        output.extend(rfdetr_result_to_coco(result, id_by_filename,
                                             labels_are_coco=labels_are_coco))
    return output


def write_coco_predictions(
    predictions: Iterable[Mapping[str, Any]],
    out_json: str | Path,
) -> Path:
    """Validate and atomically write a COCO detection list (possibly empty)."""
    normalized: list[dict[str, Any]] = []
    for item in predictions:
        try:
            image_id = int(item["image_id"])
            category_id = int(item["category_id"])
            bbox = tuple(float(value) for value in item["bbox"])
            score = float(item["score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PredictionFormatError(f"invalid COCO detection: {item!r}") from exc
        if len(bbox) != 4 or not all(math.isfinite(value) for value in bbox + (score,)):
            raise PredictionFormatError(f"invalid bbox or score: {item!r}")
        if category_id < 1 or category_id > len(DET_CLASSES):
            raise PredictionFormatError(f"category id out of range: {category_id}")
        normalized.append(CocoDetection(image_id, category_id, bbox, score).to_dict())
    target = Path(out_json)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(normalized), encoding="utf-8")
    os.replace(tmp, target)
    return target


def write_prediction_metadata(path: str | Path, **metadata: Any) -> Path:
    """Write benchmark provenance next to a prediction export."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, target)
    return target


__all__ = [
    "CocoDetection", "PredictionFormatError", "detections_from_xyxy",
    "image_ids_by_filename", "image_id_map", "rfdetr_result_to_coco", "ultralytics_result_to_coco",
    "rfdetr_results_to_coco",
    "write_coco_predictions", "write_prediction_metadata",
]
