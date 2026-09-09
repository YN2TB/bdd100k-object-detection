"""Run a trained YOLO checkpoint over the val split -> COCO predictions json.

conf is deliberately near zero: mAP integrates precision over the full recall
range, so filtering detections early silently depresses the score.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bddcv.constants import DATA_DIR  # noqa: E402
from bddcv.paths import (  # noqa: E402
    PREDICTIONS_DIR, prepare_ultralytics, resolve_output, resolve_weights,
)
from bddcv.evaluation import image_id_map  # noqa: E402
from bddcv.prediction import (  # noqa: E402
    ultralytics_result_to_coco, write_coco_predictions, write_prediction_metadata,
)
from bddcv.registry import ModelResolutionError, resolve_model_spec  # noqa: E402

SUBSET = DATA_DIR / "source_daytime_clear"
GT = SUBSET / "annotations" / "instances_val.json"
IMAGES = SUBSET / "images" / "val"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", type=Path)
    ap.add_argument("--model-id", default=None,
                    help="registry id; required for generic best.pt/last.pt")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: runs/predictions/yolo.json")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    a.out = resolve_output(a.out, PREDICTIONS_DIR / "yolo.json")
    try:
        spec = resolve_model_spec(a.model_id, a.weights)
    except ModelResolutionError as exc:
        raise SystemExit(str(exc)) from exc
    if spec.backend != "ultralytics":
        raise SystemExit(f"{spec.model_id} is not an Ultralytics backend")
    prepare_ultralytics()

    from ultralytics import RTDETR, YOLO

    id_of = image_id_map(GT)
    checkpoint = a.weights if a.weights.exists() else resolve_weights(a.weights)
    model_class = RTDETR if spec.model_id == "rtdetr-l" else YOLO
    model = model_class(str(checkpoint))
    preds = []

    stream = model.predict(
        source=str(IMAGES), imgsz=a.imgsz, conf=a.conf, iou=a.iou,
        max_det=a.max_det, device=a.device, stream=True, verbose=False, save=False,
        project=str(a.out.parent), name=a.out.stem, exist_ok=True,
    )
    for r in stream:
        name = Path(r.path).name
        img_id = id_of.get(name)
        if img_id is None:
            raise SystemExit(f"{name} is not in the ground truth file")
        # Keep the filename->ID lookup explicit; the converter accepts empty
        # boxes and emits no records for those images.
        result = ultralytics_result_to_coco(r, id_of)
        preds.extend(result)

    write_coco_predictions(preds, a.out)
    write_prediction_metadata(
        a.out.with_suffix(a.out.suffix + ".meta.json"),
        model_id=spec.model_id, backend=spec.backend, checkpoint=str(checkpoint),
        image_size=a.imgsz, confidence=a.conf, iou=a.iou,
        max_detections=a.max_det, precision=spec.precision,
        postprocessing="Ultralytics native",
    )
    print(f"{len(preds):,} detections over {len(id_of):,} images -> {a.out}")
