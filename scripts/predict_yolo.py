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
from bddcv.evaluation import image_id_map  # noqa: E402

SUBSET = DATA_DIR / "source_daytime_clear"
GT = SUBSET / "annotations" / "instances_val.json"
IMAGES = SUBSET / "images" / "val"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()

    from ultralytics import YOLO

    id_of = image_id_map(GT)
    model = YOLO(str(a.weights))
    preds = []

    stream = model.predict(
        source=str(IMAGES), imgsz=a.imgsz, conf=a.conf, iou=a.iou,
        max_det=a.max_det, device=a.device, stream=True, verbose=False,
    )
    for r in stream:
        name = Path(r.path).name
        img_id = id_of.get(name)
        if img_id is None:
            raise SystemExit(f"{name} is not in the ground truth file")
        b = r.boxes
        if b is None or len(b) == 0:
            continue
        xyxy = b.xyxy.cpu().numpy()
        conf = b.conf.cpu().numpy()
        cls = b.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), s, c in zip(xyxy, conf, cls):
            preds.append({
                "image_id": img_id,
                "category_id": int(c) + 1,       # YOLO idx -> COCO id
                "bbox": [round(float(x1), 2), round(float(y1), 2),
                         round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                "score": round(float(s), 5),
            })

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(preds), encoding="utf-8")
    print(f"{len(preds):,} detections over {len(id_of):,} images -> {a.out}")
