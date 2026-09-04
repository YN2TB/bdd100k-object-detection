"""Single COCO evaluation path shared by both detectors.

Ultralytics reports its own mAP, which is not COCO mAP. Using its internal
number for one model and pycocotools for the other would make the headline
comparison invalid. Every reported number in this project comes from here.

Both models must emit predictions in standard COCO detection format:
    [{"image_id": int, "category_id": int, "bbox": [x, y, w, h], "score": float}]
with category_id = DET_CLASSES index + 1, and image_id taken from the ground
truth file (map it by file_name, never by enumeration order).
"""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .constants import DET_CLASSES

# Classes with too few validation instances for AP to be meaningful.
# BDD100K daytime+clear holds exactly 2 'train' boxes in val, so its AP is
# noise; it is reported but excluded from the headline mean.
MIN_VAL_INSTANCES = 10


def image_id_map(gt_json: Path) -> dict[str, int]:
    """file_name -> image_id, so predictions never rely on ordering."""
    data = json.loads(Path(gt_json).read_text(encoding="utf-8"))
    return {im["file_name"]: im["id"] for im in data["images"]}


def evaluate(gt_json: Path, pred_json: Path) -> dict:
    """Run COCOeval and return overall, per-size and per-class metrics."""
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(gt_json))
        preds = json.loads(Path(pred_json).read_text(encoding="utf-8"))
        if not preds:
            raise ValueError(f"{pred_json} contains no detections")
        coco_dt = coco_gt.loadRes(preds)
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()

    s = ev.stats
    overall = {
        "mAP50_95": float(s[0]), "mAP50": float(s[1]), "mAP75": float(s[2]),
        "mAP_small": float(s[3]), "mAP_medium": float(s[4]), "mAP_large": float(s[5]),
        "AR_100": float(s[8]),
    }

    # Per-class AP@[.5:.95]: precision is [iou, recall, cls, area, maxdet]
    prec = ev.eval["precision"]
    gt_counts = {c: 0 for c in DET_CLASSES}
    for ann in coco_gt.dataset["annotations"]:
        gt_counts[DET_CLASSES[ann["category_id"] - 1]] += 1

    per_class = {}
    for i, name in enumerate(DET_CLASSES):
        p = prec[:, :, i, 0, 2]
        p = p[p > -1]
        per_class[name] = {
            "AP50_95": float(np.mean(p)) if p.size else float("nan"),
            "val_instances": gt_counts[name],
            "reliable": gt_counts[name] >= MIN_VAL_INSTANCES,
        }

    reliable = [v["AP50_95"] for v in per_class.values()
                if v["reliable"] and not np.isnan(v["AP50_95"])]
    overall["mAP50_95_reliable_classes"] = float(np.mean(reliable)) if reliable else float("nan")
    overall["n_reliable_classes"] = len(reliable)

    return {"overall": overall, "per_class": per_class}


def format_report(results: dict, title: str) -> str:
    o, pc = results["overall"], results["per_class"]
    w = 62
    out = [f"\n{title}", "=" * w,
           f"{'mAP@[.5:.95]':<26}{o['mAP50_95']:>10.4f}",
           f"{'mAP@.50':<26}{o['mAP50']:>10.4f}",
           f"{'mAP@.75':<26}{o['mAP75']:>10.4f}",
           "-" * w,
           f"{'mAP small':<26}{o['mAP_small']:>10.4f}",
           f"{'mAP medium':<26}{o['mAP_medium']:>10.4f}",
           f"{'mAP large':<26}{o['mAP_large']:>10.4f}",
           "-" * w,
           f"{'class':<18}{'AP@[.5:.95]':>14}{'val boxes':>12}{'':>4}"]
    for name, v in pc.items():
        flag = "" if v["reliable"] else "  (too few)"
        out.append(f"{name:<18}{v['AP50_95']:>14.4f}{v['val_instances']:>12,}{flag}")
    out += ["-" * w,
            f"mean over {o['n_reliable_classes']} reliable classes"
            f"{o['mAP50_95_reliable_classes']:>16.4f}", "=" * w]
    return "\n".join(out)
