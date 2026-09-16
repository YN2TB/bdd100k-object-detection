"""Verify the completed RT-DETR baseline by loading and exporting one image."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bddcv.paths import DATA_DIR, TRAIN_DIR, VERIFICATION_DIR, prepare_ultralytics
from bddcv.prediction import ultralytics_result_to_coco, write_coco_predictions


def checksum(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=VERIFICATION_DIR / "rtdetr_baseline")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    baseline = TRAIN_DIR / "rtdetr-l"
    protected = [baseline / name for name in (
        "weights/best.pt", "weights/last.pt", "results.csv", "logs/supervisor.log",
    )]
    before = {str(path.relative_to(baseline)): checksum(path) for path in protected}
    subset = DATA_DIR / "source_daytime_clear"
    gt = json.loads((subset / "annotations/instances_val.json").read_text())
    image = sorted(gt["images"], key=lambda item: item["file_name"])[0]
    prepare_ultralytics()
    import torch
    import ultralytics
    from ultralytics import RTDETR

    model = RTDETR(str(baseline / "weights/best.pt"))
    shapes: set[tuple[int, ...]] = set()
    hook = model.model.register_forward_pre_hook(
        lambda module, inputs: shapes.add(tuple(inputs[0].shape))
        if inputs and isinstance(inputs[0], torch.Tensor) else None
    )
    try:
        result = model.predict(
            source=str(subset / "images/val" / image["file_name"]),
            imgsz=640, conf=0.001, max_det=300, device=args.device,
            save=False, verbose=False,
        )[0]
    finally:
        hook.remove()
    records = ultralytics_result_to_coco(result, {image["file_name"]: image["id"]})
    args.out.mkdir(parents=True, exist_ok=True)
    write_coco_predictions(records, args.out / "predictions.json")
    after = {str(path.relative_to(baseline)): checksum(path) for path in protected}
    evidence = {
        "status": "pass" if before == after else "fail",
        "model_id": "rtdetr-l", "image": image, "detections": len(records),
        "tensor_shapes": sorted(shapes), "precision": str(next(model.model.parameters()).dtype),
        "torch": torch.__version__, "ultralytics": ultralytics.__version__,
        "device": args.device, "before_sha256": before, "after_sha256": after,
        "centralized_evaluation_run": False,
    }
    (args.out / "verification.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))
    return 0 if before == after else 1


if __name__ == "__main__":
    raise SystemExit(main())
