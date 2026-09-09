"""Registry-driven prediction export for any approved detector."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bddcv.constants import DATA_DIR  # noqa: E402
from bddcv.paths import (  # noqa: E402
    PREDICTIONS_DIR, RFDETR_ENV_DIR, prepare_runtime, prepare_rfdetr_runtime,
    resolve_output, resolve_weights,
)
from bddcv.prediction import (  # noqa: E402
    image_ids_by_filename, rfdetr_result_to_coco, ultralytics_result_to_coco,
    write_coco_predictions, write_prediction_metadata,
    validated_prediction_images,
)
from bddcv.registry import ModelResolutionError, resolve_model_spec  # noqa: E402

SUBSET = DATA_DIR / "source_daytime_clear"
GT = SUBSET / "annotations" / "instances_val.json"
IMAGES = SUBSET / "images" / "val"


def predict_ultralytics(spec, checkpoint: Path, output: Path, args) -> Path:
    id_of = image_ids_by_filename(GT)
    validated_prediction_images(IMAGES, id_of)
    from bddcv.paths import prepare_ultralytics
    prepare_ultralytics()
    from ultralytics import RTDETR, YOLO
    model = (RTDETR if spec.model_id == "rtdetr-l" else YOLO)(str(checkpoint))
    records = []
    stream = model.predict(
        source=str(IMAGES), imgsz=args.imgsz, conf=args.conf, iou=args.iou,
        max_det=args.max_det, device=args.device, stream=True, verbose=False,
        save=False, project=str(output.parent), name=output.stem, exist_ok=True,
    )
    for result in stream:
        records.extend(ultralytics_result_to_coco(result, id_of))
    write_coco_predictions(records, output)
    return output


def predict_frcnn(spec, checkpoint: Path, output: Path, args) -> Path:
    import torch
    from torch.utils.data import DataLoader
    from bddcv.frcnn import CocoDetectionDataset, build_model, collate, predict_to_coco
    prepare_runtime()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CocoDetectionDataset(SUBSET / "images/val", GT)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, collate_fn=collate, pin_memory=True)
    model = build_model(pretrained=False).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state.get("model", state))
    return predict_to_coco(model, loader, device, output)


def predict_rfdetr(spec, checkpoint: Path, output: Path, args) -> Path:
    from bddcv.rfdetr import RFDETRAdapter
    id_of = image_ids_by_filename(GT)
    images = validated_prediction_images(args.adapter / "valid", id_of)
    prepare_rfdetr_runtime()
    model = RFDETRAdapter(
        model_id=spec.model_id, checkpoint=checkpoint, python=sys.executable,
    ).construct(for_training=False)
    records = []
    for image in images:
        detections = model.predict(
            str(image), threshold=args.conf, include_source_image=False,
        )
        converted = rfdetr_result_to_coco(
            {"path": str(image), "detections": detections}, id_of,
        )
        converted.sort(key=lambda item: item["score"], reverse=True)
        records.extend(converted[:args.max_det])
    return write_coco_predictions(records, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument("--model-id", required=True,
                        help="registry id; generic best.pt requires this")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--adapter", type=Path, default=Path("data/adapters/rfdetr"))
    args = parser.parse_args()
    try:
        spec = resolve_model_spec(args.model_id, args.weights)
    except ModelResolutionError as exc:
        parser.error(str(exc))
    expected_imgsz = 512 if spec.backend == "rfdetr" else 640
    if args.imgsz is not None and args.imgsz != expected_imgsz:
        parser.error(f"{spec.model_id} requires --imgsz {expected_imgsz}")
    args.imgsz = expected_imgsz
    if spec.backend == "rfdetr" and Path(sys.prefix).resolve() != RFDETR_ENV_DIR.resolve():
        from bddcv.rfdetr import isolated_python
        python = isolated_python()
        os.execve(
            str(python), [str(python), str(Path(__file__).resolve()), *sys.argv[1:]],
            os.environ.copy(),
        )
    output = resolve_output(args.out, PREDICTIONS_DIR / f"{spec.model_id}.json")
    checkpoint = args.weights if args.weights.exists() else resolve_weights(args.weights)
    if spec.backend == "ultralytics":
        result = predict_ultralytics(spec, checkpoint, output, args)
    elif spec.backend == "frcnn":
        result = predict_frcnn(spec, checkpoint, output, args)
    else:
        result = predict_rfdetr(spec, checkpoint, output, args)
    write_prediction_metadata(
        result.with_suffix(result.suffix + ".meta.json"), model_id=spec.model_id,
        backend=spec.backend, input_policy=spec.input_policy,
        imgsz=args.imgsz, confidence=args.conf, iou=args.iou,
        precision="backend default (unmeasured)", max_detections=args.max_det,
    )
    print(f"prediction export -> {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
