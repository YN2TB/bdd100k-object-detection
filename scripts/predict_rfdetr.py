"""Export RF-DETR Small predictions from the isolated environment to COCO JSON."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bddcv.constants import DATA_DIR  # noqa: E402
from bddcv.paths import PREDICTIONS_DIR, resolve_output, resolve_weights  # noqa: E402
from bddcv.prediction import (  # noqa: E402
    image_ids_by_filename, rfdetr_result_to_coco, write_coco_predictions,
    write_prediction_metadata,
    validated_prediction_images,
)
from bddcv.rfdetr import DEFAULT_ADAPTER, RFDETRAdapter  # noqa: E402

GT = DATA_DIR / "source_daytime_clear" / "annotations" / "instances_val.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument("--model-id", default="rfdetr-small")
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.model_id != "rfdetr-small":
        parser.error("only rfdetr-small is supported")
    output = resolve_output(args.out, PREDICTIONS_DIR / "rfdetr-small.json")
    ids = image_ids_by_filename(GT)
    images = validated_prediction_images(args.adapter / "valid", ids)
    adapter = RFDETRAdapter(model_id=args.model_id, checkpoint=resolve_weights(args.weights), python=sys.executable)
    model = adapter.construct(for_training=False)
    records = []
    for image in images:
        result = model.predict(str(image))
        # Some releases return a bare Detections object without a path.
        if not hasattr(result, "path") and not isinstance(result, dict):
            result = {"path": str(image), "detections": result}
        elif isinstance(result, dict) and not any(key in result for key in ("path", "file_name", "filename")):
            result = {**result, "path": str(image)}
        records.extend(rfdetr_result_to_coco(result, ids))
    write_coco_predictions(records, output)
    write_prediction_metadata(
        output.with_suffix(output.suffix + ".meta.json"), model_id=args.model_id,
        backend="rfdetr", input_policy="512 native", precision="amp",
    )
    print(f"{len(records):,} detections over {len(ids):,} images -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
