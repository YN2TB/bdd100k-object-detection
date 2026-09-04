"""CLI: evaluate a COCO-format predictions file against the val ground truth."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bddcv.constants import DATA_DIR  # noqa: E402
from bddcv.evaluation import evaluate, format_report  # noqa: E402

DEFAULT_GT = DATA_DIR / "source_daytime_clear" / "annotations" / "instances_val.json"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", type=Path)
    ap.add_argument("--gt", type=Path, default=DEFAULT_GT)
    ap.add_argument("--title", default=None)
    ap.add_argument("--save", type=Path, default=None)
    a = ap.parse_args()

    res = evaluate(a.gt, a.predictions)
    print(format_report(res, a.title or a.predictions.stem))
    if a.save:
        a.save.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"\nsaved -> {a.save}")
