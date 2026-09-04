"""Run a trained Faster R-CNN checkpoint over the val split -> COCO predictions.

Counterpart to predict_yolo.py. Both write the same format, and both feed
scripts/evaluate.py, which is the only place a reported number comes from.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bddcv.constants import DATA_DIR  # noqa: E402
from bddcv.frcnn import (  # noqa: E402
    CocoDetectionDataset,
    build_model,
    collate,
    predict_to_coco,
)

SUBSET = DATA_DIR / "source_daytime_clear"
GT = SUBSET / "annotations" / "instances_val.json"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", type=Path, help="best.pt from train_frcnn.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_ds = CocoDetectionDataset(SUBSET / "images/val", GT)
    val_ld = DataLoader(val_ds, batch_size=a.batch, shuffle=False,
                        num_workers=a.workers, collate_fn=collate, pin_memory=True)

    model = build_model(pretrained=False).to(device)
    ckpt = torch.load(a.weights, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"loaded epoch {ckpt.get('epoch')} from {a.weights}")

    predict_to_coco(model, val_ld, device, a.out)
    n = len(a.out.read_text(encoding="utf-8").split('"image_id"')) - 1
    print(f"{n:,} detections over {len(val_ds):,} images -> {a.out}")
