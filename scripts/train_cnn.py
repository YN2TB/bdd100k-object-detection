"""Train the hand-written simple or complex CNN object detector."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bddcv.checkpointing import (  # noqa: E402
    capture_rng_state, dataset_fingerprint, restore_rng_state,
)
from bddcv.cnn_detector import (  # noqa: E402
    GridDetectionDataset, build_cnn_detector, detection_loss, export_predictions,
)
from bddcv.constants import DATA_DIR  # noqa: E402
from bddcv.evaluation import evaluate  # noqa: E402

DATASET = DATA_DIR / "source_full"
ANNOTATIONS = DATASET / "annotations"


def atomic_save(value: dict, path: Path) -> None:
    """Write a checkpoint through a temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("simple-cnn", "complex-cnn"), required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--stop-after-epoch", type=int, default=0)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.workers < 0:
        parser.error("epochs/batch must be positive and workers cannot be negative")

    args.out = (args.out or Path("runs/train_full") / args.model).resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = GridDetectionDataset(
        DATASET / "images/train", ANNOTATIONS / "instances_train.json", train=True
    )
    val_data = GridDetectionDataset(
        DATASET / "images/val", ANNOTATIONS / "instances_val.json"
    )
    if args.limit_train:
        train_data.images = train_data.images[:args.limit_train]
    if args.limit_val:
        val_data.images = val_data.images[:args.limit_val]

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        pin_memory=True, generator=generator,
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch, shuffle=False, num_workers=args.workers,
        pin_memory=True,
    )
    model = build_cnn_detector(args.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    fingerprint = dataset_fingerprint(DATASET)
    config = {
        "model": args.model,
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "seed": args.seed,
        "dataset_fingerprint": fingerprint,
        "limit_train": args.limit_train,
        "limit_val": args.limit_val,
    }

    start_epoch, best = 1, -1.0
    last_checkpoint = args.out / "last.pt"
    results_csv = args.out / "results.csv"
    if args.resume:
        if not last_checkpoint.is_file():
            raise SystemExit(f"resume checkpoint not found: {last_checkpoint}")
        checkpoint = torch.load(last_checkpoint, map_location=device, weights_only=False)
        if checkpoint.get("config") != config:
            raise SystemExit("cannot resume: model, data, or training arguments changed")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng_state(checkpoint["rng"], generator)
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint["best"])
    else:
        with results_csv.open("w", newline="", encoding="utf-8") as output:
            csv.writer(output).writerow(["epoch", "loss", "mAP50_95", "seconds"])
        (args.out / "model.json").write_text(json.dumps(config, indent=2) + "\n")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss, batches, started = 0.0, 0, time.time()
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = {key: value.to(device, non_blocking=True) for key, value in targets.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                losses = detection_loss(model(images), targets)
            scaler.scale(losses["total"]).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(losses["total"].detach())
            batches += 1
            if batches % 100 == 0:
                print(f"epoch {epoch} batch {batches}/{len(train_loader)} loss {total_loss / batches:.4f}", flush=True)

        predictions = args.out / f"val_epoch{epoch}.json"
        export_predictions(model, val_loader, device, predictions)
        metrics = evaluate(ANNOTATIONS / "instances_val.json", predictions)
        score = metrics["overall"]["mAP50_95"]
        elapsed = time.time() - started
        print(f"epoch {epoch}: loss {total_loss / max(1, batches):.4f}, mAP50-95 {score:.4f}, {elapsed:.0f}s", flush=True)

        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best": max(best, score),
            "config": config,
            "rng": capture_rng_state(generator),
        }
        if score > best:
            best = score
            state["best"] = best
            atomic_save(state, args.out / "best.pt")
            (args.out / "best_metrics.json").write_text(json.dumps(metrics, indent=2))
        atomic_save(state, last_checkpoint)
        with results_csv.open("a", newline="", encoding="utf-8") as output:
            csv.writer(output).writerow([
                epoch, round(total_loss / max(1, batches), 6), round(score, 6), round(elapsed),
            ])
        predictions.unlink(missing_ok=True)
        if args.stop_after_epoch and epoch >= args.stop_after_epoch:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
