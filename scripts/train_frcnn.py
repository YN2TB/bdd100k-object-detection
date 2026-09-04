"""Train Faster R-CNN R50-FPN v2 on the BDD100K source domain.

Per-epoch validation goes through the same pycocotools evaluator as YOLO, so
the two models' numbers are directly comparable. Best checkpoint is selected
on val mAP@[.5:.95], never the last epoch.

Two checkpoints are kept in <out>:
  best.pt  weights at the highest val mAP@[.5:.95], for evaluation
  last.pt  full training state every epoch, for --resume

Resuming restores the optimizer, the GradScaler and the global iteration
counter. That counter drives the cosine LR schedule, so losing it would
restart the schedule at warmup and quietly change the experiment rather than
fail loudly - hence the schedule guard in main().
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bddcv.constants import DATA_DIR  # noqa: E402
from bddcv.evaluation import evaluate, format_report  # noqa: E402
from bddcv.frcnn import CocoDetectionDataset, build_model, collate  # noqa: E402

SUBSET = DATA_DIR / "source_daytime_clear"
ANN = SUBSET / "annotations"


@torch.no_grad()
def predict(model, loader, device, out_json: Path) -> Path:
    model.eval()
    preds = []
    for images, targets in loader:
        images = [i.to(device, non_blocking=True) for i in images]
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            outputs = model(images)
        for t, o in zip(targets, outputs):
            img_id = int(t["image_id"])
            boxes = o["boxes"].float().cpu()
            for (x1, y1, x2, y2), s, c in zip(
                boxes.tolist(), o["scores"].float().cpu().tolist(),
                o["labels"].cpu().tolist()
            ):
                preds.append({
                    "image_id": img_id, "category_id": int(c),
                    "bbox": [round(x1, 2), round(y1, 2),
                             round(x2 - x1, 2), round(y2 - y1, 2)],
                    "score": round(float(s), 5),
                })
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(preds), encoding="utf-8")
    return out_json


def atomic_save(obj, path: Path) -> None:
    """torch.save via a temp file plus rename.

    Power loss part-way through a plain torch.save leaves a truncated file -
    precisely the failure checkpointing exists to survive. os.replace is atomic
    within a filesystem on Windows as well as POSIX.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


# Arguments that determine the LR schedule or the iteration count. Resuming
# with any of these changed would silently produce a different LR trajectory
# from the one the run started with, so a mismatch is refused rather than
# quietly accepted.
SCHEDULE_KEYS = ("epochs", "batch", "lr", "warmup_iters", "limit_train")


def schedule_of(args) -> dict:
    return {k: getattr(args, k) for k in SCHEDULE_KEYS}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=4, help="4 fits 8GB; use 8 on 12GB")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--warmup-iters", type=int, default=500)
    p.add_argument("--out", type=Path, default=Path("runs/frcnn"))
    p.add_argument("--limit-train", type=int, default=0, help="smoke test on N images")
    p.add_argument("--resume", action="store_true",
                   help="continue from <out>/last.pt, restoring optimizer, scaler and LR position")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)

    train_ds = CocoDetectionDataset(SUBSET / "images/train", ANN / "instances_train.json", train=True)
    val_ds = CocoDetectionDataset(SUBSET / "images/val", ANN / "instances_val.json")
    if args.limit_train:
        train_ds.images = train_ds.images[: args.limit_train]

    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, collate_fn=collate, pin_memory=True)
    val_ld = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, collate_fn=collate, pin_memory=True)

    model = build_model(pretrained=True).to(device)
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    total_iters = args.epochs * len(train_ld)
    def lr_at(it: int) -> float:
        if it < args.warmup_iters:
            return args.lr * (it + 1) / args.warmup_iters
        prog = (it - args.warmup_iters) / max(1, total_iters - args.warmup_iters)
        return args.lr * 0.5 * (1 + math.cos(math.pi * prog))

    start_epoch, it, best = 1, 0, -1.0
    ckpt_path = args.out / "last.pt"

    if args.resume:
        if not ckpt_path.exists():
            raise SystemExit(f"--resume given but {ckpt_path} does not exist")
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        drift = {k: (ck.get("schedule", {}).get(k), v)
                 for k, v in schedule_of(args).items()
                 if ck.get("schedule", {}).get(k) != v}
        if drift:
            raise SystemExit(
                "cannot resume: these arguments define the LR schedule and differ "
                f"from the checkpoint (saved, given) -> {drift}\n"
                "Re-run with the original values, or drop --resume to start fresh."
            )
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, it, best = ck["epoch"] + 1, ck["it"], ck["best"]
        if start_epoch > args.epochs:
            raise SystemExit(
                f"checkpoint already finished all {args.epochs} epochs (best mAP50-95 {best:.4f})"
            )
        print(f"resumed at epoch {start_epoch}/{args.epochs} "
              f"(iteration {it}, best mAP50-95 so far {best:.4f})", flush=True)

    csv_path = args.out / "results.csv"
    if not (args.resume and csv_path.exists()):
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(["epoch", "train_loss", "mAP50_95", "mAP50", "seconds"])

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running, n, t0 = 0.0, 0, time.time()
        for images, targets in train_ld:
            for g in opt.param_groups:
                g["lr"] = lr_at(it)
            images = [i.to(device, non_blocking=True) for i in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.autocast("cuda", enabled=device.type == "cuda"):
                loss = sum(model(images, targets).values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += float(loss)
            n += 1
            it += 1
            if n % 100 == 0:
                print(f"  ep{epoch} it{n}/{len(train_ld)} loss {running / n:.4f}", flush=True)

        pred_json = predict(model, val_ld, device, args.out / f"pred_epoch{epoch}.json")
        res = evaluate(ANN / "instances_val.json", pred_json)
        m = res["overall"]
        secs = time.time() - t0
        print(f"epoch {epoch}: loss {running / max(n,1):.4f}  "
              f"mAP50-95 {m['mAP50_95']:.4f}  mAP50 {m['mAP50']:.4f}  ({secs:.0f}s)", flush=True)

        with open(csv_path, "a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([epoch, round(running / max(n, 1), 5),
                                     round(m["mAP50_95"], 5), round(m["mAP50"], 5), round(secs)])

        if m["mAP50_95"] > best:
            best = m["mAP50_95"]
            atomic_save({"model": model.state_dict(), "epoch": epoch, "metrics": res},
                        args.out / "best.pt")
            (args.out / "best_metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

        # Written every epoch, not only on improvement: without this a crash
        # falls back to whenever the score last improved. Saved after `best` is
        # updated so the checkpoint records the correct high-water mark.
        atomic_save({
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "it": it,
            "best": best,
            "schedule": schedule_of(args),
        }, ckpt_path)

        pred_json.unlink(missing_ok=True)

    print(format_report(json.loads((args.out / "best_metrics.json").read_text()),
                        f"Faster R-CNN R50-FPN v2 (best of {args.epochs} epochs)"))


if __name__ == "__main__":
    main()
