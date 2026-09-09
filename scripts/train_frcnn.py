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
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bddcv.constants import DATA_DIR  # noqa: E402
from bddcv.paths import TRAIN_DIR, prepare_runtime, resolve_output  # noqa: E402
from bddcv.evaluation import evaluate, format_report  # noqa: E402
from bddcv.frcnn import (  # noqa: E402
    CocoDetectionDataset, build_model, collate, predict_to_coco,
)
from bddcv.checkpointing import (  # noqa: E402
    capture_rng_state,
    dataset_fingerprint,
    reconcile_results_csv,
    require_resume_state,
    restore_rng_state,
    resume_mismatches,
    UnsafeResumeError,
)
from bddcv.registry import get_model_spec, write_model_metadata  # noqa: E402

SUBSET = DATA_DIR / "source_daytime_clear"
ANN = SUBSET / "annotations"


@torch.no_grad()
def predict(model, loader, device, out_json: Path) -> Path:
    """Compatibility wrapper around the shared Faster R-CNN exporter."""
    return predict_to_coco(model, loader, device, out_json)


def atomic_save(obj, path: Path) -> None:
    """torch.save via a temp file plus rename.

    Power loss part-way through a plain torch.save leaves a truncated file -
    precisely the failure checkpointing exists to survive. os.replace is atomic
    within a filesystem on Windows as well as POSIX.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


# Arguments that determine the LR schedule or the iteration count. Resuming
# with any of these changed would silently produce a different LR trajectory
# from the one the run started with, so a mismatch is refused rather than
# quietly accepted.
SCHEDULE_KEYS = ("epochs", "batch", "lr", "warmup_iters", "limit_train")

# State required for a faithful continuation.  ``config`` is nested so a
# future backend can add recipe details without changing the checkpoint's
# top-level compatibility contract.
RESUME_CONFIG_KEYS = (
    "model_id", "dataset_fingerprint", "resolution", "physical_batch",
    "effective_batch", "optimizer", "schedule", "workers", "limit_train",
    "validation_batch", "seed", "cache", "prefetch",
)
RESUME_STATE_KEYS = (
    "model", "optimizer", "scaler", "epoch", "it", "best", "schedule",
    "config", "rng",
)


def schedule_of(args) -> dict:
    return {k: getattr(args, k) for k in SCHEDULE_KEYS}


def training_config(args, fingerprint: str) -> dict:
    """Return immutable run settings used to reject unsafe resume drift."""
    batch = int(args.batch)
    imgsz = int(getattr(args, "imgsz", 640))
    validation_batch = int(getattr(args, "validation_batch", 4))
    seed = int(getattr(args, "seed", 0))
    return {
        "model_id": getattr(args, "model_id", "frcnn-r50-fpn-v2"),
        "dataset_fingerprint": fingerprint,
        "resolution": {
            "imgsz": imgsz,
            "min_size": 360,
            "max_size": 640,
        },
        # Keep these aliases at the top level because the supervisor writes the
        # same launch contract before starting the child trainer.  A child
        # overwriting model.json must remain restart-compatible.
        "batch": batch,
        "imgsz": imgsz,
        "epochs": int(args.epochs),
        "seed": seed,
        "physical_batch": batch,
        "effective_batch": batch * int(getattr(args, "accumulate", 1)),
        "validation_batch": validation_batch,
        "optimizer": {
            "name": str(getattr(args, "optimizer", "SGD")),
            "lr": float(args.lr),
            "momentum": 0.9,
            "weight_decay": 1e-4,
        },
        "schedule": {
            "name": str(getattr(args, "schedule", "warmup+cosine")),
            **schedule_of(args),
        },
        "workers": int(args.workers),
        "cache": getattr(args, "cache", "none"),
        "prefetch": int(getattr(args, "prefetch", 2)),
        "limit_train": int(args.limit_train),
    }


def seed_everything(seed: int) -> None:
    """Seed every process-level RNG used by the FRCNN dataset/loader."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_seed(worker_id: int) -> None:
    """Derive deterministic worker Python/NumPy streams from torch's seed."""
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def validate_recipe(args) -> None:
    """Reject CLI values that the fixed comparison implementation cannot honor."""
    if args.model_id != "frcnn-r50-fpn-v2":
        raise ValueError("only model_id=frcnn-r50-fpn-v2 is supported")
    if args.optimizer != "SGD":
        raise ValueError("only optimizer=SGD is implemented")
    if args.schedule != "warmup+cosine":
        raise ValueError("only schedule=warmup+cosine is implemented")
    if args.imgsz != 640:
        raise ValueError("only imgsz=640 preserves the matched-resolution contract")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="frcnn-r50-fpn-v2")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=4, help="4 fits 8GB; use 8 on 12GB")
    p.add_argument("--validation-batch", type=int, default=4,
                   help="validation batch; profiling and the default recipe use 4")
    p.add_argument("--accumulate", type=int, default=1)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--optimizer", default="SGD")
    p.add_argument("--schedule", default="warmup+cosine")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--warmup-iters", type=int, default=500)
    p.add_argument("--out", type=Path, default=None,
                   help="run directory (default: runs/train/frcnn)")
    p.add_argument("--limit-train", type=int, default=0, help="smoke test on N images")
    p.add_argument("--stop-after-epoch", type=int, default=0,
                   help="cleanly stop after this checkpoint (records incomplete status)")
    p.add_argument("--resume", action="store_true",
                   help="continue from <out>/last.pt, restoring optimizer, scaler and LR position")
    p.add_argument("--cache", choices=("none", "ram"), default="none")
    p.add_argument("--prefetch", type=int, choices=(2, 4), default=2)
    args = p.parse_args()
    if args.accumulate < 1:
        p.error("--accumulate must be positive")
    if args.batch < 1 or args.validation_batch < 1:
        p.error("batch and validation-batch must be positive")
    if args.epochs < 1:
        p.error("--epochs must be positive")
    try:
        validate_recipe(args)
    except ValueError as exc:
        p.error(str(exc))
    args.out = resolve_output(args.out, TRAIN_DIR / "frcnn")
    prepare_runtime()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    fingerprint = dataset_fingerprint(SUBSET)
    config = training_config(args, fingerprint)
    if not args.resume:
        write_model_metadata(args.out / "model.json", get_model_spec(args.model_id), **config)

    train_ds = CocoDetectionDataset(SUBSET / "images/train", ANN / "instances_train.json", train=True)
    val_ds = CocoDetectionDataset(SUBSET / "images/val", ANN / "instances_val.json")
    if args.limit_train:
        train_ds.images = train_ds.images[: args.limit_train]

    if args.cache == "ram":
        train_ds.cache_images()
        val_ds.cache_images()
    loader_options = {"prefetch_factor": args.prefetch} if args.workers else {}
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    train_ld = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        collate_fn=collate, pin_memory=True, generator=loader_generator,
        worker_init_fn=worker_seed, **loader_options,
    )
    val_ld = DataLoader(
        val_ds, batch_size=args.validation_batch, shuffle=False, num_workers=args.workers,
        collate_fn=collate, pin_memory=True, worker_init_fn=worker_seed, **loader_options,
    )

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
    csv_path = args.out / "results.csv"

    if args.resume:
        if not ckpt_path.exists():
            raise SystemExit(f"--resume given but {ckpt_path} does not exist")
        try:
            ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            require_resume_state(ck, RESUME_STATE_KEYS)
        except UnsafeResumeError as exc:
            raise SystemExit(str(exc)) from exc
        drift = resume_mismatches(ck["config"], config, RESUME_CONFIG_KEYS)
        if drift:
            raise SystemExit(
                "cannot resume: immutable run configuration differs from checkpoint "
                f"(saved, given) -> {drift}\n"
                "Re-run with the original values, or drop --resume to start fresh."
            )
        # Keep this explicit check for old callers that inspect schedule data.
        schedule_drift = {k: (ck.get("schedule", {}).get(k), v)
                          for k, v in schedule_of(args).items()
                          if ck.get("schedule", {}).get(k) != v}
        if schedule_drift:
            raise SystemExit(f"cannot resume: schedule differs from checkpoint: {schedule_drift}")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, it, best = int(ck["epoch"]) + 1, int(ck["it"]), float(ck["best"])
        # The checkpoint is authoritative.  Any CSV row written after a crash
        # is retained in a sidecar and cannot make the wrapper skip work.
        reconciliation = reconcile_results_csv(csv_path, int(ck["epoch"]))
        print(f"resume reconciliation: {reconciliation}", flush=True)
        restore_rng_state(ck["rng"], loader_generator)
        if start_epoch > args.epochs:
            status = {
                "status": "complete", "epoch": int(ck["epoch"]),
                "target_epochs": args.epochs, "best": best,
            }
            (args.out / "run_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
            print(f"checkpoint already finished all {args.epochs} epochs (best mAP50-95 {best:.4f})")
            return 0
        print(f"resumed at epoch {start_epoch}/{args.epochs} "
              f"(iteration {it}, best mAP50-95 so far {best:.4f})", flush=True)
    else:
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(["epoch", "train_loss", "mAP50_95", "mAP50", "seconds"])

    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running, n, t0 = 0.0, 0, time.time()
        opt.zero_grad(set_to_none=True)
        for images, targets in train_ld:
            for group in opt.param_groups:
                group["lr"] = lr_at(it)
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

            with torch.autocast("cuda", enabled=device.type == "cuda"):
                loss = sum(model(images, targets).values())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}, iteration {it}")
            scaler.scale(loss / args.accumulate).backward()
            n += 1
            if n % args.accumulate == 0 or n == len(train_ld):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            running += float(loss)
            it += 1
            if n % 100 == 0:
                print(f"  ep{epoch} it{n}/{len(train_ld)} loss {running / n:.4f}", flush=True)

        pred_json = predict(model, val_ld, device, args.out / f"pred_epoch{epoch}.json")
        res = evaluate(ANN / "instances_val.json", pred_json)
        metrics = res["overall"]
        secs = time.time() - t0
        print(f"epoch {epoch}: loss {running / max(n, 1):.4f}  "
              f"mAP50-95 {metrics['mAP50_95']:.4f}  mAP50 {metrics['mAP50']:.4f}  ({secs:.0f}s)", flush=True)

        if metrics["mAP50_95"] > best:
            best = metrics["mAP50_95"]
            best_payload = {
                "model": model.state_dict(), "epoch": epoch, "metrics": res,
                "model_id": args.model_id, "config": config,
            }
            atomic_save(best_payload, args.out / "best.pt")
            (args.out / "best_metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

        # Save the full state before appending the CSV row.  If power is lost
        # between these writes, checkpoint progress remains authoritative.
        rng_state = capture_rng_state(loader_generator)
        atomic_save({
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "it": it,
            "best": best,
            "schedule": schedule_of(args),
            "config": config,
            "model_id": args.model_id,
            "dataset_fingerprint": fingerprint,
            "resolution": config["resolution"],
            "physical_batch": args.batch,
            "effective_batch": config["effective_batch"],
            "validation_batch": args.validation_batch,
            "optimizer_name": args.optimizer,
            "schedule_name": args.schedule,
            "workers": args.workers,
            "rng": rng_state,
            "rng_state": rng_state,
            "loader_generator_state": rng_state.get("loader_generator"),
        }, ckpt_path)
        with csv_path.open("a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([epoch, round(running / max(n, 1), 5),
                                     round(metrics["mAP50_95"], 5),
                                     round(metrics["mAP50"], 5), round(secs)])

        pred_json.unlink(missing_ok=True)
        last_epoch = epoch
        if args.stop_after_epoch and epoch >= args.stop_after_epoch:
            break

    completed = last_epoch >= args.epochs
    status = {
        "status": "complete" if completed else "incomplete",
        "epoch": last_epoch, "target_epochs": args.epochs,
        "best": best, "model_id": args.model_id, "config": config,
    }
    (args.out / "run_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    if not completed:
        print(f"INCOMPLETE: clean stop at {last_epoch}/{args.epochs} epochs", flush=True)
    best_metrics = args.out / "best_metrics.json"
    if best_metrics.exists():
        print(format_report(json.loads(best_metrics.read_text(encoding="utf-8")),
                            f"Faster R-CNN R50-FPN v2 (best of {last_epoch} epochs)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
