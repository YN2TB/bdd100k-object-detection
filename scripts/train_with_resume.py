r"""Supervise a long training run across GPU driver faults.

The 5060 machine's nvlddmkm driver intermittently faults under sustained CUDA
load (two faults in 4.5h on 2026-09-03/04). The fault kills every CUDA context,
so training dies with no traceback - killed, not crashed. Checkpoints survive;
only the restart is missing.

This wrapper launches training as a child process and relaunches it with resume
whenever it exits before finishing.

Stall guard: a restart that completes no further epoch is treated as a real
failure rather than a transient fault. Without this the wrapper would reproduce
the pathological pattern in this machine's event log on 2026-08-15, where a job
faulted and auto-restarted every 5m05s for over an hour without progressing.

Run it detached, not as a child of an agent session - a session teardown
otherwise takes the supervisor down with the run it is supervising:

    Start-Process python -ArgumentList "-u","scripts/train_with_resume.py",... `
        -WorkingDirectory D:\CV -RedirectStandardOutput runs/logs/<name>.log `
        -WindowStyle Hidden
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Ultralytics nests runs under <runs_dir>/detect/<project>/<name>.
ULTRA_RUNS_ROOT = PROJECT_ROOT / "runs" / "detect" / "runs"

DEFAULT_BATCH = {"ultra": 16, "frcnn": 4}

# RT-DETR needs Ultralytics' RTDETR class, not YOLO; both expose the same
# .train() interface, so only the constructor differs.
ULTRA_FRESH = """
from ultralytics import {cls}
{cls}('{model}').train(
    data='configs/bdd_source.yaml', epochs={epochs}, imgsz={imgsz}, batch={batch},
    amp=True, device=0, workers={workers}, project='runs', name='{name}',
    exist_ok=True, val=True, plots=True, seed=0,
)
"""

ULTRA_RESUME = """
from ultralytics import {cls}
{cls}('{last}').train(resume=True)
"""


def ultra_class(model: str) -> str:
    return "RTDETR" if "rtdetr" in Path(model).stem.lower() else "YOLO"


def log(msg: str) -> None:
    print(f"[wrapper {datetime.now():%H:%M:%S}] {msg}", flush=True)


def epochs_done(results_csv: Path) -> int:
    """Completed epochs, read from the trainer's own results.csv."""
    if not results_csv.exists():
        return 0
    try:
        with open(results_csv, newline="", encoding="utf-8") as fh:
            return max(0, sum(1 for _ in csv.reader(fh)) - 1)
    except OSError:
        return 0


def build_command(args, resuming: bool) -> list[str]:
    if args.mode == "ultra":
        cls = ultra_class(args.model)
        code = (
            ULTRA_RESUME.format(
                cls=cls, last=(args.run_dir / "weights" / "last.pt").as_posix()
            )
            if resuming
            else ULTRA_FRESH.format(
                cls=cls, model=args.model, epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, workers=args.workers, name=args.name,
            )
        )
        return [sys.executable, "-u", "-c", code]

    cmd = [
        sys.executable, "-u", str(PROJECT_ROOT / "scripts" / "train_frcnn.py"),
        "--epochs", str(args.epochs), "--batch", str(args.batch),
        "--workers", str(args.workers), "--out", str(args.out),
    ]
    if resuming:
        cmd.append("--resume")
    return cmd


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["ultra", "yolo", "frcnn"],
                   help="'ultra' drives any Ultralytics model (YOLO, RT-DETR); "
                        "'yolo' is a backwards-compatible alias")
    p.add_argument("--model", default="yolo11s.pt",
                   help="ultra only: yolo11n.pt, yolo11s.pt, yolo11m.pt, rtdetr-l.pt ...")
    p.add_argument("--name", default=None,
                   help="ultra only: run name; defaults to the model stem")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=None,
                   help="defaults to 16 for ultra, 4 for frcnn")
    p.add_argument("--imgsz", type=int, default=640, help="ultra only")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("runs/frcnn"), help="frcnn only")
    p.add_argument("--max-attempts", type=int, default=20)
    p.add_argument("--max-stalls", type=int, default=3,
                   help="consecutive restarts with no epoch completed before giving up")
    p.add_argument("--backoff", type=int, default=45,
                   help="seconds to let the GPU settle after a fault")
    args = p.parse_args()

    if args.mode == "yolo":
        args.mode = "ultra"
    if args.batch is None:
        args.batch = DEFAULT_BATCH[args.mode]

    if args.mode == "ultra":
        args.name = args.name or Path(args.model).stem
        args.run_dir = ULTRA_RUNS_ROOT / args.name
        results_csv = args.run_dir / "results.csv"
        checkpoint = args.run_dir / "weights" / "last.pt"
        label = f"{args.model} -> runs/detect/runs/{args.name}"
    else:
        results_csv = args.out / "results.csv"
        checkpoint = args.out / "last.pt"
        label = f"faster-rcnn -> {args.out}"

    log(f"supervising {label} (batch {args.batch}, {args.epochs} epochs)")

    started_at = time.time()
    stalls = 0

    for attempt in range(1, args.max_attempts + 1):
        before = epochs_done(results_csv)
        if before >= args.epochs:
            log(f"target reached: {before}/{args.epochs} epochs complete")
            return 0

        resuming = checkpoint.exists() and before > 0
        verb = f"resuming from epoch {before + 1}" if resuming else "starting fresh"
        log(f"attempt {attempt}/{args.max_attempts}: {verb} ({before}/{args.epochs} done)")

        rc = subprocess.call(build_command(args, resuming), cwd=PROJECT_ROOT)
        after = epochs_done(results_csv)
        gained = after - before
        elapsed = (time.time() - started_at) / 3600

        log(f"child exited rc={rc}; epochs {before} -> {after} (+{gained}); "
            f"{elapsed:.2f}h elapsed")

        if after >= args.epochs:
            log(f"COMPLETE: {after} epochs in {elapsed:.2f}h across {attempt} attempt(s)")
            return 0

        if rc == 0:
            # A clean exit short of target means the trainer chose to stop
            # (early stopping, or a target already satisfied). Restarting would
            # loop, so surface it instead.
            log(f"child exited cleanly at {after}/{args.epochs} epochs - not a fault. Stopping.")
            return 0

        stalls = stalls + 1 if gained == 0 else 0
        if stalls >= args.max_stalls:
            log(f"ABORT: {stalls} consecutive restarts completed no epoch. "
                f"This is a real failure, not a transient driver fault.")
            return 1

        log(f"waiting {args.backoff}s before restart "
            f"(consecutive stalls: {stalls}/{args.max_stalls})")
        time.sleep(args.backoff)

    log(f"ABORT: exhausted {args.max_attempts} attempts at "
        f"{epochs_done(results_csv)}/{args.epochs} epochs")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
