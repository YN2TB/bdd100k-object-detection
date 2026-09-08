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
        -WorkingDirectory D:\CV -RedirectStandardOutput runs/train/<name>/logs/launcher.log `
        -WindowStyle Hidden
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bddcv.paths import (  # noqa: E402
    PROJECT_ROOT, DATA_CONFIG, TRAIN_DIR, prepare_runtime, resolve_output, resolve_weights,
)

DEFAULT_BATCH = {"ultra": 16, "frcnn": 4}
LOG_PATH: Path | None = None

# RT-DETR needs Ultralytics' RTDETR class, not YOLO; both expose the same
# .train() interface, so only the constructor differs.
def ultra_class(model: str) -> str:
    return "RTDETR" if "rtdetr" in Path(model).stem.lower() else "YOLO"


def log(msg: str) -> None:
    line = f"[wrapper {datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    if LOG_PATH:
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


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
        model = args.run_dir / "weights/last.pt" if resuming else resolve_weights(args.model)
        kwargs = ({"resume": True, "save_dir": str(args.run_dir)} if resuming else {
            "data": str(DATA_CONFIG), "epochs": args.epochs, "imgsz": args.imgsz,
            "batch": args.batch, "amp": True, "device": 0, "workers": args.workers,
            "project": str(args.run_dir.parent), "name": args.run_dir.name,
            "exist_ok": True, "val": True, "plots": True, "seed": 0,
        })
        # repr handles spaces, quotes and Windows separators without code injection.
        code = (
            f"import sys; sys.path.insert(0, {str(PROJECT_ROOT / 'src')!r})\n"
            "from bddcv.paths import prepare_ultralytics\nprepare_ultralytics()\n"
            f"from ultralytics import {cls}\n{cls}({str(model)!r}).train(**{kwargs!r})\n"
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
    global LOG_PATH
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
    p.add_argument("--out", type=Path, default=None,
                   help="exact run directory; default: runs/train/<name> (frcnn: runs/train/frcnn)")
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
        if Path(args.name).name != args.name or args.name in (".", ".."):
            p.error("--name must be a single directory name; use --out for a path")
        args.run_dir = resolve_output(args.out, TRAIN_DIR / args.name)
        results_csv = args.run_dir / "results.csv"
        checkpoint = args.run_dir / "weights" / "last.pt"
        label = f"{args.model} -> {args.run_dir}"
    else:
        args.out = resolve_output(args.out, TRAIN_DIR / "frcnn")
        args.run_dir = args.out
        results_csv = args.out / "results.csv"
        checkpoint = args.out / "last.pt"
        label = f"faster-rcnn -> {args.out}"

    logs = args.run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    LOG_PATH = logs / "supervisor.log"
    (logs / "supervisor.pid").write_text(str(os.getpid()) + "\n")
    (logs / "supervisor_launch.json").write_text(json.dumps({
        "pid": os.getpid(), "started_at": datetime.now().astimezone().isoformat(),
        "argv": sys.argv, "run_dir": str(args.run_dir),
    }, indent=2), encoding="utf-8")
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

        prepare_runtime()
        args.run_dir.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as output:
            child = subprocess.Popen(build_command(args, resuming), cwd=PROJECT_ROOT,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding="utf-8", errors="replace")
            try:
                for line in child.stdout:
                    print(line, end="", flush=True)
                    output.write(line)
                    output.flush()
                rc = child.wait()
            finally:
                if child.poll() is None:
                    child.terminate()
                    child.wait()
                child.stdout.close()
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
