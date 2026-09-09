"""Train RF-DETR Small from its dedicated environment.

The supervisor invokes this script with ``.venv-rfdetr/bin/python``.  It is
kept separate from the main Ultralytics/Torchvision entrypoints so importing
RF-DETR can never mutate the constrained main environment.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bddcv.paths import DATA_DIR, prepare_rfdetr_runtime, resolve_output  # noqa: E402
from bddcv.checkpointing import dataset_fingerprint, resume_mismatches  # noqa: E402
from bddcv.registry import get_model_spec, write_model_metadata  # noqa: E402
from bddcv.rfdetr import (  # noqa: E402
    DEFAULT_ADAPTER,
    RFDETRAdapterError,
    RFDETRBackendUnavailable,
    RFDETRAdapter,
    build_adapter,
    isolated_python,
)

LIGHTWEIGHT_CHECKPOINTS = {
    "checkpoint_best_regular.pth", "checkpoint_best_ema.pth",
    "checkpoint_best_total.pth", "last_ema.pth",
}
FULL_CHECKPOINT_KEYS = {
    "epoch", "global_step", "state_dict", "optimizer_states",
    "lr_schedulers", "loops",
}


def build_train_kwargs(*, adapter: Path, out: Path, epochs: int, batch: int,
                       workers: int, accumulation: int,
                       resume: Path | None) -> dict[str, object]:
    """Build the fixed RF-DETR Small training recipe."""
    values: dict[str, object] = {
        "dataset_dir": str(adapter), "output_dir": str(out),
        "epochs": epochs, "batch_size": batch,
        "grad_accum_steps": accumulation, "eval_batch_size": 1,
        "num_workers": workers, "resolution": 512,
        # RF-DETR's pinned backend defaults to 42 when this is omitted.  The
        # comparison contract is seed 0, so pass it explicitly to the native
        # trainer rather than relying on a process-level seed.
        "seed": 0,
        # RF-DETR 1.10.1 suppresses its latest-checkpoint callback when the
        # archive interval is 1.  An interval of 2 keeps an atomic last.ckpt
        # every epoch for stop/resume while retaining periodic archives.
        "checkpoint_interval": 2, "run_test": False,
        "early_stopping": False, "class_names": list(__import__(
            "bddcv.constants", fromlist=["DET_CLASSES"]
        ).DET_CLASSES),
        # RF-DETR 1.10.1 mishandles the explicit [0] device list internally;
        # on this single-GPU assignment, "cuda" resolves to the same GPU 0.
        "device": "cuda",
    }
    if resume is not None:
        values["resume"] = str(resume)
    return values


def validate_full_checkpoint(path: Path, checkpoint: dict) -> int:
    """Reject lightweight or incomplete state before a resume attempt."""
    if path.name in LIGHTWEIGHT_CHECKPOINTS or path.suffix != ".ckpt":
        raise RFDETRAdapterError(f"unsafe RF-DETR resume checkpoint: {path}")
    missing = sorted(key for key in FULL_CHECKPOINT_KEYS if checkpoint.get(key) is None)
    if missing:
        raise RFDETRAdapterError(
            "unsafe RF-DETR resume checkpoint; missing " + ", ".join(missing)
        )
    if not checkpoint["optimizer_states"] or not checkpoint["lr_schedulers"]:
        raise RFDETRAdapterError("RF-DETR resume checkpoint has no optimizer/scheduler state")
    return int(checkpoint["epoch"]) + 1


def _stop_after_checkpoint(path: Path, target_epoch: int, done: threading.Event) -> None:
    """Interrupt after Lightning has atomically published the requested epoch."""
    import torch
    while not done.wait(1):
        if not path.exists():
            continue
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            completed = validate_full_checkpoint(path, checkpoint)
        except (OSError, RuntimeError, EOFError, RFDETRAdapterError):
            continue
        if completed >= target_epoch:
            os.kill(os.getpid(), signal.SIGINT)
            return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="rfdetr-small")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int, default=0)
    parser.add_argument("--cache", choices=("none", "ram"), default="none")
    parser.add_argument("--prefetch", type=int, choices=(2, 4), default=2)
    args = parser.parse_args()
    if args.model_id != "rfdetr-small":
        parser.error("only rfdetr-small is in the approved registry")
    out = resolve_output(args.out, args.out)
    out.mkdir(parents=True, exist_ok=True)
    source = DATA_DIR / "source_daytime_clear"
    fingerprint = dataset_fingerprint(source)
    config = {
        "model_id": args.model_id, "dataset_fingerprint": fingerprint,
        "cache": args.cache, "prefetch": args.prefetch,
        # Match the supervisor's top-level launch fields so a child rewrite of
        # model.json remains compatible with a later supervisor restart.
        "resolution": 512, "imgsz": 512, "batch": args.batch,
        "epochs": args.epochs, "seed": 0,
        "physical_batch": args.batch,
        "effective_batch": args.batch * args.accumulate,
        "optimizer": "RF-DETR native", "schedule": "RF-DETR native",
        "workers": args.workers, "validation_batch": 1,
    }
    metadata_path = out / "model.json"
    resume_path = out / "last.ckpt" if args.resume else None
    try:
        prepare_rfdetr_runtime()
        build_adapter(source, args.adapter)
        if args.resume:
            if not resume_path or not resume_path.is_file() or not metadata_path.is_file():
                raise RFDETRAdapterError("RF-DETR resume requires model.json and last.ckpt")
            import torch
            saved = json.loads(metadata_path.read_text(encoding="utf-8"))
            drift = resume_mismatches(saved, config, config)
            if drift:
                raise RFDETRAdapterError(f"RF-DETR resume configuration differs: {drift}")
            validate_full_checkpoint(
                resume_path, torch.load(resume_path, map_location="cpu", weights_only=False)
            )
        else:
            write_model_metadata(metadata_path, get_model_spec(args.model_id), **config)
        backend = RFDETRAdapter(model_id=args.model_id, python=sys.executable)
        model = backend.construct()
        kwargs = build_train_kwargs(
            adapter=args.adapter.resolve(), out=out, epochs=args.epochs,
            batch=args.batch, workers=args.workers, accumulation=args.accumulate,
            resume=resume_path,
        )
        kwargs["prefetch_factor"] = args.prefetch
        stop = threading.Event()
        watcher = None
        if args.stop_after_epoch:
            watcher = threading.Thread(
                target=_stop_after_checkpoint,
                args=(out / "last.ckpt", args.stop_after_epoch, stop), daemon=True,
            )
            watcher.start()
        interrupted = False
        import rfdetr.training as training
        original_datamodule = training.RFDETRDataModule
        if args.cache == "ram":
            from bddcv.cache import cached_rfdetr_datamodule
            training.RFDETRDataModule = cached_rfdetr_datamodule(original_datamodule)
        try:
            model.train(**kwargs)
        except KeyboardInterrupt:
            interrupted = True
        finally:
            training.RFDETRDataModule = original_datamodule
            stop.set()
            if watcher:
                watcher.join(timeout=2)
    except (RFDETRAdapterError, RFDETRBackendUnavailable, ImportError, OSError,
            RuntimeError, ValueError) as exc:
        status = {"status": "blocked", "model_id": args.model_id, "reason": str(exc)}
        (out / "run_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        print(f"RF-DETR training blocked: {exc}", file=sys.stderr)
        return 2
    completed_epoch = 0
    if (out / "last.ckpt").is_file():
        import torch
        completed_epoch = validate_full_checkpoint(
            out / "last.ckpt",
            torch.load(out / "last.ckpt", map_location="cpu", weights_only=False),
        )
    complete = completed_epoch >= args.epochs
    status = {
        "status": "complete" if complete else "incomplete",
        "model_id": args.model_id, "epoch": completed_epoch,
        "target_epochs": args.epochs, "config": config,
    }
    (out / "run_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return 0 if complete or interrupted else 2


if __name__ == "__main__":
    raise SystemExit(main())
