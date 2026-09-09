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
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bddcv.paths import (  # noqa: E402
    PROJECT_ROOT, DATA_CONFIG, TRAIN_DIR, prepare_runtime, resolve_output, resolve_weights,
)
from bddcv.registry import (  # noqa: E402
    ModelResolutionError, get_model_spec, resolve_model_id, write_model_metadata,
)

DEFAULT_BATCH = {"ultra": 16, "frcnn": 4, "rfdetr": 1}
DEFAULT_IMGSZ = {"ultra": 640, "frcnn": 640, "rfdetr": 512}
LOG_PATH: Path | None = None

# RT-DETR needs Ultralytics' RTDETR class, not YOLO; both expose the same
# .train() interface, so only the constructor differs.  The model id is the
# authority; the filename is used only for registered pretrained names.
def ultra_class(model: str, model_id: str | None = None) -> str:
    resolved = model_id or resolve_model_id(checkpoint=model)
    return "RTDETR" if resolved == "rtdetr-l" else "YOLO"


def log(msg: str) -> None:
    line = f"[wrapper {datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    if LOG_PATH:
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def _csv_epoch(results_csv: Path) -> int:
    """Return the largest valid epoch in a backend results CSV."""
    if not results_csv.is_file():
        return 0
    highest = 0
    try:
        with results_csv.open(newline="", encoding="utf-8") as fh:
            for row in csv.reader(fh):
                try:
                    highest = max(highest, int(row[0]))
                except (IndexError, TypeError, ValueError):
                    continue
    except OSError:
        return 0
    return highest


def _checkpoint_value(checkpoint: Path) -> Mapping[str, object]:
    """Load a checkpoint mapping without constructing model tensors on a GPU."""
    import torch

    value = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint is not a mapping: {checkpoint}")
    return value


def _normalise_scalar(value):
    """Normalize Ultralytics scalar/list forms used by different releases."""
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def _checkpoint_model_matches(actual, expected) -> bool:
    if actual is None or expected is None:
        return False
    actual_name = Path(str(actual)).name.lower()
    expected_names = {Path(str(expected)).name.lower()}
    return actual_name in expected_names


def validate_ultra_checkpoint(
    checkpoint: Mapping[str, object] | Path,
    expected: Mapping[str, object],
    *,
    require_full_state: bool = True,
) -> int:
    """Validate full Ultralytics state and immutable launch arguments.

    The inline supervisor path predates ``train_ultralytics.py`` and therefore
    cannot rely on that entrypoint's checks.  Keep the preflight here so a
    restart never resumes a checkpoint produced with another model, geometry,
    batch, seed, or epoch horizon.
    """
    value = _checkpoint_value(checkpoint) if isinstance(checkpoint, Path) else checkpoint
    if not isinstance(value, Mapping):
        raise ValueError("Ultralytics checkpoint is not a mapping")
    if require_full_state:
        # Reuse the entrypoint's state-shape validator; the supervisor adds
        # the launch-argument checks below because its inline RT-DETR/YOLO
        # path does not call that entrypoint.
        try:
            from scripts.train_ultralytics import validate_resume_checkpoint
            validate_resume_checkpoint(
                checkpoint if isinstance(checkpoint, Path) else Path("<inline>"),
                dict(value),
            )
        except ImportError as exc:
            raise ValueError("Ultralytics resume validator is unavailable") from exc
    epoch = value.get("epoch")
    try:
        completed_epoch = int(epoch)
    except (TypeError, ValueError) as exc:
        raise ValueError("unsafe Ultralytics checkpoint; missing epoch") from exc
    if require_full_state and completed_epoch < 0:
        raise ValueError("unsafe stripped Ultralytics checkpoint cannot be resumed")
    train_args = value.get("train_args")
    if not isinstance(train_args, Mapping):
        raise ValueError("unsafe Ultralytics resume checkpoint; train_args is invalid")

    drift: dict[str, tuple[object, object]] = {}
    for key in ("epochs", "imgsz", "batch", "seed"):
        saved = _normalise_scalar(train_args.get(key))
        current = _normalise_scalar(expected.get(key))
        if saved != current:
            drift[key] = (saved, current)

    model_id = expected.get("model_id")
    model_name = expected.get("model")
    if model_name is None and model_id is not None:
        model_name = get_model_spec(str(model_id)).checkpoint
    marked_resume = (Path(str(train_args.get("model"))).name == "last.pt"
                     and value.get("bddcv_model_id") == model_id)
    if not marked_resume and not _checkpoint_model_matches(train_args.get("model"), model_name):
        drift["model"] = (train_args.get("model"), model_name)
    if drift:
        raise ValueError(f"Ultralytics checkpoint configuration differs: {drift}")
    if require_full_state:
        from scripts.train_ultralytics import validate_resume_configuration
        validate_resume_configuration(expected, value, expected)
    return max(0, completed_epoch + 1)


def _validate_ultra_artifacts(
    checkpoint: Path,
    results_csv: Path,
    expected: Mapping[str, object],
    target_epochs: int,
) -> None:
    """Reject unsafe existing state while allowing a verified clean completion."""
    if not checkpoint.exists():
        return
    value = _checkpoint_value(checkpoint)
    try:
        epoch = int(value.get("epoch", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid Ultralytics checkpoint epoch: {checkpoint}") from exc
    if epoch < 0 or value.get("optimizer") is None or value.get("scaler") is None:
        # A successful Ultralytics run strips optimizer state.  It is usable as
        # completion evidence only when the append-only CSV also reaches the
        # requested horizon; it must never be treated as resumable state.
        if _csv_epoch(results_csv) >= target_epochs:
            validate_ultra_checkpoint(value, expected, require_full_state=False)
            return
        raise ValueError(
            "existing Ultralytics checkpoint has no full resume state and its "
            f"CSV reaches only {_csv_epoch(results_csv)}/{target_epochs} epochs"
        )
    validate_ultra_checkpoint(value, expected)


def _ultra_dataset_fingerprint() -> str:
    """Hash the YAML, fixed manifests, and annotations used by Ultralytics."""
    from scripts.train_ultralytics import dataset_fingerprint_from_config

    return dataset_fingerprint_from_config(DATA_CONFIG)


def checkpoint_epoch(checkpoint: Path, mode: str) -> int | None:
    """Read authoritative progress from a checkpoint, if it is readable."""
    if not checkpoint.exists():
        return None
    try:
        value = _checkpoint_value(checkpoint)
        if mode == "rfdetr":
            from scripts.train_rfdetr import validate_full_checkpoint
            return validate_full_checkpoint(checkpoint, value)
        if mode == "frcnn":
            from scripts.train_frcnn import RESUME_STATE_KEYS
            from bddcv.checkpointing import require_resume_state
            require_resume_state(value, RESUME_STATE_KEYS)
        epoch = int(value.get("epoch", -1))
        # Faster R-CNN stores one-based epochs; Ultralytics stores zero-based.
        # A completed Ultralytics checkpoint may deliberately strip state and
        # write epoch=-1; use CSV only as a compatibility fallback then.
        if epoch < 0:
            return None
        if mode == "ultra" and (
            value.get("optimizer") is None or value.get("scaler") is None
        ):
            return None
        if mode == "ultra":
            train_args = value.get("train_args")
            if not isinstance(train_args, Mapping) or any(
                train_args.get(key) is None for key in ("model", "imgsz", "batch", "seed")
            ):
                return None
        return epoch if mode == "frcnn" else epoch + 1
    except (OSError, RuntimeError, ValueError, TypeError, KeyError, EOFError):
        return None


def epochs_done(results_csv: Path, checkpoint: Path | None = None,
                mode: str = "frcnn") -> int:
    """Return checkpoint-authoritative progress; CSV is diagnostics only."""
    if checkpoint is not None:
        authoritative = checkpoint_epoch(checkpoint, mode)
        if authoritative is not None and authoritative >= 0:
            return authoritative
        if mode == "ultra" and checkpoint.exists():
            # The completed Ultralytics checkpoint has intentionally stripped
            # optimizer state.  CSV rows are the only remaining progress
            # evidence, and are accepted only for this known terminal form.
            try:
                value = _checkpoint_value(checkpoint)
                if int(value.get("epoch", -1)) < 0 and value.get("optimizer") is None:
                    return _csv_epoch(results_csv)
            except (OSError, RuntimeError, ValueError, TypeError, KeyError, EOFError):
                pass
    return 0


def run_checkpoint_path(run_dir: Path, mode: str) -> Path:
    """Return the full-state checkpoint used as progress authority."""
    if mode == "ultra":
        return run_dir / "weights" / "last.pt"
    if mode == "rfdetr":
        return run_dir / "last.ckpt"
    return run_dir / "last.pt"


def write_or_validate_model_metadata(path: Path, spec, **config) -> bool:
    """Create metadata once; validate immutable launch settings thereafter."""
    if path.exists():
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid existing model metadata: {path}") from exc
        def saved_value(key):
            if key == "batch" and "batch" not in saved:
                return saved.get("physical_batch")
            if key == "imgsz" and "imgsz" not in saved:
                resolution = saved.get("resolution")
                return resolution.get("imgsz") if isinstance(resolution, dict) else resolution
            if key == "validation_batch" and "validation_batch" not in saved:
                return saved.get("eval_batch_size")
            return saved.get(key)

        expected = {"model_id": spec.model_id, **config}
        drift = {key: (saved_value(key), value) for key, value in expected.items()
                 if saved_value(key) != value}
        if drift:
            raise ValueError(f"existing model metadata differs from launch: {drift}")
        return False
    write_model_metadata(path, spec, **config)
    return True


def build_command(args, resuming: bool) -> list[str]:
    model_id = getattr(args, "model_id", None)
    if args.mode == "ultra":
        spec = get_model_spec(model_id or resolve_model_id(checkpoint=args.model))
        cls = ultra_class(args.model, spec.model_id)
        model = args.run_dir / "weights/last.pt" if resuming else resolve_weights(args.model)
        if not resuming and model_id and args.model == "yolo11s.pt":
            model = resolve_weights(spec.checkpoint)
        kwargs = ({"resume": True, "save_dir": str(args.run_dir)} if resuming else {
            "data": str(DATA_CONFIG), "epochs": args.epochs, "imgsz": args.imgsz,
            "batch": args.batch, "amp": True, "device": 0, "workers": args.workers,
            "project": str(args.run_dir.parent), "name": args.run_dir.name,
            "exist_ok": True, "val": True, "plots": True, "seed": 0,
            "cache": "ram" if getattr(args, "cache", "none") == "ram" else False,
        })
        provenance = {
            "model_id": spec.model_id, "batch": args.batch, "epochs": args.epochs,
            "imgsz": args.imgsz, "workers": args.workers, "seed": 0,
            "cache": getattr(args, "cache", "none"), "prefetch": getattr(args, "prefetch", 2),
            "data_config": str(DATA_CONFIG.resolve()),
            "dataset_fingerprint": _ultra_dataset_fingerprint(),
        }
        # repr handles spaces, quotes and Windows separators without code injection.
        code = (
            f"import sys; sys.path.insert(0, {str(PROJECT_ROOT / 'src')!r})\n"
            "from bddcv.paths import prepare_ultralytics\nprepare_ultralytics()\n"
            f"from ultralytics import {cls}\n"
            f"detector = {cls}({str(model)!r})\n"
            "def enforce_fixed_batch(trainer):\n    trainer._oom_retries = 3\n"
            "detector.add_callback('on_train_epoch_start', enforce_fixed_batch)\n"
            f"sys.path.insert(0, {str(PROJECT_ROOT)!r})\n"
            "from scripts.train_ultralytics import checkpoint_provenance_callback, loader_configuration_callback\n"
            f"detector.add_callback('on_train_start', loader_configuration_callback({getattr(args, 'prefetch', 2)!r}))\n"
            f"detector.add_callback('on_model_save', checkpoint_provenance_callback({provenance!r}))\n"
            f"detector.train(**{kwargs!r})\n"
        )
        return [sys.executable, "-u", "-c", code]

    if args.mode == "rfdetr":
        # Deliberately point at the dedicated interpreter.  Building a command
        # remains CPU-testable even when that environment has not been created.
        interpreter = PROJECT_ROOT / ".venv-rfdetr" / (
            "Scripts/python.exe" if os.name == "nt" else "bin/python"
        )
        command = [
            str(interpreter), "-u", str(PROJECT_ROOT / "scripts" / "train_rfdetr.py"),
            "--model-id", model_id or "rfdetr-small", "--epochs", str(args.epochs),
            "--batch", str(args.batch), "--workers", str(args.workers),
            "--accumulate", str(args.accumulate),
            "--cache", getattr(args, "cache", "none"),
            "--prefetch", str(getattr(args, "prefetch", 2)),
            "--out", str(args.run_dir),
        ]
        if resuming:
            command.append("--resume")
        return command

    cmd = [
        sys.executable, "-u", str(PROJECT_ROOT / "scripts" / "train_frcnn.py"),
        "--epochs", str(args.epochs), "--batch", str(args.batch),
        "--validation-batch", str(getattr(args, "validation_batch", 4)),
        "--accumulate", str(args.accumulate),
        "--cache", getattr(args, "cache", "none"),
        "--prefetch", str(getattr(args, "prefetch", 2)),
        "--workers", str(args.workers), "--imgsz", str(args.imgsz),
        "--seed", "0", "--out", str(args.out),
        "--model-id", model_id or "frcnn-r50-fpn-v2",
    ]
    if resuming:
        cmd.append("--resume")
    return cmd


def main() -> int:
    global LOG_PATH
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["ultra", "yolo", "frcnn", "rfdetr"],
                   help="'ultra' drives YOLO/RT-DETR; 'yolo' is a backwards-compatible alias")
    p.add_argument("--model", default="yolo11s.pt",
                   help="ultra only: yolo11n.pt, yolo11s.pt, yolo11m.pt, rtdetr-l.pt ...")
    p.add_argument("--model-id", default=None,
                   help="registry id; required for generic checkpoints such as best.pt")
    p.add_argument("--name", default=None,
                   help="ultra only: run name; defaults to the model stem")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=None,
                   help="defaults to 16 for ultra, 4 for frcnn")
    p.add_argument("--imgsz", type=int, default=None,
                   help="input size; defaults to 640, or RF-DETR native 512")
    p.add_argument("--validation-batch", type=int, default=None,
                   help="Faster R-CNN/RF-DETR validation batch (registry default)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--accumulate", type=int, default=1)
    p.add_argument("--out", type=Path, default=None,
                   help="exact run directory; default: runs/train/<name> (frcnn: runs/train/frcnn)")
    p.add_argument("--max-attempts", type=int, default=20)
    p.add_argument("--max-stalls", type=int, default=3,
                   help="consecutive restarts with no epoch completed before giving up")
    p.add_argument("--backoff", type=int, default=45,
                   help="seconds to let the GPU settle after a fault")
    p.add_argument("--cache", choices=("none", "ram"), default="none")
    p.add_argument("--prefetch", type=int, choices=(2, 4), default=2)
    args = p.parse_args()
    if args.accumulate < 1:
        p.error("--accumulate must be positive")

    if args.mode == "yolo":
        args.mode = "ultra"
    if args.mode == "rfdetr":
        args.model_id = args.model_id or "rfdetr-small"
        if args.model_id != "rfdetr-small":
            p.error("rfdetr mode supports only --model-id rfdetr-small")
    elif args.mode == "ultra":
        try:
            args.model_id = resolve_model_id(args.model_id, args.model)
        except ModelResolutionError as exc:
            p.error(str(exc))
        if args.model_id and args.model == "yolo11s.pt":
            args.model = get_model_spec(args.model_id).checkpoint
    elif args.mode == "frcnn":
        args.model_id = args.model_id or "frcnn-r50-fpn-v2"
        if args.model_id != "frcnn-r50-fpn-v2":
            p.error("frcnn mode supports only --model-id frcnn-r50-fpn-v2")
    if args.imgsz is None:
        args.imgsz = DEFAULT_IMGSZ[args.mode]
    if args.mode == "rfdetr" and args.imgsz != DEFAULT_IMGSZ[args.mode]:
        p.error("RF-DETR Small uses its native imgsz=512")
    if args.validation_batch is None:
        args.validation_batch = get_model_spec(args.model_id).validation_batch or 4
    if args.validation_batch < 1:
        p.error("--validation-batch must be positive")
    if args.batch is None:
        args.batch = 8 if args.model_id == "rtdetr-l" else DEFAULT_BATCH.get(args.mode, 1)

    if args.mode in {"ultra", "rfdetr"}:
        args.name = args.name or (args.model_id if args.mode == "rfdetr" else Path(args.model).stem)
        if Path(args.name).name != args.name or args.name in (".", ".."):
            p.error("--name must be a single directory name; use --out for a path")
        args.run_dir = resolve_output(args.out, TRAIN_DIR / args.name)
        results_csv = args.run_dir / "results.csv"
        checkpoint = run_checkpoint_path(args.run_dir, args.mode)
        label = f"{args.model_id} ({args.model}) -> {args.run_dir}"
    else:
        args.out = resolve_output(args.out, TRAIN_DIR / "frcnn")
        args.run_dir = args.out
        results_csv = args.out / "results.csv"
        checkpoint = run_checkpoint_path(args.out, args.mode)
        label = f"{args.model_id} -> {args.out}"

    logs = args.run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    LOG_PATH = logs / "supervisor.log"
    (logs / "supervisor.pid").write_text(str(os.getpid()) + "\n")
    (logs / "supervisor_launch.json").write_text(json.dumps({
        "pid": os.getpid(), "started_at": datetime.now().astimezone().isoformat(),
        "argv": sys.argv, "run_dir": str(args.run_dir),
        "model_id": args.model_id, "backend": "rfdetr" if args.mode == "rfdetr" else args.mode,
    }, indent=2), encoding="utf-8")
    metadata_config = {
        "batch": args.batch, "epochs": args.epochs, "imgsz": args.imgsz,
        "workers": args.workers, "seed": 0,
        "cache": args.cache, "prefetch": args.prefetch,
    }
    if args.mode in {"frcnn", "rfdetr"}:
        metadata_config["validation_batch"] = args.validation_batch
    ultra_expected = None
    if args.mode == "ultra":
        try:
            metadata_config["dataset_fingerprint"] = _ultra_dataset_fingerprint()
            metadata_config["data_config"] = str(DATA_CONFIG.resolve())
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            p.error(f"cannot fingerprint Ultralytics dataset: {exc}")
        ultra_expected = {
            **metadata_config,
            "model_id": args.model_id,
            # Compare the checkpoint against the actual requested source
            # weight.  Explicit model ids may legitimately use a custom
            # pretrained filename; the registry id still selects the backend.
            "model": Path(args.model).name,
        }
    try:
        write_or_validate_model_metadata(
            args.run_dir / "model.json", get_model_spec(args.model_id),
            **metadata_config,
        )
    except ValueError as exc:
        p.error(str(exc))
    log(f"supervising {label} (batch {args.batch}, {args.epochs} epochs)")
    if ultra_expected is not None:
        try:
            _validate_ultra_artifacts(
                checkpoint, results_csv, ultra_expected, args.epochs
            )
        except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
            log(f"ABORT: unsafe existing Ultralytics state: {exc}")
            return 1
    if args.mode == "rfdetr":
        interpreter = PROJECT_ROOT / ".venv-rfdetr" / (
            "Scripts/python.exe" if os.name == "nt" else "bin/python"
        )
        if not interpreter.is_file():
            reason = f"RF-DETR isolated interpreter is unavailable: {interpreter}"
            (args.run_dir / "run_status.json").write_text(json.dumps({
                "status": "unavailable", "model_id": args.model_id, "reason": reason,
            }, indent=2) + "\n", encoding="utf-8")
            log(f"BLOCKED: {reason}")
            return 2

    started_at = time.time()
    stalls = 0

    for attempt in range(1, args.max_attempts + 1):
        before = epochs_done(results_csv, checkpoint, args.mode)
        if before >= args.epochs:
            log(f"target reached: {before}/{args.epochs} epochs complete")
            return 0

        resuming = checkpoint.exists() and before > 0
        if ultra_expected is not None and resuming:
            try:
                _validate_ultra_artifacts(
                    checkpoint, results_csv, ultra_expected, args.epochs
                )
            except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
                log(f"ABORT: unsafe Ultralytics resume state: {exc}")
                return 1
            # A CSV row written after a checkpoint is stale evidence.  Keep it
            # as a sidecar before the backend appends fresh progress.
            try:
                from bddcv.checkpointing import reconcile_results_csv
                reconciliation = reconcile_results_csv(results_csv, before)
                if reconciliation["stale_rows"]:
                    log(f"reconciled CSV to checkpoint: {reconciliation}")
            except (OSError, ValueError) as exc:
                log(f"ABORT: could not reconcile Ultralytics CSV: {exc}")
                return 1
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
        if ultra_expected is not None:
            try:
                _validate_ultra_artifacts(
                    checkpoint, results_csv, ultra_expected, args.epochs
                )
            except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
                log(f"ABORT: child left unsafe Ultralytics state: {exc}")
                return 1
        after = epochs_done(results_csv, checkpoint, args.mode)
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
            log(f"INCOMPLETE: child exited cleanly at {after}/{args.epochs} epochs - not a fault. Stopping.")
            return 2

        stalls = stalls + 1 if gained == 0 else 0
        if stalls >= args.max_stalls:
            log(f"ABORT: {stalls} consecutive restarts completed no epoch. "
                f"This is a real failure, not a transient driver fault.")
            return 1

        log(f"waiting {args.backoff}s before restart "
            f"(consecutive stalls: {stalls}/{args.max_stalls})")
        time.sleep(args.backoff)

    log(f"ABORT: exhausted {args.max_attempts} attempts at "
        f"{epochs_done(results_csv, checkpoint, args.mode)}/{args.epochs} epochs")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
