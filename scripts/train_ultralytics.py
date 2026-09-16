"""Train a registry Ultralytics detector with fixed batch and resumable smoke stops."""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bddcv.paths import DATA_CONFIG, TRAIN_DIR, prepare_ultralytics, resolve_output  # noqa: E402
from bddcv.checkpointing import (  # noqa: E402
    dataset_fingerprint,
    reconcile_results_csv,
)
from bddcv.registry import get_model_spec, write_model_metadata  # noqa: E402


class ControlledStop(RuntimeError):
    """End a smoke stage after a full-state checkpoint has been published."""


RESUME_METADATA_KEYS = (
    "model_id", "dataset_fingerprint", "data_config", "batch", "epochs", "imgsz",
    "workers", "seed", "cache", "prefetch",
)
RESUME_CHECKPOINT_KEYS = ("epochs", "batch", "imgsz", "workers", "seed")


def _resolve_dataset_path(value: str | Path, config_path: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def _dataset_files(config_path: Path) -> tuple[Path, list[Path]]:
    """Resolve the config and fixed subset files used by an Ultralytics run."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - runtime dependency contract
        raise RuntimeError("PyYAML is required to fingerprint the Ultralytics dataset") from exc

    config_path = Path(config_path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not payload.get("path"):
        raise ValueError(f"dataset config has no path: {config_path}")
    subset = _resolve_dataset_path(payload["path"], config_path)
    files = [config_path]
    for split in ("train", "val"):
        configured = payload.get(split)
        candidate = _resolve_dataset_path(configured, subset) if configured else None
        manifest = candidate if candidate and candidate.is_file() else subset / f"{split}_images.txt"
        if manifest.is_file():
            files.append(manifest)
        else:
            raise FileNotFoundError(f"dataset manifest is missing: {manifest}")
        annotation = subset / "annotations" / f"instances_{split}.json"
        if not annotation.is_file():
            raise FileNotFoundError(f"dataset annotation is missing: {annotation}")
        files.append(annotation)

    unique = list(dict.fromkeys(files))
    return subset, unique


def dataset_fingerprint_from_config(config_path: str | Path) -> str:
    """Hash the resolved YAML, manifests, and COCO annotations for a run."""
    subset, files = _dataset_files(Path(config_path))
    return dataset_fingerprint(subset, files)


def _metadata_config(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = metadata.get("config")
    return nested if isinstance(nested, Mapping) else metadata


def _checkpoint_model_matches(
    value: Any,
    model_id: str,
    checkpoint: Mapping[str, Any],
) -> bool:
    if value is None:
        return False
    marker = checkpoint.get("bddcv_model_id") or checkpoint.get("model_id")
    if marker is not None and str(marker) != model_id:
        return False
    expected = Path(get_model_spec(model_id).checkpoint).name.lower()
    actual = Path(str(value)).name.lower()
    if actual == expected:
        return True
    # Ultralytics rewrites train_args.model to last.pt when a resumed trainer
    # saves its next checkpoint.  Accept that generic name only when the
    # checkpoint carries the provenance marker written by this wrapper.
    return actual == "last.pt" and str(marker) == model_id


def validate_resume_configuration(
    metadata: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    """Require metadata and native checkpoint arguments to agree before resume."""
    if not isinstance(metadata, Mapping):
        raise ValueError("resume metadata must be a JSON object")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("resume checkpoint must be a mapping")
    saved = _metadata_config(metadata)
    metadata_drift = {
        key: (saved.get(key), config.get(key))
        for key in RESUME_METADATA_KEYS
        if saved.get(key) != config.get(key)
    }
    if metadata_drift:
        raise ValueError(f"resume metadata configuration differs: {metadata_drift}")

    train_args = checkpoint.get("train_args")
    if not isinstance(train_args, Mapping):
        raise ValueError("unsafe Ultralytics resume checkpoint; missing train_args configuration")
    checkpoint_drift: dict[str, tuple[Any, Any]] = {}
    for key in RESUME_CHECKPOINT_KEYS:
        if train_args.get(key) != config.get(key):
            checkpoint_drift[key] = (train_args.get(key), config.get(key))
    configured_data = train_args.get("data")
    expected_data = config.get("data_config")
    if expected_data is not None:
        if configured_data is None:
            checkpoint_drift["data"] = (configured_data, expected_data)
        elif Path(str(configured_data)).expanduser().resolve() != Path(str(expected_data)).expanduser().resolve():
            checkpoint_drift["data"] = (configured_data, expected_data)
    if not _checkpoint_model_matches(
        train_args.get("model"), str(config["model_id"]), checkpoint
    ):
        checkpoint_drift["model"] = (train_args.get("model"), config.get("model_id"))
    checkpoint_config = checkpoint.get("bddcv_config")
    if checkpoint_config is None:
        raise ValueError("checkpoint lacks dataset/config provenance; unsafe legacy resume")
    if checkpoint_config is not None:
        if not isinstance(checkpoint_config, Mapping):
            checkpoint_drift["bddcv_config"] = (checkpoint_config, config)
        else:
            config_drift = {
                key: (checkpoint_config.get(key), config.get(key))
                for key in RESUME_METADATA_KEYS
                if checkpoint_config.get(key) != config.get(key)
            }
            if config_drift:
                checkpoint_drift["bddcv_config"] = config_drift
    if checkpoint_drift:
        raise ValueError(f"resume checkpoint configuration differs: {checkpoint_drift}")


def ensure_fresh_output(out: str | Path) -> None:
    """Refuse fresh training when the run directory contains prior artifacts."""
    root = Path(out)
    names = (
        "model.json", "run_status.json", "results.csv", "results.csv.stale",
        "args.yaml", "weights", "logs",
    )
    existing = [root / name for name in names if (root / name).exists()]
    existing.extend(root.glob("*.pt"))
    if existing:
        listed = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"existing training artifacts in fresh output: {listed}")


def reconcile_resume_progress(out: str | Path, checkpoint_completed_epochs: int) -> dict[str, Any]:
    """Trim backend CSV rows beyond the checkpoint's one-based progress."""
    if checkpoint_completed_epochs < 0:
        raise ValueError("checkpoint_completed_epochs must be non-negative")
    return reconcile_results_csv(Path(out) / "results.csv", checkpoint_completed_epochs)


def annotate_resume_checkpoint(
    path: str | Path,
    checkpoint: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist wrapper provenance without changing native checkpoint state."""
    target = Path(path)
    payload = dict(checkpoint)
    payload["bddcv_model_id"] = str(config["model_id"])
    payload["bddcv_config"] = dict(config)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        import torch

        torch.save(payload, temporary)
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return payload


def loader_configuration_callback(prefetch: int):
    """Apply selected prefetch depth before the training iterator is consumed."""
    def configure(trainer) -> None:
        from bddcv.native_probes import configure_loader_prefetch
        for name in ("train_loader", "test_loader"):
            loader = getattr(trainer, name, None)
            if loader is not None:
                configure_loader_prefetch(loader, prefetch)
    return configure


def checkpoint_provenance_callback(config: Mapping[str, Any]):
    """Publish run provenance after every native last-checkpoint save."""
    def publish(trainer) -> None:
        import torch
        path = Path(trainer.last)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        annotate_resume_checkpoint(path, checkpoint, config)
    return publish


def _csv_completed_epochs(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames or "epoch" not in reader.fieldnames:
                return 0
            epochs = []
            for row in reader:
                try:
                    epochs.append(int(row["epoch"]))
                except (KeyError, TypeError, ValueError):
                    continue
            return max(epochs, default=0)
    except OSError:
        return 0


def _checkpoint_evidence(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"valid": False, "stripped": False, "completed_epochs": 0}
    try:
        import torch

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if (not isinstance(checkpoint, Mapping)
                or "epoch" not in checkpoint
                or not isinstance(checkpoint.get("train_args"), Mapping)):
            return {"valid": False, "stripped": False, "completed_epochs": 0}
        epoch = int(checkpoint["epoch"])
        if epoch == -1:
            return {"valid": True, "stripped": True, "completed_epochs": 0}
        if epoch < 0:
            return {"valid": False, "stripped": False, "completed_epochs": 0}
        return {"valid": True, "stripped": False, "completed_epochs": epoch + 1}
    except (OSError, RuntimeError, TypeError, ValueError, EOFError,
            ImportError, pickle.UnpicklingError):
        return {"valid": False, "stripped": False, "completed_epochs": 0}


def inspect_run_progress(
    out: str | Path,
    checkpoint_path: str | Path,
    target_epochs: int,
) -> dict[str, Any]:
    """Read completed epochs from backend artifacts after a clean return."""
    if target_epochs < 1:
        raise ValueError("target_epochs must be positive")
    root = Path(out)
    checkpoint = _checkpoint_evidence(Path(checkpoint_path))
    checkpoint_progress = int(checkpoint["completed_epochs"])
    csv_progress = _csv_completed_epochs(root / "results.csv")
    if checkpoint["valid"] and checkpoint_progress:
        completed = min(checkpoint_progress, csv_progress) if csv_progress else checkpoint_progress
        source = "checkpoint"
    elif checkpoint["valid"] and checkpoint["stripped"]:
        # A normal Ultralytics completion strips optimizer state and resets the
        # checkpoint epoch.  In that case results.csv is the backend's durable
        # per-epoch record; target arguments never prove progress by themselves.
        completed = csv_progress
        source = "results.csv" if csv_progress else "none"
    else:
        completed = 0
        source = "none"
    status = "complete" if completed >= target_epochs else "incomplete"
    return {
        "status": status,
        "completed_epochs": completed,
        "target_epochs": target_epochs,
        "checkpoint_epochs": checkpoint_progress,
        "checkpoint_valid": bool(checkpoint["valid"]),
        "csv_epochs": csv_progress,
        "evidence": source,
    }


def validate_resume_checkpoint(path: Path, checkpoint: dict) -> int:
    """Validate full Ultralytics state and return one-based completed epochs."""
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"unsafe Ultralytics resume checkpoint; invalid payload: {path}")
    required = ("epoch", "optimizer", "scaler", "train_args")
    missing = [key for key in required if checkpoint.get(key) is None]
    if missing:
        raise ValueError(f"unsafe Ultralytics resume checkpoint; missing {', '.join(missing)}: {path}")
    epoch = int(checkpoint["epoch"])
    if epoch < 0:
        raise ValueError(f"unsafe stripped Ultralytics resume checkpoint: {path}")
    if not isinstance(checkpoint["train_args"], Mapping):
        raise ValueError(f"unsafe Ultralytics resume checkpoint; train_args is invalid: {path}")
    if int(checkpoint["train_args"].get("epochs", 0)) < epoch + 1:
        raise ValueError(f"invalid epoch horizon in Ultralytics checkpoint: {path}")
    return epoch + 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--epochs", type=int, default=50,
                        help="schedule horizon; smoke runs keep this at 50")
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int, default=0)
    parser.add_argument("--cache", choices=("none", "ram"), default="none")
    parser.add_argument("--prefetch", type=int, choices=(2, 4), default=2)
    args = parser.parse_args()
    spec = get_model_spec(args.model_id)
    if spec.backend != "ultralytics" or spec.model_id == "rtdetr-l":
        parser.error("this entrypoint supports unfinished YOLO registry models only")
    if args.epochs < 1 or args.batch < 1 or args.workers < 0:
        parser.error("epochs/batch must be positive and workers non-negative")
    if args.stop_after_epoch < 0 or args.stop_after_epoch > args.epochs:
        parser.error("stop-after-epoch must be within the schedule horizon")

    out = resolve_output(args.out, TRAIN_DIR / spec.output_subdir)
    metadata_path = out / "model.json"
    try:
        fingerprint = dataset_fingerprint_from_config(DATA_CONFIG)
    except (OSError, RuntimeError, ValueError, FileNotFoundError) as exc:
        parser.error(f"cannot fingerprint dataset: {exc}")
    config = {
        "model_id": spec.model_id, "batch": args.batch, "epochs": args.epochs,
        "imgsz": args.imgsz, "workers": args.workers, "seed": spec.seed,
        "cache": args.cache, "prefetch": args.prefetch,
        "dataset_fingerprint": fingerprint, "data_config": str(DATA_CONFIG.resolve()),
    }
    checkpoint_path = out / "weights" / "last.pt"

    import torch
    if args.resume:
        if not metadata_path.is_file() or not checkpoint_path.is_file():
            parser.error("resume requires model.json and weights/last.pt")
        try:
            saved = json.loads(metadata_path.read_text(encoding="utf-8"))
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            completed = validate_resume_checkpoint(checkpoint_path, checkpoint)
            validate_resume_configuration(saved, checkpoint, config)
        except (OSError, json.JSONDecodeError, TypeError, ValueError, EOFError,
                RuntimeError, pickle.UnpicklingError) as exc:
            parser.error(str(exc))
        reconciliation = reconcile_resume_progress(out, completed)
        print(f"resume reconciliation: {reconciliation}", flush=True)
    else:
        try:
            ensure_fresh_output(out)
        except FileExistsError as exc:
            parser.error(str(exc))
        out.mkdir(parents=True, exist_ok=True)
        write_model_metadata(metadata_path, spec, **config)

    prepare_ultralytics()
    from ultralytics import YOLO

    model = YOLO(str(checkpoint_path if args.resume else spec.checkpoint_path))

    def enforce_fixed_batch(trainer) -> None:
        # The pinned trainer checks this counter before attempting a smaller batch.
        trainer._oom_retries = 3

    def stop_after_checkpoint(trainer) -> None:
        completed = int(trainer.epoch) + 1
        if args.stop_after_epoch and completed >= args.stop_after_epoch:
            if not trainer.last.is_file():
                raise RuntimeError(f"controlled stop checkpoint is missing: {trainer.last}")
            raise ControlledStop(f"checkpointed epoch {completed}")

    model.add_callback("on_train_start", loader_configuration_callback(args.prefetch))
    model.add_callback("on_model_save", checkpoint_provenance_callback(config))
    model.add_callback("on_train_epoch_start", enforce_fixed_batch)
    model.add_callback("on_fit_epoch_end", stop_after_checkpoint)
    controlled = False
    try:
        if args.resume:
            model.train(resume=True)
        else:
            model.train(
                data=str(DATA_CONFIG), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, amp=True, device=0, workers=args.workers,
                project=str(out.parent), name=out.name, exist_ok=True,
                val=True, plots=True, seed=spec.seed,
                cache="ram" if args.cache == "ram" else False,
            )
    except ControlledStop as exc:
        controlled = True
        print(f"CONTROLLED_STOP: {exc}", flush=True)

    if controlled:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint = annotate_resume_checkpoint(checkpoint_path, checkpoint, config)
        completed = validate_resume_checkpoint(checkpoint_path, checkpoint)
        evidence = {
            "status": "complete" if completed >= args.epochs else "incomplete",
            "completed_epochs": completed,
            "target_epochs": args.epochs,
            "checkpoint_epochs": completed,
            "checkpoint_valid": True,
            "csv_epochs": _csv_completed_epochs(out / "results.csv"),
            "evidence": "checkpoint",
        }
    else:
        evidence = inspect_run_progress(out, checkpoint_path, args.epochs)
        completed = int(evidence["completed_epochs"])
        if evidence["status"] == "incomplete":
            print(
                f"INCOMPLETE: backend returned cleanly at {completed}/{args.epochs} "
                f"epochs ({evidence['evidence']} evidence)",
                flush=True,
            )
    status = {
        "status": evidence["status"],
        "epoch": completed, "target_epochs": args.epochs, "model_id": spec.model_id,
        "controlled_stop": controlled, "config": config, "evidence": evidence,
    }
    (out / "run_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return 0 if evidence["status"] == "complete" or controlled else 2


if __name__ == "__main__":
    raise SystemExit(main())
