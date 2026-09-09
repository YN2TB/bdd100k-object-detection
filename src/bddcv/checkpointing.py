"""Reusable checkpoint, reproducibility and progress helpers."""
from __future__ import annotations

import csv
import hashlib
import os
import random
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch


class UnsafeResumeError(RuntimeError):
    """Raised when a checkpoint lacks state needed for a faithful resume."""


def dataset_fingerprint(
    subset_root: str | Path,
    files: Iterable[str | Path] | None = None,
) -> str:
    """Hash immutable subset manifests and annotations for resume locking.

    Image bytes are intentionally not read: the source manifests and generated
    COCO annotations define membership and geometry, while image-byte checks are
    the adapter's separate validation responsibility.
    """
    root = Path(subset_root).resolve()
    relative = list(files) if files is not None else [
        "train_images.txt", "val_images.txt",
        "annotations/instances_train.json", "annotations/instances_val.json",
    ]
    digest = hashlib.sha256()
    for value in sorted((Path(item) for item in relative), key=str):
        path = value if value.is_absolute() else root / value
        if not path.is_file():
            raise FileNotFoundError(f"dataset fingerprint input is missing: {path}")
        try:
            label = path.relative_to(root)
        except ValueError:
            label = value
        digest.update(str(label).replace("\\", "/").encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def capture_rng_state(loader_generator: torch.Generator | None = None) -> dict[str, Any]:
    """Capture process, CUDA and optional DataLoader generator state."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if loader_generator is not None:
        state["loader_generator"] = loader_generator.get_state()
    return state


def restore_rng_state(
    state: Mapping[str, Any] | None,
    loader_generator: torch.Generator | None = None,
) -> None:
    """Restore a state captured by :func:`capture_rng_state`.

    Missing CUDA state is tolerated for CPU prediction/read compatibility, but
    callers performing a resume should reject such a checkpoint explicitly if
    deterministic continuation is required.
    """
    if state and "rng" in state and not any(key in state for key in ("python", "numpy", "torch")):
        state = state["rng"]
    elif state and "rng_state" in state and not any(key in state for key in ("python", "numpy", "torch")):
        state = state["rng_state"]
    required = ["python", "numpy", "torch"]
    if loader_generator is not None:
        required.append("loader_generator")
    if torch.cuda.is_available():
        required.append("cuda")
    require_resume_state(state or {}, required)
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"].cpu())
    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_state])
    loader_state = state.get("loader_generator")
    if loader_state is not None and loader_generator is not None:
        loader_generator.set_state(loader_state.cpu())


def resume_mismatches(
    saved: Mapping[str, Any],
    current: Mapping[str, Any],
    required: Iterable[str],
) -> dict[str, tuple[Any, Any]]:
    """Return saved/current differences, including missing required values."""
    result: dict[str, tuple[Any, Any]] = {}
    for key in required:
        old = saved.get(key)
        new = current.get(key)
        if old != new:
            result[key] = (old, new)
    return result


def require_resume_state(
    checkpoint: Mapping[str, Any],
    required: Iterable[str],
) -> None:
    """Reject legacy/incomplete resume state with an actionable message."""
    missing = [key for key in required if checkpoint.get(key) is None]
    if missing:
        raise UnsafeResumeError(
            "cannot safely resume checkpoint: missing required state "
            + ", ".join(missing)
        )


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def reconcile_results_csv(
    csv_path: str | Path,
    checkpoint_epoch: int,
) -> dict[str, Any]:
    """Make CSV progress no later than authoritative checkpoint progress.

    Rows beyond the checkpoint are retained in ``results.csv.stale`` before
    the active CSV is atomically rewritten.  The sidecar is evidence of a
    crash between writes and prevents it from being mistaken for progress.
    """
    path = Path(csv_path)
    if checkpoint_epoch < 0:
        raise ValueError("checkpoint_epoch must be non-negative")
    if not path.exists():
        return {"checkpoint_epoch": checkpoint_epoch, "kept_rows": 0,
                "stale_rows": 0, "stale_path": None}
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return {"checkpoint_epoch": checkpoint_epoch, "kept_rows": 0,
                "stale_rows": 0, "stale_path": None}
    header, data = rows[0], rows[1:]
    kept: list[list[str]] = []
    stale: list[list[str]] = []
    for row in data:
        try:
            epoch = int(row[0])
        except (IndexError, TypeError, ValueError):
            # A malformed row cannot prove progress; preserve it as stale.
            stale.append(row)
            continue
        (kept if epoch <= checkpoint_epoch else stale).append(row)

    stale_path: Path | None = None
    if stale:
        import io
        stale_path = path.with_suffix(path.suffix + ".stale")
        prior = stale_path.read_text(encoding="utf-8") if stale_path.exists() else ""
        stream = io.StringIO(newline="")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(stale)
        content = stream.getvalue()
        # Avoid destroying evidence from an earlier reconciliation.
        _atomic_text(stale_path, prior + content)

    output = [header, *kept]
    lines: list[str] = []
    for row in output:
        # csv.writer to an in-memory stream avoids platform-specific quoting;
        # importing io only for this tiny operation keeps the file simple.
        import io
        stream = io.StringIO(newline="")
        csv.writer(stream, lineterminator="\n").writerow(row)
        lines.append(stream.getvalue())
    _atomic_text(path, "".join(lines))
    return {
        "checkpoint_epoch": checkpoint_epoch,
        "kept_rows": len(kept),
        "stale_rows": len(stale),
        "stale_path": str(stale_path) if stale_path else None,
    }


def csv_epoch(csv_path: str | Path) -> int:
    """Return the largest valid CSV epoch, or zero for absent/empty files."""
    path = Path(csv_path)
    if not path.exists():
        return 0
    highest = 0
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.reader(fh):
                try:
                    highest = max(highest, int(row[0]))
                except (IndexError, TypeError, ValueError):
                    continue
    except OSError:
        return 0
    return highest


__all__ = [
    "UnsafeResumeError", "capture_rng_state", "csv_epoch", "dataset_fingerprint",
    "reconcile_results_csv", "require_resume_state", "restore_rng_state",
    "resume_mismatches",
]
