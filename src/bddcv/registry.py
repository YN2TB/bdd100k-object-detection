"""The canonical detector registry and safe backend resolution.

The registry is intentionally data-only.  Training and prediction entrypoints
use a model id to choose an implementation; a checkpoint named ``best.pt`` is
never used as a proxy for the backend.  This matters because all six backends
use the same conventional checkpoint names.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .paths import RUNS_DIR, WEIGHTS_DIR


class ModelResolutionError(ValueError):
    """Raised when a checkpoint cannot be mapped to one unambiguous model."""


REGISTRY_VERSION = 1


@dataclass(frozen=True)
class ModelSpec:
    """Immutable configuration shared by training, profiling and prediction."""

    model_id: str
    backend: str
    checkpoint: str
    input_policy: str
    default_epochs: int
    batch_candidates: tuple[int, ...]
    precision: str
    seed: int = 0
    output_subdir: str = ""
    optimizer: str = "native"
    schedule: str = "native"
    augmentation: str = "native"
    validation_batch: int | None = None
    accumulation: int = 1
    schema_version: int = REGISTRY_VERSION

    @property
    def checkpoint_path(self) -> Path:
        """Return the project-local path for the pretrained checkpoint."""
        return WEIGHTS_DIR / self.checkpoint

    @property
    def output_dir(self) -> Path:
        """Return the default training output path for this model."""
        return RUNS_DIR / "train" / (self.output_subdir or self.model_id)

    @property
    def is_ultralytics(self) -> bool:
        return self.backend == "ultralytics"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["batch_candidates"] = list(self.batch_candidates)
        return value


# Keep this table in canonical experiment order.  DET_CLASSES remains the
# source of class order; this order is only the approved model roster.
MODEL_REGISTRY: dict[str, ModelSpec] = {
    "yolo11s": ModelSpec(
        "yolo11s", "ultralytics", "yolo11s.pt", "640 letterbox", 50,
        (8, 16, 32, 64), "amp", output_subdir="yolo11s",
        optimizer="Ultralytics native", schedule="Ultralytics native",
        augmentation="Ultralytics native",
    ),
    "yolo11m": ModelSpec(
        "yolo11m", "ultralytics", "yolo11m.pt", "640 letterbox", 50,
        (4, 8, 16, 32), "amp", output_subdir="yolo11m",
        optimizer="Ultralytics native", schedule="Ultralytics native",
        augmentation="Ultralytics native",
    ),
    "yolo26s": ModelSpec(
        "yolo26s", "ultralytics", "yolo26s.pt", "640 letterbox", 50,
        (8, 16, 32, 64), "amp", output_subdir="yolo26s",
        optimizer="Ultralytics native", schedule="Ultralytics native",
        augmentation="Ultralytics native",
    ),
    "frcnn-r50-fpn-v2": ModelSpec(
        "frcnn-r50-fpn-v2", "frcnn", "frcnn-r50-fpn-v2.pt",
        "1280x720 -> 640x360", 50, (2, 4, 8, 16), "amp",
        output_subdir="frcnn", optimizer="SGD", schedule="warmup+cosine",
        augmentation="horizontal flip", validation_batch=4,
    ),
    "rtdetr-l": ModelSpec(
        "rtdetr-l", "ultralytics", "rtdetr-l.pt", "640 native", 50, (8,),
        "amp", output_subdir="rtdetr-l", optimizer="Ultralytics native",
        schedule="Ultralytics native", augmentation="Ultralytics native",
    ),
    "rfdetr-small": ModelSpec(
        "rfdetr-small", "rfdetr", "rfdetr/rf-detr-small.pth", "512 native", 50,
        (1, 2, 4, 8, 16), "amp", output_subdir="rfdetr-small",
        optimizer="RF-DETR native", schedule="RF-DETR native",
        augmentation="RF-DETR native", validation_batch=1,
    ),
}

# A read-only-friendly alias is useful to callers and makes the public API
# explicit without exposing a mutable implementation detail.
MODEL_SPECS = MODEL_REGISTRY
MODEL_IDS = tuple(MODEL_REGISTRY)
model_registry = MODEL_REGISTRY
MODELS = MODEL_REGISTRY


def list_models() -> tuple[ModelSpec, ...]:
    """Return the approved roster in deterministic order."""
    return tuple(MODEL_REGISTRY.values())


def get_model_spec(model_id: str) -> ModelSpec:
    """Look up a model id, with a useful error for typos."""
    key = str(model_id).strip().lower()
    try:
        return MODEL_REGISTRY[key]
    except KeyError as exc:
        available = ", ".join(MODEL_REGISTRY)
        raise ModelResolutionError(
            f"unknown model id {model_id!r}; choose one of: {available}"
        ) from exc


get_model = get_model_spec


def _metadata_model_id(metadata: Mapping[str, Any] | Path | str | None) -> str | None:
    if metadata is None:
        return None
    if isinstance(metadata, (str, Path)):
        path = Path(metadata)
        if not path.exists():
            raise ModelResolutionError(f"model metadata file does not exist: {path}")
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelResolutionError(f"invalid model metadata file: {path}") from exc
    value = metadata.get("model_id") if isinstance(metadata, Mapping) else None
    return str(value) if value else None


def metadata_path_for_checkpoint(checkpoint: str | Path) -> Path:
    """Return the sidecar metadata path used for a checkpoint."""
    path = Path(checkpoint)
    return path.with_suffix(path.suffix + ".json")


def resolve_model_id(
    model_id: str | None = None,
    checkpoint: str | Path | None = None,
    metadata: Mapping[str, Any] | Path | str | None = None,
) -> str:
    """Resolve a model id without guessing from a generic checkpoint name.

    An explicit id wins, but it must agree with metadata when both are given.
    For a checkpoint-only call, a sidecar ``<checkpoint>.json`` containing
    ``model_id`` is accepted.  Otherwise resolution fails deliberately.  This
    prevents ``best.pt`` from silently selecting the wrong backend.
    """
    from_metadata = _metadata_model_id(metadata)
    if from_metadata is None and checkpoint is not None:
        sidecar = metadata_path_for_checkpoint(checkpoint)
        if sidecar.exists():
            from_metadata = _metadata_model_id(sidecar)
        if from_metadata is None:
            checkpoint_path = Path(checkpoint)
            # Run-level metadata is useful for Ultralytics' conventional
            # ``weights/{best,last}.pt`` files and remains explicit provenance.
            for candidate in (
                checkpoint_path.parent.parent / "model.json",
                checkpoint_path.parent.parent / "model_metadata.json",
            ):
                if candidate.exists():
                    from_metadata = _metadata_model_id(candidate)
                    if from_metadata is not None:
                        break

    if model_id is not None:
        spec = get_model_spec(model_id)
        if from_metadata is not None and spec.model_id != get_model_spec(from_metadata).model_id:
            raise ModelResolutionError(
                f"model id {spec.model_id!r} disagrees with metadata {from_metadata!r}"
            )
        return spec.model_id
    if from_metadata is not None:
        return get_model_spec(from_metadata).model_id

    if checkpoint is None:
        raise ModelResolutionError("an explicit model id is required")
    # A registered pretrained filename is unambiguous.  Do not extend this to
    # output names such as best.pt/last.pt, which are shared by every backend.
    basename = Path(checkpoint).name.lower()
    registered = [spec.model_id for spec in MODEL_REGISTRY.values()
                  if Path(spec.checkpoint).name.lower() == basename]
    if len(registered) == 1:
        return registered[0]
    raise ModelResolutionError(
        f"cannot resolve backend for checkpoint {checkpoint!r}; pass --model-id "
        "or provide a sidecar metadata file containing model_id"
    )


def resolve_model_spec(
    model_id: str | None = None,
    checkpoint: str | Path | None = None,
    metadata: Mapping[str, Any] | Path | str | None = None,
) -> ModelSpec:
    """Resolve and return the complete model specification."""
    return get_model_spec(resolve_model_id(model_id, checkpoint, metadata))


def backend_for_model(
    model_id: str | None = None,
    checkpoint: str | Path | None = None,
    metadata: Mapping[str, Any] | Path | str | None = None,
) -> str:
    """Return a backend only after unambiguous model resolution."""
    return resolve_model_spec(model_id, checkpoint, metadata).backend


def resolve_backend(
    model_id: str | None = None,
    checkpoint: str | Path | None = None,
    metadata: Mapping[str, Any] | Path | str | None = None,
) -> str:
    """Alias for :func:`backend_for_model` used by generic callers."""
    return backend_for_model(model_id, checkpoint, metadata)


def write_model_metadata(path: str | Path, spec: ModelSpec, **extra: Any) -> Path:
    """Persist a small, JSON-safe model provenance sidecar atomically."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = spec.to_dict()
    payload.update(extra)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(target)
    return target


__all__ = [
    "REGISTRY_VERSION", "MODEL_REGISTRY", "MODEL_SPECS", "MODEL_IDS", "model_registry", "MODELS",
    "ModelResolutionError", "ModelSpec", "backend_for_model", "resolve_backend",
    "get_model_spec", "get_model", "list_models",
    "metadata_path_for_checkpoint", "resolve_model_id", "resolve_model_spec",
    "write_model_metadata",
]
