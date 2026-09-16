"""Small backend-routing facade used by generic entrypoints."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .registry import (
    ModelResolutionError, ModelSpec, backend_for_model, resolve_model_spec,
)


@dataclass(frozen=True)
class BackendResolution:
    """Resolved model and backend identity for one checkpoint operation."""

    model_id: str
    backend: str
    spec: ModelSpec
    checkpoint: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "backend": self.backend,
            "checkpoint": str(self.checkpoint) if self.checkpoint else None,
            "spec": self.spec.to_dict(),
        }


def resolve_backend(
    model_id: str | None = None,
    checkpoint: str | Path | None = None,
    metadata: Mapping[str, Any] | str | Path | None = None,
) -> BackendResolution:
    """Resolve backend by explicit id or explicit model metadata."""
    spec = resolve_model_spec(model_id, checkpoint, metadata)
    return BackendResolution(spec.model_id, spec.backend, spec,
                             Path(checkpoint) if checkpoint is not None else None)


__all__ = ["BackendResolution", "ModelResolutionError", "backend_for_model", "resolve_backend"]
