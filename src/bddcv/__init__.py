"""BDD100K detector-comparison course project."""

from .registry import MODEL_REGISTRY, ModelSpec, get_model_spec, resolve_model_spec

__all__ = ["MODEL_REGISTRY", "ModelSpec", "get_model_spec", "resolve_model_spec"]
