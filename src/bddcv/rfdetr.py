"""RF-DETR's isolated dataset view and optional backend adapter.

RF-DETR expects one COCO annotation file per split and does not consume the
project's Ultralytics YAML directly.  This module creates a view of the
already-selected files; it never regenerates labels or touches the source
subset.  Hard links are used when possible, with a byte-verified copy as a
fallback for filesystems that do not support them.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .constants import DET_CLASSES, DATA_DIR
from .paths import WEIGHTS_DIR

DEFAULT_SOURCE = DATA_DIR / "source_daytime_clear"
DEFAULT_ADAPTER = DATA_DIR / "adapters" / "rfdetr"
RFDETR_REQUIREMENT = "rfdetr==1.10.1"
RFDETR_ENV_DIRNAME = ".venv-rfdetr"


class RFDETRAdapterError(RuntimeError):
    """Raised when the generated view would not be an exact source view."""


class RFDETRBackendUnavailable(RuntimeError):
    """Raised when the isolated RF-DETR environment/backend cannot be used."""


@dataclass
class AdapterValidation:
    """Structured validation evidence for one RF-DETR dataset view."""

    valid: bool
    train_images: int = 0
    valid_images: int = 0
    train_categories: tuple[str, ...] = field(default_factory=tuple)
    valid_categories: tuple[str, ...] = field(default_factory=tuple)
    disjoint: bool = False
    filenames_exact: bool = False
    bytes_match: bool = False
    category_mapping_exact: bool = False
    errors: tuple[str, ...] = field(default_factory=tuple)

    @property
    def val_images(self) -> int:
        """Alias matching the source split name used by project manifests."""
        return self.valid_images

    @property
    def categories(self) -> tuple[str, ...]:
        return self.train_categories

    def to_dict(self) -> dict[str, Any]:
        value = {
            "valid": self.valid,
            "train_images": self.train_images,
            "valid_images": self.valid_images,
            "val_images": self.valid_images,
            "train_categories": list(self.train_categories),
            "valid_categories": list(self.valid_categories),
            "disjoint": self.disjoint,
            "filenames_exact": self.filenames_exact,
            "bytes_match": self.bytes_match,
            "category_mapping_exact": self.category_mapping_exact,
            "errors": list(self.errors),
        }
        return value

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _manifest(source_root: Path, split: str) -> list[str]:
    path = source_root / f"{split}_images.txt"
    if not path.exists():
        raise RFDETRAdapterError(f"source manifest is missing: {path}")
    names = path.read_text(encoding="utf-8").split()
    if len(names) != len(set(names)):
        raise RFDETRAdapterError(f"source manifest contains duplicate names: {path}")
    unsafe = [name for name in names if Path(name).name != name or Path(name).is_absolute()]
    if unsafe:
        raise RFDETRAdapterError(f"source manifest contains unsafe filenames: {unsafe[:3]}")
    return names


def _annotation(source_root: Path, split: str) -> dict[str, Any]:
    path = source_root / "annotations" / f"instances_{split}.json"
    if not path.exists():
        raise RFDETRAdapterError(f"source annotation is missing: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RFDETRAdapterError(f"invalid source annotation: {path}") from exc
    if not isinstance(data, dict):
        raise RFDETRAdapterError(f"source annotation must be an object: {path}")
    return data


def _expected_categories() -> list[dict[str, Any]]:
    return [{"id": i + 1, "name": name} for i, name in enumerate(DET_CLASSES)]


def _annotation_for_split(source_root: Path, split: str, names: list[str]) -> dict[str, Any]:
    data = _annotation(source_root, split)
    if list(data.get("categories", [])) != _expected_categories():
        raise RFDETRAdapterError(f"{split} source categories do not match DET_CLASSES")
    images = list(data.get("images", []))
    image_names = [str(image.get("file_name")) for image in images]
    image_ids_list = [image.get("id") for image in images]
    if len(image_names) != len(set(image_names)):
        raise RFDETRAdapterError(f"{split} annotation contains duplicate filenames")
    if len(image_ids_list) != len(set(image_ids_list)):
        raise RFDETRAdapterError(f"{split} annotation contains duplicate image ids")
    by_name = {str(image.get("file_name")): image for image in images}
    missing = [name for name in names if name not in by_name]
    extra = sorted(set(by_name) - set(names))
    if missing or extra:
        detail = []
        if missing:
            detail.append(f"missing={missing[:3]}")
        if extra:
            detail.append(f"extra={extra[:3]}")
        raise RFDETRAdapterError(
            f"{split} annotation/manifest mismatch ({', '.join(detail)})"
        )
    # Preserve all source annotation fields and ordering, while limiting the
    # view to exactly the immutable manifest.  Image IDs remain unchanged.
    image_ids = {by_name[name]["id"] for name in names}
    annotations = list(data.get("annotations", []))
    annotation_ids = [ann.get("id") for ann in annotations]
    if len(annotation_ids) != len(set(annotation_ids)):
        raise RFDETRAdapterError(f"{split} annotation contains duplicate annotation ids")
    unknown_image_ids = {ann.get("image_id") for ann in annotations} - set(image_ids_list)
    if unknown_image_ids:
        raise RFDETRAdapterError(
            f"{split} annotations reference unknown image ids: {sorted(unknown_image_ids)[:3]}"
        )
    valid_categories = set(range(1, len(DET_CLASSES) + 1))
    unknown_categories = {ann.get("category_id") for ann in annotations} - valid_categories
    if unknown_categories:
        raise RFDETRAdapterError(
            f"{split} annotations use unknown category ids: {sorted(unknown_categories)[:3]}"
        )
    output = dict(data)
    output["images"] = [by_name[name] for name in names]
    output["annotations"] = [
        ann for ann in annotations if ann.get("image_id") in image_ids
    ]
    output["categories"] = _expected_categories()
    return output


def _write_json_if_needed(path: Path, data: Mapping[str, Any]) -> None:
    encoded = (json.dumps(data, indent=2, sort_keys=False) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != encoded:
            raise RFDETRAdapterError(f"refusing to overwrite conflicting adapter file: {path}")
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(encoded)
    os.replace(tmp, path)


def _link_or_copy(source: Path, target: Path, copy_mode: str = "auto") -> str:
    if target.exists():
        if target.is_file() and target.stat().st_size == source.stat().st_size \
                and _sha256(target) == _sha256(source):
            return "existing"
        raise RFDETRAdapterError(f"refusing to overwrite conflicting adapter image: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if copy_mode not in {"auto", "hardlink", "copy"}:
        raise ValueError("copy_mode must be auto, hardlink, or copy")
    if copy_mode != "copy":
        try:
            os.link(source, target)
            return "hardlink"
        except OSError:
            if copy_mode == "hardlink":
                raise RFDETRAdapterError(f"cannot create hard link: {source} -> {target}")
    temporary = target.with_name(target.name + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)
    if target.stat().st_size != source.stat().st_size or _sha256(target) != _sha256(source):
        target.unlink(missing_ok=True)
        raise RFDETRAdapterError(f"byte verification failed after copying {source}")
    return "copy"


def build_adapter(
    source_root: str | Path = DEFAULT_SOURCE,
    adapter_root: str | Path = DEFAULT_ADAPTER,
    copy_mode: str = "auto",
) -> dict[str, Any]:
    """Build the RF-DETR ``train``/``valid`` COCO view and validate it.

    The operation is additive and refuses conflicting existing files.  Source
    manifests and annotations are read before any adapter file is created.
    """
    source = Path(source_root).resolve()
    adapter = Path(adapter_root).resolve()
    source_names = {split: _manifest(source, split) for split in ("train", "val")}
    source_coco = {
        split: _annotation_for_split(source, split, source_names[split])
        for split in ("train", "val")
    }
    if set(source_names["train"]) & set(source_names["val"]):
        raise RFDETRAdapterError("train and validation manifests are not disjoint")

    actions: dict[str, int] = {"hardlink": 0, "copy": 0, "existing": 0}
    for source_split, target_split in (("train", "train"), ("val", "valid")):
        source_images = source / "images" / source_split
        target_dir = adapter / target_split
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in source_names[source_split]:
            source_image = source_images / name
            if not source_image.is_file():
                raise RFDETRAdapterError(f"source image is missing: {source_image}")
            action = _link_or_copy(source_image, target_dir / name, copy_mode)
            actions[action] += 1
        # The release convention is ``_annotations.coco.json``.  Keep a
        # descriptive alias as well for tools that use annotations.json.
        _write_json_if_needed(target_dir / "_annotations.coco.json", source_coco[source_split])
        _write_json_if_needed(target_dir / "annotations.json", source_coco[source_split])

    report = validate_adapter(source, adapter)
    if not report.valid:
        raise RFDETRAdapterError(
            "RF-DETR adapter validation failed: " + "; ".join(report.errors)
        )
    result = report.to_dict()
    result["source_root"] = str(source)
    result["adapter_root"] = str(adapter)
    result["actions"] = actions
    return result


def build_probe_adapter(source_root: str | Path, adapter_root: str | Path,
                        train_names: Iterable[str], val_names: Iterable[str]) -> Path:
    """Build an additive COCO view containing exactly the profiling filenames."""
    source = Path(source_root).resolve()
    adapter = Path(adapter_root).resolve()
    selections = {"train": list(train_names), "val": list(val_names)}
    for split, names in selections.items():
        if not names or len(names) != len(set(names)):
            raise RFDETRAdapterError(f"{split} probe names must be non-empty and unique")
        data = _annotation(source, split)
        if list(data.get("categories", [])) != _expected_categories():
            raise RFDETRAdapterError(f"{split} source categories do not match DET_CLASSES")
        by_name = {str(image.get("file_name")): image for image in data.get("images", [])}
        missing = sorted(set(names) - set(by_name))
        if missing:
            raise RFDETRAdapterError(f"{split} probe images are missing: {missing[:3]}")
        image_ids = {by_name[name]["id"] for name in names}
        subset = dict(data)
        subset["images"] = [by_name[name] for name in names]
        subset["annotations"] = [
            annotation for annotation in data.get("annotations", [])
            if annotation.get("image_id") in image_ids
        ]
        target_split = "valid" if split == "val" else split
        target_dir = adapter / target_split
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            _link_or_copy(source / "images" / split / name, target_dir / name)
        _write_json_if_needed(target_dir / "_annotations.coco.json", subset)
    return adapter


def _read_adapter_annotation(adapter: Path, target_split: str) -> dict[str, Any]:
    path = adapter / target_split / "_annotations.coco.json"
    if not path.exists():
        path = adapter / target_split / "annotations.json"
    if not path.exists():
        raise RFDETRAdapterError(f"adapter annotation is missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RFDETRAdapterError(f"invalid adapter annotation: {path}") from exc


def validate_adapter(
    source_root: str | Path = DEFAULT_SOURCE,
    adapter_root: str | Path = DEFAULT_ADAPTER,
) -> AdapterValidation:
    """Check counts, disjointness, categories, filenames and image bytes."""
    source = Path(source_root).resolve()
    adapter = Path(adapter_root).resolve()
    errors: list[str] = []
    source_names: dict[str, list[str]] = {}
    adapter_data: dict[str, dict[str, Any]] = {}
    for source_split, target_split in (("train", "train"), ("val", "valid")):
        try:
            source_names[source_split] = _manifest(source, source_split)
            adapter_data[source_split] = _read_adapter_annotation(adapter, target_split)
        except RFDETRAdapterError as exc:
            errors.append(str(exc))
            source_names.setdefault(source_split, [])
            adapter_data.setdefault(source_split, {})

    train_names = set(source_names.get("train", []))
    val_names = set(source_names.get("val", []))
    disjoint = not (train_names & val_names)
    if not disjoint:
        errors.append("source train/valid filenames overlap")

    filename_exact = True
    bytes_match = True
    category_exact = True
    categories: dict[str, tuple[str, ...]] = {}
    counts: dict[str, int] = {}
    for source_split, target_split in (("train", "train"), ("val", "valid")):
        data = adapter_data.get(source_split, {})
        images = data.get("images", []) if isinstance(data, dict) else []
        names = [str(image.get("file_name")) for image in images]
        categories[source_split] = tuple(
            str(category.get("name")) for category in data.get("categories", [])
        ) if isinstance(data, dict) else tuple()
        counts[source_split] = len(images)
        expected = source_names.get(source_split, [])
        if names != expected:
            filename_exact = False
            errors.append(f"{target_split} filenames do not match the manifest")
        if list(data.get("categories", [])) != _expected_categories():
            category_exact = False
            errors.append(f"{target_split} categories do not match DET_CLASSES")
        for name in expected:
            source_image = source / "images" / source_split / name
            adapter_image = adapter / target_split / name
            if not adapter_image.is_file():
                bytes_match = False
                errors.append(f"missing adapter image: {adapter_image}")
                continue
            if not source_image.is_file() or source_image.stat().st_size != adapter_image.stat().st_size:
                bytes_match = False
                errors.append(f"adapter byte size differs: {name}")
            elif _sha256(source_image) != _sha256(adapter_image):
                bytes_match = False
                errors.append(f"adapter bytes differ: {name}")

    valid = not errors and disjoint and filename_exact and bytes_match and category_exact \
        and counts.get("train", 0) == len(source_names.get("train", [])) \
        and counts.get("val", 0) == len(source_names.get("val", []))
    return AdapterValidation(
        valid=valid,
        train_images=counts.get("train", 0),
        valid_images=counts.get("val", 0),
        train_categories=categories.get("train", tuple()),
        valid_categories=categories.get("val", tuple()),
        disjoint=disjoint,
        filenames_exact=filename_exact,
        bytes_match=bytes_match,
        category_mapping_exact=category_exact,
        errors=tuple(errors),
    )


def isolated_python(
    repository_root: str | Path | None = None,
    python: str | Path | None = None,
) -> Path:
    """Resolve the RF-DETR interpreter without falling back to main env."""
    if python is not None:
        value = Path(python).expanduser().absolute()
    else:
        root = Path(repository_root or Path(__file__).resolve().parents[2])
        value = root / RFDETR_ENV_DIRNAME / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not value.is_file():
        raise RFDETRBackendUnavailable(
            f"RF-DETR isolated interpreter is unavailable: {value}. "
            f"Create it separately and install {RFDETR_REQUIREMENT}."
        )
    return value


def isolated_install_command(python: str | Path) -> list[str]:
    """Return (but never execute) the isolated RF-DETR install command."""
    root = Path(__file__).resolve().parents[2]
    return [
        str(python), "-m", "pip", "install", "-r", str(root / "requirements-rfdetr.txt"),
        "-c", str(root / "constraints-rfdetr.txt"),
        "--extra-index-url", "https://download.pytorch.org/whl/cu128",
    ]


def check_isolated_environment(python: str | Path) -> dict[str, Any]:
    """Probe an isolated interpreter and return versions without installing."""
    code = (
        "import importlib.util, sys; "
        "print(sys.version.split()[0]); "
        "print(bool(importlib.util.find_spec('rfdetr'))); "
        "print(bool(importlib.util.find_spec('torch')))"
    )
    result = subprocess.run([str(python), "-c", code], capture_output=True, text=True)
    lines = result.stdout.splitlines()
    available = result.returncode == 0 and len(lines) >= 3 and lines[1].strip() == "True"
    return {
        "python": str(python), "returncode": result.returncode,
        "version": lines[0] if lines else None, "rfdetr": available,
        "torch": len(lines) >= 3 and lines[2].strip() == "True",
        "stderr": result.stderr.strip(),
    }


class RFDETRAdapter:
    """Thin optional adapter around the pinned RF-DETR Python API.

    Importing this module remains safe in the main environment.  A missing
    isolated package is reported as an explicit unavailable outcome, not as a
    silent fallback to Ultralytics.
    """

    def __init__(
        self,
        model_id: str = "rfdetr-small",
        checkpoint: str | Path | None = None,
        python: str | Path | None = None,
    ) -> None:
        if model_id != "rfdetr-small":
            raise ValueError(f"unsupported RF-DETR model id: {model_id}")
        self.model_id = model_id
        self.checkpoint = Path(checkpoint) if checkpoint is not None else (
            WEIGHTS_DIR / "rfdetr" / "rf-detr-small.pth"
        )
        self.python = isolated_python(python=python)

    @staticmethod
    def _load_class():
        try:
            from rfdetr import RFDETRSmall  # type: ignore
            return RFDETRSmall
        except ImportError as exc:
            raise RFDETRBackendUnavailable(
                f"{RFDETR_REQUIREMENT} is not available in this interpreter"
            ) from exc

    def construct(self, *, for_training: bool = True, **kwargs: Any) -> Any:
        cls = self._load_class()
        if not for_training:
            if not self.checkpoint.is_file():
                raise RFDETRBackendUnavailable(
                    f"RF-DETR checkpoint is unavailable: {self.checkpoint}"
                )
            return cls.from_checkpoint(str(self.checkpoint))
        if self.checkpoint is not None:
            kwargs.setdefault("pretrain_weights", str(self.checkpoint))
        return cls(**kwargs)


# Backwards-friendly names used by scripts and external smoke tests.
build_rfdetr_adapter = build_adapter
validate_rfdetr_adapter = validate_adapter
create_dataset_view = build_adapter
validate_dataset_view = validate_adapter
RFDETRDatasetAdapter = RFDETRAdapter


__all__ = [
    "AdapterValidation", "DEFAULT_ADAPTER", "DEFAULT_SOURCE", "RFDETRAdapter",
    "RFDETRAdapterError", "RFDETRBackendUnavailable", "RFDETR_ENV_DIRNAME",
    "RFDETR_REQUIREMENT", "build_adapter", "build_probe_adapter", "build_rfdetr_adapter",
    "check_isolated_environment", "isolated_install_command", "isolated_python",
    "validate_adapter", "validate_rfdetr_adapter", "create_dataset_view",
    "validate_dataset_view", "RFDETRDatasetAdapter",
]
