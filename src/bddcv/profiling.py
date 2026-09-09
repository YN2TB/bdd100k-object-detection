"""Structured, subprocess-isolated hardware profiling interfaces."""
from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .paths import PROJECT_ROOT
from .registry import ModelSpec, get_model_spec, list_models


# Version 1 was the historical post-fetch compute-step metric.  Keep its
# artifacts readable for reporting, but never use them to select hardware
# settings after switching to loader-inclusive intervals.
PROFILE_SCHEMA_VERSION = 3
TIMING_SCHEMA = "end_to_end_loader_inclusive"
TIMING_SCHEMA_VERSION = 3


class ProfileStatus(str, Enum):
    SUCCESS = "success"
    OOM = "oom"
    NONFINITE = "nonfinite"
    VALIDATION_FAILURE = "validation_failure"
    GPU_UNAVAILABLE = "gpu_unavailable"
    ERROR = "error"


@dataclass
class ProfileOutcome:
    """Serializable outcome of one model/batch candidate."""

    model_id: str
    batch: int
    status: ProfileStatus | str
    reason: str = ""
    median_step_seconds: float | None = None
    p95_step_seconds: float | None = None
    images_per_second: float | None = None
    peak_allocated_bytes: int | None = None
    peak_reserved_bytes: int | None = None
    peak_total_memory_bytes: int | None = None
    peak_memory_bytes: int | None = None
    total_memory_bytes: int | None = None
    validation_steps: int | None = None
    warmup_steps: int | None = None
    measured_steps: int | None = None
    measured_elapsed_seconds: float | None = None
    profile_schema_version: int | None = None
    timing_schema: str | None = None
    timing_schema_version: int | None = None
    command: tuple[str, ...] = field(default_factory=tuple)
    returncode: int | None = None
    elapsed_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.status, str):
            try:
                self.status = ProfileStatus(self.status)
            except ValueError:
                self.status = ProfileStatus.ERROR
        self.batch = int(self.batch)
        if self.profile_schema_version is not None:
            self.profile_schema_version = int(self.profile_schema_version)
        if self.timing_schema_version is not None:
            self.timing_schema_version = int(self.timing_schema_version)

    @property
    def successful(self) -> bool:
        return self.status == ProfileStatus.SUCCESS

    @property
    def memory_fraction(self) -> float | None:
        if not (self.peak_total_memory_bytes or self.peak_memory_bytes) or not self.total_memory_bytes:
            return None
        return (self.peak_total_memory_bytes or self.peak_memory_bytes) / self.total_memory_bytes

    @property
    def throughput(self) -> float:
        if (
            self.measured_steps is not None
            and self.measured_elapsed_seconds is not None
            and self.measured_steps > 0
            and math.isfinite(float(self.measured_elapsed_seconds))
            and self.measured_elapsed_seconds > 0
        ):
            return self.batch * self.measured_steps / self.measured_elapsed_seconds
        return float(self.images_per_second or 0.0)

    @property
    def timing_compatible(self) -> bool:
        """Whether this artifact uses the current loader-inclusive schema."""
        versions_match = (
            self.profile_schema_version == PROFILE_SCHEMA_VERSION
            and self.timing_schema == TIMING_SCHEMA
            and self.timing_schema_version == TIMING_SCHEMA_VERSION
        )
        if not versions_match:
            return False
        if self.status != ProfileStatus.SUCCESS:
            return True
        elapsed = self.measured_elapsed_seconds
        steps = self.measured_steps
        return (
            steps is not None and steps > 0
            and elapsed is not None and math.isfinite(float(elapsed)) and elapsed > 0
        )

    @property
    def total_memory(self) -> int | None:
        return self.total_memory_bytes

    @property
    def memory_bytes(self) -> int:
        return int(self.peak_total_memory_bytes or self.peak_memory_bytes
                   or self.peak_reserved_bytes or 0)

    def to_dict(self) -> dict[str, Any]:
        value = {
            "model_id": self.model_id,
            "batch": self.batch,
            "status": self.status.value if isinstance(self.status, ProfileStatus) else str(self.status),
            "reason": self.reason,
            "median_step_seconds": self.median_step_seconds,
            "p95_step_seconds": self.p95_step_seconds,
            "images_per_second": self.images_per_second,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "peak_reserved_bytes": self.peak_reserved_bytes,
            "peak_total_memory_bytes": self.peak_total_memory_bytes,
            "peak_memory_bytes": self.peak_memory_bytes,
            "total_memory_bytes": self.total_memory_bytes,
            "validation_steps": self.validation_steps,
            "warmup_steps": self.warmup_steps,
            "measured_steps": self.measured_steps,
            "measured_elapsed_seconds": self.measured_elapsed_seconds,
            "profile_schema_version": self.profile_schema_version,
            "timing_schema": self.timing_schema,
            "timing_schema_version": self.timing_schema_version,
            "command": list(self.command),
            "returncode": self.returncode,
            "elapsed_seconds": self.elapsed_seconds,
            "metadata": self.metadata,
        }
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProfileOutcome":
        data = dict(value)
        if data.get("peak_total_memory_bytes") is None:
            data["peak_total_memory_bytes"] = data.get("peak_memory_bytes") or data.get("peak_gpu_memory_bytes")
        if data.get("total_memory_bytes") is None:
            data["total_memory_bytes"] = data.get("gpu_total_memory_bytes")
        data["command"] = tuple(data.get("command") or ())
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


def classify_profile_output(
    model_id: str,
    batch: int,
    *,
    returncode: int,
    stdout: str = "",
    stderr: str = "",
    elapsed_seconds: float | None = None,
) -> ProfileOutcome:
    """Classify a child process without treating failures as measurements."""
    text = f"{stdout}\n{stderr}".lower()
    structured: dict[str, Any] | None = None
    for line in reversed((stdout or "").splitlines()):
        candidate = line.strip()
        if candidate.startswith("PROFILE_RESULT"):
            candidate = candidate[len("PROFILE_RESULT"):].strip()
        if not candidate.startswith("{"):
            continue
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "status" in value:
            structured = value
            break
    structured_status = str(structured.get("status", "")) if structured else ""
    if structured_status in {item.value for item in ProfileStatus}:
        status = ProfileStatus(structured_status)
        reason = str(structured.get("reason", ""))
        return ProfileOutcome(
            model_id=model_id, batch=batch, status=status, reason=reason,
            median_step_seconds=structured.get("median_step_seconds"),
            p95_step_seconds=structured.get("p95_step_seconds"),
            images_per_second=structured.get("images_per_second"),
            peak_allocated_bytes=structured.get("peak_allocated_bytes"),
            peak_reserved_bytes=structured.get("peak_reserved_bytes"),
            peak_total_memory_bytes=structured.get("peak_total_memory_bytes"),
            total_memory_bytes=structured.get("total_memory_bytes"),
            validation_steps=structured.get("validation_steps"),
            warmup_steps=structured.get("warmup_steps"),
            measured_steps=structured.get("measured_steps"),
            measured_elapsed_seconds=structured.get("measured_elapsed_seconds"),
            profile_schema_version=structured.get("profile_schema_version"),
            timing_schema=structured.get("timing_schema"),
            timing_schema_version=structured.get("timing_schema_version"),
            returncode=returncode, elapsed_seconds=elapsed_seconds,
            metadata={key: value for key, value in structured.items()
                      if key not in {
                          "status", "reason", "model_id", "batch",
                          "median_step_seconds", "p95_step_seconds", "images_per_second",
                          "peak_allocated_bytes", "peak_reserved_bytes",
                          "peak_total_memory_bytes", "total_memory_bytes",
                          "validation_steps", "warmup_steps", "measured_steps",
                          "measured_elapsed_seconds", "profile_schema_version",
                          "timing_schema", "timing_schema_version",
                      }},
        )
    if any(token in text for token in (
        "validation failure", "validation failed", "validation oom", "val failed",
    )):
        status = ProfileStatus.VALIDATION_FAILURE
        reason = "validation probe failed"
    elif any(token in text for token in (
        "out of memory", "cuda out of memory", "cudnn_status_alloc_failed",
        "resourceexhaustederror",
    )):
        status = ProfileStatus.OOM
        reason = "GPU memory allocation failed"
    elif any(token in text for token in (
        "non-finite", "nonfinite", "nan loss", "loss nan", "loss is nan",
        "inf loss", "loss inf",
    )):
        status = ProfileStatus.NONFINITE
        reason = "non-finite loss or metric"
    elif any(token in text for token in (
        "gpu unavailable", "cuda unavailable", "cuda is not available",
        "no cuda device", "gpu_unavailable",
    )):
        status = ProfileStatus.GPU_UNAVAILABLE
        reason = "CUDA device is unavailable"
    elif returncode == 0:
        status = ProfileStatus.SUCCESS
        reason = ""
    else:
        status = ProfileStatus.ERROR
        reason = (stderr.strip() or stdout.strip() or f"child exited {returncode}")[-1000:]
    return ProfileOutcome(
        model_id=model_id, batch=batch, status=status, reason=reason,
        returncode=returncode, elapsed_seconds=elapsed_seconds,
    )


def _as_outcome(value: ProfileOutcome | Mapping[str, Any]) -> ProfileOutcome:
    return value if isinstance(value, ProfileOutcome) else ProfileOutcome.from_dict(value)


def select_profile(
    outcomes: Iterable[ProfileOutcome | Mapping[str, Any]],
    *,
    max_memory_fraction: float = 0.90,
    tie_fraction: float = 0.03,
) -> ProfileOutcome | None:
    """Choose max throughput under 90% total memory, preferring lower memory.

    Candidates within ``tie_fraction`` below the fastest are considered tied;
    among those the lower-memory candidate wins.  OOM, nonfinite, validation
    failure and unavailable outcomes never enter selection.
    """
    if not 0 < max_memory_fraction <= 1:
        raise ValueError("max_memory_fraction must be in (0, 1]")
    if tie_fraction < 0:
        raise ValueError("tie_fraction must be non-negative")
    candidates: list[ProfileOutcome] = []
    for raw in outcomes:
        item = _as_outcome(raw)
        if not item.timing_compatible or not item.successful or item.throughput <= 0:
            continue
        fraction = item.memory_fraction
        if fraction is None or not math.isfinite(fraction) or not 0 < fraction <= max_memory_fraction:
            continue
        candidates.append(item)
    if not candidates:
        return None
    fastest = max(item.throughput for item in candidates)
    tied = [item for item in candidates if item.throughput >= fastest * (1 - tie_fraction)]
    return min(tied, key=lambda item: (item.memory_bytes or 2**63 - 1, -item.throughput, item.batch))


def should_try_ram_cache(baseline: ProfileOutcome) -> bool:
    """Try caching only for measured low utilization and loader-bound steps."""
    util = baseline.metadata.get("median_utilization_percent")
    wait = baseline.metadata.get("data_wait_fraction")
    return (baseline.timing_compatible and baseline.successful
            and isinstance(util, (int, float)) and math.isfinite(util) and util < 80
            and isinstance(wait, (int, float)) and math.isfinite(wait) and wait > 0.20)


def accept_ram_cache(baseline: ProfileOutcome, candidate: ProfileOutcome) -> bool:
    """Require 5% throughput improvement and both measured memory budgets."""
    ram = candidate.metadata.get("host_ram_peak_percent")
    memory = candidate.memory_fraction
    return (candidate.model_id == baseline.model_id and candidate.batch == baseline.batch
            and candidate.metadata.get("cache") == "ram"
            and candidate.timing_compatible and candidate.successful
            and baseline.throughput > 0 and candidate.throughput >= baseline.throughput * 1.05
            and isinstance(ram, (int, float)) and math.isfinite(ram) and ram < 75
            and memory is not None and 0 < memory <= 0.90)


def make_probe_manifest(
    source_root: str | Path,
    output: str | Path,
    *,
    dense_count: int = 128,
    sample_count: int = 384,
    seed: int = 0,
) -> Path:
    """Write the deterministic dense-plus-sampled 512-image profile manifest."""
    root = Path(source_root)
    ann_path = root / "annotations" / "instances_train.json"
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    images = data.get("images", [])
    counts = {image["id"]: 0 for image in images}
    for ann in data.get("annotations", []):
        if ann.get("image_id") in counts:
            counts[ann["image_id"]] += 1
    by_id = {image["id"]: image["file_name"] for image in images}
    names = sorted(
        ((filename, counts.get(image_id, 0)) for image_id, filename in by_id.items()),
        key=lambda item: (-item[1], item[0]),
    )
    names = [name for name, _ in names]
    dense = names[: min(dense_count, len(names))]
    remainder = [name for name in names if name not in set(dense)]
    rng = random.Random(seed)
    sampled = rng.sample(remainder, min(sample_count, len(remainder)))
    selected = dense + sorted(sampled)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(selected) + ("\n" if selected else ""), encoding="utf-8")
    return target


class ProfileController:
    """Run each candidate in a fresh child and persist one JSON outcome."""

    def __init__(
        self,
        output_root: str | Path,
        *,
        python: str | Path | None = None,
        candidate_script: str | Path | None = None,
    ) -> None:
        self.output_root = Path(output_root)
        self.python = str(python or sys.executable)
        self.candidate_script = Path(candidate_script) if candidate_script else (
            Path(__file__).resolve().parents[2] / "scripts" / "profile_candidate.py"
        )

    def command(self, model_id: str, batch: int, output_dir: Path, **kwargs: Any) -> list[str]:
        command = [
            self.python, "-u", str(self.candidate_script),
            "--model-id", model_id, "--batch", str(batch), "--out", str(output_dir),
        ]
        for key, value in kwargs.items():
            if value is None:
                continue
            option = "--" + key.replace("_", "-")
            command.extend([option, str(value)])
        return command

    def candidate_output(self, model_id: str, batch: int,
                         variant: str | None = None) -> Path:
        """Return a collision-free output directory for one probe variant."""
        root = self.output_root / model_id
        if variant:
            if Path(variant).name != variant:
                raise ValueError(f"unsafe profile variant: {variant!r}")
            root /= variant
        return root / f"batch-{int(batch)}"

    def run_candidate(self, model_id: str, batch: int, **kwargs: Any) -> ProfileOutcome:
        spec = get_model_spec(model_id)
        variant = kwargs.pop("variant", None)
        output_dir = self.candidate_output(spec.model_id, batch, variant)
        output_dir.mkdir(parents=True, exist_ok=True)
        command = self.command(spec.model_id, batch, output_dir, **kwargs)
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, cwd=str(PROJECT_ROOT),
                env={**os.environ, "PYTHONHASHSEED": str(spec.seed)},
            )
            elapsed = time.monotonic() - started
            outcome = classify_profile_output(
                spec.model_id, batch, returncode=completed.returncode,
                stdout=getattr(completed, "stdout", "") or "",
                stderr=getattr(completed, "stderr", "") or "",
                elapsed_seconds=elapsed,
            )
            # A successful candidate writes its detailed payload.  A malformed
            # payload is a validation failure, never a silent success.
            payload = output_dir / "profile.json"
            if outcome.successful:
                if not payload.exists():
                    # A structured stdout result is sufficient for a test
                    # runner; an unstructured success is not.
                    if outcome.images_per_second is None:
                        outcome.status = ProfileStatus.VALIDATION_FAILURE
                        outcome.reason = "successful child produced no profile payload"
                else:
                    try:
                        detail = json.loads(payload.read_text(encoding="utf-8"))
                        outcome = ProfileOutcome.from_dict({
                            **outcome.to_dict(), **detail,
                            "command": tuple(command), "returncode": completed.returncode,
                            "elapsed_seconds": elapsed,
                        })
                    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                        outcome.status = ProfileStatus.VALIDATION_FAILURE
                        outcome.reason = f"invalid profile payload: {exc}"
                if outcome.successful and not outcome.timing_compatible:
                    outcome.status = ProfileStatus.VALIDATION_FAILURE
                    outcome.reason = "successful child produced incompatible timing schema"
            outcome.command = tuple(command)
        except OSError as exc:
            outcome = ProfileOutcome(
                spec.model_id, batch, ProfileStatus.ERROR, reason=str(exc),
                command=tuple(command), returncode=None,
                elapsed_seconds=time.monotonic() - started,
            )
        (output_dir / "outcome.json").write_text(
            json.dumps(outcome.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return outcome

    def run(self, model_ids: Iterable[str] | None = None) -> list[ProfileOutcome]:
        """Run all registry candidates using their approved batch lists."""
        selected = [get_model_spec(value) for value in (model_ids or [s.model_id for s in list_models()])]
        outcomes: list[ProfileOutcome] = []
        for spec in selected:
            for batch in spec.batch_candidates:
                outcomes.append(self.run_candidate(spec.model_id, batch))
        return outcomes

    profile_candidate = run_candidate


__all__ = [
    "PROFILE_SCHEMA_VERSION", "TIMING_SCHEMA", "TIMING_SCHEMA_VERSION",
    "ProfileController", "ProfileOutcome", "ProfileStatus", "classify_profile_output",
    "make_probe_manifest", "select_profile", "OutcomeStatus", "ProfileResult",
    "choose_profile",
]

OutcomeStatus = ProfileStatus
ProfileResult = ProfileOutcome
choose_profile = select_profile
