"""Run one production-backend profiling candidate in a fresh process."""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bddcv.native_probes import (  # noqa: E402
    run_frcnn_probe,
    run_rfdetr_probe,
    run_ultralytics_probe,
    ValidationProbeError,
)
from bddcv.paths import RFDETR_ENV_DIR  # noqa: E402
from bddcv.profiling import (  # noqa: E402
    PROFILE_SCHEMA_VERSION, TIMING_SCHEMA, TIMING_SCHEMA_VERSION,
)
from bddcv.registry import get_model_spec  # noqa: E402
from bddcv.rfdetr import isolated_python  # noqa: E402


def write_result(out: Path, result: dict[str, Any]) -> None:
    """Atomically persist and emit the structured profiling result."""
    out.mkdir(parents=True, exist_ok=True)
    target = out / "profile.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    print("PROFILE_RESULT " + json.dumps(result, sort_keys=True), flush=True)


def result_metadata() -> dict[str, Any]:
    """Return the schema markers required for reusable profile outcomes."""
    return {
        "profile_schema_version": PROFILE_SCHEMA_VERSION,
        "timing_schema": TIMING_SCHEMA,
        "timing_schema_version": TIMING_SCHEMA_VERSION,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("data/source_daytime_clear"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--validation-steps", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--cache", choices=("none", "ram"), default="none")
    args = parser.parse_args()
    spec = get_model_spec(args.model_id)
    args.out.mkdir(parents=True, exist_ok=True)

    if spec.backend == "rfdetr" and Path(sys.prefix).resolve() != RFDETR_ENV_DIR.resolve():
        python = isolated_python()
        os.execve(
            str(python), [str(python), str(Path(__file__).resolve()), *sys.argv[1:]],
            os.environ.copy(),
        )

    import torch

    if not torch.cuda.is_available():
        result = {"status": "gpu_unavailable", "model_id": spec.model_id,
                  "batch": args.batch, "reason": "CUDA is not available",
                  **result_metadata()}
        write_result(args.out, result)
        return 0

    try:
        if spec.backend == "ultralytics":
            if spec.model_id == "rtdetr-l":
                raise ValueError("completed RT-DETR-L baseline is excluded from profiling")
            result = run_ultralytics_probe(
                model_id=spec.model_id,
                checkpoint=spec.checkpoint_path,
                source=args.source.resolve(),
                manifest=args.probe_manifest.resolve(),
                out=args.out.resolve(),
                batch=args.batch,
                workers=args.workers,
                cache=args.cache,
                prefetch=args.prefetch,
                warmup=args.warmup,
                steps=args.steps,
                validation_steps=args.validation_steps,
                imgsz=args.imgsz or 640,
            )
        elif spec.backend == "frcnn":
            result = run_frcnn_probe(
                model_id=spec.model_id,
                source=args.source.resolve(),
                manifest=args.probe_manifest.resolve(),
                out=args.out.resolve(),
                batch=args.batch,
                validation_batch=spec.validation_batch or args.batch,
                workers=args.workers,
                cache=args.cache,
                prefetch=args.prefetch,
                warmup=args.warmup,
                steps=args.steps,
                validation_steps=args.validation_steps,
                imgsz=args.imgsz or 640,
            )
        elif spec.backend == "rfdetr":
            result = run_rfdetr_probe(
                model_id=spec.model_id,
                checkpoint=spec.checkpoint_path,
                source=args.source.resolve(),
                manifest=args.probe_manifest.resolve(),
                out=args.out.resolve(),
                batch=args.batch,
                workers=args.workers,
                cache=args.cache,
                prefetch=args.prefetch,
                warmup=args.warmup,
                steps=args.steps,
                validation_steps=args.validation_steps,
                imgsz=args.imgsz or 512,
            )
        else:
            raise NotImplementedError(f"native {spec.backend} probe is not implemented")
    except torch.cuda.OutOfMemoryError as exc:
        result = {"status": "oom", "model_id": spec.model_id, "batch": args.batch,
                  "reason": str(exc), **result_metadata()}
    except FloatingPointError as exc:
        result = {"status": "nonfinite", "model_id": spec.model_id, "batch": args.batch,
                  "reason": str(exc), **result_metadata()}
    except ValidationProbeError as exc:
        result = {"status": "validation_failure", "model_id": spec.model_id,
                  "batch": args.batch, "reason": str(exc), **result_metadata()}
    except Exception as exc:  # Fresh process must serialize unexpected backend failures.
        traceback.print_exc()
        result = {"status": "error", "model_id": spec.model_id, "batch": args.batch,
                  "reason": f"{type(exc).__name__}: {exc}", **result_metadata()}

    write_result(args.out, result)
    return 0 if result["status"] in {"success", "gpu_unavailable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
