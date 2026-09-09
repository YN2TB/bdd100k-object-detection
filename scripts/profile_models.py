"""Profile registry candidates in fresh subprocesses and select a batch."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bddcv.paths import RUNS_DIR, resolve_output  # noqa: E402
from bddcv.profiling import (  # noqa: E402
    PROFILE_SCHEMA_VERSION, TIMING_SCHEMA, TIMING_SCHEMA_VERSION,
    ProfileController, should_try_ram_cache, accept_ram_cache,
    make_probe_manifest,
    select_profile,
)
from bddcv.registry import ModelResolutionError, get_model_spec, list_models  # noqa: E402


UNFINISHED_MODELS = tuple(
    spec.model_id for spec in list_models() if spec.model_id != "rtdetr-l"
)


def load_existing_outcome(path: Path):
    """Return one valid serialized outcome, or None for a missing/broken file."""
    from bddcv.profiling import ProfileOutcome
    try:
        outcome = ProfileOutcome.from_dict(json.loads(path.read_text(encoding="utf-8")))
        return outcome if outcome.timing_compatible else None
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def load_batch_outcomes(out: Path, model_id: str):
    """Load prior batch-sweep outcomes for a loader-only continuation."""
    from bddcv.profiling import ProfileOutcome
    values = []
    for path in sorted((out / model_id).glob("batch-*/outcome.json")):
        value = load_existing_outcome(path)
        if value is not None:
            values.append(value)
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", action="append", dest="model_ids")
    parser.add_argument("--out", type=Path, default=None,
                        help="default: runs/profile")
    parser.add_argument("--source", type=Path,
                        default=Path("data/source_daytime_clear"))
    parser.add_argument("--probe-manifest", type=Path, default=None)
    parser.add_argument("--phase", choices=("all", "batch", "workers"), default="all")
    parser.add_argument("--force", action="store_true",
                        help="rerun completed loader variants instead of reusing artifacts")
    args = parser.parse_args()
    out = resolve_output(args.out, RUNS_DIR / "profile")
    models = args.model_ids or list(UNFINISHED_MODELS)
    manifest = args.probe_manifest or (out / "probe_images.txt")
    try:
        make_probe_manifest(args.source, manifest)
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"profile probe manifest unavailable: {exc}", file=sys.stderr)
        return 2

    try:
        specs = [get_model_spec(model_id) for model_id in models]
    except ModelResolutionError as exc:
        parser.error(str(exc))
    controller = ProfileController(out)
    outcomes = []
    if args.phase != "workers":
        for spec in specs:
            for batch in spec.batch_candidates:
                outcomes.append(controller.run_candidate(
                    spec.model_id, batch, probe_manifest=manifest, source=args.source,
                    warmup=10, steps=30, validation_steps=10,
                ))
    else:
        for spec in specs:
            outcomes.extend(load_batch_outcomes(out, spec.model_id))

    batch_selected = {
        model_id: select_profile(item for item in outcomes if item.model_id == model_id)
        for model_id in models
    }
    loader_outcomes = []
    if args.phase != "batch":
        for spec in specs:
            selected = batch_selected[spec.model_id]
            if selected is None:
                continue
            # The batch sweep already measured workers=0, prefetch=2.
            loader_outcomes.append(selected)
            for workers in (4, 8):
                variant = f"workers-{workers}-prefetch-2"
                existing = None if args.force else load_existing_outcome(
                    controller.candidate_output(spec.model_id, selected.batch, variant) / "outcome.json"
                )
                loader_outcomes.append(existing or controller.run_candidate(
                    spec.model_id, selected.batch, variant=variant,
                    probe_manifest=manifest, source=args.source, workers=workers,
                    prefetch=2, warmup=10, steps=30, validation_steps=10,
                ))
            nonzero = [
                item for item in loader_outcomes
                if item.model_id == spec.model_id
                and int(item.metadata.get("workers", 0)) > 0
            ]
            best_nonzero = select_profile(nonzero)
            if best_nonzero is not None:
                workers = int(best_nonzero.metadata["workers"])
                variant = f"workers-{workers}-prefetch-4"
                existing = None if args.force else load_existing_outcome(
                    controller.candidate_output(spec.model_id, selected.batch, variant) / "outcome.json"
                )
                loader_outcomes.append(existing or controller.run_candidate(
                    spec.model_id, selected.batch, variant=variant,
                    probe_manifest=manifest, source=args.source, workers=workers,
                    prefetch=4, warmup=10, steps=30, validation_steps=10,
                ))
    summary = {
        "profile_schema_version": PROFILE_SCHEMA_VERSION,
        "timing_schema": TIMING_SCHEMA,
        "timing_schema_version": TIMING_SCHEMA_VERSION,
        "probe_manifest": str(manifest),
        "outcomes": [item.to_dict() for item in outcomes],
        "batch_selected": {},
        "loader_outcomes": [item.to_dict() for item in loader_outcomes],
        "selected": {},
        "cache_trials": {},
    }
    for model_id in models:
        batch_choice = batch_selected[model_id]
        summary["batch_selected"][model_id] = batch_choice.to_dict() if batch_choice else None
        final_pool = [item for item in loader_outcomes if item.model_id == model_id]
        selected = select_profile(final_pool) if final_pool else batch_choice
        if selected is not None and args.phase != "batch" and should_try_ram_cache(selected):
            workers = int(selected.metadata.get("workers", 0))
            prefetch = int(selected.metadata.get("prefetch", 2))
            variant = f"workers-{workers}-prefetch-{prefetch}-cache-ram"
            existing = None if args.force else load_existing_outcome(
                controller.candidate_output(model_id, selected.batch, variant) / "outcome.json"
            )
            trial = existing or controller.run_candidate(
                model_id, selected.batch, variant=variant, cache="ram",
                probe_manifest=manifest, source=args.source, workers=workers,
                prefetch=prefetch, warmup=10, steps=30, validation_steps=10,
            )
            retained = accept_ram_cache(selected, trial)
            summary["cache_trials"][model_id] = {
                "retained": retained, "outcome": trial.to_dict(),
            }
            if retained:
                selected = trial
        else:
            summary["cache_trials"][model_id] = {
                "retained": False, "reason": "trigger not met or batch-only phase",
            }
        summary["selected"][model_id] = selected.to_dict() if selected else None
    path = out / "summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    # Unavailable hardware is a valid outcome, but malformed or unexpected
    # failures remain a non-zero command result for automation.
    checked = outcomes + loader_outcomes
    return 0 if checked and all(item.status.value != "error" for item in checked) else 1


if __name__ == "__main__":
    raise SystemExit(main())
