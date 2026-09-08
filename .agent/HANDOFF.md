# Handoff

Updated: 2026-09-08

## Latest work

Artifact organization and the detailed README were committed as `51fd8dc` and
pushed directly to `origin/main`. The current branch is
`codex/add-model-profiling`, created from that exact commit.

The next phase is fully specified in
`.agent/plans/active/2026-09-08-six-model-profiling.md`. It adds YOLO11m,
YOLO26s, and RF-DETR Small, profiles five unfinished models for the RTX 3060
12 GB, and validates two-epoch stop/resume smoke runs. No implementation or
training for this phase has started.

## Artifact state

39 legacy files migrated with hashes preserved. Receipt:
`runs/maintenance/migration-b4b9d2f7126747dc848ed7b0e2135191.json`.
Production RT-DETR is now `runs/train/rtdetr-l/`, with 50 epoch rows and the
historical COMPLETE log. Do not restart this completed experiment.
Pretrained weight: `weights/rtdetr-l.pt`. Legacy smoke: `runs/smoke/smoke_rtdetr/`.
A separate reduced-data GPU integration smoke completed in 106.1 seconds at
`runs/smoke/layout_check/`; this is not a scientific comparison run.

## Preservation and limits

The machine-specific `configs/bdd_source.yaml` path and manifest line endings were
already present before artifact-layout work and are included in commit `51fd8dc`.
Historical experiment snapshot and migrated checkpoints/logs remain byte-identical.
Existing metadata may refer to old paths; use wrapper `--out` for migrated resume,
not raw old commands.
Tests cover routing and migration; independent reviewer was unavailable due to
usage limits. Detailed logic defects remain deferred by user-approved scope.

## Next step

Wait for the user's command. When authorized, implement the active plan in order,
starting with model registry and dependency compatibility checks. Do not launch
50-epoch runs, retrain RT-DETR-l, regenerate manifests, or rewrite historical logs.
