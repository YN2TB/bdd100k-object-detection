# Handoff

Updated: 2026-09-08

## Latest work

A repository-scoped Codex and Claude subagent architecture is specified in
`docs/superpowers/specs/2026-09-08-shared-subagents-and-skills-design.md` and is
awaiting final user review. It defines explorer, worker, validator, and reviewer
roles, with a mandatory confirmation gate before delegating difficult or long
tasks. No agent or skill configuration has been implemented.

Previous completed work remains unchanged:

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

Ask the user to review the shared subagent design. After approval, create a focused
implementation plan before adding agent and skill files. The separate six-model
profiling plan remains approved but must not start without a specific user command.
Do not launch 50-epoch runs, retrain RT-DETR-l, regenerate manifests, or rewrite
historical logs.
