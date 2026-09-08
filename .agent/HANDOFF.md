# Handoff

Updated: 2026-09-08

## Latest work

Implemented the approved artifact organization and detailed README on branch
`codex/organize-artifacts`. No commit or push. See `README.md` for the complete
current tree, commands and output rules; see `docs/reviews/2026-09-08-project-audit.md`
for findings and validation. `.agent/TODO.md` contains only deferred logic work.

## Artifact state

39 legacy files migrated with hashes preserved. Receipt:
`runs/maintenance/migration-b4b9d2f7126747dc848ed7b0e2135191.json`.
Production RT-DETR is now `runs/train/rtdetr-l/`, with 50 epoch rows and the
historical COMPLETE log. Do not restart this completed experiment.
Pretrained weight: `weights/rtdetr-l.pt`. Legacy smoke: `runs/smoke/smoke_rtdetr/`.
A separate reduced-data GPU integration smoke completed in 106.1 seconds at
`runs/smoke/layout_check/`; this is not a scientific comparison run.

## Preservation and limits

`configs/bdd_source.yaml` and both manifests were dirty before this task and have
been preserved; their Git diff is not introduced here. Historical experiment
snapshot and migrated checkpoints/logs remain byte-identical. Existing metadata
may refer to old paths; use wrapper `--out` for migrated resume, not raw old commands.
Tests cover routing and migration; independent reviewer was unavailable due to
usage limits. Detailed logic defects remain deferred by user-approved scope.

## Next step

Review the diff and select a backlog item when requested. Do not auto-commit,
push, rerun production training, regenerate manifests or rewrite historical logs.
