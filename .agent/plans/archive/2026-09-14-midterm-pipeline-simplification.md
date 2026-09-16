# Midterm pipeline simplification

Status: complete.

## Goal

Keep the course workflow easy to explain, fix four data-integrity gaps, and make
the evaluator return the essential object-detection metrics.

## Scope

- Keep the flat `src/bddcv/` and `scripts/` structure.
- Keep existing trainers, checkpoints, outputs, experiments, and optional tools.
- Do not install packages, retrain models, or change official artifacts.
- Report standard COCO metrics only; do not add custom threshold-dependent
  accuracy or F1 calculations.

## Checkpoints

- [x] Checkpoints 1-4: data fixes, concise README, validation, and initial closeout.
- [x] Checkpoint 5: print/document mAP50-95, mAP50, mAP75, size AP, AR@100,
  reliable-class mAP, and per-class AP; verify against official predictions.
- [x] Checkpoint 6: final tests, state update, and re-archive.
- [x] Checkpoint 7: rerun centralized evaluation for all six saved prediction
  files and publish the essential metrics table, including AR@100.

## Acceptance evidence

- The main suite passed 95 tests with one expected isolated RF-DETR skip.
- Real label verification matched 34,096 val and 239,901 train boxes with a
  maximum 0.006 px deviation; the montage was inspected.
- Real manifest/raw/image input coverage passed for both splits.
- Existing official YOLO11s predictions evaluated at overall mAP50-95 0.2566,
  reliable-class mAP 0.2851, and AR@100 0.3537.
- All six official prediction files were re-evaluated successfully on 2026-09-14;
  the reliable-class ranking remains unchanged.
- Compilation and `git diff --check` passed; protected manifest and ranking
  hashes did not change.
