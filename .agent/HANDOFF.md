# Handoff

Updated: 2026-10-05

## Project

This repository is a Deep Learning course midterm on image reading (đọc ảnh). It
compares two hand-written CNN detectors (`simple-cnn` and `complex-cnn`) with a
YOLO11s baseline on all publicly labelled BDD100K detection images. All three
models produce bounding boxes and are scored with the same COCO evaluator.

## Current state: complete

The full-data experiment is finished and published. Its plan is
`.agent/plans/archive/2026-09-14-full-bdd100k-retraining.md`. No training, queue,
or hourly automation is running, and no plan is active.

- Data: 79,863 labelled records, with no time or weather filter, split with
  `seed=0` into 55,904 train, 15,973 validation, and 7,986 test images. The
  splits contain 1,029,446, 295,515, and 146,998 boxes. Membership is fixed by
  `data/source_full/*_images.txt`.
- Training: each model ran 30 epochs on the RTX 3060 without restarts, finishing
  on 2026-09-15. Totals were SimpleCNN 3 h 20 min, ComplexCNN 3 h 51 min, and
  YOLO11s 5 h 23 min.
- Test scores used the best validation checkpoints, evaluated once through
  `bddcv.evaluation` with at most 100 detections per image:

  | Model | mAP50-95 | mAP50 | mAP75 | AR@100 |
  |---|---:|---:|---:|---:|
  | SimpleCNN | 0.0027 | 0.0107 | 0.0011 | 0.0147 |
  | ComplexCNN | 0.0187 | 0.0585 | 0.0076 | 0.0651 |
  | YOLO11s | 0.2761 | 0.4867 | 0.2631 | 0.3644 |

- Artifacts are in `runs/train_full/`, `runs/predictions_full/`, and
  `runs/evaluation_full/`. The full write-up and per-class AP are in
  `docs/full-data-experiment.md`. The commands are in `README.md`.
- The invalid run that was started before the push is preserved under
  `runs/smoke_full/aborted-before-push-simple-cnn-20260915`. Never use it in
  results.
- The superseded 90/10 link-only view (`data/source_full_90_10_superseded`) is no
  longer present under `data/`.

## Historical context (do not rewrite)

- Daytime/clear subset and six-model ranking: `docs/model-ranking.md`,
  `docs/model-profiling-status.md`, and
  `.agent/plans/archive/2026-09-08-six-model-profiling.md`. Its open review items
  were closed as superseded on 2026-10-05.
- The RT-DETR assignment is in `docs/HANDOFF_3060.md`. Older experiment notes are
  in `docs/experiments.md`.
- The detailed handoff narrative from before 2026-10-05 is in git history
  (`git show 8c0c8f3:.agent/HANDOFF.md`).

## Standing user requirements

- Keep the course pipeline simple: prepare/build, verify, train, predict,
  evaluate. Do not add a CLI framework, config hierarchy, CI, or serving layer.
- Commit and push code before launching any GPU training. While any new training
  runs, send hourly reports in Vietnamese covering model, epoch, % progress,
  elapsed and recent epoch time, ETA, GPU utilization, VRAM, peak memory,
  temperature, power, and any restarts or errors.
- Do not tune checkpoints or thresholds on the test split.

## Verification

When the experiment was published, the unit tests, Python compilation, and diff
whitespace checks passed. Run `python -m unittest discover -s tests -v` after any
code change.

## Next step

None pending. Wait for the user's next request, such as report writing or
analysis.
