# Full BDD100K retraining plan

Updated: 2026-09-15

## Goal

Replace the filtered daytime/clear experiment with a new, separately stored
experiment using every publicly labelled BDD100K detection record. Train three
models from scratch and report final metrics on a held-out test split.

The existing `data/source_daytime_clear`, checkpoints, predictions, evaluations,
and ranking remain unchanged as historical artifacts.

## Fixed data policy

- Combine all 69,863 official training records and all 10,000 official
  validation records without filtering time-of-day or weather.
- Shuffle all 79,863 filenames deterministically with `seed=0` and split them
  70%/20%/10% for train/validation/test.
- Do not use the official 20,000-image test split for metrics because its labels
  are not public.
- Preserve `DET_CLASSES`, category aliases, box filtering, YOLO `i` to COCO
  `i + 1`, and filename-based image IDs.

Expected logical split sizes after generation:

| split | images | purpose |
|---|---:|---|
| train | 55,904 | parameter fitting |
| val | 15,973 | checkpoint selection |
| test | 7,986 | final metrics only |
| total | 79,863 | all public labelled records |

## Simplicity constraints

- Keep the existing flat `scripts/` and `src/bddcv/` layout.
- Do not add a framework, config schema, CI, API, or report generator.
- Reuse the existing trainers and centralized evaluator.
- Add only the minimum data-root/split options needed to keep old artifacts
  usable and point new runs at `data/source_full`.
- Update README and state Markdown at every completed checkpoint.

## Checkpoints

### 0. Authorization and plan — complete

- User explicitly authorized full-data conversion and retraining from scratch on
  2026-09-14. The final primary roster is one simple CNN detector, one complex
  CNN detector, and one YOLO baseline.
- Full images and raw labels are present locally under `data/kagglehub` and
  `data/labels_raw`; no download or dependency install is planned.
- GPU preflight and bounded smoke runs now pass on the RTX 3060.

### 1. Full-data preparation — complete

- Generate protected train/val/test manifests before creating dataset links.
- Reject manifest drift before modifying the generated dataset.
- Link labelled images into `data/source_full/images/{train,val,test}` without
  duplicating image bytes.
- Confirm counts, disjoint filenames, source coverage, and missing files.
- Update this plan, handoff, TODO, and decisions.

The first generated view used an earlier split policy and passed the checks below,
but was superseded before label conversion or training. It will be preserved
temporarily while the authorized 70/20/10 view is generated.

Final evidence:

- Generated 55,904 train, 15,973 val, and 7,986 test image links.
- Every link resolves, each directory exactly matches its manifest, pairwise
  overlap is zero, and the union contains 79,863 filenames.
- `scripts/prepare_data.py` has no time-of-day or weather filtering.
- Final manifest SHA-256 values are
  `0c941e26478196a8baf30436ebd928ee3d26a12676d313a3576a5b9796655ef0`,
  `2c6b09fd0d65345c114e8ec5e1521ee45e4625794b2d1dccc1afb9bb06f94a81`,
  and `2aa547fada11bc50e70b890042bd8cb75c17fe08ebe27c116f6d247bc63e20d1`
  for train, val, and test respectively.
- The superseded generated view is temporarily retained at
  `data/source_full_90_10_superseded`; it contains links only, not duplicate image
  bytes, and is outside the new pipeline.

Superseded evidence:

- `scripts/prepare_data.py` does not inspect time-of-day or weather attributes.
- Generated 62,877 train, 6,986 val, and 10,000 test links under
  `data/source_full`; all 79,863 links resolve.
- Pairwise filename overlap is zero and the union contains 79,863 names.
- Manifest SHA-256 values are `19043e709a1369595d365914ac6d9c5911ad4f1e0ce2c81c3146bd29a5c102d5`,
  `42858af7b38e0f5b392967eb4102262635d225caab5171eb853c68b0cf774820`,
  and `b18740be50a93685aa9c931bde376e358e1f7149a335adf3fd4645112f5d7fdb`
  for train, val, and test respectively.

### 2. Labels and configuration — complete

- Generate YOLO labels and COCO annotations for train, val, and test.
- Point the primary config and main pipeline at `data/source_full` while keeping
  explicit paths available for historical evaluation.
- Verify empty labels, image coverage, class IDs, box geometry, and split counts.
- Update README and state Markdown.

Progress evidence:

- YOLO and COCO labels were generated successfully for all three final splits.
- Train: 55,904 images and 1,029,446 retained boxes.
- Validation: 15,973 images and 295,515 retained boxes after removing one exact
  duplicate source annotation from both output formats.
- Test: 7,986 images and 146,998 retained boxes.
- All images are 1280x720; no box required boundary clamping. The existing
  minimum-side rule removed 315/76/46 tiny boxes in train/val/test.
- Primary config and model roster are confirmed for bounding-box detection.

Final implementation:

- The user confirmed bounding-box detection. The primary config, label verifier,
  unified predictor, and evaluator now use `data/source_full` and its held-out test.
- `src/bddcv/cnn_detector.py` contains both hand-written CNNs, their shared grid
  head/loss, class-aware NMS, and COCO export.
- `scripts/train_cnn.py` is the single short trainer for both CNNs, with atomic
  best/last checkpoints and complete resume state.
- Five slots per grid cell reduce dense same-cell unassigned boxes from about
  4.0% with three slots to about 0.6%.
- README and `docs/full-data-experiment.md` document the current simple pipeline.

### 3. Focused validation — complete

- Add only small tests for deterministic splitting, split disjointness, and the
  official-validation-to-test mapping.
- Run the existing unit suite, compile check, label verification, and one short
  CPU/read-only data smoke where practical.
- Do not start three production runs until these checks pass.
- Update all execution-state Markdown.

Evidence:

- 100 unit tests pass with one expected isolated historical RF-DETR skip.
- Four focused custom-CNN tests cover the primary roster, relative model size,
  common output shape, finite loss, and valid COCO box decoding.
- SimpleCNN and ComplexCNN each completed a bounded CPU epoch. SimpleCNN also
  resumed from its epoch-1 checkpoint and completed epoch 2.
- Label verification passed 1,029,446 train, 295,515 validation, and 146,998 test
  boxes with zero mismatches and 0.006 px maximum deviation.
- The generated test montage was inspected and boxes align with traffic objects.

### 4. GPU preflight and smoke — complete

- Require a successful CUDA kernel check and NVIDIA-SMI telemetry.
- Re-profile only where the fivefold dataset-size change makes the saved recipe
  unsafe; otherwise preserve model hyperparameters and input resolution.
- Run one bounded smoke/resume check per backend, sequentially.
- Never overwrite historical runs.
- Update all execution-state Markdown.

Evidence:

- NVIDIA-SMI reports an RTX 3060 with 12,288 MiB, and a real CUDA tensor kernel
  passed with Torch 2.11.0+cu128.
- SimpleCNN passed a GPU epoch with batch 16; ComplexCNN passed with batch 8.
- YOLO11s passed with batch 16, 640 input, and about 4.1 GB peak allocated VRAM.
- The YOLO smoke exposed one duplicate raw box. Label generation now removes
  exact duplicates before both formats are written, its focused regression test
  passes, and full numeric verification again reports zero mismatches.
- After this repair, the full suite passed 100 tests with one expected isolated
  RF-DETR skip; compilation, queue shell syntax, and `git diff --check` passed.

### 5. Three fresh training runs — running after publish

- Train a hand-written four-block grid detector (`simple-cnn`), a hand-written
  residual grid detector (`complex-cnn`), and YOLO11s. Both CNNs share the same
  objectness/class/box prediction format and NMS so architecture depth is the
  main controlled difference.
- Use separate output roots under `runs/train_full/` and run one GPU workload at
  a time.
- Restore hourly Vietnamese progress reporting when the queue starts.
- Preserve atomic/native checkpoint and resume behavior.
- Update the plan and handoff after every model completes.

Initialization policy: both custom CNNs start from random weights. YOLO starts a
new epoch-1 run from the standard COCO-pretrained `yolo11s.pt` baseline; no old
BDD100K checkpoint is resumed. This transfer-learning difference is disclosed in
the report.

Pre-launch evidence and correction:

- A production queue was started at 2026-09-15 09:44 +07 before the validated
  code was committed and pushed. This violated the user's publication gate and
  the queue is stopped; no run from that attempt is valid experiment evidence.
- Initial telemetry showed 92% GPU utilization, 4,077/12,288 MiB VRAM, 58 C,
  and about 90 W.
- SimpleCNN completed its epoch-1 training batches but produced no checkpoint;
  validation accumulated a 439 MiB prediction JSON and about 10.7 GiB process
  memory. Treat this as a failed pre-production probe and bound validation output
  before publishing.
- Required gate: fix the validation-memory issue, revalidate, commit and push the
  passing code, then start a clean sequential queue. Hourly monitoring remains
  paused until that new queue begins.
- Validation export is now capped at 100 detections per image, matching COCO
  `maxDets=100`. A worst-case full-validation smoke passed in 108 seconds with
  6,834,448 KiB maximum RSS and exit 0, down from the failed attempt's 10.7 GiB.
- Acceptance passed, local commit `d5e4f38` was created, and the user explicitly
  confirmed the configured GitHub remote. Commit `d5e4f38` was pushed successfully
  to `origin/codex/add-model-profiling`. No training ran before that push.
- The publication-state checkpoint `8d36d3d` was also pushed. Local HEAD and the
  remote branch matched before the clean production queue started at 2026-09-15
  10:13 +07.
- SimpleCNN is the active first stage. Initial clean-run telemetry showed 97% GPU,
  4,032/12,288 MiB VRAM, 63 C, and about 145 W. Hourly Vietnamese monitoring is
  active again.
- SimpleCNN completed all 30 epochs at 2026-09-15 13:34 +07 with no restart.
  Total recorded epoch time was 12,014 seconds (3 h 20 min), averaging 400 seconds;
  the best validation mAP50-95 was 0.002768 at epoch 19.
- ComplexCNN then started automatically. At 2026-09-15 15:20 +07 it had 13/30
  completed checkpoints and was finishing epoch 14. Its recent epoch time is
  about 462 seconds and its best validation mAP50-95 so far is 0.019130 at epoch
  13. Estimated ComplexCNN completion is around 17:25 +07.
- ComplexCNN completed 30/30 epochs at 2026-09-15 17:25 +07 with no restart.
  Total recorded epoch time was 13,850 seconds (3 h 51 min), averaging 462
  seconds; its best validation mAP50-95 was 0.019779 at epoch 23.
- YOLO11s started automatically and had 16/30 completed checkpoints at 20:25
  +07 (53.3%), while training epoch 17. Its best native validation mAP50-95 so
  far is 0.28133 at epoch 16, recent epochs take about 647 seconds, and estimated
  training completion is around 22:55 +07. Peak sampled VRAM is 6,440 MiB;
  no restart or error is recorded.

### 6. Test prediction and evaluation — pending

- Export all predictions on the same 7,986-image held-out test split.
- Evaluate every model through `bddcv.evaluation` with the same confidence and
  maximum-detection policy.
- Report COCO mAP50-95, mAP50, mAP75, AP by size, AR@100, reliable-class mAP,
  and per-class AP in one table.
- Do not select or tune checkpoints using test metrics.

### 7. Final documentation and review — pending

- Replace the README quick-start counts and explain the three splits plainly.
- Publish a separate full-data ranking without changing the historical ranking.
- Run final tests and independent acceptance review, then archive this plan.

## Current blocker

None. The clean sequential production run is active.
