# Handoff

Updated: 2026-09-15

## Completed checkpoint: full BDD100K retraining

The user superseded the fixed daytime/clear scope and explicitly authorized a
new experiment that retrained three models on every publicly labelled BDD100K
detection record. The completed ExecPlan is
`.agent/plans/archive/2026-09-14-full-bdd100k-retraining.md`.

Checkpoints 0 through 4 are complete. All 79,863 public labelled records were combined
without time/weather filtering and split with `seed=0` into 55,904 train, 15,973
validation, and 7,986 test images. All links resolve, the directories match their
manifests, and pairwise overlap is zero. The finalized roster is a hand-written
four-block grid detector (`simple-cnn`), a hand-written residual grid detector
(`complex-cnn`), and YOLO11s (baseline). Both CNNs must output objectness, class,
and box coordinates and use NMS. The earlier MobileNetV3 Faster R-CNN suggestion
was withdrawn because a two-stage R-CNN is not a simple CNN. Keep the
old subset and all old artifacts unchanged. Full source images and raw labels are
already local, so no download is needed. NVIDIA-SMI and a CUDA tensor kernel now
pass on the RTX 3060.

The superseded 62,877/6,986/10,000 link-only view is temporarily preserved at
`data/source_full_90_10_superseded`; it was never labelled or trained.

Full label conversion has completed: train has 1,029,446 retained boxes,
validation 295,515, and test 146,998. One exact duplicate validation annotation
was removed from both COCO and YOLO outputs. All ten classes occur in every split. No
production training had started at that checkpoint. The user then confirmed
object-level bounding boxes plus classes, so all three models use common COCO
metrics.

Checkpoints 2 and 3 are complete. `src/bddcv/cnn_detector.py` implements the
408,171-parameter SimpleCNN and 11,190,123-parameter residual ComplexCNN with a
shared five-slot grid head, loss, class-aware NMS, and COCO export.
`scripts/train_cnn.py` trains either model and atomically saves best/last complete
resume state. Unified prediction and evaluation target the held-out test split.
All 100 tests pass with one expected historical RF-DETR environment skip. Both
CNNs completed CPU smoke epochs; SimpleCNN resume reached epoch 2. Full numeric
label verification passed with zero mismatches and the test montage was visually
correct. README and `docs/full-data-experiment.md` contain current commands.

GPU smoke runs passed sequentially with batch 16 for SimpleCNN, batch 8 for
ComplexCNN, and batch 16 for YOLO11s. YOLO's smoke peaked at about 4.1 GB
allocated VRAM. The full label verifier passes 1,471,959 boxes with zero
mismatches after common deduplication. A sequential production queue was started
too early at 2026-09-15 09:44 +07, before commit/push. It is stopped and hourly
monitoring is paused. No epoch checkpoint was produced, so that attempt is not
valid experiment evidence. It did reveal that custom validation created a 439
MiB prediction JSON and reached about 10.7 GiB process memory. Fix and validate
that issue, commit and push the passing code, and only then start a clean queue.
The repair caps every validation/test export at 100 detections per image, matching
COCO `maxDets=100`. A full 15,973-image validation smoke passed in 108 seconds
with 6,834,448 KiB peak RSS.
The full 100-test suite, compilation, CLI help, shell syntax, and diff checks pass.
The user explicitly approved the configured GitHub destination and commit
`d5e4f38` was pushed to `origin/codex/add-model-profiling`. The publication gate
is satisfied. Publication-state commit `8d36d3d` was also pushed. A clean queue
started only afterward at 2026-09-15 10:13 +07. The invalid earlier artifact is
preserved separately under
`runs/smoke_full/aborted-before-push-simple-cnn-20260915` and must not be used in
results.

SimpleCNN completed 30/30 epochs without a restart at 2026-09-15 13:34 +07.
Recorded epoch time totals 12,014 seconds (3 h 20 min), with 400 seconds/epoch
on average; its best validation mAP50-95 is 0.002768 at epoch 19. ComplexCNN
started next; at an intermediate checkpoint it had 13/30 completed checkpoints
at 15:20 +07 while finishing epoch 14. It averaged about 462 seconds/epoch and
its best validation score at that point was 0.019130 at epoch 13.

ComplexCNN completed 30/30 without a restart at 17:25 +07. Recorded epoch time
totals 13,850 seconds (3 h 51 min), averaging 462 seconds/epoch; its best
validation mAP50-95 is 0.019779 at epoch 23. YOLO11s completed 30/30 in one
attempt at 22:48 +07, taking 19,299 seconds (5 h 23 min); its best native
validation mAP50-95 is 0.29102 at epoch 30. The queue then predicted and
centrally evaluated every best checkpoint and reached `COMPLETE` at 22:53 +07.

Final COCO test mAP50-95 / mAP50 / mAP75 / AR@100 values are:
SimpleCNN 0.0027 / 0.0107 / 0.0011 / 0.0147, ComplexCNN 0.0187 / 0.0585 /
0.0076 / 0.0651, and YOLO11s 0.2761 / 0.4867 / 0.2631 / 0.3644. All predictions
cover 7,986 test image IDs, use valid category IDs and finite boxes/scores, and
contain at most 100 detections per image. Unit tests, Python compilation, and
diff whitespace checks passed after documentation. The final results are in
`docs/full-data-experiment.md`; the hourly automation is retired because no run
remains active.

## Completed checkpoint: midterm pipeline simplification

The user requested a deliberately simple undergraduate midterm workflow. The
completed plan is `.agent/plans/archive/2026-09-14-midterm-pipeline-simplification.md`.
Do not introduce a CLI framework, config hierarchy, CI, serving layer, or new
training abstraction.

All four checkpoints are complete. `scripts/prepare_data.py` now rejects manifest drift
before subset writes and extracts raw labels with bounded atomic copies.
`scripts/build_labels.py` validates duplicate/missing manifest-to-raw coverage
and image presence before writes. `scripts/verify_labels.py` checks every COCO
image, including empty-label images, and returns failure status. README now
presents the five-step course pipeline and marks profiling/monitoring/migration
as optional.

Four focused tests pass. The full main suite passes 95 tests with one expected
RF-DETR-environment skip. Python compilation passes. Real label verification
passes for 34,096 val and 239,901 train boxes with 0.006 px maximum deviation;
the montage was inspected and boxes align visually. SHA-256 values for both
tracked manifests and `docs/model-ranking.md` were identical before and after.

Read-only preflight accepted exact manifest/raw/image coverage for both splits.
The existing official YOLO11s prediction evaluated through the centralized path
at reliable-class mAP50-95 0.2851, matching the published ranking. No prediction,
label, training, or production artifact was regenerated.

Checkpoint 5 is complete. Evaluation output now shows mAP50-95, mAP50, mAP75,
AP by object size, AR@100, reliable-class mAP, and per-class AP. The official
YOLO11s prediction still reports overall mAP50-95 0.2566, reliable-class mAP
0.2851, and AR@100 0.3537. Do not add custom threshold-dependent accuracy/F1,
regenerate predictions, or launch GPU work.

Checkpoint 6 is complete. The final main suite passed 95 tests with one expected
RF-DETR-environment skip; compilation and `git diff --check` passed. Protected
manifest and ranking hashes remain unchanged. The next repository task remains
the independent acceptance review for the older six-model plan.

Checkpoint 7 is complete. All six saved prediction JSON files were re-evaluated
through `bddcv.evaluation` on 2026-09-14 without regenerating predictions. The
full essential-metrics table, now including AR@100, is in
`docs/model-ranking.md`; the reliable-class ordering is unchanged.

## Latest work

The six-model plan is implemented in the working tree on
`codex/add-model-profiling`; it is being committed and pushed before GPU work. The previous
handoff incorrectly said implementation had not started. The active plan records
prior authorization for implementation and delegation. On 2026-09-09 the user
explicitly authorized training the remaining models after this code is committed
and pushed to GitHub.

Initial verification: `.venv/bin/python -m unittest discover -s tests -v` passed
52 tests. The default shell Conda Python lacks Torch; use `.venv/bin/python`.
Both `.venv` and `.venv-rfdetr` pass `python -m pip check`. RF-DETR is isolated,
and `validate_adapter()` reports exact split/category membership and byte matches
for 12,454 training and 1,764 validation images. Source manifest hashes match
`docs/HANDOFF_3060.md`; filenames/counts/disjointness also pass.

Saved sweeps exist under `runs/profile/official/`, but independent review found
compute-only timing excludes loader wait, so their selections are superseded.
See `docs/model-profiling-status.md` for historical values and limitations.
Review repairs were applied for timing/selection, native RF-DETR seed,
supervisor metadata/resume compatibility, YOLO early-stop and dataset locking,
Faster R-CNN validation batch alignment, and prediction input coverage/resolution.

The saved RT-DETR baseline loading/prediction receipt reports pass. On 2026-09-09,
all four protected file hashes were recomputed and still match its receipt.
No baseline retraining or centralized evaluation was performed.

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

Final repair verification: `.venv/bin/python -m unittest discover -s tests -v`
passed 77 tests (exit 0); `git diff --check` and
`.venv/bin/python -m compileall -q scripts src/bddcv` passed (exit 0).
The final independent reviewer and two workers hit usage limits. The primary
completed remaining checkpoint provenance and timing fixes locally, but final
independent review remains pending. YOLO now publishes provenance at every native
checkpoint save and refuses unsafe unmarked legacy resume. Prediction reading
compatibility is unchanged. No commits or GPU training were performed.

Next: finish independent review, then re-run corrected GPU profiling in a new output directory when training/probe
execution resumes. Old profiling outcomes must not be treated as accepted results.
Five full-data two-epoch stop/resume smoke validations remain outstanding; the
existing `runs/smoke/official/yolo11s` directory has no completed checkpoint.
Do not launch 50-epoch runs, retrain RT-DETR-l, regenerate manifests, or rewrite
historical logs. The active plan remains in progress until measured acceptance.

## Latest continuation: RAM-cache implementation

Implemented conditional RAM-cache trials, explicit `--cache none|ram` and
`--prefetch 2|4` options in the training entrypoints/supervisor, resume locking,
copy-on-read decoded caching, sampled peak host RAM, and strict device-memory
selection evidence. Timing schema is now 3; older profiles remain ineligible.
Fixed supervisor sibling imports from another working directory. No GPU runs.

Verification: 87 main-environment tests discovered, 86 passed and one isolated
RF-DETR integration skipped; that test passed separately under `.venv-rfdetr`.
Compilation, CLI help and diff whitespace checks passed. See
`docs/model-profiling-status.md` for commands and limitations. Remaining work is
independent review and actual GPU/ full-data acceptance, including measurement of
resolved recipes, precision, tensor shapes and ETA. Full-data cache fit is not
established by a 512-image cache trial.

## User reporting requirement

When new training starts, enable Vietnamese reports every hour: model/epoch/%
progress, elapsed/recent epoch time, remaining time and completion ETA, GPU
utilization/VRAM/peak/temperature/power, plus restarts or errors. Start the per-run
GPU sampler and reuse the paused previous hourly schedule with current paths and
this task as destination. Do not reactivate the historical RT-DETR run prompt.
Activate hourly reporting when the first new GPU run is launched.

## Pre-push verification for model expansion

Before publishing the branch, `.venv/bin/python -m unittest discover -s tests -v`
discovered 87 tests: 86 passed and the isolated RF-DETR integration test skipped.
That integration test passed under `.venv-rfdetr`. Both environments passed
`pip check`; source/adapter counts, filenames, split disjointness, categories and
image bytes passed validation. Python compilation, CLI help, and `git diff
--check` passed. The RTX 3060 was visible with 12,288 MiB VRAM. No new GPU workload
had been launched at this checkpoint.

## Corrected GPU profiling

Timing-schema-3 profiling completed under `runs/profile/official-v3/` for all
five unfinished models. Selected batch/workers/prefetch are YOLO11s 32/4/2,
YOLO11m 16/8/4, YOLO26s 16/4/4, Faster R-CNN 16/4/2, and RF-DETR Small 8/4/2.
All use `cache=none`; conditional cache triggers were not met. Peak sampled VRAM
was 81.67%, 80.68%, 57.91%, 67.05%, and 74.97%, respectively. YOLO26s batch 32
was excluded at 96.61% VRAM. The user superseded the earlier 90%/3% policy with
speed-first selection up to 95% VRAM and a 1% throughput tie.

The initial Faster R-CNN sweep failed because `data_wait` was constructed after
its first use. A regression test reproduced the ordering defect; moving the
buffer initialization before `TimedLoader` fixed it, and all FRCNN batches and
worker variants then completed. Before any smoke or production training, verify,
commit, and push this supplemental policy/probe/monitor change.

## Active training queue

Commit `0829aa9` (`Prioritize training throughput in GPU profiles`) was pushed to
`origin/codex/add-model-profiling` before training. The sequential queue is now
running independently as user service `bddcv-remaining-models-v2.service` from
`runs/control/remaining-models-queue.sh`. Current state and event history are in
`runs/control/remaining-models-current.tsv` and
`runs/control/remaining-models-events.tsv`; per-stage launcher and GPU telemetry
are under `runs/control/stage-logs/`.

The first detached attempt was cleaned up by the tool environment. The first
systemd attempt then failed before training because the queue created an output
`logs/` directory ahead of the trainer's fresh-output guard. Its evidence was
moved to `runs/control/failed-launch-yolo11s-20260909-0930/`. The queue now keeps
launcher/monitor files under `runs/control`, leaving fresh run outputs empty.
The v2 service was verified active with YOLO11s smoke epoch 1 on the GPU at
batch/workers/prefetch 32/4/2. Initial telemetry showed 100% utilization,
9,581/12,288 MiB device memory, 62 C, and 148.77/170 W.

The existing hourly automation was updated and enabled as
`BDD CV: báo train mỗi giờ`, targeting this task. It monitors the queue and
reports progress/ETA/GPU telemetry in Vietnamese, pauses on failure or verified
completion, never launches duplicates, and never retrains completed RT-DETR-l.

## RF-DETR smoke checkpoint repair

The first RF-DETR smoke exposed a native checkpoint naming mismatch: with
`checkpoint_interval=1`, RF-DETR 1.10.1 suppresses `last.ckpt` and writes only
`checkpoint_0.ckpt`, so the epoch-1 watcher could not stop the run. The queue and
automation were paused before production. The failed smoke evidence is preserved
under `runs/control/failed-rfdetr-checkpoint-contract-20260909-1507/`.

Commit `8be9cfe` changes the interval to 2, enabling full-state `last.ckpt` every
epoch plus numbered archives every two epochs. All 89 main tests passed with one
isolated integration skip; the isolated RF-DETR cache integration test passed in
`.venv-rfdetr`. The commit was pushed before GPU execution resumed. Queue v3 is
running independently as `bddcv-remaining-models-v3.service`, starting directly
at fresh RF-DETR epoch-1 smoke; earlier successful smoke runs were not repeated.
The hourly automation is active again and references the v3 service.

## RF-DETR native stop repair and queue v4

Queue v3 exposed a second, separate smoke-control defect. Its full-state
`last.ckpt` was valid, but the background watcher sent `SIGINT` and PyTorch
Lightning did not stop at the requested epoch; the run continued through epoch 6.
The later `DataLoader worker ... terminated` status was caused by deliberately
stopping the service after detecting this overrun. No production stage started.
Evidence is preserved under
`runs/control/failed-rfdetr-signal-stop-20260909-2053/`.

Commit `384b161` replaces the signal watcher with a native Lightning callback
that sets `trainer.should_stop` after validation at the requested epoch and skips
Lightning's pre-training sanity check. The callback is appended after RF-DETR's
checkpoint callbacks, so `last.ckpt` is published before the clean stop. The
main suite passed 91 tests with one isolated integration skip, and the callback
was confirmed to inherit Lightning's `Callback` in `.venv-rfdetr`. The commit was
pushed before GPU execution.

Queue v4 is active as `bddcv-remaining-models-v4.service`, starting from a fresh
RF-DETR epoch-1 smoke at 2026-09-10 10:49 Asia/Ho_Chi_Minh. Its initial telemetry
showed 100% GPU utilization and 9,207/12,288 MiB VRAM. On successful epoch-1 and
resume-to-epoch-2 acceptance, the same queue automatically begins the five
authorized production runs in order. The hourly automation is active and now
references queue v4 and commit `384b161`.

## Production completion and centralized ranking

Queue v4 completed all remaining smoke/resume checks and all five authorized
50-epoch production runs on 2026-09-11. RT-DETR-l remains the preserved baseline;
it was not retrained. The completion sequence and elapsed training times were
YOLO11s 1.83 h, YOLO26s 2.32 h, YOLO11m 4.00 h, Faster R-CNN 10.52 h, and
RF-DETR Small 8.03 h. `runs/control/remaining-models-events.tsv` ends with a
`COMPLETE` queue record.

Centralized evaluation completed on 2026-09-13 for all six best checkpoints.
All predictions use the same 1,764-image validation split, confidence 0.001,
maximum 300 exported detections per image, and `bddcv.evaluation`. The headline
metric excludes only the sparse `train` class. Ranking: RT-DETR-l 0.3612,
YOLO11m 0.3243, RF-DETR Small 0.3025, YOLO26s 0.3008, YOLO11s 0.2851, and Faster
R-CNN 0.2623. See `docs/model-ranking.md` and
`runs/evaluation/official-v1/metrics.json`. Prediction coverage, IDs, categories,
finite values, checkpoint hashes, metadata, rank ordering, and the nine-class
policy passed validation. Final independent code review remains outstanding.
