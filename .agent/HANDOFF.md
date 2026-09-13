# Handoff

Updated: 2026-09-13

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
