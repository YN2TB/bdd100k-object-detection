# Handoff

Updated: 2026-09-09

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
