# Durable decisions

## 2026-09-15: Publish held-out full-data results as the current comparison

The final midterm comparison uses the best validation-selected checkpoint from
each 30-epoch run and scores it once on the fixed 7,986-image test split through
`bddcv.evaluation`. The headline test mAP50-95 values are 0.0027 for SimpleCNN,
0.0187 for ComplexCNN, and 0.2761 for YOLO11s. YOLO11s is the final baseline
winner; ComplexCNN improves over SimpleCNN but neither custom detector is close
to the transfer-learned baseline.

Keep these test results separate from trainer-native validation metrics. Do not
tune checkpoints or thresholds using the test split. The full aggregate and
per-class table lives in `docs/full-data-experiment.md`; the old daytime/clear
six-model ranking remains historical and must not be rewritten.

## 2026-09-14: Supersede the filtered subset with a full labelled experiment

The user explicitly authorized retraining the final three-model roster using all
publicly labelled BDD100K detection records. Preserve the completed daytime/clear
experiment as history. The final policy combines 69,863 official train and 10,000
official validation records, then splits all 79,863 images with `seed=0` into
55,904 train, 15,973 validation, and 7,986 test images. There is no time-of-day or
weather filter. The official 20,000-image test split remains inference-only
because its labels are not public. New data and runs use separate paths and may
not overwrite historical artifacts.

The final primary roster is a hand-written four-block grid detector
(`simple-cnn`), a hand-written residual grid detector (`complex-cnn`), and
YOLO11s as the baseline. Both custom CNNs use standard PyTorch layers, a shared
objectness/class/box head, and NMS; they are detectors rather than image-level
classifiers. All three therefore produce bounding boxes and share the same COCO
metrics. Older model code and artifacts are historical and stay outside the new
report.

"Retrain from scratch" means three new runs beginning at epoch 1 with no old
daytime/clear checkpoint. The two custom CNNs use random initialization. YOLO11s
uses its standard COCO-pretrained `yolo11s.pt` initialization as the declared
transfer-learning baseline; its initialization and native recipe are reported
as a comparison difference.

One exact duplicate source box occurs in validation image
`75055858-7d04a650.jpg`. Label conversion removes exact duplicates before
writing either format, leaving 295,515 validation boxes and preventing
Ultralytics-only deduplication from changing one backend's ground truth.

## 2026-09-14: Keep the course pipeline simple

The user chose five simple course stages: prepare/build, verify, train, predict,
and evaluate. Keep the current flat package/scripts layout and
existing CLI. Advanced profiling, monitoring, migration, and backend-specific
entrypoints remain available but are documented as optional. Do not add a CLI
framework, config hierarchy, CI, deployment, artifact schema, or broad new test
matrix for this cleanup.

## 2026-09-08: Shared instructions and state

User approved the folder demo in this task. `AGENTS.md` is the shared rule source;
`CLAUDE.md` imports it. `.agent/` stores common state; `.codex/` and `.claude/`
contain only tool-specific configuration. Start with no configuration overrides.
Custom agents and skills are deferred until needed.

## 2026-09-08: Preserve experiment context

Preserve the former `CLAUDE.md` verbatim in `docs/experiments.md`, clearly labeled
as historical. Keep `HANDOFF_3060.md` intact. This reorganization does not redefine
the detector comparison or establish current training status.

## Existing experiment contracts

Source: `AGENTS.md` and `HANDOFF_3060.md`.
Keep subset manifests, canonical class order and category/image-ID mappings,
streamed raw-label reads, centralized evaluation, and matched input resolution.
Use `constraints.txt` for installations that can pull PyTorch.

## 2026-09-08: Artifact organization

User approved migrating all old artifacts and adding a detailed README. Defaults
are repository-root anchored; explicit CLI output paths remain caller-relative.
Weights live under `weights/`, production/smoke runs under `runs/train/` and
`runs/smoke/`, and library runtime configuration under `runs/cache/`.
Historical checkpoint/log/metadata bytes remain intact; the migration receipt maps
old paths to new paths. The earlier decision to keep the root handoff is superseded:
current instructions are in `docs/HANDOFF_3060.md`; `docs/experiments.md` stays verbatim.
Logic defects found during review are backlog items, not part of this refactor.

## 2026-09-08: Six-model RTX 3060 profiling scope

The approved roster is YOLO11s, YOLO11m, YOLO26s, Faster R-CNN R50-FPN v2,
RT-DETR-l, and RF-DETR Small. Add all three new candidates in one implementation
phase. This phase profiles and smoke-tests; it does not start full 50-epoch runs.

The completed RT-DETR-l run remains the baseline and must not be retrained.
Each backend keeps its native preprocessing and training recipe; comparisons must
record actual tensor shape, optimizer, augmentation, batch, accumulation, precision,
and compute cost. Hardware tuning targets maximum stable throughput with at least
10% total VRAM headroom. Batch is selected before production training and never
reduced silently within a run. RF-DETR dependencies use a separate environment.

The implementation branch is `codex/add-model-profiling`. Work begins only after
the user gives a separate start command.

## 2026-09-08: Shared subagent architecture

Use repository-scoped native agent definitions for Codex and Claude Code with
shared, tool-neutral orchestration and experiment-validation skills. The primary
handles easy, short tasks directly. Before delegating difficult or long work, it
must ask the user and include recommended roles and models. The specialized roles
are explorer, worker, validator, and reviewer. The approved model profile and full
design are in
`docs/superpowers/specs/2026-09-08-shared-subagents-and-skills-design.md`.

The design is implemented with canonical skills under `.agents/skills/`, native
Codex agents under `.codex/agents/`, and native Claude agents plus skill routers
under `.claude/`. Primary model choices remain recommendations; the repository
sets role-specific subagent models only and leaves context-window settings unset.

## 2026-09-09: Profiling evidence and continuation scope

The current continuation focuses on implementation, review and CPU validation;
no new training jobs are launched in this pass. Independent review found the
saved probe throughput excluded loader wait. Preserve `runs/profile/official/`
as historical evidence, but invalidate its hardware selections and rerun with
versioned end-to-end timing before production use. Do not publish an epoch ETA
until full-data measurements exist. RF-DETR must pass seed 0 explicitly; its
backend default is not the experiment seed.

## 2026-09-09: Hourly reporting when training starts

User requests the same hourly reporting cadence used for the previous RT-DETR
run. When an authorized new training run starts, enable an hourly report in
Vietnamese covering model/run, checkpoint-authoritative epoch progress and
percentage, elapsed time, recent epoch duration, remaining duration and expected
completion time (Asia/Ho_Chi_Minh), restart/stall/error status, GPU utilization,
VRAM used/total and sampled peak, temperature, and power when available.
Use measured full-data timings for ETA and disclose uncertainty; do not invent
missing telemetry. Start the per-run GPU sampler with training and retain its
logs. Link current logs/artifacts. Reuse/update the paused prior hourly schedule
when launching, correcting its historical paths and target task; never reactivate
its old RT-DETR launch instructions or retrain the completed baseline. While no
new run is active, do not start periodic empty reports. Pause reporting after
verified completion or a terminal failure. This reporting request does not
itself authorize starting a new 50-epoch run.

## 2026-09-09: Train remaining models after publishing code

The user explicitly authorized profiling, smoke/resume validation, and subsequent
50-epoch GPU training for YOLO11s, YOLO11m, YOLO26s, Faster R-CNN R50-FPN v2,
and RF-DETR Small. Commit and push the implementation branch before starting GPU
work. Keep the completed RT-DETR-l baseline untouched. Run only one training or
profiling workload on the RTX 3060 at a time. Enable the existing hourly reporting
cadence when the first new run starts.

## 2026-09-09: Prefer training speed in hardware selection

The user prioritizes training speed and does not require conservative GPU/VRAM
limits. Select the fastest valid measured configuration up to 95% sampled total
VRAM; use lower memory only for candidates within 1% throughput. OOM, non-finite
loss, validation failure, and candidates above the ceiling remain ineligible.

## 2026-09-09: RF-DETR latest-checkpoint interval

Use RF-DETR `checkpoint_interval=2`. In RF-DETR 1.10.1, an interval of 1
suppresses the separate latest-checkpoint callback and produces only numbered
archives. An interval of 2 preserves periodic archives while atomically updating
full-state `last.ckpt` every epoch for controlled stops, supervisor progress, and
faithful resume.

## 2026-09-10: Stop RF-DETR smoke runs through Lightning

Use a native PyTorch Lightning callback for bounded RF-DETR smoke runs. Append it
after RF-DETR constructs its trainer and checkpoint callbacks, ignore validation
sanity checks, and set `trainer.should_stop=True` after the requested completed
epoch. Do not use a background signal watcher: `SIGINT` did not stop RF-DETR's
Lightning loop reliably and allowed the smoke run to exceed its target.
