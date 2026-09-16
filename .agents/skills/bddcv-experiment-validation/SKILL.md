---
name: bddcv-experiment-validation
description: Use when validating changes to BDDCV data, training, checkpointing, profiling, prediction, or evaluation behavior.
---

# BDDCV Experiment Validation

Read `AGENTS.md` and `docs/experiments.md`. Also read `docs/HANDOFF_3060.md`
for RTX 3060 work and the relevant plan located through `.agent/PLANS.md`.

Select only checks relevant to the change, but preserve every applicable contract:

- Dataset: fixed daytime/clear manifests, expected split counts, disjoint filenames,
  and no writes before drift is rejected.
- Labels: class order from `bddcv.constants`, legacy aliases through
  `canonical_category()`, boxed annotations through `detection_boxes()`, YOLO class
  `i` to COCO category `i + 1`, and streaming raw-label reads.
- Inputs: matched comparison geometry and recorded backend-native tensor shape.
- Resume: atomic writes; model, optimizer, scaler, scheduler/iteration, epoch, best
  score, and RNG state when required; explicit compatibility rejection; checkpoint
  progress authoritative over CSV; incomplete stop distinct from completion.
- Artifacts: repository-root defaults, caller-relative explicit paths, protected
  manifests/history, and no modification or retraining of completed RT-DETR-l.
- Prediction/evaluation: image IDs by filename, common COCO detection JSON,
  zero-detection handling, centralized `bddcv.evaluation`, and reliable-class
  metric policy.
- Comparison reporting: actual resolution/tensor shape, optimizer, schedule,
  augmentation, precision, batch, accumulation, versions, and compute cost.

Run focused unit checks first. Expand to label verification or smoke tests only
when the change and authorization require them. Never launch a 50-epoch run,
install dependencies, regenerate data, or modify production artifacts without
explicit user authority.

## Result

Begin the final result with exactly one status:

- `PASS` — every applicable acceptance check succeeded.
- `FAIL` — a requirement is violated or a check failed.
- `BLOCKED` — required evidence cannot be obtained within current permissions or
  environment.

Then list commands with exit status, observed evidence, checked requirements, and
remaining uncertainty. Do not fix implementation code; return failures to the
primary.

