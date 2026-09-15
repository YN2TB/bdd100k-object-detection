# Outstanding work

New active execution plan:
[full BDD100K retraining](plans/active/2026-09-14-full-bdd100k-retraining.md).

- [x] Regenerate full-data manifests and links as 70/20/10 over all 79,863 labels.
- [x] Build and verify YOLO/COCO labels for 79,863 labelled images.
- [x] Implement and CPU-smoke SimpleCNN/ComplexCNN bounding-box detectors.
- [x] Update the simple README pipeline for train/val/test and three models.
- [x] Restore NVIDIA driver access and pass GPU preflight.
- [x] Smoke/resume-check simple CNN, complex CNN, and YOLO11s.
- [x] Bound custom-CNN validation at 100 detections/image; full-val smoke passed.
- [x] Rerun complete acceptance: 100 tests pass (one expected skip), compilation,
  CLI help, shell syntax, and diff whitespace checks pass.
- [x] Commit the passing implementation locally as `d5e4f38`.
- [x] Obtain explicit trust confirmation and push `d5e4f38` to the configured
  GitHub remote.
- [x] Start the clean sequential train queue after push and reactivate hourly
  reporting (2026-09-15 10:13 +07; SimpleCNN first).
- [ ] Complete all 30 epochs for SimpleCNN, ComplexCNN, and YOLO11s.
- [ ] Predict and centrally evaluate all three models on the 7,986-image test split.

Active execution plan:
[six-model profiling](plans/active/2026-09-08-six-model-profiling.md).

- [ ] Complete final independent review of applied repairs (86 main-environment tests plus isolated RF-DETR cache test pass).
- [x] Re-run corrected end-to-end profiling for five unfinished models; current
  selections are in `runs/profile/official-v3/summary.json`.
- [x] Evaluate conditional RAM-cache triggers on GPU; no selected profile met
  both trigger conditions, so all resolved profiles use `cache=none`.
- [x] Finish RF-DETR Small two-epoch full-data stop/resume validation. YOLO11s,
  YOLO26s, YOLO11m, and Faster R-CNN have passed both smoke stages.
- [x] Record resolved recipes, measured epoch/setup/validation times and ETA.
- [ ] Complete final independent acceptance review and archive the plan only
  after its GPU profiling and smoke criteria pass.

Registry entries, backend interfaces and RF-DETR isolation are present in the
working tree. Adapter split/category/image-byte validation passes. RT-DETR-l
loading/prediction has a saved passing receipt, with protected hashes rechecked
on 2026-09-09. See `docs/model-profiling-status.md` for evidence and limitations.

Implemented repairs requiring runtime acceptance:

- [ ] RESUME-1: reconcile CSV progress with checkpoint epoch.
- [ ] RESUME-2: distinguish early clean stop from completed budget.
- [ ] REPRO-1: define Faster R-CNN seed/RNG resume policy.
- [x] PRED-1: consolidate prediction and define empty-output evaluation.
- [x] PRED-2: verified RT-DETR prediction export for centralized evaluation.
- [x] Evaluate and rank all six best checkpoints through the centralized COCO path;
  see `docs/model-ranking.md` and `runs/evaluation/official-v1/`.

Deferred beyond the active plan:

- [x] DATA-1: refuse manifest drift before writes.
- [x] DATA-2: verify empty-label images and return failure status.
- [x] DATA-3: validate manifest coverage and duplicate raw records.
- [x] DATA-4: bounded, atomic raw-label extraction.
- [ ] DOC-1: confirm full model roster before final comparison report.
