# Outstanding work

Active execution plan:
[six-model profiling](plans/active/2026-09-08-six-model-profiling.md).

- [ ] Add registry entries for YOLO11s, YOLO11m, YOLO26s, Faster R-CNN,
  RT-DETR-l, and RF-DETR Small.
- [ ] Pin/verify backend dependencies and isolate RF-DETR.
- [ ] Build and validate the RF-DETR dataset adapter.
- [ ] Add common training, profiling, resume, and prediction interfaces.
- [ ] Profile batch/workers on RTX 3060 with 10% VRAM headroom.
- [ ] Run two-epoch stop/resume smoke validation for five unfinished models.
- [ ] Verify RT-DETR-l checkpoint loading/prediction without retraining it.
- [ ] Document measured hardware profiles, tensor shapes, smoke status, and ETA.

Related defects required by the active plan:

- [ ] RESUME-1: reconcile CSV progress with checkpoint epoch.
- [ ] RESUME-2: distinguish early clean stop from completed budget.
- [ ] REPRO-1: define Faster R-CNN seed/RNG resume policy.
- [ ] PRED-1: consolidate prediction and define empty-output evaluation.
- [ ] PRED-2: verified RT-DETR prediction export for centralized evaluation.

Deferred beyond the active plan:

- [ ] DATA-1: refuse manifest drift before writes.
- [ ] DATA-2: verify empty-label images and return failure status.
- [ ] DATA-3: validate manifest coverage and duplicate raw records.
- [ ] DATA-4: bounded, atomic raw-label extraction.
- [ ] DOC-1: confirm full model roster before final comparison report.
