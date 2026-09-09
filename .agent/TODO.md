# Outstanding work

Active execution plan:
[six-model profiling](plans/active/2026-09-08-six-model-profiling.md).

- [ ] Complete final independent review of applied repairs (86 main-environment tests plus isolated RF-DETR cache test pass).
- [x] Re-run corrected end-to-end profiling for five unfinished models; current
  selections are in `runs/profile/official-v3/summary.json`.
- [x] Evaluate conditional RAM-cache triggers on GPU; no selected profile met
  both trigger conditions, so all resolved profiles use `cache=none`.
- [ ] Run two-epoch full-data stop/resume validation for five unfinished models.
- [ ] Record resolved recipes, measured epoch/setup/validation times and ETA.
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
- [ ] PRED-1: consolidate prediction and define empty-output evaluation.
- [ ] PRED-2: verified RT-DETR prediction export for centralized evaluation.

Deferred beyond the active plan:

- [ ] DATA-1: refuse manifest drift before writes.
- [ ] DATA-2: verify empty-label images and return failure status.
- [ ] DATA-3: validate manifest coverage and duplicate raw records.
- [ ] DATA-4: bounded, atomic raw-label extraction.
- [ ] DOC-1: confirm full model roster before final comparison report.
