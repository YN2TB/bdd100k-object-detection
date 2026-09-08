# Project audit — 2026-09-08

## Scope and implemented changes

Reviewed all 14 original Python source/entrypoint files, configuration, project
instructions, experiment notes, dataset metadata and existing run artifacts.
Implemented repository-root output defaults, explicit caller-relative overrides,
pretrained weight routing, fresh/resume output routing, isolated library caches,
smoke and GPU monitor entrypoints, migration with checksum receipts and rollback,
and a detailed README. No model algorithm or experiment hyperparameter was changed.

39 legacy files were moved: root pretrained RT-DETR weight, completed RT-DETR run,
logs, smoke metadata, historical helper scripts and the label montage. Checkpoint,
CSV, log and historical metadata bytes were preserved. The receipt is
`runs/maintenance/migration-b4b9d2f7126747dc848ed7b0e2135191.json`.
The original experiments snapshot was preserved byte-for-byte. The current
RTX 3060 instructions moved to `docs/HANDOFF_3060.md` with updated commands.

## Deferred findings (not fixed in this change)

| ID | Priority | Evidence | Impact / proposed follow-up |
|---|---|---|---|
| DATA-1 | High | `scripts/prepare_data.py:156`, `main()` writes manifests before the drift warning | Refuse drift before modifying either split; preserve fixed experiment membership. |
| DATA-2 | High | `scripts/verify_labels.py:42` iterates only annotated images; line 66 only prints status | Check all COCO images including empty labels and return a failing exit code. |
| RESUME-1 | High | `scripts/train_frcnn.py:193` appends CSV before checkpoint save; wrapper `epochs_done()` counts rows | Crash between CSV and checkpoint can cause duplicate rows / false completion; reconcile progress with saved epoch. |
| DATA-3 | Medium | `scripts/build_labels.py:39`, `build()` filters raw records against a set without checking coverage | Missing/duplicate raw records can silently yield incomplete output; compare manifest and emitted image names. |
| DATA-4 | Medium | `scripts/prepare_data.py:73`, `ensure_raw_labels()` uses full `src.read()` and direct destination writes | Large allocation and partial file retained after interruption; use bounded copy and atomic completion. |
| RESUME-2 | Medium | `scripts/train_with_resume.py`, `main()` returns 0 for clean child exit before target | Callers cannot distinguish completed budget from early stop; define explicit result status. |
| REPRO-1 | Medium | `scripts/train_frcnn.py`, `main()` shuffles/augments without explicit seed or checkpoint RNG state | Reproducibility and interrupted-run trajectories need a dedicated design and tests. |
| PRED-1 | Medium | `scripts/train_frcnn.py:42` and `src/bddcv/frcnn.py:89` duplicate prediction conversion | Consolidate conversion without altering numeric output; test empty predictions (`evaluation.py:43` currently raises). |
| PRED-2 | Medium | `scripts/predict_yolo.py:40` constructs YOLO only | Add a separately verified RT-DETR export path before centralized RT-DETR scoring. |
| DOC-1 | Low | Historical notes describe two models; RTX 3060 assignment describes four | Keep history distinct; obtain the complete comparison roster before final report. |

## Verification

- `python -m unittest discover -s tests -v`: 9 tests passed, including interruption
  rollback and process-guard refusal. CLI help checks passed for all changed entrypoints.
- Unit tests exercise fresh/resume command arguments with spaces and quotes, output
  resolution from another working directory, migration dry-run, collision rejection,
  checksum preservation, repeat invocation and rollback refusal after modification.
- Actual GPU smoke: `scripts/smoke_ultra.py --model rtdetr-l.pt --name layout_check
  --fraction 0.001 --batch 8 --workers 0`, invoked from `/tmp`. One epoch on the
  reduced training fraction, full validation; success in 106.1 seconds, peak allocated
  6,094,811,648 bytes. Output: `runs/smoke/layout_check/`. These are smoke artifacts,
  not experimental comparison scores.
- Initial smoke failed before training because CUDA memory stats were reset before
  initialization; reproduced with a minimal host check and corrected by initializing CUDA.
- Legacy RT-DETR results retain 50 epoch rows and the original COMPLETE log.
- Migration checked host process visibility before applying and SHA-256 after moving.
- Dataset configuration, manifests, annotations and experiment snapshot are compared
  with a pre-change checksum baseline. Existing manifest CRLF differences from Git are preserved.
- Independent reviewer dispatch failed because of usage limits; final review is local.

## Operational limits

Migration requires same-filesystem hard links and host process visibility. Run only
while project workers are stopped; the guard is a point-in-time check, not a lock
against a trainer started concurrently. Receipt rollback restores moved artifacts,
not code/documentation. Historical PID files are not evidence of a live process.

Final static check found an existing invalid-escape SyntaxWarning in
`scripts/scan_labels.py:79`; it is unrelated to output routing and remains unchanged.
