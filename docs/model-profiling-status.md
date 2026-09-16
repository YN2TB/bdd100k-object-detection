# Six-model implementation and profiling status

Updated: 2026-09-09

## Acceptance status

Review fixes have been applied; CPU regression and cache integration tests pass.
Corrected timing-schema-3 GPU profiling is complete for all five unfinished
models. Full-data two-epoch stop/resume acceptance has not completed yet.

## Current RTX 3060 selections

Source: `runs/profile/official-v3/summary.json`. Selection prioritizes measured
throughput up to 95% sampled total VRAM and uses lower memory only within a 1%
throughput tie. Every selected configuration passed training and validation
probes without OOM or non-finite values.

| Model | Batch | Workers | Prefetch | Cache | Images/s | Peak device MiB | Peak VRAM |
|---|---:|---:|---:|---|---:|---:|---:|
| YOLO11s | 32 | 4 | 2 | none | 95.07 | 10,035 | 81.67% |
| YOLO11m | 16 | 8 | 4 | none | 43.29 | 9,914 | 80.68% |
| YOLO26s | 16 | 4 | 4 | none | 71.93 | 7,116 | 57.91% |
| Faster R-CNN R50-FPN v2 | 16 | 4 | 2 | none | 17.15 | 8,239 | 67.05% |
| RF-DETR Small | 8 | 4 | 2 | none | 20.64 | 9,212 | 74.97% |

YOLO26s batch 32 reached 49.00 images/s in the workers-0 sweep but used 96.61%
sampled total VRAM, so it was excluded. No selected result met both conditional
RAM-cache triggers; no cache trial was needed. The first Faster R-CNN sweep
exposed an uninitialized loader-wait buffer, which was fixed with a regression
test before rerunning all four batch candidates successfully.

The saved profiling sweep below is historical evidence, not an approved hardware
configuration. Review found that step timing excluded data-loader wait, making
worker selection unreliable. Re-run the sweeps after the timing fix before
selecting production settings. Do not derive a full-training ETA from these values.

## Saved RTX 3060 sweep (superseded timing)

Source: `runs/profile/official/summary.json`. The GPU is an RTX 3060 with
12,288 MiB total memory. Values below reproduce the saved selections; throughput
is the old compute-step metric, not end-to-end training throughput.

| Model | Batch | Workers | Prefetch | Old step images/s | Peak device MiB |
|---|---:|---:|---:|---:|---:|
| frcnn-r50-fpn-v2 | 8 | 8 | 2 | 17.04 | 5271 |
| rfdetr-small | 8 | 4 | 4 | 21.10 | 9196 |
| yolo11m | 16 | 0 | 2 | 43.49 | 9841 |
| yolo11s | 32 | 0 | 2 | 95.11 | 10173 |
| yolo26s | 16 | 4 | 4 | 71.35 | 7120 |

The saved selected candidates were below 90% device memory in those probes.
This does not establish full-data memory safety. The original sweep recorded OOM
at YOLO11s/YOLO26s batch 64, YOLO11m batch 32, and RF-DETR Small batch 16.

Observed model input tensors:

- YOLO: `[batch, 3, 640, 640]` in the saved training probes.
- Faster R-CNN: `[8, 3, 384, 640]` training and `[4, 3, 384, 640]` validation.
- RF-DETR Small: square inputs from 352 through 672 in steps of 32 under native
  multi-scale training. The nominal registry input is 512.

Native recipes differ: YOLO uses Ultralytics augmentation and optimizer policy;
Faster R-CNN uses SGD, warmup/cosine scheduling and horizontal flips; RF-DETR uses
its native training recipe. These timings are not a controlled detector speed
comparison. Full-data runs must record resolved optimizer, accumulation,
augmentation, precision, validation batch and schedule before comparison.

## RT-DETR baseline preservation

Saved receipt: `runs/verification/rtdetr_baseline/verification.json` reports a
successful checkpoint load and prediction for one validation image (300 detections,
`[1, 3, 640, 640]`, float32). This is a loading/export check, not full evaluation.
On 2026-09-09, SHA-256 hashes were recomputed for `weights/best.pt`,
`weights/last.pt`, `results.csv`, and `logs/supervisor.log` under
`runs/train/rtdetr-l/`; all match the receipt. The completed baseline was not retrained.

## Environment and remaining acceptance

Use `.venv/bin/python`, not the shell default Conda Python (which lacks Torch).
The pinned main stack is Torch 2.11.0+cu128, torchvision 0.26.0+cu128, and
Ultralytics 8.4.142. RF-DETR 1.10.1 is isolated in `.venv-rfdetr` with its
resolved dependencies in `requirements-rfdetr-lock.txt`.

Still required before declaring the plan complete:

- Finish independent review of the applied fixes.
- Run five full-data epoch-1 stop / epoch-2 resume smoke experiments with a
  50-epoch schedule horizon, each in its own output directory.
- Record measured epoch/setup/validation times and an ETA with a ±30% interval.

No full-training ETA is available yet.

## Repair verification (2026-09-09)

- `.venv/bin/python -m unittest discover -s tests -v`: exit 0, 77 tests passed.
- `git diff --check`: exit 0.
- `.venv/bin/python -m compileall -q scripts src/bddcv`: exit 0.
- Both environment `pip check` commands: exit 0, no broken requirements.
- Read-only RF-DETR adapter validation: exact filenames/categories, disjoint
  splits and source-image byte matches.

Repairs cover checkpoint provenance on repeated YOLO resumes, early clean exits,
dataset/config drift rejection, CSV reconciliation, supervisor/trainer metadata
compatibility, RF-DETR seed 0, Faster R-CNN validation batch alignment, prediction
coverage/resolution checks, and end-to-end probe timing. Native GPU execution of
the repaired code has not yet been validated. RAM-cache support is implemented; GPU validation and
full recipe/precision measurement remain outstanding; this report does not claim
that all implementation or experiment acceptance criteria are complete.

## RAM-cache and entrypoint continuation

Conditional cache trials and native cache/prefetch options are implemented for
YOLO, Faster R-CNN and RF-DETR. Cached decoded inputs are copied before transforms,
so stochastic augmentation is not frozen. The CPU native RF-DETR integration test
checks pixels, boxes and JPEG draft scaling. The host RAM monitor samples peak
usage during profiling; cache construction refuses the 75% ceiling. Full-data
cache capacity remains unmeasured.

Timing schema 3 measures non-overlapping step completion intervals, including
loader overhead, and records fetch wait separately. Memory selection refuses
missing total-device measurements. The supervisor also resolves sibling modules
from an unrelated working directory without inherited PYTHONPATH.

Final commands for this continuation:

- `.venv/bin/python -m unittest discover -s tests -v`: 87 discovered tests;
  86 passed, one RF-DETR integration test skipped in the main environment.
- `.venv-rfdetr/bin/python -m unittest discover -s tests -p test_rfdetr_cache_integration.py -v`:
  exit 0, the RF-DETR integration test passed in its isolated environment.
- Training entrypoint `--help`, compilation and `git diff --check`: exit 0.

The corrected GPU profiling artifacts are stored under `runs/profile/official-v3/`.
Smoke and production training remain pending at this checkpoint. Independent
final review remains unavailable because of the recorded usage limit.
