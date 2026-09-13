# Six-model profiling and RTX 3060 optimization plan

Status: profiling, smoke/resume, production training, and centralized evaluation
complete; final independent review pending.

## Goal

Support and profile six detectors on the fixed BDD100K daytime/clear subset:
YOLO11s, YOLO11m, YOLO26s, Faster R-CNN R50-FPN v2, RT-DETR-l, and RF-DETR
Small. Produce measured RTX 3060 12 GB configurations, complete smoke/resume
validation, then train the five unfinished models for 50 epochs sequentially.

## Fixed decisions

- Add all three proposed models: YOLO11m, YOLO26s, and RF-DETR Small.
- After code is committed and pushed, run the five authorized 50-epoch jobs
  sequentially on the GPU and report progress every hour.
- Preserve the completed 50-epoch RT-DETR-l run as the baseline. Do not retrain it.
- Keep each backend's native preprocessing, optimizer, augmentation, and input
  policy. Report actual tensor shapes and recipe differences.
- Prioritize training throughput up to 95% sampled total VRAM. Prefer lower
  memory only when measured throughput is within 1% of the fastest candidate.
- Never change resolution after OOM. Select the next smaller tested batch.
- Do not modify dataset manifests, source images, annotations, or class order.

## Implementation phases

### 1. Model registry and dependency isolation

- Add a versioned model registry for all six models, including backend, pretrained
  checkpoint, input policy, default epoch budget, batch candidates, precision,
  seed, and output paths.
- Keep the existing Torch CUDA stack constrained. Pin the working Ultralytics
  version without upgrading it during implementation.
- Install RF-DETR training dependencies in a separate ignored environment. Select
  a stable release compatible with the constrained CUDA stack; record the resolved
  package versions. If no compatible release exists, mark RF-DETR blocked without
  changing the main environment.
- Route RF-DETR weights and caches under `weights/rfdetr/` and `runs/cache/rfdetr/`.

### 2. RF-DETR dataset and backend adapter

- Build a generated dataset view under `data/adapters/rfdetr/` from the existing
  manifests, images, and COCO annotations. Prefer hard links on the same filesystem;
  use verified copies only when links are unavailable.
- Validate 12,454 train images, 1,764 validation images, disjoint splits, exact
  filenames, ten categories, and category mapping through `bddcv.constants`.
- Add RF-DETR Small train, checkpoint/resume, and COCO prediction adapters using
  the API of the pinned stable release. Disable automatic test-set evaluation.
- Keep RF-DETR's native 512 input policy unless the pinned release's pretrained
  model requires another documented value; record the actual value in results.

### 3. Common train, profile, and prediction interfaces

- Add registry-driven commands for model training, profiling, and prediction while
  keeping the current entrypoints compatible.
- Resolve backend from model ID rather than checkpoint filename. Persist resolved
  configuration, library versions, GPU identity, dataset fingerprint, seed,
  pretrained provenance, physical batch, accumulation, and validation batch.
- Support all six models in prediction and emit the common COCO detection JSON
  format. Map image IDs by filename and categories by the shared class order.
- Keep native postprocessing, but save confidence, max detections, NMS/head mode,
  preprocessing shape, and precision with every benchmark result.

### 4. Reliable resume before new long experiments

- Make checkpoint state the progress authority; reconcile CSV rows to checkpoint
  epoch so a crash between writes cannot report false completion.
- Save and restore Python, NumPy, Torch, CUDA, sampler/worker RNG, optimizer,
  scaler, scheduler position, and best score for Faster R-CNN.
- Lock dataset fingerprint, model, resolution, physical/effective batch, optimizer,
  and schedule on resume. Reject drift explicitly.
- Treat clean exit before target as incomplete, distinct from completion.
- Disable automatic batch reduction during production runs through supported
  adapter configuration; never patch installed packages.
- Maintain read compatibility for existing checkpoints, but refuse unsafe resume
  when required state cannot be reconstructed.

### 5. RTX 3060 profiling

- Create a deterministic 512-image probe: the 128 most box-dense training images
  plus 384 seed-0 sampled images. Save the generated manifest under `runs/profile/`.
- Run every candidate in a fresh subprocess: 10 warm-up training steps, 30 measured
  training steps, and 10 validation steps using the production model, losses,
  optimizer, augmentation, and data loader.
- Collect median/p95 step time, images/s, GPU utilization, allocated/reserved and
  total device memory, host RAM, temperature, and power. Include data wait where
  the backend exposes it.
- Batch candidates: YOLO11s and YOLO26s `8,16,32,64`; YOLO11m `4,8,16,32`;
  Faster R-CNN `2,4,8,16`; RF-DETR Small `1,2,4,8,16`. RT-DETR-l retains batch 8.
- Reject OOM, non-finite loss, validation OOM, or total GPU memory above 95%.
  Choose maximum throughput; within 1%, choose lower memory use.
- After batch selection, test workers `0,4,8` with prefetch 2; test prefetch 4 only
  for the best nonzero worker count. Try RAM cache only when utilization is below
  80% and data wait exceeds 20%, and retain it only for a gain of at least 5% with
  host RAM below 75%.

### 6. Smoke/resume acceptance

- For YOLO11s, YOLO11m, YOLO26s, Faster R-CNN, and RF-DETR Small, run two full-data
  smoke epochs with the selected hardware profile: checkpoint after epoch 1, stop,
  resume into epoch 2. Keep the scheduler horizon at 50 epochs.
- If a full-data smoke OOMs, start a new smoke run using the next smaller profiled
  batch. Do not modify the failed run in place.
- For RT-DETR-l, verify baseline checkpoint loading and prediction only. Do not run
  training or full centralized evaluation on the RTX 3060.
- Compute ETA from measured full-data epoch time with separate setup/validation
  times and a disclosed +/-30% planning interval.

## Tests and acceptance criteria

- Registry/backend selection works for generic checkpoint names such as `best.pt`.
- RF-DETR dataset view preserves source bytes and exact split/category membership.
- Profile subprocesses correctly record success, OOM, NaN, validation failure, and
  GPU-unavailable states without leaving root artifacts.
- Resume restores state and rejects configuration drift; CSV-ahead-of-checkpoint
  recovery cannot create false completion.
- Prediction handles zero-detection images and produces valid COCO IDs for all
  supported backends.
- Existing artifact-layout and migration tests continue to pass.
- README and handoff report only measured batch, memory, throughput, tensor shapes,
  smoke results, dependency versions, and ETA. Estimates must remain labeled.

Implementation is complete only when the three new models are integrated, the five
unfinished models pass profile plus two-epoch smoke/resume, the RT-DETR-l baseline
remains intact, and the measured RTX 3060 profile is documented. A dependency or
hardware incompatibility must be recorded as `blocked`, not represented as ready.

## 2026-09-09 continuation evidence

Implementation/review repairs are in the working tree; 77 CPU tests pass. See
`docs/model-profiling-status.md` for checked commands and preserved evidence.
The old official sweep excludes loader wait and is superseded. Corrected timing
uses aggregate measured elapsed time and rejects incompatible saved outcomes.
No training was launched in this continuation. Final independent review,
corrected GPU sweeps with implemented conditional RAM-cache trials, and full-data smoke/resume
acceptance remain outstanding. Do not mark this plan complete from unit tests.

RAM-cache continuation: timing schema 3, cache/prefetch resume locking, and CPU
native cache equivalence checks are implemented. 86 main-environment tests and
one separate RF-DETR cache integration pass. GPU/full-data acceptance is pending.
