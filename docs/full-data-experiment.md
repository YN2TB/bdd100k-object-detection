# Full-data three-model experiment

Updated: 2026-09-15

## Scope

Current report compares three object detectors on the fixed 70/20/10 split of
all 79,863 publicly labelled BDD100K images:

1. `simple-cnn`: four hand-written convolution blocks and a grid detection head.
2. `complex-cnn`: hand-written residual backbone and the same grid detection head.
3. `yolo11s`: Ultralytics baseline.

The shared custom head uses a 12x20 grid with five box slots per cell. Each slot
predicts objectness, normalized box coordinates, and ten class logits. Five slots
leave approximately 0.6% of dense same-cell ground-truth boxes unassigned, versus
approximately 4.0% with three slots. Class-aware NMS produces the final boxes.
Validation and test export retain at most 100 detections per image, matching the
COCO evaluator's `maxDets=100` and bounding prediction memory.

Both custom models use 640x360 inputs, random initialization, Adam, horizontal
flip, AMP when CUDA is available, and 30 epochs. YOLO starts a fresh run at epoch
1 from the standard COCO-pretrained `yolo11s.pt`, uses `imgsz=640`, and keeps its
native optimizer and augmentation for the same 30-epoch budget. These recipe and
initialization differences must be disclosed with final results.

## Current status

- Data split and label conversion: complete.
- Label counts: 1,029,446 train, 295,515 validation, 146,998 test boxes. One
  exactly duplicated source box was removed from both YOLO and COCO validation
  labels so all backends use identical ground truth.
- Full unit suite: 100 passed with one expected isolated historical RF-DETR skip.
- CPU smoke: SimpleCNN and ComplexCNN completed one epoch; SimpleCNN checkpoint
  also resumed successfully from epoch 1 to epoch 2.
- GPU preflight and one-epoch smoke runs: passed on the RTX 3060. Safe batches
  are 16 for SimpleCNN, 8 for ComplexCNN, and 16 for YOLO11s.
- Production GPU training: not started validly. An early queue attempt was
  stopped before any epoch checkpoint because it preceded commit/push and exposed
  excessive custom-validation RAM use. That issue is fixed and the passing code
  is now pushed. The clean queue started at 2026-09-15 10:13 +07 with SimpleCNN
  first; hourly reporting is active.
- SimpleCNN completed 30/30 epochs in 3 h 20 min with no restart. Its best
  validation mAP50-95 is 0.002768 at epoch 19.
- ComplexCNN had completed 13/30 checkpoints at 2026-09-15 15:20 +07 and was
  finishing epoch 14. Its average is about 7.7 minutes/epoch; estimated finish is
  17:25 +07. YOLO11s remains queued.
- ComplexCNN completed 30/30 epochs in 3 h 51 min with no restart. Its best
  validation mAP50-95 is 0.019779 at epoch 23.
- YOLO11s had completed 16/30 checkpoints at 20:25 +07 and was training epoch
  17. Recent epochs take about 10.8 minutes; estimated finish is 22:55 +07.

Pre-production RTX 3060 estimates for 30 epochs:

| Model | Estimated epoch | Estimated 30 epochs | Confidence |
|---|---:|---:|---|
| SimpleCNN | 6.7 min observed | 3 h 20 min observed | complete |
| ComplexCNN | 7.7 min observed | 3 h 51 min observed | complete |
| YOLO11s | 10.8 min observed | about 5 h 24 min | high; 16 epochs observed |

The sequential training total is approximately 19-25 hours, excluding final
test prediction/evaluation. Replace these ranges with observed epoch timings once
each clean production run records its first checkpoint.

## Final result table

Fill this table only after predictions are exported on the 7,986-image test set.

| Model | mAP50-95 | mAP50 | mAP75 | AR@100 | Parameters | Train time |
|---|---:|---:|---:|---:|---:|---:|
| SimpleCNN | pending | pending | pending | pending | 408,171 | pending |
| ComplexCNN | pending | pending | pending | pending | 11,190,123 | pending |
| YOLO11s | pending | pending | pending | pending | ~9.4M | pending |

Do not tune a checkpoint or threshold using test results.
