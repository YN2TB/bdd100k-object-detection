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
- Production GPU training, test prediction, and centralized evaluation: complete.
  The clean queue ran after the passing implementation was pushed, from
  2026-09-15 10:13 to 22:53 +07, with no restart or error.
- SimpleCNN completed 30/30 epochs in 3 h 20 min with no restart. Its best
  validation mAP50-95 is 0.002768 at epoch 19.
- ComplexCNN completed 30/30 epochs in 3 h 51 min with no restart. Its best
  validation mAP50-95 is 0.019779 at epoch 23.
- YOLO11s completed 30/30 epochs in 5 h 23 min with no restart. Its best native
  validation mAP50-95 is 0.29102 at epoch 30.
- All three `best.pt` checkpoints were exported on the same 7,986-image held-out
  test split and scored through `bddcv.evaluation` with `maxDets=100`.

Observed RTX 3060 training time for 30 epochs:

| Model | Mean epoch | Total train time | Peak sampled VRAM |
|---|---:|---:|---|
| SimpleCNN | 6.7 min | 3 h 20 min | 5,021 MiB |
| ComplexCNN | 7.7 min | 3 h 51 min | 2,645 MiB |
| YOLO11s | 10.7 min | 5 h 23 min | 6,459 MiB |

Sequential training took approximately 12 h 35 min. Prediction and evaluation
then completed in about five minutes.

## Final test results

The following values come from one centralized COCO evaluation path on the
7,986-image test split with 146,998 ground-truth boxes. All ten classes have at
least ten test instances, so reliable-class mAP equals overall mAP50-95.

| Model | mAP50-95 | mAP50 | mAP75 | AP small | AP medium | AP large | AR@100 | Parameters | Train time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SimpleCNN | 0.0027 | 0.0107 | 0.0011 | 0.0003 | 0.0033 | 0.0077 | 0.0147 | 408,171 | 3 h 20 min |
| ComplexCNN | 0.0187 | 0.0585 | 0.0076 | 0.0013 | 0.0115 | 0.0733 | 0.0651 | 11,190,123 | 3 h 51 min |
| YOLO11s | **0.2761** | **0.4867** | **0.2631** | **0.1100** | **0.3299** | **0.5165** | **0.3644** | 9,431,662 | 5 h 23 min |

YOLO11s ranks first on every aggregate metric. ComplexCNN improves materially
over SimpleCNN, but both hand-written grid detectors remain far below the
transfer-learned YOLO baseline.

## Per-class AP50-95 on test

| Class | Test boxes | SimpleCNN | ComplexCNN | YOLO11s |
|---|---:|---:|---:|---:|
| pedestrian | 10,333 | 0.0015 | 0.0082 | **0.2979** |
| rider | 507 | 0.0069 | 0.0040 | **0.1883** |
| car | 81,826 | 0.0099 | 0.0657 | **0.4673** |
| truck | 3,422 | 0.0025 | 0.0383 | **0.4183** |
| bus | 1,339 | 0.0012 | 0.0467 | **0.4180** |
| train | 14 | 0.0000 | 0.0000 | 0.0000 |
| motorcycle | 338 | 0.0012 | 0.0069 | **0.1857** |
| bicycle | 862 | 0.0034 | 0.0062 | **0.2160** |
| traffic light | 21,304 | 0.0002 | 0.0033 | **0.2298** |
| traffic sign | 27,053 | 0.0004 | 0.0080 | **0.3397** |

The `train` class has only 14 test boxes and every model scores 0.0000 on it;
although it passes the project's ten-instance reliability threshold, this class
should still be interpreted cautiously. No checkpoint or threshold was selected
using these test results.

Final JSON artifacts are stored under `runs/predictions_full/` and
`runs/evaluation_full/`. The historical six-model/daytime-clear ranking remains
unchanged in `docs/model-ranking.md`.
