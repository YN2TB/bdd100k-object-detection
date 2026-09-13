# Six-model centralized ranking

Evaluated on 2026-09-13 with `bddcv.evaluation` against the fixed 1,764-image
BDD100K daytime/clear validation split. Predictions use confidence 0.001 and at
most 300 exported detections per image. The headline ranking is COCO mAP50–95
averaged over the nine reliable classes. The `train` class has only two
validation boxes, so it remains in standard COCO mAP but is excluded from the
headline mean.

| Rank | Model | Reliable mAP50–95 | COCO mAP50–95 | mAP50 | mAP75 | Small | Medium | Large | Train h | Input | Peak VRAM MiB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | RT-DETR-l | 0.3612 | 0.3275 | 0.5749 | 0.3094 | 0.1817 | 0.4046 | 0.6056 | 9.03 | 640 | 8,695 |
| 2 | YOLO11m | 0.3243 | 0.2919 | 0.5088 | 0.2785 | 0.1364 | 0.3727 | 0.5953 | 4.00 | 640 | 9,975 |
| 3 | RF-DETR Small | 0.3025 | 0.3126 | 0.5428 | 0.3039 | 0.1052 | 0.3391 | 0.7077 | 8.03 | 512 | 9,199 |
| 4 | YOLO26s | 0.3008 | 0.2712 | 0.4752 | 0.2613 | 0.1230 | 0.3365 | 0.5678 | 2.32 | 640 | 7,127 |
| 5 | YOLO11s | 0.2851 | 0.2566 | 0.4533 | 0.2429 | 0.1127 | 0.3261 | 0.5431 | 1.83 | 640 | 10,051 |
| 6 | Faster R-CNN R50-FPN v2 | 0.2623 | 0.2361 | 0.4357 | 0.2211 | 0.0961 | 0.2926 | 0.5516 | 10.52 | 640 | 8,230 |

RT-DETR-l has the highest headline score and the highest AP on every reliable
class. YOLO11m is second while requiring less than half RT-DETR-l's recorded
training time. YOLO26s is the strongest speed/resource choice: its reliable mAP
is only 0.0016 below RF-DETR Small, while its training completed 5.72 hours
sooner and used 2,072 MiB less peak VRAM. YOLO11s has the shortest training time.
RF-DETR Small has the best large-object AP, but uses its approved native 512
input, while the other five use 640.

The production JSON files contain 187 low-confidence boxes clipped to zero area
across four backends. COCOeval accepts these records. A filtered re-evaluation
changed reliable mAP by at most 0.00000011, so the displayed precision and rank
are unchanged.

Detailed metrics, checkpoint hashes, prediction counts, and per-class AP are in
`runs/evaluation/official-v1/metrics.json`. The sortable table is
`runs/evaluation/official-v1/ranking.csv`, and COCO prediction files are under
`runs/predictions/official-v1/`.
