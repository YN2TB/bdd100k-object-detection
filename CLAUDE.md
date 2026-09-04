# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A deep learning course project: a controlled comparison of a one-stage detector (YOLO11s, Ultralytics) against a two-stage detector (Faster R-CNN R50-FPN v2, torchvision) on a fixed BDD100K subset, with per-class analysis under severe class imbalance.

Scope is exactly that comparison. A separate research project on domain shift and target-data efficiency reuses this dataset and this code, but lives in its own folder with its own `CLAUDE.md` and is **paused**. Nothing here implements it, and nothing here should change to accommodate it — if a request is about domain shift, target-data budgets, or recovery curves, it belongs to that project, not this one.

The full plan lives at `~/.claude/plans/velvet-finding-kurzweil.md`.

## Environment — read before installing anything

Python 3.14.3, `torch 2.11.0+cu128`, `torchvision 0.26.0+cu128`, `ultralytics 8.4.138`, `pycocotools 2.0.11`.

The RTX 5060 is Blackwell (`sm_120`). A stock `pip install torch` pulls a CUDA 12.1 build with no `sm_120` kernels and fails at the first training step with `no kernel image is available for execution on the device`. Worse, `pip install ultralytics` on its own wants to upgrade torch to 2.14.0 **from PyPI**, which on Windows is the CPU-only wheel — this silently disables GPU training rather than erroring.

**Always pass `-c constraints.txt` when installing anything that can pull torch:**

```bash
python -m pip install <pkg> -c constraints.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

Verify the stack before trusting any run — the last line is the real test, since capability reporting can succeed while kernel launches still fail:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability(0)); torch.zeros(8,device='cuda').sum().item()"
```

Expect `2.11.0+cu128 (12, 0)` on the 5060, `(8, 6)` on the teammate's RTX 3060.

## Pipeline commands

Run in this order. Steps 1–4 are already done; they are idempotent and safe to re-run.

```bash
# 1. Census: image counts per (timeofday x weather), class histograms -> data/label_census.json
python scripts/scan_labels.py

# 2. Select the daytime+clear subset and extract only those images from archive.zip
python scripts/prepare_data.py

# 3. Write YOLO txt + COCO json from one source pass
python scripts/build_labels.py

# 4. Verify: numeric YOLO<->COCO cross-check, plus a visual montage
python scripts/verify_labels.py

# 5. Train
python -c "from ultralytics import YOLO; YOLO('yolo11s.pt').train(data='configs/bdd_source.yaml', epochs=50, imgsz=640, batch=16, amp=True, device=0, project='runs', name='yolo11s')"
python scripts/train_frcnn.py --epochs 50 --batch 4          # --batch 8 on the 12GB card

# Resume an interrupted run (both models checkpoint every epoch)
python scripts/train_frcnn.py --epochs 50 --batch 4 --resume
python -c "from ultralytics import YOLO; YOLO('runs/detect/runs/yolo11s/weights/last.pt').train(resume=True)"

# 6. Evaluate (both models must go through this)
python scripts/predict_yolo.py runs/detect/runs/yolo11s/weights/best.pt --out runs/preds_yolo.json
python scripts/evaluate.py runs/preds_yolo.json --title "YOLO11s"
```

Smoke-test either trainer before committing to a full run: `train_frcnn.py --limit-train 200 --epochs 1`, or the same YOLO call with `epochs=1`.

There is no test suite and no linter configured. `scripts/verify_labels.py` is the correctness check for the data pipeline.

## Architecture

Four invariants hold this together. Breaking any one produces plausible-looking but invalid results.

**`src/bddcv/constants.py` owns class order.** `DET_CLASSES` is the contract binding the YOLO label files, the COCO annotation file, and every results table. Reordering it invalidates already-converted labels and trained checkpoints. Never re-declare class order locally.

The archive ships the **legacy** label release, which names three classes differently from the newer `det_20` release (`person`/`bike`/`motor` rather than `pedestrian`/`bicycle`/`motorcycle`). `canonical_category()` maps them; results are always reported under the canonical names.

**`src/bddcv/labels.py` streams rather than loads.** `bdd100k_labels_images_train.json` is 1.45 GB and `json.load` on it needs roughly 8–12 GB, which does not coexist with a training process on a 23 GB machine. `stream_records()` walks the top-level array with a bounded buffer. Use it for any pass over the raw labels.

`labels[]` also carries `poly2d` drivable-area and lane annotations with no `box2d`; `detection_boxes()` drops those.

**`src/bddcv/evaluation.py` is the only evaluation path.** Ultralytics reports its own mAP, which is not COCO mAP. Using its internal number for one model and `pycocotools` for the other would invalidate the headline comparison. Every reported number comes from `evaluate()`.

Two conventions it depends on:
- category id: YOLO index `i` ↔ COCO `i + 1`. Prediction converters add 1.
- `image_id` comes from the ground truth file, mapped **by `file_name` via `image_id_map()`** — never by enumeration order.

**`src/bddcv/frcnn.py` matches YOLO's input resolution.** Ultralytics `imgsz=640` letterboxes 1280×720 to 640×360. torchvision's default `min_size=800` would feed roughly 1422×800 — about 8× the pixels — silently handing Faster R-CNN a large advantage. `MIN_SIZE=360, MAX_SIZE=640` reproduce YOLO's geometry. Do not change one side without the other.

What deliberately **cannot** be matched, and must be disclosed in the report rather than papered over: optimizer, LR schedule, and augmentation pipeline (Ultralytics applies mosaic; the torchvision loop applies only horizontal flip).

## Checkpointing

`train_frcnn.py` writes two files into `--out`: `best.pt` (weights at the highest val mAP@[.5:.95], used for evaluation) and `last.pt` (full training state, written **every** epoch — saving only on improvement would fall back to whenever the score last improved). Both go through `atomic_save()`, which writes a temp file and renames, because a truncated `torch.save` from power loss is exactly the failure being insured against.

`last.pt` carries model, optimizer, `GradScaler`, epoch, the global iteration counter and the best score so far. **The iteration counter drives the cosine LR schedule**, so resuming without it would restart the schedule at warmup — a run that completes and reports plausible numbers from a different LR trajectory than the one it started with. `--resume` therefore refuses to continue when any of `epochs`, `batch`, `lr`, `warmup_iters` or `limit_train` differ from the checkpoint, and `results.csv` appends rather than truncating.

Ultralytics handles its own equivalent: `last.pt` and `best.pt` every epoch, resumed with `resume=True`. Note it strips optimizer state on *successful completion*, so a finished run shows `epoch: -1` and cannot be resumed — only an interrupted one can.

## Dataset facts

Established by `scripts/scan_labels.py`; no need to re-derive.

Source domain is `timeofday == "daytime"` and `weather == "clear"`: **12,454 train / 1,764 val images**, uniformly 1280×720. That is the entire available pool, so it is used in full — there is no subsampling and no sampling seed. The subset is reproducible from the attribute filter alone; `data/source_daytime_clear/{train,val}_images.txt` records it.

The 20k test split has no public labels. Validation comes from the 10k `val` split.

Class imbalance is severe. Train / val instance counts:

| class | train | val |
|---|---|---|
| car | 139,376 | 19,627 |
| traffic sign | 43,858 | 6,339 |
| traffic light | 25,991 | 3,785 |
| pedestrian | 16,772 | 2,450 |
| truck | 7,411 | 1,021 |
| bus | 2,727 | 380 |
| bicycle | 1,616 | 207 |
| rider | 1,209 | 174 |
| motorcycle | 905 | 111 |
| **train** | **36** | **2** |

`train` has two validation boxes. Its AP is noise — the 1-epoch smoke run scored it P=1.0, R=0.0, AP=0.0. `evaluation.py` reports it but excludes any class under `MIN_VAL_INSTANCES` from the headline mean, which is published separately as `mAP50_95_reliable_classes`.

## Measured performance

On the RTX 5060 Laptop (8 GB), YOLO11s at `imgsz=640, batch=16, amp=True`:

- **9:42 per training epoch** (779 iterations, 1.3 it/s), ~15 s validation → **50 epochs ≈ 8.3 hours**
- VRAM peak 4.3 GB of 8 GB, so there is headroom to raise batch size
- GPU sustained 87 °C; expect thermal throttling on long runs
- After 1 epoch from COCO-pretrained: mAP50 0.279, mAP50-95 0.141

## Data layout

`archive.zip` (8.2 GB) is a BDD100K repack. Its train images are **not stored flat** — they are scattered across `train/trainA`, `train/trainB`, `train/testA`, `train/testB` and `train/` itself. Basenames are globally unique, so `prepare_data.py` resolves them through a basename index scoped to the split subtree. Do not assume a flat layout.

The pipeline needs only two things from the archive: `data/labels_raw/` (the two label JSONs) and `data/source_daytime_clear/` (the 14,218 selected images plus converted labels, ~1.1 GB). A full extraction of the archive under `data/bdd100k*/` is present but **not required by any script** — it can be deleted to reclaim space.

This directory is not a git repository. If one is initialised, exclude `data/`, `runs/`, `datasets/`, and `archive.zip`.
