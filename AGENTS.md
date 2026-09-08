# Repository Guidelines

## Project Structure & Module Organization

This Python project compares object detectors on the fixed BDD100K daytime/clear subset.

- `src/bddcv/`: shared class definitions, streaming label readers, Faster R-CNN components, and COCO evaluation.
- `scripts/`: data preparation, label verification, training/resume, prediction, and evaluation entry points.
- `configs/bdd_source.yaml`: dataset location and class names.
- `data/`: tracked census and train/validation manifests; generated images and annotations are ignored.
- `runs/`: ignored checkpoints, logs, predictions, and plots.
- `docs/experiments.md` and `docs/HANDOFF_3060.md`: historical experiment details and RT-DETR training handoff instructions.

## Setup & Development Commands

Run commands from the repository root. Set `path` in `configs/bdd_source.yaml` to your local `data/source_daytime_clear` directory; preserve the split paths and class names.

```bash
python -m pip install ultralytics pycocotools -c constraints.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

Use `constraints.txt` for dependency installs that can pull PyTorch to preserve the pinned CUDA build. No separate build step is configured.

- Data pipeline, in order: `python scripts/scan_labels.py`, `python scripts/prepare_data.py`, then `python scripts/build_labels.py`. Requires the BDD100K archive and raw labels.
- Label check: `python scripts/verify_labels.py` cross-checks YOLO/COCO labels and produces a visual montage.
- Training smoke run: `python scripts/train_frcnn.py --limit-train 200 --epochs 1 --out runs/smoke/smoke_frcnn`.
- Supervised YOLO training: `python scripts/train_with_resume.py ultra --model yolo11s.pt --name yolo11s --epochs 50 --batch 16`.
- Score exported predictions: `python scripts/evaluate.py runs/predictions/yolo.json --title YOLO11s`.

## Coding Style & Experiment Contracts

Use four-space indentation, `snake_case` functions/variables, uppercase constants, type hints, and concise docstrings. No formatter or linter is configured.

Import class order from `bddcv.constants`; never duplicate or reorder it. YOLO class `i` maps to COCO category `i + 1`. Resolve image IDs by filename. Stream raw labels with `stream_records()`. Report metrics through `bddcv.evaluation`, and preserve matched input resolution across detectors.

## Testing Guidelines

Run `python -m unittest discover -s tests -v` for artifact routing and migration regression checks. No coverage threshold is configured. For data changes, run label verification and inspect the montage. For trainer changes, run a short smoke experiment in a separate output directory. Record commands and outcomes.

## Commit & Pull Request Guidelines

History contains one descriptive initial commit; no formal message convention is established. Use concise, action-oriented subjects. PRs should explain the change, experiment impact, and validation performed, linking issues when relevant. Preserve tracked subset manifests; exclude datasets, weights, and run artifacts.

## Shared Agent Workflow

- This file is the shared project instruction source. `CLAUDE.md` imports it for Claude Code.
- Before continuing unfinished work, read `.agent/HANDOFF.md`, `.agent/TODO.md`, and `.agent/DECISIONS.md`.
- Use `.agent/PLANS.md` to locate only the relevant plan in `.agent/plans/active/`.
- Before handing work off, update the handoff with changes, verification, uncertainties, and the next concrete step. Keep TODO limited to outstanding work.
- Record durable decisions in `.agent/DECISIONS.md`; move completed plans to `.agent/plans/archive/` and update the index.
- Keep state tool-neutral: use repository-relative paths and commands, without credentials or machine-local settings.

### Subagent Routing

- Use `$bddcv-orchestration` when classifying work or considering delegation.
- Handle work directly when it is both easy and short; do not propose a subagent for casual tasks.
- Before delegating difficult or long work, tell the user the classification, reason, recommended roles and models, execution order, and material actions, then wait for confirmation.
- Use `bddcv-explorer` for read-heavy mapping, `bddcv-worker` for scoped implementation, `bddcv-validator` for acceptance evidence, and `bddcv-reviewer` for independent review.
- Keep the primary conversation focused by sending compact task packets and receiving concise evidence summaries.

## Additional Experiment Contracts

- Domain shift, target-data budgets, and recovery curves belong to a separate project; do not expand this repository into that work.
- Read `docs/experiments.md` before changing training, checkpointing, data conversion, or evaluation. It preserves historical context, not current run status.
- Use `canonical_category()` for legacy category aliases and `detection_boxes()` to exclude annotations without boxes.
- Preserve atomic checkpoint writes and complete resume state, including optimizer, scaler, epoch, iteration counter, and best score. Preserve resume compatibility checks and append-only result logging.
- Preserve the reliable-class metric policy for sparse validation classes. Disclose optimizer, schedule, and augmentation differences in comparisons.
- For the RTX 3060 RT-DETR assignment, follow `docs/HANDOFF_3060.md`: fixed subset, 50 epochs, `imgsz=640`, `seed=0`, and verification before training. Benchmark all models on the same GPU; the 3060 assignment returns training artifacts for centralized evaluation.
