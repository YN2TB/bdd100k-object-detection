# Durable decisions

## 2026-09-08: Shared instructions and state

User approved the folder demo in this task. `AGENTS.md` is the shared rule source;
`CLAUDE.md` imports it. `.agent/` stores common state; `.codex/` and `.claude/`
contain only tool-specific configuration. Start with no configuration overrides.
Custom agents and skills are deferred until needed.

## 2026-09-08: Preserve experiment context

Preserve the former `CLAUDE.md` verbatim in `docs/experiments.md`, clearly labeled
as historical. Keep `HANDOFF_3060.md` intact. This reorganization does not redefine
the detector comparison or establish current training status.

## Existing experiment contracts

Source: `AGENTS.md` and `HANDOFF_3060.md`.
Keep subset manifests, canonical class order and category/image-ID mappings,
streamed raw-label reads, centralized evaluation, and matched input resolution.
Use `constraints.txt` for installations that can pull PyTorch.

## 2026-09-08: Artifact organization

User approved migrating all old artifacts and adding a detailed README. Defaults
are repository-root anchored; explicit CLI output paths remain caller-relative.
Weights live under `weights/`, production/smoke runs under `runs/train/` and
`runs/smoke/`, and library runtime configuration under `runs/cache/`.
Historical checkpoint/log/metadata bytes remain intact; the migration receipt maps
old paths to new paths. The earlier decision to keep the root handoff is superseded:
current instructions are in `docs/HANDOFF_3060.md`; `docs/experiments.md` stays verbatim.
Logic defects found during review are backlog items, not part of this refactor.

## 2026-09-08: Six-model RTX 3060 profiling scope

The approved roster is YOLO11s, YOLO11m, YOLO26s, Faster R-CNN R50-FPN v2,
RT-DETR-l, and RF-DETR Small. Add all three new candidates in one implementation
phase. This phase profiles and smoke-tests; it does not start full 50-epoch runs.

The completed RT-DETR-l run remains the baseline and must not be retrained.
Each backend keeps its native preprocessing and training recipe; comparisons must
record actual tensor shape, optimizer, augmentation, batch, accumulation, precision,
and compute cost. Hardware tuning targets maximum stable throughput with at least
10% total VRAM headroom. Batch is selected before production training and never
reduced silently within a run. RF-DETR dependencies use a separate environment.

The implementation branch is `codex/add-model-profiling`. Work begins only after
the user gives a separate start command.

## 2026-09-08: Shared subagent architecture

Use repository-scoped native agent definitions for Codex and Claude Code with
shared, tool-neutral orchestration and experiment-validation skills. The primary
handles easy, short tasks directly. Before delegating difficult or long work, it
must ask the user and include recommended roles and models. The specialized roles
are explorer, worker, validator, and reviewer. The approved model profile and full
design are in
`docs/superpowers/specs/2026-09-08-shared-subagents-and-skills-design.md`.

The design is implemented with canonical skills under `.agents/skills/`, native
Codex agents under `.codex/agents/`, and native Claude agents plus skill routers
under `.claude/`. Primary model choices remain recommendations; the repository
sets role-specific subagent models only and leaves context-window settings unset.
