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
