# Shared subagents and skills design

Status: awaiting user review

## Goal

Add repository-scoped orchestration for Codex and Claude Code. Both tools must
understand the BDD100K detector-comparison context, distinguish casual work from
long or difficult work, and ask the user before delegating. The confirmation must
include the task classification, recommended roles, and recommended models.

Four specialized roles are required:

- `bddcv-explorer`: read-heavy investigation and execution-path mapping.
- `bddcv-worker`: plan-driven implementation in an explicitly assigned scope.
- `bddcv-validator`: evidence-based tests, smoke checks, and artifact validation.
- `bddcv-reviewer`: independent correctness and scientific-validity review.

## Selected architecture

Use native project-scoped agent definitions for each tool:

```text
.codex/agents/*.toml
.claude/agents/*.md
```

Keep project knowledge and workflow policy in shared, tool-neutral sources:

```text
AGENTS.md
.agent/HANDOFF.md
.agent/TODO.md
.agent/DECISIONS.md
.agent/PLANS.md
.agents/skills/bddcv-orchestration/SKILL.md
.agents/skills/bddcv-experiment-validation/SKILL.md
```

Codex loads the shared skills by repository-relative paths from
`.codex/config.toml`. Claude-facing files in `.claude/skills/` are thin routers to
the canonical shared skill content. Agent definitions differ in syntax and tool
controls, while responsibilities and acceptance contracts remain equivalent.

Use plain files rather than filesystem symlinks so shared configuration behaves
consistently on Linux and Windows clones.

## Orchestrator policy

The primary owns the user conversation, scope, sequencing, delegation, and final
synthesis. Before material work, it classifies the request on two axes:

- Complexity: easy or difficult.
- Expected duration: short or long.

A task is casual only when it is both easy and short. Examples include a narrow
question, targeted read-only lookup, or small edit with an obvious execution path
and quick verification. The primary handles casual tasks directly without a
delegation prompt.

A task is difficult or long when it has one or more of these properties:

- It crosses multiple modules, backends, or implementation phases.
- It requires broad codebase exploration or version-specific research.
- It changes training, checkpointing, data conversion, prediction, or evaluation.
- It needs independent validation or scientific-comparison review.
- It is expected to generate substantial logs or consume significant context.
- It can be divided into two or more independent workstreams.

Before spawning any subagent for such a task, the primary must ask for user
confirmation. The request states:

1. The easy/difficult and short/long classification with a brief reason.
2. Whether delegation is recommended.
3. Which roles will be used and whether they run in parallel or sequence.
4. The recommended primary and subagent models.
5. Expected material activity such as code writes, dependency changes, GPU jobs,
   or lengthy tests.

No subagent starts until the user confirms. Confirmation applies only to the
described scope. The primary may steer or re-run the same assigned agents within
that scope without repeatedly asking; materially broader delegation needs a new
confirmation.

## Role contracts

### Explorer

The explorer is read-only. It reads `AGENTS.md`, current `.agent/` state, and the
relevant plan and experiment documentation before investigating. It traces entry
points, data flow, configuration, tests, and dependencies with file and symbol
references. It reports concise evidence and uncertainties. It does not edit files,
install packages, or run training.

### Worker

The worker receives a bounded assignment and, for non-trivial coding, a
user-approved plan. It reads shared project state before editing, preserves
unrelated user changes, and follows repository coding and experiment contracts.
It runs focused verification and returns changed files, commands, outcomes, and
remaining risks.

It must not launch full 50-epoch training, retrain the completed RT-DETR-l
experiment, change manifests, or expand into the separate domain-shift project
unless the user explicitly authorizes that work. Only one worker owns a writable
subsystem at a time. Parallel workers require independent file ownership and
outputs.

### Validator

The validator checks an implementation against approved requirements. It is
read-only except for explicitly assigned disposable or ignored test artifacts. It
runs the smallest sufficient tests first, then required integration, data, or
smoke checks. It verifies command exit status and actual outputs rather than
accepting summaries.

For experiment-related changes it checks dataset identity, class/category
mapping, input policy, checkpoint/resume state, prediction format, centralized
evaluation, and artifact routing as applicable. It returns `PASS`, `FAIL`, or
`BLOCKED` followed by evidence. It never silently fixes implementation code.

### Reviewer

The reviewer is read-only and independent of the worker. It examines the diff and
requirements for behavioral defects, regressions, unsafe resume behavior,
reproducibility gaps, invalid detector comparisons, and missing meaningful tests.
Findings are ordered by severity and include file references and a concrete
failure scenario. Style-only observations are omitted unless they obscure a real
risk. No findings means the reviewer says so and lists residual risks or
unverified assumptions.

## Workflow

For a confirmed difficult or long coding task:

1. The primary freezes scope and locates the relevant active plan.
2. The explorer maps unfamiliar code or external APIs when needed.
3. The primary turns the evidence into a plan and obtains any required approval.
4. The worker implements the bounded assignment.
5. The validator runs acceptance checks.
6. The reviewer independently reviews the stable diff.
7. The primary sends concrete failures to the same worker, then repeats affected
   validation and review.
8. The primary updates shared state and reports completion only when required
   checks pass, or accurately reports a blocker.

Exploration and independent documentation research may run in parallel.
Validation and review begin after a stable worker diff and may run in parallel
when neither writes shared state.

## Model routing

Use aliases or models available in the installed client instead of dated model
IDs that a teammate's client may not resolve. Do not set a global primary model in
the repository: the primary recommends a model, while the user retains control of
the main session selection.

### Codex Plus recommendation

| Role | Model | Reasoning |
|---|---|---|
| Primary/orchestrator | `gpt-6-astra` | `low` |
| Worker | `gpt-5.6-luna` | `max` |
| Explorer | `gpt-5.6-luna` | `xhigh` |
| Validator | `gpt-5.6-terra` | `max` |
| Reviewer | `gpt-5.6-sol` | `high` |
| Escalation/adjudication | `gpt-6-astra` | `medium` |

This profile optimizes completed, high-quality work per Plus allowance. A compact
task packet makes Luna Max suitable for implementation. Terra Max validates with
a model independent of the worker. Sol High supplies strong review and model
diversity without consuming Astra for every review. Astra Medium is reserved for
cross-backend architecture, scientific-method disputes, or unresolved disagreement
between validator and reviewer.

Leave `model_context_window` and `model_auto_compact_token_limit` unset. Use the
repository state files and distilled task packets as external memory.

### Claude Code recommendation

| Role | Model alias | Effort |
|---|---|---|
| Primary/orchestrator | `opus` | `medium` |
| Worker | `sonnet` | `high` |
| Explorer | `haiku` | `high` |
| Validator | `sonnet` | `high` |
| Reviewer | `opus` | `high` |
| Escalation/adjudication | `opus` | `max` |

Claude has no one-to-one equivalents for the Codex models. Opus owns orchestration
and scientific judgment, Sonnet owns implementation and validation, and Haiku
handles bounded read-heavy exploration.

The explorer is promoted to Terra Max on Codex or Sonnet High on Claude for
checkpoint state, RNG restoration, dependency compatibility, or investigation
across several detector backends. If a recommended model is unavailable, the
primary discloses the fallback in the confirmation prompt.

## Project context required by every role

Every role inherits or reads these facts before acting:

- The dataset is the fixed BDD100K daytime/clear subset with 12,454 train and
  1,764 validation images.
- Class order comes only from `bddcv.constants`; YOLO class `i` maps to COCO
  category `i + 1`.
- Raw labels use `stream_records()`; `detection_boxes()` excludes annotations
  without boxes; legacy aliases use `canonical_category()`.
- Reported accuracy comes through `bddcv.evaluation`, with image IDs resolved by
  filename and the reliable-class policy preserved.
- Detector inputs remain comparable while backend-native optimizer, schedule,
  augmentation, and preprocessing differences are recorded.
- Checkpoint writes remain atomic and resume state remains complete and compatible.
- The completed RT-DETR-l run is preserved and must not be retrained.
- The active six-model profiling phase does not authorize full 50-epoch runs.
- Data manifests, images, annotations, historical logs, and class order are
  protected experiment artifacts.

Agents consult `docs/experiments.md` before changing training, checkpointing,
data conversion, prediction, or evaluation. RTX 3060 assignments also consult
`docs/HANDOFF_3060.md`.

## Tool and permission boundaries

- Explorer and reviewer use read-only sandbox configuration where supported.
- Validator is read-only by default. Necessary caches or test artifacts go to
  repository-ignored paths or `/tmp` within the approved validation scope.
- Worker inherits the primary workspace permission policy and cannot bypass user
  approval or repository restrictions.
- Subagents do not create more subagents unless the platform supports nested
  delegation and the primary explicitly assigns coordination responsibility.
- Network access, installs, destructive actions, long GPU runs, and external
  writes retain their normal approval requirements.

## Skills

`bddcv-orchestration` contains classification, confirmation, routing, task-packet,
handoff, and failure-loop guidance. It applies when deciding whether or how to
delegate work. It links to current repository state instead of copying changing
plan details.

`bddcv-experiment-validation` contains non-obvious scientific and artifact checks
for training, data, checkpointing, prediction, evaluation, or profiling changes.
It does not activate for casual documentation or unrelated Python edits.

The skills complement `AGENTS.md`; they do not duplicate it or grant authority.

## Acceptance criteria

- Codex discovers all four agents from `.codex/agents/`.
- Claude Code discovers all four agents from `.claude/agents/`.
- Both primary agents handle easy, short work without a delegation prompt.
- Both ask for confirmation with role and model recommendations before delegating
  difficult or long work.
- Explorer and reviewer are read-only under their declared controls.
- Worker prompts require explicit scope and an approved plan for non-trivial work.
- Validator output follows the required status and evidence format.
- A dry-run checkpointing scenario routes through explorer, worker, validator,
  reviewer, and optional adjudication in the intended order.
- Skill structure and frontmatter pass available validation tooling.
- Existing unit tests continue to pass.

Implementation of this configuration does not include live training, dependency
installation, dataset regeneration, or production artifact modification.
