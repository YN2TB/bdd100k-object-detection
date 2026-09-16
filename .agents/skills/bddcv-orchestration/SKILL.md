---
name: bddcv-orchestration
description: Use when classifying project work or deciding whether to delegate BDDCV tasks to specialized agents.
---

# BDDCV Orchestration

Read `AGENTS.md`, `.agent/HANDOFF.md`, `.agent/TODO.md`, and
`.agent/DECISIONS.md`. Use `.agent/PLANS.md` to locate only the relevant plan.

## Classify before work

Classify complexity as **easy** or **difficult** and expected duration as
**short** or **long**.

- Easy and short: handle directly. Do not discuss or spawn subagents.
- Difficult or long: prepare a delegation recommendation and ask for user
  confirmation. Stop before spawning any subagent.

Treat cross-module or cross-backend work, broad exploration, version research,
experiment-pipeline changes, independent scientific review, substantial logs, or
multiple independent workstreams as difficult or long.

## Confirmation

Tell the user:

1. The classification and brief reason.
2. Whether delegation is recommended.
3. Recommended roles and parallel or sequential order.
4. Recommended primary and subagent model settings.
5. Expected writes, installs, GPU work, or long tests.

Confirmation covers only that scope. Reuse and steer the assigned agents within
it; ask again before materially broader delegation.

## Model recommendations

| Role | Codex Plus | Claude Code |
|---|---|---|
| Primary | Astra low | Opus medium |
| Worker | Luna max | Sonnet high |
| Explorer | Luna xhigh | Haiku high |
| Validator | Terra max | Sonnet high |
| Reviewer | Sol high | Opus high |
| Adjudication | Astra medium | Opus max |

Promote the explorer to Terra max or Sonnet high for checkpoint/RNG state,
dependency compatibility, or multi-backend reasoning. Use adjudication only for
unresolved agent disagreement, architecture risk, or scientific-method disputes.
Disclose any unavailable-model fallback. Do not force a primary model or context
window in repository configuration.

## Task packet

Every delegated task states: goal, exact scope, relevant files, constraints,
acceptance criteria, permitted writes, tests, expected return format, and relevant
decisions or plan path. Send distilled evidence, not the parent conversation.

Use `bddcv-explorer` for read-heavy mapping, `bddcv-worker` for approved scoped
implementation, `bddcv-validator` for evidence-based acceptance, and
`bddcv-reviewer` for independent diff review. Validation and review consume a
stable worker diff. Return failures to the same worker and repeat affected gates.

