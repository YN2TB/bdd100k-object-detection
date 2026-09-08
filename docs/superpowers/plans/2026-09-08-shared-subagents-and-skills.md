# Shared Subagents and Skills Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add team-shared Codex and Claude agent definitions plus reusable orchestration and experiment-validation skills for the BDD100K detector project.

**Architecture:** Keep project invariants in `AGENTS.md` and canonical reusable workflows under `.agents/skills/`. Add native, repository-scoped role definitions under `.codex/agents/` and `.claude/agents/`, with thin Claude skill routers so both clients consume the same workflow. Verify structure and required behavioral clauses with repository unit tests and the bundled skill validator.

**Tech Stack:** TOML, Markdown with YAML frontmatter, Python `unittest`, Python `tomllib`

**Spec:** `docs/superpowers/specs/2026-09-08-shared-subagents-and-skills-design.md`

## Global Constraints

- The primary must ask for confirmation before delegating any difficult or long task.
- Easy and short tasks must be handled directly without a delegation prompt.
- Codex role defaults are Luna Max worker, Luna XHigh explorer, Terra Max validator, and Sol High reviewer; Astra Low primary and Astra Medium escalation remain recommendations rather than forced global settings.
- Claude role defaults are Sonnet High worker and validator, Haiku High explorer, and Opus High reviewer; Opus Medium primary and Opus Max escalation remain recommendations rather than forced global settings.
- Explorer and reviewer are read-only; validator writes only disposable or ignored verification artifacts.
- Do not set `model_context_window` or `model_auto_compact_token_limit`.
- Do not run training, install dependencies, regenerate data, or modify production artifacts.
- Preserve all BDD100K experiment contracts from `AGENTS.md`.

---

### Task 1: Add structural contract tests

**Files:**
- Create: `tests/test_agent_configuration.py`

**Interfaces:**
- Consumes: repository files under `.codex/`, `.claude/`, and `.agents/skills/`.
- Produces: `unittest` coverage for discoverability, model routing, read-only roles, canonical skill presence, and delegation confirmation language.

- [ ] **Step 1: Write the failing structural tests**

Create `tests/test_agent_configuration.py` using `unittest`, `tomllib`, and
`pathlib.Path`. Define constants for the four role names and helpers
`load_codex_agent(role: str) -> dict` and `load_claude_agent(role: str) -> str`.
Add tests that assert:

```python
EXPECTED_CODEX = {
    "bddcv-worker": ("gpt-5.6-luna", "max"),
    "bddcv-explorer": ("gpt-5.6-luna", "xhigh"),
    "bddcv-validator": ("gpt-5.6-terra", "max"),
    "bddcv-reviewer": ("gpt-5.6-sol", "high"),
}
EXPECTED_CLAUDE = {
    "bddcv-worker": ("sonnet", "high"),
    "bddcv-explorer": ("haiku", "high"),
    "bddcv-validator": ("sonnet", "high"),
    "bddcv-reviewer": ("opus", "high"),
}
```

For every Codex file, parse TOML and verify `name`, `description`,
`developer_instructions`, `model`, and `model_reasoning_effort`. Assert explorer
and reviewer declare `sandbox_mode = "read-only"`. For each Claude file, inspect
the frontmatter and assert the expected `name`, `model`, and `effort`; assert
explorer and reviewer list only read-oriented tools.

Assert both canonical `SKILL.md` files exist and contain valid `name` and
`description` frontmatter. Assert `.claude/skills/*/SKILL.md` points to its matching
canonical skill. Assert orchestration text contains all of `easy`, `difficult`,
`short`, `long`, `confirmation`, `recommended`, and `model`. Assert neither
`.codex/config.toml` nor any agent file contains `model_context_window` or
`model_auto_compact_token_limit`.

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
python -m unittest tests.test_agent_configuration -v
```

Expected: FAIL because the agent and skill files do not exist.

- [ ] **Step 3: Commit the contract test**

```bash
git add tests/test_agent_configuration.py
git commit -m "Test shared agent configuration contracts"
```

### Task 2: Create canonical shared skills and Claude routers

**Files:**
- Create: `.agents/skills/bddcv-orchestration/SKILL.md`
- Create: `.agents/skills/bddcv-experiment-validation/SKILL.md`
- Create: `.claude/skills/bddcv-orchestration/SKILL.md`
- Create: `.claude/skills/bddcv-experiment-validation/SKILL.md`

**Interfaces:**
- Consumes: `AGENTS.md`, `.agent/PLANS.md`, relevant active plans, `docs/experiments.md`, and `docs/HANDOFF_3060.md`.
- Produces: `$bddcv-orchestration` and `$bddcv-experiment-validation` workflows; Claude routers import the corresponding canonical file using repository-relative references.

- [ ] **Step 1: Write the orchestration skill**

Create a valid skill with name `bddcv-orchestration`. Its description activates
only when classifying or delegating project work. Its workflow must:

```text
read shared state
classify complexity as easy/difficult
classify duration as short/long
handle easy+short directly
for difficult or long: prepare a recommendation, ask for confirmation, then stop
after confirmation: send a compact task packet and coordinate roles
```

Define the confirmation packet fields from the spec and require an explicit scope,
goal, relevant files, constraints, acceptance criteria, permitted writes, tests,
and expected return format in every worker task packet. Include the Codex and
Claude recommendation tables from the spec and the escalation conditions.

- [ ] **Step 2: Write the experiment-validation skill**

Create a valid skill with name `bddcv-experiment-validation`. Its description
activates for changes to data, training, checkpoints, profiling, prediction, or
evaluation. Require the validator to select applicable checks for dataset identity,
class/category mapping, input geometry, resume completeness and compatibility,
artifact routing, common COCO output, centralized evaluation, reliable-class
metrics, backend recipe disclosure, and protection of the completed RT-DETR-l run.
Require final output beginning with exactly one of `PASS`, `FAIL`, or `BLOCKED`,
followed by commands, observed evidence, and remaining uncertainty.

- [ ] **Step 3: Add Claude skill routers**

Create minimal valid Claude skill files with matching names and descriptions. Each
body imports its canonical source:

```markdown
Follow the canonical project skill exactly:
@../../../.agents/skills/<skill-name>/SKILL.md
```

- [ ] **Step 4: Validate the canonical skills**

Run:

```bash
python /home/orlab/.codex/skills/.system/skill-creator/scripts/quick_validate.py .agents/skills/bddcv-orchestration
python /home/orlab/.codex/skills/.system/skill-creator/scripts/quick_validate.py .agents/skills/bddcv-experiment-validation
```

Expected: both commands report valid skills.

- [ ] **Step 5: Run the focused structural test**

```bash
python -m unittest tests.test_agent_configuration -v
```

Expected: role-file assertions still fail; skill assertions pass.

- [ ] **Step 6: Commit the shared skills**

```bash
git add .agents/skills .claude/skills
git commit -m "Add shared BDDCV orchestration skills"
```

### Task 3: Add native Codex role definitions

**Files:**
- Create: `.codex/agents/bddcv-worker.toml`
- Create: `.codex/agents/bddcv-explorer.toml`
- Create: `.codex/agents/bddcv-validator.toml`
- Create: `.codex/agents/bddcv-reviewer.toml`
- Modify: `.codex/config.toml`

**Interfaces:**
- Consumes: canonical shared skills and repository state.
- Produces: four project-scoped custom Codex agents discoverable by their `name` fields.

- [ ] **Step 1: Configure global project agent settings**

Add this section to `.codex/config.toml` without setting a primary model or context
window:

```toml
[agents]
enabled = true
max_concurrent_threads_per_session = 4
interrupt_message = true

[[skills.config]]
path = ".agents/skills/bddcv-orchestration/SKILL.md"
enabled = true

[[skills.config]]
path = ".agents/skills/bddcv-experiment-validation/SKILL.md"
enabled = true
```

- [ ] **Step 2: Add the worker agent**

Create a TOML agent named `bddcv-worker` using `gpt-5.6-luna`, effort `max`, and
`workspace-write`. Its instructions require a bounded task packet, relevant shared
state reads, minimal scoped edits, focused verification, no delegation, and a
return containing files, commands, results, and risks.

- [ ] **Step 3: Add the explorer agent**

Create a TOML agent named `bddcv-explorer` using `gpt-5.6-luna`, effort `xhigh`,
and `read-only`. Its instructions require evidence with file/symbol references and
forbid edits, installs, and training. It must recommend promotion to Terra Max for
the complex cases named in the spec.

- [ ] **Step 4: Add the validator agent**

Create a TOML agent named `bddcv-validator` using `gpt-5.6-terra`, effort `max`,
and `read-only`. Its instructions require the experiment-validation skill when
applicable, direct evidence, the three-state result format, and no implementation
fixes.

- [ ] **Step 5: Add the reviewer agent**

Create a TOML agent named `bddcv-reviewer` using `gpt-5.6-sol`, effort `high`, and
`read-only`. Its instructions require independent diff review, severity-ordered
findings, concrete failure scenarios, file references, and residual risks when no
findings exist.

- [ ] **Step 6: Run the focused structural test**

```bash
python -m unittest tests.test_agent_configuration -v
```

Expected: Codex and skill assertions pass; Claude role assertions fail.

- [ ] **Step 7: Commit the Codex agents**

```bash
git add .codex tests/test_agent_configuration.py
git commit -m "Configure project Codex subagents"
```

### Task 4: Add native Claude role definitions

**Files:**
- Create: `.claude/agents/bddcv-worker.md`
- Create: `.claude/agents/bddcv-explorer.md`
- Create: `.claude/agents/bddcv-validator.md`
- Create: `.claude/agents/bddcv-reviewer.md`

**Interfaces:**
- Consumes: canonical shared skills through Claude routers and repository state.
- Produces: four project-scoped Claude Code agents discoverable by frontmatter name.

- [ ] **Step 1: Add the worker agent**

Create a Markdown agent named `bddcv-worker` with `model: sonnet`, `effort: high`,
`skills: [bddcv-orchestration]`, and tools `Read, Grep, Glob, Bash, Edit, Write`.
Mirror the Codex worker contract and explicitly exclude subagent spawning.

- [ ] **Step 2: Add the explorer agent**

Create a Markdown agent named `bddcv-explorer` with `model: haiku`, `effort: high`,
and tools `Read, Grep, Glob`. Mirror the Codex explorer contract and request Sonnet
High promotion for complex checkpoint, RNG, dependency, or cross-backend work.

- [ ] **Step 3: Add the validator agent**

Create a Markdown agent named `bddcv-validator` with `model: sonnet`, `effort:
high`, `skills: [bddcv-experiment-validation]`, and tools `Read, Grep, Glob, Bash`.
Mirror the validator evidence and result-format contract. State that Bash writes
must be limited to ignored test artifacts or `/tmp`.

- [ ] **Step 4: Add the reviewer agent**

Create a Markdown agent named `bddcv-reviewer` with `model: opus`, `effort: high`,
and tools `Read, Grep, Glob, Bash`. Mirror the independent review contract and
permit Bash only for read-only Git inspection and tests that do not alter tracked
or production artifacts.

- [ ] **Step 5: Run the focused structural test**

```bash
python -m unittest tests.test_agent_configuration -v
```

Expected: PASS.

- [ ] **Step 6: Commit the Claude agents**

```bash
git add .claude/agents tests/test_agent_configuration.py
git commit -m "Configure project Claude subagents"
```

### Task 5: Integrate orchestration guidance and verify the repository

**Files:**
- Modify: `AGENTS.md`
- Modify: `.agent/DECISIONS.md`
- Modify: `.agent/HANDOFF.md`
- Modify: `.agent/TODO.md`
- Modify: `.agent/PLANS.md`
- Move: `docs/superpowers/plans/2026-09-08-shared-subagents-and-skills.md` to `.agent/plans/archive/2026-09-08-shared-subagents-and-skills.md` after completion

**Interfaces:**
- Consumes: all agent definitions and skills from Tasks 2–4.
- Produces: primary-agent routing instructions, complete shared state, and repository-level verification evidence.

- [ ] **Step 1: Add concise primary routing to AGENTS.md**

Add a `Subagent Routing` subsection that points to `$bddcv-orchestration`, states
the easy+short direct-work rule, and mandates confirmation with task/model
recommendations before difficult or long delegation. Do not copy the full skill.

- [ ] **Step 2: Run all structural and repository tests**

```bash
python -m unittest discover -s tests -v
```

Expected: all tests pass.

- [ ] **Step 3: Run skill validation again**

```bash
python /home/orlab/.codex/skills/.system/skill-creator/scripts/quick_validate.py .agents/skills/bddcv-orchestration
python /home/orlab/.codex/skills/.system/skill-creator/scripts/quick_validate.py .agents/skills/bddcv-experiment-validation
```

Expected: both skills are valid.

- [ ] **Step 4: Inspect the final diff and protected paths**

```bash
git diff --check
git status --short
git diff -- AGENTS.md .agent .agents .codex .claude tests docs/superpowers
```

Expected: only configuration, test, plan, spec, and shared-state files change; no
files under `data/`, `runs/`, or `weights/` change.

- [ ] **Step 5: Update shared project state**

Record the configuration and validation outcome in `.agent/DECISIONS.md` and
`.agent/HANDOFF.md`. Remove completed configuration items from `.agent/TODO.md`
without changing the separate six-model profiling backlog. Add this plan to the
archive section in `.agent/PLANS.md` and move the plan file to the archive path.

- [ ] **Step 6: Commit the integration**

```bash
git add AGENTS.md .agent .agents .codex .claude tests docs/superpowers
git commit -m "Integrate shared subagent workflow"
```
