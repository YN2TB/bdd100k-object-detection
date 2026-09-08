# Shared project state

Start with `../AGENTS.md`, then read `HANDOFF.md`, `TODO.md`, and `DECISIONS.md`.
Use `PLANS.md` to find the plan relevant to the current task.

- `HANDOFF.md`: latest verified state, uncertainties, and next steps.
- `TODO.md`: outstanding work only; remove completed entries.
- `DECISIONS.md`: durable decisions with rationale and source.
- `PLANS.md`: index of active and archived plans.
- `plans/active/`: plans for work in progress.
- `plans/archive/`: completed or superseded plans, retained for context.

Both Codex and Claude Code use these same files. Update them at meaningful handoffs;
do not copy state into tool-specific folders. Record commands and actual outcomes,
and distinguish historical notes from fresh verification. Keep secrets, checkpoints,
and logs out of this directory.

Tool adapters are `../.codex/config.toml` and `../.claude/settings.json`.
They start with no project overrides. Personal Claude settings belong in the ignored
`../.claude/settings.local.json`. Add custom agents or skills only when needed.

Configuration references:
- [Codex configuration](https://learn.chatgpt.com/docs/config-file/config-basic)
- [Claude memory imports](https://code.claude.com/docs/en/memory)
- [Claude settings](https://code.claude.com/docs/en/settings)
