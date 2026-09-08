# Artifact layout implementation — completed 2026-09-08

User-approved scope: repository-root output defaults, explicit path overrides,
all legacy artifact migration with SHA-256/rollback, detailed README, and separate
backlog for logic defects. No commit/push or experimental hyperparameter changes.

- [x] Add shared paths and route fresh/resume/prediction/verification outputs.
- [x] Add reusable smoke and GPU monitoring entrypoints.
- [x] Test dry-run, collisions, interrupted migration, guard failure and rollback.
- [x] Migrate 39 legacy artifacts; verify hashes and repeat no-op.
- [x] Move current 3060 handoff and update detailed README.
- [x] Record review/backlog and preserve historical snapshot and dataset files.
- [x] Run 9 unit tests and separate reduced-data GPU smoke (106.1 seconds).

See `docs/reviews/2026-09-08-project-audit.md` and `README.md` for results and limits.
