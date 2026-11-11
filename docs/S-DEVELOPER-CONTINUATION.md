# S Developer continuation checkpoint

Read `prompts/S-UNIFIED-DEVELOPMENT-MASTER.md`, current Git state and related PRs
before resuming. Preserve all existing checkpoints. Owner remains Md Shiyam
Talukder. Repository changes do not prove installation on the owner's devices.

## Verified base

PRs #14–18 are merged. Main before this milestone was
`fa1f354f9b31569343177935f060f551459216a9` (PR #18). Existing requirements,
structured specifications, controlled generation/build/repair, rollback,
portable inference and independent evaluation interfaces remain preserved.
The old native checkpoint remains unchanged and failed 0/3 bundle checks in the
prior baseline; semantic correctness was unassessed.

## Current milestone

Branch: `feature/s-native-coding-curriculum`.

Implemented a reproducible self-authored synthetic curriculum, completion-masked
full-record training, fixed-validation checkpoint selection, checksum-verified
random/old/trained controls, unconstrained bounded greedy JSON decoding, exact
output replay and independent isolated arithmetic oracle. Added an explicit
operator decoder selection; sampled inference remains default. Candidate models
are separate opt-in experiment artifacts, not replacements.

Two predeclared runs completed: 1,500 and 2,000 CPU updates. Each independently
initialized model has 115,008 parameters. Selected steps are 400 and 100 based on
validation NLL. **Both produced 0/8 valid held-out source bundles; no candidate code
reached semantic execution and neither model is promoted.** Training loss and
lower validation NLL do not prove a working neural coder. v2 uses fresh held-out
constants and records its follow-up rationale; no further test-tuned run was made.

See `docs/S-NATIVE-CODING-CURRICULUM.md` and
`experiments/coding-curriculum-v{1,2}/` for exact plans, data, safe weights, hashes,
logs, raw generation failures and reproducible commands. Check the related PR
for current CI and merge status; never infer it from this checkpoint alone.

Local verification: 108 application tests, 801 neural tests and 25 UI tests passed.
Package 0.2.3 builds; entry-point smoke is recorded in the related PR. The local
host has no Docker; actual restricted builds/oracle fixture execution is checked
in GitHub CI. Exact frozen neural output replay and experimental quality results
are separate from infrastructure success.

## Next concrete work

Complete/inspect all CI, preserve actual execution evidence, integrate only the
reviewed tested tree. For a future learning experiment, investigate robust
free-running validation generation and operation/constant-sensitive metrics on
training/validation splits. Predeclare a fresh test set and compute budget.
Do not present memorization or teacher-forced NLL as held-out coding ability.
Continue master acceptance requirements, including useful real application
behavior and clean Windows installer H. Actual Android/PC upgrades, physical phone
inference, unrestricted autonomous host access and a general neural coder are
not claimed. Persist a concrete checkpoint at a genuine session limit.
