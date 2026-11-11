# S Developer continuation — independent coding evaluator

Read `prompts/S-UNIFIED-DEVELOPMENT-MASTER.md`, current Git state and the related
PRs before resuming. Do not restart existing phases or infer device installation
from GitHub changes. Owner remains Md Shiyam Talukder.

## Verified base

PR #17 passed all four required workflows and was merged into main at
`6a4191d9a9ca43944768e67bcacf14f378b82bc3`. It preserves the existing weights and
adds a stdlib small-model runtime/probe, numerical parity tests and the revised
master prompt. Actual phone inference remains unverified; the supplied screenshot
proved only requirements-page visibility. Earlier PRs #14–16 remain preserved.

## Active milestone

Branch: `feature/s-capability-evaluation`, based on PR #17 source commit
`7df7707bd457357a3d8afc05f72337338b8a6ece`.
Goal: independently judge generated function outputs, refuse false-success exit
codes, preserve provider/source provenance and keep withheld answers out of model
prompts/repair feedback. Add CLI and consented GUI evaluation without expanding
host access. Correct native prompt truncation instead of hiding context limits.

## Implementation and local evidence

`capability.py` contains three public synthetic diagnostic tasks. Expected values
are compared on the host; only one input is sent to the isolated fixed driver.
Candidate tests are not the oracle. Reports separate format success, initial/final
correctness, assessed tasks, runtime blocks and context blocks. One benchmark
repair maximum; no general promotion from this tiny public suite.

102 local app tests, 798 neural tests and 25 UI tests passed. Actual S neural
baseline: 0/3 valid source bundles, zero tasks independently executed; semantic
accuracy therefore unassessed. Exact unchanged weight hash, failed results and
commands are retained in `docs/S-CODING-CAPABILITY-EVALUATION.md` and
`reports/s-developer/native-coding-baseline-v1.json`.

The new CI script uses clearly labelled correct/wrong/exit-zero/repair source
fixtures with real Docker and host comparisons. These verify the evaluator,
not neural ability. Required current CI/merge evidence is in the feature PR;
check it before claiming all checks passed or this branch merged.

## Next concrete actions

Complete CI and integrate only the tested source when authorized checks pass.
For actual learning improvements, declare a separate data/model/context/compute
budget and held-out evaluation before training. Do not learn these exact public
benchmark answers and label memorization independent generalization. Preserve the
failed baseline and every source/model checkpoint. Continue master acceptance
checks, especially useful real project behavior and clean Windows installer H.
No full autonomous neural coder, ownership transfer, unlimited runtime or invisible
background work is claimed. Persist unfinished work before a real session limit.
