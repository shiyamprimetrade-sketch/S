# S Developer continuation checkpoint

Owner: Md Shiyam Talukder. Read the master directive, current Git state and related
PR before resuming. Never infer device installation from a GitHub merge.

## Preserved base

PRs #14–19 are merged. Main before this milestone is
`2947aa80b797295c8a32e7f7d612083a79ec85ec`. All prior architecture, memory, tools,
interfaces, model checkpoints and tests are preserved. v1/v2 coding experiments
failed all eight source-bundle format checks; their complete evidence remains.

## Current milestone

Branch: `feature/s-generation-validation`.

Fixed NLL-only model selection by adding free-running validation source/contract
checks. Added semantic-token training weights and arithmetic boundary inputs to
catch incorrect divisors. New v3 run: 329,376 random-initialized parameters,
2,400 updates, selected checkpoint step 1,200 by validation generation metrics.
Held-out constants are fresh; wording templates are shared. This is a narrow
arithmetic experiment, never a claim of general programming ability.

Actual held-out generation: candidate 8/8 valid bundles; old and random controls
0/8. All eight raw candidate programs copy an incorrect constant. Their original
source is frozen. No test-tuned retraining is allowed. A separately predeclared
one-step deterministic literal repair copies only an explicit approved numeric
requirement, preserving the neural operator. Initial/final execution and repair
provenance must remain separate; never attribute repaired results to untouched
neural generation. No model is automatically promoted.

See `docs/S-GENERATION-VALIDATION-FIX.md` and `experiments/coding-curriculum-v3/`.
Local verification: 115 app tests, 807 neural tests and 25 UI tests passed;
old/new frozen outputs replay; package/installed launcher smoke passes. Local
Docker is absent. Inspect the feature PR for real isolated semantic results,
CI, published evidence and merge status before claiming those checks passed.

## Next action

Finish CI and preserve actual reports; integrate only the tested tree. Retain the
pure-neural quality target even if the separately labelled deterministic repair
succeeds. Future independent training must address literal-copy reliability with
fresh held-out data and predeclared budgets. Full general autonomous coding,
Windows clean-install acceptance H and physical Android/PC upgrades remain open.
