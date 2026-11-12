# S Developer continuation checkpoint

Owner: Md Shiyam Talukder. Preserve project boundaries and every prior verified
checkpoint. Read the master directive and related PR before resuming. A GitHub
merge does not prove installation on the owner's PC/phone.

## Preserved base

PRs #14–20 are merged. Main before this milestone is
`af6db819a6d710e1d7a00cee3e77ce7108f6c83a`. v3 SBrain still copies wrong constants;
PR #20 verified 8/8 arithmetic tasks only after a separate deterministic literal
repair. Its raw 0/8 and every earlier failure remain unchanged and preserved.

## Current milestone

Branch `feature/s-learned-byte-copy`: explicit learned source-copy generation,
without post-generation repair. Frozen 329,376-parameter independently trained
SBrain; independently initialized copy adapters on fresh synthetic CC0 corpora.

Byte attention/gate: 6,241 parameters, 1,200 updates, selected 1,100. Validation
8/8; held-out source-contract agreement 5/8, below its six-task target. Span
endpoint/gate follow-up: 18,819 parameters, 2,000 updates, selected 900. Validation
8/8; fresh held-out source-contract agreement 6/8. One span action, 32-byte limit,
explicitly labelled controller. No literal parser, expected answers, source patch
or repair at inference. Known wording templates are shared; no general coding or
unseen-template claim. Validation-only prototypes preceded the isolated final
span run; no held-out test output was used to retrain either frozen candidate.

`docs/S-LEARNED-COPY.md` and `experiments/learned-{byte,span}-copy-v1/` contain
plans, data, safe weights, hashes, logs and raw results. New `native-copy` provider
and explicit local adapter installation are available in CLI/GUI. Default model
and execution authority are unchanged. Full prompts exceeding context are refused.

Local: 120 app, 814 neural, 25 UI tests passed. New frozen outputs replay exactly;
package/installed launcher/persistence/stdlib smoke passes. Local Docker is absent.
Real Docker CI in PR #21 verified byte 5/8 and span 6/8 tasks correct with zero
repairs; base/random controls 0/8 in both. Span meets its six-task target; byte
misses it. Exact reports and artifact provenance are preserved in
`reports/s-developer/learned-copy/`. Span has 46/56 individual cases correct, with
two programs still failing. Inspect PR #21 for final Windows/Linux and merge
status. Never relabel this new controller as unassisted frozen-base improvement.

## Next concrete work

Complete real isolated seven-case-per-task acceptance and required CI; preserve
actual reports and integrate the tested tree. Never relabel copy-assisted results
as the unmodified base model's unassisted improvement. Do not retune the now-public
held-out constants. Remaining failures, general autonomous programming, Windows
clean-installer acceptance H and physical device upgrades remain open. No model
is automatically promoted based on this small public synthetic diagnostic.
