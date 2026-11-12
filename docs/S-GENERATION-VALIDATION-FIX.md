# S generated-code validation and bounded literal repair

## Measured defect and correction

PR #19's checkpoints reproduce their recorded validation loss after safe reload;
the observed generation failure was not a lost-weight or cache-load defect.
The selection policy measured teacher-forced average byte loss, which did not
establish valid free-running code. This upgrade adds generated validation bundle
checks and arithmetic contract AST agreement. Checkpoint rank is contract matches,
then valid bundles, then lowest unweighted validation NLL. AST inspection never
executes model source; actual acceptance still requires isolated execution.

A second defect was oracle coverage: the original small inputs cannot distinguish
some wrong floor divisors. v3 adds `k-1`, `k`, `-k-1`, and `2*k` to negative,
zero and positive probes. Wrong adjacent divisors must fail a boundary case.
v1/v2 plans, outputs and original evidence are preserved unchanged.

## New fixed experiment

`experiments/coding-curriculum-v3/plan.json` was declared before training.
It uses the existing SBrain architecture with width 96, three layers, 512-byte
context and **329,376 independently initialized parameters**. No pretrained
weights, external teacher or model service was used. The exact synthetic CC0
corpus, safe weights, immutable plan, training log and raw outputs are retained.

Training uses 1,920 records and eight fixed validation records. Eight test records
use entirely new constants, 257 and 293; validation constants are 239 and 281.
All constants are outside v1/v2. Prompt and complete answer hashes are split
disjoint, but wording templates are shared. This is **constant transfer within
known arithmetic templates**, not unseen-paraphrase or general coding ability.

The CPU run completed its 2,400-update budget in about 461 seconds. Completion
training weights operation/constant bytes by eight, retaining the rest of the
source. The historical `train_nll` field is this weighted optimizer objective;
`validation_nll` is ordinary unweighted completion cross entropy. Checkpoint
selection does not use test outputs. Step 1,200 was selected with three validation
AST contracts matched; later lower-loss/invalid-contract candidates cannot outrank
it simply through boilerplate prediction.

Selected SHA-256:
`be18814895673f2a80075e0d2baedbda9277f927291ac614b7d3b2ebaa344711`.
The old demo and random-weight control each emit 0/8 valid bundles. The candidate
emits **8/8 valid bundles** using the same unconstrained greedy JSON decoder,
without forcing syntax, inserting source templates or repairing generation.
All eight raw programs nevertheless copy the wrong numeric constant. Raw test
outputs are frozen; no test-tuned training run follows. Semantic execution is
assessed separately by CI, not inferred from these format results.

## Explicit bounded repair

After validation exposed wrong constants and **before test generation**, the
separate `repair-plan.json` declared one optional deterministic literal repair.
It does not change the original six-of-eight unassisted neural target. A separate
supplementary target requires all eight final programs correct after repair.
Reports must show initial and final outcomes separately.

`requirements-literal` can change exactly one positive integer literal, 1–999,
using the explicit approved English arithmetic requirement. It preserves the
model-generated operator, parameter, function, keys, files and all other source.
Only one-expression `solve(p)` arithmetic grammar is accepted. Imports, calls,
extra statements, unsupported operators and unknown paths are refused. No source
is executed by repair. Ambiguous requirements, unchanged literals and unsupported
shapes stop the attempt; there is no fallback code generator.

The provider is labelled **deterministic repair**, not independent neural output.
It receives the original requirement, not test inputs, expected answers or oracle
diagnostics. It cannot correct a wrong operator. Initial and repaired hashes,
source, outcomes and repair provenance remain distinguishable. The existing
engine preserves artifact revisions, specification approval, consent, host
execution opt-in, sandbox restrictions and bounded retry rules.

For an approved supported Python project with meaningful independent tests:

```sh
S project build --project PROJECT_ID --enable-builds --consent --max-repairs 1 --repair-provider requirements-literal
```

This is a repair-only provider. It cannot generate a project or turn a successful
process exit into business acceptance. The build engine's toolchain status still
requires independent requirements verification. The repair is not enabled by
untrusted project text or by a model; an operator selects it explicitly.

## Reproduction and verification

From the repository root with optional neural dependencies already available:

```sh
python scripts/train_coding_curriculum.py --plan experiments/coding-curriculum-v3/plan.json --output build/reproduce-v3
python scripts/evaluate_coding_curriculum.py --experiment experiments/coding-curriculum-v3 --verify experiments/coding-curriculum-v3/generated.json
python scripts/verify_coding_curriculum.py --experiment experiments/coding-curriculum-v3 --approve-runtime-download --approve-requirements-literal-repair --report build/v3-execution.json
python -m unittest discover -s tests -v
python -m unittest discover -s s_brain/tests -v
```

The verifier runs seven cases per valid raw program in the existing restricted
Docker runner. A failed program may receive the one declared repair, then seven
fresh executions. Case inputs/answers stay in the host oracle. Reports retain
`initial_tasks_correct`, final `tasks_correct`, raw `target_met`, and the distinct
`repair_target_met`. A green infrastructure workflow is not a raw-neural success
claim. There is no host execution fallback; this local executor has no Docker.

Local checks: 115 application tests, 807 neural tests and 25 UI tests passed.
Both old frozen neural experiments replay exactly, and the new output also
replays exactly. Package 0.2.4 builds and installed launcher/persistence/stdlib
inference smoke passes. Actual runtime results and current CI/merge state must
be checked in the related PR before claiming execution or delivery success.

No default model is replaced or automatically promoted. These 512-context models
are not supported by the current stdlib portable runtime; physical phone/PC
installation and general autonomous coding remain unverified. The next independent
learning gap is faithful literal copying without deterministic repair.

## Actual isolated outcome

Code commit `1a7790c49c718e112181c5effaea8a73eeb17ede` (PR #20) was exercised
in [real Docker CI](https://github.com/shiyamprimetrade-sketch/S/actions/runs/43193762090).
The candidate passed **0/8 tasks before repair and 8/8 after repair**, with **56/56
independent final cases passing**. Every task used exactly one declared literal
repair. The old and random controls remained 0/8. The raw neural target remains
false; the distinct repair target is true. No score is attributed to a hidden
neural fix or external model.

The complete initial/final cases, repaired source and provenance are preserved in
`reports/s-developer/generation-validation-v3/execution.json`. Its companion
provenance records the exact code/tree, workflow/job/artifact IDs and verified
archive digest. CI now explicitly requires the predeclared repair acceptance
target, while keeping unassisted quality separate. The ordinary website/Python/C
build, fixture repair and isolation regressions also passed in that Docker job.
The neural workflow independently reproduced the frozen outputs. Inspect PR #20
for final Windows/Linux matrix and merge status.
