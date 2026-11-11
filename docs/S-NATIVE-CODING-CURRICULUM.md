# Independent native coding curriculum experiments

## Outcome and scope

This milestone implements reproducible, bounded training and independent coding
assessment using the existing SBrain byte Transformer. It does **not** deliver a
reliable neural coder. Both candidate checkpoints failed all eight held-out JSON
source-bundle checks. No generated candidate reached semantic execution. Neither
candidate is promoted, packaged as the default, or a substitute for the existing
working starter/build/repair interfaces.

All data is self-authored synthetic arithmetic, released as CC0-1.0. No external
model, teacher service, pretrained weights, or private data was used. These public
experiments measure narrow within-family transfer, not broad software engineering
or a secret benchmark. The original three public coding diagnostic tasks are not
used as training examples.

## Fixed experiments

The plan in each `experiments/coding-curriculum-v*/plan.json` was written before
its training run. The generator and compressed exact dataset are retained beside
safe random-control/selected weights, manifests, training logs, training report,
and all raw model outputs. Completion and prompt hashes are split-disjoint.
Training consumes only training rows. Validation uses a fixed full batch, not
random windows; test generation starts only after checkpoint selection is frozen.

| Experiment | Training rows | Validation / test rows | Budget | Selected update | Validation NLL: initial → selected | Valid held-out bundles |
|---|---:|---:|---:|---:|---:|---:|
| v1 | 240 | 8 / 8 | 1,500 updates | 400 | 56.65428 → 0.40014 | 0/8 |
| v2 | 1,104 | 8 / 8 | 2,000 updates | 100 | 59.96558 → 0.15964 | 0/8 |

Each model has **115,008 independently initialized parameters**, two layers,
width 64 and a 512-byte context. CPU training uses two threads, batch eight,
AdamW, warmup/cosine decay, finite-gradient checks and clipping. Completion-only
loss masks prompt/padding positions. Full records exceeding context are rejected.
The selected checkpoint is the lowest validation NLL sampled every 100 updates;
it is not selected by test score or final training loss. Wall times were about
137 s and 165 s on this host; these are observations, not performance guarantees.

The predeclared quality target was six independently correct tasks out of eight,
with three input cases per task. Both experiments miss that target at the format
gate. Lower teacher-forced NLL did not establish autoregressive coding ability.
The v2 follow-up adds an explicit answer boundary, broader constants and wholly
new held-out constants; its rationale was declared after the v1 failure and
before v2 training. No test-feedback repair/retraining loop was used.

The old demo and same-architecture random controls also produced zero valid
bundles on each experiment. All three models use the same unconstrained greedy
byte decoder and 200-token budget. It stops at the first complete JSON object;
it does not force syntax, strip a bad prefix, insert templates, repair output or
retry samples. Existing strict source-bundle validation remains mandatory.
A direct trained-model check confirmed full and cached next-byte choices agree;
invalid generation is not hidden by a cache or template fallback.

## Preserved checkpoint identities

| Checkpoint | SHA-256 |
|---|---|
| Existing demo, unchanged | `a6caeb1f219c01f42cec3ab54fb5dcdb61e20d6530dc91173a7805bdd7d89859` |
| v1 selected | `5c2deca97553c02c36a227f96fbce3459ae7c11045808a20db87743f0a1dbd1b` |
| v2 selected | `40434006382446f0595486a64e698d2a365fd67c430e1caf023725dd15ef624c` |

Architecture/seed/dataset/plan hashes and safe-weight checksums accompany each
checkpoint. No optimizer pickle is included. Training and evaluation refuse to
overwrite existing output directories/files. The compressed dataset is reproducible
with `records(plan)`; its report fingerprint is canonical record JSON, not gzip
bytes. Reproduction commands below create fresh experiment outputs.

## Reproduction

Use an existing environment with the optional neural dependencies (`S[neural]`).
No dataset/model download is required. Run from the repository root:

```sh
python scripts/train_coding_curriculum.py --plan experiments/coding-curriculum-v1/plan.json --output build/reproduce-v1
python scripts/train_coding_curriculum.py --plan experiments/coding-curriculum-v2/plan.json --output build/reproduce-v2
python scripts/evaluate_coding_curriculum.py --experiment experiments/coding-curriculum-v1 --verify experiments/coding-curriculum-v1/generated.json
python scripts/evaluate_coding_curriculum.py --experiment experiments/coding-curriculum-v2 --verify experiments/coding-curriculum-v2/generated.json
```

Training writes `random-control/` and `selected/`. For evaluating a new run,
assemble a **new** experiment directory with its plan/training report and these
two directories under `checkpoints/`, then call evaluation with `--output` naming
a new file. Checkpoint metadata timestamps/runtime may differ on reproduction;
CI replay checks the published weights and exact generated bytes, not identical
wall times or training NLL across different hardware/PyTorch releases.

Actual isolated semantics requires Docker and explicit runtime preparation:

```sh
python scripts/verify_coding_curriculum.py --experiment experiments/coding-curriculum-v1 --approve-runtime-download --report build/execution-v1.json
python scripts/verify_coding_curriculum.py --experiment experiments/coding-curriculum-v2 --approve-runtime-download --report build/execution-v2.json
```

Only a fixed driver and validated candidate files enter the restricted container.
The host oracle independently computes negative/zero/positive input results; the
model receives no case inputs or expected answers. Candidate test files cannot
approve their own code. There is no host execution fallback. Reports distinguish
format failure from actual execution and preserve diagnostics/image provenance.
CI verifies checkpoint-output replay separately from container/oracle integrity;
**a green infrastructure job does not mean the experimental quality target passed**.
Here all candidate bundles fail format, so no neural source is executed. Existing
correct/wrong/repair fixture tests still exercise real Docker and the host oracle.

## Optional operator-controlled decoder

Existing sampled native inference remains the default. A local operator may opt
into bounded greedy JSON decoding with the PyTorch backend:

```sh
S model configure-runtime --backend torch
S model configure-decoding --decoding greedy-json
S model configure-decoding --decoding sampled
```

Install a particular experiment with `S model install-native --checkpoint
experiments/coding-curriculum-v2/checkpoints/selected` only if deliberately
investigating it. Installation uses the existing content-addressed model store;
it is not a capability endorsement. These candidates are not useful production
coders. The 512-byte models exceed the portable runtime's supported context and
are **not verified on Android**. Full application requirements may exceed their
context; S must report that limit, never discard requirements.

## Next learning decision

Preserve both negative results. The immediate research problem is robust
free-running source format and semantic transfer, rather than lower average
teacher-forced boilerplate loss. Before another run, declare a new dataset/model
budget and separate test split; consider validation generation metrics and
operation/constant-sensitive diagnostics on training/validation data. Do not
retune these now-public test answers and call that independent generalization.
Full autonomous coding, clean Windows installer acceptance H and physical-device
installation remain unverified.

## Published verification evidence

PR #19 code commit `baacd32db3ff80a4385e583149da5ed5a294d40b`
(source tree `60a3b9a3e807af9b1dda6d80dfd96c6de3e6f60a`) passed all four
workflows: S Developer foundation (Windows/Linux package and interface checks),
S-Brain experimental checks (801 tests and exact frozen-output replay), AxiomFlow
Continuum matrix, and controlled development acceptance.

Actual Docker reports are preserved under
`reports/s-developer/native-coding-curriculum/`. Their provenance file records
the source commit, workflow/job/artifact IDs and verified archive SHA-256.
[Container execution log](https://github.com/shiyamprimetrade-sketch/S/actions/runs/43069519135)
and [neural replay log](https://github.com/shiyamprimetrade-sketch/S/actions/runs/43069519308)
remain separate evidence. Both experimental reports explicitly record
`target_met: false` and `improved_over_controls: false`. Zero candidate programs
executed because none passed format. The real container fixture oracle accepted
correct code 3/3, rejected exit-zero and constant-answer attempts 0/3, and verified
fixture repair from 0/3 to 3/3. Those successes are never attributed to the native
neural checkpoints. Documentation/evidence-only follow-up commits do not alter
the tested executable source or weights.
