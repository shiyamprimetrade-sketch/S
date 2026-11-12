# S learned source-copy generation

## Purpose and scope

This milestone tests faithful literal transfer during generation, without the
post-generation deterministic repair used in PR #20. It preserves the existing
SBrain architecture/weights, all previous checkpoints, S-Memory/S-Tools and
permissioned build interfaces. Neither new adapter becomes the default.

Two explicitly different neural copy mechanisms are evaluated. These are
self-authored synthetic, known-template arithmetic experiments, not general
software-development benchmarks. All prompt/complete-answer hashes and constant
values are disjoint across splits. The new constants are outside earlier runs.
Wording templates are shared; no unfamiliar-English generalization is claimed.

## Mechanisms and frozen experiments

Both adapters use the unchanged, independently trained v3 SBrain checkpoint:
`be18814895673f2a80075e0d2baedbda9277f927291ac614b7d3b2ebaa344711`.
Its 329,376 parameters are frozen, excluded from optimization and checked again
after training. No external model weights or teacher/model service is used; the previously trained SBrain base is reused locally.
The new adapters are initialized independently and trained on synthetic CC0 data.

| Experiment | Adapter parameters | Update budget / selected | Validation contracts | Held-out source-contract agreement |
|---|---:|---:|---:|---:|
| Learned byte attention/gate | 6,241 | 1,200 / 1,100 | 8/8 | 5/8 |
| Learned span endpoints/gate | 18,819 | 2,000 / 900 | 8/8 | 6/8 |

The first method mixes the frozen model's next-byte distribution with a learned
attention distribution over all prompt bytes, using a learned copy/generate gate.
It receives no digit mask or extracted number at inference. Its six-task target
was missed; the failure and all raw outputs are retained.

The follow-up predicts source start/end positions and a copy-action gate. A
declared controller allows **one span-copy action of at most 32 bytes**; ordinary
bytes come from frozen SBrain. Start/end/gate are learned, not a literal parser.
Invalid selected spans fail rather than invoking a fallback. The controller is
an explicit architectural change, not ordinary unassisted SBrain decoding.

At inference, neither method receives structured constant values, expected code,
case inputs, expected answers, number annotations or ASTs. There is no regex
number extraction, forced Python/JSON syntax, output patch, repair or fallback
source template. Training-only source labels use known synthetic literals;
validation chooses checkpoints by free-running contract agreement, then valid
bundle count, then the declared validation objective. Test generation starts
only after weights are frozen. No test-tuned retraining follows either result.

Each final training corpus has 3,920 records, eight validation records and eight
test records. Byte tests use 413/487; span tests use fresh 593/683. Validation-only
linear span prototypes were not tested. A metadata selection-label correction
and isolated nonlinear-head rerun happened before span test generation; final
plans, exact compressed corpora, reports, logs, weights and raw outputs are saved
under `experiments/learned-{byte,span}-copy-v1/`.

## Checkpoint identities

| Artifact | SHA-256 |
|---|---|
| Selected byte adapter | `2c4c89e4ddad1798e27b9c03097abf586871b036ed07654c1eae0988d435cd19` |
| Selected span adapter | `88d846a7d4e5d0cc4377e69302697d7f8dbd6f724f9a0c028a4bfe531cd0e2d7` |

Base and random-adapter controls use the same complete requests and output budget.
Plain SBrain decoding remains a separately labelled control. Random and trained
adapters share their respective controller; model origin and decoder changes are
never hidden. Reports distinguish source-format/AST checks from actual isolated
semantic execution. Six source matches alone are not a runtime success claim.

## Local operation

Optional neural dependencies must already be installed on a supported host.
Install the required base and then the selected adapter explicitly:

```sh
S model install-native --checkpoint experiments/coding-curriculum-v3/checkpoints/selected
S model configure-runtime --backend torch
S model install-copy-adapter --adapter experiments/learned-span-copy-v1/selected
S develop application
```

The graphical workspace offers `S Native + learned copy — experimental arithmetic`;
CLI generation/evaluation supports `--provider native-copy`. Installation is a
checksum-bound local copy, never a model download or capability endorsement.
The provider is experimental and complete project prompts may exceed its 512-byte
context. Such prompts are refused without discarding requirements. The existing
requirements editor, approval, revision, diagnostics and rollback rules remain.
This is not a general application generator. Android inference/install and clean
Windows installer acceptance H remain unverified.

Safe manifests bind an adapter to an exact base SHA, architecture and safe-weight
checksum. Wrong-base, random-control installation, corruption, symlinked targets
and oversized architectures are rejected. No optimizer pickle is loaded. Model
installation is host-operator CLI only; project instructions cannot change it.
Adapters do not grant execution, filesystem, dependency or network permissions.
Native-copy inference requires PyTorch and has no portable/backend fallback.

## Reproduction

From the repository root, use new output directories/files:

```sh
python scripts/train_byte_copy.py --plan experiments/learned-byte-copy-v1/plan.json --output build/reproduce-byte
python scripts/train_span_copy.py --plan experiments/learned-span-copy-v1/plan.json --output build/reproduce-span
python scripts/evaluate_byte_copy.py --experiment experiments/learned-byte-copy-v1 --verify experiments/learned-byte-copy-v1/generated.json
python scripts/evaluate_byte_copy.py --experiment experiments/learned-span-copy-v1 --verify experiments/learned-span-copy-v1/generated.json
python scripts/verify_byte_copy.py --experiment experiments/learned-byte-copy-v1 --approve-runtime-download --report build/byte-copy-execution.json
python scripts/verify_byte_copy.py --experiment experiments/learned-span-copy-v1 --approve-runtime-download --require-target --report build/span-copy-execution.json
```

The verifier never repairs programs. Seven independent host-oracle cases per task
include negative/zero/positive and divisor-boundary inputs. Every candidate is
executed only in the existing digest-pinned restricted Docker runner, with no
host fallback. The span CI gate requires the predeclared six-task target and
improvement over both controls. A byte-experiment diagnostic does not relax its
missed target. The local host lacks Docker; actual results require CI or a
suitable isolated host.

Local validation: 120 application tests, 814 neural tests, 25 UI tests; all passed.
Both new frozen reports replay exactly. Package 0.2.5 and installed launcher,
packaged UI, persistence and stdlib-inference smoke passed. The related PR records
current actual execution/Windows/Linux CI and merge status. Remaining inaccurate
span outputs and the failed byte candidate are kept visible for future work.

## Actual zero-repair runtime evidence

Code commit `a6fbe7fa6241b98fc0280ac0679e66e26d02865d` (PR #21) passed
[restricted Docker execution](https://github.com/shiyamprimetrade-sketch/S/actions/runs/43219172983).
The byte adapter passed **5/8 tasks**, below its declared target, and the span
adapter passed **6/8 tasks**, meeting its six-task target. In both experiments,
the unchanged base and random-adapter controls passed **0/8**. **No repairs were
performed.** The span result has 46/56 individual cases correct; six complete
programs each passed all seven cases. Two programs for 683 still emit 686 and
remain failed. These are not hidden, patched or excluded from the denominator.

Exact case inputs, expected/actual outputs and statuses are preserved under
`reports/s-developer/learned-copy/`, with the source/tree, workflow/job/artifact
IDs and checksum-verified archive digest. The neural CI independently replayed
both frozen reports. A follow-up hardens origin labels so an adapter of unknown
or external origin cannot inherit the base model's independent-training claim;
this changes provenance metadata, not trained weights or generation outputs.
The local registered `native-copy` provider also reproduced a frozen bundle
through the content-addressed installer without executing source on the host.

These results establish narrow copy-assisted generation without repair, not an
improvement in the frozen base's unassisted score or general coding reliability.
Check PR #21 for final Windows/Linux status and merge evidence.
