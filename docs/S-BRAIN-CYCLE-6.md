# S-Brain engineering checkpoint — Cycle 6

## Objective

Build an audited pipeline from **authorized real-world data** to reviewed local
Ollama teacher responses, deterministic curriculum sampling, reproducible
learning-curve experiments and independent checkpoint selection.

Cycle 6 adds machinery and quality gates. It does **not** upload a real-world
training corpus, create live teacher samples or prove Gemini-level performance.
The experimental weights remain separately trained; no new model is promoted
or deployed automatically.

## Added code

| Module | Purpose |
|---|---|
| `s_brain/curation_v2.py` | Offline curation of source text; license/permission metadata, explicit reviews, exact and near-duplicate screening |
| `s_brain/teacher_review.py` | Explicit reviewer approval, hash-locked review actions and auditable teacher-example ledger |
| `s_brain/curriculum_v2.py` | Deterministic easy/medium/hard sampling schedules and category weights |
| `s_brain/learning_curves.py` | Independent seed/step-budget runs, reproducible metadata, loss summaries and confidence intervals |
| `s_brain/checkpoint_selection.py` | SHA-256 checked checkpoint quality gates and explicit operator promotion |
| `s_brain/tests/test_cycle6.py` | Regressions for each of the modules, including negative quality and privacy cases |

All earlier cycles and Continuum behavior are preserved. Existing PR #14
is kept in draft until independent real-world model evaluation is established.

## What counts as high-quality training evidence

Use owned, permissioned and reviewed text, with train/validation/test split
boundaries. The audit can detect exact and overlapping word n-grams, but not
all semantic paraphrases or intellectual-property problems. Human review
must independently verify source rights, accuracy, relevance and the
absence of sensitive data.

For teacher supervision, create candidate answers only using the
locally running Ollama service. `teacher_data.py` from Cycle 5 already
implements that step. Cycle 6 adds review workflow support.

### 1. Curate a private source corpus

Create a locally stored JSONL file containing approved source records.
Example (self-authored text only):

```json
{"identifier":"doc-1","text":"A transformer is a neural architecture that computes attention over tokens.","source":"self-authored","permission":"user-owned","split":"train","category":"machine-learning","origin":"human","review":"approved"}
```

Run:

```bash
python -m s_brain.curation_v2 --input private_sources.jsonl --accepted private_accepted.jsonl --audit private_curation_audit.json
```

The audit contains hashes and metadata, not the source text.

### 2. Generate temporary teacher examples

Use the existing Cycle 5 tool with an allowed, already installed local
Ollama model:

```bash
python -m s_brain.teacher_data --questions private_questions.jsonl --output private_candidates.jsonl --model qwen2.5:1.5b
```

This does not require the user to install or run Ollama at inference time.
Responses start in `pending` status. The tool does not automatically judge
them factual or permitted for training.

### 3. Review teacher outputs

Create explicit JSONL actions referencing the current SHA-256 of each
question and answer. Never auto-approve teacher-generated text.

```json
{"identifier":"q-1","decision":"approve","reviewer":"human-reviewer","reason":"Checked accuracy, originality and permission","expected_answer_sha256":"<64-hex-digest>","expected_question_sha256":"<64-hex-digest>"}
```

Apply:

```bash
python -m s_brain.teacher_review --candidates private_candidates.jsonl --actions private_actions.jsonl --output private_reviewed.jsonl --ledger private_review_ledger.jsonl
```

Approved response data can be converted to the supervised byte format
using `s_brain.supervised_data` from Cycle 5. Holdout questions should
be independently authored and kept separate.

### 4. Audit curriculum

```bash
python -m s_brain.curriculum_v2 --train private_train.jsonl --context 64 --steps 100
```

This generates a deterministic sampling **plan**, not a trained model.
Integrating this sampler into the training loop requires a later explicit
model-training experiment and comparison against uniform sampling.

### 5. Run learning-curve experiments

The following command trains the independent S model from scratch for
each seed/budget combination:

```bash
python -m s_brain.learning_curves --train private_train.jsonl --holdout private_holdout.jsonl --output private_learning_curve --seeds 11 29 --budgets 10 30 --context 64 --dim 128 --layers 2
```

Keep the dataset, seed, optimizer, architecture, compute host and
evaluation code version constant when comparing runs. These experiments
are potentially expensive; GitHub CI covers the code through small
deterministic tests, not full-scale real-data training.

### 6. Checkpoint selection

For each baseline/candidate checkpoint, run an **independent** evaluation
and preserve the same test-set fingerprint. Use the selection utility:

```bash
python -m s_brain.checkpoint_selection --baseline path/to/baseline --candidate path/to/candidate --before path/to/baseline_eval.json --after path/to/candidate_eval.json --report private_selection.json
```

The selector requires SHA-256-valid model weights, checkpoint progress,
sufficient holdout coverage, improved loss, no unacceptable accuracy
regression and matching training/evaluation provenance. It never changes
the S chat application, promotes weights to production or bypasses
operator approval.

## What is verified versus still pending

**Software verification:** GitHub Actions runs the S-Brain regression
suite on Linux/Python 3.12 with CPU PyTorch, and separate Continuum tests.

**Not verified by software tests alone:** a licensed real training corpus,
human review of Ollama's output, real-world improvements in S reasoning,
correctness on unseen external benchmarks, multimodal ability or live
deployment on the owner's Windows computer.

The Windows Code Integrity block on `torch.dll` remains a separate host
restriction. Do not disable Smart App Control to run this code.

## Cycle 7 proposed focus

Hook curriculum sampling into the actual optimizer loop; add honest
uniform-versus-curriculum comparisons, data provenance version control,
and non-synthetic benchmark suites. Once improvement is measured on real
held-out tasks, develop S-Memory and controlled S-Tools with their own
evaluation gates.
