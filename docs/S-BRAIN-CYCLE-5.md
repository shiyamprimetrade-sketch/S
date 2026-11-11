# S-Brain — Cycle 5: Real Training and Temporary Ollama Teacher

## Purpose and boundaries

S-Brain is an independently trainable byte-level Transformer. Ollama serves
as a **temporary optional data-generation teacher**. It never supplies S's
inference responses when the independent model is selected, and the S-Brain
training module does not require Ollama to run. This cycle builds a measured,
reviewable chain from approved training questions to actual gradient updates.

Training is not equivalent to deploying an advanced general intelligence.
The user has not yet provided a licensed large-scale corpus or trained S
weights. Tests use tiny self-authored fixtures only.

## What was coded

| Module | Capability |
|---|---|
| `s_brain/teacher_data.py` | Loopback-only Ollama question/answer generator with consent metadata, response hashing and explicit review |
| `s_brain/supervised_data.py` | Byte-level prompt completion examples, mask all non-answer targets, audit and dataset fingerprints |
| `s_brain/train_supervised.py` | AdamW optimizer, gradient accumulation, prediction error, measured weight changes, fixed held-out validation, baseline/best/final weights |
| `s_brain/evaluate_learning.py` | Independent teacher-forced held-out likelihood, token-level accuracy, category breakdown, errors and bootstrap confidence bounds |
| `s_brain/tests/test_cycle5_data.py` | Teacher isolation, metadata, review state, JSONL integrity and loss-mask tests |
| `s_brain/tests/test_cycle5_learning.py` | Real optimization steps, checkpoint integrity, baseline/candidate evaluation and regression thresholds |

## Learning contract

Training examples are represented as:

```text
Question: <human-reviewed question>
Answer: <reviewed answer>
<END>
```

S sees the question and previous bytes as its input. Loss is applied only
to the answer bytes. The saved weight update is measured through the L2
distance between parameters before and after AdamW. Every checkpoint is
bound to the training/validation dataset fingerprints and stores SHA-256
checksums for weights. The best validation checkpoint is saved **only**
when it performs better than the untrained baseline on the same samples.

This is **supervised response distillation**: teacher-generated answer text
is training data. It is not the stronger knowledge-distillation method
involving the teacher's full next-token probability distribution. Ollama
does not expose those teacher logits through the ordinary chat endpoint.

## Review before training

Teacher-generated text may contain errors, hallucinations, rights issues
or personally identifying material. The teacher questions require
`source` and `permission` attributes. All answers start in `pending`
review status and must be independently checked and marked `approved`
before admission to training. Holdout questions are excluded from
teacher generation by default. Do not commit raw training data,
answers, secrets or user conversations to a public GitHub repository.

Human-authored blind validation data must be collected separately.
A future model benchmark should use questions and expected answers that
were not shown to the teacher or student during training.

## Development flow

From the checked-out S repository root and a PyTorch-capable Linux/WSL host:

```bash
python -m pip install -r s_brain/requirements.txt
python -m unittest discover -s s_brain/tests -v
```

Create a local JSONL file of questions with these fields:

```json
{"identifier":"q-001","question":"What is a neural network?","category":"machine-learning","source":"self-authored","permission":"user-owned","split":"train"}
```

Run local Ollama temporarily to generate **candidate responses**:

```bash
python -m s_brain.teacher_data --questions private_questions.jsonl --output private_teacher_candidates.jsonl --model qwen2.5:1.5b
```

The output is intentionally in `pending` review state. Manually verify,
correct/reject and mark appropriate examples `approved` through
`review_example()` or an audited review interface. A textual edit must
recompute answer SHA-256 to satisfy integrity verification.

Convert reviewed examples:

```bash
python -m s_brain.supervised_data --approved-examples private_reviewed.jsonl --output private_train.jsonl
```

Separately create human-authored holdout responses using
`encode_record(...,split='validation')` and `export_jsonl()`.
Do not derive these from the same teacher prompts.

Run a limited, explicit training experiment:

```bash
python -m s_brain.train_supervised --train private_train.jsonl --holdout private_holdout.jsonl --output private_sbrain_experiment --steps 100 --context 64 --dim 128 --layers 2 --device cpu
```

The output directory contains a baseline checkpoint, best checkpoint when
validation improves, final checkpoint, metrics and an experiment report.
The checkpoint files are local and ignored by Git where placed under
`s_brain/checkpoints/`. Be careful with other output locations.

Evaluate real checkpoint progress independently:

```bash
python -m s_brain.evaluate_learning --baseline private_sbrain_experiment/baseline --candidate private_sbrain_experiment/final --holdout private_holdout.jsonl --output private_evaluation.json
```

## Meaning of completion evidence

- **Working training code:** all tests, including optimizer steps, pass CI.
- **Weights changed:** measured parameter delta is strictly positive.
- **Model improved:** held-out loss improves against the baseline on the
  same data. Training loss decline alone is not sufficient.
- **Independent:** trained checkpoint runs without Ollama.
- **Useful intelligence:** requires much larger and representative
  benchmarks covering reasoning, factuality, coding and tools.
- **Gemini-class:** no claim until reproducible, comparable evaluation
  evidence exists against frontier reference systems.

## Platform notes

The Windows Python installation previously showed Code Integrity events
blocking `torch.dll`. GitHub CI can verify small Linux-based experiments
but cannot train the production model on the user's Windows PC. Keep Smart
App Control enabled; use a trusted, compatible development environment
(approved WSL/Linux or an authorized machine).

## Next milestone

Cycle 6 should expand independently reviewed data and formalize learning
curves, dataset-version management and curriculum selection. Then add
S-Memory retrieval and tool use behind explicit permissions. Prefer
demonstrated holdout improvement over adding artificial code volume.
