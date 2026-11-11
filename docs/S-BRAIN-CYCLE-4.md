# S-Brain — Cycle 4: measurable learning, prefix reuse, fair microbatching

## Purpose

Cycle 4 shifts the work from architecture alone to independently observable
learning. The milestone has three distinct engineering claims to test:
(1) weights change after gradient backpropagation, (2) logits from reused
prefixes match a full prefill, and (3) queued requests have bounded, fair
microbatch admission without replacing the existing S app.

This does **not** claim frontier-model intelligence or production deployment.
S currently uses an experimental byte-level decoder-only Transformer and must
still train on licensed, diverse data before generating useful language.

## Additions

| File | Purpose |
| --- | --- |
| `s_brain/prefix_reuse.py` | Store logits and KV states; skip prefill for exact repeats and compute only uncached suffixes on prefix hits |
| `s_brain/scheduler.py` | Fair owner round-robin, bounded queues, compatible microbatches, futures, cancellation and deadlines |
| `s_brain/learning_probe.py` | Inspect prediction/loss, gradients, parameter delta and validation metrics across optimizer steps |
| `s_brain/data_quality.py` | Dataset-size, duplication, repetition, leakage and potential secret-pattern screening |
| `continuum/s_brain_bridge.py` | Standard-library-only, localhost-only optional S-Brain client |
| `continuum/engine.py` | An explicit `s-brain:experimental` model routing branch, no fallback |
| `continuum/server.py` | Advertise S-Brain only when opt-in flag is set and the local brain is healthy |
| `s_brain/tests/test_cycle4.py` | Numerical parity, training and scheduling regression tests |
| `s_brain/tests/test_data_quality.py` | Quality gate and leakage regression tests |
| `tests/test_s_brain_bridge.py` | Existing S app route protection and opt-in enforcement |

## Why the prefix cache is genuinely different

Cycle 3 only stored KV snapshots and recomputed the full prefill for logits.
Cycle 4 stores the **last-token logits alongside the exact KV state**, enabling
a true exact-hit reuse. On a verified prefix of a longer prompt, the remaining
tokens are appended by the cached path rather than pre-filling the whole prompt.

The saved work is observable by cache-hit kind, matched token count and
uncached suffix length. Byte-token equality and weight fingerprint scope
must match before reuse; separate projects do not share entries by default.
This is not persistent disk memory; cache is process-local and evictable.

On reaching the context limit the cache still rebuilds relative positions as
required by the model's RoPE implementation. Cache hits are not a guarantee of
lower wall-clock time; use an actual benchmark on target hardware.

## Why scheduling is not yet full continuous batching

The new scheduler groups compatible queued requests by effective byte length
and executes each selected microbatch together. It enforces per-owner quotas,
global queue bounds and deadlines, with explicit start and shutdown.

Once a microbatch is executing, new requests wait for the next batch. This is
**fair microbatch admission**, not continuous injection into in-flight
autoregressive decoding. Fairness is at service selection, not strict latency
guarantees. Future work may add token-level continuous batching.

## Learning experiment

Use separate self-authored training and held-out files. Never copy secrets,
API keys, private customer data, confidential conversations or scraped
copyrighted material into a public GitHub repository.

On an approved Linux/WSL host with working PyTorch:

```bash
python -m pip install -r s_brain/requirements.txt
python -m s_brain.learning_probe --train local_data/train.txt --validation local_data/validation.txt --steps 50 --sequence 16 --output local_probe.json
```

The probe records prediction correctness fraction, pre- and post-update train
loss, gradient norm, L2 change in learned weights and validation loss.
A positive weight delta proves an update occurred. It **does not prove**
better reasoning; validation on genuinely unseen examples is essential.

Data quality check:

```bash
python -m s_brain.data_quality --manifest private_corpus/manifest.jsonl
```

This is a conservative screening tool, not an automated privacy guarantee or
a replacement for rights review and independent evaluation.

## Optional integration

S's current Continuum chat remains on `http://127.0.0.1:8765`; the
independent S-Brain service remains on `http://127.0.0.1:8766`.

By default S-Brain is disabled. For an operator-controlled local test:

1. Start S-Brain v2 with a **verified local checkpoint**:
   `python -m s_brain.server_v2 --checkpoint path/to/step-checkpoint`
2. In the PowerShell session that starts S, set:
   `$env:AXIOM_ENABLE_S_BRAIN='1'`
3. Start the original S Continuum server using its normal command.
4. Select `s-brain:experimental` only if the model appears in the model list.

S still has Ollama as its default. Selecting S-Brain does not automatically
switch the default. If the experimental service fails, an error is shown;
the program does not silently pretend Ollama is S's own learned model.

To disable: remove the environment flag or set it to `0` and restart S.
The UI may show limited-quality text because S is not yet instruction tuned.
This bridge accepts one prompt only; it does not claim genuine multi-turn
conversation training or complete tool access.

## Verification criteria

- All current S-Brain GitHub CI tests pass, including true weight deltas.
- Existing Continuum CI passes with optional route disabled by default.
- Reused-prefix logits numerically match normal full-forward logits.
- No cross-model or cross-project cached state is accepted.
- Scheduler queue limits, futures, and stopping are tested.
- Dataset screening reports clear accepted/rejected rules.
- No measured speedup or general intelligence is claimed without evidence.

## Next priority (Cycle 5)

Improve training-data diversity and deterministic validation, add genuine
teacher–student distillation with clearly labeled source provenance, develop
checkpoint-resume reproducibility and evaluation suites for unseen prompts.
Only expand S-Memory and S-Tools after S's own weight updates and independent
inference have been demonstrated against meaningful baselines.
