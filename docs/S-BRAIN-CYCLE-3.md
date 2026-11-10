# S-Brain Cycle 3 — Batched Inference and Integration Readiness

## Project boundaries

The existing S Continuum application and its Ollama bridge remain unchanged.
The independent experimental model lives under `s_brain/` on the current
feature branch. Nothing in this cycle enables it in the public UI or exposes
its inference endpoint to an external network.

This milestone has four engineering objectives:

1. Execute multiple same-length prompts in one causal-attention batch.
2. Manage KV-cache snapshots with bounded memory accounting.
3. Improve the integrity of benchmark and regression evaluation.
4. Provide reversible, explicit opt-in compatibility contracts for future S integration.

## New modules

| File | Function |
|---|---|
| `s_brain/batching.py` | Same-length grouped prefill, batched sampling, dynamic row compaction |
| `s_brain/cache_pool.py` | LRU and TTL cache accounting, identity checks and explicit eviction |
| `s_brain/evaluation_suite.py` | Duplicate prompts, training overlap, paired accuracy bootstrap, provenance |
| `s_brain/integration.py` | Disabled-by-default bridge policy, readiness audit and model router |
| `s_brain/batch_benchmark.py` | Batched-versus-single timing, output checks, environment metadata |
| `s_brain/tests/test_cycle3.py` | Cache, batching, evaluation and integration regression tests |
| `s_brain/tests/test_batch_benchmark.py` | Benchmark reproducibility and compatibility tests |

## Correctness decisions

**Padding:** Variable-length prompts are grouped by their *effective byte
length* before batch prefill. This avoids assigning incorrect RoPE positions
or accidentally treating padded bytes as real training or inference tokens.

**Completion:** Finished requests are removed from active batch tensors,
KV states and sampling loops. Remaining requests keep their associated
independent random number generators. Each output returns to the original
request position and identifier.

**Context rollover:** When cache context fills, a full prefill is used on
the current window. That is necessary for numerical parity with the
original model's relative RoPE scheme. It is not as efficient as a paged
attention implementation with trained absolute-position handling.

**Cache pool:** Keys contain model fingerprints and full token digests.
Hits can be independently checked against full byte-token histories.
The cache pool is a storage/quota prototype; its current `cached_prefill`
function still recomputes logits. Thus, **a cache hit does not imply faster
inference**. No speed claim should be made without benchmark evidence.

## Training and inference remain separate

Existing training and checkpoints are preserved from Cycles 1 and 2.
New batching and cache-pool files do not modify learned model weights,
training data, optimizers, user data, or the main Continuum database.

No model has Gemini-level capabilities merely because the architecture
uses grouped-query attention, batching or a KV cache.

## Run on an approved Python and PyTorch host

From the S repository root:

```powershell
python -m pip install -r s_brain/requirements.txt
python -m unittest discover -s s_brain/tests -v
```

The user's Windows Smart App Control issue blocking `torch.dll` is separate.
Use an approved Windows policy configuration or WSL/Linux training host;
the code does not bypass or weaken host security.

## Batched performance experiment

After creating a valid advanced checkpoint:

```powershell
python -m s_brain.batch_benchmark --checkpoint s_brain/checkpoints/advanced/step-00000100 --batches 1 2 4 --prompt-bytes 24 --tokens 12 --warmups 1 --repeats 3 --output private-batch-results.json
```

This executes the same synthetic workload in single and batched modes,
records latencies and compares generated content digests. It is a timing
experiment on the actual host, **not** a guarantee that batching is faster.

Use separate homogeneous and heterogeneous prompt experiments to understand
grouping overhead. Short prompts and tiny models can perform worse with
batching due to attention-launch and cache-compaction overhead.

## Opt-in integration readiness

Create a disabled policy template:

```powershell
python -m s_brain.integration --create-disabled-template --config s_brain/local_integration.json
```

The generated JSON has `"enabled": false`. No automatic fallback to
Ollama, no modification to `continuum/engine.py`, and no unsupported
multi-turn history behavior are introduced by this cycle.

Once the independent service works, operator can explicitly enable the
local policy and run readiness checks:

```powershell
python -m s_brain.integration --config s_brain/local_integration.json
```

Readiness requires the model service to be responding on the loopback
endpoint with a valid independent-model health report. It does not imply
the model is useful or has been independently benchmarked.

## Evaluation evidence standards

A serious evaluation must disclose its data source, authorship, license,
whether it was independently written, the exact benchmark cases,
contamination checks, model checkpoint identifier and test hardware.

A paired before/after accuracy comparison should keep the same held-out
cases and scoring contract, with confidence intervals disclosed rather
than interpreting a small one-off gain as proof of reasoning improvement.

## Known limitations and follow-up

- Batch groups are based on prompt length; a full continuous batching
  scheduler with fair queuing and dynamic prompt admission is not yet built.
- Cache snapshots incur memory costs; current cache pool does not skip prefill.
- Training data may still contain paraphrase or semantic near-duplicates.
- The benchmark integrity harness uses simple n-gram similarity and may
  produce both false positives and false negatives.
- S is not instruction-tuned or integrated into the default user interface.
- No personal Windows installation or production deployment is verified.

**Cycle 4 recommendation:** continuous batching, batch fairness, effective
prefix reuse, improved long-context memory and a real user-approved
integration switch, followed by deployment and training benchmarks.
