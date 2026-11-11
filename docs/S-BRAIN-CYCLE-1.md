# S-Brain engineering cycle 1 — independent training infrastructure

This milestone extends the experimental Transformer in draft PR #14. It does not
change the existing S application, automatically call Ollama, upload private data,
or claim production-readiness. The independent model must be trained before it can
generate meaningful responses.

## New architecture

| Module | Capability | Limitations |
| --- | --- | --- |
| `dataset.py` | Corpus provenance, basic redaction, deterministic splits, manifests, hashes | Redaction is not exhaustive; near-duplicates may leak |
| `checkpoint.py` | Checksummed safetensors weights, optimizer state, retention | Optimizer state is trusted-local only |
| `sampling.py` | Top-k/top-p/min-p, repetition controls, seeded sampling | No KV caching; UTF-8 byte-level generation |
| `evaluate.py` | NLL, perplexity, exact/normalized task evaluation, regression gates | Statistical scores are not intelligence tests |
| `train_advanced.py` | Gradient accumulation, AdamW groups, LR schedule, held-out validation | CPU training may be slow; no multi-GPU |
| `devtools.py` | CLI inspect, compare, audit, sample, evaluate and QA | Requires a trained checkpoint |
| `tests/test_dataset.py` | Privacy/provenance tests | Only representative patterns |
| `tests/test_advanced.py` | Training, checkpoint, sampling and benchmark smoke tests | Tiny configurations and no performance guarantees |

## Create an authorized dataset

Place original approved `.txt` or `.md` files in a local folder named
`approved_corpus`. Do not upload this folder or personal data to the public
GitHub repository.

```powershell
python -m s_brain.dataset build --input approved_corpus --output private_corpus --source self-authored --license user-owned
python -m s_brain.dataset audit --manifest private_corpus/manifest.jsonl
```

The builder assigns a stable split by relative filename and removes exact
duplicates. **A small set of files can leave validation or test empty.** Use a
sufficiently diverse corpus and verify `private_corpus/stats.json` first.

## Train

Use an environment with a working PyTorch build. The original Windows
Code Integrity block for `torch.dll` must be resolved using a trusted,
policy-compliant environment; do not turn off Windows security to train.

```powershell
python -m pip install -r s_brain/requirements.txt
python -m s_brain.train_advanced --corpus-dir private_corpus --steps 100 --batch 2 --sequence 64 --dim 128 --layers 2 --heads 4 --kv-heads 2
python -m s_brain.devtools list --root s_brain/checkpoints/advanced
```

Optimizers are saved to support resuming **only with explicit trust**. Load the
optimizer state only when it was produced locally and not modified by an
untrusted party:

```powershell
python -m s_brain.train_advanced --corpus-dir private_corpus --steps 200 --resume-from s_brain/checkpoints/advanced/step-00000100 --trusted-optimizer
```

## Inspect, sample, evaluate

```powershell
python -m s_brain.devtools inspect --checkpoint s_brain/checkpoints/advanced/step-00000100
python -m s_brain.devtools sample --checkpoint s_brain/checkpoints/advanced/step-00000100 --prompt "S is" --tokens 80
python -m s_brain.devtools evaluate --checkpoint s_brain/checkpoints/advanced/step-00000100 --file private_corpus/test.bin
```

These commands evaluate the independent model. The standalone API in `api.py`
continues to load the **legacy** `weights.pt` format from the earlier basic
trainer; the advanced trainer uses `model.safetensors` in versioned directories.
The API must be upgraded to load both formats and tested before integration.

## Verification and exit criteria

1. GitHub Actions confirms all prior and new regression tests pass.
2. The dataset reader rejects bad hashes and cross-split exact duplication.
3. The training smoke test saves checkpoints that reload successfully.
4. Seeded sampling produces reproducible results.
5. No model claims are made without separate held-out evidence.
6. The current S chat server remains unchanged on port 8765.

## Next milestone proposal

- Add a tested cached-attention inference path and streaming API.
- Add an opt-in S model adapter rather than replacing the existing Ollama path.
- Improve the S-Brain dataset ingestion rules and near-duplicate detection.
- Introduce actual benchmark fixtures with provenance, licensing and blind scoring.
- Resolve the advanced-checkpoint/standalone-API format mismatch.

When these criteria are met, ask for the next engineering cycle by replying
`Continue`. Development happens during each active session, not indefinitely
in the background.
