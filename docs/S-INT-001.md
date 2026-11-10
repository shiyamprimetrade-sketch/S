# S-INT-001 — Independently initialized native neural experiment

This module trains a small causal character-attention network from random weights using CPU PyTorch. Training and inference do not use Ollama, pretrained weights, or remote inference. The installed S v0.6 application is separate: this research branch predates it, and must not replace its Continuum files.

## Existing Windows environment

The verified PC already has Python 3.12.10 and CPU PyTorch 2.15.0 in `C:\S\.venv`. Reuse that working environment. A fresh installation needs Python 3.11+ and PyTorch from its official package index. PyTorch is software, not pretrained intelligence.

Open PowerShell in this repository root. Each block is one command.

```powershell
& 'C:\S\.venv\Scripts\python.exe' -m unittest discover -s tests -v
```

Train a new 80-step experiment. Existing checkpoint/metrics paths are never overwritten:

```powershell
& 'C:\S\.venv\Scripts\python.exe' -m s_intelligence.train --steps 80 --seed 20261110 --threads 2 --output artifacts/new-smoke.pt
```

```powershell
& 'C:\S\.venv\Scripts\python.exe' -m s_intelligence.infer --checkpoint artifacts/new-smoke.pt --prompt 'S ' --count 80 --seed 20261110
```

```powershell
& 'C:\S\.venv\Scripts\python.exe' -m s_intelligence.evaluate --checkpoint artifacts/new-smoke.pt --output artifacts/new-evaluation.json
```

`--seed` selects independently initialized model weights and deterministic CPU sampling. The trainer saves a `.metrics.json` beside each checkpoint. Initial/final training and validation losses use the same frozen evaluation batches. The training corpus is intentionally repeated synthetic text; low validation loss is not an independent test of generalization.

## Controlled experiment

The bounded runner executes one 80-step smoke test, three 240-step runs with seeds 20261110/20261111/20261112, and an exact 80-step reproducibility check. It denies all `socket.connect` calls throughout training, evaluation and generation. Run it once in a fresh checkout; it refuses to replace an existing experiment path.

```powershell
& 'C:\S\.venv\Scripts\python.exe' scripts/run_s_int001.py
```

Checkpoints and raw machine-local results are under `artifacts/s-int-001/`, excluded from Git. The sanitized report is `reports/s_int_001_windows_results.json`. No user chats, credentials, or private datasets are training inputs. Checkpoint inference uses `torch.load(..., weights_only=True)`.

The held-out file `data/evaluation/s-int-001.json` was authored separately and frozen before these experiments. It is not read by the trainer. Evaluation compares trained next-character loss with the matched random initialization, a training-character unigram baseline, and a uniform baseline. It discloses and skips whole records containing unsupported characters; no unknown text is silently rewritten. This narrow tokenizer is not suitable for unrestricted conversation.

## Acceptance and actual results

All 38 compatible repository tests passed, including six native tests. Coverage includes tokenizer round-trip, causal masking, finite logits and nonzero finite gradients, checkpoint reload, deterministic training/generation, no-network inference, output preservation, and independent evaluation accounting.

Architecture: 18-character vocabulary, width 64, learned positions, context 32, one four-head causal attention layer with residual/LayerNorm, linear output. Trainable parameters: 21,138. No feed-forward block or deep stack is present.

The 80-step run reduced repeated-pattern validation loss from 2.970508 to 0.432258. Three 240-step runs reached 0.092303, 0.073604 and 0.075491. However, independent-text loss worsened to 5.874169, 6.310066 and 6.211445, versus random initialization near 3.03 and unigram 2.472995. Only 7 of 12 independent texts fit the vocabulary; the other 5 are explicitly reported. These observations demonstrate executed training and overfitting, not useful generalization or reasoning.

Measured training took 0.26 seconds for the smoke run and 0.78–0.85 seconds for each 240-step run, using two CPU threads. Peak process resident memory was about 268 MiB, including PyTorch import. Warm-process inference including checkpoint load took 31–33 ms for 80 generated characters. These are local measurements, not hardware-independent promises.

## Next proof gate

Use a diverse, authorized corpus, an adequate open-vocabulary tokenizer, source-disjoint validation/test sets, and explicit reasoning tasks with negative controls. Preserve this experiment as a baseline. Do not claim human-level intelligence, GPT-level capability, autonomous invention, or neural training from imported chats.
