# S-INT-002: broader vocabulary and controlled synthetic diversity

## Scope and provenance

Continues S-INT-001 on the same research branch. Native model code is unchanged: one causal attention layer, width 64, context 32. The fixed 100-character printable ASCII vocabulary increases trainable parameters from 21,138 to 31,716. Unicode remains unsupported; this is not an open-vocabulary tokenizer.

The new dataset has 102 training records and 48 validation records. Every record is explicitly synthetic. Validation uses different sentence templates with shared subjects and objects; this is template-disjoint, not independently sourced real-world data. No private chats, downloaded datasets, pretrained weights, new dependencies, or remote teacher calls were used. Existing checkpoints and browser application remain unchanged.

Training accepts an optional JSON dataset via --data. The default S-INT-001 experiment remains supported. Validation never enters gradient updates. Split validation rejects duplicate or overlapping normalized records, missing provenance, unsupported characters and insufficient text. Checkpoints retain corpus counts, dataset hashes and normalized training-record hashes. Evaluation rejects exact normalized training-record leakage and uses checkpoint-specific add-one-smoothed unigram counts. Old checkpoints retain their original unsmoothed baseline.

## Fixed experiment and observed results

Budget fixed before execution: one 80-step smoke and three independently initialized 600-step runs, seeds 20261110 through 20261112. AdamW, learning rate 0.003, batch 16, two CPU threads. All socket.connect calls were denied during the experiment. Every model was evaluated; no held-out-driven retraining was performed.

| Steps | Seed | Initial validation | Final validation | Benchmark CE, all 12 | Benchmark CE, original 7 |
|---:|---:|---:|---:|---:|---:|
| 80 | 20261110 | 4.7205 | 2.4483 | 2.4176 | 2.3239 |
| 600 | 20261110 | 4.7205 | 3.0348 | 4.1118 | 4.2153 |
| 600 | 20261111 | 4.8621 | 2.8160 | 3.7694 | 3.8452 |
| 600 | 20261112 | 4.7900 | 3.0508 | 4.1245 | 4.1849 |

Lower cross-entropy is better. Coverage is now 12/12, versus 7/12 previously. The new unigram baseline over all 12 records is 2.7799. At 80 steps, trained loss is 2.4176, below this baseline. At 600 steps, all three seeds regress to 3.7694–4.1245. This is evidence of overfitting, not a successful general-purpose model. On the original seven records, the prior 80-step model scored 4.1494; the new 80-step model scores 2.3239. Vocabulary and corpus both changed, so this is not a causal ablation.

The original evaluation set was already known during dataset design. Its results are a development benchmark, not a blind estimate of generalization. Exact-record exclusion does not prevent semantic or template overlap. Validation uses fixed sampled batches rather than the entire split. No reasoning benchmark was run.

Training took about 0.27 seconds for 80 steps and 2.00–2.02 seconds for each 600-step run on the existing CPU environment. Full measured losses, resource metrics, hashes, outputs and per-record scores are in reports/s_int_002_windows_results.json. Native checkpoints remain local under artifacts/s-int-002 and are excluded from Git.

## Verification

40 compatible repository tests passed, including two new dataset tests. New checks cover malformed splits, normalized train/validation overlap, checkpoint vocabulary, ASCII prompts, all-12-record evaluation, finite smoothed baseline, and training-record leakage rejection. The existing tests still verify old checkpoint behavior, seeded reproducibility, gradients, causality and offline inference.

## Reproduction

From the repository root, with the existing PyTorch environment:

```powershell
& 'C:\S\.venv\Scripts\python.exe' -m unittest discover -s tests -v
```

In a fresh checkout with no artifacts/s-int-002 directory:

```powershell
& 'C:\S\.venv\Scripts\python.exe' scripts/run_s_int002.py
```

Generate from the preserved 80-step experiment:

```powershell
& 'C:\S\.venv\Scripts\python.exe' -m s_intelligence.infer --checkpoint artifacts/s-int-002/steps-80-seed-20261110.pt --prompt 'S ' --count 80 --seed 20261110
```

For a separate training run, use a new checkpoint path; existing files are never intentionally overwritten:

```powershell
& 'C:\S\.venv\Scripts\python.exe' -m s_intelligence.train --steps 80 --seed 20261110 --threads 2 --data data/training/s-int-002.json --output artifacts/new-ascii-run.pt
```

## Next milestone

Use a substantially more diverse authorized corpus, source-disjoint data splits and a sealed evaluation set. Select stopping rules on validation only, then evaluate once on the sealed test. Compare corpus and vocabulary changes separately. Do not increase training duration on the current narrow templates or claim reasoning from character likelihood.
