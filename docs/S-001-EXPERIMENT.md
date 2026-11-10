# S-001 — Meaning-compiled symbolic instructions

## Purpose
A bounded experiment: derive executable rewrite rules from demonstrations, challenge them with unseen inputs, and compare against the pre-existing one-operation append-s baseline. It is **not** a new physical computer, general semantic understanding, or autonomous invention.

## Source reuse and project boundary
Reviewed `Slickk/slickk-native-runtime` (source branch) and selectively adapted its architecture principles rather than copying the education application. The source has a replaceable `SlickkEngine` interface, a manifest registry, and a persistent Node/Transformers.js ONNX worker. S already has its own optional Ollama integration, memory, and bounded symbolic synthesis. The source's worker references `tools/slickk-wasm/persistent-worker.mjs`, which is not present in the inspected source tree, so it cannot be claimed operational or imported as a functioning runtime. No source repo was modified.

## Hypothesis
For some transformations, S can infer a compact instruction table from examples rather than hard-coding every example. Test against held-out cases and an append-s baseline.

## Reproduction
```powershell
py -3 -m unittest discover -s tests -v
py -3 -m examples.run_s001
```

The report includes exact outputs, rule support, baseline score, held-out failures, and SHA-256 digest. An irregular plural is intentionally included to prevent perfect-score theater. No speed or energy superiority is claimed. Inference uses conventional Python; it does not bypass physical computing constraints.

## Completion gates
1. All existing tests pass.
2. Held-out score improves over baseline without training/holdout leakage.
3. Report preserves failures, evidence digest, and learned rules.
4. Novelty relative to existing inductive programming and rewrite systems must be researched separately; this experiment alone is not patent evidence.

## Next experiments
Compare against finite-state transducers, program synthesis systems, and learned transducers; add ambiguity handling, out-of-distribution detection, and CPU/memory benchmarks. Only then decide whether the representation is genuinely advantageous.
