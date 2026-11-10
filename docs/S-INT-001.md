# S-INT-001 — Independent neural model experiment

This module uses PyTorch to initialize and train a small causal character-level attention network from scratch. No pretrained weights, Ollama calls, external inference APIs or teacher data are required. PyTorch is a software dependency, not a pretrained intelligence dependency.

## Windows 11 quick start

From the repository root in PowerShell:

```powershell
py -3 -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install torch
.\\.venv\\Scripts\\python.exe -m unittest tests.test_s_native -v
.\\.venv\\Scripts\\python.exe -m s_intelligence.train --steps 80
.\\.venv\\Scripts\\python.exe -m s_intelligence.infer --prompt 'S '
```

## Acceptance criteria

1. Model creates finite logits and gradients.
2. Causal masking prevents later characters from changing earlier predictions.
3. Training creates a checkpoint and reports both validation losses.
4. Checkpoint reloads and generates characters with Ollama stopped.
5. Run at least three independent seeds and evaluate on separately authored text before claiming generalization.

## Limits

The built-in repeated synthetic corpus is a pipeline smoke test, not a real-world training set. The validation partition repeats training patterns and cannot establish generalization. No training results or benchmark improvements are claimed until commands are executed and evidence is recorded. Never commit private dialogues or credentials.
