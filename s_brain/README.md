# S-Brain — independent experimental Transformer

This is an **untrained-by-default**, standalone autoregressive byte-level language model, not a Gemini-equivalent assistant. S-Brain has its own learned weights after training and needs **no Ollama** at inference time. Do not equate low training loss with independent reasoning.

## Isolation
- Existing S Continuum API remains on `127.0.0.1:8765`.
- The optional native console integrates at `/brain` on port 8765; standalone development serving remains separate on port 8766. See [integration evidence and limits](../docs/brain-console-integration.md).
- No pretrained weights, datasets, private conversations, API keys, or secrets are committed.
- Existing TeacherLab is a different intent classifier and remains unchanged.

## Install (from repository root)

```powershell
py -3 -m venv .venv-sbrain
.\.venv-sbrain\Scripts\python.exe -m pip install -r s_brain\requirements.txt
.\.venv-sbrain\Scripts\python.exe -m unittest discover -s s_brain/tests -v
```

If Windows Smart App Control blocks `torch.dll` (Code Integrity 3077), **do not disable security**; use an authorized trusted Linux/WSL development environment or a Microsoft-approved allow policy. The repository code cannot override endpoint protection.

## Train (use your own approved text file)

```powershell
.\.venv-sbrain\Scripts\python.exe -m s_brain.train --corpus .\my_authorized_text.txt --steps 1000
.\.venv-sbrain\Scripts\python.exe -m s_brain.api
```

## Test standalone inference

```powershell
Invoke-RestMethod http://127.0.0.1:8766/health
$body = @{prompt="Hello S";max_new_tokens=32} | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8766/generate -Method Post -ContentType application/json -Body $body
```

Training is byte-level next-token prediction using GQA, RoPE, RMSNorm, and SwiGLU. The checkpoint stores the best validation loss observed at intervals. Cycles 1–10 added KV cache, supervised training and evaluation infrastructure; multimodality and distributed training remain unsupported. The 10% suffix validation split may still contain near-duplicates: real benchmarks require independent held-out data.

## Integration criteria
1. Unit tests pass in the intended training environment.
2. A checkpoint trains, reloads, and generates without Ollama.
3. Independent test prompts are evaluated for accuracy and robustness.
4. Resolve any conflict with the original server's port and request format (standalone module deliberately uses 8766).
5. Add an explicit opt-in model selection adapter to S; retain the original Ollama backend and avoid automatic unsafe switching.

**Status:** Experimental integration tested on Windows; the tiny trained checkpoint failed its synthetic reasoning diagnostic. See the integration report for measured results and remaining gates.

Final browser and preservation evidence is documented in [the integration report](../docs/brain-console-integration.md).
