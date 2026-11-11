# S-Brain Cycle 2 — KV-cache, streaming inference and opt-in compatibility

## Scope and isolation

Cycle 2 extends the independent experimental byte-level Transformer introduced
in Draft PR #14. It leaves S's working Continuum server, persistent project
storage, local Ollama bridge and TeacherLab classifier untouched.

- Existing S server: `127.0.0.1:8765`
- Experimental S-Brain v2: `127.0.0.1:8766`
- Ollama is NOT needed by the new Transformer inference path.
- This is an experimental next-byte model, NOT an equivalent to Gemini.
- No automatic model replacement, execution of generated code or internet exposure.
- No trained weights or private training corpora are committed.

## Cycle 2 modules

| File | Purpose |
| --- | --- |
| `s_brain/kv_cache.py` | GQA key/value cache, relative RoPE offsets, prefill, incremental append, context rollover |
| `s_brain/streaming.py` | Cached generation with UTF-8 incremental decoding, stop strings, cancellation, output events |
| `s_brain/server_v2.py` | Localhost HTTP server with JSON/SSE/NDJSON, concurrency caps and checkpoint compatibility |
| `s_brain/client.py` | Loopback-only streaming client and protocol decoders |
| `s_brain/benchmark.py` | Cached/uncached timing and numerical parity comparison |
| `s_brain/compat.py` | Explicit backend selection contracts, no silent fallback |
| `s_brain/tests/test_cycle2.py` | Cache, streaming, HTTP request, Unicode and benchmark tests |
| `s_brain/tests/test_compat.py` | Model-registry and opt-in integration tests |

## KV-cache design

Cached prefill processes a complete causal prompt and retains grouped KV heads
without repeating KV tensors across attention heads in persistent storage.
For single-token decoding it computes only the current token's new projections,
appends them to cached K/V tensors and uses noncausal SDPA over the full past.
The full original block MLP and normalization paths are reused exactly.

RoPE in the original model is relative to the current active context window.
When the window becomes full, the implementation **re-prefills** the shifted
window to preserve output equivalence. A faster rolling-position scheme would
require explicit model architecture and training changes; pretending a simple
KV crop is equivalent would be incorrect.

## Start a versioned checkpoint

From the repository root, in an approved environment that can load PyTorch:

```powershell
python -m pip install -r s_brain/requirements.txt
python -m unittest discover -s s_brain/tests -v
python -m s_brain.server_v2 --checkpoint s_brain/checkpoints/advanced/step-00000100
```

Use an actual checkpoint directory containing `manifest.json` and
`model.safetensors`. The v2 loader also accepts the earlier trusted-local
`config.json` and `weights.pt` format.

The Windows Smart App Control restriction involving `torch.dll` must still be
resolved using a policy-compliant installation or approved WSL/Linux host.
No instruction here disables the PC's security protections.

## HTTP endpoints

`GET /health` — ready, model architecture and admission statistics.

`GET /v1/models` — experimental S-Brain model ID.

`POST /generate` — non-streaming JSON response with text and usage.

`POST /generate/stream` — server-sent events (SSE), one JSON object per event.

`POST /generate/ndjson` — newline-delimited event objects.

Example PowerShell:

```powershell
$body = @{prompt='Hello S';max_new_tokens=20;seed=42;use_cache=$true} | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8766/generate -Method Post -ContentType application/json -Body $body
```

For streaming, invoke the Python client:

```powershell
python -m s_brain.client health
python -m s_brain.client stream 'Hello S' --tokens 20 --seed 42
python -m s_brain.client generate 'S is' --tokens 20
```

Stream events: `start`, zero or more `delta` and exactly one
`complete` when the client stays connected. A disconnect or process failure
can terminate early. Network buffering and clients may alter rendering timing.

Example event:

```json
{"type":"delta","sequence":1,"text":"H","model":"s-brain:experimental"}
```

The `complete` event contains finish reason, generated token count, cache
rebuild count and elapsed wall-clock time. Values are diagnostic metadata.

## Benchmarking

```powershell
python -m s_brain.benchmark --checkpoint s_brain/checkpoints/advanced/step-00000100 --prompt-bytes 32 --tokens 24 --warmup 2 --rounds 5 --output local-benchmark.json
```

This compares the same model on the same host in cached and uncached modes.
It also checks numerical parity over a token sequence. The cache must not be
claimed faster until actual benchmark evidence exists. Short contexts can
make caching overhead dominate.

## Security boundaries

- Loopback-only, no public listener or remote API key support.
- Origin and Host headers validated for browser requests.
- JSON body and prompt limits enforced before generation.
- Concurrency capped; excess requests receive 429.
- Generated output is returned as text, never executed.
- No password, secret or private training data is committed.
- Not yet hardened for untrusted local processes. Localhost alone is not
  authentication against an adversarial process on the same machine.

## Verification criteria

- GitHub Actions complete all tests, including existing S regression tests.
- Cached prefill/incremental output agrees numerically with full decoding.
- Randomized cached sampling matches uncached results under the same seed.
- Incremental UTF-8 decoder handles split multibyte characters and bad bytes.
- Stop strings are not leaked across event boundaries.
- Checkpoint formats are loaded only from trusted local paths.
- API request size, origin and concurrency controls are tested.
- No changes to existing S server are required.

## Known limitations

This implementation recomputes a full prefill at the context boundary;
it does not use continuous RoPE position IDs and paged attention. SSE is a
blocking thread-per-request implementation, not an async inference scheduler.
It is not instruction-trained, multimodal or optimized for large batches.
Latency metrics are architecture and hardware dependent, and the cache
does not prove improved reasoning or answer quality.

## Next milestone

Cycle 3: advanced batching, cache-efficient context management, checkpoint
format migration, independent instruction dataset/evaluation pipeline, optional
S UI integration with explicit approval, and performance regression gates.

All future work should preserve the current branch, existing tests and PR
history. Stop after each cycle for the user's next `Continue`.
