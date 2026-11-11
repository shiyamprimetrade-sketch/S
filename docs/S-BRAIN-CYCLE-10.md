# S-Brain Cycle 10 — opt-in assistant HTTP integration and checkpoint evaluation

## Purpose

Cycle 10 extends the independent S-Brain experimental server with an opt-in
assistant interface. It preserves all Cycle 1–9 implementation, the original
Continuum application and its Ollama settings. The new route calls only S's
independent model weights; it does not silently fall back to Ollama.

The goal is a usable, audited research pathway for optional memory,
read-only tools and measured answer improvement, not a claim of Gemini-class
intelligence.

## Implemented code

- s_brain/assist_runtime.py — bounded requests, RAG context snapshots, per-project scope, independent generation, read-only tool results, optional source checks and redacted audits.
- s_brain/server_v2.py — disabled-by-default POST /v1/assist and GET /v1/assist/capabilities, along with separate startup flags for assistant, memory and tools.
- s_brain/real_rag_benchmark.py — loads a real checksum-verified S checkpoint for BOTH RAG and no-RAG generation arms, with fixed seed, question set fingerprints, answer exact match, token F1 and paired record bootstrap intervals.
- s_brain/tool_answer_benchmark.py — isolates read-only tool execution, compares tool-output fidelity separately from the answer and NEVER passes gold reference answers to the generation callback.
- s_brain/tests/test_cycle10_assist.py — opt-in assistant route, source hash and revision checks, localhost restrictions, unauthorized tool and memory access.
- s_brain/tests/test_cycle10_benchmarks.py — actual randomly initialized S neural checkpoint smoke test, checksum verification, benchmark integrity and tool-gold leakage regressions.
- s_brain/tests/test_cycle10_security.py — adversarial scope, schema, shell and write-denial tests.

## Starting the independent server

The existing S-Brain v2 service binds localhost by default. Previous endpoints
such as /generate and /generate/stream remain intact. No flags means the new
assistant route is unavailable.

Without the experimental assistant:

    python -m s_brain.server_v2 --checkpoint private_model/best --port 8766

Enable experimental assistant generation only:

    python -m s_brain.server_v2 --checkpoint private_model/best --port 8766 --enable-assistant

Enable project-scoped local retrieval:

    python -m s_brain.server_v2 --checkpoint private_model/best --port 8766 --enable-assistant --enable-memory --memory-db private_data/s_memory.sqlite3

Enable read-only calculator, text statistics and JSON extraction too:

    python -m s_brain.server_v2 --checkpoint private_model/best --port 8766 --enable-assistant --enable-memory --memory-db private_data/s_memory.sqlite3 --enable-tools

All arguments above refer to operator-owned local files. Do not commit
private memory databases, training examples or model outputs to public GitHub.

## PowerShell client

Query the feature flags:

    Invoke-RestMethod "http://127.0.0.1:8766/v1/assist/capabilities"

Submit an independent S prompt with memory off:

    $body = @{
        question = "What is gradient descent?"
        scope = "S"
        memory = $false
        cite_sources = $false
        max_new_tokens = 16
        seed = 42
    } | ConvertTo-Json -Depth 8
    Invoke-RestMethod -Uri "http://127.0.0.1:8766/v1/assist" -Method Post -ContentType "application/json" -Body $body

Set memory to true only when local approved excerpts exist and the server was
started with the memory switch. Cite_sources requests a conservative
source/citation validation. A small-context model may refuse RAG generation
because the entire question and reference evidence must fit the checkpoint's
byte-token context. Silent truncation would invalidate source provenance.

The assistant endpoint does not expose a live memory-write operation.
Tool results are untrusted context and are inserted BEFORE the answer cue.

## Read-only tool request example

Start the assistant with --enable-tools, then send a JSON POST body:

    {
      "question": "Multiply 6 by 7.",
      "scope": "S",
      "memory": false,
      "tool_intents": [
        {
          "id": "calc-1",
          "name": "calculator",
          "arguments": {"operation":"multiply","left":6,"right":7}
        }
      ]
    }

The server allows only the three named read-only tools. It never runs shell,
Python, subprocesses, arbitrary URLs, external accounts, file reads or memory
writes. Tool calls still have argument and result quotas. This experimental
HTTP route has no per-user authentication; local processes sharing OS access
may call it. Do not expose the port through tunnels or reverse proxies.

## Evaluating genuine S model memory effects

Run paired memory-on/memory-off generations from a LOCAL VERIFIED checkpoint
and gold test data collected independently of model development:

    python -m s_brain.real_rag_benchmark --checkpoint private_model/best --memory-db private_data/s_memory.sqlite3 --gold private_gold_test.jsonl --output private_real_rag.json --max-new-tokens 16 --seed 101

A benchmark answer label must contain an identifier, question,
expected_answer, category, scope, permission, and split=test. Require a
representative set; the default benchmark minimum is two cases and a
model context of at least 512 byte tokens.

The benchmark runs the actual S model, not a mocked answer callback. In CI,
it uses a tiny random-weight S checkpoint to demonstrate the computation
path and provenance validation. Such a test does NOT prove practical
memory usefulness or knowledge gains.

Exact match, token F1, per-case hashes and paired confidence intervals are
reported. A positive improvement requires a positive confidence lower bound
under a threshold chosen BEFORE evaluation. Do not repeatedly tune on the
independent test set.

## Tool-assisted benchmark and scientific boundaries

The tool-answer benchmark compares actual deterministic tool results with
pre-authored expected outputs, while separately scoring a model/fixture
answer. ToolQuestion passed to the answer callback excludes expected_answer
and expected_output. A correct calculator result is not evidence that S
can choose the appropriate tool or produce a correct explanation.

Citation validation checks whether the exact memory snapshot used in
generation still matches source ID, revision and content SHA-256. A valid
source citation and keyword overlap do not prove semantic entailment,
truthfulness, or that the answer actually follows from evidence.

The experimental server is opt-in, not an automatic upgrade of the main
Continuum web user interface. No external teacher dataset was generated,
no large-scale training was performed, and the Windows Code Integrity
torch.dll block remains unresolved.

## Verification gate

Cycle 10 is complete only after:
- At least 2,000 meaningful implementation and test Python lines committed.
- All S-Brain GitHub Actions checks pass on the FINAL branch head.
- Original Continuum tests remain green.
- No automatic memory capture, write tools or arbitrary command execution.
- PR #14 remains a draft and documents the exact scientific limitations.
- The GitHub commit and Actions run IDs are reported without fabricated claims.

Stop at this checkpoint. Pursue the next upgrade only after a new explicit
Continue message and preferably after collecting representative real
benchmark data.
