# S-Brain Cycle 9 — Opt-in RAG, controlled tools and answer-quality evidence

## Purpose

Cycle 9 is a **feature-branch research milestone**, not a live deployment.
It extends the independent S-Brain inference code with explicitly opted-in
memory retrieval, a strict host-authorized tool dispatcher, conservative
citation/source checks and paired answer-quality evaluation.

The existing S Continuum/Ollama interface, S-Brain v2 HTTP endpoints and all
previous cycles are preserved without changing defaults. Nothing listens for
conversations automatically, grants internet access, or self-approves tools.

## Implemented modules

| File | Purpose |
| --- | --- |
| `s_brain/rag_inference.py` | Assemble scoped source-attributed prompts; call S's independent inference path; compare enabled/disabled memory with identical generation seeds |
| `s_brain/grounded_answers.py` | Verify included citation IDs, source revision/content hashes, deletion and expiration; keyword heuristic screens claims, with explicit limitation |
| `s_brain/tool_orchestration.py` | Parse a model's tool *proposal*; require host authorization tied to intent, scope and name; execute bounded existing S-Tools allowlist |
| `s_brain/rag_evaluation.py` | Human-labeled paired answer exact-match/token-F1 evaluation, bootstrap uncertainty, categories and immutable benchmark files |
| `s_brain/tests/test_cycle9_pipeline.py` | RAG context boundaries, project isolation, current-source checking, tool denial and small independent-model execution |
| `s_brain/tests/test_cycle9_evaluation.py` | Gold answer benchmarks, answer ablations, score integrity and no-overwrite rules |

## 1. Retrieval-augmented independent inference

Use a local S-Memory database and an already loaded SBrain checkpoint:

```python
from pathlib import Path
from s_brain.checkpoint import load_weights
from s_brain.memory_store import MemoryStore
from s_brain.rag_inference import RetrievalOptions, generate_with_memory

model, _ = load_weights(Path("private_model/best"), device="cpu")
with MemoryStore(Path("private_data/s_memory.sqlite3")) as memory:
    answer = generate_with_memory(
        model,
        "What did my approved engineering note say about optimization?",
        store=memory,
        options=RetrievalOptions(
            enabled=True, scope="S",
            top_k=3, evidence_budget=512,
        ),
        max_new_tokens=24,
        seed=42,
    )
    print(answer.answer)
    print(answer.evidence)
```

**Important limitation:** S is currently a small byte-level model, and many
checkpoints have short context windows. The RAG prompt has framing overhead
and will explicitly raise `ContextOverflow` rather than silently truncate
the question or source identity. Larger-context trained checkpoints are
required for meaningful memory-augmented responses. Adding raw memory text
does not train new weights or guarantee useful answers.

The RAG prompt treats retrieved memory as **untrusted reference material**.
Records containing obvious instruction-override patterns are excluded by a
heuristic guard. Other prompt-injection attempts may evade this guard.

## 2. Source verification

Example using captured context evidence:

```python
from s_brain.rag_inference import assemble_query, RetrievalOptions
from s_brain.grounded_answers import verify_and_gate

options = RetrievalOptions(enabled=True, scope="S")
assembly = assemble_query(
    memory, "What did the research note say?", options,
    model_context=model.cfg.context,
)
packet = generate_with_memory(
    model, "What did the research note say?",
    store=memory, options=options, max_new_tokens=24,
)
checked = verify_and_gate(
    packet, list(assembly.included), store=memory, scope="S",
)
```

In applications, preserve the **exact** evidence snapshot used for the
prompt, instead of independently retrieving a potentially different set.
A citation ID is valid only if it belongs to the injected scope and its
source/content hash and revision still match live memory. Source deletion,
updates or expiry invalidate old citations. A lexical coverage heuristic
checks if cited content shares important terms with generated claims.

**Not proof of factual grounding:** keyword overlap and a valid citation
marker do not establish semantic entailment, truth or correctness.
For consequential claims, require stronger independent verification.

## 3. Host-controlled S-Tools

```python
from s_brain.tools_control import ToolController
from s_brain.tool_orchestration import (
    ToolIntent, ToolAuthorization, ToolOrchestrator,
)

controller = ToolController(scope="S")
controller.enable()  # Explicit host/operator action
intent = ToolIntent(
    identifier="tool-001",
    name="calculator",
    arguments={"operation": "multiply", "left": 9, "right": 7},
    scope="S",
)
approval = ToolAuthorization(
    tool_name="calculator", scope="S",
    intent_id="tool-001", approved_by="operator",
    enabled=True,
)
trace = ToolOrchestrator(controller).run(
    [intent], approvals={"tool-001": approval}
)
print(trace.public())
```

The model cannot grant its own approval. The host must authorize the exact
intent. Permission failures are logged with codes, not sensitive arguments.
Memory modifications additionally require one-use tokens from the tool
controller. The available tools still cannot run arbitrary subprocesses,
shell commands, external network calls or Python code.

## 4. Paired answer evaluation

The harness needs independently authored reference answers and a fixed model
checkpoint, memory store and generation seed. One example:

```python
from s_brain.rag_evaluation import (
    AnswerCase, AnswerEvalPolicy, run_answer_evaluation,
)

case = AnswerCase(
    identifier="human-labeled-001",
    question="What is a transformer?",
    expected_answer="A neural network architecture using attention.",
    scope="S", category="science", split="test",
    permission="self-authored",
)
report = run_answer_evaluation(
    model, memory, [case],
    policy=AnswerEvalPolicy(
        minimum_cases=1, bootstrap_iterations=200,
    ),
)
print(report["summary"])
```

This evaluates both modes using the same independent S generator. Scores
include normalized exact match, token F1, per-category results and
record-level paired bootstrap confidence intervals. The *gold answer* is
used only for scoring, never to construct memory or influence generation.

CI fixture tests use deterministic mock generators to prove the evaluation
pipeline can detect a controlled retrieval benefit. They are not proof that
S's actual trained model has improved factuality or reasoning.

## 5. Verification and scientific limitations

Required:
- Existing S application CI remains green.
- Existing S-Brain tests remain green plus Cycle 9 tests.
- Opt-in retrieval is project scoped and refuses context overflow.
- Source updates/deletions invalidate citation integrity.
- Model-generated tool proposals cannot execute without separate host approval.
- Paired evaluations are deterministic and report negative results honestly.

Not established:
- A production trained model with broad reasoning performance.
- A real-world, representative RAG improvement over no-memory.
- Validity of all model-generated citations under semantic entailment.
- Live integration in the main S Continuum UI.
- A fix for Windows App Control blocking `torch.dll`.

Cycle 10 should benchmark genuine small-model RAG and tool-assisted answers
against a fixed owned test suite, then integrate opt-in controls into S's
user-facing interface after security review. Preserve source and benchmark
fingerprints throughout.
