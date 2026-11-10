# S-Brain — Cycle 8: S-Memory, controlled S-Tools, independent learning evaluation

## Mission

Develop three **separate** subsystems without confusing retrieval or tool use
with updated neural-network weights:

1. **S-Memory** stores operator-approved text with source and scope evidence.
2. **S-Tools** executes a narrow, bounded set of deterministic local functions.
3. **S-Learning evaluation** tests held-out prediction quality under domain
   changes and measures memory retrieval relevance.

All Cycle 1–7 code remains in the same branch and Draft PR #14. No default
S Continuum chat route or Ollama behavior is replaced.

## Files

| File | Implementation |
|---|---|
| `s_brain/memory_store.py` | SQLite project scopes, consent, TTL, source digests, deletion, audit, BM25-style lexical retrieval |
| `s_brain/tools_control.py` | Disabled-by-default allowlisted tools, per-session limits, scope isolation, write approvals and redacted audit |
| `s_brain/shift_evaluation.py` | Paired domain-shift reports, lexical leakage checks, category comparison and record-level confidence intervals |
| `s_brain/memory_evaluation.py` | Labeled retrieval precision/recall, citation checks and explicit cross-scope leakage probes |
| `s_brain/tests/test_cycle8_memory_tools.py` | Memory persistence, expiry, deletion, permission and tool security regression tests |
| `s_brain/tests/test_cycle8_evaluation.py` | Held-out shift, uncertainty, retrieval relevance, split isolation and no-overwrite reporting |

## S-Memory usage

S-Memory is explicitly local. It does not automatically scrape user messages,
make claims of remembering everything, or train S's neural weights.

```python
from pathlib import Path
from s_brain.memory_store import MemoryStore, citation_context

with MemoryStore(Path("private_data/s_memory.sqlite3")) as memory:
    saved = memory.remember(
        scope="S",
        content="Transformer weights change through optimizer updates.",
        source="user-approved engineering note",
        consent="explicit",
        ttl_seconds=86400,
    )
    hits = memory.retrieve("S", "transformer weight updates", limit=3)
    print(citation_context(hits))
    memory.forget("S", saved.id)
```

**Project scope is mandatory.** A record saved under S cannot be retrieved,
modified, or removed using Prime International's scope. The SQLite database
includes content until explicit deletion or expiry. The scope is a logical
application boundary; it is not a replacement for file/disk encryption or
OS-user separation. Read/write access to the database file must also be
restricted by the host operating system.

The retrieval algorithm uses lexical term scoring. Semantically equivalent
paraphrases may be missed, even if a more advanced retriever could find them.
Retrieved text is **untrusted evidence**, not commands for the model.

## S-Tools usage

S-Tools intentionally does not offer arbitrary Python execution, shell,
filesystem browsing, command-line tools, external HTTP, GitHub writes or
third-party account access. It is a small, auditable foundation.

```python
from s_brain.tools_control import ToolController, ToolCall

tools = ToolController(scope="S")
tools.enable()  # Explicit controlling-application action

result = tools.invoke(ToolCall(
    name="calculator",
    arguments={"operation":"multiply","left":"1.5","right":"3"},
    scope="S",
))
assert result.success
print(result.output)  # {'value': '4.5', ...}
```

Default allowlist:

- Calculator with bounded Decimal arithmetic.
- Text length and word statistics.
- JSON value extraction from *supplied* input only.
- Project-scoped memory retrieval, if a memory store is configured.

**Write operations are disabled by default**. An application must instantiate
an explicit policy permitting memory writes, allowlist the exact write tool,
and obtain a separate one-use approval token before each mutation. The
approval token is a controller interface, not an authentication secret;
production integration must connect it to a real user approval control.

Tool outputs are checked against a configured byte quota. Audit entries
record argument hashes and status rather than the raw argument payload.
Do not treat hashes as anonymization of low-entropy secret text.

## S-Learning: held-out distribution-shift evaluation

Run against an approved model checkpoint and separately authored
`validation` or `test` supervised byte datasets:

```powershell
python -m s_brain.shift_evaluation --checkpoint private_checkpoint --base private_holdout.jsonl --shift private_new_domain.jsonl --output private_shift_report.json
```

The evaluator checks exact question and source-hash overlap between the
baseline and shifted datasets. Reports include group-specific cross-entropy,
accuracy, categories, record-bootstrap intervals, and declared tolerances
for accuracy and loss regression.

These are **teacher-forced next-byte** measurements. They do not establish
free-form reasoning, factuality, instruction following, agent planning,
or Gemini-level intelligence. Unknown dataset overlap and semantic
paraphrases remain an evaluation risk.

## S-Memory relevance tests

```python
from s_brain.memory_evaluation import RetrievalCase, evaluate_retrieval
from s_brain.memory_store import MemoryStore
from pathlib import Path

with MemoryStore(Path("private_data/s_memory.sqlite3")) as memory:
    # Gold IDs must be labeled by a human, not inferred from the retriever.
    case = RetrievalCase(
        identifier="relevance-1", scope="S",
        query="transformer optimizer",
        expected_ids=("human-labeled-id",),
    )
    print(evaluate_retrieval(memory, [case]))
```

Measures precision-at-k, recall-at-k and whether retrieved memories
carry source and content digests. This is retrieval evaluation, not
an answer correctness score. The cross-scope probe checks whether a query
can return records from an unrelated project.

## Threat model and non-goals

- No background operation or automatic listening for user activity.
- No automatic online tool activation, internet search or Ollama fallback.
- No silent memory extraction from conversations.
- No ability to bypass Windows Code Integrity policy.
- No inference that storing a memory equals training learned weights.
- Secret/email patterns are heuristic: they can miss credentials.
- A caller with unrestricted write access to the SQLite file can bypass
  Python-level scope checks; OS filesystem access controls are necessary.
- The model-generated request must not be treated as user approval.
- Tests in GitHub CI use synthetic, temporary fixtures rather than user data.

## Milestone completion criteria

1. Python compiles in GitHub Actions and the S-Brain regression suite passes.
2. Existing Continuum CI remains green.
3. Cross-scope retrieval and memory writes are blocked or specifically authorized.
4. Explicit memory deletion and expiry work in a temporary database.
5. Tool calls are allowlisted, quota-limited and audit-logged without raw payload.
6. Shift evaluation refuses overlapping reference/shift data and compares
   reports with matching fingerprints.
7. The branch and PR retain all previous S-Brain development work.

## Next research milestone

Cycle 9 should benchmark real memory-augmented answers against non-memory
baselines; build *explicitly selected* tools with sandboxing and permission
tests; evaluate free-form predictions on independently authored tasks; and
only then integrate memory/tool routing into S's user interface.
