# S v0.2.2: independent coding capability diagnostics

Purpose: distinguish valid source, successful process exit and actual expected
results. This continues the merged portable runtime (PR #17); it does not retrain
or promote neural weights, restart S, or claim a generally capable coding model.

## What is measured

The new versioned suite `s-public-coding-diagnostic-v1` requests three small
Python functions: sum positive integers, sorted unique lowercase words and a
case/ASCII-space-insensitive palindrome check. Each function receives a JSON
object through `solve(p)` and returns the specified JSON object.

There are three cases per task: one public case and two withheld from generation
and repair. **The suite is public, synthetic and reproducible**, not a secret,
uncontaminated or representative industry benchmark. Withheld means excluded
from the current model prompts/repair feedback, not undiscoverable on GitHub.
A three-task pass is never sufficient for general model promotion.

Generation receives only the task contract and source-bundle protocol. Candidate
source is validated using the existing bounded-file rules. Its own `test_app.py`
is not the acceptance oracle. For independent execution, a fixed driver imports
`app.py`, supplies one case input and calls `solve`. Only that input is sent over
container stdin; expected answers and other cases are not mounted in the
container. The trusted host separately compares the finite JSON result, including
key/type distinctions. Extra stdout, empty output or a plain success exit is not
an answer. For example JSON boolean true is not accepted as integer 1.

This is observable input/output testing of a small number of examples. It cannot
prove absence of malicious behavior, source correctness for all inputs, useful
reasoning, business acceptance or freedom from memorization. Docker isolation
and the existing permission boundary still apply.

## Permissions and commands

From a current source checkout use `python -m s_developer.cli` instead of `S`, or
`.\scripts\S.cmd` on Windows. Format-only checks do not execute generated code:

```sh
S evaluate coding --provider native --consent --output new-native-report.json
```

Prepare the official runtime explicitly when needed, then request independent
container cases with a separate host gate and action approval:

```sh
S runtime prepare python --approve-download
S --enable-builds evaluate coding --provider native --execute-tests --max-repairs 1 --consent --output new-code-report.json
```

Existing output files are refused rather than replaced. The CLI returns nonzero
unless all three tasks pass actual independent execution. Format-only execution
is correctly non-passing; an unassessed result is not zero measured accuracy.
Reports identify provider/model, prompt/source hashes, source history, initial
and final task correctness, assessed task count, actual errors and scope.

The Developer coding panel has an **Evaluate selected provider** action. A saved
project and action consent are required; container execution additionally needs
the operator's existing build opt-in and prepared pinned runtime. Evaluation uses
synthetic contracts, not private project requirements, and records its report in
that selected project's history. Changing projects clears displayed results.

At most one repair is allowed per benchmark task (the ordinary project workflow
still has its own 0–2 budget). Repair receives only public-case feedback. If the
public case passed, withheld inputs/answers/observations are not disclosed in
repair feedback. Prior sources and diagnostics remain in the report. Blocked
runtimes, invalid generation, unmatched/no-change repairs and context limits remain
explicit non-passing outcomes. No dependency installation or host fallback occurs.

## Context integrity correction

Native providers previously truncated prompts to their context budgets. That
could discard requirements before generation and invalidate capability claims.
They now raise `ContextLimitError` instead of dropping required input. Portable
source generation has a 64-byte input budget; PyTorch generation uses the declared
checkpoint context. The evaluator uses compact complete contracts under the
bundled PyTorch checkpoint's 256-byte budget. Portable evaluation reports an
input-context block when the complete contract does not fit, rather than
pretending to evaluate an unseen task. No external model/template fallback exists.

## Actual neural baseline

The unchanged independently initialized S checkpoint has SHA-256
`a6caeb1f219c01f42cec3ab54fb5dcdb61e20d6530dc91173a7805bdd7d89859`.
An actual local PyTorch run on all three complete compact contracts produced
**0/3 valid source bundles**. All failed JSON decoding. **Zero tasks reached
independent execution**, so semantic accuracy was not measured. No model was
promoted. The full report is retained at
`reports/s-developer/native-coding-baseline-v1.json`.

This failure is a development baseline, not a claim that independent neural
coding is impossible. The next learning milestone needs authorized training data,
adequate model/context/compute, and evaluation tasks kept separate from training.
Do not train on these exact evaluation answers and report the resulting score as
independent generalization.

## Verification

Local suites: 102 app/evaluator/security/upgrader tests, 798 neural tests and 25 UI
state tests passed. Unit tests that inject candidate/runner fixtures test
orchestration and answer-leakage controls; they are not actual neural successes.

`scripts/verify_capability_evaluation.py` performs real isolated container runs
using explicitly labelled, self-authored **test-fixture source providers**. It
checks correct functions, an `os._exit(0)` false-success attempt, a constant public
answer that fails other cases, and wrong functions corrected by a fixture repair.
Generated test files deliberately raise an error if imported, proving they are
not the acceptance oracle. Only the fixture provider is substituted; Docker and
host comparisons are real. A fixture pass is never labelled S neural capability.
The GitHub controlled-development job preserves the full JSON report and prints
labelled summaries. Its actual status is recorded in the related PR.

The local executor has no Docker service, so actual container verification must
come from that CI job or an appropriately configured PC. Physical phone inference
and the full Windows installer/business acceptance matrix remain separate checks.
