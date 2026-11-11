# S-Brain application integration

This integration extends the existing Continuum backend with an optional local experimental neural runtime. It preserves Cycles 1–10 at commit `676e17da5e5208a6d91fda2fc190ba29717c2df9`. It does not promote the experimental checkpoint as a useful production assistant.

## Architecture and boundaries

`web/brain.html`, `brain.js`, and `brain.css` provide project selection, native chat, explicit memory controls, tool approval, and measured learning status. The main app links to `/brain`. `continuum/server.py` routes `/api/brain/*` and the explicit `s-brain:experimental` chat model to `continuum/brain_console.py`.

The adapter reuses `checkpoint.py`, `model.py`, `assist_runtime.py`, `memory_store.py`, `tools_control.py`, and existing grounding and retrieval modules. Operator-owned `s-brain-host.json` enables the model, memory and tools independently. Requests cannot edit host permissions or choose checkpoint paths. With no configuration the integration remains disabled.

The existing curation, teacher review, curriculum training, evaluation and checkpoint-selection modules remain command-line development workflows. They are not automatic background training. The optional Ollama Teacher is identified separately and is never a native inference fallback. Teacher candidates require review before training; this integration used no teacher output or private conversations.

Memory requires explicit consent and an existing project. It supports search, revision hashes, expiry and deletion. Memory is stored separately beside the application database in `s-brain-memory.sqlite3`. Chat still persists through the original app's conversation store; chat is not automatically copied into S-Memory or training data.

Tool proposals are entered explicitly and require approval, host enablement, the existing allowlist and quotas. The adapter never treats generated text as permission. Only the existing read-only tools are exposed. Successful assistant audit summaries enter the app event log; memory operations retain their existing audit records. This is a trusted local desktop boundary, not multi-user authentication or public hosting.

Context overflow is rejected rather than silently truncated. Existing KV-cache code and standalone `server_v2` streaming remain intact. The embedded browser chat returns one completed response; SSE and NDJSON are available through the standalone runtime, not streamed into this console.

## Installation and recovery

Use a separate checkout for research. Do not replace an installed S v0.6 tree with this branch's older Continuum files. `scripts/install_brain_console.py` applies guarded additive changes to a stopped v0.6 installation, verifies a trained safetensors checkpoint, and backs up the two modified original files. It refuses an already integrated installation. Back up the app database separately before installation.

From the repository root with CPU PyTorch, NumPy and safetensors installed:

```powershell
python scripts/install_brain_console.py C:\S s_brain/checkpoints/integration-run/step-00000080
```

The installed operator configuration is:

```json
{"enabled":true,"memory_enabled":true,"tools_enabled":true,"checkpoint":"s_brain/checkpoints/integrated"}
```

Restart the existing launcher and open `http://127.0.0.1:8765/brain`. Disable individual features by changing the local host configuration and restarting. Configuration, weights and private databases must remain outside Git. To recover, stop the app and restore the backed-up server and main index; preserve the added files and databases until recovery is verified. Never weaken Windows security policy to load a runtime.

## Measured Windows evidence

Baseline: 789 S-Brain tests and 45 branch Continuum tests passed. Integrated: 795 S-Brain tests and 45 Continuum tests passed. The staged installed v0.6 suite passed 115 tests with one Windows symlink skip. Thirteen live HTTP checks covered actual checkpoint inference, scoped memory, consent, revision/deletion, unapproved and approved tools, foreign-origin rejection, fail-closed grounding, ordinary generation, SSE and NDJSON. Native tests prohibited socket connections to prove no teacher dependency.

An independently initialized byte-level model trained for 80 optimizer steps on authorized synthetic S-INT-002 fixtures. It has 23,648 parameters and a 1,024-token context. Final training loss was 25.76681137084961; validation loss was 25.528810501098633. Measured gradient norm was 9.03590202331543 and parameter L2 change from seeded initialization was 1.5517800100262125. These values prove training occurred, not useful intelligence.

Reproduction after preparing permissioned train/validation/test byte files:

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
python -m s_brain.train_advanced --corpus-dir s_brain/checkpoints/integration-corpus --output-dir s_brain/checkpoints/integration-run --steps 80 --batch 1 --sequence 1024 --accumulate 1 --dim 32 --layers 1 --heads 4 --kv-heads 2 --evaluate-every 20 --save-every 40 --evaluation-batches 2 --seed 42
```

The local corpus and checkpoint are not distributed with this source. Exact reproduction requires the same authorized corpus and dependency versions. The tested Windows environment used Python 3.12.10, CPU PyTorch 2.15.0, safetensors 0.8.0 and NumPy 2.5.4.

`reports/brain_integration_benchmark.json` records two held-out synthetic cases, evaluated with the same checkpoint and seed with and without memory. Both arms scored zero exact match and zero F1. No improvement or promotion is supported. The declared minimum gain is now passed into the evaluation policy, fixing a misleading nested gate result. This two-case diagnostic is not a representative capability benchmark.

Browser checks exercised explicit consent rejection, add/search/edit memory, fail-closed sourced answers and an approved calculator call. The calculator executed, but the neural model did not produce a useful answer. Empty generation now displays an honest placeholder. The final placeholder was verified in the in-app browser after restarting the installed app. Both the prior backup and current SQLite database passed integrity checks; all prior rows in all 15 application tables were preserved. No private content was exported.

## Remaining release gates

The implementation commit passed S-Brain Linux CPU CI and all four existing Continuum Windows/Ubuntu jobs. Final documentation-commit CI is recorded in PR14 and the local delivery checkpoint. Browser verification, installed startup and database preservation passed. The next release phase still needs recovery rehearsal and a release manifest. The dashboard reports missing best-checkpoint evidence as `Not measured`. No broad reasoning, generalization, production readiness, public-access security, or real-world RAG improvement has been demonstrated.

The next model milestone is a predeclared instruction-following benchmark with independent data and a fixed training budget. Promote only if that gate and regression checks pass; retain failed results.
