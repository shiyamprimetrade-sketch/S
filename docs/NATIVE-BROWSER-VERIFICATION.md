# Native browser integration and reasoning verification

## Outcome

S v0.6 now exposes **S Native Lab** (`s-native:local`) in its browser model selector. It loads the independently trained S-INT-002 80-step checkpoint, generates a bounded response locally, saves the answer with the correct model identifier, and displays an explicit warning. Chrome selection, generation and conversation persistence after page reload were verified. The live browser reply matched the independently scored checkpoint output exactly.

The reasoning diagnostic scored **0/24**. There is no evidence of usable reasoning or reliable instruction following in this model. This is a successful software integration with a failed capability diagnostic, not an intelligence milestone.

## Changes

- `s_intelligence/browser.py`: lazy PyTorch loading, checkpoint availability, ASCII/length validation, serialized seeded generation, 80-character output bound, raw generation separate from UI warning.
- `scripts/integrate_native_browser.py`: guarded installer for the existing v0.6 engine, health/model listings and frontend catalog. Backs up changed files and refuses existing native integrations/packages/fixtures. It does not replace the installed app with the older research-branch Continuum code or change the database schema.
- `s_intelligence/reasoning.py` and `data/evaluation/native-reasoning.json`: scored diagnostic using the exact browser generation adapter.
- Unit and installation integration tests; sanitized measured results.

Installed app changes are limited to `C:\S\continuum\engine.py`, `C:\S\continuum\server.py`, `C:\S\web\app.js`, the new `C:\S\s_intelligence` package and synthetic evaluation fixture. Other engines remain available, and S Research remains the default. Reloading resets the selector to that existing default; saved messages retain their Native Lab attribution.

## Browser verification

Opened `http://127.0.0.1:8765` in Chrome. Created a separate project named `Native browser verification - synthetic, not teacher history`. Selected S Native Lab and sent `2 + 2 =`. The response was persisted, with S Native Lab attribution and an unreliability notice. Reloaded the page, selected the synthetic project, reopened the conversation and confirmed the same saved response. The screenshot accompanies this report.

The response was not `4`. Its raw generated text was:

```text
nuleevhord, am ranca a.
Thneden scre * sed lhidniirenibes catl s g panevi sorl s
```

## Capability diagnostic

The 24 synthetic cases were authored after the checkpoint was frozen and before scoring. No training or tuning used these cases. All prompts, including the appended `Answer:` suffix, fit within the 32-character context window. Scoring compares the complete generated answer after case, whitespace and terminal-period normalization. It does not award credit for accidentally containing an expected digit or word.

| Category | Correct | Total |
|---|---:|---:|
| Copying control | 0 | 4 |
| Arithmetic | 0 | 4 |
| Letter counting | 0 | 4 |
| Ordering and simple sequence | 0 | 4 |
| Transitive/equality logic | 0 | 4 |
| Insufficient-information control | 0 | 4 |

Copy failures demonstrate basic instruction-following limitations. These short-answer failures cannot isolate abstract reasoning from language generation. This small diagnostic cannot prove an absolute absence of all reasoning, but it gives no basis for claiming reliable reasoning. The expected unknown response in insufficient-information cases is a convention the model has not been taught.

The installed and research evaluations matched exactly. Native inference and evaluation ran with socket.connect denied. This verifies the native path does not need Ollama or a remote AI service. The app health endpoint still probes the optional loopback Ollama service to discover other engines; no external inference service was used.

## Validation

- Research suite: 42 passed, 0 failed.
- Existing v0.6 suite on the staged patched app: 116 discovered, 115 passed, 1 Windows symlink skip.
- Native integration tests: 3 passed on the staged app, then 3 passed against the installed app using temporary databases. Checks cover health advertisement, absent-checkpoint hiding, HTTP generation, Unicode error handling, model attribution, deterministic output and persistence.
- Frontend renderer: 5 regression cases passed.
- All prior database rows preserved across 15 tables. Only the explicitly labeled synthetic browser project/conversation was added to the live database.
- Python compilation and Git whitespace checks passed.

A test-copy invocation initially used the wrong working-directory-relative path; it was corrected before the integration tests ran. Final test logs are included. Optional NumPy is absent; PyTorch prints its existing warning, but the tested tensor paths run successfully.

## Installation and recovery

Checkpoint: `C:\S\native-research\S-INT-002\artifacts\s-int-002\steps-80-seed-20261110.pt`.

Original app file backup: `C:\S\upgrade-backups\native-browser-20261110-100329-374270`.

Pre-change SQLite backup: `C:\S\backups\before-native-browser-20261110.sqlite3`.

The existing desktop shortcut launches the updated app. Choose **S Native Lab** from the top model selector. ASCII only; 32-character context; no conversation-history or Teacher Memory conditioning. Unicode prompts receive an explicit error rather than silent substitution. The browser saves the user's attempted prompt even when inference fails, consistent with the existing chat route.

To reproduce the diagnostic from `C:\S`, choose a new output filename:

```powershell
& 'C:\S\.venv\Scripts\python.exe' -m s_intelligence.reasoning --output native-reasoning-new.json
```

For a fresh compatible v0.6 installation with the verified checkpoint already present, stop its server, run `scripts/integrate_native_browser.py <installation>` from the research checkout, then restart. The installer refuses an already-integrated installation. To roll back the adapter, stop S, restore the three backed-up app files, and restart. Restoring the database is not needed for adapter rollback and would discard newer messages.

## Remaining limits

No reasoning training, architecture expansion, Unicode tokenizer, chat-memory conditioning or hosted CI validation was added. The model remains an experimental text generator. Next work should establish basic copying and instruction following on dedicated training/validation data, then evaluate a separate sealed test set before claiming reasoning improvements.
