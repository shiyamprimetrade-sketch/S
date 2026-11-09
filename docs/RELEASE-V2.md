# AxiomFlow Continuum v2 — provenance and completion criteria

## Purpose

Preserve real human teacher–student conversations, allow locally controlled
retrieval, and explore an original from-scratch language model based on
individual UTF-8 byte tokens (including the user's letter/word philosophy).

## Implementation status (local container, not user's Windows PC)

- ChatGPT `conversations.json` export parser: implemented and integration-tested.
- Active-branch reconstruction, timestamps, text, role and media placeholders: tested on fixtures.
- Local SQLite archive project: persisted and searchable; non-chat role records stored separately.
- Browser JSON import, search, current conversation view: implemented; browser integration test only at API level, not manual cross-browser QA.
- Character-level causal Transformer from scratch: implemented and trained for 250 synthetic-corpus steps.
- Baseline checkpoint: 138,240 parameters; 250 steps; train loss last 2.285; one held-out sample 2.1478.
- Automatic transformer conversational quality: **NOT VERIFIED; currently low quality**. Don't call it a GPT clone.
- Existing Ollama support: preserved; requires separately installed open-weight model for useful replies.
- Tests: 40/40 local Python tests passed when created.
- Actual Windows 11 user device installation: NOT VERIFIED.
- Full original ChatGPT discussion imported: NOT VERIFIED (the export must be provided locally).
- Complete original image/audio bytes display: NOT IMPLEMENTED; placeholders only.

## Privacy and project separation

Only self-contained **source code** and synthetic assets are fit for public GitHub.
No raw private conversation text, private model checkpoints trained from user
content, Cloudflare tunnel credentials, backups or SQLite databases are to be
committed. The unredacted import option applies only to the owner's local disk.

## Tests that still need to run on Windows

1. Start server with `py -3 -m continuum.cli serve`.
2. Import *your actual* ChatGPT data export using the web Import ChatGPT button or CLI.
3. Verify complete chat sequence against the export, including tool/media placeholders.
4. Search for `keyboard`, `terminal`, `memory`, `ABCA`, and review correct sessions.
5. Install Ollama and a compatible open-weight model, choose it in UI, ask
   `What did my teacher say about errors?` and verify retrieval and citation of
   the remembered concept. Note the prompt context is bounded.
6. Run ByteLM training with your private imported data, evaluate fresh human
   prompts, save model and report errors; compare with synthetic-only baseline.
7. Back up the SQLite file, reopen, and verify retrieval after restart.

## Scientific limits

A 256-token vocabulary can encode arbitrary UTF-8 text; it cannot turn a finite
physical memory into infinite storage. Synthetic data is not new human input.
Generating characters demonstrates learned probability distributions, not the
same architecture, model weights, capabilities, or subjective experience as the
hosted ChatGPT assistant. Imported chat is context data unless explicitly used
for a local weight-training experiment.
