# AxiomFlow Continuum — independent invention workspace

**Status:** A runnable local-first chat prototype, persistent project memory and backup/export system. Uses a **separately installed local open-weight AI model** via Ollama. V2 bundles a tiny 138,240-parameter byte-token checkpoint trained on synthetic templates; it does **not** recreate GPT-6 or provide comparable reasoning. No cloud subscriptions or API keys required *after* Python, Ollama and a model have been downloaded. No automatic tool execution or full self-modification is claimed.

## Quick start — Windows 11

1. Extract the ZIP to e.g. `C:\AxiomFlow-Continuum` on a drive with adequate free space. Do not use the 16 GB Ubuntu live USB for the model files.
2. In Windows PowerShell, check Python: `py -3 --version`. Requires Python 3.11+; if missing, use `winget install -e --id Python.Python.3.12` and reopen PowerShell.
3. Install Ollama for Windows (via `winget install -e --id Ollama.Ollama` if available), start Ollama, and download a small open-weight model: `ollama pull qwen2.5:1.5b`. This is a suggested model ID, not an assertion that it is already installed. Models must be downloaded while online, but inference can run without the internet.
4. Open PowerShell in the extracted project directory and run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\Test-Continuum.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\Start-Continuum.ps1
```

5. Open `http://127.0.0.1:8765`. Create independent projects, start sessions, record lessons and hypotheses, and chat with the local model.

## If no local model is installed

- The web UI, project notes, conversation history, word relations, export and backup work. `teacherlab:local` offers a limited prebuilt teaching-response classifier; `charllm:local` offers experimental from-scratch character generation. Fluently following broad questions requires a separately installed open-weight Ollama model.
- Check `ollama list`, `ollama serve` (if the service is not running), and `py -3 -m continuum.cli doctor`.
- Default model is `qwen2.5:1.5b`; set `$env:AXIOM_MODEL='your-model-id'` or edit the UI model field to use a model already installed locally.

## Backup and migrate between computers or model providers

```powershell
# Stop the UI with Ctrl+C before restoring; backup may run while the app is serving.
py -3 -m continuum.cli backup
py -3 -m continuum.cli doctor
# Export one project: get its ID from the UI API or the exported JSON.
# The UI has an Export Project button; copy the JSON to another computer.
py -3 -m continuum.cli import .\your-project.json
# Restore a whole SQLite snapshot ONLY with server stopped:
py -3 -m continuum.cli restore .\backups\continuum-YYYYMMDD-HHMMSS.sqlite3 --confirm
```

All messages and user-approved knowledge notes are held in `data/continuum.sqlite3`. Backups go to `backups/`; keep at least one **encrypted copy on another device**. The **Handoff** button creates a Markdown recap of recent conversations, project lessons, hypotheses and evidence for any other chat model. It does not export model weights. Exported JSON is *plaintext*: keep it private. Backup/restore and data migration use standard SQLite/JSON, not a proprietary cloud format.

## Project boundaries

The app creates a separate workspace named `Independent Computing & KEY26`. It includes **user-taught lessons** (symbols, word counting, rule generation, error analysis) and explicitly labelled **hypotheses**, rather than claiming any memory miracle or autonomous consciousness. Other projects can be created separately.

## What is not yet built

- A local model with GPT-6-level reasoning (not promised).
- Automatic web browsing, arbitrary shell execution, full IDE integration, model fine-tuning, or autonomous self-training.
- Encrypted-at-rest SQLite; protect the Windows account and device, and encrypt portable backups.
- Exposure to the public internet. The web server is deliberately bound to `127.0.0.1`.
- Always-on execution when Windows is off. The OS, machine, model and electricity must be available.

## Technical construction

- `continuum/server.py`: Python standard-library HTTP server, loopback-only, host and Origin checks.
- `continuum/storage.py`: SQLite WAL-compatible portable project/session/message/notes storage, backup/restore and import/export.
- `continuum/engine.py`: local Ollama `/api/chat` bridge with real error reporting, time measurement and context notes.
- `continuum/relations.py`: deterministic word-length/anagram/co-occurrence discoveries. These are statistical/structural associations, **not** proof of general intelligence.
- `web/`: responsive local chat UI, project/notes/relationship explorer.
- `docs/INVENTION-PROTOCOL.md`: preserved teaching checkpoints.
- `tests/`: automated local tests.

## Threat model

Localhost-only is not complete authentication against a malicious program already running on the same PC. The application does not include a public-user login system. Do not bind it to `0.0.0.0`, forward it through Cloudflare Tunnel, or expose it to the LAN without adding authentication, CSRF protection, TLS and access controls. Do not paste secrets into project notes. Model output can be wrong; test all code before execution.

## Teacher–Student ML invention lab (new)

This repository also includes `continuum/teacherlab.py`, a deterministic offline
simulation generator and a trained **intent-classification** model. There are 4,000
synthetic teacher/student exchanges and 15 teacher-lesson paraphrases in separate
provenance-marked JSONL files. A 350-example heldout-template report is committed.
This does not train a general conversational AI from scratch or reproduce GPT-6.

```powershell
py -3 -m continuum.lab_cli build --simulations 4000
py -3 -m continuum.lab_cli evaluate
py -3 -m continuum.lab_cli ask "Why should errors trigger experiments?"
py -3 -m unittest discover -s tests -v
py -3 -m continuum.cli serve
```

In the browser, select `teacherlab:local` to try the classifier without Ollama.
For a more fluent local dialogue, install an open-weight model in Ollama; its system
context will include bounded TeacherLab teaching examples and stored project notes.
See `docs/TEACHER-STUDENT-ML.md`. The original private chat is **not** included in
this public repository; only curated technical paraphrases were used.

**Reproducibility:** The complete 4,000-row synthetic dataset is committed compressed as `data/simulations/teacher_student_4000.jsonl.gz`; `lab_cli build` recreates the working plaintext JSONL.
