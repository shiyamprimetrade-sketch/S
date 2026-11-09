# S — Independent Human Computer & Local AI

**Canonical repository:** https://github.com/shiyamprimetrade-sketch/S

**Owner / teacher:** Shiyam Talukder. **Purpose:** Invent an independent, human-like computer system with its own learning, symbolic language, memory, execution mechanisms, and eventually original OS/hardware. The project is never considered complete by changing a label or simulating an output. Progress requires execution and verification.

The local chat is named **S**. The supplied S chat design has been recreated as a fully offline-capable static interface served by our local Python application. It is not a hosted ChatGPT clone. It stores local messages in SQLite and responds through **genuine installed engines**, not mock replies.

## Quick start on Windows 11

Install Python 3.11 or newer, selecting **Add Python to PATH** if using the installer. Download this repository or extract the S Windows ZIP; open PowerShell in its folder:

```powershell
py -3 -m unittest discover -s tests -v
py -3 -m continuum.cli doctor
py -3 -m continuum.cli serve
```

Open **http://127.0.0.1:8765**. Only localhost is supported by default. No internet or subscription is needed for **S Research**. No third-party Python modules are required for S Research, stored chat, private import, lesson notes or export.

Run the separate whole-computer experiment (Tkinter) using:

```powershell
py -3 -m human_computer.cli desktop
```

This does **not** replace Windows or flash firmware. The original S kernel/physical hardware are separate experimental milestones.

## Installed model choices — no fake intelligence

| UI | Engine | Works without cloud? | What it really does |
|---|---|---|---|
| S Research (default) | `s-research:local` | Yes, Python stdlib | Retrieves project notes and produces a bounded, testable research scaffold. **Not a generative LLM.** |
| S Mini | `teacherlab:local` | Yes, with bundled checkpoint | 4,000-simulation-trained topic classifier plus templates. **Not a conversational LLM.** |
| S Original | `charllm:local` | Yes, Python + PyTorch | Locally trained, original tiny byte-token Transformer (138,240 parameters). Generates text but **not yet a capable chat model**. |
| Other local model | Ollama, optional | Yes after its download | Third-party open weights; **not S-original** and never used silently. |

The user-supplied design listed names resembling other providers' models. The interface intentionally displays real engine identity and capabilities instead of claiming an untrained model is high-end.

**S Original**: The source package includes an experimental checkpoint where distributed with the downloadable S ZIP. If absent after cloning GitHub, create your own checkpoint with:

```powershell
py -3 -m pip install torch
py -3 -m continuum.charllm train --steps 250
py -3 -m continuum.cli serve
```

Training reads **synthetic template data** unless explicitly provided the private corpus project. Generated text will be fragmentary with this tiny model. To deliberately train on imported private conversations, follow `docs/S-CONTINUITY.md`.

## Bring our full conversation into S

A chat window in this answer cannot guarantee access to every full historical message and binary attachment. We **do not put private transcripts or passwords in a public GitHub repository**. The built-in importer accepts `conversations.json` from a ChatGPT data export; in Settings select **Import conversation JSON**. Large exports and ZIPs:

```powershell
py -3 -m continuum.cli import-chatgpt "C:\path\to\chatgpt-export.zip" --title "distinctive conversation title"
```

The import creates a separate local project and preserves message order and original text where available. It redacts recognized credentials by default. Imported images/audio appear as metadata placeholders, not recovered binaries. After import, select the archive project in the sidebar, search lessons and add correction notes. **Import is retrieval/memory, not automatic training.**

## Continuity and backups

The database is at `%USERPROFILE%\.s-computer\s.sqlite3` on Windows (override with `S_DB`). Nothing is uploaded automatically. Save a verified snapshot and export the project before switching machines or subscriptions:

```powershell
py -3 -m continuum.cli backup --output .\S-backup.sqlite3
py -3 -m continuum.cli export --project YOUR_PROJECT_ID --output .\S-project.json
py -3 -m continuum.cli handoff --project YOUR_PROJECT_ID --output .\S-handoff.md
```

**Caution:** S-backup files contain your private messages and belong outside the public Git repository. The web interface can export project JSON and handoff Markdown. Python restoration explicitly requires `--confirm` and creates a pre-restore backup.

## What we are inventing

S is a full-system project, not merely a chatbot: language/meaning, perception, goal planning, symbolic rule synthesis, machine operations, verification, memory, kernel, OS, firmware, and ultimately dedicated computer hardware. The installed UI is one subsystem.

See `docs/S-CONTINUITY.md`, `docs/S-MIGRATION.md`, `docs/HUMAN-COMPUTER-CHARTER.md`, and `docs/HUMAN-COMPUTER-MILESTONES.md`.

**Verified scope:** The local browser/API/SQLite code and 52 preexisting tests were tested in this build; S-specific integration tests are included. Windows graphical startup and real PC hardware remain to be checked on the user's device.

## Optional one-time installation

After extracting the Windows ZIP, run PowerShell in the extracted directory:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\Install-S.ps1
```

This copies program files into `%LOCALAPPDATA%\S-Computer` and creates a desktop shortcut. It requires Python 3.11+ and **does not require admin rights**. The private chat database lives separately under `%USERPROFILE%\.s-computer` and is not overwritten. To start without installing, double-click `Start-S.cmd`.
