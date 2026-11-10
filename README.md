# S — Independent Human Computer & Local AI

**v0.7 Apprenticeship checkpoint:** Added private teaching-history evaluation, held-out question controls, baseline vs teacher-defined meaning retrieval, snapshot evidence, and project-bound evaluation backups. See `docs/S-APPRENTICESHIP-EVALUATION.md`. The actual private ChatGPT archive has not been supplied to this build; we have **not** verified its import or trained a conversational model on it.

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

## S-issued API keys

S can generate its **own** cryptographically random API keys for authorized local clients. The **Settings → S API keys** panel supports generating, copying once, listing and revoking keys. Only SHA-256 hashes are saved in SQLite; private key material is never returned in a list or stored in GitHub. All API traffic is loopback-only. No third-party API key is generated.

CLI alternative (Windows PowerShell in `C:\S`):

```powershell
py -3 -m continuum.cli api-key create --name "My app" --scopes read,chat
py -3 -m continuum.cli api-key list
py -3 -m continuum.cli api-key revoke YOUR_KEY_ID
```

Call the authenticated local models endpoint (replace the placeholder with your privately saved API key):

```powershell
$SKey = Read-Host "Paste your S API key"
Invoke-RestMethod -Uri http://127.0.0.1:8765/v1/models -Headers @{ Authorization = "Bearer $SKey" }
```

Call the local chat API by providing an **existing** `session_id` from the S desktop:

```powershell
$payload = @{session_id="YOUR_SESSION_ID";text="Describe our next invention test";model="s-research:local"} | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8765/v1/chat -Method Post -ContentType application/json -Headers @{ Authorization="Bearer $SKey" } -Body $payload
```

The key protects `/v1/*` endpoints. Existing `/api/*` browser routes intentionally remain available to programs already running on the same computer to support the local S UI. This is **not an Internet-exposed multi-user authentication service**. Do not forward port 8765 through a router or tunnel. On a shared/untrusted Windows user account, local processes may access the browser API and the database; operating-system account separation is still required for stronger isolation.

## S v0.5 — Invention Lab (teacher → experiment)

Use the **⚗ Invention Lab** button in the S top bar to teach symbolic transformation examples, infer a candidate instruction, run independent held-out tests, and correct mistakes. Previous failures and later results remain in the project-local evidence ledger.

The lab runs only a fixed, inspected instruction vocabulary and optional two-operation compositions. **It is not yet a general autonomous coding agent.** See `docs/S-INVENTION-LAB.md` for proof criteria and the next development gates.

To update an already installed Windows v0.4 instance: stop S, extract this release, and run `powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\Upgrade-S-v0.5.ps1`. This tests the extracted source, backs up the private SQLite database, and then copies application files into the existing installation. The S database stays under your Windows user profile and is not committed publicly.

## S v0.6 — Meaning and Teacher Memory

S now provides **◇ Knowledge → Teacher Memory** for explicitly teaching principles, naming equivalent single/multiword expressions, recalling lessons in new questions, reviewing suggested corrections from your own imported conversations, revising mistakes without deleting history, and intentionally transferring a reviewed lesson between projects. Unreviewed suggestions and negative feedback never silently become trusted lessons. The app stays localhost-only and does not require an AI subscription.

**Important:** This is a transparent local memory and retrieval capability, *not* neural-network fine-tuning or human-level semantic comprehension. Full private conversation data cannot be reconstructed without your ChatGPT export, which must remain on your own PC. S Research and explicitly selected local Ollama models receive the reviewed teaching relevant to the active project only.

To upgrade an installed Windows S v0.5, stop S, extract v0.6, then run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\Upgrade-S-v0.6.ps1
```

This runs tests, checks whether the local app is still running, backs up the default local SQLite database if present, preserves a copy of installed program files, and installs the update. It does not delete chats or API keys. If your installation directory is nonstandard, supply `-Destination "C:\Your\S\Path"`. See `docs/S-MEANING-MEMORY.md` and run `py -3 -m examples.evaluate_meaning` for the 9-case illustrative test.
