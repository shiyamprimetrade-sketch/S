# S v0.7 — Apprenticeship and real teaching-history evaluation

## Purpose

S's goal is to learn from the human teacher and eventually become capable of inventing. A high score on memorized or synthetic questions is not evidence of that. This release adds a **teacher-controlled, local evaluation system** that measures whether approved teachings transfer to **new questions** and compares two clearly defined methods.

**What it measures:** top-1 retrieval of an approved teacher lesson or a deliberately expected no-match response. Baseline ignores teacher-defined phrase synonyms; taught-meaning recall uses them. Held-out questions live only in the benchmark table and are never added to lesson cues or a neural training corpus. The system rejects exact copies of an expected lesson cue.

**What it does not measure:** broad comprehension, quality of generated answers, successful code invention, self-training, or correctness of the teacher's beliefs. Results from hand-authored questions must not be described as representative of the real world.

## Using your actual chat export on Windows (private, local)

1. Get the official ChatGPT data export from your own account. Make sure you have its ZIP or `conversations.json` on your PC. **S does not have the full private conversation by default.**
2. Back up your existing S data before updating. Keep the backup outside public Git repositories.
3. Stop S; extract the v0.7 package into a separate folder and run the upgrade script from that folder:

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\Upgrade-S-v0.7.ps1
   ```

4. Import the export into a separate local project (substitute the real file path):

   ```powershell
   cd "$env:LOCALAPPDATA\S-Computer"
   py -3 -m continuum.cli import-chatgpt "$HOME\Downloads\YOUR-CHATGPT-EXPORT.zip"
   ```

   `conversations.json` also works; if your existing installation is at `C:\S`, navigate there instead. The importer redacts credential-like text by default, but do not assume automatic redaction is exhaustive.
5. Launch S at `http://127.0.0.1:8765`, select the archive project in the left sidebar, then select **Knowledge → Find corrections in this project's chats**. Treat results as suggestions. Inspect the source and approve or reject each one.
6. To teach the main **S — Human Computer** project, use **Teach another project** on a reviewed lesson; the archive's messages remain separate. Approved lessons are not scientific facts, and imported data is not neural training.
7. In the main S project, open **Knowledge → S Apprenticeship**. Add questions that were not used as teaching examples. Select the expected approved lesson, or **No lesson should match** for negative controls. Run **Compare retrieval methods**.
8. Review both successes and failures, revise teaching as appropriate, and rerun the identical dataset. The evidence journal records separate snapshots and SHA-256 fingerprints of the dataset and relevant approved memory.

## Score interpretation

- **Baseline correct / evaluated** = number of cases whose top-1 result matches the teacher expectation using literal word matching.
- **Taught meaning correct / evaluated** = the same criterion with explicitly taught concept/phrase equivalences.
- **No-match control** = correct only if the retrieval returns no lesson.
- **Stale excluded** = a benchmark referencing a lesson that was revised or removed. It is **not** silently scored; replace the test with a new reference.
- **Dataset SHA** = fingerprint of the questions and expected answers, not a proof of their correctness.
- **Memory SHA** = fingerprint of approved lessons and the defined concept vocabulary used in that run.

The benchmark does not inspect generated model answers. New, genuinely unseen invention challenges with runnable outcomes must be introduced and evaluated in the later engineering-action phase.

## Evidence and continuity

- New project-local data tables: `apprenticeship_cases`, `apprenticeship_runs`.
- New localhost routes: `GET /api/apprenticeship/{cases,runs,readiness}` and `POST /api/apprenticeship/{cases,delete,run}`.
- Project exports include benchmark cases and historical run snapshots. Imported cases rebind to restored lesson IDs. Historical hashes describe the old snapshot, so rerun after import for fresh comparable evidence.
- SQLite backup includes every case, lesson, run and local conversation. Handoff Markdown includes the latest run's digest and summary.
- `/v1` scoped bearer-token API remains unchanged; local browser routes are **not an internet-ready authentication boundary**.

## Verify this build

```powershell
py -3 -m unittest discover -s tests -v
py -3 -m examples.evaluate_apprenticeship
py -3 -m continuum.cli serve
```

The second command intentionally uses only **two designer-authored questions**: one paraphrase and one negative control. It is an executable regression demonstration, not evidence of human-like intelligence.

## Known blockers and next gate

- A real export is required to measure performance on the teacher's actual past conversations. No private data was provided with this build, so **actual ChatGPT import remains unverified**.
- No complete offline general-purpose AI model is trained by this release. S's original tiny byte model remains experimental; installed open-weight models are separate.
- S cannot yet autonomously propose arbitrary source-code changes and safely execute tests. This is the next planned functional milestone, with explicit per-project permissions and rollback.
- Windows installation and GUI startup on the teacher's physical PC require checking locally. Automated tests and a Chromium browser smoke test only verify the development environment.
