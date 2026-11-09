# S — Permanent local continuity and teacher–student learning

## Purpose

Keep the S invention usable after an online subscription ends. The program, user conversations, synthetic simulations, experiments, and model checkpoints must be portable, locally inspectable, and restorable. This is an engineering requirement, not a promise of autonomous continuation without a running computer.

## What is original and what is not

1. **Human-origin teaching**: The original chat export comes from the user and is imported privately. Public GitHub contains only **curated paraphrases** from earlier teaching sessions, not the entire raw conversation.
2. **Simulations**: `data/simulations/teacher_student_4000.jsonl.gz` contains 4,000 programmatically generated teacher–student examples. They are synthetic and must not be presented as real human dialogues.
3. **Classifier**: `models/teacher_student_nb.json.gz` is an independent small offline classifier, not an LLM.
4. **Original byte LLM**: `continuum/charllm.py` is a tiny trainable Transformer; the optional checkpoint is at `models/s_byte_lm.pt`. This model does not yet generalize well to natural conversation.
5. **Research scaffold**: `continuum/s_reasoning.py` makes deterministic, evidence-labeled notes and plans using recorded project knowledge. It is not a neural network.
6. **Other local AI**: Ollama integration remains optional, explicitly selected only; its weights are not S-owned.

## Private import workflow

1. Export ChatGPT data using your ChatGPT account's available data export interface. Download a ZIP or extract `conversations.json` to Windows.
2. Start S and select Settings → Import conversation JSON. Use the CLI for ZIPs or exports larger than 35 MiB.
3. S creates a **new archive project**. Each message is stored locally in chronological order using original timestamps where available. Non-text media is represented by placeholders. The default importer redacts credentials; examine its results for sensitive information.
4. Search the imported archive, choose a thread, then add notes in the project knowledge area. Correct any inferred interpretation that is wrong.
5. Export project JSON and a handoff capsule. Keep a SQLite backup. Test restoring the copy before assuming continuity.

## Private model training workflow

After importing, find the selected project's ID from the model's local database using `py -3 -m continuum.cli export` or inspect the browser project selector. Run this **deliberately**; the program never silently trains on all private chat:

```powershell
py -3 -m pip install torch
py -3 -m continuum.charllm train --corpus private --db "$HOME\.s-computer\s.sqlite3" --project "YOUR_PROJECT_ID" --steps 400
```

This creates a local checkpoint, not an exact replica of a hosted model. Never claim that training loss proves broad reasoning capacity. Evaluate the checkpoint on messages held out from training and on genuinely new teacher questions.

## Current milestones and proof gates

- Source preserved in S repository with historical branches linked to old Axiom PRs.
- Offline S interface: sends actual requests to the locally installed engine; saves messages and feedback in SQLite.
- Original S model: trained with small synthetic corpus (research baseline), optional PyTorch installation on Windows.
- Whole computer: kernel boot under an emulator not yet shown, native OS not yet built, physical S computer not yet prototyped.

Do not delete the old Axiom repository until full migration and tests are accepted. Preserve raw historical commits and PRs there as an audit trail.
