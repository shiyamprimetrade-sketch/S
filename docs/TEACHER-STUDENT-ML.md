# AxiomFlow Teacher–Student Training Lab

## Research goal

Preserve the teacher's principles of symbol combinations, word-number relationships,
self-discovery through interaction, pattern reconstruction, and terminal feedback.
Generate a reproducible local experimental training set and measure what a small
learner actually extracts. **Do not confuse algorithmic intent detection with
human-level intelligence or imitation of two real people.**

## Data provenance and permission

- `data/lessons/curated.jsonl`: **15 manually curated paraphrases** of technical
  teaching exchanges available in the source conversation. These are NOT a complete
  transcript and are NOT labelled as verbatim quotes. Names, credentials, personal
  history, and unrelated projects are deliberately excluded from this **public** repo.
- `data/simulations/teacher_student_4000.jsonl`: **4,000 synthetic template-generated
  teacher prompt/student reply pairs**. Each line includes `source`, `scenario`,
  `label`, and a false `human_verified` flag. This is reproducible with a seed.
- `models/teacher_student_nb.json.gz`: a small learned **multinomial Naive Bayes
  classifier** for topic/interaction classification. This is ML, but it is not a
  language model, not a fine-tuned ChatGPT and not evidence of consciousness.
- `reports/teacher_student_evaluation.json`: a measured test on **350 differently
  templated examples** not included in training. Because all examples come from
  hand-authored templates, even these results do not establish real-conversation
  generalization.

**No full chat was copied to a public repository.** To use the entire exact chat,
export it from the chat platform under your own control, carefully remove secrets,
private information, and unwanted conversation threads, and import approved excerpts
into a *private* dataset. Public data publishing is a separate opt-in step. Never
commit passwords, API tokens or private conversations to GitHub.

## Commands (Python 3.11+, Windows 11 / Linux; no Python packages required)

```powershell
py -3 -m continuum.lab_cli build --simulations 4000 --seed 20261109
py -3 -m continuum.lab_cli evaluate
py -3 -m continuum.lab_cli ask "How can a word discover a relationship?"
py -3 -m unittest discover -s tests -v
py -3 -m continuum.cli serve
```

The local UI is at `http://127.0.0.1:8765`. Select `teacherlab:local` for an
explicit *offline, non-LLM* classifier-and-template response. For free-form natural
conversation, install a suitable open-weight model in local Ollama; the app adds
bounded teaching notes and retrieved examples to its system context. That is
**retrieval/context conditioning, not modifying Ollama's model weights**.

## How to judge progress

1. **Reproduction:** same seed gives identical 4,000 synthetic records.
2. **Honesty:** synthetic records are not mislabelled as actual teacher messages.
3. **No leakage:** all test templates differ from train templates.
4. **Classification:** record the actual heldout confusion matrix and accuracy.
5. **Behavior:** use 30 new, manually written prompts and have the teacher assess
   whether responses are accurate, appropriately curious, and evidence-aware.
6. **Real-world capability:** run local model inference and measure latency,
   memory and usefulness on the actual Windows computer. It is not tested here.
7. **Continuity:** confirm conversation records survive restart and migrate via
   project export/import and a restored backup.

## Next research stages

- Collect teacher-approved, diverse real dialogue examples **privately**.
- Create an evaluation set written independently of the synthetic generator.
- Compare retrieval-based prompting vs. small-model fine-tuning on the same heldout
  test cases; do not claim improvement without a measured baseline.
- Add richer role-play sequences, reward/critique feedback, and provenance tracking.
- Add redaction, access control and encrypted backups before supporting public access.
