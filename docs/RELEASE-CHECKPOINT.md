# AxiomFlow Continuum — initial public code checkpoint

- Source: local-first chat application, project-scoped knowledge, SQLite persistence,
  browser UI, backup/restore and handoff capsule.
- Training: 4,000 synthetic paired dialogue *simulations* derived from seven
  learning categories and 15 curated human-lesson paraphrases.
- ML: pure-Python multinomial Naive Bayes; not a general language model.
- Synthetic heldout: 251/350 correct (71.71%) on different templates. This is not
  a real-human generalization score.
- Tests: 32 Python unittests passed in the authoring environment.
- Reality check: no Windows installer executed here, no downloaded Ollama model
  inference tested, no training of a large language model, no independent agent
  autonomy, no hardware-limit breakthrough.
- Security: public repository excludes original verbatim conversation, credentials,
  personally identifying details, and live secret tokens. Lesson-derived paraphrases
  are marked as such. Never publish raw private conversation without review.
- Simulation data: compressed at `data/simulations/teacher_student_4000.jsonl.gz`;
  re-create plaintext with `python -m continuum.lab_cli build --simulations 4000`.

## Proof commands

```shell
python -m unittest discover -s tests -v
python -m continuum.lab_cli evaluate
python -m continuum.lab_cli ask "How do we learn from terminal errors?"
python -m continuum.cli doctor
```

## Next independent checkpoint

Test the app on the actual Windows machine with Ollama installed. Make a private
benchmark of teacher-selected prompts that are not synthetic templates, score
behavioral alignment with explanations, and test full project transfer to a second
machine. Preserve separate logs for synthetic and human data.
