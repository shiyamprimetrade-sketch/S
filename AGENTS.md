# Codex / Engineering agent directive — AxiomFlow Continuum

Read `README.md`, `docs/INVENTION-PROTOCOL.md`, and `docs/TEACHER-STUDENT-ML.md`
first. The purpose is an **independent local research assistant**, preserving the
teacher's instruction method and project boundaries. Work from verified checkpoints;
implement experiments, run tests, report actual results, and do not claim invented
capacities. Preserve existing human data and separate it from synthetic simulations.

## Priorities

1. Run tests and verify Windows 11 startup; keep the server strictly bound to loopback.
2. Preserve notes, lessons, sessions, backup, export/import and handoff continuity.
3. Record the source and label of every training example. Never treat synthetic text
   as verified human statements. Do not publish exact private conversations.
4. Train and evaluate TeacherLab before changing the conversation model; report
   heldout results, gaps and your precise test commands.
5. Use locally downloaded models via Ollama when available; use `teacherlab:local`
   transparently as a simple offline non-LLM fallback. Never quietly call cloud APIs.
6. Do not give this agent unsupervised shell or network powers. Significant actions
   require explicit authorization. Test generated commands before real deployment.
7. Do not alter SLICKK Mail, SLICKK-OS or other independent repositories.
8. Never commit passwords, tokens, private transcripts or user-specific secrets.
9. Do not claim that prompts or 4,000 simulations create the actual assistant's
   weights, inner experiences, ongoing background work, or identical capabilities.

Completion criteria: reproducible source, passing tests, runnable Windows commands,
local SQLite continuity, accurate evidence report and no fabricated AI claims.
