# S — Autonomous engineering continuation protocol

This is the canonical repository for **S**, an independent local human-like computer research project. Previously called Axiom Human Computer / AxiomFlow Continuum. Do not restart it or transfer work back to the old repository.

## Permanent operating rules

- Build the whole project across intelligence, symbolic meaning, training, memory, OS, hardware and user interaction; keep experiments separate from unverified claims.
- Understand the objective before applying keyword triggers; reasoning, verification and learning are default behaviors, not special commands activated by phrases such as "deep thinking".
- User is the teacher; preserve actual human messages separately from synthetic simulations and model-generated text. Never misattribute synthetic output to the user.
- Code and test real changes; log test evidence and known blockers. No fake success, simulated web searches or fake high-end model labels.
- Use **GitHub `shiyamprimetrade-sketch/S`** as the only canonical target for new work. The Axiom repository is read-only historical evidence.
- Preserve SQLite backups and model checkpoints locally; do not commit private transcripts, tokens, secrets or exported chat histories to a public repository.
- The local UI must work without CDN assets or paid AI APIs. Original S byte-model training uses PyTorch (installed locally), and optional third-party models must be clearly identified.
- Machine modifications (disk format, firmware flash, boot loader, kernel install, drivers) require independent emulation tests and explicit user approval; don't damage the existing Windows PC.
- Keep unrelated projects (Slickk Mail, Slickk-OS, Slickk Pay, Prime International, PixelForge) isolated.

## Engineering checkpoints

1. Run `py -3 -m unittest discover -s tests -v`.
2. Run `py -3 -m continuum.cli doctor` and `py -3 -m continuum.cli serve` on Windows; inspect the browser at localhost.
3. Import a small synthetic ChatGPT export, confirm conversation counts, search and feedback persistence.
4. Evaluate the local original byte model on held-out examples, separately from the rule scaffold.
5. Build the first S kernel in QEMU; verify boot and reproducibility before any hardware work.

Status is never "complete" without evidence. The final invention is not yet complete.
