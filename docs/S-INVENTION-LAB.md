# S Invention Lab v0.5 — teacher → hypothesis → experiment → correction

This milestone adds executable, *bounded* symbolic program discovery to the complete S Human Computer project. It is not an unrestricted autonomous inventor or a new operating system.

## Why this exists

The teacher can introduce an observable goal, show examples, reserve independent tests, then evaluate S's inferred operation. A failure stays a failure. The teacher can correct a mistaken expected result and trigger a new synthesis attempt. The lab preserves original failing runs and new passing runs, with SHA-256 fingerprints of experiment evidence.

## Example: S as a meaningful transformation

1. In S, open **⚗ Invention Lab**, select the S project, and create a goal.
2. Teach `CAT → CATS` and `DOG → DOGS` as **teach** examples.
3. Reserve `SUN → SUNS` as a **held-out test**.
4. Click **Discover rule**: S selects `append_s` from a bounded library.
5. Click **Run tests**: the actual candidate runs on the unseen input. S records whether it matched.
6. Make a deliberate wrong test, observe **FAIL**, correct it, and rerun. Earlier evidence is preserved.

S never trains on the held-out examples. It cannot run arbitrary Python, shell commands, or network searches from this lab.

## What is implemented

- Per-project goals with observable success criteria.
- Distinct teaching examples and held-out tests.
- Small auditable operations (`append_s`, `reverse`, `duplicate`, etc.) and up to two-step compositions.
- Evidence-labeled hypothesis/verified-on-tests/falsified/unresolved statuses.
- Teacher corrections that invalidate previous candidate and trigger new synthesis.
- Append-only experiment journal, test inputs/outputs, SHA-256 fingerprints, persistent SQLite storage.
- API endpoints at `/api/inventions` and `/api/invention/...` accessible only through the existing loopback application security boundary.
- Project export/import and text handoff contain invention evidence; verified SQLite backups include all tables automatically.

## Interpretation limits

A passing held-out test proves the program produced specified outputs on **those** cases. It does not prove all language semantics, generalized human understanding, autonomous computer invention, or unlimited ability. The `append_s` operation is a surface transformation; the English plural has exceptions.

## Next capability gates

- **v0.6:** Add explicit teacher correction retrieval from real ChatGPT exports and benchmark S on previously unseen teacher problems; distinguish retrieval from weight training.
- **v0.7:** Add supervised local tool planning and code-writing experiments with manual approval, process isolation, time limits, test evidence and rollback. Do not run generated host commands unrestricted.
- **v0.8:** Train/evaluate original S models on privately selected datasets, compare against baseline models with held-out generalization tests, and report failures honestly.
- **v0.9:** Add invention-task dependency graphs, checkpoint/resume, research citations and evidence-only self-improvement proposals.
- **OS/Hardware:** Build an original S kernel under QEMU and compare observable behavior with existing kernels before physical device changes.
