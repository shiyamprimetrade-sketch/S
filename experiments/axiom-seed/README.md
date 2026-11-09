# Axiom Seed v0.1 — a small independently runnable learning skill

AxiomFlow research prototype. **No LLM, third-party packages, Internet, cloud or paid API required.** Requires Python 3.10+ on Windows/Linux/macOS.

Purpose: demonstrate a terminal that accepts teaching *examples* rather than a pre-written command for every new input. It searches a **fixed library of 12 candidate transformations**, selects a unique candidate only after at least two examples, predicts new results, and records held-out tests locally. This is **rule induction**, not new general intelligence, unconstrained program synthesis or an ability to execute arbitrary tasks.

## Windows PowerShell

In the folder containing `axiom_seed.py`:

```powershell
py -3 .\axiom_seed.py demo
py -3 .\axiom_seed.py teach letters APPLE FIVE
py -3 .\axiom_seed.py teach letters BANANA SIX
py -3 .\axiom_seed.py run letters CAT
py -3 .\axiom_seed.py prove letters CAT THREE
py -3 .\axiom_seed.py trace letters APPLE
py -3 .\axiom_seed.py inspect letters
```

`trace` demonstrates the user's teaching example `APPLE → FIVE → FOUR → FOUR` and detects the repeated result without looping infinitely.

All learned examples are written to `axiom_memory.json` in the current directory. Keep it private if it contains sensitive material.

## Principle

A terminal response is not reduced to arbitrary `ERROR` or `SUCCESS` labels. It can report `NEEDS_TEACHING`, `NEEDS_MORE_EXAMPLES`, `PREDICTION`, `TEST_PASS`, `TEST_FAIL`, and other meaningful states. An unverified prediction is **not** labeled a fact; a failed test does not get relabeled as success.

Example discovery is constrained by its available transformations. To create genuinely new operators, future versions must propose candidate programs, safely evaluate them, and demonstrate generalization using untouched tests. The version included here has no ability to modify its own source code.

The character counter currently counts Python Unicode code points, not user-perceived grapheme clusters or physical keys. Some non-English scripts may be surprising; improving that is a future testable change.

## Tests

```powershell
py -3 -m unittest discover -s tests -v
```