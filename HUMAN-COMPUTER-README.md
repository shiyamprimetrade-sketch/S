# Axiom Human Computer — foundation for a complete human-like computer

This repo addition starts the **whole-computer** project: hardware, firmware, kernel, OS, desktop, cognition, memory, symbol-based learning and verified action. It deliberately **does not claim that the entire computer has already been built**.

The first runnable component is an independent Windows desktop/control plane that can accept goals without requiring "deep thinking", preserve feedback and memories, observe workspace files, synthesize new symbolic instructions from examples, validate outputs, and show real machine information. If integrated with AxiomFlow Continuum v2, it can also invoke the project's **own tiny byte-token model** for experimental unverified text generation. No subscription or cloud inference is necessary.

## Windows 11

1. Install Python 3.11+ (with Tkinter) if missing.
2. Extract the release ZIP or clone the Axiom repository containing these new files.
3. In PowerShell, enter the extracted directory and run:

```powershell
py -3 -m unittest discover -s tests -v
py -3 -m human_computer.cli desktop
```

Or open `scripts/Start-Axiom-Human-PC.ps1` from a PowerShell terminal.

Optional original Axiom byte-model generation requires AxiomFlow v2 files, its saved model checkpoint and `py -3 -m pip install torch`. It will be labeled **UNVERIFIED_GENERATION**, not a verified intelligent action.

## Commands in the desktop or terminal

```text
status
Invent a computer that learns to execute new instructions
goal Build the Axiom boot sequence
teach mirror CAT => CATTAC
teach mirror DOG => DOGGOD
invent mirror
perform mirror SUN
verify mirror SUN => SUNNUS
remember rule1 Labels are not outcomes
recall rule1
evidence
generate Discuss learning and language
```

For terminal mode: `py -3 -m human_computer.cli shell`.

Data is local at `%USERPROFILE%\.axiom-human-computer` including `axiom_pc.sqlite3`, learned skills, and a dedicated `workspace/`. Do not upload that directory to a public repository. No disk partitioning, driver installation, firmware flashes or host command execution occurs.

## Program scope

Read `docs/HUMAN-COMPUTER-CHARTER.md`, `docs/HUMAN-COMPUTER-ARCHITECTURE.md`, and `docs/HUMAN-COMPUTER-MILESTONES.md`. The ultimate target is **an independent, physical, human-like computer**. All stages are measured. This code is the **first measured foundation**, not a final system.