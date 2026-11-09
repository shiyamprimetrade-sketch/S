# Axiom Human Computer — milestones and proof rules

| Stage | Deliverable | Completion gate | Current |
|---|---|---|---|
| 0A | local desktop & goal/skill memory | offline launch, tests, restart continuity | tested in Linux environment, Windows untested |
| 0B | original AI generation attached | model loads/generates locally, response clearly labeled | existing trained AxiomFlow v2 model; UI adapter added, local integration pending validation |
| 1 | bootable Axiom kernel under QEMU | boot log, shutdown, exception test, image SHA | not started |
| 2 | own kernel services | process, memory, file, device, stress tests | not started |
| 3 | usable Axiom desktop | app lifecycle, accessibility, recovery, benchmarks | not started |
| 4 | continual teacher-driven learning | held-out transfer, source attribution, rollback | not started |
| 5 | hardware proof | FPGA/physical hardware running own ISA or defined compute substrate | not started |

## First hardware proof experiment

Create a freestanding kernel and bootable disk image for an **emulator target**, then demonstrate boot-to-console and recovery with automated logs. Use UEFI and QEMU documentation as reference, but treat firmware/bootloader/kernel/device boundaries as design choices. First physically test only after emulator evidence and a separate boot medium.

## Failure conditions

- A displayed SUCCESS that does not satisfy the task's postcondition is a failed experiment.
- A synthetic training score does not establish human conversational understanding.
- A runtime on Windows is an application, not an independently booting OS.
- The machine's owner must be able to inspect raw evidence and stop any action.