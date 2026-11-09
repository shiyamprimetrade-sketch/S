# Axiom Human Computer — full-stack architecture

The **complete computer** is the product. Today's code is the first **user-mode control-plane implementation**. It is intentionally installed *alongside* Windows to collect accurate evidence without overwriting the boot disk.

```
                       HUMAN TEACHER / OWNER
             speech · vision · keyboard · sensors · I/O
                                 ↓
                     AXIOM PERSONAL INTERFACE
       intention · conversation · accessibility · preferences
                                 ↓
                    AXIOM UNIFIED GOAL ENGINE
       interpret → hypothesize → plan → act → check → learn
          ↙             ↓             ↓               ↘
    SYMBOL/LANGUAGE   MEMORY       MODEL CORE          EVIDENCE
    tokens/words/ISA   events       original weights   reproducible
    new functions      skills       training/retrieval  tests/metrics
          ↘             ↓             ↓               ↙
                        AXIOM ACTION ENGINE
            explicit capability grants · testable goals
                                 ↓
                      AXIOM OS SERVICES
            processes · memory · files · networking · UI
                                 ↓
                        AXIOM KERNEL
          scheduler · address spaces · interrupts · drivers
                                 ↓
                          FIRMWARE
                 verified boot · startup · recovery
                                 ↓
                      HARDWARE INTERFACE
            CPU/ISA · RAM · storage · display · peripherals
```

## Implementation stages and contracts

**Stage 0 (implemented foundation):** user-mode desktop and terminal, SQLite evidence/memory, symbolic instruction synthesis, scoped file observations, hardware status, tests; optional original tiny byte-token model from AxiomFlow v2. This **does not replace Windows**.

**Stage 1 (boot work, not yet implemented):** architecture-neutral machine contracts, own freestanding kernel and boot image, virtual serial-console first boot; QEMU-only experimentation. Do not repartition the real computer or flash firmware.

**Stage 2:** scheduler, heap/virtual memory, interrupts, storage and verified boot persistence; crash and recovery measurement.

**Stage 3:** secure permissions, IPC and a human-centered graphical environment, keyboard and accessibility interfaces.

**Stage 4:** offline language/vision/speech input, independent from-scratch model training, retrieval, goal decomposition, verification; compare with a specified reference system.

**Stage 5:** new instruction-set experiments on a virtual CPU, then FPGA prototype and physical measurement; a truly new physical computer requires design/manufacturing/electronics validation.

## Design versus physical facts

Symbols and OS interfaces are changeable. Available physical storage, CPU time, RAM capacity, voltage, heat and energy are measurable properties; new compression, scheduling, architectures or external memory can change *effective utility*, but don't turn an unperformed action into a performed one.

## Reference interfaces

A device adapter reports capabilities and actual measured results. A task is a goal with evidence-bearing postconditions. An action may fail without aborting investigation, but only a validated postcondition permits a success status. The full machine's system state and autonomous decisions must remain inspectable by its owner.

The current demo reads host system properties, not a replacement kernel. It makes no unrestricted external shell calls. File reading is restricted to a separate workspace; writes require explicit authorization. GitHub source must not include private transcripts, tokens or local memory files.

## Relevant public references

- UEFI Forum specifications: https://uefi.org/specifications
- QEMU full-system emulation: https://www.qemu.org/docs/master/system/index.html
- OSDev bare-bones bootstrap background: https://wiki.osdev.org/Barebones

Research the standards, but design original Axiom contracts and verify them independently. No dependency here implies a ban on inventing better abstractions.