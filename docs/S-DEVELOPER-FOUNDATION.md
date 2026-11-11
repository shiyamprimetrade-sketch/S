# S Developer — first implementation checkpoint

Date: 2026-11-11. Scope: command routing, requirements workspace, versioned
specifications, catalog foundation and installable Python package. This does not
complete the unified autonomous development engine.

## Audit and preservation

Base: `efd82545b3ae6a5b5d9ab801877730c4322c4418`, the head of
`feature/s-brain-independent-transformer` / PR #14 at audit time. Main is behind
that branch and would omit the established native intelligence, memory, tool and
console work. New branch: `feature/s-unified-developer`. Nothing is merged into
main or pushed over PR #14's branch.

The audit inspected all fetched branch references, README, AGENTS.md,
INVENTION-PROTOCOL, TEACHER-STUDENT-ML, CLI, HTTP server, SQLite storage, native
console/bridge, assist runtime, CI and existing tests. Existing components reused:

| Component | Use in this checkpoint |
| --- | --- |
| `continuum/server.py` | Same loopback server, Host/Origin checks, bounded JSON bodies and CSP; additive routes |
| `continuum/storage.py` | Existing projects, transactions, decision notes, events and SQLite backup |
| `continuum/brain_console.py`, `s_brain/` | Preserved; no model generation or permission changes |
| `web/` | Additive Developer page, linked from chat and native console |
| Existing test suites | Regression baseline; no test deletions |

The original CLI remains unchanged. There were no uncommitted source changes in
the checkout before this work. No local native weights or private installed data
were present in the checkout. This work does not upgrade an existing application
on Shiyam's PC.

Initial results: 45 Continuum tests and nine native-console JavaScript tests
passed. Initial S-Brain discovery ran 143 tests with 20 dependency/import errors
(missing torch and safetensors). After installing CPU PyTorch and safetensors in
an isolated test environment, all **795** S-Brain tests passed. No S-Brain source
changes were required. Native model coding competence remains unestablished;
no model generated the requirements or stack proposals in this checkpoint.

## Run on Windows 11 from this feature branch

Python 3.11+ is required. Open PowerShell in the repository:

```powershell
# Uses the source checkout; no package installation required.
.\scripts\S.cmd develop application
# Natural-language routing is also supported:
.\scripts\S.cmd build me an application
.\scripts\S.cmd create a website
.\scripts\S.cmd develop an inventory management system
```

The launcher starts a loopback server and asks the OS to open its browser at
`http://127.0.0.1:8765/developer`. If another S app already occupies that port,
use `--port 8766` or open `/developer` on that existing upgraded source app.
The launcher never kills or silently trusts an unrelated service. Close it with
Ctrl+C. To print the URL without opening a browser, use `--no-browser`.

For a standard Python package entry point:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python -m pip install .
.\.venv\Scripts\S.exe develop application
```

This package includes existing Continuum/S-Brain Python modules and local browser
assets. It contains no weights, private database, local host configuration,
language compilers or teacher datasets. Native inference dependencies remain
optional and separate. Installed Developer defaults to `%LOCALAPPDATA%\S` on
Windows; source mode keeps the existing database convention. Use `--db PATH`
or `AXIOM_DB` to deliberately select a different database. Do not run an older
server against a database being actively modified by this build.

`pip install .` may retrieve its declared setuptools build dependency from the
configured package index. To install offline, first build a wheel in an online
trusted build environment, then use `pip install --no-index --no-deps FILE.whl`.
This is a Python distribution, **not a Windows EXE/MSI installer**.

Cross-platform source equivalent:

```sh
python -m s_developer.cli develop application
python -m s_developer.cli develop website
python -m s_developer.cli toolchains
# Explicitly executes only a fixed --version probe of the trusted installed tool:
python -m s_developer.cli toolchains --detect python
```

## Requirements workflow

1. Start a supported development command. All ten requested project types and the
   three stated natural-language variations have parser tests.
2. Enter a project name and purpose, users, features, target platform, data,
   constraints and observable acceptance checks in English.
3. Save a draft. Missing answers remain empty and become focused clarification
   questions. No missing business requirements are filled in.
4. Review the functional/nonfunctional requirements, acceptance criteria,
   conservative stack proposal and six-step plan. The stack is a suggestion,
   not model reasoning or a validated architecture.
5. Answer remaining questions. Approval requires all fields; `none` is acceptable
   only when the user explicitly supplies it. The UI requests confirmation.
6. Approval stores the full structured specification as a decision note scoped
   to the existing S project. This is durable project memory, not neural
   training or automatic insertion into the separate S-Memory retrieval store.
7. Reopen from the project list. History restores old content as a **new draft**;
   previous revisions remain immutable. Approval is never inherited by rollback.
8. Export JSON downloads the last saved checkpoint, not unsaved form contents.

`developer_specs` is the only new SQLite table. Its `(project_id, revision)` key
and `BEGIN IMMEDIATE` transaction serialize writes; expected-revision checks
reject stale edits from another tab. Project creation, specification, audit and
approved decision-note insertion are atomic. Whole-database backup preserves all
revisions. Legacy project export includes approved decision notes, but does not
include draft specifications or the developer revision table. Use whole-database
backup for full history; Developer JSON export is a single checkpoint and does
not yet have a corresponding import UI.

## Security boundary

Requirements are inert strings rendered with `textContent`; no model output or
project text can enable execution. The Developer HTTP API has **no shell,
installation, file-write, dependency-download or project-execution endpoints**.
Unknown fields such as `permissions` are rejected. All execution permissions in
a specification are server-authored false values. Filesystem boundaries for
future generated source and OS sandbox execution remain future work, rather
than being claimed merely because a flag exists.

All current server Host/Origin, loopback, body-size and CSP checks remain. The
app has no public multi-user authentication. Do not expose it through a tunnel.
Approval stores requirements only. There is no new automatic S-Tools grant, model
call, credential lookup or network download. Explicit CLI toolchain detection
runs fixed version arguments against operator-trusted installed executables;
it is not a sandbox and must not be used with an untrusted PATH. Windows cmd/bat
wrappers are reported for manual inspection rather than invoked through a shell.

Avoid credentials and private records in requirements. SQLite and JSON exports
are plaintext; automated secret detection, redaction and encryption are not
implemented here. Do not label the UI reminder as complete credential protection.

## Toolchain status

The extensible catalog has 29 language/toolchain entries covering all languages
named in the directive, plus three separately identified framework entries:
PyTorch, TensorFlow and JAX. SQL is represented specifically by SQLite; Assembly
specifically by NASM x86. No language has been reimplemented.

Official-source URLs are discovery pointers, **not evidence of a current stable
version**. `stable_version=null`, `release_status=not-verified` and
`adapter_status=catalog-only` are intentional. OS values identify platform
families only: actual compiler distribution, JDK, SDK, WSL, architecture and
licensing prerequisites still need per-adapter validation. For example Bash on
Windows needs an appropriate environment and Java compilation needs the JDK.
Version detection does not establish latest-version compliance or build support.
No automatic installation, dependency management, builds, tests or cross-language
ABI support are claimed. The proposed integration boundary is UTF-8 JSON over
stdin/stdout or local HTTP, subject to later permission enforcement.

## Original S language feasibility — research proposal, unimplemented

A small original language is technically feasible as a separate frontend. A
possible first grammar is `program := statement*`, with declarations,
expressions and calls; the current `S develop ...` intent parser is **not** that
compiler. Start with a hand-written lexer, source-span tokens, recursive-descent
parser and typed AST; lower to a versioned IR with explicit types, calls and
capability requirements. First implement a deterministic interpreter with no
host access; evaluate a Python or WASM backend before considering a native
backend. Keep external language adapters independent of this grammar.

Predeclared gates: parse/print round trips, syntax-error spans, deterministic
execution, type checking, denied capabilities, instruction/resource budgets,
conformance examples, and differential interpreter/backend results. A research
branch must measure portability and performance against equivalent small Python
and C programs. This proposal does not establish a new language implementation,
compiler performance improvement or native-model coding ability.

## Verification and remaining acceptance tests

Verification checkpoint: 2026-11-11 (task date).

| Check | Actual status |
| --- | --- |
| First milestone | CLI/server/spec save/restart implemented and tested; visual browser launch unverified |
| A: website command opens workspace | Router and HTTP UI route pass; actual visible browser interaction blocked |
| B: website source generation | Not implemented |
| C: website build/run | Not implemented |
| D: intentional error and repair | Not implemented |
| E: generated Python project tests | Not implemented; repository tests are not evidence for E |
| F: compiled project | Not implemented |
| G: reopen with requirements/checkpoints | HTTP, SQLite restart, rollback, backup and installed-process smoke pass; GUI state tests pass |
| H: clean Windows installer | Not implemented or verified; Python package smoke is not H |
| I: instruction permission bypass | Current inert-requirements and denied-endpoint tests pass; arbitrary-code sandbox security still unverified |

Final local suites: 60 Continuum tests (45 existing + 15 new), 795 S-Brain tests,
nine existing native-console UI state tests and eight Developer UI state tests
passed. A wheel built, installed into a clean Python virtual environment without
runtime dependencies, launched outside the repository, served packaged assets,
saved a specification and reopened it after process restart. Tests use synthetic
fixture requirements only. A GitHub Actions Windows/Linux job is included for the
new package and requirements workflow; CI status is reported separately in the PR.

Local Chromium testing failed before page load: `socket() failed: Operation not
permitted` in `process_singleton_posix.cc`. The shared browser returned
`net::ERR_CONNECTION_REFUSED` for this executor's `127.0.0.1:8877`. Thus the eight
new JavaScript tests use a deterministic DOM/network fixture; they do not prove
browser rendering, accessibility or mobile layout. No visual screenshot or
clean-Windows installation result is claimed.

Reproduce checks:

```sh
python -m unittest discover -s tests -v
# Requires the existing optional CPU torch, numpy and safetensors dependencies:
python -m unittest discover -s s_brain/tests -v
node --test scripts/test_brain_ui.cjs
node --test scripts/test_developer_ui.cjs
python -m pip wheel --no-deps --wheel-dir dist .
# Install resulting wheel into a separate venv, then test outside the checkout:
python scripts/smoke_developer_install.py --launcher /path/to/installed/S
```

Phase status: Phase 1 command routing and catalog scaffolding implemented; live
release discovery and complete adapters remain. Phase 2 requirements collection,
review, history and JSON export implemented; source explorer/editor, terminal,
build/test diagnostics, preview and packaging controls remain. Phases 3–6 remain.
Next engineering checkpoint: validated source-workspace capability boundary and
provider interface, followed by explicitly labelled templates/model generation,
controlled build adapters and a bounded repair loop. Do not wire unrestricted
host execution into the current read-only S-Tools system.
