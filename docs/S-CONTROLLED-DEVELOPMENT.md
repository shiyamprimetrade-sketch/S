# S v0.2 controlled development and bundled neural checkpoint

This extends the merged S main checkpoint `b348ecfd97c527bc5b9d2c728a8fed413fa1475a`.
It preserves the existing native model architecture, independent training,
S-Memory, S-Tools, chat interfaces and every prior Git checkpoint. It does not
replace installed model weights or give existing read-only S-Tools host powers.

## What this checkpoint implements

Approved English requirements can produce an immutable source bundle, appear in
the file explorer/editor, export as a ZIP, and enter an explicitly approved
container build/test loop. Up to two repairs may create new source checkpoints
and rerun the fixed language adapter. Generation failures, compiler/test errors,
blocked runtimes and repair failures are recorded without replacing prior source.
A successful process is labelled `toolchain_passed`; business acceptance remains
unverified until independently checked. Generated tests can be incomplete or
misleading, so their exit status alone is not full requirements verification.

Three initial recipe/adapter targets exist:

| Recipe | Template output | Actual container check |
| --- | --- | --- |
| static-site | Informational HTML/CSS page containing the approved name, purpose and requested-feature list | Parse HTML title, start an actual internal HTTP server and GET the page |
| python-cli | Standard-library CLI returning the approved description and a unittest | Discover and execute at least one Python test |
| c-cli | C program printing the approved name/purpose | Compile with C17 warnings-as-errors and execute the binary |

These **starter templates do not implement arbitrary requested business
features**. For example an inventory-management request produces a starter,
not a stock database or a working inventory application. The UI and provenance
say so. Generic neural-generated code can target these adapters if it meets the
source-bundle contract, but useful general autonomous coding is not demonstrated.

## Neural model: actual training, installation and failed coding gate

The package now includes a small genuine checkpoint of the existing `SBrain`
Transformer: **23,648 parameters, 256 AdamW updates**, independently initialized
with seed 20261111. Training uses self-authored synthetic code-format examples;
no private conversations, teacher output or externally pretrained weights.
Weights use safetensors; optimizer pickle is not shipped or loaded.

Final bundled weight SHA-256:
`a6caeb1f219c01f42cec3ab54fb5dcdb61e20d6530dc91173a7805bdd7d89859`.

On fixed held-out windows, validation NLL changed from **29.09885597229004** to
**1.6647034883499146**. Parameter update L2: **25.20088813773971**. Held-out names
are disjoint from training names, but the examples share a synthetic format.
This is a small learning experiment, not evidence of real-world generalization.
The final checkpoint is the predeclared 256-update endpoint, not a claim that
its validation loss is the minimum over the learning curve.

The actual neural code-format diagnostic **FAILED**: output was not valid JSON.
The failed result, full measured curve and training provenance are shipped in
`resources/native-demo/training-report.json`. An earlier sampled-validation run
is retained in `reports/s-developer/native-initial-sampled-validation.json`; it
is not substituted for the final fixed-window evaluation. The code never labels
a deterministic template or Ollama response as S's independently generated code.

An actual local installation of these weights was exercised in this executor.
This does not mean the model is installed on Shiyam's PC or phone. Neural code
inference failed its format gate; no model-generated application was verified.

### Install/use on a supported PC

From the reviewed source checkout, with Python 3.11+:

```powershell
# Optional runtime packages come from your configured trusted Python index.
py -3 -m pip install ".[neural]"
.\scripts\S.cmd model install-demo
.\scripts\S.cmd develop application
```

`install-demo` copies the checksum-verified bundled weights and manifest into the
selected database directory's `s-native-models/<hash>` and configures the **S
Developer provider only**. The existing native-console host settings/checkpoint
are not overwritten. The command installs files without needing PyTorch;
`torch_dependency_present` is not a claim of successful native inference.
If Windows Application Control blocks torch DLLs, that runtime remains blocked;
this change does not bypass operating-system protections.

To configure an existing separately trained S checkpoint instead:

```powershell
.\scripts\S.cmd model install-native --checkpoint "C:\Path\To\SCheckpoint"
```

To reproduce the small experiment into a **new** directory:

```powershell
.\scripts\S.cmd model train-demo --output "C:\New\S-Demo-Experiment" --steps 256
```

Training does not overwrite previous checkpoints. Installation verifies digest,
training counters, byte vocabulary and architecture/resource limits. Manifest
provenance is checkpoint-declared; it is not a cryptographic audit of who trained
arbitrary third-party supplied weights.

### Optional external local neural provider

If an operator already has a supported local Ollama runtime, S can explicitly
install/configure an optional external starter model:

```powershell
.\scripts\S.cmd model pull-ollama --model qwen2.5-coder:0.5b --approve-download
# Or select an already installed local model without downloading:
.\scripts\S.cmd model configure-ollama --model qwen2.5-coder:0.5b
```

This model is external and is always labelled `ollama-external`; it is not S's
independent model and is not installed by default. The adapter calls only
`127.0.0.1:11434`, bypasses environment proxies, forbids redirects and rejects
cloud/remote model metadata. The operator is responsible for the local service's
own configuration; S does not administer its daemon or outbound policy. No
Ollama runtime/model was available in this executor, so its code-generation
capability is **unverified**, not represented by mock tests or template success.

Protocol references: [generate](https://docs.ollama.com/api/generate.md),
[pull](https://docs.ollama.com/api/pull.md). The API is not called silently after
native-model failures: there is no provider fallback.

## Build and automatic repair permissions

Docker with a working local Linux-container engine is required. Preparing images
is a separate explicit operator operation; no project or model output can pull
an image or install packages:

```powershell
.\scripts\S.cmd runtime prepare python --approve-download
.\scripts\S.cmd runtime prepare gcc --approve-download
.\scripts\S.cmd --enable-builds develop application
```

The fixed official image tags are `python:3.12-slim` and `gcc:14`. Preparation
records the actual official repository digest in `s-developer-host.json`; later
builds use that digest with `--pull=never`. No claim is made that these are the
latest toolchain releases. Image provenance is registry/digest based, not a
package-vulnerability or vendor-signature audit. See Docker's official
[container execution documentation](https://docs.docker.com/engine/containers/run/).

In the workspace:

1. Complete and approve the saved requirements.
2. Open the source/model/build workspace, choose a recipe and explicitly choose
   template, native S, or the configured external provider.
3. Approve source generation; inspect its provider, files and source hash.
4. Edit source if needed and save a new source checkpoint.
5. Choose 0–2 repairs, check action consent and confirm container execution.
6. Review actual diagnostics. Restore an earlier source checkpoint or export the
   saved ZIP with `S-PROVENANCE.json`.

The host flag is distinct from browser action consent. A request cannot change
host config, select a model path/image/command, grant filesystem/network access
or increase the repair budget beyond two. Updated/unapproved requirements block
execution of old source. Stale source edits are rejected transactionally.
Changing project clears source, consent and diagnostics; unsaved-source guards
prevent accidental loss during ordinary project changes.

A native or external provider proposes complete replacement files for a repair;
prior source checkpoints remain available for rollback. Template repair is an
explicit narrow heuristic: a missing colon on the known Python `describe`
function, or a missing semicolon on a C `return 0` line. It is not model reasoning
or a general diagnosis engine. An unmatched/repeated/no-change repair stops.

CLI equivalents for an approved project ID:

```powershell
.\scripts\S.cmd project generate --project PROJECT_ID --recipe python-cli --provider template --consent
.\scripts\S.cmd --enable-builds project build --project PROJECT_ID --provider template --max-repairs 1 --consent
.\scripts\S.cmd project status --project PROJECT_ID
```

Failed/blocked builds return a nonzero CLI exit status. No source is considered
working merely because it was generated.

## Execution and filesystem boundary

Source is stored in project-scoped, immutable SQLite artifact revisions, not
written into the application or host's working repository. Before a build, only
validated source text is materialized into a new temporary directory. Paths are
relative, portable and bounded; traversal, drive paths, hidden files, reserved
Windows names, case collisions, file/directory collisions and oversized bundles
are rejected. Maximum: 20 files, 64 KB/file and 256 KB/bundle. Export is built
from these same validated entries, never from arbitrary host directories.

A container receives **only** read-only source and a code-owned read-only driver.
It has no host network, Docker socket, repository, user home, model weights,
database or credentials mounted. Root filesystem is read-only; build output
uses ephemeral tmpfs. UID/GID 65534, no capabilities, no-new-privileges, 64 PIDs,
512 MB memory/swap, one CPU, file/output/time limits and default Docker seccomp
remain. Proxy environment variables are cleared. Source strings cannot choose
or interpolate shell commands. The C shell driver is fixed trusted S code.

Output is bounded; timeouts/output overflow kill the Docker CLI **and explicitly
remove its container**. The Docker logging driver is disabled for these jobs.
Build products remain ephemeral; ZIP export currently contains source, not a
Windows/mobile executable. Containers share a kernel and are not proof against
all container escapes; this is not an audited hostile multi-user service.

No runtime dependency installation occurs during builds. Missing imports or
libraries are reported; a model cannot resolve them by silently running pip/npm.
The starter recipes use only standard/system libraries. Full dependency
resolution/provenance scanning, full terminal, application preview, broad
language adapters and distributable app packaging remain future work.

## Phone and existing-PC upgrade

The Termux installer now also installs the tiny bundled checkpoint into phone
storage when no existing native Developer provider is configured. It preserves
an existing configured provider. It does not install a PyTorch runtime or prove
Android neural inference. Standard Termux normally cannot run Docker containers;
there is deliberately no unrestricted-host build fallback. Requirements, source
editing, history and export remain usable without a model or container runtime.

The reviewed-file PC updater includes the new source files, browser assets and
bundled checkpoint. Use the dry-run/backup/apply workflow in
[S-PHONE-AND-PC-UPGRADE.md](S-PHONE-AND-PC-UPGRADE.md), then run `model install-demo`
inside that installed copy if desired. Existing data/model settings remain local.
No device was remotely updated in this session and PC/phone data are not synced.

## Verification scope

Baseline: 70 existing app/updater tests passed. Current local suites: 88
app/orchestration/security tests, 796 S-Brain tests (including actual bundled
neural generation) and 23 UI-state tests passed. The fresh wheel installed in a
clean virtual environment, installed its packaged neural checkpoint, served
assets, saved a specification and reopened it after restart. GUI rendering is
unverified; JavaScript checks use deterministic DOM/network fixtures.

This executor has no Docker CLI/service, and Bubblewrap cannot create the
required socket. Therefore **local actual container builds were blocked**. The
new `S controlled development acceptance` GitHub job runs real, unmocked Docker
checks for website HTTP, Python tests, C compilation/execution, intentional
syntax error/one repair, host-path/env/source-write/network denial and output
limits. Its results are reported separately in the PR; a mock result is never
counted as those acceptance tests.

Reproduce:

```sh
python -m unittest discover -s tests -v
python -m unittest discover -s s_brain/tests -v  # optional neural dependencies
node --test scripts/test_brain_ui.cjs scripts/test_developer_ui.cjs scripts/test_coding_ui.cjs
python scripts/verify_controlled_development.py --approve-runtime-download --report build/controlled-acceptance.json
python scripts/smoke_developer_install.py --launcher /path/to/installed/S
```

This is an implementation checkpoint, not completion of the entire master
prompt. Neural general coding competence, clean Windows 11 installer H,
Android model inference, arbitrary-business acceptance and the full original
language/toolchain expansion have not been demonstrated.

Initial real acceptance CI (run 43055092461) passed website HTTP and Python
execution, then found that the temporary C build mount was non-executable by
Docker default. Compilation succeeded, execution returned permission denied.
The runtime now explicitly allows execution only on the disposable `/work`
tmpfs, with `nosuid,nodev`; `/tmp` remains `noexec`. The failing C acceptance
check remains enabled. The acceptance script now saves partial evidence before
assertions, so a failed step retains earlier build results as an artifact.
