# S phone installation and existing-PC upgrade

Checkpoint date: 2026-11-11. These are installation paths for the requirements
workspace. No direct connection to Shiyam's phone or PC exists in this session.
Publishing or merging GitHub source does not update either installed device.
No Android APK or native-model inference result is claimed.

## Phone: Termux + local browser

The installer uses the existing source without pip, root privileges, shared
storage permissions, model downloads or PC access. It writes a launcher to
`~/.local/bin/S` and keeps the phone's SQLite database at
`~/.local/share/S/continuum.sqlite3`. The source checkout must stay in place.
If it finds an unmanaged S launcher, it stops and preserves it. Reinstalling a
managed launcher retains the previous version as `S.previous-<id>`.

In Termux, install prerequisites from its configured package repository, then
clone S into a fresh private directory. The integration branch below is retained
after merge; it contains the tested native-console base plus Developer changes.

```sh
pkg install -y python git
s_phone_dir="$(mktemp -d "$HOME/S-phone-XXXXXX")" &&
git clone --branch feature/s-unified-developer https://github.com/shiyamprimetrade-sketch/S.git "$s_phone_dir/source" &&
python "$s_phone_dir/source/scripts/install_termux.py" &&
"$HOME/.local/bin/S" develop application
```

The last command starts a loopback server. S attempts to open the Android browser
using the installed `termux-open-url` helper. If no browser appears, manually open
`http://127.0.0.1:8765/developer`. Keep Termux running. Ctrl+C stops S. Restart with:

```sh
"$HOME/.local/bin/S" develop application
# Website command:
"$HOME/.local/bin/S" develop website
# Or print the URL without trying to open a browser:
"$HOME/.local/bin/S" --no-browser develop application
```

Termux's official URL helper is documented by its
[source](https://github.com/termux/termux-tools/blob/master/scripts/termux-open-url.in).
Only the fixed locally generated localhost URL is passed as an argument; a shell
is not invoked. We do not infer browser behavior from a helper returning success.

The requirements workspace, history and export need no neural model or Ollama.
The normal S chat and native console remain present, but neural chat needs its
separately supported runtime and checkpoint. Android CPU/RAM compatibility,
model inference, background lifetime and actual browser rendering remain
unverified. Android may stop background apps; this is not an always-on service.
Do not copy private PC databases to the phone without explicitly reviewing that
transfer. There is no automatic PC/phone synchronization.

Phone verification still required: see the page, save a draft, stop S, restart,
reopen the project and check its saved purpose and revision. A phone screenshot
or terminal output can establish what actually ran; host tests cannot.

## PC: preserve the installed S system

When back at the PC, use a separate checkout of the integrated source. Do not
replace the installed application's directory wholesale or copy a fresh empty
database over the existing one. `scripts/upgrade_developer.py` is a source-only,
reviewed-file updater for the known native-console checkpoint and the initial
Developer checkpoint. Unknown local code changes stop the upgrade. EXE-only or
other installed layouts require a matching upgrade package and are refused.

Stop the installed S app, then run from the new checkout (replace the path with
the actual installed S folder):

```powershell
# Review the exact file plan first; this command makes no changes.
py -3 scripts\upgrade_developer.py --app-root "C:\Path\To\Installed-S"
# After stopping S, apply the reviewed-file upgrade.
py -3 scripts\upgrade_developer.py --app-root "C:\Path\To\Installed-S" --apply --stopped
# Launch the installed copy's new Developer workspace:
& "C:\Path\To\Installed-S\scripts\S.cmd" develop application
```

The updater checks known SHA-256 hashes (allowing Git's LF/CRLF text conversion),
rejects symlinked target paths, creates file backups plus an integrity-checked
snapshot of `data/continuum.sqlite3` if present, then replaces only a code-owned
file allowlist. Native weights, host configuration, existing memory, unrelated
files and private data are not replaced. If a copy fails, changed files are
restored. A live listener on the selected S port blocks application. `--stopped`
is an explicit operator assertion: it cannot discover every custom S process or
nonstandard database path. Back up any separately configured databases manually.

The output includes the backup directory. Roll back code using:

```powershell
py -3 scripts\upgrade_developer.py --app-root "C:\Path\To\Installed-S" --rollback "C:\Path\To\Installed-S\upgrade-backups\s-developer-ID" --apply --stopped
```

Rollback checks that no post-upgrade code edits would be overwritten. It restores
old code, removes only unchanged newly installed files, and does not roll back
or delete the live database. The backup database remains available for a separate
operator-reviewed recovery; full database replacement is never automatic.

## Windows CI failure and correction

Initial Developer CI succeeded on Linux but failed on Windows in the occupied-port
regression test: a second server did not exit. This was a real platform bug,
not a reason to remove or skip the test. Python HTTPServer's address-reuse option
has different Windows semantics. AppServer now disables `SO_REUSEADDR` on Windows
and sets `SO_EXCLUSIVEADDRUSE` before binding. Unix restart behavior is retained.
The existing subprocess test is rerun on both Windows Python versions.

## Evidence boundary

Host tests exercise the Termux launcher's generated source path, version probe,
existing-launcher preservation and symlink refusal. Separate tests perform a real
upgrade of a temporary known-baseline install, verify database backup and config
preservation, exercise idempotency and rollback, inject a copy failure, and refuse
post-upgrade edits. These are not installation evidence for Shiyam's actual PC
or phone. Final CI links and commit evidence are recorded in PR #15.

Local continuation verification: **69** app/requirements/updater tests passed,
**17** combined UI state tests passed, and a freshly built wheel installed in a
clean virtual environment passed its outside-checkout launch/save/restart smoke.
The Termux-specific tests are host-side fixtures, not an Android execution claim.
Windows CI is the gate for the exclusive-port fix before integration.
