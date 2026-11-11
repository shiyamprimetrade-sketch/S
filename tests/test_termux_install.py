"""Host-side launcher tests; these do not emulate Android."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPT=Path(__file__).resolve().parents[1]/'scripts'/'install_termux.py'
spec=importlib.util.spec_from_file_location('termux_install',SCRIPT)
installer=importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class TermuxTests(unittest.TestCase):
    def test_android_browser_helper_receives_url_as_one_argument(self):
        from s_developer.cli import open_browser
        with tempfile.TemporaryDirectory() as temporary:
            helper=Path(temporary)/'bin'/'termux-open-url';helper.parent.mkdir();helper.write_text('fixture')
            with patch.dict(os.environ,{'PREFIX':temporary}), patch('s_developer.cli.subprocess.run') as run:
                self.assertTrue(open_browser('http://127.0.0.1:8765/developer#request=develop+application'))
                self.assertEqual(run.call_args.args[0],[str(helper),'http://127.0.0.1:8765/developer#request=develop+application'])
                self.assertFalse(run.call_args.kwargs['shell'])

    def test_launcher_runs_existing_source_without_packages_or_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            result=installer.install(root,root,installer.SOURCE,sys.executable)
            command=[sys.executable,result['launcher'],'toolchains','--detect','python']
            run=subprocess.run(command,capture_output=True,text=True,timeout=15)
            self.assertEqual(run.returncode,0,run.stderr)
            self.assertIn('"installed": true',run.stdout)
            self.assertFalse(result['model_installed'])
            self.assertFalse(result['phone_execution_verified'])
            self.assertFalse(Path(result['database']).exists())
            second=installer.install(root,root,installer.SOURCE,sys.executable)
            self.assertTrue(Path(second['previous_launcher']).is_file())

    def test_unmanaged_launcher_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            home=Path(temporary)
            target=home/'.local'/'bin'/'S';target.parent.mkdir(parents=True)
            target.write_text('existing user launcher')
            with self.assertRaises(FileExistsError):
                installer.install(home,home,installer.SOURCE,sys.executable)
            self.assertEqual(target.read_text(),'existing user launcher')

    @unittest.skipIf(os.name=='nt','Symlink creation can require elevated Windows privileges')
    def test_symlink_launcher_cannot_redirect_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            home=Path(temporary)
            target=home/'.local'/'bin'/'S';target.parent.mkdir(parents=True)
            external=home/'original';external.write_text('preserve')
            target.symlink_to(external)
            with self.assertRaises(ValueError):installer.install(home,home,installer.SOURCE,sys.executable)
            self.assertEqual(external.read_text(),'preserve')

    def test_non_termux_main_refuses_installation(self):
        with patch.dict(os.environ,{'PREFIX':''}):
            with self.assertRaisesRegex(SystemExit,'No files changed'):installer.main()
