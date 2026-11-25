import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from test_termux_install import installer


@unittest.skipUnless(os.name == 'nt', 'Windows junction test')
class LauncherReparseTests(unittest.TestCase):
    def test_junction_install_directory_cannot_redirect_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            external = home / 'external'
            external.mkdir()
            sentinel = external / 'S'
            sentinel.write_text('preserve')
            local = home / '.local'
            local.mkdir()
            junction = local / 'bin'
            result = subprocess.run(['cmd.exe', '/d', '/c', 'mklink', '/J', str(junction), str(external)], capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            try:
                with self.assertRaises(ValueError):
                    installer.install(home, home, installer.SOURCE, sys.executable)
                self.assertEqual(sentinel.read_text(), 'preserve')
                self.assertEqual(list(external.iterdir()), [sentinel])
            finally:
                junction.rmdir()
