import tempfile, unittest
from pathlib import Path
from project_s.verification import verify

class TrustedConsumerTests(unittest.TestCase):
    def test_modified_python_never_executes(self):
        with tempfile.TemporaryDirectory() as root:
            for capability in ('calculator','inventory'):
                p=Path(root)/(capability+'.py')
                marker=Path(root)/'executed'
                p.write_text("from pathlib import Path\nPath("+repr(str(marker))+").touch()\n")
                with self.assertRaisesRegex(ValueError,'Untrusted generated Python'):
                    verify(p,capability)
                self.assertFalse(marker.exists())
