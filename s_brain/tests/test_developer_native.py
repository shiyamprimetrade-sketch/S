"""Actual bundled native weights execute; failed coding diagnostic is retained."""
from pathlib import Path
import tempfile
import unittest
import torch
from s_developer.models import install_demo,load_config,validate_native
from s_developer.providers import NativeProvider


class DeveloperNativeTests(unittest.TestCase):
    def test_real_neural_inference_not_a_template_fallback(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temporary:
            db=Path(temporary)/'s.sqlite3';installed=install_demo(db)
            checkpoint=Path(load_config(db)['native_checkpoint'])
            manifest=validate_native(checkpoint)
            self.assertEqual(manifest.step,256)
            self.assertTrue(manifest.extras['independently_initialized'])
            native=NativeProvider(checkpoint)
            self.assertEqual(native.provenance()['model_sha256'],installed['weights_sha256'])
            # The fixed demo has not passed its format gate. A failure must stay
            # a failure rather than returning a deterministic template silently.
            with self.assertRaises(ValueError):
                native.generate({'name':'Kepler','answers':{'purpose':'Print a greeting'}},'python-cli')
