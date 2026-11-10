"""Dataset isolation, broad vocabulary and evaluation provenance checks."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, 'PyTorch not installed')
class DatasetTests(unittest.TestCase):
    def test_split_validation(self):
        from s_intelligence.train import load_dataset
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.json'
            base = {'provenance': 'synthetic test', 'train': ['a' * 40], 'validation': ['b' * 40]}
            for changes in ({'provenance': ''}, {'train': []}, {'validation': ['A' * 40]},
                            {'train': ['a' * 40, 'A' * 40]}, {'train': ['Ω' * 40]},
                            {'validation': ['short']}):
                path.write_text(json.dumps(dict(base, **changes)), encoding='utf8')
                with self.assertRaises(ValueError):
                    load_dataset(path)

    def test_ascii_checkpoint_and_independent_evaluation(self):
        from s_intelligence.train import run, load_dataset
        from s_intelligence.evaluate import evaluate
        from s_intelligence.infer import run as infer
        data = Path(__file__).resolve().parents[1] / 'data/training/s-int-002.json'
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), patch(
                'socket.socket.connect', side_effect=AssertionError('Network forbidden')):
            checkpoint = Path(directory) / 'model.pt'
            run(steps=2, output=checkpoint, seed=17, threads=1, data=data)
            state = torch.load(checkpoint, weights_only=True)
            self.assertEqual(len(state['vocabulary']), 100)
            self.assertEqual(infer(checkpoint, prompt='Q: 19 + 2 = ?', count=0), 'Q: 19 + 2 = ?')
            result = evaluate(checkpoint)
            self.assertEqual(result['evaluated_texts'], 12)
            self.assertEqual(result['unsupported_texts'], 0)
            self.assertEqual(result['unigram_additive_smoothing'], 1)
            self.assertTrue(torch.isfinite(torch.tensor(result['training_unigram_cross_entropy'])))
            leaked = Path(directory) / 'leak.json'
            leaked.write_text(json.dumps({'provenance': 'test', 'texts': [load_dataset(data)['train'][0].upper()]}))
            with self.assertRaises(ValueError):
                evaluate(checkpoint, leaked)
