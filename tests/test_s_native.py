"""Structural tests for the independent model; PyTorch required."""
import unittest
try:
    import torch
except ImportError:
    torch = None

@unittest.skipUnless(torch is not None, 'PyTorch not installed')
class NativeModelTests(unittest.TestCase):
    def test_tokenizer_roundtrip(self):
        from s_intelligence.model import CharTokenizer
        tok=CharTokenizer('S tests.')
        self.assertEqual(tok.decode(tok.encode('S tests.')),'S tests.')
    def test_causal_shape_and_backward(self):
        from s_intelligence.model import SModel
        model=SModel(12,width=32,context=8)
        x=torch.randint(0,12,(2,8))
        logits,loss=model(x,x)
        self.assertEqual(tuple(logits.shape),(2,8,12))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
    def test_future_token_cannot_change_past_logits(self):
        from s_intelligence.model import SModel
        torch.manual_seed(42)
        model=SModel(12,width=32,context=8).eval()
        x=torch.tensor([[1,2,3,4]])
        y=torch.tensor([[1,2,9,8]])
        with torch.no_grad():
            a,_=model(x)
            b,_=model(y)
        self.assertTrue(torch.allclose(a[:,:2],b[:,:2],atol=1e-5))



class NativeTrainingTests(unittest.TestCase):
    def test_checkpoint_reload_reproducibility_and_offline_inference(self):
        if torch is None:
            self.skipTest('PyTorch not installed')
        import contextlib
        import io
        import json
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        from s_intelligence.train import run
        from s_intelligence.infer import run as infer
        from s_intelligence.model import SModel

        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            first = Path(directory) / 'first.pt'
            second = Path(directory) / 'second.pt'
            with patch('socket.socket.connect', side_effect=AssertionError('Network forbidden')):
                metrics = run(steps=3, output=first, seed=17, threads=1)
                repeated = run(steps=3, output=second, seed=17, threads=1)
                text = infer(first, prompt='S ', count=12, seed=5)
                self.assertEqual(text, infer(first, prompt='S ', count=12, seed=5))
            self.assertEqual(len(text), 14)
            a = torch.load(first, map_location='cpu', weights_only=True)
            b = torch.load(second, map_location='cpu', weights_only=True)
            self.assertTrue(all(torch.equal(a['weights'][key], b['weights'][key]) for key in a['weights']))
            for metric in ('initial_training_loss', 'initial_validation_loss', 'final_training_loss', 'final_validation_loss'):
                self.assertTrue(torch.isfinite(torch.tensor(metrics[metric])))
                self.assertEqual(metrics[metric], repeated[metric])
            self.assertEqual(json.loads(first.with_suffix('.metrics.json').read_text())['seed'], 17)
            self.assertEqual(a['initialization'], 'random_from_scratch')
            self.assertGreater(metrics['checkpoint_bytes'], 0)
            model = SModel(len(a['vocabulary']), a['width'], a['context'])
            model.load_state_dict(a['weights'])
            tokens = torch.tensor([[0, 1, 2]])
            self.assertTrue(torch.isfinite(model(tokens)[0]).all())
            from s_intelligence.evaluate import run as evaluate_run
            evaluation = evaluate_run([first], Path(directory) / 'evaluation.json')['experiments'][0]
            self.assertGreater(evaluation['evaluated_texts'], 0)
            self.assertGreater(evaluation['unsupported_texts'], 0)
            self.assertTrue(torch.isfinite(torch.tensor(evaluation['trained_cross_entropy'])))
            row = next(r for r in evaluation['records'] if r['status'] == 'evaluated')
            model.eval()
            ids = torch.tensor([a['vocabulary'].index(char) for char in row['text']])
            logits, _ = model(ids[:-1].unsqueeze(0))
            expected_loss = float(torch.nn.functional.cross_entropy(logits[0], ids[1:]).detach())
            self.assertAlmostEqual(row['trained_loss'], expected_loss, places=5)
            with self.assertRaises(FileExistsError):
                run(steps=1, output=first)
            with self.assertRaises(ValueError):
                infer(first, prompt='unknown symbol Ω')
            with self.assertRaises(ValueError):
                infer(first, count=-1)

    def test_finite_nonzero_gradients(self):
        if torch is None:
            self.skipTest('PyTorch not installed')
        from s_intelligence.model import SModel
        torch.manual_seed(9)
        model = SModel(12, width=32, context=8)
        x = torch.randint(0, 12, (2, 8))
        logits, loss = model(x, x)
        self.assertTrue(torch.isfinite(logits).all())
        loss.backward()
        gradients = [p.grad for p in model.parameters()]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
        self.assertTrue(any(torch.count_nonzero(g) > 0 for g in gradients))
        with self.assertRaises(ValueError):
            model(torch.ones((1, 9), dtype=torch.long))

    def test_invalid_training_parameters_preserve_output(self):
        if torch is None:
            self.skipTest('PyTorch not installed')
        from s_intelligence.train import run
        for args in ({'steps': 0}, {'steps': -1}, {'seed': -1}, {'threads': 0}):
            with self.assertRaises(ValueError):
                run(**args)

if __name__ == '__main__':
    unittest.main()
