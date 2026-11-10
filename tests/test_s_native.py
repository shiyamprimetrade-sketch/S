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

if __name__ == '__main__': unittest.main()
