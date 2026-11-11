"""S-Brain smoke/regression tests. Run with unittest after installing torch."""
import unittest
import torch
from s_brain.model import Config, SBrain

class TransformerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.cfg = Config(vocab_size=256,dim=64,layers=2,heads=4,kv_heads=2,context=16)
        self.model = SBrain(self.cfg)

    def test_forward_shape_and_finite_loss(self):
        tokens = torch.randint(0,256,(2,12))
        logits,loss = self.model(tokens,tokens)
        self.assertEqual(tuple(logits.shape),(2,12,256))
        self.assertTrue(torch.isfinite(loss))

    def test_training_changes_weights(self):
        x = torch.randint(0,256,(2,12))
        optim = torch.optim.AdamW(self.model.parameters(),lr=0.001)
        before = self.model.embed.weight.detach().clone()
        _,loss = self.model(x,x)
        loss.backward()
        optim.step()
        self.assertFalse(torch.equal(before,self.model.embed.weight))

    def test_causal_mask(self):
        self.model.eval()
        a = torch.tensor([[3,4,5,6,7]])
        b = torch.tensor([[3,4,5,99,100]])
        with torch.inference_mode():
            left = self.model(a)[0][:,:3]
            right = self.model(b)[0][:,:3]
        self.assertTrue(torch.allclose(left,right,atol=1e-5,rtol=1e-5))

    def test_generation_shape(self):
        result = self.model.generate(torch.tensor([[72,105]]),steps=4)
        self.assertEqual(tuple(result.shape),(1,6))
        self.assertTrue((result >= 0).all() and (result < 256).all())

    def test_context_limit(self):
        with self.assertRaises(ValueError):
            self.model(torch.zeros((1,17),dtype=torch.long))

if __name__ == "__main__":
    unittest.main()
