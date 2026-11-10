import tempfile,unittest
from pathlib import Path
from continuum.code_eval import unigram_loss, evaluate

class HeldoutTests(unittest.TestCase):
    def test_unigram_finite(self):
        self.assertGreater(unigram_loss(b'aaaa',b'ab'),0)
    def test_empty_rejected(self):
        with self.assertRaises(ValueError):unigram_loss(b'',b'a')
    def test_missing_checkpoint_does_not_fake_success(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for i in range(40):
                (root/f'example{i}.py').write_text('def function_'+str(i)+'():\n    return '+str(i)+'\n'+'#'*120)
            with self.assertRaises(FileNotFoundError):evaluate(root,root/'missing.pt')
    def test_external_loss_checkpoint(self):
        try:import torch
        except ImportError:self.skipTest('torch unavailable')
        from continuum.charllm import architecture
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'checkpoint.pt'
            model=architecture()(context=16,dim=16,heads=4,layers=1)
            torch.save({'model_type':'tiny-causal-byte-transformer-v1','config':{'context':16,'dim':16,'heads':4,'layers':1},'state_dict':model.state_dict()},path)
            from continuum.code_eval import model_loss
            result=model_loss(path,b'abcdef'*20,max_tokens=64)
            self.assertEqual(result['bytes_scored'],64)
            self.assertGreater(result['nll'],0)
if __name__=='__main__':unittest.main()
