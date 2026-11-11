"""Full-sequence supervision and exact unconstrained decoder behavior."""
import json
from pathlib import Path
from unittest.mock import patch
import unittest
import torch
from s_brain.coding_curriculum import batches, records, read_plan
from s_brain.coding_decode import greedy_json
from s_brain.model import Config, SBrain


class CurriculumNeuralTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)
    def test_targets_supervise_only_full_completion_and_reject_truncation(self):
        plan=read_plan(Path(__file__).resolve().parents[2]/'experiments/coding-curriculum-v1/plan.json')
        rows=records(plan)[:8]
        x,y=batches(rows,512)
        for i,row in enumerate(rows):
            n=len(row['prompt'].encode());completion=list(row['completion'].encode())
            self.assertTrue((y[i,:n-1]==-100).all())
            self.assertEqual(y[i,n-1:n-1+len(completion)].tolist(),completion)
            self.assertTrue((y[i,n-1+len(completion):]==-100).all())
            self.assertEqual(x[i,:n].tolist(),list(row['prompt'].encode()))
        with self.assertRaisesRegex(ValueError,'exceeds'):batches(rows,256)
    def test_cached_greedy_matches_full_model_greedy_without_json(self):
        torch.manual_seed(3);model=SBrain(Config(dim=32,layers=1,heads=4,kv_heads=2,context=64)).eval()
        prompt='abc';ids=torch.tensor([list(prompt.encode())]);expected=[]
        with torch.no_grad():
            for _ in range(12):
                logits,_=model(ids);token=int(logits[0,-1].argmax());expected.append(token)
                ids=torch.cat((ids,torch.tensor([[token]])),dim=1)
        self.assertEqual(greedy_json(model,prompt,12),bytes(expected).decode('utf-8',errors='replace'))
    def test_json_stop_is_lexical_not_a_source_repair_or_wrapper(self):
        model=SBrain(Config(dim=32,layers=1,heads=4,kv_heads=2,context=64))
        raw=json.dumps({'text':'brace } inside a string'})
        def logits(byte):
            value=torch.full((1,1,256),-1000.0);value[0,0,byte]=1;return value
        remaining=iter(raw.encode()[1:])
        with patch('s_brain.kv_cache.prefill',return_value=(logits(raw.encode()[0]),None)),patch(
            's_brain.kv_cache.append',side_effect=lambda *args:(logits(next(remaining)),None)) as append:
            self.assertEqual(greedy_json(model,'x',200),raw)
            self.assertEqual(append.call_count,len(raw.encode())-1)
        with self.assertRaisesRegex(ValueError,'context'):greedy_json(model,'x'*65,5)
        with self.assertRaises(ValueError):greedy_json(model,'x',0)


if __name__=='__main__':unittest.main()
