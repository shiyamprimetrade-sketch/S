"""Learned attention/gate invariants; fixtures are not capability evidence."""
import unittest
import torch
from s_brain.byte_copy import ByteCopy,generate_copy
from s_brain.coding_decode import greedy_json
from s_brain.model import Config,SBrain
from s_brain.coding_curriculum import records,read_plan,batches
from scripts.train_byte_copy import supervision
from pathlib import Path


class ByteCopyTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(3)
    def test_disabled_gate_matches_unchanged_base_and_hooks_are_removed(self):
        base=SBrain(Config(dim=32,layers=1,heads=4,kv_heads=2,context=64)).eval();adapter=ByteCopy(32,16)
        expected=greedy_json(base,'abc',12)
        count=len(base.norm._forward_hooks)
        self.assertEqual(generate_copy(base,adapter,'abc',12,gate_override=0),expected)
        self.assertEqual(len(base.norm._forward_hooks),count)
        with self.assertRaises(ValueError):generate_copy(base,adapter,'x'*65,12)
        with self.assertRaises(ValueError):generate_copy(base,adapter,'x',0)
    def test_copy_distribution_uses_source_bytes_and_mask_not_a_literal_parser(self):
        adapter=ByteCopy(4,2)
        with torch.no_grad():adapter.query.weight.zero_();adapter.key.weight.zero_()
        hidden=torch.zeros(1,1,4);logits=torch.zeros(1,1,256);source=torch.zeros(1,3,4)
        p,_,_=adapter.distribution(hidden,logits,source,torch.tensor([[65,66,67]]),torch.tensor([[True,False,False]]),1)
        self.assertEqual(int(p[0,0].argmax()),65);self.assertAlmostEqual(float(p.detach().sum()),1)
        p,_,_=adapter.distribution(hidden,logits,source,torch.tensor([[88,66,67]]),torch.tensor([[True,False,False]]),1)
        self.assertEqual(int(p[0,0].argmax()),88)
    def test_probabilities_normalize_and_gradients_reach_only_adapter(self):
        adapter=ByteCopy(4,2);h=torch.randn(2,3,4);source=torch.randn(2,5,4)
        p,s,g=adapter.distribution(h,torch.randn(2,3,256),source,torch.randint(0,256,(2,5)),torch.ones(2,5,dtype=torch.bool))
        self.assertTrue(torch.allclose(p.sum(-1),torch.ones(2,3),atol=1e-6))
        (-p[:,:,65].clamp_min(1e-12).log().mean()+s.square().mean()).backward()
        self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() for v in adapter.parameters()))
    def test_supervision_is_training_only_with_exact_source_positions(self):
        plan=read_plan(Path(__file__).resolve().parents[2]/'experiments/learned-byte-copy-v1/plan.json')
        row=next(r for r in records(plan) if r['split']=='train')
        x,y=batches([row],512);gate,pointer=supervision([row],y);copy=pointer!=-100
        self.assertEqual(int(copy.sum()),3)
        self.assertEqual(bytes(x[0,pointer[copy]].tolist()).decode(),str(row['constant']))
        self.assertEqual(bytes(y[copy].tolist()).decode(),str(row['constant']))
        self.assertEqual(int(gate.sum()),3)


class SpanCopyTests(unittest.TestCase):
    def test_random_gate_preserves_base_and_invalid_budgets_refuse(self):
        from s_brain.span_copy import SpanCopy,generate_span
        torch.set_num_threads(2);torch.manual_seed(3)
        base=SBrain(Config(dim=32,layers=1,heads=4,kv_heads=2,context=64));adapter=SpanCopy(32,16)
        self.assertEqual(generate_span(base,adapter,'abc',12),greedy_json(base,'abc',12))
        self.assertEqual(len(base.norm._forward_hooks),0)
        with self.assertRaises(ValueError):generate_span(base,adapter,'a'*65,12)
        with self.assertRaises(ValueError):generate_span(base,adapter,'abc',False)
    def test_span_selection_is_learned_unmasked_source_positions(self):
        from s_brain.span_copy import SpanCopy
        adapter=SpanCopy(4,8);source=torch.randn(1,4,4)
        starts,ends=adapter.positions(source,torch.tensor([[True,True,False,False]]))
        self.assertTrue(torch.isneginf(starts[0,2:]).all());self.assertTrue(torch.isneginf(ends[0,2:]).all())
        (-starts[0,:2].log_softmax(0)[0]-ends[0,:2].log_softmax(0)[1]).backward()
        self.assertTrue(all(p.grad is not None for p in adapter.start.parameters()))
        self.assertTrue(all(p.grad is not None for p in adapter.end.parameters()))


    def test_invalid_learned_span_stops_without_fallback_and_removes_hook(self):
        from unittest.mock import patch
        from s_brain.span_copy import SpanCopy,generate_span
        base=SBrain(Config(dim=32,layers=1,heads=4,kv_heads=2,context=64));adapter=SpanCopy(32,16)
        with torch.no_grad():adapter.gate[-1].bias.fill_(100)
        with patch.object(adapter,'positions',return_value=(torch.tensor([[0.,0.,10.]]),torch.tensor([[10.,0.,0.]]))):
            with self.assertRaisesRegex(ValueError,'span outside'):generate_span(base,adapter,'abc',12)
        self.assertEqual(len(base.norm._forward_hooks),0)


if __name__=='__main__':unittest.main()
