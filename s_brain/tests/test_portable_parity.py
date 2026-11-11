"""Differential tests against actual PyTorch using the same unchanged weights."""
from pathlib import Path
import unittest
import torch
from s_brain.checkpoint import load_weights
from s_developer.portable_native import PortableBrain

CHECKPOINT=Path(__file__).resolve().parents[2]/'s_developer'/'resources'/'native-demo'


class PortableParity(unittest.TestCase):
    def test_every_prefix_logits_match_actual_pytorch(self):
        torch.set_num_threads(2);model,_=load_weights(CHECKPOINT);portable=PortableBrain(CHECKPOINT)
        for text in ('A','Hello','Request: Hello\n','😀 S'):
            ids=list(text.encode());cache=portable.new_cache()
            for index,token in enumerate(ids):
                actual=portable.step(token,cache)
                with torch.inference_mode():expected=model(torch.tensor([ids[:index+1]]))[0][0,-1].tolist()
                self.assertLess(max(abs(a-b) for a,b in zip(actual,expected)),0.0003)
    def test_greedy_tokens_match_pytorch_with_context_cache(self):
        torch.set_num_threads(2);model,_=load_weights(CHECKPOINT);portable=PortableBrain(CHECKPOINT)
        ids=list(b'Request: Hello\n');generated=[]
        for _ in range(16):
            with torch.inference_mode():scores=model(torch.tensor([ids]))[0][0,-1]
            token=int(scores.argmax());generated.append(token);ids.append(token)
        self.assertEqual(portable.generate('Request: Hello\n',16)['generated_byte_ids'],generated)
