"""Cycle 3 targeted edge-case tests and cross-module consistency contracts.

The purpose of these tests is to protect boundaries where a future integration
might introduce state reuse, unexpected training overlap, silent fallback,
GPU memory leaks or incorrect batch ordering.
"""
import json
import tempfile
import unittest
from pathlib import Path

import torch

from s_brain.model import Config,SBrain
from s_brain.batching import BatchRequest,batch_generate,group_by_length
from s_brain.cache_pool import CachePolicy,CachePool,key_for_prompt,validate_pool_invariants
from s_brain.evaluation_suite import (
    DataProvenance,EvaluationPolicy,contamination_pairs,
    duplicate_pairs,inspect_benchmark,paired_accuracy_delta,
    performance_gate,
)
from s_brain.evaluate import BenchmarkCase,run_cases
from s_brain.integration import IntegrationPolicy,OptInModelRouter,ExperimentalBridge
from s_brain.kv_cache import prefill


class BoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(2026)
        cls.net=SBrain(Config(
            dim=32,layers=1,heads=4,kv_heads=2,context=8
        ))
        cls.net.eval()

    def test_context_length_grouping(self):
        groups=group_by_length([
            BatchRequest("a","a"*8),BatchRequest("b","b"*100)
        ],8)
        self.assertEqual(len(groups),1)
        self.assertEqual(len(groups[8]),2)

    def test_batch_order_restoration(self):
        requests=[
            BatchRequest("first","a"*5,max_new_tokens=1,seed=1),
            BatchRequest("second","b"*2,max_new_tokens=1,seed=2),
            BatchRequest("third","c"*5,max_new_tokens=1,seed=3),
        ]
        result=batch_generate(self.net,requests)
        self.assertEqual([x.identifier for x in result],[
            "first","second","third"
        ])

    def test_overflow_generations_complete(self):
        result=batch_generate(self.net,[
            BatchRequest("a","abcdefgh",max_new_tokens=4,seed=22)
        ])
        self.assertEqual(result[0].generated_tokens,4)
        self.assertGreaterEqual(result[0].cache_rebuilds,3)

    def test_seed_changes_do_not_corrupt_others(self):
        first=batch_generate(self.net,[
            BatchRequest("a","hi",max_new_tokens=3,seed=100),
            BatchRequest("b","hi",max_new_tokens=3,seed=101),
        ])
        second=batch_generate(self.net,[
            BatchRequest("b","hi",max_new_tokens=3,seed=101),
            BatchRequest("a","hi",max_new_tokens=3,seed=100),
        ])
        a={x.identifier:x.response for x in first}
        b={x.identifier:x.response for x in second}
        self.assertEqual(a,b)

    def test_pool_replacement_consistency(self):
        source=torch.tensor([[1,2,3]],dtype=torch.long)
        _,state=prefill(self.net,source)
        pool=CachePool(CachePolicy(max_entries=2,max_bytes=100000))
        identity=key_for_prompt("same-model",source)
        pool.insert(identity,state)
        pool.insert(identity,state)
        validate_pool_invariants(pool)
        self.assertEqual(len(pool),1)

    def test_cache_wrong_model_never_matches(self):
        source=torch.tensor([[1,2,3]],dtype=torch.long)
        _,state=prefill(self.net,source)
        pool=CachePool(CachePolicy(max_entries=2,max_bytes=100000))
        pool.insert(key_for_prompt("a",source),state)
        self.assertIsNone(pool.get(key_for_prompt("b",source)))

    def test_cache_explicit_clear_frees_quota(self):
        source=torch.tensor([[1,2,3]],dtype=torch.long)
        _,state=prefill(self.net,source)
        pool=CachePool(CachePolicy(max_entries=2,max_bytes=100000))
        pool.insert(key_for_prompt("a",source),state)
        pool.clear()
        self.assertEqual(pool.statistics()["payload_bytes"],0)

    def test_invalid_policy_endpoint(self):
        with self.assertRaises(ValueError):
            IntegrationPolicy(endpoint="https://example.com/api").validate()

    def test_policy_unknown_fields_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"policy.json"
            path.write_text(json.dumps({"enabled":False,"unknown":1}))
            with self.assertRaises(ValueError):
                IntegrationPolicy.load(path)

    def test_disabled_router_never_routes(self):
        route=OptInModelRouter()
        self.assertEqual(route.known_models(),[])
        with self.assertRaises(PermissionError):
            route.dispatch("s-brain:experimental","test")

    def test_case_overlap_ids(self):
        cases=[
            BenchmarkCase("a","calculate one two three","x","math","human"),
            BenchmarkCase("b","calculate one two three","x","math","human"),
        ]
        self.assertEqual(duplicate_pairs(cases),[("a","b")])

    def test_contamination_threshold(self):
        cases=[BenchmarkCase("a","one two three four","x","math","human")]
        hits=contamination_pairs(cases,["one two three four"],threshold=1)
        self.assertEqual(len(hits),1)

    def test_gate_prevents_accuracy_collapse(self):
        baseline={"accuracy":.99,"p95_latency_ms":10,"failure_rate":0}
        candidate={"accuracy":.4,"p95_latency_ms":10,"failure_rate":0}
        self.assertFalse(performance_gate(baseline,candidate)["passed"])

    def test_gate_prevents_latency_regression(self):
        baseline={"accuracy":.9,"p95_latency_ms":10,"failure_rate":0}
        candidate={"accuracy":.9,"p95_latency_ms":50,"failure_rate":0}
        self.assertFalse(performance_gate(baseline,candidate)["passed"])

    def test_paired_evaluation_rejects_different_ids(self):
        a=BenchmarkCase("a","abc","x","test","human")
        b=BenchmarkCase("b","abc","x","test","human")
        before,_=run_cases([a],lambda _:"x")
        after,_=run_cases([b],lambda _:"x")
        with self.assertRaises(ValueError):
            paired_accuracy_delta(before,after)

    def test_provenance_requires_source(self):
        bad=DataProvenance("name","v1","","owned","blind","hash",True)
        with self.assertRaises(ValueError):
            bad.validate()


if __name__=="__main__":
    unittest.main()
