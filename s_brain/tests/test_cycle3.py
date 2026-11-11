"""Cycle 3 unit and integration tests: batched decoding, caches and evaluation."""
import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

import torch

from s_brain.model import Config,SBrain
from s_brain.batching import (
    BatchRequest,batch_generate,batch_summary,build_prompt_batch,
    compare_batch_to_single,group_by_length,naive_reference,
)
from s_brain.cache_pool import (
    CacheIdentity,CachePolicy,CachePool,cached_prefill,
    fingerprint_model,key_for_prompt,validate_pool_invariants,
)
from s_brain.evaluation_suite import (
    DataProvenance,EvaluationPolicy,bootstrap_paired_delta,
    contamination_pairs,duplicate_pairs,evaluate_with_provenance,
    fingerprint,inspect_benchmark,jaccard,normalize_text,
    paired_accuracy_delta,performance_gate,token_overlap,
    word_ngrams,
)
from s_brain.evaluate import BenchmarkCase,run_cases
from s_brain.integration import (
    ExperimentalBridge,IntegrationPolicy,OptInModelRouter,
    ReadinessAuditor,check_local_port_conflict,
    disabled_by_default_config,explain_integration_status,
)
from s_brain.kv_cache import prefill


def model(context=16):
    torch.manual_seed(101)
    result=SBrain(Config(vocab_size=256,dim=32,layers=1,heads=4,kv_heads=2,context=context))
    result.eval()
    return result


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.net=model(context=12)

    def test_group_equal_lengths(self):
        requests=[
            BatchRequest("a","ab"),BatchRequest("b","cd"),
            BatchRequest("c","longer"),
        ]
        groups=group_by_length(requests,12)
        self.assertEqual(len(groups[2]),2)
        self.assertEqual(len(groups[6]),1)

    def test_effective_lengths_clamp_at_context(self):
        groups=group_by_length([
            BatchRequest("a","a"*20),BatchRequest("b","b"*19)
        ],12)
        self.assertEqual(len(groups[12]),2)

    def test_unicode_byte_grouping(self):
        groups=group_by_length([
            BatchRequest("a","é"),BatchRequest("b","ab")
        ],12)
        self.assertEqual(len(groups[2]),2)

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            batch_generate(self.net,[BatchRequest("x","a"),BatchRequest("x","b")])

    def test_empty_batch(self):
        self.assertEqual(batch_generate(self.net,[]),[])

    def test_bad_max_batch(self):
        with self.assertRaises(ValueError):
            batch_generate(self.net,[BatchRequest("a","a")],max_batch_size=0)

    def test_invalid_prompt(self):
        with self.assertRaises(ValueError):
            BatchRequest("a","").validate(12)

    def test_invalid_identifier(self):
        with self.assertRaises(ValueError):
            BatchRequest("","abc").validate(12)

    def test_invalid_samples(self):
        with self.assertRaises(ValueError):
            BatchRequest("a","abc",max_new_tokens=0).validate(12)

    def test_equal_length_batch(self):
        batch=[
            BatchRequest("a","ab",max_new_tokens=3,seed=1),
            BatchRequest("b","cd",max_new_tokens=3,seed=2),
        ]
        result=batch_generate(self.net,batch)
        self.assertEqual(len(result),2)
        self.assertEqual([r.identifier for r in result],["a","b"])
        self.assertTrue(all(r.generated_tokens==3 for r in result))

    def test_mixed_length_batch(self):
        batch=[
            BatchRequest("a","a",max_new_tokens=2),
            BatchRequest("b","abcdef",max_new_tokens=2),
            BatchRequest("c","ab",max_new_tokens=2),
        ]
        output=batch_generate(self.net,batch)
        self.assertEqual([x.identifier for x in output],["a","b","c"])

    def test_batch_chunking(self):
        requests=[
            BatchRequest(str(i),"xy",max_new_tokens=2,seed=i)
            for i in range(5)
        ]
        results=batch_generate(self.net,requests,max_batch_size=2)
        self.assertEqual(len(results),5)

    def test_staggered_finish(self):
        requests=[
            BatchRequest("a","aa",max_new_tokens=1,seed=1),
            BatchRequest("b","bb",max_new_tokens=4,seed=2),
        ]
        results=batch_generate(self.net,requests)
        self.assertEqual([x.generated_tokens for x in results],[1,4])

    def test_context_rollover(self):
        requests=[
            BatchRequest("a","A"*12,max_new_tokens=5,seed=3),
            BatchRequest("b","B"*12,max_new_tokens=5,seed=4),
        ]
        results=batch_generate(self.net,requests)
        self.assertTrue(all(item.cache_rebuilds>0 for item in results))

    def test_batch_parity(self):
        requests=[
            BatchRequest("a","AA",max_new_tokens=3,seed=11),
            BatchRequest("b","BB",max_new_tokens=4,seed=12),
        ]
        report=compare_batch_to_single(self.net,requests)
        self.assertTrue(report["match"],str(report))

    def test_ragged_batch_rejected(self):
        rows=[
            (0,BatchRequest("a","a"),b"a"),
            (1,BatchRequest("b","ab"),b"ab"),
        ]
        with self.assertRaises(ValueError):
            build_prompt_batch(rows,"cpu")

    def test_summary(self):
        results=batch_generate(self.net,[
            BatchRequest("a","a",max_new_tokens=2),
            BatchRequest("b","b",max_new_tokens=3)
        ])
        stats=batch_summary(results)
        self.assertEqual(stats["requests"],2)
        self.assertEqual(stats["tokens"],5)


class MemoryPoolTests(unittest.TestCase):
    def setUp(self):
        self.net=model()
        self.prompt=torch.tensor([[1,2,3,4]],dtype=torch.long)
        self.clock=[0.0]
        self.pool=CachePool(CachePolicy(max_entries=2,max_bytes=100000,ttl_seconds=10),
                            clock=lambda:self.clock[0])
        self.fingerprint="test-model"

    def key_and_state(self,prompt):
        _,state=prefill(self.net,prompt)
        return key_for_prompt(self.fingerprint,prompt),state

    def test_insert_and_get(self):
        key,state=self.key_and_state(self.prompt)
        self.assertTrue(self.pool.insert(key,state))
        self.assertIsNotNone(self.pool.get(key,self.prompt))
        self.assertEqual(self.pool.statistics()["hits"],1)

    def test_cache_key_stable(self):
        a=key_for_prompt("m",self.prompt)
        b=key_for_prompt("m",self.prompt.clone())
        self.assertEqual(a,b)

    def test_cache_key_model_sensitive(self):
        a=key_for_prompt("m",self.prompt)
        b=key_for_prompt("other",self.prompt)
        self.assertNotEqual(a,b)

    def test_cache_key_token_sensitive(self):
        a=key_for_prompt("m",self.prompt)
        b=key_for_prompt("m",self.prompt+1)
        self.assertNotEqual(a,b)

    def test_cache_state_mismatch_rejected(self):
        key,state=self.key_and_state(self.prompt)
        with self.assertRaises(ValueError):
            self.pool.insert(key_for_prompt("m",self.prompt+1),state)

    def test_cache_ttl_expiry(self):
        key,state=self.key_and_state(self.prompt)
        self.pool.insert(key,state)
        self.clock[0]=11.0
        self.assertIsNone(self.pool.get(key))
        self.assertEqual(len(self.pool),0)

    def test_cache_lru(self):
        for i in range(2):
            p=torch.tensor([[i+1,i+2]],dtype=torch.long)
            k,s=self.key_and_state(p)
            self.pool.insert(k,s)
        key_first=key_for_prompt(self.fingerprint,torch.tensor([[1,2]]))
        self.pool.get(key_first)
        p=torch.tensor([[8,9]])
        k,s=self.key_and_state(p)
        self.pool.insert(k,s)
        self.assertIsNotNone(self.pool.get(key_first))
        self.assertLessEqual(len(self.pool),2)

    def test_cache_miss_counter(self):
        self.assertIsNone(self.pool.get(key_for_prompt("m",self.prompt)))
        self.assertEqual(self.pool.statistics()["misses"],1)

    def test_clear(self):
        key,state=self.key_and_state(self.prompt)
        self.pool.insert(key,state)
        self.pool.clear()
        self.assertEqual(len(self.pool),0)

    def test_evict(self):
        key,state=self.key_and_state(self.prompt)
        self.pool.insert(key,state)
        self.assertTrue(self.pool.evict(key))
        self.assertFalse(self.pool.evict(key))

    def test_bytes_accounting(self):
        key,state=self.key_and_state(self.prompt)
        self.pool.insert(key,state)
        validate_pool_invariants(self.pool)
        self.assertGreater(self.pool.statistics()["payload_bytes"],0)

    def test_oversized_entry_refused(self):
        tiny=CachePool(CachePolicy(max_entries=1,max_bytes=1))
        key,state=self.key_and_state(self.prompt)
        self.assertFalse(tiny.insert(key,state))

    def test_prefill_adapter_matches_direct(self):
        _,direct=prefill(self.net,self.prompt)
        _,cached,hit=cached_prefill(self.net,self.prompt,self.pool,self.fingerprint)
        self.assertFalse(hit)
        self.assertEqual(direct.positions,cached.positions)
        _,cached,hit=cached_prefill(self.net,self.prompt,self.pool,self.fingerprint)
        self.assertTrue(hit)

    def test_model_fingerprint(self):
        first=fingerprint_model(self.net)
        second=fingerprint_model(self.net)
        self.assertEqual(first,second)

    def test_policy_validation(self):
        with self.assertRaises(ValueError):
            CachePolicy(max_entries=0).validate()


class IntegrityTests(unittest.TestCase):
    def cases(self):
        return [
            BenchmarkCase(str(i),f"Question {i}?","answer", "math" if i%2 else "code","human")
            for i in range(30)
        ]

    def test_fingerprint(self):
        self.assertEqual(fingerprint("hello"),fingerprint("hello"))

    def test_word_normalization(self):
        self.assertEqual(normalize_text("Hello, WORLD!"),"hello world")

    def test_ngrams(self):
        self.assertIn(("one","two"),word_ngrams("one two three",2))

    def test_jaccard(self):
        self.assertAlmostEqual(jaccard({1,2},{2,3}),1/3)

    def test_overlap(self):
        self.assertAlmostEqual(token_overlap("one two three","one two three"),1)

    def test_duplicate_prompt_detection(self):
        cases=self.cases()
        cases.append(BenchmarkCase("x","Question 0?","answer","code","human"))
        self.assertEqual(len(duplicate_pairs(cases)),1)

    def test_contamination_detected(self):
        cases=self.cases()
        matching=contamination_pairs(cases,["Question 0?"])
        self.assertTrue(any(x["case_id"]=="0" for x in matching))

    def test_inspection_good(self):
        result=inspect_benchmark(self.cases(),EvaluationPolicy())
        self.assertTrue(result["accepted"],str(result))

    def test_inspection_short(self):
        result=inspect_benchmark(self.cases()[:2],EvaluationPolicy())
        self.assertFalse(result["accepted"])

    def test_inspection_contaminated(self):
        result=inspect_benchmark(self.cases(),EvaluationPolicy(),["Question 0?"])
        self.assertFalse(result["accepted"])

    def test_provenance_requires_independence(self):
        bad=DataProvenance("dataset","v1","author","owned","blind","digest",False)
        with self.assertRaises(ValueError):
            bad.validate()

    def test_scoring_report(self):
        provenance=DataProvenance("qa","v1","author","owned","human","sha",True)
        report=evaluate_with_provenance(
            self.cases(),lambda _: "answer",provenance
        )
        self.assertEqual(report["summary"]["accuracy"],1)

    def test_paired_accuracy(self):
        cases=self.cases()
        baseline,_=run_cases(cases,lambda _: "wrong")
        improved,_=run_cases(cases,lambda _: "answer")
        report=paired_accuracy_delta(baseline,improved)
        self.assertEqual(report["wins"],30)
        self.assertEqual(report["accuracy_delta"],1)

    def test_bootstrap_paired(self):
        cases=self.cases()
        before,_=run_cases(cases,lambda _: "wrong")
        after,_=run_cases(cases,lambda _: "answer")
        report=bootstrap_paired_delta(before,after,iterations=100)
        self.assertEqual(report["lower"],1)

    def test_regression_gate(self):
        before={"accuracy":.8,"p95_latency_ms":100,"failure_rate":0}
        after={"accuracy":.81,"p95_latency_ms":105,"failure_rate":0}
        self.assertTrue(performance_gate(before,after)["passed"])


class FakeClient:
    def __init__(self,*args,**kwargs):
        self.calls=[]

    def health(self):
        return {"ready":True,"independent":True,"context":128}

    def generate(self,prompt,**kwargs):
        self.calls.append(prompt)
        return {"response":"fixture","usage":{"generated_tokens":1}}

    def stream(self,prompt,**kwargs):
        return iter([{"type":"delta","text":"fixture"}])


class IntegrationTests(unittest.TestCase):
    def test_off_by_default(self):
        self.assertFalse(IntegrationPolicy().enabled)

    def test_fallback_disallowed(self):
        with self.assertRaises(ValueError):
            IntegrationPolicy(allow_automatic_fallback=True).validate()

    def test_multiturn_disallowed(self):
        with self.assertRaises(ValueError):
            IntegrationPolicy(allow_multi_turn=True).validate()

    def test_policy_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"policy.json"
            IntegrationPolicy().save(path)
            self.assertFalse(IntegrationPolicy.load(path).enabled)

    def test_disabled_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"policy.json"
            result=disabled_by_default_config(path)
            self.assertFalse(result["enabled"])
            self.assertTrue(path.exists())

    def test_no_network_when_disabled(self):
        policy=IntegrationPolicy()
        report=ReadinessAuditor(policy,client_factory=lambda *_:self.fail("called")).check()
        self.assertFalse(report.ready)

    def test_readiness_enabled(self):
        report=ReadinessAuditor(IntegrationPolicy(enabled=True),FakeClient).check()
        self.assertTrue(report.ready,str(report.as_dict()))

    def test_bridge_disabled(self):
        with self.assertRaises(PermissionError):
            ExperimentalBridge(IntegrationPolicy())

    def test_bridge_enabled(self):
        bridge=ExperimentalBridge(IntegrationPolicy(enabled=True),FakeClient())
        output=bridge.generate("Hello")
        self.assertEqual(output["content"],"fixture")
        self.assertTrue(output["experimental"])

    def test_bridge_reject_history(self):
        bridge=ExperimentalBridge(IntegrationPolicy(enabled=True),FakeClient())
        with self.assertRaises(ValueError):
            bridge.generate("hi",[{"role":"user","content":"old"}])

    def test_router_does_not_fallback(self):
        router=OptInModelRouter()
        with self.assertRaises(PermissionError):
            router.dispatch("s-brain:experimental","hi")
        with self.assertRaises(KeyError):
            router.dispatch("qwen2.5:1.5b","hi")

    def test_local_port_separation(self):
        self.assertTrue(check_local_port_conflict())
        self.assertFalse(check_local_port_conflict(8765,8765))

    def test_status_is_not_production_claim(self):
        status=explain_integration_status(IntegrationPolicy())
        self.assertFalse(status["ready_for_production"])


if __name__=="__main__":
    unittest.main()
