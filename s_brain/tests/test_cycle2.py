"""Cycle 2 regression suite: exact KV cache parity, streaming, wire protocol, API.

These tests use very small randomly initialized weights. Passing them confirms
software behavior and shape/caching semantics, not language intelligence.
"""
import io
import json
import tempfile
import threading
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from s_brain.model import Config,SBrain
from s_brain.kv_cache import (
    KVState,LayerKV,append,attention_cached,cache_report,
    compare_full_and_cached,extend,memory_estimate,next_token_logits,
    prefill,rebuild_if_modified,rope_at,
)
from s_brain.streaming import (
    GenerationOptions,GenerationStats,StopMatcher,StreamEvent,
    UTF8Accumulator,collect_events,compare_stream_modes,
    jsonl_event,prepare_prompt,sse_event,stream_events,
)
from s_brain.sampling import SamplingOptions
from s_brain.server_v2 import (
    AdmissionControl,BusyError,RequestParameters,Runtime,ServerLimits,
    make_handler,
)
from s_brain.client import (
    ClientConfig,SBrainClient,StreamProtocolError,parse_ndjson,parse_sse,
)
from s_brain.benchmark import (
    BenchmarkPlan,benchmark,collect_distribution,compare_benchmarks,
    model_profile,
)


def small_model(context=16):
    torch.manual_seed(102)
    model = SBrain(Config(vocab_size=256,dim=32,layers=2,heads=4,kv_heads=2,context=context))
    model.eval()
    return model


class CacheParityTests(unittest.TestCase):
    def setUp(self):
        self.model = small_model(12)

    def test_prefill_matches_full(self):
        tokens = torch.randint(0,256,(1,7))
        with torch.no_grad():
            full,_ = self.model(tokens)
            cached,state = prefill(self.model,tokens)
        self.assertTrue(torch.allclose(full,cached,atol=1e-5,rtol=1e-5))
        self.assertEqual(state.positions,7)
        self.assertEqual(len(state.layers),2)

    def test_incremental_matches_reference(self):
        tokens = torch.randint(0,256,(1,11))
        comparison = compare_full_and_cached(self.model,tokens)
        self.assertTrue(comparison["match"],str(comparison))

    def test_rollover_matches_reference(self):
        tokens = torch.randint(0,256,(1,28))
        comparison = compare_full_and_cached(self.model,tokens)
        self.assertTrue(comparison["match"],str(comparison))

    def test_rope_zero_offset(self):
        from s_brain.model import rope
        x = torch.randn(1,4,5,8)
        self.assertTrue(torch.allclose(rope(x),rope_at(x,0)))

    def test_rope_offset_suffix(self):
        x = torch.randn(1,4,9,8)
        full = rope_at(x,0)
        sliced = rope_at(x[:,:,-3:,:],6)
        self.assertTrue(torch.allclose(full[:,:,-3:,:],sliced,atol=1e-6))

    def test_rollover_context_constant(self):
        tokens = torch.randint(0,256,(1,12))
        _,cache = prefill(self.model,tokens)
        next_token = torch.tensor([[7]])
        _,updated = append(self.model,next_token,cache)
        self.assertEqual(updated.positions,12)
        self.assertTrue(torch.equal(updated.tokens,torch.cat((tokens[:,1:],next_token),dim=1)))

    def test_cache_memory_calculation(self):
        _,state = prefill(self.model,torch.randint(0,256,(1,8)))
        estimate = memory_estimate(self.model.cfg,1,8,dtype_bytes=4)
        self.assertEqual(state.bytes,estimate)

    def test_cache_report(self):
        _,state = prefill(self.model,torch.randint(0,256,(1,8)))
        report = cache_report(state)
        self.assertEqual(report["cached_tokens"],8)
        self.assertGreater(report["bytes"],0)

    def test_cache_bad_length_rejected(self):
        _,state = prefill(self.model,torch.randint(0,256,(1,5)))
        state.positions = 9
        with self.assertRaises(ValueError):
            state.validate(2)

    def test_cache_bad_batch_rejected(self):
        _,state = prefill(self.model,torch.randint(0,256,(1,5)))
        with self.assertRaises(ValueError):
            append(self.model,torch.tensor([[1],[2]]),state)

    def test_cache_bad_append_shape(self):
        _,state = prefill(self.model,torch.randint(0,256,(1,5)))
        with self.assertRaises(ValueError):
            append(self.model,torch.tensor([[1,2]]),state)

    def test_cache_not_shared_by_reference(self):
        _,state = prefill(self.model,torch.randint(0,256,(1,5)))
        old_keys = state.layers[0].key.clone()
        _,future = append(self.model,torch.tensor([[7]]),state)
        self.assertTrue(torch.equal(old_keys,state.layers[0].key))
        self.assertNotEqual(state.positions,future.positions)

    def test_extend_equals_successive_append(self):
        tokens = torch.randint(0,256,(1,5))
        append_tokens = torch.randint(0,256,(1,3))
        _,state = prefill(self.model,tokens)
        fast,new_state = extend(self.model,append_tokens,state)
        reference,_ = self.model(torch.cat((tokens,append_tokens),dim=1))
        self.assertTrue(torch.allclose(fast[:,-1,:],reference[:,-1,:],atol=1e-5))
        self.assertEqual(new_state.positions,8)

    def test_prefill_rejects_context_overflow(self):
        with self.assertRaises(ValueError):
            prefill(self.model,torch.randint(0,256,(1,13)))

    def test_rebuild_if_modified(self):
        tokens = torch.randint(0,256,(1,4))
        _,state = prefill(self.model,tokens)
        self.assertIs(rebuild_if_modified(self.model,state,tokens),state)
        changed = tokens.clone()
        changed[0,0] += 1
        result = rebuild_if_modified(self.model,state,changed)
        self.assertIsNot(result,state)


class UTF8Tests(unittest.TestCase):
    def test_one_byte_character(self):
        decoder = UTF8Accumulator()
        self.assertEqual(decoder.push(65),"A")
        self.assertEqual(decoder.finish(),"")

    def test_multibyte_character(self):
        decoder = UTF8Accumulator()
        parts = [decoder.push(value) for value in "বাংলা".encode("utf-8")]
        self.assertEqual("".join(parts)+decoder.finish(),"বাংলা")

    def test_partial_sequence_not_emitted(self):
        decoder = UTF8Accumulator()
        self.assertEqual(decoder.push(0xE2),"")
        self.assertEqual(decoder.push(0x82),"")
        self.assertEqual(decoder.push(0xAC),"€")

    def test_truncated_sequence_replacement(self):
        decoder = UTF8Accumulator()
        decoder.push(0xE2)
        self.assertEqual(decoder.finish(),"�")

    def test_reject_nonbyte(self):
        with self.assertRaises(ValueError):
            UTF8Accumulator().push(256)

    def test_reject_feed_after_finish(self):
        decoder = UTF8Accumulator()
        decoder.finish()
        with self.assertRaises(RuntimeError):
            decoder.push(65)


class StopTests(unittest.TestCase):
    def test_stop_in_single_chunk(self):
        stop = StopMatcher(("END",))
        self.assertEqual(stop.push("abcENDtail"),"abc")
        self.assertEqual(stop.matched,"END")

    def test_stop_split_over_chunks(self):
        stop = StopMatcher(("END",))
        self.assertEqual(stop.push("hello E"),"hello ")
        self.assertEqual(stop.push("N"),"")
        self.assertEqual(stop.push("D"),"")
        self.assertEqual(stop.matched,"END")

    def test_stop_prefix_rolls_back(self):
        stop = StopMatcher(("END",))
        self.assertEqual(stop.push("E"),"")
        self.assertEqual(stop.push("X"),"EX")

    def test_partial_stop_flushed(self):
        stop = StopMatcher(("END",))
        stop.push("E")
        self.assertEqual(stop.finish(),"E")

    def test_two_stop_choices(self):
        stop = StopMatcher(("STOP","END"))
        self.assertEqual(stop.push("aENDbSTOP"),"a")
        self.assertEqual(stop.matched,"END")

    def test_empty_stop_invalid(self):
        with self.assertRaises(ValueError):
            StopMatcher(("",))

    def test_no_stop_unbuffered(self):
        stop = StopMatcher()
        self.assertEqual(stop.push("hello"),"hello")


class StreamTests(unittest.TestCase):
    def setUp(self):
        self.model = small_model(12)

    def test_stream_produces_terminal_event(self):
        events = list(stream_events(
            self.model,"S",GenerationOptions(
                sampling=SamplingOptions(max_new_tokens=3,seed=7)
            )
        ))
        self.assertEqual(events[0].type,"start")
        self.assertEqual(events[-1].type,"complete")
        self.assertEqual(events[-1].usage["generated_tokens"],3)

    def test_cached_uncached_parity(self):
        result = compare_stream_modes(self.model,"Hello",tokens=8,seed=10)
        self.assertTrue(result["match"],str(result))

    def test_cache_rollover_parity(self):
        result = compare_stream_modes(self.model,"A"*12,tokens=18,seed=10)
        self.assertTrue(result["match"],str(result))

    def test_collect_events(self):
        result = collect_events(stream_events(
            self.model,"A",GenerationOptions(
                sampling=SamplingOptions(max_new_tokens=3,seed=7)
            )
        ))
        self.assertIn("response",result)
        self.assertEqual(result["usage"]["generated_tokens"],3)

    def test_cancel_immediately(self):
        events = list(stream_events(
            self.model,"A",GenerationOptions(
                sampling=SamplingOptions(max_new_tokens=20)
            ),cancelled=lambda: True
        ))
        self.assertEqual(events[-1].finish_reason,"cancelled")
        self.assertEqual(events[-1].usage["generated_tokens"],0)

    def test_prompt_too_big(self):
        with self.assertRaises(ValueError):
            prepare_prompt("a"*100,12,10)

    def test_utf8_prompt_is_byte_counted(self):
        token_ids,bytes_used = prepare_prompt("é",12,10)
        self.assertEqual(token_ids,[195,169])
        self.assertEqual(bytes_used,2)

    def test_jsonl_wire_format(self):
        event = StreamEvent(type="delta",sequence=2,text="a\nb")
        payload = jsonl_event(event)
        self.assertEqual(json.loads(payload)["text"],"a\nb")

    def test_sse_wire_format(self):
        event = StreamEvent(type="delta",sequence=2,text="a\nb")
        payload = sse_event(event)
        self.assertIn(b"event: delta\n",payload)
        self.assertIn(b"data: ",payload)

    def test_usage_finish(self):
        stats = GenerationStats()
        stats.finish("length")
        self.assertEqual(stats.report()["finish_reason"],"length")


class RequestValidationTests(unittest.TestCase):
    def setUp(self):
        self.limits = ServerLimits()
        self.limits.validate()

    def test_simple_request(self):
        request = RequestParameters.parse({"prompt":"hi"},self.limits)
        self.assertEqual(request.prompt,"hi")

    def test_reject_extra_field(self):
        with self.assertRaises(ValueError):
            RequestParameters.parse({"prompt":"hi","unsafe":True},self.limits)

    def test_reject_boolean_integer(self):
        with self.assertRaises(ValueError):
            RequestParameters.parse({"prompt":"hi","top_k":True},self.limits)

    def test_reject_nonboolean_cache(self):
        with self.assertRaises(ValueError):
            RequestParameters.parse({"prompt":"hi","use_cache":"yes"},self.limits)

    def test_reject_large_prompt(self):
        with self.assertRaises(ValueError):
            RequestParameters.parse({"prompt":"a"*16001},self.limits)

    def test_reject_excess_tokens(self):
        with self.assertRaises(ValueError):
            RequestParameters.parse({"prompt":"hi","max_new_tokens":1000},self.limits)

    def test_reject_broken_stop(self):
        with self.assertRaises(ValueError):
            RequestParameters.parse({"prompt":"hi","stop":"ABC"},self.limits)

    def test_reject_public_bind(self):
        with self.assertRaises(ValueError):
            ServerLimits(host="0.0.0.0").validate()

    def test_admission_control(self):
        gate = AdmissionControl(1)
        gate.acquire()
        with self.assertRaises(BusyError):
            gate.acquire()
        self.assertEqual(gate.snapshot()["rejected"],1)
        gate.release()
        self.assertEqual(gate.snapshot()["active"],0)

    def test_invalid_capacity(self):
        with self.assertRaises(ValueError):
            AdmissionControl(0)


class ClientProtocolTests(unittest.TestCase):
    def test_sse_parser(self):
        lines = [
            b"event: start\n",b'data: {"type":"start","sequence":0}\n',
            b"\n",b"event: complete\n",
            b'data: {"type":"complete","sequence":1}\n',b"\n",
        ]
        self.assertEqual(len(list(parse_sse(iter(lines)))),2)

    def test_ndjson_parser(self):
        rows = [b'{"type":"delta","text":"hi"}\n']
        self.assertEqual(list(parse_ndjson(iter(rows)))[0]["text"],"hi")

    def test_sse_truncation_rejected(self):
        with self.assertRaises(StreamProtocolError):
            list(parse_sse(iter([b"data: {}\n"])))

    def test_invalid_json_rejected(self):
        with self.assertRaises(StreamProtocolError):
            list(parse_ndjson(iter([b"{error}\n"])))

    def test_nonlocal_client_rejected(self):
        with self.assertRaises(ValueError):
            ClientConfig(base_url="http://example.com:8766").validate()

    def test_local_client_allowed(self):
        ClientConfig(base_url="http://localhost:8766").validate()


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.model = small_model(12)

    def test_profile(self):
        profile = model_profile(self.model)
        self.assertGreater(profile["parameters"],0)
        self.assertEqual(profile["layers"],2)

    def test_distribution(self):
        result = collect_distribution([1.,2.,3.])
        self.assertEqual(result["median"],2.)
        self.assertEqual(result["p95"],3.)

    def test_empty_distribution(self):
        with self.assertRaises(ValueError):
            collect_distribution([])

    def test_invalid_plan(self):
        with self.assertRaises(ValueError):
            BenchmarkPlan(measurement_rounds=0).validate(12)

    def test_benchmark_smoke(self):
        result = benchmark(self.model,BenchmarkPlan(
            prompt_bytes=6,new_tokens=2,warmup_rounds=0,measurement_rounds=1
        ))
        self.assertTrue(result["logit_parity"]["match"])
        self.assertIn("cached",result["measurements"])

    def test_compare_benchmark_mismatch(self):
        report = benchmark(self.model,BenchmarkPlan(
            prompt_bytes=3,new_tokens=1,warmup_rounds=0,measurement_rounds=1,
            verify_parity=False
        ))
        altered = json.loads(json.dumps(report))
        altered["model"]["parameters"] += 1
        with self.assertRaises(ValueError):
            compare_benchmarks(report,altered)


if __name__=="__main__":
    unittest.main()
