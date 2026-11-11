"""Cycle 10: opt-in assistant integration and local HTTP security tests.

The default S endpoints remain unchanged. These tests exercise request
validation, memory scope, source snapshotting, authorization, tool limits,
read-only guarantees, and localhost routing. The generated response fixture
is clearly synthetic; it does not establish model intelligence.
"""
from __future__ import annotations

import dataclasses
import http.client
import json
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

import torch

from s_brain.assist_runtime import (
    AssistLimits,AssistRequest,AssistConfiguration,AssistAudit,
    AssistantEngine,AssistantDisabled,AssistantRequestError,
    request_hash,make_tool_controller,parse_read_only_intents,
    run_read_only_tools,evidence_snapshot,append_tool_evidence,
    response_envelope,
)
from s_brain.memory_store import MemoryStore
from s_brain.server_v2 import Runtime,ServerLimits,make_handler
from s_brain.tools_control import ToolPermissionError
from s_brain.rag_inference import ANSWER_HEADING


class DummyConfig:
    context=4096


class DummyModel:
    cfg=DummyConfig()
    def parameters(self):
        return iter((torch.nn.Parameter(torch.ones(1)),))


def controlled_generator(model,prompt,*,max_new_tokens=32,seed=7):
    # A fixture generator, NOT an S model learning benchmark.
    result="The research text is unknown."
    if "Gradient descent updates neural weights" in prompt:
        result="Gradient descent updates neural weights."
    return {"response":result,"model":"fixture-only",
            "usage":{"generated_tokens":4,"seed":seed}}


def wrong_citation_generator(model,prompt,*,max_new_tokens=32,seed=7):
    return {"response":"Gradient descent works [mem:11111111-1111-1111-1111-111111111111].",
            "model":"fixture-only","usage":{}}


def echo_first_marker(model,prompt,*,max_new_tokens=32,seed=7):
    import re
    found=re.search(r"\[mem:[a-f0-9-]{36}\]",prompt)
    citation=found.group(0) if found else ""
    return {"response":"Gradient descent updates neural weights "+citation+".",
            "model":"fixture-only","usage":{}}


class AssistedRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.store=MemoryStore(Path(self.tmp.name)/"owned.sqlite3")
        self.record=self.store.remember(
            scope="S",
            content="Gradient descent updates neural weights.",
            source="user-owned experimental note",
            consent="explicit",
            ttl_seconds=3600,
        )
        self.model=DummyModel()
        self.configuration=AssistConfiguration(
            enabled=True,memory_enabled=True,tools_enabled=True,
        )
        self.engine=AssistantEngine(
            self.model,configuration=self.configuration,
            memory=self.store,generator=controlled_generator,
        )

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def request(self,**kwargs):
        values={"question":"gradient descent","scope":"S","memory":True}
        values.update(kwargs)
        return AssistRequest(**values)

    def test_defaults_are_disabled(self):
        settings=AssistConfiguration()
        self.assertFalse(settings.enabled)
        self.assertFalse(settings.memory_enabled)
        self.assertFalse(settings.tools_enabled)

    def test_runtime_requires_explicit_assistant_flag(self):
        with self.assertRaises(ValueError):
            AssistConfiguration(memory_enabled=True).validate()

    def test_runtime_requires_assistant_for_tools(self):
        with self.assertRaises(ValueError):
            AssistConfiguration(tools_enabled=True).validate()

    def test_runtime_forbids_mutating_tool_names(self):
        with self.assertRaises(ValueError):
            AssistConfiguration(
                enabled=True,tools_enabled=True,
                read_only_tools=("memory_remember",),
            ).validate()

    def test_runtime_forbids_shell_names(self):
        with self.assertRaises(ValueError):
            AssistConfiguration(
                enabled=True,tools_enabled=True,
                read_only_tools=("run_shell",),
            ).validate()

    def test_limits_validate(self):
        AssistLimits().validate()

    def test_limits_zero_invalid(self):
        with self.assertRaises(ValueError):
            AssistLimits(max_new_tokens=0).validate()

    def test_limits_query_over_request_invalid(self):
        with self.assertRaises(ValueError):
            AssistLimits(max_query_bytes=20000).validate()

    def test_parse_minimum_request(self):
        parsed=AssistRequest.parse(
            {"question":"What is S?"},AssistLimits()
        )
        self.assertFalse(parsed.memory)
        self.assertEqual(parsed.scope,"S")

    def test_parse_rejects_unknown_fields(self):
        with self.assertRaises(AssistantRequestError):
            AssistRequest.parse(
                {"question":"x","command":"shell"},AssistLimits()
            )

    def test_parse_rejects_empty_question(self):
        with self.assertRaises(AssistantRequestError):
            AssistRequest.parse({"question":""},AssistLimits())

    def test_parse_rejects_bad_scope(self):
        with self.assertRaises(AssistantRequestError):
            AssistRequest.parse(
                {"question":"hello","scope":"../Prime"},AssistLimits()
            )

    def test_parse_rejects_boolean_tokens(self):
        with self.assertRaises(AssistantRequestError):
            AssistRequest.parse(
                {"question":"x","max_new_tokens":True},AssistLimits()
            )

    def test_parse_rejects_boolean_seed(self):
        with self.assertRaises(AssistantRequestError):
            AssistRequest.parse(
                {"question":"x","seed":True},AssistLimits()
            )

    def test_parse_rejects_large_query(self):
        with self.assertRaises(AssistantRequestError):
            AssistRequest.parse(
                {"question":"q"*5000},AssistLimits()
            )

    def test_parse_rejects_excess_tools(self):
        with self.assertRaises(AssistantRequestError):
            AssistRequest.parse(
                {"question":"x","tool_intents":[
                    {"id":str(x),"name":"calculator","arguments":{}}
                    for x in range(4)
                ]},AssistLimits()
            )

    def test_parse_rejects_bad_tool_json(self):
        with self.assertRaises(AssistantRequestError):
            AssistRequest.parse(
                {"question":"x","tool_intents":"not a list"},AssistLimits()
            )

    def test_parse_rejects_unknown_tool_fields(self):
        with self.assertRaises(AssistantRequestError):
            AssistRequest.parse(
                {"question":"x","tool_intents":[{
                    "id":"1","name":"calculator","arguments":{},
                    "extra":"not allowed"
                }]},AssistLimits()
            )

    def test_disabled_engine_rejects(self):
        engine=AssistantEngine(self.model)
        with self.assertRaises(AssistantDisabled):
            engine.respond(self.request(memory=False))

    def test_scope_outside_s_rejected(self):
        with self.assertRaises(PermissionError):
            self.engine.respond(self.request(scope="Prime"))

    def test_memory_requires_server_enablement(self):
        engine=AssistantEngine(
            self.model,configuration=AssistConfiguration(enabled=True),
            generator=controlled_generator,
        )
        with self.assertRaises(AssistantDisabled):
            engine.respond(self.request())

    def test_memory_requires_db(self):
        engine=AssistantEngine(
            self.model,
            configuration=AssistConfiguration(enabled=True,memory_enabled=True),
            generator=controlled_generator,
        )
        with self.assertRaises(AssistantDisabled):
            engine.respond(self.request())

    def test_health_shows_feature_flags(self):
        report=self.engine.health()
        self.assertTrue(report["assistant_enabled"])
        self.assertTrue(report["memory_enabled"])
        self.assertFalse(report["auto_capture_enabled"])

    def test_request_hash_deterministic(self):
        self.assertEqual(request_hash(self.request()),request_hash(self.request()))

    def test_request_hash_changes_with_mode(self):
        self.assertNotEqual(
            request_hash(self.request(memory=False)),
            request_hash(self.request(memory=True)),
        )

    def test_independent_generation_uses_memory(self):
        payload=self.engine.respond(self.request())
        self.assertEqual(payload["memory_mode"],"memory")
        self.assertIn("Gradient descent",payload["response"])
        self.assertTrue(payload["independent_inference"])
        self.assertTrue(payload["no_ollama_fallback"])

    def test_no_memory_mode(self):
        payload=self.engine.respond(self.request(memory=False))
        self.assertEqual(payload["memory_mode"],"no-memory")
        self.assertEqual(payload["audit"]["memory_excerpts"],0)

    def test_memory_does_not_auto_write(self):
        before=self.store.count("S")
        self.engine.respond(self.request())
        self.assertEqual(self.store.count("S"),before)
        self.assertFalse(self.engine.respond(self.request())["live_memory_write_allowed"])

    def test_citation_is_unverified_by_default(self):
        payload=self.engine.respond(self.request())
        self.assertFalse(payload["source_verification_passed"])
        self.assertEqual(payload["verified_sources"],[])

    def test_correct_source_citation_passes_heuristic(self):
        engine=AssistantEngine(
            self.model,configuration=self.configuration,
            memory=self.store,generator=echo_first_marker,
        )
        payload=engine.respond(self.request(cite_sources=True))
        self.assertTrue(payload["source_verification_passed"],payload)
        self.assertEqual(len(payload["verified_sources"]),1)

    def test_fabricated_citation_fails_closed(self):
        engine=AssistantEngine(
            self.model,configuration=self.configuration,
            memory=self.store,generator=wrong_citation_generator,
        )
        payload=engine.respond(self.request(cite_sources=True))
        self.assertFalse(payload["source_verification_passed"])
        self.assertEqual(payload["rejection_reason"],
                         "source_or_claim_verification_failed")
        self.assertIn("Insufficient",payload["response"])

    def test_citation_request_without_relevant_source_denied(self):
        with self.assertRaises(RuntimeError):
            self.engine.respond(
                self.request(question="unrelated lunar sky",cite_sources=True)
            )

    def test_deleted_memory_never_retrieved(self):
        self.store.forget("S",self.record.id)
        payload=self.engine.respond(self.request())
        self.assertEqual(payload["audit"]["memory_excerpts"],0)

    def test_snapshot_is_source_attributed(self):
        snapshot=evidence_snapshot(
            self.store,self.request(),self.configuration,context=4096
        )
        self.assertEqual(len(snapshot.included),1)
        self.assertIn(self.record.content_sha256,snapshot.prompt)

    def test_snapshot_cannot_cross_project(self):
        with self.assertRaises(PermissionError):
            evidence_snapshot(
                self.store,self.request(scope="Prime"),
                self.configuration,context=4096,
            )

    def test_prompt_context_limit_fails_closed(self):
        with self.assertRaises(RuntimeError):
            evidence_snapshot(
                self.store,self.request(),self.configuration,context=16
            )

    def test_no_tools_requested_executes_none(self):
        payload=self.engine.respond(self.request())
        self.assertEqual(payload["audit"]["tools_requested"],0)
        self.assertEqual(payload["tool_trace"]["execution_audits"],[])

    def test_calculator_is_available_if_enabled(self):
        request=self.request(memory=False,tool_intents=({
            "id":"calc","name":"calculator",
            "arguments":{"operation":"multiply","left":"6","right":"7"},
        },))
        result=self.engine.respond(request)
        self.assertEqual(result["audit"]["tools_executed"],1)
        self.assertEqual(result["audit"]["tools_denied"],0)
        self.assertTrue(result["tool_trace"]["raw_results_excluded"])
        self.assertNotIn("result",result["tool_trace"]["execution_audits"][0])
        self.assertNotIn("output",result["tool_trace"]["execution_audits"][0])

    def test_shell_tool_denied_without_execution(self):
        request=self.request(memory=False,tool_intents=({
            "id":"shell","name":"run_shell",
            "arguments":{"command":"whoami"},
        },))
        payload=self.engine.respond(request)
        self.assertEqual(payload["audit"]["tools_executed"],0)
        self.assertEqual(payload["audit"]["tools_denied"],1)

    def test_memory_write_tool_denied(self):
        request=self.request(memory=False,tool_intents=({
            "id":"write","name":"memory_remember",
            "arguments":{"content":"new private note","source":"owner"},
        },))
        before=self.store.count("S")
        result=self.engine.respond(request)
        self.assertEqual(result["audit"]["tools_denied"],1)
        self.assertEqual(before,self.store.count("S"))

    def test_tools_disabled_rejects_any_requested_tool(self):
        configuration=AssistConfiguration(
            enabled=True,memory_enabled=True,tools_enabled=False
        )
        engine=AssistantEngine(
            self.model,configuration=configuration,
            memory=self.store,generator=controlled_generator,
        )
        with self.assertRaises(AssistantDisabled):
            engine.respond(self.request(memory=False,tool_intents=({
                "id":"calc","name":"calculator",
                "arguments":{"operation":"add","left":1,"right":1},
            },)))

    def test_tool_results_precede_answer_prefix(self):
        prompt="Question"+ANSWER_HEADING
        modified=append_tool_evidence(prompt,["[tool:calculator]\n42"],4096)
        self.assertTrue(modified.endswith(ANSWER_HEADING))
        self.assertLess(modified.index("42"),modified.index(ANSWER_HEADING))

    def test_tool_results_never_truncate_context(self):
        prompt="Question"+ANSWER_HEADING
        with self.assertRaises(ValueError):
            append_tool_evidence(prompt,["long"*500],100)

    def test_prompt_structure_guard(self):
        with self.assertRaises(ValueError):
            append_tool_evidence("not-a-rag-prompt",["42"],4096)

    def test_unknown_generator_result_rejected(self):
        engine=AssistantEngine(
            self.model,configuration=self.configuration,
            memory=self.store,generator=lambda *a,**kw:{"invalid":"value"},
        )
        with self.assertRaises(TypeError):
            engine.respond(self.request(memory=False))

    def test_assistant_output_audit_no_raw_question(self):
        result=self.engine.respond(self.request())
        self.assertNotIn("gradient descent",json.dumps(result["audit"]))
        self.assertEqual(len(result["audit"]["request_sha256"]),64)


class LocalHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.memory=MemoryStore(Path(cls.temp.name)/"memory.db")
        cls.memory.remember(
            scope="S",content="Gradient descent updates neural weights.",
            source="approved",consent="explicit",
        )
        cls.runtime=Runtime(DummyModel(),ServerLimits(port=8779))
        cls.runtime.assistant=AssistantEngine(
            cls.runtime.model,
            configuration=AssistConfiguration(
                enabled=True,memory_enabled=True,tools_enabled=True
            ),
            memory=cls.memory,generator=controlled_generator,
        )
        cls.server=ThreadingHTTPServer(
            ("127.0.0.1",0),make_handler(cls.runtime)
        )
        cls.thread=threading.Thread(
            target=cls.server.serve_forever,daemon=True
        )
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=3)
        cls.memory.close()
        cls.temp.cleanup()

    def request(self,method,path,payload=None,host="127.0.0.1:8779",
                origin=None):
        connection=http.client.HTTPConnection(
            "127.0.0.1",self.server.server_port,timeout=10
        )
        headers={"Host":host,"Content-Type":"application/json"}
        if origin:
            headers["Origin"]=origin
        data=json.dumps(payload).encode() if payload is not None else None
        try:
            connection.request(method,path,body=data,headers=headers)
            response=connection.getresponse()
            result=response.read()
            return response.status,json.loads(result.decode("utf-8"))
        finally:
            connection.close()

    def test_capabilities_visible(self):
        status,result=self.request("GET","/v1/assist/capabilities")
        self.assertEqual(status,200)
        self.assertTrue(result["assistant_enabled"])

    def test_assistant_post_returns_answer(self):
        status,result=self.request(
            "POST","/v1/assist",
            {"question":"gradient descent","memory":True},
        )
        self.assertEqual(status,200,result)
        self.assertEqual(result["memory_mode"],"memory")

    def test_local_host_required(self):
        status,_=self.request(
            "POST","/v1/assist",
            {"question":"hello"},host="malicious.example"
        )
        self.assertEqual(status,403)

    def test_external_origin_denied(self):
        status,_=self.request(
            "POST","/v1/assist",
            {"question":"hello"},origin="https://evil.example"
        )
        self.assertEqual(status,403)

    def test_foreign_scope_denied(self):
        status,_=self.request(
            "POST","/v1/assist",
            {"question":"secret","scope":"Prime"},
        )
        self.assertEqual(status,403)

    def test_excessive_generation_denied(self):
        status,_=self.request(
            "POST","/v1/assist",
            {"question":"hello","max_new_tokens":10000},
        )
        self.assertEqual(status,400)

    def test_no_remote_mutation(self):
        before=self.memory.count("S")
        status,result=self.request(
            "POST","/v1/assist",
            {"question":"hello","tool_intents":[{
                "id":"mutate","name":"memory_remember",
                "arguments":{"content":"another note","source":"user"}
            }]},
        )
        self.assertEqual(status,200)
        self.assertEqual(result["audit"]["tools_denied"],1)
        self.assertEqual(self.memory.count("S"),before)

    def test_default_generation_route_not_removed(self):
        # Only the assistant route is under test; original /generate
        # remains registered and checked by existing server_v2 tests.
        status,result=self.request("GET","/v1/models")
        self.assertEqual(status,200)
        self.assertIn("models",result)

    def test_capabilities_rejects_external_origin(self):
        status,_=self.request(
            "GET","/v1/assist/capabilities",origin="https://evil.example"
        )
        self.assertEqual(status,403)

    def test_assistant_404_other_path(self):
        status,_=self.request("POST","/v1/assist/unknown",{"question":"hello"})
        self.assertEqual(status,404)

    def test_disabled_assistant_endpoint_403(self):
        # Mutate runtime configuration only while all other tests idle.
        previous=self.runtime.assistant
        try:
            self.runtime.assistant=None
            status,_=self.request("POST","/v1/assist",{"question":"hello"})
            self.assertEqual(status,403)
        finally:
            self.runtime.assistant=previous


if __name__=="__main__":
    unittest.main()
