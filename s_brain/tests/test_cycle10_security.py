"""Cycle 10 adversarial regression tests for tool-assisted S interface.

These explicitly negative cases guard scope, write boundaries, prompt
truncation, output provenance and untrusted model/tool content. They run
without external network access or unapproved training data.
"""
from __future__ import annotations

import dataclasses
import tempfile
import unittest
from pathlib import Path

from s_brain.assist_runtime import (
    AssistConfiguration,AssistLimits,AssistRequest,AssistantEngine,
    AssistantDisabled,run_read_only_tools,append_tool_evidence,
)
from s_brain.memory_store import MemoryStore
from s_brain.tools_control import ToolPermissionError
from s_brain.rag_inference import ANSWER_HEADING
from s_brain.tool_answer_benchmark import (
    ToolAnswerCase,public_tool_question,
)


class FakeModel:
    class Config:
        context=4096
    cfg=Config()


def generator(model,prompt,*,max_new_tokens=32,seed=7):
    return {"response":"Unverified generated output","usage":{}}


class AdversarialPolicyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.database=MemoryStore(Path(self.tmp.name)/"safe.sqlite3")
        self.config=AssistConfiguration(
            enabled=True,memory_enabled=True,tools_enabled=True,
        )
        self.engine=AssistantEngine(
            FakeModel(),configuration=self.config,
            memory=self.database,generator=generator,
        )

    def tearDown(self):
        self.database.close()
        self.tmp.cleanup()

    def test_explicit_scope_remains_unmodified(self):
        question=AssistRequest(question="unrelated text",scope="S")
        original=question.scope
        self.engine.respond(question)
        self.assertEqual(question.scope,original)

    def test_cross_scope_tool_does_not_return_data(self):
        self.database.remember(
            scope="Prime",source="other",consent="explicit",
            content="Confidential import research data.",
        )
        req=AssistRequest(
            question="Find confidential import research",
            memory=False,scope="S",
            tool_intents=({
                "id":"search","name":"memory_search",
                "arguments":{"query":"confidential"},
            },),
        )
        outcome=self.engine.respond(req)
        self.assertEqual(outcome["audit"]["tools_executed"],0)
        self.assertEqual(outcome["audit"]["tools_denied"],1)
        self.assertNotIn("Confidential import",str(outcome))

    def test_model_cannot_insert_write_token(self):
        req=AssistRequest(
            question="Write note",scope="S",
            tool_intents=({
                "id":"write",
                "name":"memory_remember",
                "arguments":{
                    "content":"Attacker-directed memory",
                    "source":"model",
                    "approval":"fake",
                },
            },),
        )
        report=self.engine.respond(req)
        self.assertEqual(report["audit"]["tools_denied"],1)
        self.assertEqual(self.database.count("S"),0)

    def test_explicit_memory_enabled_does_not_allow_auto_capture(self):
        req=AssistRequest(
            question="Remember everything I say",scope="S",memory=True,
        )
        output=self.engine.respond(req)
        self.assertFalse(output["live_memory_write_allowed"])
        self.assertEqual(self.database.count("S"),0)

    def test_localhost_only_does_not_enable_arbitrary_shell(self):
        req=AssistRequest(
            question="Execute user code",scope="S",
            tool_intents=({
                "id":"run","name":"run_shell",
                "arguments":{"command":"echo no"},
            },),
        )
        output=self.engine.respond(req)
        self.assertEqual(output["audit"]["tools_denied"],1)

    def test_tool_name_case_cannot_bypass_allowlist(self):
        req=AssistRequest(
            question="Calculate",scope="S",
            tool_intents=({
                "id":"case","name":"Calculator",
                "arguments":{
                    "operation":"add","left":1,"right":1,
                },
            },),
        )
        output=self.engine.respond(req)
        self.assertEqual(output["audit"]["tools_denied"],1)

    def test_negative_tool_depth_ignored_safely(self):
        req=AssistRequest(
            question="Compute",scope="S",
            tool_intents=({
                "id":"json","name":"json_extract",
                "arguments":{
                    "object":{"a":{"b":{"c":{"d":{"e":{"f":1}}}}}},
                    "path":["a","b","c","d","e","f"],
                },
            },),
        )
        output=self.engine.respond(req)
        self.assertEqual(output["audit"]["tools_denied"],1)
        # Excess nesting is explicitly denied by the schema depth quota.
        self.assertFalse(output["live_memory_write_allowed"])

    def test_malformed_tool_arguments_fail_without_raw_log(self):
        req=AssistRequest(
            question="Check math",scope="S",
            tool_intents=({
                "id":"badmath","name":"calculator",
                "arguments":{
                    "operation":"divide","left":"1","right":"0",
                },
            },),
        )
        output=self.engine.respond(req)
        self.assertEqual(output["audit"]["tools_executed"],1)
        self.assertNotIn("divide",str(output["tool_trace"]))

    def test_unknown_permissioned_scope_rejected(self):
        with self.assertRaises(PermissionError):
            self.engine.respond(AssistRequest(
                question="Hello",scope="S:another",
            ))

    def test_content_overflow_never_silently_drops_end_prompt(self):
        prompt="Question"+ANSWER_HEADING
        with self.assertRaises(ValueError):
            append_tool_evidence(
                prompt,["danger"*800],context=100
            )

    def test_no_empty_answer_prefix_after_tools(self):
        prompt="Question"+ANSWER_HEADING
        final=append_tool_evidence(
            prompt,["[tool:calculator]\n123\n[/tool-data]"],context=1000
        )
        self.assertTrue(final.endswith(ANSWER_HEADING))

    def test_tool_gold_answer_never_exposed_in_public_question(self):
        benchmark=ToolAnswerCase(
            identifier="test",
            question="What is 9 plus 10?",scope="S",
            tool_name="calculator",
            arguments={"operation":"add","left":9,"right":10},
            expected_output={"value":"19"},
            expected_answer="19",category="arithmetic",
        )
        public=public_tool_question(benchmark)
        self.assertNotIn("expected_answer",public.__dataclass_fields__)
        self.assertNotIn("expected_output",public.__dataclass_fields__)

    def test_assistant_disabled_even_if_memory_database_exists(self):
        engine=AssistantEngine(
            FakeModel(),memory=self.database,generator=generator
        )
        with self.assertRaises(AssistantDisabled):
            engine.respond(AssistRequest(question="Hello"))

    def test_tools_require_host_startup_consent(self):
        with self.assertRaises(ValueError):
            AssistConfiguration(
                enabled=False,tools_enabled=True
            ).validate()

    def test_s_brain_stays_independent_of_ollama(self):
        reply=self.engine.respond(AssistRequest(
            question="Hello experimental model",
        ))
        self.assertTrue(reply["independent_inference"])
        self.assertTrue(reply["no_ollama_fallback"])


if __name__=="__main__":
    unittest.main()
