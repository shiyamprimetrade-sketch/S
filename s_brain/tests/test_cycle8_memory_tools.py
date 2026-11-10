"""Cycle 8 S-Memory and S-Tools tests.

The tests use temporary databases and exact mock data. The tool router is
disabled by default and must never acquire shell, subprocess, filesystem
or network privileges. Memory stays separate between project scopes.
"""
from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

from s_brain.memory_store import (
    MemoryPolicy,MemoryStore,MemoryRecord,RetrievalHit,
    content_hash,tokens,validate_scope,validate_consent,
    validate_source,evaluate_content,rank_records,bm25,
    citation_context,memory_quality,
)
from s_brain.tools_control import (
    ToolPolicy,ToolCall,ToolController,ToolDefinition,ToolFailure,
    ToolPermissionError,ToolSchemaError,ToolLimitError,
    decimal_calculator,text_statistics,json_extract,memory_search,
    serialized_size,nesting_depth,validate_json_value,validate_arguments,
    DEFAULT_TOOLS,
)


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.now=[100000]
        self.store=MemoryStore(
            Path(self.temp.name)/"memory.sqlite3",
            clock=lambda:float(self.now[0]),
        )

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def remember(self,text="S model training uses gradient descent.",
                 scope="project-S",ttl=200):
        return self.store.remember(
            scope=scope,content=text,source="self-authored-note",
            consent="explicit",ttl_seconds=ttl,tags=("research",),
        )

    def test_policy(self):
        MemoryPolicy().validate()

    def test_policy_rejects_zero_quota(self):
        with self.assertRaises(ValueError):
            MemoryPolicy(maximum_items=0).validate()

    def test_scope_sanitized(self):
        validate_scope("S-brain:training")
        with self.assertRaises(ValueError):
            validate_scope("../bad scope")

    def test_source_required(self):
        with self.assertRaises(ValueError):
            validate_source("")

    def test_consent_required(self):
        with self.assertRaises(PermissionError):
            validate_consent("auto-inferred")

    def test_secret_refused(self):
        with self.assertRaises(ValueError):
            self.remember("api_key=secret123")

    def test_email_refused(self):
        with self.assertRaises(ValueError):
            self.remember("contact john@example.com")

    def test_empty_content_refused(self):
        with self.assertRaises(ValueError):
            self.remember("   ")

    def test_oversized_content_refused(self):
        with self.assertRaises(ValueError):
            self.remember("z"*9000)

    def test_invalid_tag_refused(self):
        with self.assertRaises(ValueError):
            self.store.remember(
                scope="p",content="verified content",source="owned",
                consent="explicit",tags=("unsafe tag",),
            )

    def test_create_and_get(self):
        entry=self.remember()
        self.assertEqual(self.store.count("project-S"),1)
        loaded=self.store.get("project-S",entry.id)
        self.assertEqual(loaded.content,entry.content)
        self.assertEqual(loaded.revision,1)

    def test_cross_scope_get_isolated(self):
        entry=self.remember()
        self.assertIsNone(self.store.get("project-Prime",entry.id))

    def test_scope_list_isolated(self):
        self.remember(scope="project-S")
        self.remember("Fruit shipments arrive in Bangladesh",scope="project-Prime")
        self.assertEqual(len(self.store.list_scope("project-S")),1)
        self.assertEqual(len(self.store.list_scope("project-Prime")),1)

    def test_search_finds_relevant(self):
        self.remember("A Transformer uses attention for token prediction")
        self.remember("Oranges are imported by fruit companies")
        results=self.store.retrieve("project-S","transformer attention",limit=3)
        self.assertEqual(len(results),1)
        self.assertIn("attention",results[0].matched_terms)

    def test_search_no_cross_scope(self):
        self.remember("Company private records for testing",scope="project-Prime")
        self.assertEqual(self.store.retrieve("project-S","private"),[])

    def test_memory_expiry(self):
        entry=self.remember(ttl=5)
        self.now[0]+=6
        self.assertIsNone(self.store.get("project-S",entry.id))
        self.assertEqual(self.store.count("project-S"),0)

    def test_memory_update_increments_revision(self):
        entry=self.remember()
        changed=self.store.update(
            "project-S",entry.id,"Gradient descent updates neural weights.",
            consent="explicit",
        )
        self.assertEqual(changed.revision,2)
        self.assertNotEqual(changed.content_sha256,entry.content_sha256)

    def test_cross_scope_update_denied(self):
        entry=self.remember()
        with self.assertRaises(KeyError):
            self.store.update(
                "project-Prime",entry.id,"Not allowed",consent="explicit",
            )

    def test_delete_single(self):
        entry=self.remember()
        self.assertTrue(self.store.forget("project-S",entry.id))
        self.assertFalse(self.store.forget("project-S",entry.id))
        self.assertIsNone(self.store.get("project-S",entry.id))

    def test_cross_scope_delete_denied(self):
        entry=self.remember()
        self.assertFalse(self.store.forget("project-Prime",entry.id))
        self.assertIsNotNone(self.store.get("project-S",entry.id))

    def test_delete_project(self):
        self.remember("one note")
        self.remember("two note")
        self.remember("Other project",scope="project-Prime")
        self.assertEqual(self.store.forget_scope("project-S"),2)
        self.assertEqual(self.store.count("project-Prime"),1)

    def test_metadata_omits_content(self):
        entry=self.remember("Private owned research finding")
        result=self.store.export_metadata("project-S")
        self.assertEqual(result["count"],1)
        self.assertNotIn("Private owned research",json.dumps(result))
        self.assertEqual(result["memories"][0]["content_sha256"],
                         entry.content_sha256)

    def test_audit_does_not_log_content(self):
        entry=self.remember("Distinct original experimental record")
        self.store.forget("project-S",entry.id)
        result=self.store.audit("project-S")
        self.assertEqual([item["operation"] for item in result],
                         ["remember","forget"])
        self.assertNotIn("Distinct original",str(result))

    def test_scope_capacity(self):
        self.store.policy=MemoryPolicy(maximum_items=1)
        self.remember()
        with self.assertRaises(ValueError):
            self.remember("another valid note")

    def test_invalid_ttl(self):
        with self.assertRaises(ValueError):
            self.remember(ttl=0)

    def test_ranking_zero_query(self):
        entry=self.remember()
        self.assertEqual(rank_records("!!!",[entry]),[])

    def test_bm25_positive(self):
        score=bm25(["neural"],["neural","model"],{"neural":1},2,2)
        self.assertGreater(score,0)

    def test_bm25_absent_is_zero(self):
        score=bm25(["absent"],["neural","model"],{},1,2)
        self.assertEqual(score,0)

    def test_rank_is_deterministic(self):
        a=self.remember("neural model uses gradients")
        b=self.remember("neural network weights update from gradients")
        rows=self.store.list_scope("project-S")
        left=rank_records("neural gradients",rows)
        right=rank_records("neural gradients",rows)
        self.assertEqual([x.record.id for x in left],
                         [x.record.id for x in right])

    def test_citation_context_has_source(self):
        self.remember("Gradient descent is an optimizer")
        hits=self.store.retrieve("project-S","gradient optimizer")
        rendered=citation_context(hits,max_bytes=1000)
        self.assertIn("source=",rendered)
        self.assertIn("[/memory]",rendered)

    def test_context_budget_respected(self):
        self.remember()
        hits=self.store.retrieve("project-S","gradient")
        self.assertEqual(citation_context(hits,max_bytes=2),"")

    def test_retrieval_quality(self):
        one=self.remember("Neural model training and gradients")
        two=self.remember("Orange fruit shipping imports")
        hits=self.store.retrieve("project-S","neural model")
        report=memory_quality({one.id},hits,2)
        self.assertEqual(report["hits_at_k"],1)
        self.assertEqual(report["recall_at_k"],1)

    def test_retrieve_limit_rejected(self):
        with self.assertRaises(ValueError):
            self.store.retrieve("project-S","query",limit=10000)

    def test_persistence_across_connections(self):
        entry=self.remember()
        with MemoryStore(Path(self.temp.name)/"memory.sqlite3",
                         clock=lambda:float(self.now[0])) as connection:
            self.assertEqual(connection.get("project-S",entry.id).content,
                             entry.content)


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.store=MemoryStore(Path(self.temp.name)/"tools.sqlite3")
        self.controller=ToolController(
            scope="S",store=self.store,policy=ToolPolicy(),
        )

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def call(self,name,args,approval=None):
        return ToolCall(name=name,arguments=args,scope="S",approval=approval)

    def test_tool_policy_validation(self):
        ToolPolicy().validate()

    def test_invalid_policy_quota(self):
        with self.assertRaises(ValueError):
            ToolPolicy(max_calls=0).validate()

    def test_tool_disabled_by_default(self):
        with self.assertRaises(ToolPermissionError):
            self.controller.invoke(self.call("calculator",{
                "operation":"add","left":1,"right":2,
            }))

    def test_enable_and_disable(self):
        self.controller.enable()
        self.assertTrue(self.controller.enabled)
        self.controller.disable()
        self.assertFalse(self.controller.enabled)

    def test_allowlist_hides_write_tools(self):
        allowed=[x["name"] for x in self.controller.describe()]
        self.assertIn("calculator",allowed)
        self.assertNotIn("memory_remember",allowed)

    def test_unknown_tool_denied(self):
        self.controller.enable()
        with self.assertRaises(ToolPermissionError):
            self.controller.invoke(self.call("run_shell",{"command":"whoami"}))

    def test_cross_scope_denied(self):
        self.controller.enable()
        with self.assertRaises(ToolPermissionError):
            self.controller.invoke(ToolCall(
                "calculator",{"operation":"add","left":1,"right":2},"Prime"
            ))

    def test_calculator_add(self):
        self.controller.enable()
        result=self.controller.invoke(self.call(
            "calculator",{"operation":"add","left":"1.25","right":"2.75"}
        ))
        self.assertTrue(result.success)
        self.assertEqual(result.output["value"],"4.00")

    def test_calculator_zero_division_is_controlled(self):
        self.controller.enable()
        result=self.controller.invoke(self.call(
            "calculator",{"operation":"divide","left":1,"right":0}
        ))
        self.assertFalse(result.success)
        self.assertEqual(result.error_code,"ToolSchemaError")

    def test_calculator_rejects_eval(self):
        with self.assertRaises(ToolSchemaError):
            decimal_calculator({
                "operation":"multiply","left":"__import__('os')","right":"3",
            },"S")

    def test_calculator_power_limit(self):
        with self.assertRaises(ToolSchemaError):
            decimal_calculator({
                "operation":"power","left":"3","right":"1000",
            },"S")

    def test_text_stats(self):
        self.controller.enable()
        result=self.controller.invoke(self.call("text_stats",{
            "text":"Hello world hello"
        }))
        self.assertEqual(result.output["words"],3)
        self.assertEqual(result.output["unique_words"],2)

    def test_json_path(self):
        self.controller.enable()
        result=self.controller.invoke(self.call("json_extract",{
            "object":{"rows":[{"name":"S"}]},"path":["rows",0,"name"]
        }))
        self.assertEqual(result.output["value"],"S")

    def test_json_path_absent_fails(self):
        self.controller.enable()
        result=self.controller.invoke(self.call("json_extract",{
            "object":{"x":1},"path":["absent"]
        }))
        self.assertFalse(result.success)

    def test_memory_search(self):
        self.store.remember(
            scope="S",content="Transformer training uses loss",
            source="owned",consent="explicit",
        )
        self.controller.enable()
        result=self.controller.invoke(self.call("memory_search",{
            "query":"transformer loss","limit":5,
        }))
        self.assertEqual(result.output["count"],1)

    def test_memory_search_other_scope_isolated(self):
        self.store.remember(
            scope="Prime",content="fruit import secret strategies",
            source="owned",consent="explicit",
        )
        self.controller.enable()
        result=self.controller.invoke(self.call("memory_search",{
            "query":"fruit import",
        }))
        self.assertEqual(result.output["count"],0)

    def test_memory_write_disabled(self):
        self.controller.enable()
        with self.assertRaises(ToolPermissionError):
            self.controller.permit_write("memory_remember")

    def test_memory_write_requires_approval(self):
        self.controller=ToolController(
            scope="S",store=self.store,
            policy=ToolPolicy(
                allow_memory_write=True,
                permitted_tools=("calculator","memory_remember","memory_forget"),
            ),
        )
        self.controller.enable()
        with self.assertRaises(ToolPermissionError):
            self.controller.invoke(self.call("memory_remember",{
                "content":"Learning is measurable","source":"user",
            }))

    def test_memory_write_one_use(self):
        self.controller=ToolController(
            scope="S",store=self.store,
            policy=ToolPolicy(
                allow_memory_write=True,
                permitted_tools=("memory_remember","memory_forget"),
            ),
        )
        self.controller.enable()
        token=self.controller.permit_write("memory_remember")
        result=self.controller.invoke(self.call(
            "memory_remember",{
                "content":"Research facts need citations","source":"owned"
            },approval=token,
        ))
        self.assertTrue(result.success)
        self.assertEqual(self.store.count("S"),1)
        with self.assertRaises(ToolPermissionError):
            self.controller.invoke(self.call(
                "memory_remember",{
                    "content":"A second record","source":"owned"
                },approval=token,
            ))

    def test_approval_scope_is_name_specific(self):
        self.controller=ToolController(
            scope="S",store=self.store,
            policy=ToolPolicy(
                allow_memory_write=True,
                permitted_tools=("memory_remember","memory_forget"),
            ),
        )
        self.controller.enable()
        token=self.controller.permit_write("memory_remember")
        with self.assertRaises(ToolPermissionError):
            self.controller.invoke(self.call(
                "memory_forget",{"memory_id":"nope"},approval=token
            ))

    def test_call_quota(self):
        self.controller=ToolController(
            scope="S",store=self.store,
            policy=ToolPolicy(max_calls=1),
        )
        self.controller.enable()
        self.controller.invoke(self.call(
            "calculator",{"operation":"add","left":1,"right":1}
        ))
        with self.assertRaises(ToolLimitError):
            self.controller.invoke(self.call(
                "calculator",{"operation":"add","left":1,"right":1}
            ))

    def test_input_quota(self):
        self.controller=ToolController(
            scope="S",store=self.store,
            policy=ToolPolicy(max_input_bytes=30),
        )
        self.controller.enable()
        with self.assertRaises(ToolLimitError):
            self.controller.invoke(self.call(
                "text_stats",{"text":"long input string"*20}
            ))

    def test_batch_quota(self):
        self.controller.enable()
        with self.assertRaises(ToolLimitError):
            self.controller.run_batch([self.call(
                "calculator",{"operation":"add","left":1,"right":1}
            ) for _ in range(20)])

    def test_batch_success(self):
        self.controller.enable()
        results=self.controller.run_batch([self.call(
            "calculator",{"operation":"add","left":1,"right":i}
        ) for i in range(3)])
        self.assertEqual(len(results),3)

    def test_unknown_arguments_rejected(self):
        self.controller.enable()
        with self.assertRaises(ToolSchemaError):
            self.controller.invoke(self.call(
                "calculator",{
                    "operation":"add","left":1,"right":1,"extra":"bad"
                },
            ))

    def test_missing_arguments_rejected(self):
        self.controller.enable()
        with self.assertRaises(ToolSchemaError):
            self.controller.invoke(self.call(
                "calculator",{"left":1,"right":1}
            ))

    def test_nonfinite_json_rejected(self):
        with self.assertRaises(ToolSchemaError):
            validate_json_value({"value":float("nan")})

    def test_nonjson_object_rejected(self):
        with self.assertRaises(ToolSchemaError):
            validate_json_value({"value":object()})

    def test_audit_redacts_arguments(self):
        self.controller.enable()
        self.controller.invoke(self.call(
            "text_stats",{"text":"private authored excerpt"}
        ))
        audit=self.controller.audit()
        self.assertNotIn("private authored excerpt",str(audit))
        self.assertTrue(audit["arguments_redacted"])

    def test_disable_drops_approvals(self):
        self.controller=ToolController(
            scope="S",store=self.store,
            policy=ToolPolicy(
                allow_memory_write=True,
                permitted_tools=("memory_remember",),
            ),
        )
        self.controller.enable()
        self.controller.permit_write("memory_remember")
        self.controller.disable()
        self.assertEqual(self.controller.quota()["write_approvals_remaining"],0)

    def test_nesting_depth(self):
        self.assertEqual(nesting_depth({"x":[{"z":1}]}),3)

    def test_result_metadata(self):
        self.controller.enable()
        result=self.controller.invoke(self.call(
            "calculator",{"operation":"subtract","left":5,"right":3}
        ))
        self.assertIsNotNone(result.audit_id)
        self.assertGreaterEqual(result.elapsed_seconds,0)
        self.assertFalse(DEFAULT_TOOLS["calculator"].mutates)


if __name__=="__main__":
    unittest.main()
