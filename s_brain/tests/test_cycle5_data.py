"""Cycle 5: permission-aware local teacher and supervised answer training tests."""
from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from s_brain.teacher_data import (
    TeacherSettings, TeacherQuestion, TeacherExample, TeacherResponseError,
    TeacherUnavailable, digest, question_digest, ollama_payload,
    validate_teacher_response, send_ollama, create_example, load_questions,
    save_questions, load_examples, save_examples, generate_dataset,
    review_example, eligible_training_examples, audit_examples,
)
from s_brain.supervised_data import (
    FormatPolicy, SupervisedRecord, TrainingWindow, WindowSampler,
    formatted_prompt, encode_record, from_teacher_examples, record_fingerprint,
    detect_train_test_leakage, audit_records, export_jsonl, import_jsonl,
    make_windows, dense_completion_windows, build_windows,
    masked_cross_entropy, fingerprint_dataset,
)


def teacher_question(number=1, split="train"):
    return TeacherQuestion(
        identifier=f"q-{number}",
        question=f"Explain the difference between variable {number} and a constant.",
        category="programming", source="self-authored",
        permission="user-owned", split=split
    )


def example(number=1, review="approved", split="train"):
    question=teacher_question(number, split)
    result=create_example(
        question,TeacherSettings(),
        generator=lambda q,s:f"Variable {number} changes; a constant does not."
    )
    if review=="pending":
        return result
    return review_example(result, approved=review=="approved")


class TeacherSettingsTests(unittest.TestCase):
    def test_valid_settings(self):
        TeacherSettings().validate()

    def test_reject_cloud_url(self):
        with self.assertRaises(ValueError):
            TeacherSettings(base_url="https://api.example.com").validate()

    def test_reject_fake_loopback_domain(self):
        with self.assertRaises(ValueError):
            TeacherSettings(base_url="http://127.0.0.1.evil.com").validate()

    def test_reject_non_ollama_port(self):
        with self.assertRaises(ValueError):
            TeacherSettings(base_url="http://localhost:1234").validate()

    def test_reject_embedded_credentials(self):
        with self.assertRaises(ValueError):
            TeacherSettings(base_url="http://u:p@localhost:11434").validate()

    def test_reject_invalid_model(self):
        with self.assertRaises(ValueError):
            TeacherSettings(model="../malicious model").validate()

    def test_reject_negative_temperature(self):
        with self.assertRaises(ValueError):
            TeacherSettings(temperature=-1).validate()

    def test_reject_oversize_question(self):
        with self.assertRaises(ValueError):
            dataclasses.replace(teacher_question(),question="x"*16001).validate()

    def test_reject_wrong_split(self):
        with self.assertRaises(ValueError):
            teacher_question(split="unspecified").validate()

    def test_payload_has_no_cloud_or_tools(self):
        payload=ollama_payload(teacher_question(),TeacherSettings(seed=11))
        self.assertEqual(payload["stream"],False)
        self.assertEqual(payload["options"]["seed"],11)
        self.assertEqual(payload["messages"][-1]["role"],"user")

    def test_question_digest_is_deterministic(self):
        q=teacher_question()
        self.assertEqual(question_digest(q),question_digest(q))

    def test_generated_examples_start_pending(self):
        item=create_example(
            teacher_question(),TeacherSettings(),
            generator=lambda q,s:"An answer to the question."
        )
        self.assertEqual(item.review_status,"pending")
        self.assertEqual(item.question_sha256,digest(item.question))

    def test_teacher_response_validation(self):
        raw={"message":{"content":"Useful answer"}}
        self.assertEqual(validate_teacher_response(raw),"Useful answer")

    def test_teacher_refuses_empty_answer(self):
        with self.assertRaises(TeacherResponseError):
            validate_teacher_response({"message":{"content":"  "}})

    def test_teacher_refuses_non_object(self):
        with self.assertRaises(TeacherResponseError):
            validate_teacher_response(["bad"])

    def test_response_tampering_rejected(self):
        item=example()
        obj=item.as_dict()
        obj["answer"]="tampered"
        with self.assertRaises(ValueError):
            TeacherExample.from_dict(obj)

    def test_human_review_rejects(self):
        item=example(review="pending")
        rejected=review_example(item,approved=False,notes="incorrect")
        self.assertEqual(rejected.review_status,"rejected")

    def test_only_approved_train_is_eligible(self):
        rows=[
            example(1,"approved","train"),
            example(2,"pending","train"),
            example(3,"approved","validation"),
        ]
        eligible=eligible_training_examples(rows)
        self.assertEqual([r.identifier for r in eligible],["q-1"])

    def test_holdout_questions_not_sent_to_teacher(self):
        called=[]
        def generate(q,s):
            called.append(q.identifier)
            return "Valid original answer."
        results,stats=generate_dataset(
            [teacher_question(1),teacher_question(2,"test")],
            TeacherSettings(),generator=generate
        )
        self.assertEqual(called,["q-1"])
        self.assertEqual(stats["skipped_holdouts"],1)
        self.assertEqual(len(results),1)

    def test_audit_pending(self):
        audit=audit_examples([example(1,"pending")])
        self.assertEqual(audit["status"]["pending"],1)
        self.assertEqual(audit["approved_train"],0)

    def test_save_load_questions(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/"questions.jsonl"
            save_questions(target,[teacher_question()])
            self.assertEqual(load_questions(target),[teacher_question()])

    def test_duplicate_question_ids_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                save_questions(Path(tmp)/"q.jsonl",[
                    teacher_question(),teacher_question()
                ])

    def test_save_load_examples(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/"examples.jsonl"
            item=example()
            save_examples(target,[item])
            self.assertEqual(load_examples(target),[item])

    def test_no_teacher_network_on_mock(self):
        with patch("s_brain.teacher_data.send_ollama",side_effect=AssertionError("network")):
            result=create_example(
                teacher_question(),TeacherSettings(),
                generator=lambda q,s:"Locally generated fixture."
            )
        self.assertEqual(result.answer,"Locally generated fixture.")


class SupervisedDataTests(unittest.TestCase):
    def test_format_prompt(self):
        self.assertTrue(formatted_prompt("Hello?",FormatPolicy()).startswith("Question: "))

    def test_record_has_expected_byte_values(self):
        record=encode_record(
            "a","What is S?","A small experimental model.",
            category="ai",source_hash="digest",permission="user-owned"
        )
        record.validate()
        self.assertTrue(all(0<=x<256 for x in record.prompt_bytes))
        self.assertTrue(bytes(record.completion_bytes).startswith(b"A small"))

    def test_only_approved_teacher_examples(self):
        rows=from_teacher_examples([
            example(1,"approved"),example(2,"pending"),example(3,"rejected")
        ])
        self.assertEqual(len(rows),1)

    def test_reject_short_answers(self):
        with self.assertRaises(ValueError):
            encode_record(
                "a","Explain this?","x",category="test",
                source_hash="hash",permission="owned",
            )

    def test_record_fingerprint_changes_with_answer(self):
        a=encode_record("a","Q?","Answer one.",category="t",
                        source_hash="a",permission="owned")
        b=encode_record("a","Q?","Answer two.",category="t",
                        source_hash="a",permission="owned")
        self.assertNotEqual(record_fingerprint(a),record_fingerprint(b))

    def test_cross_split_overlap_rejected(self):
        a=encode_record("a","Same?","Answer one.",category="t",
                        source_hash="a",permission="owned",split="train")
        b=encode_record("b","Same?","Answer two.",category="t",
                        source_hash="b",permission="owned",split="test")
        self.assertEqual(detect_train_test_leakage([a,b]),["b"])

    def test_dense_window_shapes(self):
        record=encode_record(
            "a","What is a transformer?","A neural network attention mechanism.",
            category="ai",source_hash="hash",permission="owned"
        )
        windows=dense_completion_windows(record,16)
        self.assertGreater(len(windows),0)
        self.assertEqual(len(windows[0].token_ids),16)
        self.assertEqual(windows[0].supervised_tokens,1)

    def test_completion_only_labels(self):
        record=encode_record(
            "a","Q?","Supervised answer bytes.",category="ai",
            source_hash="digest",permission="owned"
        )
        windows=dense_completion_windows(record,16)
        self.assertTrue(all(sum(v!=-100 for v in item.target_ids)==1 for item in windows))

    def test_sampled_masks(self):
        record=encode_record(
            "a","Why?","Because reasons and evidence.",category="ai",
            source_hash="digest",permission="owned"
        )
        windows=build_windows([record],16)
        sampler=WindowSampler(windows,16,seed=9)
        x,y,supervised=sampler.sample(3)
        self.assertEqual(tuple(x.shape),(3,16))
        self.assertEqual(supervised,3)
        self.assertEqual(int((y!=-100).sum()),3)

    def test_stride_windows(self):
        record=encode_record(
            "a","A long question?","An answer with multiple informative words.",
            category="a",source_hash="s",permission="owned"
        )
        windows=make_windows(record,16,stride=4)
        self.assertTrue(all(window.supervised_tokens>0 for window in windows))

    def test_masked_loss_valid(self):
        logits=torch.randn(2,8,256,requires_grad=True)
        targets=torch.full((2,8),-100,dtype=torch.long)
        targets[:, -1]=42
        loss=masked_cross_entropy(logits,targets)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(logits.grad)

    def test_fully_masked_loss_rejected(self):
        with self.assertRaises(ValueError):
            masked_cross_entropy(
                torch.zeros((1,8,256)),torch.full((1,8),-100,dtype=torch.long)
            )

    def test_wrong_logits_vocab_rejected(self):
        with self.assertRaises(ValueError):
            masked_cross_entropy(
                torch.zeros((1,8,10)),torch.zeros((1,8),dtype=torch.long)
            )

    def test_jsonl_roundtrip(self):
        record=encode_record(
            "a","What is AI?","Intelligent software.",category="ai",
            source_hash="digest",permission="owned"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"data.jsonl"
            export_jsonl(path,[record])
            self.assertEqual(import_jsonl(path),[record])

    def test_tampered_jsonl_rejected(self):
        record=encode_record(
            "a","What is AI?","Intelligent software.",category="ai",
            source_hash="digest",permission="owned"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"data.jsonl"
            export_jsonl(path,[record])
            row=json.loads(path.read_text())
            row["completion_bytes"]=[42]*12
            path.write_text(json.dumps(row))
            with self.assertRaises(ValueError):
                import_jsonl(path)

    def test_dataset_fingerprint_order_independent(self):
        a=encode_record("a","One?","Answer one.",category="t",
                        source_hash="1",permission="owned")
        b=encode_record("b","Two?","Answer two.",category="t",
                        source_hash="2",permission="owned")
        self.assertEqual(fingerprint_dataset([a,b]),fingerprint_dataset([b,a]))

    def test_audit_reports_counts(self):
        record=encode_record(
            "a","What is a token?","A vocabulary unit.",category="ai",
            source_hash="hash",permission="owned",
        )
        report=audit_records([record])
        self.assertEqual(report["records"],1)
        self.assertGreater(report["target_tokens"],0)


if __name__=="__main__":
    unittest.main()
