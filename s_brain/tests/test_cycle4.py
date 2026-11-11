"""Cycle 4: prefix reuse, scheduler, and measured learning regression tests.

All test models are tiny, randomly initialized Transformers. Results establish
software behavior only. They do not establish conversational intelligence.
"""
from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from concurrent.futures import CancelledError
from pathlib import Path

import torch

from s_brain.model import Config, SBrain
from s_brain.kv_cache import prefill
from s_brain.prefix_reuse import (
    PrefixCache, PrefixPolicy, PrefixKey, prefill_reusing_prefix,
    prove_prefix_equivalence,
)
from s_brain.scheduler import (
    FairMicrobatchScheduler, SchedulerPolicy, QueueFull, SchedulerClosed,
    RequestExpired, wait_result,
)
from s_brain.batching import BatchRequest, BatchResult
from s_brain.learning_probe import (
    ProbeConfig, build_model, capture_parameters, dataset_fingerprint,
    make_batch, parameter_delta, prediction_fraction, run_probe,
    split_bytes, summarize_probe, validation_loss, write_report,
)


def small_model(context=12):
    torch.manual_seed(77)
    m = SBrain(Config(dim=32, layers=1, heads=4, kv_heads=2, context=context))
    m.eval()
    return m


class PrefixTests(unittest.TestCase):
    def setUp(self):
        self.model = small_model()
        self.cache = PrefixCache(PrefixPolicy(min_prefix_tokens=1))
        self.ids = torch.tensor([[5, 6, 7, 8, 9]], dtype=torch.long)

    def run_cached(self, ids, scope="project-a", model_id="weights-a"):
        return prefill_reusing_prefix(
            self.model, ids, cache=self.cache, model_id=model_id,
            project_scope=scope
        )

    def test_first_is_miss(self):
        _, state, report = self.run_cached(self.ids)
        self.assertEqual(report.kind, "miss")
        self.assertEqual(state.positions, 5)

    def test_exact_hit(self):
        self.run_cached(self.ids)
        cached, state, hit = self.run_cached(self.ids)
        reference, _ = self.model(self.ids)
        self.assertEqual(hit.kind, "exact")
        self.assertEqual(hit.remaining_tokens, 0)
        self.assertTrue(torch.allclose(cached[:, -1, :], reference[:, -1, :]))

    def test_prefix_hit(self):
        self.run_cached(self.ids[:, :3])
        cached, state, hit = self.run_cached(self.ids)
        ref, _ = self.model(self.ids)
        self.assertEqual(hit.kind, "prefix")
        self.assertEqual(hit.matched_tokens, 3)
        self.assertEqual(hit.remaining_tokens, 2)
        self.assertTrue(torch.allclose(cached[:, -1, :], ref[:, -1, :], atol=1e-5))
        self.assertEqual(state.positions, 5)

    def test_project_isolation(self):
        self.run_cached(self.ids, scope="project-a")
        _, _, hit = self.run_cached(self.ids, scope="project-b")
        self.assertEqual(hit.kind, "miss")

    def test_model_isolation(self):
        self.run_cached(self.ids, model_id="first")
        _, _, hit = self.run_cached(self.ids, model_id="second")
        self.assertEqual(hit.kind, "miss")

    def test_modified_prompt_is_not_exact_hit(self):
        self.run_cached(self.ids)
        modified = self.ids.clone()
        modified[0, 0] = 100
        _, _, hit = self.run_cached(modified)
        self.assertEqual(hit.kind, "miss")

    def test_explicit_purge(self):
        self.run_cached(self.ids)
        self.assertEqual(self.cache.clear(project_scope="project-a"), 1)
        self.assertEqual(self.cache.stats()["entries"], 0)

    def test_lru_limit(self):
        cache = PrefixCache(PrefixPolicy(max_entries=2, min_prefix_tokens=1))
        for i in range(3):
            ids = torch.tensor([[i + 1, i + 2]], dtype=torch.long)
            prefill_reusing_prefix(
                self.model, ids, cache=cache, model_id="m", project_scope="p",
            )
        self.assertLessEqual(cache.stats()["entries"], 2)
        cache.validate()

    def test_ttl_expiry(self):
        clock = [0.0]
        cache = PrefixCache(
            PrefixPolicy(ttl_seconds=2, min_prefix_tokens=1),
            clock=lambda: clock[0],
        )
        prefill_reusing_prefix(
            self.model, self.ids, cache=cache, model_id="m", project_scope="p",
        )
        clock[0] = 3.0
        result = cache.lookup("m", self.ids, project_scope="p")
        self.assertEqual(result[2].kind, "miss")

    def test_tiny_byte_limit(self):
        cache = PrefixCache(
            PrefixPolicy(max_bytes=1, min_prefix_tokens=1)
        )
        _, state = prefill(self.model, self.ids)
        _, state = prefill(self.model, self.ids)
        self.assertFalse(cache.put("m", state, self.model(self.ids)[0], project_scope="p"))

    def test_key_is_stable(self):
        self.assertEqual(
            PrefixKey.create("m", self.ids),
            PrefixKey.create("m", self.ids.clone()),
        )

    def test_invalid_cache_input(self):
        with self.assertRaises(ValueError):
            PrefixKey.create("m", self.ids.float())

    def test_proven_prefix_equivalence(self):
        result = prove_prefix_equivalence(self.model, self.ids, split_at=2)
        self.assertTrue(result["match"], str(result))
        self.assertEqual(result["saved_prefill_tokens"], 2)

    def test_multiple_prefix_candidates(self):
        self.run_cached(self.ids[:, :2])
        self.run_cached(self.ids[:, :4])
        _, _, hit = self.run_cached(self.ids)
        self.assertEqual(hit.matched_tokens, 4)

    def test_cache_recomputes_after_full_context(self):
        ids = torch.randint(0, 256, (1, self.model.cfg.context))
        report = prove_prefix_equivalence(self.model, ids, split_at=6)
        self.assertTrue(report["match"], str(report))


class DummyExecutor:
    def __init__(self):
        self.groups = []

    def __call__(self, requests):
        self.groups.append([entry.identifier for entry in requests])
        return [
            BatchResult(
                identifier=entry.identifier,
                response=entry.prompt.upper(),
                prompt_bytes=len(entry.prompt.encode("utf-8")),
                generated_tokens=1,
                finish_reason="length",
                stop_sequence=None,
                cache_rebuilds=0,
                elapsed_seconds=0.001,
            )
            for entry in requests
        ]


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.model = small_model(context=8)
        self.executor = DummyExecutor()
        self.policy = SchedulerPolicy(batch_wait_ms=0, max_queue=8,
                                      max_pending_per_owner=4)

    def test_unstarted_scheduler_rejects(self):
        scheduler = FairMicrobatchScheduler(
            self.model, policy=self.policy, executor=self.executor
        )
        with self.assertRaises(SchedulerClosed):
            scheduler.submit(BatchRequest("a", "hi"))

    def test_one_request_completes(self):
        with FairMicrobatchScheduler(
            self.model, policy=self.policy, executor=self.executor
        ) as scheduler:
            result = wait_result(
                scheduler.submit(BatchRequest("a", "hi")), timeout=3
            )
            self.assertEqual(result.response, "HI")
            self.assertEqual(scheduler.stats()["completed"], 1)

    def test_multiple_requests_complete(self):
        with FairMicrobatchScheduler(
            self.model, policy=self.policy, executor=self.executor
        ) as scheduler:
            futures = [
                scheduler.submit(BatchRequest(str(i), "same"), owner=str(i % 2))
                for i in range(4)
            ]
            results = [wait_result(item, timeout=3) for item in futures]
            self.assertEqual(len(results), 4)
            self.assertEqual(
                {result.identifier for result in results},
                {"0", "1", "2", "3"},
            )

    def test_duplicate_pending_identifier(self):
        policy = dataclasses.replace(self.policy, batch_wait_ms=100)
        with FairMicrobatchScheduler(
            self.model, policy=policy, executor=self.executor
        ) as scheduler:
            scheduler.submit(BatchRequest("same", "abc"))
            with self.assertRaises(ValueError):
                scheduler.submit(BatchRequest("same", "xyz"))

    def test_invalid_scheduler_policy(self):
        with self.assertRaises(ValueError):
            SchedulerPolicy(max_queue=0).validate()

    def test_invalid_prompt_limit(self):
        policy = dataclasses.replace(self.policy, max_prompt_bytes=2)
        with FairMicrobatchScheduler(
            self.model, policy=policy, executor=self.executor
        ) as scheduler:
            with self.assertRaises(ValueError):
                scheduler.submit(BatchRequest("x", "long"))

    def test_stop_without_start(self):
        scheduler = FairMicrobatchScheduler(
            self.model, policy=self.policy, executor=self.executor
        )
        scheduler.stop()
        self.assertFalse(scheduler.stats()["running"])

    def test_no_automatic_restart_after_close(self):
        scheduler = FairMicrobatchScheduler(
            self.model, policy=self.policy, executor=self.executor
        )
        scheduler.start()
        scheduler.stop()
        with self.assertRaises(SchedulerClosed):
            scheduler.start()

    def test_context_manager_exit(self):
        scheduler = FairMicrobatchScheduler(
            self.model, policy=self.policy, executor=self.executor
        )
        with scheduler:
            self.assertTrue(scheduler.stats()["running"])
        self.assertFalse(scheduler.stats()["running"])

    def test_result_type_checked(self):
        from concurrent.futures import Future
        f = Future()
        f.set_result("not-a-result")
        with self.assertRaises(TypeError):
            wait_result(f, timeout=1)


class LearningTests(unittest.TestCase):
    def setUp(self):
        self.config = ProbeConfig(
            steps=2, batch_size=2, sequence=8,
            dim=32, layers=1, heads=4, kv_heads=2,
        )
        self.training = (b"abc123def456" * 12)
        self.validation = (b"abcdef123456" * 12)

    def test_training_data_split_rejects_identical(self):
        with self.assertRaises(ValueError):
            split_bytes(b"abc"*20, b"abc"*20, 8)

    def test_dataset_fingerprints(self):
        report = dataset_fingerprint(self.training, self.validation)
        self.assertFalse(report["identical"])
        self.assertEqual(len(report["training_sha256"]), 64)

    def test_train_batch_labels_shifted(self):
        values = torch.tensor(list(self.training), dtype=torch.long)
        x, y = make_batch(
            values, 8, 2, torch.Generator().manual_seed(3), "cpu"
        )
        self.assertTrue(torch.equal(x[:, 1:], y[:, :-1]))

    def test_invalid_probe_config(self):
        with self.assertRaises(ValueError):
            ProbeConfig(lr=-1).validate()

    def test_zero_loss_not_assumed(self):
        model = build_model(self.config)
        ids = torch.randint(0, 256, (2, 8))
        logits, loss = model(ids, ids)
        self.assertGreater(float(loss), 0)
        self.assertEqual(tuple(logits.shape), (2, 8, 256))

    def test_parameter_delta_detects_optimizer_update(self):
        model = build_model(self.config)
        original = capture_parameters(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        ids = torch.randint(0, 256, (2, 8))
        _, loss = model(ids, ids)
        loss.backward()
        optimizer.step()
        self.assertGreater(parameter_delta(original, model), 0)

    def test_prediction_fraction_range(self):
        logits = torch.randn(2, 3, 256)
        targets = torch.randint(0, 256, (2, 3))
        score = prediction_fraction(logits, targets)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

    def test_probe_runs_actual_weight_updates(self):
        report = run_probe(self.training, self.validation, self.config)
        self.assertEqual(report["recorded_updates"], 2)
        self.assertTrue(report["all_recorded_updates_changed_weights"])
        self.assertFalse(report["assessment"]["general_intelligence_proven"])

    def test_validation_is_finite(self):
        model = build_model(self.config)
        source = torch.tensor(list(self.validation))
        loss = validation_loss(
            model, source, sequence=8, batch_size=2,
            generator=torch.Generator().manual_seed(1), device="cpu",
        )
        self.assertTrue(torch.isfinite(torch.tensor(loss)))

    def test_probe_summary(self):
        report = run_probe(self.training, self.validation, self.config)
        summary = summarize_probe(report)
        self.assertEqual(summary["recorded_steps"], 2)
        self.assertGreater(summary["mean_weight_change_l2"], 0)

    def test_report_is_saved(self):
        report = run_probe(self.training, self.validation, self.config)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "probe.json"
            write_report(path, report)
            self.assertIn(
                "baseline_validation_loss", json.loads(path.read_text())
            )


if __name__ == "__main__":
    unittest.main()
