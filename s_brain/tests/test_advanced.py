"""S-Brain checkpoint, sampling, benchmarking and training regression tests."""
import json
import math
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from s_brain.checkpoint import (
    best_checkpoint,
    checkpoint_summary,
    compare_checkpoints,
    fingerprint_files,
    list_checkpoints,
    load_manifest,
    load_weights,
    remove_old_checkpoints,
    resume_training,
    save_checkpoint,
    sha256_file,
)
from s_brain.evaluate import (
    BenchmarkCase,
    bootstrap_accuracy_ci,
    case_fingerprint,
    compare_reports,
    evaluate_token_stream,
    keyword_recall,
    load_cases,
    normalized,
    regression_gate,
    run_cases,
    save_cases,
    score_answer,
    sensitivity_by_category,
    summarize_cases,
)
from s_brain.model import Config, SBrain
from s_brain.sampling import (
    SamplingOptions,
    decode_bytes,
    encode_bytes,
    filter_min_p,
    filter_top_k,
    filter_top_p,
    generate_tokens,
    penalize_logits,
    select_token,
    truncate_stop,
)
from s_brain.train_advanced import (
    ByteStream,
    TrainOptions,
    cosine_rate,
    parameter_groups,
    run_training,
)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cfg = Config(vocab_size=256, dim=32, layers=1, heads=4, kv_heads=2, context=16)
        self.model = SBrain(self.cfg)

    def tearDown(self):
        self.temp.cleanup()

    def save(self, name="one", step=1, validation=2.5):
        return save_checkpoint(
            self.root/name, self.model, self.cfg,
            step=step, train_tokens=32*step,
            train_loss=2.0, validation_loss=validation,
            seed=42, dataset_fingerprint="data-fingerprint",
        )

    def test_roundtrip(self):
        expected = self.save()
        actual = load_manifest(self.root/"one")
        self.assertEqual(expected.weights_sha256, actual.weights_sha256)
        loaded, manifest = load_weights(self.root/"one")
        x = torch.randint(0, 256, (1,5))
        self.assertTrue(torch.allclose(self.model(x)[0], loaded(x)[0]))
        self.assertEqual(manifest.model_config, asdict(self.cfg))

    def test_tampering_rejected(self):
        self.save()
        with (self.root/"one"/"model.safetensors").open("ab") as handle:
            handle.write(b"tamper")
        with self.assertRaises(ValueError):
            load_manifest(self.root/"one")

    def test_manifest_schema_rejected(self):
        self.save()
        path = self.root/"one"/"manifest.json"
        value = json.loads(path.read_text())
        value["schema"] = 99
        path.write_text(json.dumps(value))
        with self.assertRaises(ValueError):
            load_manifest(self.root/"one")

    def test_checkpoint_comparison(self):
        self.save("a",step=1,validation=3.0)
        self.save("b",step=5,validation=2.5)
        result = compare_checkpoints(self.root/"a", self.root/"b")
        self.assertTrue(result["comparable"])
        self.assertAlmostEqual(result["validation_loss_delta"], -0.5)

    def test_best_checkpoint(self):
        self.save("slow", step=1, validation=3.2)
        self.save("fast", step=2, validation=1.9)
        self.assertEqual(best_checkpoint(self.root).name, "fast")

    def test_retention_preview(self):
        for index in range(1,6):
            self.save(f"step-{index}",step=index,validation=10-index)
        candidates = remove_old_checkpoints(self.root,keep=2)
        self.assertEqual(len(candidates), 3)
        self.assertTrue(all(path.exists() for path in candidates))

    def test_summary(self):
        self.save()
        result = checkpoint_summary(self.root/"one")
        self.assertGreater(result["weight_bytes"], 0)
        self.assertFalse(result["has_optimizer"])

    def test_missing_optimizer_is_allowed(self):
        self.save()
        optimizer = torch.optim.AdamW(self.model.parameters())
        meta = resume_training(self.root/"one", self.model, optimizer)
        self.assertEqual(meta.step, 1)

    def test_fingerprint_changes_with_data(self):
        path = self.root/"data.bin"
        path.write_bytes(b"hello")
        first = fingerprint_files([path])
        path.write_bytes(b"goodbye")
        self.assertNotEqual(first, fingerprint_files([path]))


class SamplingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.cfg = Config(vocab_size=256,dim=32,layers=1,heads=4,kv_heads=2,context=16)
        self.model = SBrain(self.cfg)

    def test_option_validation(self):
        for option in (
            SamplingOptions(max_new_tokens=0),
            SamplingOptions(temperature=-1),
            SamplingOptions(top_p=0),
            SamplingOptions(min_p=1),
            SamplingOptions(top_k=-1),
            SamplingOptions(repetition_penalty=0),
        ):
            with self.assertRaises(ValueError):
                option.validate()

    def test_top_k(self):
        logits = torch.tensor([[1.0,2.0,3.0,4.0]])
        result = filter_top_k(logits, 2)
        self.assertEqual(torch.isfinite(result).sum(), 2)

    def test_top_p_keeps_one(self):
        scores = torch.tensor([[9.0,1.0,1.0]])
        result = filter_top_p(scores, 0.1)
        self.assertEqual(torch.isfinite(result).sum(), 1)

    def test_min_p_filters_small_values(self):
        scores = torch.tensor([[15.0, -20.0]])
        self.assertEqual(torch.isfinite(filter_min_p(scores, 0.5)).sum(), 1)

    def test_repetition_penalty(self):
        scores = torch.tensor([[0.0,2.0,3.0]])
        history = torch.tensor([[1,1,2]])
        result = penalize_logits(
            scores, history, SamplingOptions(repetition_penalty=2)
        )
        self.assertAlmostEqual(float(result[0,1]),1.0)

    def test_zero_temperature_greedy(self):
        scores = torch.tensor([[0.0,1.0,5.0]])
        token = select_token(
            scores, torch.tensor([[0]]),
            SamplingOptions(temperature=0)
        )
        self.assertEqual(int(token.item()),2)

    def test_seeded_generation(self):
        options = SamplingOptions(max_new_tokens=8, seed=5)
        one = generate_tokens(self.model,[72,105],options)
        two = generate_tokens(self.model,[72,105],options)
        self.assertEqual(one,two)

    def test_byte_roundtrip(self):
        text = "S learns বাংলা"
        self.assertEqual(decode_bytes(encode_bytes(text)),text)

    def test_stop_sequence(self):
        self.assertEqual(truncate_stop("abcSTOPxyz",("STOP",)),("abc","STOP"))

    def test_generation_metadata(self):
        output, metadata = generate_tokens(
            self.model,[72,105],SamplingOptions(max_new_tokens=3,seed=1)
        )
        self.assertEqual(metadata["tokens_generated"],3)
        self.assertLessEqual(len(output), 9)

    def test_invalid_history_shape(self):
        with self.assertRaises(ValueError):
            penalize_logits(
                torch.zeros((1,5)),
                torch.zeros((2,3),dtype=torch.long),
                SamplingOptions()
            )


class EvaluationTests(unittest.TestCase):
    def example_cases(self):
        return [
            BenchmarkCase("1","capital of France?","Paris","geography","human"),
            BenchmarkCase("2","2+2?","4","math","human"),
            BenchmarkCase("3","3+3?","6","math","human"),
        ]

    def test_normalized_comparison(self):
        self.assertEqual(normalized("Hello, WORLD!"),"hello world")
        self.assertTrue(score_answer("Paris", "paris.", "normalized")[0])

    def test_contains_comparison(self):
        self.assertTrue(score_answer("4","the result is 4","contains")[0])

    def test_keyword_recall(self):
        self.assertEqual(keyword_recall("red green","red"),0.5)

    def test_case_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"cases.jsonl"
            save_cases(path,self.example_cases())
            self.assertEqual(load_cases(path),self.example_cases())

    def test_case_fingerprint_is_deterministic(self):
        self.assertEqual(
            case_fingerprint(self.example_cases()),
            case_fingerprint(self.example_cases())
        )

    def test_case_scoring(self):
        answers = {"capital of France?":"Paris","2+2?":"4","3+3?":"5"}
        rows, report = run_cases(
            self.example_cases(), lambda text: answers[text]
        )
        self.assertEqual(report["correct"],2)
        self.assertEqual(report["count"],3)
        self.assertIn("math",report["by_category"])
        self.assertEqual(sensitivity_by_category(rows)["priorities"][0],"math")

    def test_bootstrap_interval(self):
        rows,_ = run_cases(self.example_cases(), lambda text: "4")
        a,b = bootstrap_accuracy_ci(rows,draws=100)
        self.assertLessEqual(a,b)

    def test_report_comparison(self):
        result = compare_reports(
            {"nll":3.0},{"nll":2.0},"nll",True
        )
        self.assertTrue(result["improved"])

    def test_regression_gate(self):
        baseline = {"accuracy":0.9,"p95_latency_ms":100}
        candidate = {"accuracy":0.88,"p95_latency_ms":110}
        self.assertTrue(regression_gate(baseline,candidate)["passed"])

    def test_token_stream(self):
        cfg = Config(vocab_size=256,dim=32,layers=1,heads=4,kv_heads=2,context=16)
        model = SBrain(cfg)
        result = evaluate_token_stream(
            model,torch.tensor([1,2,3,4,5],dtype=torch.long)
        )
        self.assertEqual(result["token_count"],4)
        self.assertTrue(math.isfinite(result["nll"]))


class TrainingTests(unittest.TestCase):
    def test_schedule_warmup(self):
        rate1 = cosine_rate(0,100,10,0.01,0.001)
        rate10 = cosine_rate(9,100,10,0.01,0.001)
        self.assertLess(rate1, rate10)

    def test_schedule_cooldown(self):
        self.assertAlmostEqual(
            cosine_rate(100,100,10,0.01,0.001),0.001
        )

    def test_parameter_groups(self):
        cfg = Config(dim=32,layers=1,heads=4,kv_heads=2,context=16)
        model = SBrain(cfg)
        groups = parameter_groups(model,0.1)
        self.assertEqual(len(groups),2)
        self.assertEqual(groups[1]["weight_decay"],0)

    def test_training_config_rejects_bad_head_dims(self):
        with self.assertRaises(ValueError):
            TrainOptions(
                Path("."), Path("."),dim=30,heads=8
            ).validate()

    def test_train_stream_sampling(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("train","validation","test"):
                (root/f"{name}.bin").write_bytes(bytes(range(128)))
            stream = ByteStream(root,16)
            x,y = stream.sample(
                "train",2,"cpu",torch.Generator().manual_seed(1)
            )
            self.assertEqual(tuple(x.shape),(2,16))
            self.assertTrue(torch.equal(y[:,:-1],x[:,1:]))

    def test_small_training_cycle(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root/"data"
            data.mkdir()
            for name in ("train","validation","test"):
                (data/f"{name}.bin").write_bytes(bytes(range(128))*3)
            result = run_training(TrainOptions(
                corpus_dir=data,output_dir=root/"out",
                steps=2,batch=2,sequence=8,accumulate=1,
                dim=32,layers=1,heads=4,kv_heads=2,
                evaluate_every=1,save_every=1,evaluation_batches=1,
            ))
            self.assertEqual(result["steps"],2)
            self.assertTrue((root/"out"/"step-00000002"/"manifest.json").exists())
            self.assertTrue((root/"out"/"metrics.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
