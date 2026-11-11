"""Cycle 3 benchmark integrity and deterministic fixture tests."""
import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

import torch

from s_brain.model import Config,SBrain
from s_brain.batch_benchmark import (
    BatchBenchmarkPlan,benchmark_passed,compare_batch_reports,
    latency_summary,make_requests,run_benchmarks,time_inference,
    write_report,
)


class BatchBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(321)
        cls.net=SBrain(Config(dim=32,layers=1,heads=4,kv_heads=2,context=12))
        cls.net.eval()

    def test_invalid_batch_size(self):
        with self.assertRaises(ValueError):
            BatchBenchmarkPlan(batch_sizes=(0,)).validate()

    def test_invalid_repeats(self):
        with self.assertRaises(ValueError):
            BatchBenchmarkPlan(repeats=0).validate()

    def test_requests_are_deterministic(self):
        plan=BatchBenchmarkPlan(prompt_bytes=8)
        one=make_requests(plan,4)
        two=make_requests(plan,4)
        self.assertEqual(one,two)

    def test_heterogeneous_prompts_within_limit(self):
        plan=BatchBenchmarkPlan(prompt_bytes=8)
        requests=make_requests(plan,10,heterogeneous=True)
        self.assertTrue(all(1<=len(x.prompt)<=8 for x in requests))

    def test_latency_summary(self):
        report=latency_summary([1,3,2])
        self.assertEqual(report["median_seconds"],2)
        self.assertEqual(report["p95_seconds"],3)

    def test_no_latency_samples(self):
        with self.assertRaises(ValueError):
            latency_summary([])

    def test_single_batch_run(self):
        plan=BatchBenchmarkPlan(
            batch_sizes=(1,),prompt_bytes=3,
            output_tokens=1,warmups=0,repeats=1
        )
        report=run_benchmarks(self.net,plan)
        self.assertEqual(len(report["comparisons"]),1)
        self.assertTrue(report["comparisons"][0]["same_outputs"])

    def test_multi_batch_run(self):
        plan=BatchBenchmarkPlan(
            batch_sizes=(1,2),prompt_bytes=4,
            output_tokens=2,warmups=0,repeats=1
        )
        report=run_benchmarks(self.net,plan)
        self.assertTrue(benchmark_passed(report),str(report))

    def test_different_plan_comparison_rejected(self):
        plan=BatchBenchmarkPlan(
            batch_sizes=(1,),prompt_bytes=3,
            output_tokens=1,warmups=0,repeats=1
        )
        left=run_benchmarks(self.net,plan)
        right=json.loads(json.dumps(left))
        right["plan"]["prompt_bytes"]=10
        with self.assertRaises(ValueError):
            compare_batch_reports(left,right)

    def test_parity_gate(self):
        report={"comparisons":[{"same_outputs":True}]}
        self.assertTrue(benchmark_passed(report))
        report["comparisons"][0]["same_outputs"]=False
        self.assertFalse(benchmark_passed(report))

    def test_report_storage(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"report.json"
            write_report(path,{"proof":"synthetic fixture"})
            self.assertEqual(json.loads(path.read_text())["proof"],"synthetic fixture")

    def test_same_report_comparison(self):
        plan=BatchBenchmarkPlan(
            batch_sizes=(1,),prompt_bytes=3,
            output_tokens=1,warmups=0,repeats=1
        )
        baseline=run_benchmarks(self.net,plan)
        result=compare_batch_reports(baseline,baseline)
        self.assertAlmostEqual(result["comparisons"][0]["speed_ratio"],1)


if __name__=="__main__":
    unittest.main()
