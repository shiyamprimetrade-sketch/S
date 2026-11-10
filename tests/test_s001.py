import unittest
from continuum.s001 import MeaningCompiler, TRAIN, HOLDOUT, run_experiment

class S001Tests(unittest.TestCase):
    def test_independent_holdout(self):
        result=run_experiment()
        self.assertEqual(result["total"],6)
        self.assertGreater(result["correct"],result["baseline"]["correct"])
        self.assertFalse(next(c for c in result["holdout_cases"] if c["input"]=="child")["correct"])
    def test_reject_overlap(self):
        compiler=MeaningCompiler().fit(TRAIN)
        with self.assertRaises(ValueError):
            compiler.evaluate([TRAIN[0]])
    def test_unseen_rule_and_provenance(self):
        compiler=MeaningCompiler().fit(TRAIN)
        self.assertEqual(compiler.predict("party")["output"],"parties")
        self.assertGreaterEqual(compiler.predict("party")["rule"]["support"],2)
    def test_ambiguous_examples_do_not_become_rules(self):
        compiler=MeaningCompiler().fit([("cat","cats"),("dog","dogs"),("bus","buses"),("gas","gases")])
        self.assertEqual(compiler.predict("glass")["output"],"glasses")
    def test_no_unsafe_code_execution(self):
        compiler=MeaningCompiler().fit(TRAIN)
        self.assertIsNone(compiler.predict("child")["output"] if compiler.predict("child")["status"]=="abstain" else None)
        self.assertTrue(all(set(rule)<=set(("suffix","replacement","support")) for rule in compiler.program()["rules"]))
    def test_deterministic_evidence(self):
        self.assertEqual(run_experiment()["sha256"],run_experiment()["sha256"])
    def test_reject_duplicates(self):
        with self.assertRaises(ValueError):
            MeaningCompiler().fit([("cat","cats"),("cat","catz")])
if __name__=="__main__": unittest.main()
