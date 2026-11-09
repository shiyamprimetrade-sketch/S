import json
import tempfile
import unittest
from pathlib import Path
from axiom_seed import apply, matching_rules, new_memory, predict, prove, read_memory, save_memory, teach, trace, number_word

class TestSeed(unittest.TestCase):
    def test_number_words(self):
        self.assertEqual(number_word(4), 'FOUR')
        self.assertEqual(number_word(42), 'FORTYTWO')
        self.assertEqual(number_word(1005), 'ONETHOUSANDFIVE')
    def test_invalid_size(self):
        with self.assertRaises(ValueError):
            number_word(10000)
    def test_no_examples(self):
        self.assertEqual(predict(new_memory(), 'unknown', 'A')['status'], 'NEEDS_TEACHING')
    def test_teach_discovers_count_as_word(self):
        m=new_memory()
        teach(m,'letters','APPLE','FIVE')
        t=teach(m,'letters','BANANA','SIX')
        self.assertEqual(t['candidate_rules'], ['count_as_word'])
        self.assertEqual(predict(m,'letters','CAT')['output'], 'THREE')
    def test_ambiguous_requires_more(self):
        m=new_memory()
        teach(m,'amb','A','A');teach(m,'amb','B','B')
        self.assertEqual(predict(m,'amb','C')['status'], 'NEEDS_MORE_EXAMPLES')
    def test_failure_evidence(self):
        m=new_memory();teach(m,'t','APPLE','FIVE');teach(m,'t','BANANA','SIX')
        self.assertEqual(prove(m,'t','CAT','FOUR')['status'], 'TEST_FAIL')
        self.assertEqual(predict(m,'t','DOG')['evidence'], 'UNDER_REVIEW_FAILED_TEST')
    def test_pass_evidence(self):
        m=new_memory();teach(m,'t','APPLE','FIVE');teach(m,'t','BANANA','SIX')
        self.assertEqual(prove(m,'t','CAT','THREE')['status'], 'TEST_PASS')
        self.assertEqual(predict(m,'t','DOG')['evidence'], 'HAS_PASSED_UNSEEN_TEST')
    def test_no_train_test_leak(self):
        m=new_memory();teach(m,'t','APPLE','FIVE');teach(m,'t','BANANA','SIX')
        self.assertEqual(prove(m,'t','APPLE','FIVE')['status'], 'NOT_UNSEEN')
    def test_trace_fixed_point(self):
        m=new_memory();teach(m,'t','APPLE','FIVE');teach(m,'t','BANANA','SIX')
        self.assertEqual(trace(m,'t','APPLE')['sequence'],['APPLE','FIVE','FOUR','FOUR'])
    def test_conflict_does_not_change(self):
        m=new_memory();teach(m,'t','APPLE','FIVE')
        self.assertEqual(teach(m,'t','APPLE','NO')['status'],'CONFLICT')
        self.assertEqual(m['skills']['t']['examples']['APPLE'],'FIVE')
    def test_explicit_correction(self):
        m=new_memory();teach(m,'t','APPLE','FIVE');teach(m,'t','BANANA','SIX')
        self.assertEqual(teach(m,'t','APPLE','WRONG',correct=True)['status'],'NO_MATCHING_RULE')
    def test_persistence(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'memory.json';m=new_memory();teach(m,'t','APPLE','FIVE')
            save_memory(p,m)
            self.assertEqual(read_memory(p),m)
    def test_disallow_bad_schema(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'memory.json';p.write_text('{"schema":2,"skills":{}}')
            with self.assertRaises(ValueError):
                read_memory(p)
    def test_utf8_codepoint(self):
        self.assertEqual(apply('count_characters','বাংলা'),str(len('বাংলা')))

if __name__ == '__main__': unittest.main()