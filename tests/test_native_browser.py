"""Browser generation boundary and exact-answer scoring tests."""
import unittest
from s_intelligence.browser import generate_answer
from s_intelligence.reasoning import matches


class NativeBrowserTests(unittest.TestCase):
    def test_input_validation(self):
        for text in ('', ' ', None, 'x' * 4097, 'বাংলা', '\x00'):
            with self.assertRaises(ValueError):
                generate_answer(text)

    def test_exact_answer_controls(self):
        self.assertTrue(matches('  YES.\n', 'yes'))
        self.assertTrue(matches('4', '4'))
        for text in ('14', '4 maybe', 'the answer is 4', 'not yes'):
            self.assertFalse(matches(text, '4'))
