import unittest
from project_s.intelligence import interpret

class IntelligenceTests(unittest.TestCase):
    def test_consumer_intent(self):
        for phrase,label in [('compute decimal sums','calculator'),('responsive company homepage','website'),('warehouse product stock quantities','inventory')]:
            with self.subTest(phrase=phrase):
                result=interpret(phrase)
                self.assertEqual(result['capability'],label)
                self.assertEqual(result['authority'],'advisory only')
                self.assertFalse(result['calibrated'])
    def test_unknown(self):
        self.assertEqual(interpret('quasar nebula pulsar')['capability'],'unknown')
