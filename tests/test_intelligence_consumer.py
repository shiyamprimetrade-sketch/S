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

    def test_learned_end_to_end(self):
        import tempfile
        from project_s.secure import SecureEngine
        with tempfile.TemporaryDirectory() as root:
            engine=SecureEngine(root)
            for phrase,capability in [('compute decimal sums and products','calculator'),('responsive company homepage for Cedar Studio','website'),('warehouse product stock quantities','inventory')]:
                with self.subTest(phrase=phrase):
                    task,plan=engine.create(phrase)
                    self.assertEqual(plan['capability'],capability)
                    self.assertIn('learned suggestion',plan['selection'])
                    self.assertFalse((engine.root/task).exists())
                    self.assertTrue(engine.execute_token(task,engine.issue(task))['passed'])
                    self.assertTrue(engine.artifact(task).is_file())
            for phrase in ['quasar nebula pulsar','upload warehouse product stock','calculator and inventory']:
                with self.assertRaises(ValueError):engine.create(phrase)
