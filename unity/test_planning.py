import unittest
from unity.planning import architecture

class PlanningTests(unittest.TestCase):
    def test_website(self):
        p=architecture("website","Bakery",["heading","description"])
        self.assertEqual(p["files"],["index.html"])
        self.assertFalse(p["authorized"])
        self.assertFalse(p["implemented"])
    def test_unsupported(self):
        self.assertEqual(architecture("website","Bakery",["payments"])["status"],"unsupported_features")
    def test_invalid_name(self):
        with self.assertRaises(ValueError):
            architecture("website","../escape",["heading"])
    def test_inventory(self):
        self.assertEqual(architecture("inventory app","Stock",["register","search"])["dependencies"],[])
