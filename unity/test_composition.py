import unittest
from unity.composition import Component, compose

class CompositionTests(unittest.TestCase):
    def test_order(self):
        registry={"base":Component("base","x=1"),"app":Component("app","y=x+1",("base",))}
        result=compose(registry,["app","base"])
        self.assertEqual(result["components"],["base","app"])
        self.assertEqual(result["source"],"x=1\n\ny=x+1\n")
        self.assertFalse(result["verified"])
    def test_cycle(self):
        with self.assertRaises(ValueError):
            compose({"a":Component("a","",("b",)),"b":Component("b","",("a",))},["a"])
    def test_missing(self):
        with self.assertRaises(ValueError):
            compose({"a":Component("a","",("missing",))},["a"])
    def test_invalid_name(self):
        with self.assertRaises(ValueError):
            compose({},["../escape"])
