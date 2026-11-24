import tempfile
import unittest
from pathlib import Path
from unity.responsive_site import render, generate

class ResponsiveTests(unittest.TestCase):
    def test_layout(self):
        r=render("Lab","Science","Research",[("Physics","Experiments"),("Math","Numbers")])
        self.assertIn("auto-fit",r["source"])
        self.assertIn('href="#section-1"',r["source"])
        self.assertIn('id="section-1"',r["source"])
    def test_escape(self):
        self.assertNotIn("<script>",render("Lab","<script>","Research",[("Physics","<script>")])["source"])
    def test_consent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"site"
            self.assertEqual(generate(root,"Lab","Science","Research",[("Physics","Experiments")])["status"],"awaiting_authorization")
            self.assertFalse(root.exists())
    def test_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"site"
            args=(root,"Lab","Science","Research",[("Physics","Experiments")])
            self.assertEqual(generate(*args,approved=True)["status"],"generated")
            self.assertTrue((root/"index.html").is_file())
            with self.assertRaises(FileExistsError):
                generate(*args,approved=True)
