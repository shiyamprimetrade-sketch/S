import tempfile
import unittest
from pathlib import Path
from unity.generation import generate_website
from unity.website_acceptance import verify_website

class WebsiteAcceptanceTests(unittest.TestCase):
    def test_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"site"
            generate_website(root,"Library","Books & more","Open daily",approved=True)
            self.assertEqual(verify_website(root,"Library","Books & more","Open daily")["status"],"verified")
    def test_wrong_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"site"
            generate_website(root,"Library","Wrong","Open daily",approved=True)
            self.assertEqual(verify_website(root,"Library","Books","Open daily")["status"],"rejected")
    def test_injected_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"site"
            generate_website(root,"Library","Books","Open daily",approved=True)
            p=root/"index.html"
            p.write_text(p.read_text().replace("</body>","<script>alert(1)</script></body>"))
            self.assertEqual(verify_website(root,"Library","Books","Open daily")["status"],"rejected")
    def test_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(verify_website(tmp,"Library","Books","Open daily")["status"],"rejected")
