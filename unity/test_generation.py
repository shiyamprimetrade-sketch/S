import tempfile
import unittest
from pathlib import Path
from hashlib import sha256
from unity.generation import generate_website

class GenerationTests(unittest.TestCase):
    def test_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"site"
            result=generate_website(root,"Bakery","Fresh bread","Open daily",approved=True)
            data=(root/"index.html").read_bytes()
            self.assertIn(b"<h1>Fresh bread</h1>",data)
            self.assertEqual(result["sha256"],sha256(data).hexdigest())
            self.assertFalse(result["verified"])
    def test_denied(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"site"
            self.assertEqual(generate_website(root,"Bakery","Bread","Daily")["status"],"awaiting_authorization")
            self.assertFalse(root.exists())
    def test_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"site"
            generate_website(root,"Site","<script>alert(1)</script>","Text",approved=True)
            self.assertNotIn("<script>",(root/"index.html").read_text())
    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                generate_website(tmp,"Site","Title","Text",approved=True)
