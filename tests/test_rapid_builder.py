import tempfile
import unittest
from pathlib import Path
from s_brain.rapid_builder import classify_request, generate

class RapidBuilderTests(unittest.TestCase):
    def test_routing(self):
        self.assertEqual(classify_request("build me calculator"), "calculator")
        self.assertEqual(classify_request("build a chess game"), "chess")
        self.assertEqual(classify_request("create a website"), "website")
        self.assertEqual(classify_request("build S calculator install on my phone"), "android-calculator")

    def test_generates_all_artifacts(self):
        for request in ("build calculator", "build chess game", "build website", "build calculator for android"):
            with self.subTest(request=request), tempfile.TemporaryDirectory() as tmp:
                result = generate(request, Path(tmp) / "new")
                for name in result["files"]:
                    self.assertTrue((Path(tmp) / "new" / name).is_file())
                self.assertFalse(result["installed"])
                self.assertFalse(result["build_verified"])

    def test_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp)
            (destination / "important.txt").write_text("keep")
            with self.assertRaises(FileExistsError):
                generate("build calculator", destination)
            self.assertEqual((destination / "important.txt").read_text(), "keep")

    def test_unsupported(self):
        with self.assertRaises(ValueError):
            classify_request("build an operating system")

if __name__ == "__main__":
    unittest.main()
