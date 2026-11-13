import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from s_brain.android_build import build_android
from s_brain.rapid_builder import generate


class AndroidBuildTests(unittest.TestCase):
    def test_requires_gradle(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "app"
            generate("build android calculator", project)
            with patch("s_brain.android_build.shutil.which", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "Gradle executable unavailable"):
                    build_android(project)

    def test_explicit_install_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "app"
            generate("build android calculator", project)
            apk = project / "app/build/outputs/apk/debug/app-debug.apk"
            apk.parent.mkdir(parents=True)
            apk.write_bytes(b"test-placeholder")
            with patch("s_brain.android_build.shutil.which", side_effect=lambda name: name), \
                 patch("s_brain.android_build.subprocess.run") as run:
                run.return_value.returncode = 0
                result = build_android(project)
                self.assertTrue(result["built"])
                self.assertFalse(result["installed"])
                self.assertEqual(run.call_count, 1)
                result = build_android(project, install=True, serial="test-device")
                self.assertTrue(result["installed"])
                self.assertEqual(run.call_count, 3)
                self.assertEqual(run.call_args.args[0][:3], ["adb", "-s", "test-device"])

    def test_rejects_unrelated_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                build_android(Path(directory))


if __name__ == "__main__":
    unittest.main()
