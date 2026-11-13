"""Explicit Android APK build and optional ADB installation for S rapid projects.

Never installs without --install; subprocesses use fixed argv and no shell.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess


def build_android(project: Path, *, install: bool = False, serial: str | None = None) -> dict:
    project = Path(project).resolve()
    if not (project / "settings.gradle").is_file() or not (project / "app/build.gradle").is_file():
        raise ValueError("Not an S Android Gradle project")
    gradle = shutil.which("gradle")
    if not gradle:
        raise RuntimeError("Gradle executable unavailable; Android build not started")
    result = subprocess.run(
        [gradle, "--no-daemon", "assembleDebug"],
        cwd=project, capture_output=True, text=True, timeout=600, shell=False,
    )
    if result.returncode:
        raise RuntimeError("Gradle build failed: " + result.stderr[-2000:])
    apk = project / "app/build/outputs/apk/debug/app-debug.apk"
    if not apk.is_file():
        raise RuntimeError("Gradle returned success but APK is missing")
    output = {"apk": str(apk), "built": True, "installed": False}
    if install:
        adb = shutil.which("adb")
        if not adb:
            raise RuntimeError("ADB unavailable; APK built but not installed")
        command = [adb]
        if serial:
            command += ["-s", serial]
        command += ["install", "-r", str(apk)]
        installed = subprocess.run(
            command, capture_output=True, text=True, timeout=120, shell=False,
        )
        if installed.returncode:
            raise RuntimeError("APK built; installation failed: " + installed.stderr[-1000:])
        output["installed"] = True
    return output


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("project", type=Path)
    parser.add_argument("--install", action="store_true",
                        help="Explicitly authorize installing the debug APK via ADB")
    parser.add_argument("--serial", help="ADB device serial")
    args = parser.parse_args(argv)
    print(json.dumps(build_android(args.project, install=args.install, serial=args.serial), indent=2))


if __name__ == "__main__":
    main()
