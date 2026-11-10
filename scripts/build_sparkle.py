"""Embed and sign the pinned Sparkle framework without altering its entitlements."""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path


def sign_sparkle(app, identity, *, timestamp=False):
    framework = app / "Contents/Frameworks/Sparkle.framework"
    version = framework / "Versions/B"
    children = [version / "Autoupdate", *sorted((version / "XPCServices").glob("*.xpc")), version / "Updater.app", framework]
    for child in children:
        args = ["codesign", "--force", "--sign", identity, "--preserve-metadata=entitlements"]
        if identity != "-":
            args += ["--options", "runtime", "--timestamp" if timestamp else "--timestamp=none"]
        subprocess.run([*args, str(child)], check=True)
    subprocess.run(["codesign", "--verify", "--deep", "--strict", str(framework)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("app", type=Path)
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    source = args.artifacts / "sparkle/Sparkle"
    framework = args.app / "Contents/Frameworks/Sparkle.framework"
    if framework.exists():
        shutil.rmtree(framework)
    framework.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ditto", str(source / "Sparkle.xcframework/macos-arm64_x86_64/Sparkle.framework"), str(framework)], check=True)
    executable = args.app / "Contents/MacOS/TransiroYT"
    commands = subprocess.check_output(["otool", "-l", str(executable)], text=True)
    for rpath in re.findall(r"cmd LC_RPATH\s+cmdsize \d+\s+path ([^\n]+?) \(offset", commands):
        if rpath.startswith("/") and not rpath.startswith(("/usr/lib/", "/System/Library/")):
            subprocess.run(["install_name_tool", "-delete_rpath", rpath, str(executable)], check=True)
    resources = args.app / "Contents/Resources"
    shutil.copy2(source / "LICENSE", resources / "Sparkle-LICENSE.txt")
    sign_sparkle(args.app, "-")


if __name__ == "__main__":
    main()
