"""Publish a notarized update and activate its feed after the repository is public."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from update_feed import update_configuration


def publish(artifacts):
    config = update_configuration()
    repo = config["repository"]
    visibility = json.loads(subprocess.check_output(["gh", "repo", "view", repo, "--json", "visibility"], text=True))
    if visibility["visibility"] != "PUBLIC":
        raise ValueError("リポジトリの公開後に更新を配信できます。公開設定はこのスクリプトでは変更しません。")
    release = json.loads((artifacts / "release-manifest.json").read_text())
    if not release.get("notarized") or not release["signing_authority"].startswith("Developer ID Application:"):
        raise ValueError("更新の配信にはDeveloper ID署名と公証済みの成果物が必要です。")
    update = json.loads((artifacts / "update-manifest.json").read_text())
    archive, feed = artifacts / update["archive"], artifacts / update["feed"]
    with archive.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != release["sha256"]:
            raise ValueError("公証後にアーカイブの内容が変更されています。")
    # Create the immutable version release before replacing the shared feed.
    args = ["gh", "release", "create", update["tag"], str(archive), str(archive.with_suffix(archive.suffix + ".sha256")),
            str(artifacts / f"TransiroYT-{release['version']}-third-party-sources.zip"),
            "--repo", repo, "--title", f"TransiroYT {release['version']}",
            "--notes", "アプリ本体と同梱yt-dlpの更新。アプリから自動更新できます。"]
    if release["channel"] != "release":
        args.append("--prerelease")
    subprocess.run(args, check=True)
    listed = json.loads(subprocess.check_output(["gh", "release", "list", "--repo", repo, "--json", "tagName", "--limit", "100"], text=True))
    if not any(item["tagName"] == "updates" for item in listed):
        subprocess.run(["gh", "release", "create", "updates", "--repo", repo, "--title", "Update feeds",
                        "--notes", "TransiroYTの自動更新配信ファイル。", "--latest=false"], check=True)
    feeds = [feed]
    if release["channel"] == "release":
        preview_feed = artifacts / "appcast-preview.xml"
        shutil.copy2(feed, preview_feed)
        feeds.append(preview_feed)
    subprocess.run(["gh", "release", "upload", "updates", *map(str, feeds), "--repo", repo, "--clobber"], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    publish(args.artifacts.resolve())


if __name__ == "__main__":
    main()
