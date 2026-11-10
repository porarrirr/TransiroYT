"""Create an Ed25519-signed Sparkle appcast for a signed TransiroYT archive."""
from __future__ import annotations

import argparse
import base64
import datetime
import json
import plistlib
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
SPARKLE = "http://www.andymatuschak.org/xml-namespaces/sparkle"
ET.register_namespace("sparkle", SPARKLE)


def update_configuration():
    config = json.loads((ROOT / "distribution/updates.json").read_text())
    if len(base64.b64decode(config["public_key"], validate=True)) != 32:
        raise ValueError("Invalid Sparkle public key")
    for field in ("feed_url", "preview_feed_url"):
        url = urlparse(config[field])
        if url.scheme != "https" or not url.hostname or url.username or url.password:
            raise ValueError("Update feeds must use HTTPS")
    return config


def configure_app(info, channel):
    config = update_configuration()
    info.update(SUFeedURL=config["feed_url" if channel == "release" else "preview_feed_url"],
                SUPublicEDKey=config["public_key"], SUEnableAutomaticChecks=True,
                SUAutomaticallyUpdate=True, SUVerifyUpdateBeforeExtraction=True,
                SUEnableSystemProfiling=False, SUScheduledCheckInterval=86400)


def write_feed(path, info, archive_url, signature, length):
    rss = ET.Element("rss", {"version": "2.0"})
    channel = ET.SubElement(rss, "channel")
    ET.SubElement(channel, "title").text = "TransiroYT"
    item = ET.SubElement(channel, "item")
    ET.SubElement(item, "title").text = f"TransiroYT {info['CFBundleShortVersionString']}"
    ET.SubElement(item, f"{{{SPARKLE}}}version").text = info["CFBundleVersion"]
    ET.SubElement(item, f"{{{SPARKLE}}}shortVersionString").text = info["CFBundleShortVersionString"]
    ET.SubElement(item, f"{{{SPARKLE}}}minimumSystemVersion").text = info["LSMinimumSystemVersion"]
    ET.SubElement(item, "pubDate").text = datetime.datetime.now(datetime.UTC).strftime("%a, %d %b %Y %H:%M:%S +0000")
    ET.SubElement(item, "enclosure", {"url": archive_url, "length": str(length), "type": "application/octet-stream",
                                     f"{{{SPARKLE}}}edSignature": signature})
    ET.indent(rss)
    ET.ElementTree(rss).write(path, encoding="utf-8", xml_declaration=True)


def create_feed(app, archive, output, archive_url, tools):
    if urlparse(archive_url).scheme != "https":
        raise ValueError("Published update archives must use HTTPS")
    config = update_configuration()
    with (app / "Contents/Info.plist").open("rb") as stream:
        info = plistlib.load(stream)
    if info.get("SUPublicEDKey") != config["public_key"]:
        raise ValueError("Application and publisher update keys do not match")
    subprocess.run(["codesign", "--verify", "--deep", "--strict", str(app)], check=True)
    public_key = subprocess.check_output([str(tools / "generate_keys"), "--account", config["keychain_account"], "-p"], text=True).strip()
    if public_key != config["public_key"]:
        raise ValueError("Keychain public key does not match update configuration")
    signer = tools / "sign_update"
    signature = subprocess.check_output([str(signer), "--account", config["keychain_account"], "-p", str(archive)], text=True).strip()
    subprocess.run([str(signer), "--account", config["keychain_account"], "--verify", str(archive), signature], check=True)
    write_feed(output, info, archive_url, signature, archive.stat().st_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--archive-url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tools", type=Path, default=ROOT / ".local/DesktopBuild/artifacts/sparkle/Sparkle/bin")
    args = parser.parse_args()
    create_feed(args.app, args.archive, args.output, args.archive_url, args.tools)


if __name__ == "__main__":
    main()
