import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("update_feed", ROOT / "scripts/update_feed.py")
feed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feed)


def test_appcast_preserves_machine_version_signature_and_minimum_os(tmp_path):
    output = tmp_path / "appcast.xml"
    info = {"CFBundleVersion": "20261026.170000", "CFBundleShortVersionString": "0.5.0", "LSMinimumSystemVersion": "26.0"}
    feed.write_feed(output, info, "https://example.com/update.dmg?a=1&b=2", "signed", 1234)
    item = ET.parse(output).find("channel/item")
    assert item.find(f"{{{feed.SPARKLE}}}version").text == "20261026.170000"
    assert item.find(f"{{{feed.SPARKLE}}}minimumSystemVersion").text == "26.0"
    assert item.find("enclosure").attrib == {"url": "https://example.com/update.dmg?a=1&b=2", "length": "1234", "type": "application/octet-stream", f"{{{feed.SPARKLE}}}edSignature": "signed"}


def test_update_channels_and_automatic_installation_defaults():
    stable, preview = {}, {}
    feed.configure_app(stable, "release")
    feed.configure_app(preview, "preview")
    assert stable["SUFeedURL"].endswith("/appcast.xml")
    assert preview["SUFeedURL"].endswith("/appcast-preview.xml")
    assert stable["SUEnableAutomaticChecks"] and stable["SUAutomaticallyUpdate"]
    assert stable["SUVerifyUpdateBeforeExtraction"]
    assert stable["SUEnableSystemProfiling"] is False


def test_update_configuration_rejects_unverified_feed_transport(tmp_path, monkeypatch):
    config = feed.update_configuration()
    config["feed_url"] = "http://example.com/appcast.xml"
    (tmp_path / "distribution").mkdir()
    (tmp_path / "distribution/updates.json").write_text(json.dumps(config))
    monkeypatch.setattr(feed, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="HTTPS"):
        feed.update_configuration()


@pytest.mark.parametrize("public,notarized", [(False, True), (True, False)])
def test_publishing_requires_public_repository_and_notarization(tmp_path, monkeypatch, public, notarized):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import publish_update

    monkeypatch.setattr(publish_update.subprocess, "check_output", lambda *a, **kw: json.dumps({"visibility": "PUBLIC" if public else "PRIVATE"}))
    def reject_mutation(*args, **kwargs):
        pytest.fail("Must not upload an ineligible update")
    monkeypatch.setattr(publish_update.subprocess, "run", reject_mutation)
    (tmp_path / "release-manifest.json").write_text(json.dumps({"notarized": notarized}))
    with pytest.raises(ValueError):
        publish_update.publish(tmp_path)
