import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "distribution_licenses", Path(__file__).resolve().parents[1] / "scripts/distribution_licenses.py")
licenses = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(licenses)


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


@pytest.fixture
def fixture(tmp_path):
    root, runtime, build = (tmp_path / name for name in ["root", "runtime", "build"])
    write(root / "LICENSE", "MIT test application")
    write(root / "docs/licenses.md", "Third-party licenses apply separately")
    write(root / "distribution/licenses/manifest.json", '{"components": []}')
    write(root / "distribution/licenses/models.json", '{"models": []}')
    write(root / "distribution/fallback-models.lock.json", '{"models": []}')
    write(root / "distribution/licenses/native-manifest.json", '{"components": [], "files": []}')
    write(root / "distribution/models.lock.json", '{"models": []}')
    write(root / "distribution/third-party-sources.lock.json", '{"sources": []}')
    (runtime / "help").mkdir(parents=True)
    return root, runtime, build


def package(runtime, name="example", license="MIT", notice=True):
    directory = runtime / f"python-api/lib/python3.12/site-packages/{name}-1.0.dist-info"
    write(directory / "METADATA", f"Name: {name}\nVersion: 1.0\nLicense-Expression: {license}\n")
    if notice:
        write(directory / "licenses/LICENCE", "Copyright upstream; permission granted")
    return directory


def test_offline_notices_preserve_original_text_and_inventory(fixture):
    root, runtime, build = fixture
    directory = package(runtime)
    licenses.prepare_licenses(root, runtime, build)
    report = json.loads((runtime / "licenses/package-licenses.json").read_text())
    notice = runtime / "licenses" / report["api"][0]["license_files"][0]
    assert notice.read_bytes() == (directory / "licenses/LICENCE").read_bytes()
    assert (runtime / "help/licenses/TransiroYT-MIT.txt").read_bytes() == (root / "LICENSE").read_bytes()
    assert 'licenses/python/api/example-1.0.dist-info/licenses/LICENCE' in (runtime / "help/licenses.html").read_text()


@pytest.mark.parametrize("license,notice,error", [
    ("MIT", False, "Missing license text"),
    ("GPL-2.0-or-later", True, "Unreviewed GPL"),
    ("LGPL-2.1-or-later", True, "Corresponding source required"),
])
def test_unreviewed_or_incomplete_packages_stop_distribution(fixture, license, notice, error):
    root, runtime, build = fixture
    package(runtime, license=license, notice=notice)
    with pytest.raises(ValueError, match=error):
        licenses.prepare_licenses(root, runtime, build)


def test_mutagen_is_removed_without_changing_other_packages(tmp_path):
    write(tmp_path / "mutagen/__init__.py", "optional")
    write(tmp_path / "mutagen-1.0.dist-info/METADATA", "optional metadata")
    write(tmp_path / "yt_dlp/__init__.py", "subtitles")
    licenses.remove_optional_metadata_editor(tmp_path)
    assert not (tmp_path / "mutagen").exists()
    assert not list(tmp_path.glob("mutagen-*.dist-info"))
    assert (tmp_path / "yt_dlp/__init__.py").read_text() == "subtitles"


def test_legacy_notice_collection_does_not_misclassify_runtime_exception(fixture):
    root, runtime, build = fixture
    directory = package(runtime)
    write(directory / "METADATA", "Name: example\nVersion: 1.0\nLicense: BSD notices "
          + "x" * 200 + " GPL-3.0 WITH GCC-exception-3.1\n")
    licenses.prepare_licenses(root, runtime, build)


def test_matching_source_requires_exact_version_and_hash(tmp_path):
    data = b"source archive"
    source = tmp_path / "build/sources/soxr.tar.gz"
    source.parent.mkdir(parents=True)
    source.write_bytes(data)
    write(tmp_path / "distribution/third-party-sources.lock.json", json.dumps({"sources": [{
        "name": "soxr", "version": "1.0", "filename": "soxr.tar.gz", "url": "https://example.invalid/source",
        "sha256": hashlib.sha256(data).hexdigest()}]}))
    with pytest.raises(ValueError, match="version mismatch"):
        licenses.matching_sources(tmp_path, tmp_path / "build", {"soxr": {"2.0"}})
    assert licenses.matching_sources(tmp_path, tmp_path / "build", {}) == []
    assert licenses.matching_sources(tmp_path, tmp_path / "build", {"soxr": {"1.0"}}) == [source]
    source.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        licenses.matching_sources(tmp_path, tmp_path / "build", {"soxr": {"1.0"}})


def test_changed_model_revision_requires_updated_terms(fixture):
    root, runtime, build = fixture
    write(root / "distribution/fallback-models.lock.json", json.dumps({"models": [{"repo": "model", "revision": "new"}]}))
    with pytest.raises(ValueError, match="model revisions"):
        licenses.prepare_licenses(root, runtime, build)


def test_native_input_change_requires_corresponding_source_review(tmp_path):
    write(tmp_path / "distribution/licenses/native-manifest.json", json.dumps({"components": [{
        "binary": "native.dylib", "input_sha256": hashlib.sha256(b"reviewed").hexdigest()}]}))
    write(tmp_path / "packages/native.dylib", "reviewed")
    licenses.validate_native_inputs(tmp_path, tmp_path / "packages")
    write(tmp_path / "packages/native.dylib", "changed")
    with pytest.raises(ValueError, match="review corresponding source"):
        licenses.validate_native_inputs(tmp_path, tmp_path / "packages")


def test_cpython_version_change_requires_updated_dependency_notices(fixture):
    root, runtime, build = fixture
    write(root / "distribution/licenses/manifest.json", json.dumps({"components": [],
          "runtime_components": [{"prefix": "python-api", "version": "3.12.11",
                                  "header": "include/python3.12/patchlevel.h"}]}))
    write(runtime / "python-api/include/python3.12/patchlevel.h", '#define PY_VERSION "3.12.12"\n')
    with pytest.raises(ValueError, match="CPython notices need version review"):
        licenses.prepare_licenses(root, runtime, build)
