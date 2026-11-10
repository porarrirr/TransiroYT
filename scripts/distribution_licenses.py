"""Retain notices and matching LGPL sources in distributable builds."""
from __future__ import annotations

import hashlib
import html
import json
import re
import shutil
import urllib.request
from email.parser import Parser


def normalized(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def is_notice(path):
    return path.name.lower().startswith(("license", "licence", "copying", "notice", "copyright", "authors"))


def remove_optional_metadata_editor(packages):
    # yt-dlp's optional Mutagen add-on embeds music tags. TransiroYT only reads
    # subtitles and extracts audio; it never enables metadata postprocessors.
    # Do not distribute this GPL add-on as part of the MIT application.
    for path in [packages / "mutagen", *packages.glob("mutagen-*.dist-info")]:
        if path.exists():
            shutil.rmtree(path)


def validate_native_inputs(root, packages):
    """Check wheel-native inputs against the reviewed, source-matched build."""
    manifest = json.loads((root / "distribution/licenses/native-manifest.json").read_text())
    for component in manifest["components"]:
        binary = packages / component["binary"]
        if binary.exists():
            with binary.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != component["input_sha256"]:
                    raise ValueError(f"Native library input changed; review corresponding source: {binary}")


def matching_sources(root, build, versions):
    entries = json.loads((root / "distribution/third-party-sources.lock.json").read_text())["sources"]
    paths = []
    for entry in entries:
        name = normalized(entry["name"])
        if name not in versions:
            continue
        if versions.get(name) != {entry["version"]}:
            raise ValueError(f"Corresponding source version mismatch: {name}")
        path = build / "sources" / entry["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with urllib.request.urlopen(entry["url"], timeout=60) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise ValueError(f"Source checksum mismatch: {name}")
            path.write_bytes(data)
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != entry["sha256"]:
                raise ValueError(f"Source checksum mismatch: {name}")
        paths.append(path)
    return paths


def prepare_licenses(root, runtime, build):
    """Fail closed on absent notices, unreviewed GPL, or unmatched LGPL sources."""
    licenses = runtime / "licenses"
    licenses.mkdir(exist_ok=True)
    shutil.copy2(root / "LICENSE", licenses / "TransiroYT-MIT.txt")
    shutil.copy2(root / "docs/licenses.md", licenses / "README-ja.md")
    shutil.copytree(root / "distribution/licenses", licenses / "supplemental", dirs_exist_ok=True)
    shutil.copy2(root / "distribution/third-party-sources.lock.json", licenses / "third-party-sources.lock.json")
    supplemental_manifest = json.loads((licenses / "supplemental/manifest.json").read_text())
    supplements = supplemental_manifest["components"]
    by_name = {normalized(item["name"]): item for item in supplements}
    for item in supplements:
        for file in item["files"]:
            if hashlib.sha256((licenses / "supplemental" / file["file"]).read_bytes()).hexdigest() != file["sha256"]:
                raise ValueError(f"License checksum mismatch: {file['file']}")
    for component in supplemental_manifest.get("runtime_components", []):
        header = (runtime / component["prefix"] / component["header"]).read_text()
        if not re.search(r'^#define\s+PY_VERSION\s+"' + re.escape(component["version"]) + r'"', header, re.MULTILINE):
            raise ValueError(f"CPython notices need version review: {component['prefix']}")
    models = json.loads((licenses / "supplemental/models.json").read_text())["models"]
    required = {item["repo"]: item["revision"]
                for filename in ("models.lock.json", "fallback-models.lock.json")
                for item in json.loads((root / "distribution" / filename).read_text())["models"]}
    if {item["repo"]: item["revision"] for item in models} != required:
        raise ValueError("Pinned model terms do not match model revisions")
    for model in models:
        for file in model["files"]:
            if hashlib.sha256((licenses / "supplemental" / file["file"]).read_bytes()).hexdigest() != file["sha256"]:
                raise ValueError(f"Model terms checksum mismatch: {file['file']}")
    native = json.loads((licenses / "supplemental/native-manifest.json").read_text())
    for file in native["files"]:
        if hashlib.sha256((licenses / "supplemental" / file["file"]).read_bytes()).hexdigest() != file["sha256"]:
            raise ValueError(f"Native component notice checksum mismatch: {file['file']}")
    inventory, versions = {}, {}
    for kind in ["api", "tts"]:
        python = runtime / f"python-{kind}"
        records = []
        for directory in sorted(python.glob("lib/python*/site-packages/*.dist-info")):
            metadata = Parser().parsestr((directory / "METADATA").read_text())
            name, version = metadata["Name"], metadata["Version"]
            key = normalized(name)
            versions.setdefault(key, set()).add(version)
            license_id = metadata["License-Expression"] or metadata["License"] or "See license text"
            classification = " ".join(metadata.get_all("Classifier", []))
            # Legacy License fields may contain the entire notice collection
            # (NumPy includes GCC's runtime exception). Check the SPDX/short
            # declaration, not incidental license names inside those texts.
            declaration = metadata["License-Expression"] or (license_id if len(license_id) < 200 else "")
            if "GNU General Public License" in classification or re.search(r"(?<![A-Za-z])(?:A?GPL)-", declaration):
                raise ValueError(f"Unreviewed GPL dependency in distribution: {name}")
            texts = []
            for source in sorted(directory.rglob("*")):
                if source.is_file() and is_notice(source):
                    target = licenses / "python" / kind / directory.name / source.relative_to(directory)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
                    texts.append(str(target.relative_to(licenses)))
            if key in by_name:
                item = by_name[key]
                if version not in item["versions"]:
                    raise ValueError(f"Supplemental license version mismatch: {name} {version}")
                texts.extend("supplemental/" + file["file"] for file in item["files"])
                license_id = item["license"]
            if not texts:
                raise ValueError(f"Missing license text: {kind}/{name} {version}")
            if ("LGPL" in declaration or "Lesser General Public License" in classification) and key not in {"soxr", "autocommand"}:
                raise ValueError(f"Corresponding source required for unreviewed LGPL dependency: {name}")
            records.append({"name": name, "version": version, "license": license_id,
                            "license_files": texts})
        # Include CPython's license and notices shipped inside package directories
        # (e.g. vendored native libraries/fonts), as well as dist-info notices.
        for source in python.rglob("*"):
            if source.is_file() and is_notice(source) and not any(p.endswith(".dist-info") for p in source.parts):
                target = licenses / "python" / kind / "runtime" / source.relative_to(python)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        inventory[kind] = records
    (licenses / "package-licenses.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n")
    for component in native["components"]:
        if component["package"] in versions and versions[component["package"]] != {component["version"]}:
            raise ValueError(f"Native component version changed: {component['package']}")
    sources = matching_sources(root, build, versions)
    # The offline help and DMG expose these texts without entering the .app.
    help_licenses = runtime / "help/licenses"
    if help_licenses.exists():
        shutil.rmtree(help_licenses)
    shutil.copytree(licenses, help_licenses)
    links = []
    for path in sorted(help_licenses.rglob("*")):
        if path.is_file():
            relative = str(path.relative_to(runtime / "help"))
            links.append(f'<li><a href="{html.escape(relative, quote=True)}">{html.escape(str(path.relative_to(help_licenses)))}</a></li>')
    page = ('<!doctype html><html lang="ja"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>TransiroYT ライセンス</title><body><h1>ライセンス</h1>'
            '<p>TransiroYT本体はMITライセンスです。同梱部品と別途取得するモデルには、それぞれの条件が適用されます。</p>'
            '<p><a href="licenses/README-ja.md">適用範囲・モデル条件・対応ソースと改変方法</a></p>'
            '<p><a href="index.html">はじめにへ戻る</a></p><ul>' + "\n".join(links) + '</ul></body></html>')
    (runtime / "help/licenses.html").write_text(page)
    return sources
