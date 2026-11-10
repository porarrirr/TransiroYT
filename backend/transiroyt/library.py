"""Completed exports and the regenerable working files owned by file dubbing."""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

CACHE_DIRECTORIES = {
    "file-dubbing": "音声分析・翻訳・生成音声",
    "file-dubbing-downloads": "取得した動画・字幕",
    "file-dubbing-transcripts": "音声から作成した字幕",
    "reference-conversion": "調整済みの参照音声",
}
CACHE_MAX_AGE = 7 * 24 * 60 * 60
CACHE_MAX_BYTES = 30_000_000_000


class ExportReport(BaseModel):
    version: str
    output: str
    title: str | None = None
    uploader: str | None = None
    thumbnail_url: str | None = None
    duration: float = Field(default=0, ge=0, allow_inf_nan=False)


def directory_bytes(path: Path) -> int:
    """Count files once, without following symlinks out to models or source media."""
    if path.is_symlink() or not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            item = Path(root) / name
            if not item.is_symlink():
                try:
                    total += item.stat().st_size
                except FileNotFoundError:
                    pass  # A running job can replace its temporary files during a refresh.
    return total


class DubbingLibrary:
    def __init__(self, data: Path):
        self.data = data
        self.index = data / "library-directories.json"

    def directories(self):
        if not self.index.exists():
            return []
        return [Path(value) for value in TypeAdapter(list[str]).validate_json(self.index.read_bytes())]

    def remember(self, directory: Path):
        directory = directory.expanduser().resolve()
        directories = self.directories()
        if directory in directories:
            return
        directories.append(directory)
        self.data.mkdir(parents=True, exist_ok=True)
        temporary = self.index.with_suffix(".tmp")
        temporary.write_text(json.dumps([str(item) for item in directories], ensure_ascii=False))
        temporary.replace(self.index)

    def snapshot(self):
        entries, warnings = [], []
        for directory in self.directories():
            try:
                reports = list(directory.glob("*.dubbing.json"))
                for path in reports:
                    try:
                        report = ExportReport.model_validate_json(path.read_bytes())
                        output = Path(report.output).expanduser().resolve()
                        if report.version != "file-dubbing-v1" or output != path.with_suffix("").with_suffix(".mp4").resolve():
                            continue
                        exists = output.is_file()
                        entries.append({"id": str(output), "title": report.title or output.stem,
                                        "output": str(output), "created_at": path.stat().st_mtime,
                                        "uploader": report.uploader, "thumbnail_url": report.thumbnail_url,
                                        "duration": report.duration, "available": exists,
                                        "size_bytes": output.stat().st_size if exists else 0})
                    except (OSError, ValidationError):
                        warnings.append(f"レポートを読み込めません: {path.name}")
            except OSError:
                warnings.append(f"保存先を読み込めません: {directory}")
        entries.sort(key=lambda item: item["created_at"], reverse=True)
        return {"entries": entries, "total_bytes": sum(item["size_bytes"] for item in entries), "warnings": warnings}

    def cache_status(self):
        categories = [{"id": name, "title": title, "size_bytes": directory_bytes(self.data / name)}
                      for name, title in CACHE_DIRECTORIES.items()]
        return {"categories": categories, "total_bytes": sum(item["size_bytes"] for item in categories)}

    def clear_cache(self):
        self._check_cache_exports()
        for name in CACHE_DIRECTORIES:
            path = self.data / name
            if path.is_symlink() or path.is_file():
                path.unlink()
            elif path.exists():
                shutil.rmtree(path)
        return self.cache_status()

    def prune_cache(self):
        """Expire complete work units, then evict oldest units to meet the size limit."""
        self._check_cache_exports()
        entries = []
        for name in CACHE_DIRECTORIES:
            root = self.data / name
            if root.is_symlink() or not root.is_dir():
                continue
            groups = {}
            for path in root.iterdir():
                # Converted references have a WAV and a JSON sidecar; evict both together.
                key = path.stem if name == "reference-conversion" else path.name
                groups.setdefault(key, []).append(path)
            for paths in groups.values():
                size, modified = 0, 0
                for path in paths:
                    size += directory_bytes(path)
                    modified = max(modified, path.lstat().st_mtime)
                    if path.is_dir() and not path.is_symlink():
                        for directory, _, files in os.walk(path):
                            modified = max(modified, Path(directory).stat().st_mtime)
                            for filename in files:
                                modified = max(modified, (Path(directory) / filename).lstat().st_mtime)
                entries.append((modified, size, paths))
        total = sum(size for _, size, _ in entries)
        cutoff = time.time() - CACHE_MAX_AGE
        for modified, size, paths in sorted(entries, key=lambda entry: entry[0]):
            if modified > cutoff and total <= CACHE_MAX_BYTES:
                break
            for path in paths:
                if path.is_symlink() or path.is_file():
                    path.unlink()
                else:
                    shutil.rmtree(path)
            total -= size

    def _check_cache_exports(self):
        # A custom export destination can be anywhere. Never clear it as working data.
        for directory in self.directories():
            if (any(directory.is_relative_to((self.data / name).resolve()) for name in CACHE_DIRECTORIES)
                    and any(directory.glob("*.dubbing.json"))):
                raise ValueError("完成動画の保存先がキャッシュ内にあります。動画とレポートを別の場所へ移動してください。")
