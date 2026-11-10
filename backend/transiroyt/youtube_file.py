"""Download one YouTube video and original captions for offline dubbing.

A video without original-language captions is downloaded alone; the file job
then transcribes its speech.

Extraction runs in a child process so cancellation also stops network work.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .captions import CaptionError, _youtube_options, parse_json3, select_track
from .download_progress import run_download, worker_args, write_progress
from .errors import FileDubbingError
from .process import ProcessError
from .subtitle_files import format_srt
from .youtube_access import FALLBACK_CLIENT, SIGN_IN_MESSAGE, needs_sign_in

GENERIC_MESSAGE = "YouTubeの動画・字幕を取得できませんでした。時間をおいて再試行してください。"
RATE_LIMIT_MESSAGE = (
    "YouTubeが字幕・動画の取得を制限しています（HTTP 429）。"
    "時間をおいて再試行してください。制限が続く場合は、動画ファイルとSRT字幕を指定してください。"
)


def youtube_video_id(url: str) -> str:
    try:
        parsed = urlparse(url.strip())
        host = (parsed.hostname or "").lower()
        if parsed.scheme not in {"https", "http"} or parsed.username or parsed.password or parsed.port:
            raise ValueError
        if host in {"youtu.be", "www.youtu.be"}:
            video_id = parsed.path.removeprefix("/")
        elif host in {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"}:
            if parsed.path == "/watch":
                video_id = parse_qs(parsed.query).get("v", [""])[0]
            else:
                match = re.fullmatch(r"/(?:shorts|embed|live)/([A-Za-z0-9_-]{11})/?", parsed.path)
                video_id = match[1] if match else ""
        else:
            raise ValueError
        if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
            raise ValueError
        return video_id
    except ValueError as exc:
        raise FileDubbingError("YouTubeの動画URLを指定してください。") from exc


class _QuietLogger:
    # yt-dlp errors can contain signed media URLs; the worker reports safe errors.
    def debug(self, message):
        pass

    info = debug
    warning = debug
    error = debug


class VideoDownloadProgress:
    def __init__(self, path):
        self.path = path
        self.downloaded, self.totals = {}, {}
        self.last_report = 0.0
        self.speed = None

    def report(self, state, *, force=False):
        import time
        if not force and time.monotonic() - self.last_report < 0.2:
            return
        total = sum(self.totals.values()) if self.totals and all(self.totals.values()) else None
        write_progress(self.path, {"resource": "youtube", "state": state,
                       "downloaded": sum(self.downloaded.values()), "total": total,
                       "speed": self.speed if state == "downloading" else None,
                       "eta": max(0, total - sum(self.downloaded.values())) / self.speed
                       if total and self.speed and state == "downloading" else None})
        self.last_report = time.monotonic()

    def formats(self, info):
        for item in info.get("requested_formats") or [info]:
            key = item.get("format_id", "media")
            self.totals[key] = item.get("filesize") or item.get("filesize_approx")
            self.downloaded[key] = 0
        self.report("downloading", force=True)

    def hook(self, status):
        self.speed = status.get("speed")
        key = status.get("info_dict", {}).get("format_id", "media")
        self.downloaded[key] = status.get("downloaded_bytes", 0)
        self.totals[key] = status.get("total_bytes") or status.get("total_bytes_estimate") or self.totals.get(key)
        if status["status"] == "finished" and self.downloaded[key]:
            self.totals[key] = self.downloaded[key]
        self.report("downloading", force=status["status"] == "finished")


YOUTUBE_QUALITIES = ("best", "2160", "1440", "1080", "720", "480", "360")


def video_format(quality):
    if quality not in YOUTUBE_QUALITIES:
        raise FileDubbingError("YouTubeの画質を選び直してください。")
    if quality == "best":
        return "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b"
    limit = f"[height<={quality}]"
    # Apply the ceiling to every video fallback; audio quality is independent.
    return f"bv*{limit}+ba/b{limit}"


def download_once(settings, video_id, directory, language, client=None, *, quality="best"):
    import yt_dlp

    progress = VideoDownloadProgress(directory / "download-progress.json")
    progress.report("checking", force=True)
    options = _youtube_options(settings, client)
    options.update({"skip_download": False, "format": video_format(quality),
                    "merge_output_format": "mp4", "logger": _QuietLogger(), "progress_hooks": [progress.hook], "outtmpl": str(directory / "source.%(ext)s"),
                    "postprocessors": [{"key": "FFmpegVideoRemuxer", "preferedformat": "mp4"}]})
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
        if info.get("is_live") or info.get("live_status") in {"is_live", "is_upcoming", "post_live"}:
            raise CaptionError("生配信には未対応です。字幕が確定した通常動画を指定してください。")
        try:
            lang, url, _ = select_track(info, language)
        except CaptionError:
            # No original-language captions: the file job transcribes the speech.
            lang = cues = None
        else:
            with ydl.urlopen(url) as response:
                raw = response.read(20 * 1024 * 1024)
            try:
                cues = parse_json3(json.loads(raw))
            except (ValueError, TypeError, KeyError) as exc:
                raise CaptionError("YouTubeの字幕データを読み取れませんでした。") from exc
            if not cues:
                raise CaptionError("YouTubeから空の字幕が返されました。")
        # Fetch captions before downloading a potentially large video.
        progress.formats(info)
        ydl.process_info(info)
        progress.report("processing", force=True)
        video = directory / "source.mp4"
        if not video.is_file():
            raise FileDubbingError("YouTubeの動画をMP4として保存できませんでした。")
        if cues:
            (directory / "source.srt").write_text(format_srt(cues), encoding="utf-8")
        progress.report("complete", force=True)
        return {"language": lang and lang.removesuffix("-orig"), "subtitles": cues is not None,
                "title": info.get("title", video_id),
                "uploader": info.get("channel") or info.get("uploader"),
                "thumbnail_url": info.get("thumbnail")}


async def download_youtube(settings, url, directory, language="auto", *, quality="best", progress=None):
    video_id = youtube_video_id(url)
    video_format(quality)
    directory.mkdir(parents=True, exist_ok=True)
    args = worker_args("transiroyt.youtube_file", video_id, directory,
                       language, settings.root, settings.cookies_browser or "", quality)
    try:
        await run_download(args, directory / "download-progress.json", progress)
    except ProcessError as exc:
        # Worker only writes a user-facing error on stderr; never expose signed URLs.
        raise FileDubbingError(str(exc)) from exc
    except TimeoutError as exc:
        raise FileDubbingError("YouTubeの取得が制限時間を超えました。再試行してください。") from exc
    return json.loads((directory / "metadata.json").read_text())


def main():
    import yt_dlp
    from yt_dlp.networking.exceptions import HTTPError

    from .config import Settings

    video_id, directory, language, root, cookies, quality = sys.argv[1:]
    settings = Settings(root=Path(root), cookies_browser=cookies or None)
    try:
        youtube_video_id(f"https://www.youtube.com/watch?v={video_id}")
        try:
            metadata = download_once(settings, video_id, Path(directory), language, quality=quality)
        except yt_dlp.utils.YoutubeDLError as exc:
            if not needs_sign_in(exc):
                raise FileDubbingError(_download_error_message(exc)) from exc
            try:
                metadata = download_once(settings, video_id, Path(directory), language, FALLBACK_CLIENT, quality=quality)
            except yt_dlp.utils.YoutubeDLError as fallback:
                raise FileDubbingError(_download_error_message(fallback)) from fallback
        (Path(directory) / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False))
    except (FileDubbingError, CaptionError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except HTTPError as exc:
        print(RATE_LIMIT_MESSAGE if exc.status == 429 else GENERIC_MESSAGE, file=sys.stderr)
        return 1
    except Exception:
        print(GENERIC_MESSAGE, file=sys.stderr)
        return 1
    return 0


def _download_error_message(error):
    # yt-dlp wraps media download HTTP failures in DownloadError; caption
    # requests made through urlopen raise HTTPError directly instead.
    if re.search(r"\bHTTP Error 429\b", str(error)):
        return RATE_LIMIT_MESSAGE
    return SIGN_IN_MESSAGE if needs_sign_in(error) else GENERIC_MESSAGE


if __name__ == "__main__":
    sys.exit(main())
