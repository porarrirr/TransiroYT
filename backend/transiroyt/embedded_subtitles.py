"""Extract a selected text subtitle track through the shared FFmpeg runner."""
from __future__ import annotations

import json

from .errors import FileDubbingError
from .process import ProcessError, run_process
from .subtitle_files import parse_srt

TEXT_CODECS = {"subrip", "mov_text", "ass", "ssa", "webvtt", "text"}


async def extract_subtitles(video, output) -> bool:
    """Write the video's text subtitles to ``output``; False when it has none to use."""
    raw = await run_process("ffprobe", "-v", "error", "-select_streams", "s",
                            "-show_streams", "-of", "json", video)
    tracks = json.loads(raw)["streams"]
    usable = [track for track in tracks if track["codec_name"] in TEXT_CODECS
              and not track.get("disposition", {}).get("forced", 0)]
    if not usable:
        return False
    # A default track is the container author's selection. Ambiguous containers
    # need an explicit external SRT rather than silently choosing a language.
    defaults = [track for track in usable if track.get("disposition", {}).get("default", 0)]
    candidates = defaults or usable
    if len(candidates) != 1:
        raise FileDubbingError("動画に複数の字幕があります。使用する元の言語の字幕をSRTファイルとして指定してください。")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        await run_process("ffmpeg", "-v", "error", "-y", "-i", video,
                          "-map", f"0:{candidates[0]['index']}", "-c:s", "srt", output)
    except ProcessError as exc:
        raise FileDubbingError("動画内の字幕を取り出せませんでした。外部SRTファイルを指定してください。") from exc
    parse_srt(output.read_text(encoding="utf-8-sig"))
    return True
