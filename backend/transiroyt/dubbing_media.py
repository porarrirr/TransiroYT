"""Media input, PCM track placement and Apple-compatible dubbing output."""
from __future__ import annotations

import array
import json
import math
import sys
import wave
from pathlib import Path

from .audio import SAMPLE_RATE
from .errors import FileDubbingError
from .process import run_process
from .session_models import Segment

# Original speech fades out just before a dubbed line and back after it.
SPEECH_FADE_SECONDS = 0.06
SPEECH_RELEASE_SECONDS = 0.12


async def probe(video: Path) -> dict:
    raw = await run_process("ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json",
                            str(video), timeout=60)
    info = json.loads(raw)
    streams = info.get("streams", [])
    if not any(s.get("codec_type") == "video" for s in streams):
        raise FileDubbingError("動画ファイルに映像トラックがありません。")
    if not any(s.get("codec_type") == "audio" for s in streams):
        raise FileDubbingError("元動画に音声トラックがありません。話者の参照音声を作れません。")
    duration = float(info.get("format", {}).get("duration") or 0)
    if duration <= 0:
        raise FileDubbingError("動画の長さを取得できません。")
    audio = next(s for s in streams if s.get("codec_type") == "audio")
    return {"duration": duration, "channels": int(audio.get("channels") or 2)}


async def extract_audio(video: Path, target: Path, rate: int):
    if target.is_file():
        return
    temporary = target.with_suffix(".tmp.wav")
    await run_process("ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                      "-i", str(video), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(rate),
                      "-c:a", "pcm_s16le", str(temporary), timeout=3600)
    temporary.replace(target)


def place_segments(ready: list[Segment], work: Path, target: Path, duration: float):
    """Write the dub track; windows are sequential, so clips never overlap."""
    frames = math.ceil(duration * SAMPLE_RATE)
    temporary = target.with_suffix(".tmp.wav")
    block_frames = 65536
    silence = bytes(block_frames * 2)
    try:
        with wave.open(str(temporary), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(SAMPLE_RATE)
            output.setnframes(frames)
            cursor = 0
            for segment in sorted(ready, key=lambda item: item.speech_start):
                start = round(segment.speech_start * SAMPLE_RATE)
                if start < cursor:
                    raise FileDubbingError(f"吹き替え音声の区間が重なっています（区間{segment.index}）。")
                with wave.open(str(work / "clips" / f"segment-{segment.index}.wav"), "rb") as clip:
                    if (clip.getnchannels(), clip.getsampwidth(), clip.getframerate()) != (1, 2, SAMPLE_RATE):
                        raise FileDubbingError(f"吹き替え音声の形式が不正です（区間{segment.index}）。")
                    while cursor < min(start, frames):
                        count = min(block_frames, min(start, frames) - cursor)
                        output.writeframesraw(silence[:count * 2])
                        cursor += count
                    remaining = max(0, min(clip.getnframes(), frames - start))
                    while remaining:
                        data = clip.readframes(min(block_frames, remaining))
                        if not data:
                            raise FileDubbingError(f"吹き替え音声が途中で切れています（区間{segment.index}）。")
                        output.writeframesraw(data)
                        remaining -= len(data) // 2
                    cursor = start + clip.getnframes()
            while cursor < frames:
                count = min(block_frames, frames - cursor)
                output.writeframesraw(silence[:count * 2])
                cursor += count
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


GAIN_UNITY = 16384  # The gain track stores gain / 2, so 0..2 fits in 16-bit PCM.


def speech_spans(ready: list[Segment]) -> list[tuple[float, float]]:
    """Where the original speech is replaced: the source line and its dub, joined."""
    spans = []
    for segment in sorted(ready, key=lambda item: min(item.start, item.speech_start)):
        start = min(segment.start, segment.speech_start) - SPEECH_FADE_SECONDS
        end = max(segment.end, segment.speech_end) + SPEECH_RELEASE_SECONDS
        if spans and start <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], end))
        else:
            spans.append((max(0.0, start), end))
    return spans


def write_speech_gain(ready: list[Segment], target: Path, duration: float, volume: float):
    """Gain for the separated original speech: 1 outside dubbed lines, ``volume`` inside.

    Lines that were not dubbed (skipped, or speech without subtitles) keep the
    original voice at full level; linear ramps avoid clicks at the edges.
    """
    frames = math.ceil(duration * SAMPLE_RATE)
    low = round(GAIN_UNITY * volume)
    fade = max(1, round(SPEECH_FADE_SECONDS * SAMPLE_RATE))
    temporary = target.with_suffix(".tmp.wav")
    with wave.open(str(temporary), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        cursor = 0

        def constant(value, until):
            nonlocal cursor
            block = array.array("h", [min(32767, value)]) * 65536
            while cursor < until:
                count = min(until - cursor, len(block))
                wav.writeframes(block[:count].tobytes() if sys.byteorder == "little" else _swapped(block[:count]))
                cursor += count

        def ramp(start_value, end_value, until):
            nonlocal cursor
            count = max(0, until - cursor)
            if count:
                values = array.array("h", (min(32767, round(start_value + (end_value - start_value) * (i + 1) / count))
                                           for i in range(count)))
                wav.writeframes(values.tobytes() if sys.byteorder == "little" else _swapped(values))
                cursor = until

        for start, end in speech_spans(ready):
            a, b = min(frames, round(start * SAMPLE_RATE)), min(frames, round(end * SAMPLE_RATE))
            if b <= a:
                continue
            constant(GAIN_UNITY, max(cursor, a - fade))
            ramp(GAIN_UNITY, low, max(cursor, a))
            constant(low, max(cursor, b))
            ramp(low, GAIN_UNITY, min(frames, max(cursor, b + fade)))
        constant(GAIN_UNITY, frames)
    temporary.replace(target)


def _swapped(values):
    values = array.array("h", values)
    values.byteswap()
    return values.tobytes()


async def video_output_args(video: Path) -> list[str]:
    """Keep Apple-compatible video; convert other codecs for desktop playback."""
    raw = await run_process("ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_streams", "-of", "json", str(video), timeout=60)
    streams = json.loads(raw).get("streams", [])
    if not streams or streams[0].get("disposition", {}).get("attached_pic"):
        raise FileDubbingError("動画ファイルに再生できる映像トラックがありません。")
    stream = streams[0]
    codec, pixels = stream.get("codec_name"), stream.get("pix_fmt")
    if codec == "h264" and pixels in {"yuv420p", "yuvj420p"}:
        return ["-c:v", "copy", "-tag:v", "avc1"]
    if codec == "hevc" and pixels in {"yuv420p", "yuv420p10le"}:
        return ["-c:v", "copy", "-tag:v", "hvc1"]
    # MP4 alone does not imply QuickTime compatibility (notably AV1/VP9).
    # Pad odd dimensions by at most one pixel for 4:2:0 encoding.
    return ["-c:v", "h264_videotoolbox", "-allow_sw", "1", "-q:v", "65", "-pix_fmt", "yuv420p",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-tag:v", "avc1"]


async def playable_video(video: Path, target: Path) -> Path:
    """The source when its video plays on the desktop, otherwise a video-only conversion at ``target``."""
    args = await video_output_args(video)
    if args[:2] == ["-c:v", "copy"]:
        return video
    await run_process("ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                      "-i", str(video), "-map", "0:v:0", *args, str(target), timeout=7200)
    return target


async def mux(video: Path, dub_track: Path, output: Path, dub_volume: float, channels: int = 2, *,
              vocals: Path, background: Path, speech_gain: Path):
    """Write desktop-playable video; background + gained original speech + dub.

    background + vocals is the original soundtrack, so outside dubbed lines the
    original plays unchanged. A mono original stays mono. Otherwise the mono dub
    is placed at unity gain in both channels.
    """
    temporary = output.with_name(f".{output.stem}.partial.mp4")
    video_args = await video_output_args(video)
    if channels == 1:
        layout, mono_to_layout = "aformat=channel_layouts=mono", "aformat=channel_layouts=mono"
    else:
        layout, mono_to_layout = "aformat=channel_layouts=stereo", "pan=stereo|c0=c0|c1=c0"
    mix = (f"[1:a]aresample={SAMPLE_RATE},{layout}[background];"
           f"[2:a]aresample={SAMPLE_RATE},{layout}[speech];"
           f"[3:a]aresample={SAMPLE_RATE},{mono_to_layout}[gain];"
           "[speech][gain]amultiply,volume=2[original];"
           f"[4:a]aresample={SAMPLE_RATE},{mono_to_layout},volume={dub_volume:.4f}[dub];"
           "[background][original][dub]amix=inputs=3:normalize=0:duration=first,"
           "alimiter=limit=0.95:latency=1[mixed]")
    try:
        await run_process("ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                          "-i", str(video), "-i", str(background), "-i", str(vocals), "-i", str(speech_gain),
                          "-i", str(dub_track), "-filter_complex", mix,
                          "-map", "0:v:0", "-map", "[mixed]", *video_args, "-c:a", "aac", "-b:a", "192k",
                          "-map_metadata", "0", "-movflags", "+faststart", str(temporary), timeout=7200)
        await probe(temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
