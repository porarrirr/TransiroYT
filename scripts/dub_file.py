"""Dub a downloaded MP4 with its SRT subtitles into a new MP4.

Without an SRT, the subtitles are transcribed from the speech (macOS, reviewed by Whisper).

Example:
  uv run python scripts/dub_file.py video.mp4 video.en.srt -o video.ja.mp4 \
      --original-volume 0 --dub-volume 1.0
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from transiroyt.config import Settings  # noqa: E402
from transiroyt.dubbing_options import FileDubbingOptions  # noqa: E402
from transiroyt.errors import FileDubbingError  # noqa: E402
from transiroyt.file_dubbing import FileDubber  # noqa: E402
from transiroyt.gemma_audio import GemmaUnavailable  # noqa: E402
from transiroyt.logging_utils import configure_logging  # noqa: E402
from transiroyt.subtitle_files import SubtitleError  # noqa: E402
from transiroyt.vocal_separation import SeparationUnavailable  # noqa: E402
from transiroyt.whisper_transcription import WhisperUnavailable  # noqa: E402
from transiroyt.youtube_file import YOUTUBE_QUALITIES, download_youtube, youtube_video_id  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="MP4とSRTから吹き替え済みMP4を作成します。")
    parser.add_argument("video", type=Path, nargs="?", help="元の動画（MP4）")
    parser.add_argument("subtitles", type=Path, nargs="?", help="字幕（SRT、UTF-8）。省略時は音声から書き起こす")
    parser.add_argument("--youtube-url", help="動画と字幕を取得するYouTube URL")
    parser.add_argument("--youtube-quality", choices=YOUTUBE_QUALITIES, default="best",
                        help="YouTube動画の上限画質（既定: best＝最高画質）")
    parser.add_argument("-o", "--output", type=Path, help="出力MP4（既定: 元の名前.ja-dub.mp4）")
    parser.add_argument("--original-volume", type=float, default=0.0,
                        help="吹き替えた区間に残す元の声の音量（0〜2、既定0）。音楽・効果音は元の音量のまま")
    parser.add_argument("--dub-volume", type=float, default=1.0, help="吹き替え音声の音量（0〜2、既定1.0）")
    parser.add_argument("--language", default="auto", help="字幕の言語コード（既定: Gemmaまたは設定したAPIで判定）")
    parser.add_argument("--speakers", type=int, help="話者数が分かっている場合に指定")
    parser.add_argument("--title", help="動画のタイトル（翻訳の文脈。YouTube URLでは自動）")
    parser.add_argument("--steps", type=int, help="音声生成のステップ数（既定10）")
    parser.add_argument("--download-models", action="store_true",
                        help="未取得のGemma 4 E4B（約7.1GB）・Whisper・話者照合・音源分離モデルのダウンロードを許可")
    parser.add_argument("--overwrite", action="store_true", help="既存の出力ファイルを上書き")
    args = parser.parse_args()
    if args.youtube_url:
        if args.video or args.subtitles:
            parser.error("--youtube-urlと動画・字幕ファイルは同時に指定できません。")
        try:
            video_id = youtube_video_id(args.youtube_url)
        except FileDubbingError as exc:
            parser.error(str(exc))
        output = args.output or Path(f"{video_id}.ja-dub.mp4")
    else:
        if not args.video:
            parser.error("動画、または--youtube-urlを指定してください。")
        output = args.output or args.video.with_name(f"{args.video.stem}.ja-dub.mp4")
    if output.exists() and not args.overwrite:
        print(f"出力ファイルが既にあります: {output}（上書きするには --overwrite）", file=sys.stderr)
        return 2
    configure_logging()
    settings = Settings.from_env()
    import uuid
    directory = settings.data / "file-dubbing-downloads" / uuid.uuid4().hex
    if args.youtube_url:
        video, subtitles = directory / "source.mp4", directory / "source.srt"
    else:
        video, subtitles = args.video, args.subtitles or directory / "transcribed.srt"
    options = FileDubbingOptions(
        video=video, subtitles=subtitles, output=output,
        original_volume=args.original_volume, dub_volume=args.dub_volume,
        source_language=args.language, speakers=args.speakers, steps=args.steps,
        allow_download=args.download_models, title=args.title,
        transcribe=not args.youtube_url and args.subtitles is None)

    def progress(stage, fraction, message):
        print(f"[{stage} {fraction * 100:5.1f}%] {message}", file=sys.stderr, flush=True)

    def download_progress(update):
        total = update.get("total")
        fraction = min(1.0, update["downloaded"] / total) if total else 0
        label = {"youtube": "動画・字幕", "coreml": "Core ML音声モデル", "gemma": "Gemmaローカル代替", "speaker_model": "話者照合モデル", "diarization_model": "話者区間モデル（Nemotron）",
                 "separation_model": "音源分離モデル", "speech_check_model": "音声確認モデル（Whisper）"}[update["resource"]]
        detail = f"{update['downloaded']:,}" + (f" / {total:,}" if total else "") + " bytes"
        progress("download", fraction, f"{label}: {detail}（{update['state']}）")

    async def run():
        if args.youtube_url:
            options.validate(require_inputs=False)
            progress("download", 0, "YouTubeの動画と字幕をダウンロードしています")
            metadata = await download_youtube(settings, args.youtube_url, video.parent, args.language,
                                              quality=args.youtube_quality, progress=download_progress)
            if not metadata["subtitles"]:
                options.transcribe = True
            elif options.source_language == "auto":
                options.source_language = metadata["language"]
            options.title = args.title or metadata.get("title")
        return await FileDubber(settings, options, progress, download_progress=download_progress).run()

    try:
        report = asyncio.run(run())
    except (FileDubbingError, SubtitleError, GemmaUnavailable, SeparationUnavailable, WhisperUnavailable) as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1
    summary = report["summary"]
    print(f"出力: {output}")
    print(f"レポート: {output.with_suffix('.dubbing.json')}")
    print(f"区間: {summary['ready']}/{summary['segments']} を吹き替え、{summary['skipped']} を元の音声のまま")
    for name, speaker in report["speakers"].items():
        state = f"参照 {speaker['duration']:.1f}秒" if speaker.get("reference") else f"参照なし: {speaker['error']}"
        print(f"  {name}: {state}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
