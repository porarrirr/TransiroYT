# TransiroYT

Apple Silicon Macで動画を日本語に吹き替え、MP4として保存するアプリです。

[Mac版 v0.5.2をダウンロード](https://github.com/porarrirr/TransiroYT/releases/latest)（Developer ID署名・公証済みDMG）

## 使い方

1. アプリをアプリケーションフォルダに移して起動します。
2. 右上メニューの「初回の準備と利用条件」を確認します。
3. YouTube URL、動画、または動画と外部SRTを選びます。
4. 「開始」を押します。不足しているモデルはアプリが自動でダウンロードします。生成したMP4は既定で `~/Downloads/TransiroYT` に保存されます。
5. 完了後は動画の再生、Finderでの表示、区間ごとの処理レポートの確認ができます。

YouTubeでは元言語の字幕（手動・自動）を使います。ライブ配信には対応していません。ローカル動画はUTF-8の外部SRT、またはMP4/MOV/MKVなどのテキスト字幕を使用します。外部SRTを優先します。動画内の字幕が複数ある場合は既定トラックを使用し、一意に決まらない場合は外部SRTを指定します。画像字幕・焼き込み字幕の認識は行いません。字幕がない場合（YouTubeで元言語の字幕がない場合を含む）は、macOS標準の音声認識で作った字幕をWhisperで聞き直して吹き替えます。

音楽・効果音はKim Mel-Band RoFormer（MLX）で声から分離して残し、各話者の声を参照して日本語の音声を作成します。参照音声は、その話者の声で読ませた短い日本語（IndexTTS-2.5）に作り直してから音声生成に使い、聞き取りやすさと声の再現を改善します（別の声になった場合は元の参照に戻します）。参照音声が不足する区間や生成に失敗した区間は元の声を残し、レポートに記録します。出力はMac標準プレーヤーで再生できるMP4です。参照音声用のモデルはbilibili Model Use License Agreement（および補助モデルの非商用条件を含む）に従って利用します。詳細はアプリ内のライセンス表示を確認してください。

## 設定ファイル

通常画面は入力と開始のみです。音量・話者数・YouTube画質・保存先は、右上メニューの「設定…」から変更します。保存した設定は次の開始時に読み込みます。

保存先: `~/Library/Application Support/TransiroYT/product.json`

```json
{
  "outputDirectory": "~/Downloads/TransiroYT",
  "originalVolume": 0,
  "dubVolume": 1,
  "youtubeQuality": "best"
}
```

話者数は省略時に自動判定です。固定する場合は `"speakers": 2` のように追加します。

## 必要環境

Apple Silicon、macOS 27以降が必要です。翻訳はこのMacのPLaMo（英語→日本語）・Gemma、または設定したAPIで処理します。モデル取得・YouTube取得・外部API利用にはネットワークを使用します。Python・FFmpeg・Node・llama.cppと必要なライブラリはアプリに同梱しており、Homebrew・uv・Xcodeなどを利用者がインストールする必要はありません。初回はCore ML音声モデル約2 GB、Whisper約1.6 GB、話者判定・音源分離、参照音声用のIndexTTS-2.5約7.3 GB、PLaMo約1.81 GB、Gemma 4 E4B約7.1 GBを自動取得します（合計約21 GB）。取得済みモデルはアプリの更新後も再利用し、通常のキャッシュ削除では削除しません。動画の作業領域を含め35 GB以上の空きを用意してください。モデルの利用条件はアプリと各配布元で確認してください。

APIキーはmacOSのキーチェーンに保存し、設定JSONやリポジトリには含めません。CLIの使い方は[ガイド](guide/cli.html)を参照してください。

## 開発

iPhone／iPadをMacの音声生成ワーカーとして使う[TransiroYT Mobile](apple/Mobile/README.md)のソースも含みます。
Mac v0.5.2との連携に対応しています。iOS版はXcodeでビルドして実機へ導入します。
App Store／TestFlightでの配布は行っていません。

- `backend/transiroyt/`: 吹き替え・翻訳・話者判定などの既存エンジン
- `apple/Desktop/`: macOSのSwiftUIアプリ
- `apple/Mobile/`: iPhone／iPadのSwiftUIアプリ
- `apple/Inference/`: MacとiOSで共有するCore ML音声生成
- `scripts/`: セットアップ、モデルワーカー、ビルドと配布検証
- `distribution/`: 依存・モデルの固定情報、原文ライセンス、再ビルド情報
- `tests/`: エンジン・配布・接続の回帰テスト

```sh
./scripts/setup.sh --with-tts
./scripts/build_desktop.sh
uv run pytest -q
swift test --package-path apple/Desktop --scratch-path .local/DesktopTests
uv run ruff check backend scripts tests
```

翻訳用のApple独自権限は不要です。モデル・設定・トークンはGitに含めません。配布アプリは利用者のApplication Supportにデータを保持します。

ローカルの署名付きpreview:

```sh
uv run python scripts/build_dmg.py --channel preview \
  --build-dir .local/product-build --output-dir .local/product-dist
```

正式公開ではランタイムを再利用せず、Developer ID署名と公証を必須にします。

```sh
uv run python scripts/build_dmg.py --channel release \
  --sign-identity 'Developer ID Application: YOUR NAME (TEAMID)' \
  --notary-profile YOUR_KEYCHAIN_PROFILE
```

配布先はGitHub Releasesです。実際に公開する前にモデルの利用・再配布条件を確認してください。

配布アプリには自動更新機能があります。yt-dlpも検証済みの更新版アプリと一緒に更新します。
DMGとソースをGitHub Releasesで公開し、自動更新も配信します。[更新と配信手順](docs/auto-updates.md)を参照してください。

## ライセンス

独自コードは[MIT](LICENSE)です。第三者ライブラリ・モデルは別の条件が適用されます。[ライセンス一覧](docs/licenses.md)、`distribution/THIRD_PARTY_NOTICES.txt`、`distribution/licenses/`、[LGPL部品の再ビルド情報](distribution/REBUILDING-LGPL.md)を保持してください。
