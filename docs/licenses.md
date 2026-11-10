# ライセンス

TransiroYTの独自コードとドキュメントは[MITライセンス](../LICENSE)で提供します。
著作権表示とライセンス本文を保持すれば、改変・再配布・商用利用が可能です。
無保証での提供となります。第三者のコード・ライブラリ・モデル・商標・動画・字幕・参照音声には、
それぞれの権利者の条件が適用されます。本体のMITライセンスでこれらを再許諾するものではありません。

## 同梱ソフトウェア

| 対象 | ライセンス・配布方法 |
| --- | --- |
| Rust標準ライブラリ | Rust音声カーネルに静的リンク。MIT / Apache-2.0等の原文と依存部品の帰属を `distribution/licenses/rust/` に保持 |
| TransiroYT本体 | MIT。本文をアプリとDMG内の案内に同梱 |
| llama.cpp（リビジョン8ca4060b7e30e76baaa5670ee1ec1b760b7f8552） | MITおよび同梱第三者部品の条件。原文・vendorソース・固定リビジョンとSHA-256をアプリに保持し、元ソースを対応ソースZIPに同梱。Metalカーネルを内蔵した静的リンク構成 |
| Sparkle 2.10.1 | MITおよび同梱第三者部品の条件。SPMで固定し、原文ライセンスとフレームワーク内の表示を保持 |
| Irodori-TTSコード | MIT。上流の著作権表示と全文を保持 |
| IndexTTS-2.5コード（index-tts、リビジョン `d9e41aac89fd00b3d71497fddb287b7f24613712`） | **bilibili Model Use License Agreement**。全文を `Runtime/licenses/IndexTTS-bilibili-LICENSE.txt` に同梱し、モデルカードと原文の写しも保持。重みは同ライセンスで別途ダウンロード |
| DACVAEコード | Apache-2.0。Metaの著作権表示・ライセンスを保持 |
| dav1d 1.5.3 | BSD-2-Clause。AV1デコード用。固定SHA-256の公式ソースからビルドし、著作権表示・ライセンス原文・ソース・ビルド手順を同梱 |
| FFmpeg 8.1.1 | LGPL-2.1-or-later。GPL/nonfree/version3を無効にした共有ライブラリ構成。対応ソース・ビルドスクリプト・構成フラグを別添ZIPで提供 |
| Python-SoXR 1.0.0 / libsoxr | LGPL-2.1-or-later等。Python拡張とlibsoxrを含む対応ソース、ライセンス全文、ビルド手順を別添ZIPで提供 |
| SoundFile内のlibsndfile 1.2.2、mpg123 1.32.3、LAME 3.100 | LGPL。静的に組み込まれたXiph系コーデックも含め、対応ソースと上流のmacOSビルドスクリプトを別添ZIPで提供 |
| SciPy内のlibquadmath | LGPL-2.1-or-later。ビルド情報のGCC 13.4.0に対応するソース・再ビルド情報を別添ZIPで提供。libgcc/libgfortranのGCC Runtime Library Exceptionも保持 |
| autocommand 2.2.2（依存環境に含まれるビルドのみ） | LGPL-3.0。含まれる場合はPythonソースをアプリに保持し、対応ソース、LGPLv3とGPLv3の本文を別添ZIPで提供 |
| certifi・tqdm | MPL-2.0等。ライセンス表示とPythonソースを保持。各ファイルの条件に従う |
| CPython（python-api 3.12、python-tts・python-indextts 3.11）/ Node.js | PSF / MITおよび各々に含まれる第三者ライセンス。原文を保持 |
| その他のPythonパッケージ・同梱データ | ビルドごとの `package-licenses.json` に名前・バージョン・ライセンス本文の保存先を記録。参照音声用の `python-indextts` は `dependency-inventory.json` の `indextts` に一覧し、原文は同梱の各 `*.dist-info` に保持 |

元の配布パッケージで本文が欠けている場合は `distribution/licenses/` の原文で補います。
原文の入手先・対象バージョン・SHA-256は同フォルダの `manifest.json` に記録します。
CPythonの静的依存（OpenSSL、libedit、libffi、bzip2、xz、SQLite、Tcl/Tk等）も、
各ランタイムのビルド日に対応するpython-build-standaloneの原文で補います。
macOSビルドのdbmはOSのndbmを使用し、他プラットフォーム向けのBerkeley DBは同梱しません。
上流のパッケージを再ライセンスしません。音楽タグ埋め込み用の任意依存Mutagen（GPL）は同梱しません。
字幕取得と参照音声抽出ではこの機能を使用しません。

配布アプリでは「使い方」のページ末尾にある「ライセンス」から本文を確認できます。
DMG内の「はじめに/licenses.html」からも開けます。アプリ内部の保存先は
`TransiroYT.app/Contents/Resources/Runtime/licenses/` です。

## 別途ダウンロードするモデル

モデルの重みはDMG・対応ソースZIPに含めません。固定したリビジョンの
モデルカードと利用条件を `Runtime/licenses/supplemental/models/` に保存します。
使用するモデルは `models.lock.json` と `fallback-models.lock.json` で固定します。

- [Irodori-TTS-v4.1-Small-MF](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small-MF)：モデルカードはMITとし、追加の倫理的制限として、本人の明示的同意のない声の複製・なりすまし、誤認を目的とした偽情報等を禁止しています。この追加条件も確認してください。
- [Semantic-DACVAE-Japanese-32dim](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim)：モデルカードのMIT表記と元モデルへの帰属を保持します。
- [SilentCipher](https://huggingface.co/Sony/SilentCipher)：モデルカードを同梱します。コードのMIT本文はSonyの著作権表示とともに保持します。現在の音声生成経路ではSilentCipherの透かし処理を使用しません。

- [Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)：ファイル吹き替えの生成音声の書き起こし確認に使います。OpenAIの重み（MIT）をMLX形式に変換した `mlx-community/whisper-large-v3-turbo`（リビジョン `a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb`）を、利用者がダウンロードを許可したときだけ取得します。実行コードのmlx-whisper（MIT、OpenAI WhisperのMITコードを含む）は両方の原文を同梱します。

- [話者区間モデル Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization)：ファイル吹き替えで、各台詞を誰が話しているかの判定に使います。固定リビジョン `f667ed73aee57d40cc39428eb768b4fd87a0a29e` の重み（約400 MB）を、利用者がダウンロードを許可したときだけHugging Faceから取得します（DMGには含めません）。**OpenMDW License Agreement 1.1**（https://openmdw.ai/license/1-1/ 。利用・改変・再配布・商用利用を制限なく許可し、配布する場合は契約書の写しと出所の表示を残す。モデルについて特許・著作権の訴訟を起こすと権利が終了する）。モデルカードを同梱します。実行に使うtransformers（Apache-2.0）は同梱ランタイムの原文に含まれます。

- [IndexTTS-2.5](https://huggingface.co/IndexTeam/IndexTTS-2.5)（リビジョン `c39ce5ba981572cb187443877ff559dfb246ce63`）: ファイル吹き替えで参照音声を作り直す0.8BのTTS。本体と補助モデルを合わせた重み約7.3GBはDMGに含めず、利用者がダウンロードを許可したときだけHugging Faceの公開リポジトリから取得し、`Runtime/models.lock.json` の固定リビジョンとサイズで確認します。**bilibili Model Use License Agreement**が適用されます（モデルカード・原文の写しは `Runtime/licenses/supplemental/models/`、コードと重みで同じ条項）。日本語での要点は次のとおりです。

  - 直近1か月の製品・サービスの月間アクティブ利用者が1億人を超える場合、または前年度の年間収入が10億元（人民元）を超える場合は、別途ライセンスの書面による許諾が必要です（2.2）。許諾があるまで本契約の権利を行使できません。
  - モデルまたは派生物を複製・配布するときは、元の著作権表示と本契約の写しを保持し、下流の受領者にも同じ条件を課す必要があります（3.4a、3.4b）。本アプリは重みを改変せずにそのまま使います。
  - bilibili indextts2およびその派生物を、それ自身・その派生物・非商用のAIモデルの改善以外のAIモデル改善に使用できません（3.4c）。
  - 医療診断、自動運転、軍事、重要インフラの制御、大規模な生体監視、与信・採用等の自動意思決定といった高リスク用途に配備してはいけません（4.2）。
  - 利用地域の法令を守り、第三者のデータや重みを持ち込む場合は自ら必要な権利を取得する責任があります（4.1）。出力内容の適法性も利用者の責任です（4.1c、4.1d）。
  - 無保証・責任限定（3.1〜3.3）。準拠法は中国法で、紛争は上海仲裁委員会の仲裁によります（6）。英語版と中国語版が矛盾する場合は中国語版が優先します（9）。
  - 補助モデルはそれぞれの条件に従います。[w2v-bert-2.0](https://huggingface.co/facebook/w2v-bert-2.0)（リビジョン `da985ba0987f70aaeb84a80f2851cfac8c697a7b`、MIT）、[BigVGAN](https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_256x)（`633ff708ed5b74903e86ff1298cf4a98e921c513`、MIT）、[CAMPPlus](https://huggingface.co/funasr/campplus)（`e4b6ede7ce16997aff4ae69fbca1f0175e2afede`、Apache-2.0）、[MaskGCT semantic codec](https://huggingface.co/amphion/MaskGCT)（`265c6cef07625665d0c28d2faafb1415562379dc`）。**MaskGCTのモデルカードはCC-BY-NC-4.0（非商用）です。商用配布・商用利用の前に権利者への確認か代替部品への置き換えが必要です。** 感情記述用のQwenEmotion（`qwen0.6bemo4-merge`）は使わないためダウンロードしません。
  - IndexTTS-2.5が宣言する依存パッケージのうち、GPLのコードを含み推論経路で使わないものは配布物に含めません。opencv-python（macOS wheelがGPLのFFmpegコーデック libx264/libx265 を同梱）、Distance（GPL。g2p_en専用）、g2p_en・textstat・pyphen（未使用の `text_utils` ヘルパー専用。pyphenはGPL/LGPL/MPLのデュアルライセンス）です。日本語の推論経路はこれらを読み込みません。`Runtime/licenses/dependency-inventory.json` の `indextts` に実際に同梱したパッケージを記録します。

- [ファイル吹き替えの音源分離モデル](https://huggingface.co/KimberleyJSN/melbandroformer/tree/ac9b0614ab3cd7f77219e18ba494dfd93956c348)（Kim Mel-Band RoFormer、MIT、`MelBandRoformer.ckpt`、913,106,900バイト、SHA-256 `87201f4d31afb5bc79993230fc49446918425574db48c01c405e44f365c7559e`）：利用者が取得を許可したときだけ固定revisionから取得し、ハッシュを確認します。作者が2026年4月にMITへ変更したモデルカード原文を同梱します。推論はmlx-audio 0.6.5（MIT）の必要なモデル/DSP部分を取り込み、原文と変更元を `backend/transiroyt/mel_roformer/` に保持。

モデルの許諾は、他人の声・動画・字幕を使用する許可を意味しません。
参照音声は使用権限と本人の同意を得たものを使用してください。

## LGPL部品の改変と対応ソース

DMGと同じリリースの `TransiroYT-0.5.0-third-party-sources.zip` に、
FFmpeg、Python-SoXR/libsoxr、libsndfileと内包コーデック、libquadmath、および依存環境に含まれる場合はautocommandの対応ソースと再ビルド手順を含めます。
これらは上流ソースを改変せずに使用します。ビルド構成・依存の再配置・配布署名は実施します。
同じZIPに原文ライセンス、実際のパッケージ一覧、入手元とハッシュも含めます。
SoundFileの同梱ネイティブライブラリは上流の固定コミットのバイナリとの一致を確認しています。
SoundFileとSciPyの対象ライブラリは入力ハッシュを記録し、変更時には対応ソースの再確認を必須とします。
ファイル吹き替えの参照音声用Python 3.11環境（`python-indextts`）も、`python-tts` と同じSoundFile・libsoxr・SciPyのwheelを使うため、上記の対応ソースと再ビルド手順がそのまま対応します。

LGPL部品の改変・互換ライブラリへの差し替えと、その改変をデバッグするための
リバースエンジニアリングを禁止しません。手順は `REBUILDING-LGPL.md` を参照してください。
アプリを再配布するときは、本体のMIT本文に加え、同梱部品の表示・対応ソースの提供も維持してください。

根拠： [MIT](https://opensource.org/license/mit)、[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)、
[LGPL-2.1](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html)、[LGPL-3.0](https://www.gnu.org/licenses/lgpl-3.0.html)、
[MPL-2.0](https://www.mozilla.org/en-US/MPL/2.0/)。

## モバイル連携・共通Core ML音声生成

`apple/Inference/Vendor/IrodoriTTS` は [Corvelis/irodori-tts-coreml](https://github.com/Corvelis/irodori-tts-coreml/tree/948f17da32e94fdb87eeb9420b76326312665d49) の固定ソース。新規SDK部分はApache-2.0、上流由来のモデル・実装は部品別のMIT/Apache-2.0。原文・NOTICE・変更点は同ディレクトリに保持し、Mac配布物では `Runtime/licenses/mobile-inference/`、iOSではアプリの `Notices/` に同梱する。モデル取得物にも各部品の原文と変換来歴を保持する。AudioSealのモデルは原本として残すが、本アプリのファイル生成ではロード・付与しない。

iOSアプリは音声生成だけを担当し、llama.cppとテキストモデルを同梱しない（2026-10-31まで同梱していたLFM3とllama.cppは削除）。再ビルドは `scripts/build_mobile.sh`、接続方式は `apple/Mobile/Sources/WorkerConnection.swift` を参照。

- [ローカル代替のGemma 4 E4B-it](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF)：Unslothによる量子化の固定リビジョン `bfc15c382204943c3a8fff0c750b94ae2364d7a3` のQ6_Kを使用。Gemma 4は [Apache-2.0](https://ai.google.dev/gemma/docs/gemma_4_license)。Apache-2.0の原文とモデルカードを同梱する。

## PLaMo翻訳モデル

Built with PLaMo. 標準のローカル翻訳は [PLaMo 3 Translate 2B GGUF](https://huggingface.co/pfnet/plamo-3-translate-2b-GGUF) の固定版 `f2b4599dff9b3e45c62567eac15e83186b4cb64e`、Q4_K_M（約1.81GB）。重みは初回使用時に取得します。[PLaMo Community License](https://plamo.preferredai.jp/info/plamo-community-license-en) に従い、原文・モデルカードを同梱します。個人の趣味・学習など非商用利用は無償です。商用利用はPFNへの事前登録と本人・所属企業の年間売上10億円以下が条件で、超える場合は別途契約が必要です。アプリは登録を代行しません。出力を公開する場合はPLaMoの出力であることを明記してください。アプリ本体のMITライセンスとは別の条件です。

## macOSワーカーの再現と共有配置

音声処理ワーカーの依存は`requirements-tts-mac.txt`と`distribution/worker-requirements.lock`で固定する。
IndexTTSは別のTransformers環境を持ち、両環境で一致したPyTorch等はアプリ内部の相対リンクで共有する。
原文ライセンス・NOTICEと依存一覧は引き続き保持する。配布にはPyTorchのコンパイル用ヘッダー・パッケージ導入ツール・
未使用の学習/Web UI依存を含めず、同梱ライブラリを再ビルドする場合は公開ソースと固定版を使う。
