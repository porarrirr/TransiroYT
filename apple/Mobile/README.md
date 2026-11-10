# TransiroYT Mobile

iPhone／iPadをMac版TransiroYTの音声生成ワーカーとして使うアプリです。
v0.5.1はMac v0.5.1と同じCore ML実装・モデル識別子・通信形式を使います。
翻訳・話者判定・音声の検証・完成動画の出力はMacで行い、モバイル版は音声生成だけを担当します。
PCCや翻訳モデルは使いません。

## ビルド

Apple Silicon Mac、Xcode 16以降（Swift 6対応）、iOS／iPadOS 18以降、XcodeGenが必要です。
リポジトリのルートで実行します。

```sh
brew install xcodegen
scripts/build_mobile.sh
open apple/Mobile/TransiroYTMobile.xcodeproj
```

スクリプトはXcodeGenでプロジェクトを生成し、arm64シミュレーター版をビルドします。
Macと共通のSwift Packageは`apple/Inference`に含まれます。別途Python・llama.cpp・CMakeのビルドは不要です。
出力先は`.local/work/mobile/DerivedData`です。変更する場合は
`TRANSIROYT_MOBILE_BUILD_DIR=/path/to/build scripts/build_mobile.sh`を使います。

実機へ入れる場合はXcodeのTransiroYTMobileターゲットで自分のSigning Teamを選び、
接続したiPhone／iPadを実行先にしてRunします。既存のモデルと接続情報を保持するため、
更新時は同じTeamとBundle Identifierを使ってください。署名情報や接続トークンをGitへ追加しないでください。
GitHubではソースコードを公開しています。App Store／TestFlight配布や汎用の実機インストール用IPAは提供していません。

## Macとの連携

1. Mac版v0.5.1を起動し、「… → モバイル連携」でWi-FiまたはUSBケーブルを選びます。
2. 「端末を追加」で表示したQRコードをモバイル版で読み取ります。QRコードは2分で失効します。
3. 初回は端末自身が音声モデル約2 GBを取得・検証・コンパイルします。取得済みモデルは再利用します。
4. アプリを前面にしたまま、Macで動画の吹き替えを開始します。Macが自動で音声生成を割り当てます。

最大2台のiPhone／iPadと連携できます。Wi-Fiでは同じネットワーク、USBではデータ通信対応ケーブルと
「このコンピュータを信頼」が必要です。初回のモデル取得には端末のインターネット接続も必要です。
低電力モード・画面ロック・アプリの中断・通信失敗時は処理をMacへ戻します。
再接続情報は端末のキーチェーンに保存し、TLS証明書を固定してペアリングしたMacだけと通信します。
旧版でバージョン不一致が表示された場合は、Macとモバイルの両方を更新してください。

## ライセンスと検証

独自コードは[MIT](../../LICENSE)。共通エンジンの固定ソース・変更点は
[`apple/Inference/Vendor/IrodoriTTS/UPSTREAM.md`](../Inference/Vendor/IrodoriTTS/UPSTREAM.md)、
第三者の原文・NOTICEは[`Notices`](Notices)と[ライセンス一覧](../../docs/licenses.md)にあります。
音声モデル固有の利用条件はアプリ内からも確認できます。モデルの重みはリポジトリに含めません。

Mac側の通信・割り当て・切断回復の回帰検証は、リポジトリのルートで実行できます。

```sh
uv run python -m pytest -q tests/test_mobile_workers.py tests/test_mobile_connection.py
```

v0.5.1ではXcode 27.2でarm64シミュレーター版と実機用Release版（署名なし）のビルド、
iOS 27.2シミュレーターでの起動・画面表示を確認しました。
実際のSwift通信コードとMac側エンジンをループバックHTTPSで接続し、証明書不一致の拒否、ペアリング、
発話長・参照音声の受け渡し、結果の再送・完了数・切断を確認しています。この通信検証では合成したWAVを使い、
モバイル端末による実モデルの音声生成は検証していません。
実機の速度・メモリ・USBケーブル接続の連続動作は未確認です。
