# アプリとyt-dlpの自動更新

配布アプリはSparkle 2.10.1で更新を確認する。初期設定は24時間ごとの自動確認・
自動ダウンロード。メニューバーの「TransiroYT → アップデートを確認…」からも確認できる。
動画作成・モデル取得中の確認を延期し、取得済み更新の再起動も処理完了まで待つ。

yt-dlpと対応するEJSは、検証済みの更新版アプリに同梱して入れ替える。
YouTube側の変更を即座に直す仕組みではなく、配布者が新しい依存を固定し、
検証・署名・公証して公開する必要がある。署名済みアプリ内でpipや`yt-dlp -U`は実行しない。

## 配信先と署名

`distribution/updates.json`にGitHub ReleasesのURL・Ed25519公開鍵・キーチェーンの
アカウント名を保持する。秘密鍵はキーチェーンに置き、Gitへ含めない。
安定版は`appcast.xml`、previewは`appcast-preview.xml`を使用する。
各バージョンはビルド番号を含む固有のタグから取得する。
SparkleはアーカイブのEd25519署名を検証してから展開する。

ソースリポジトリの公開と更新配信は別の操作です。
更新配信は公証済み成果物の公開後に開始します。ローカルpreviewの作成だけでは配信されません。

## 公開時の手順

1. yt-dlp/EJSなど必要な依存を固定し、回帰テストと同梱ランタイムでの実処理を確認する。
2. `--reuse-runtime`なしでDeveloper ID署名・公証済みアプリを作る。

   ```sh
   uv run python scripts/build_dmg.py --channel release \
     --sign-identity 'Developer ID Application: YOUR NAME (TEAMID)' \
     --notary-profile YOUR_KEYCHAIN_PROFILE \
     --build-dir .local/release-build --output-dir .local/release-artifacts
   ```

3. `release-manifest.json`、DMG、対応ソースZIP、署名付き`appcast.xml`を確認する。
   更新鍵の利用をmacOSが確認した場合は、Mac上で許可する。
4. リポジトリを公開する承認と公開前検証を終えた後、公開済みリポジトリに配信する。

   ```sh
   uv run python scripts/publish_update.py .local/release-artifacts
   ```

公開処理は非公開リポジトリ・未公証・Developer ID署名なしの成果物を拒否する。
まず固有タグにアーカイブを公開し、その後`updates`リリースのフィードを置き換える。
安定版を公開するとpreviewフィードにも同じ版を配信する。
previewの配信には`--channel preview --notarize --update-feed`とDeveloper ID設定を使う。
署名付きフィードを作らないローカルpreviewは配信対象にしない。

## ローカル検証

実運用と異なるアプリID・一時鍵・localhostフィードを使い、既存アプリと利用者データを
変更しない場所で更新を検証する。改ざんアーカイブが拒否され、正しいアーカイブでは
バージョンが更新され、再起動することを確認する。
この検証はGitHubからの公開配信・Developer ID公証済み更新・別Macでの検証と区別する。

参考: [Sparkleの設定](https://sparkle-project.org/documentation/customization/)、
[更新API](https://sparkle-project.org/documentation/api-reference/Classes/SPUUpdater.html)。
