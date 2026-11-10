import AppKit
import SwiftUI

struct ProductView: View {
    @ObservedObject var store: BackendStore
    @State private var showSetup = false
    @State private var showLogs = false
    @State private var showMobile = false
    @State private var showLibrary = false
    @State private var showStorage = false

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 10) {
                Image(nsImage: Bundle.main.url(forResource: "AppIcon", withExtension: "icns").flatMap { NSImage(contentsOf: $0) } ?? NSApplication.shared.applicationIconImage)
                    .resizable().frame(width: 30, height: 30)
                Text("TransiroYT").font(.system(size: 16, weight: .semibold))
                Spacer()
                Picker("表示", selection: $showLibrary) {
                    Text("作成").tag(false)
                    Text("ライブラリ").tag(true)
                }.pickerStyle(.segmented).frame(width: 200)
                Menu {
                    Button("キャッシュを削除…", systemImage: "trash") { showStorage = true }
                    OpenSettingsButton()
                    Button("モバイル連携") { showMobile = true }
                    Button("初回の準備と利用条件") { showSetup = true }
                    Button("ログ") { showLogs = true }
                } label: { Image(systemName: "ellipsis").font(.title3) }
                .menuStyle(.borderlessButton).fixedSize().accessibilityLabel("その他")
            }.padding(.horizontal, 40).padding(.top, 24).padding(.bottom, 8)
            ZStack {
                FileDubbingView(store: store, preferencesRevision: store.preferencesRevision)
                    .opacity(showLibrary ? 0 : 1).disabled(showLibrary).accessibilityHidden(showLibrary)
                if showLibrary { LibraryView(store: store) }
            }
        }
        .background(Color(nsColor: .windowBackgroundColor))
        .navigationTitle("TransiroYT")
        .tint(.yellow)
        .frame(minWidth: 780, minHeight: 620)
        .sheet(isPresented: $showSetup) {
            setup.frame(width: 600, height: 500)
        }
        .sheet(isPresented: $showMobile) { MobileWorkersView(store: store) }
        .sheet(isPresented: $showStorage) { CacheStorageView(store: store) }
        .sheet(isPresented: $showLogs) {
            VStack {
                LogsView(store: store)
                Button("閉じる") { showLogs = false }.keyboardShortcut(.cancelAction).padding()
            }.frame(width: 720, height: 500)
        }
        .onChange(of: store.showGuide) { _, value in
            if value { showSetup = true; store.showGuide = false }
        }
        .alert("操作を完了できませんでした", isPresented: Binding(get: { store.errorMessage != nil }, set: { if !$0 { store.errorMessage = nil } })) {
            Button("閉じる") { store.errorMessage = nil }
        } message: { Text(store.errorMessage ?? "") }
    }

    private var setup: some View {
        VStack(alignment: .leading, spacing: 24) {
            Image(nsImage: Bundle.main.url(forResource: "AppIcon", withExtension: "icns").flatMap { NSImage(contentsOf: $0) } ?? NSApplication.shared.applicationIconImage)
                .resizable().frame(width: 56, height: 56)
            Text("初回の準備").font(.system(size: 30, weight: .semibold)).tracking(-1)
            Text("動画またはYouTube URLを指定して「開始」を押すと、必要なモデルを自動で取得します。PythonやFFmpegなどの追加ソフトは不要です。")
            Text("初回に必要なモデル約21 GBを取得します。取得済みモデルはアプリを更新しても再利用します。インターネット接続と、動画の作業領域を含む35 GB以上の空きを用意してください。")
                .font(.callout).foregroundStyle(.secondary)
            Text("Apple Silicon・macOS 27以降が必要です。翻訳にはこのMacのPLaMo・Gemma、または設定したAPIを使います。")
                .font(.callout).foregroundStyle(.secondary)
            HStack {
                Link("音声モデルの利用条件", destination: URL(string: "https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small-MF#-license--ethical-restrictions")!)
                if store.installation != nil {
                    Button("すべてのライセンス") { store.openGuide() }.buttonStyle(.link)
                }
            }.font(.callout)
            Spacer(minLength: 8)
            Button("閉じる") { showSetup = false }
                .buttonStyle(.borderedProminent).controlSize(.large).keyboardShortcut(.defaultAction)
        }.padding(36)
    }
}
