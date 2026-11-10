import AppKit
import SwiftUI

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    weak var store: BackendStore?
    func applicationWillTerminate(_ notification: Notification) { store?.shutdown() }
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }
}

@main
struct TransiroYTApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @StateObject private var store = BackendStore()
    @StateObject private var updates = AppUpdater()

    @ViewBuilder
    private var mainContent: some View {
        ProductView(store: store)
    }

    var body: some Scene {
        Window("TransiroYT", id: "main") {
            mainContent
                .task {
                    delegate.store = store
                    updates.store = store
                    updates.start()
                    await store.initializeConsumer()
                    while !Task.isCancelled {
                        await store.refresh()
                        updates.resumeWhenIdle()
                        try? await Task.sleep(for: .seconds(3))
                    }
                }
                .task {
                    while !Task.isCancelled {
                        await store.refreshDownloadProgress()
                        try? await Task.sleep(for: .milliseconds(300))
                    }
                }
        }
        .defaultSize(width: 980, height: 720)
        .commands {
            CommandGroup(replacing: .appSettings) { OpenSettingsButton() }
            CommandGroup(after: .appInfo) {
                Button("アップデートを確認…", action: updates.check).disabled(!updates.canCheck)
                if store.installation == nil {
                    Button("プロジェクトフォルダを選択…", action: store.chooseProject)
                        .disabled(store.ownsServer || store.busy)
                } else {
                    Button("はじめに・セットアップ") { store.showGuide = true }
                    Button("使い方を開く", action: store.openGuide)
                }
                Button("状態を更新") { Task { await store.refresh() } }.keyboardShortcut("r")
            }
        }
        Window("設定", id: "settings") { SettingsWindow(store: store) }
            .defaultSize(width: 900, height: 640)
            .windowResizability(.contentMinSize)
    }
}
