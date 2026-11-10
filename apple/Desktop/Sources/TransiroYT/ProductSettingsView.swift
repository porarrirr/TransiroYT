import AppKit
import SwiftUI

enum SettingsPage: Hashable {
    case task(RoutingTask), services, voice, output

    var title: String {
        switch self {
        case .task(let task): task.title
        case .services: "AIサービス"
        case .voice: "音声"
        case .output: "保存とダウンロード"
        }
    }
    var icon: String {
        switch self {
        case .task(let task): task.icon
        case .services: "key"
        case .voice: "waveform"
        case .output: "square.and.arrow.down"
        }
    }
    var lead: String {
        switch self {
        case .task(let task): task.lead
        case .services: "作業に使えるAIと、APIの接続・キーを管理します。"
        case .voice: "吹き替えの音量と音声の生成を設定します。"
        case .output: "完成した動画の保存先とダウンロード画質を設定します。"
        }
    }
    var usesAPISettings: Bool {
        switch self {
        case .task, .services: true
        case .voice, .output: false
        }
    }
}

struct OpenSettingsButton: View {
    @Environment(\.openWindow) private var openWindow
    var body: some View {
        Button("設定…", systemImage: "gearshape") { openWindow(id: "settings") }
            .keyboardShortcut(",", modifiers: .command)
    }
}

/// Every opening starts from the saved settings; closing the window discards unsaved edits.
struct SettingsWindow: View {
    @ObservedObject var store: BackendStore
    @State private var session = UUID()

    var body: some View {
        SettingsLoader(store: store).id(session).onDisappear { session = UUID() }
    }
}

private struct SettingsLoader: View {
    @ObservedObject var store: BackendStore
    @State private var loaded = Result { try ProductPreferences.load() }

    var body: some View {
        switch loaded {
        case .success(let preferences):
            ProductSettingsView(store: store, preferences: preferences) { updated in
                try updated.save()
                store.preferencesRevision += 1
            }
        case .failure(let error):
            ContentUnavailableView("設定を読み込めませんでした", systemImage: "exclamationmark.triangle",
                                   description: Text(error.localizedDescription))
                .frame(minWidth: 820, minHeight: 560)
        }
    }
}

struct ProductSettingsView: View {
    @Environment(\.dismissWindow) private var dismissWindow
    @ObservedObject var store: BackendStore
    @State private var page = SettingsPage.task(.translation)
    @State private var service: String?
    @State private var routing = TaskRoutingSettings()
    @State private var sarashina = SarashinaSettings()
    @State private var apiLoaded = false
    @State private var loadingAPI = false
    @State private var loadError: String?
    @State private var saving = false
    @State private var showDetail = false
    @State var preferences: ProductPreferences
    let save: (ProductPreferences) throws -> Void
    @State private var error: String?

    private var problem: String? { apiLoaded ? routing.problem : nil }

    var body: some View {
        NavigationSplitView {
            sidebar.navigationSplitViewColumnWidth(min: 190, ideal: 210, max: 260).toolbar(removing: .sidebarToggle)
        } detail: {
            VStack(alignment: .leading, spacing: 0) {
                HStack(spacing: 6) {
                    Text(page.lead).foregroundStyle(.secondary)
                    if case .task(let task) = page {
                        Button { showDetail = true } label: { Image(systemName: "info.circle") }
                            .buttonStyle(.borderless).accessibilityLabel("\(task.title)の詳しい説明")
                            .popover(isPresented: $showDetail) {
                                Text(task.summary).padding().frame(width: 340).fixedSize(horizontal: false, vertical: true)
                            }
                    }
                }
                .padding(.horizontal, SettingsLayout.inset).padding(.top, 14).padding(.bottom, 2)
                content.frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
            }
            .navigationTitle(page.title)
            .safeAreaInset(edge: .bottom, spacing: 0) { actions }
        }
        .task { await loadAPISettings() }
        .frame(minWidth: 820, minHeight: 560)
        .alert("保存できませんでした", isPresented: Binding(get: { error != nil }, set: { if !$0 { error = nil } })) {
            Button("閉じる") { error = nil }
        } message: { Text(error ?? "") }
    }

    private var actions: some View {
        VStack(spacing: 0) {
            Divider()
            HStack(spacing: 12) {
                if let problem {
                    Label(problem, systemImage: "exclamationmark.triangle.fill").foregroundStyle(.orange)
                } else {
                    Text("次に開始する吹き替えから適用します。").foregroundStyle(.secondary)
                }
                Spacer()
                Button("キャンセル") { dismissWindow(id: "settings") }.keyboardShortcut(.cancelAction)
                Button("保存", action: saveAll)
                    .buttonStyle(.borderedProminent).tint(.yellow).keyboardShortcut(.defaultAction)
                    .disabled(saving || problem != nil)
            }
            .font(.callout).padding(.horizontal, SettingsLayout.inset).padding(.vertical, 12)
        }
        .background(.bar)
    }

    private var sidebar: some View {
        List(selection: Binding(get: { page }, set: { if let value = $0 { page = value; service = nil } })) {
            Section("AIの使い方") {
                ForEach(RoutingTask.allCases) { task in
                    Label {
                        VStack(alignment: .leading, spacing: 1) {
                            Text(task.title)
                            if apiLoaded { Text(routeSummary(task)).font(.caption).foregroundStyle(.secondary).lineLimit(1) }
                        }
                    } icon: { Image(systemName: task.icon) }
                    .tag(SettingsPage.task(task))
                }
                Label(SettingsPage.services.title, systemImage: SettingsPage.services.icon).tag(SettingsPage.services)
            }
            Section("吹き替え") {
                ForEach([SettingsPage.voice, .output], id: \.self) { item in
                    Label(item.title, systemImage: item.icon).tag(item)
                }
            }
        }
        .listStyle(.sidebar).tint(.yellow)
    }

    @ViewBuilder private var content: some View {
        if page.usesAPISettings && !apiLoaded {
            if loadingAPI {
                ProgressView("AIの設定を読み込んでいます…").frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ContentUnavailableView {
                    Label("AIの設定を読み込めませんでした", systemImage: "exclamationmark.triangle")
                } description: {
                    Text("\(loadError ?? "")\n音声と保存の設定は変更・保存できます。")
                } actions: {
                    Button("もう一度読み込む") { Task { await loadAPISettings() } }
                }
            }
        } else {
            switch page {
            case .task(let task):
                TaskOrderPage(task: task, config: $routing, sarashinaReady: sarashina.ready) { provider in
                    page = .services
                    service = provider
                }
                .disabled(saving)
            case .services:
                AIServicesPage(store: store, config: $routing, sarashina: $sarashina, selected: $service).disabled(saving)
            case .voice: voicePage
            case .output: outputPage
            }
        }
    }

    private func routeSummary(_ task: RoutingTask) -> String {
        let chain = routing.chain(task)
        guard let first = chain.first else { return task == .shortening ? "使わない" : "未選択" }
        return chain.count == 1 ? routing.name(first) : "\(routing.name(first)) ほか\(chain.count - 1)件"
    }

    private var voicePage: some View {
        Form {
            Section("音量") {
                volume("元の声", note: "0%で元の声を消して置き換えます", value: $preferences.originalVolume, range: 0...2)
                volume("吹き替え音声", note: "100%が生成時の音量です", value: $preferences.dubVolume, range: 0.01...2)
            }
            Section("話者と生成") {
                Picker("話者数", selection: $preferences.speakers) {
                    Text("自動判定").tag(Int?.none)
                    ForEach(1...20, id: \.self) { Text("\($0)人").tag(Int?.some($0)) }
                }
                LabeledContent {
                    HStack(spacing: 12) {
                        Slider(value: Binding(get: { Double(preferences.synthesisSteps) },
                                              set: { preferences.steps = Int($0.rounded()) }),
                               in: Double(ProductPreferences.stepRange.lowerBound)...Double(ProductPreferences.stepRange.upperBound),
                               step: 1)
                            .frame(width: 180).accessibilityLabel("生成ステップ数")
                        Text("\(preferences.synthesisSteps)").monospacedDigit().frame(width: 44, alignment: .trailing)
                    }
                } label: {
                    Text("生成ステップ数")
                    Text("既定は10。多いほど時間がかかります")
                }
            }
            Section("担当するモデル") {
                LabeledContent("文字起こし", value: "macOS・Whisper")
                LabeledContent("話者判定", value: "Nemotron・ECAPA")
                LabeledContent("音源分離", value: "Kim Mel-Band RoFormer")
                LabeledContent("音声生成", value: "Irodori")
            }
        }
        .formStyle(.grouped)
    }

    private var outputPage: some View {
        Form {
            Section("完成した動画") {
                LabeledContent {
                    Button("変更…", action: chooseDirectory)
                } label: {
                    Text("保存先")
                    Text(preferences.outputDirectory).lineLimit(1).truncationMode(.middle).textSelection(.enabled)
                }
            }
            Section("YouTube") {
                Picker(selection: $preferences.youtubeQuality) {
                    Text("最高画質（自動）").tag("best")
                    Text("2160p（4K）").tag("2160")
                    Text("1440p").tag("1440")
                    Text("1080p").tag("1080")
                    Text("720p").tag("720")
                    Text("480p").tag("480")
                    Text("360p").tag("360")
                } label: {
                    Text("ダウンロード画質")
                    Text("指定以下で利用できる最高画質を取得します")
                }
            }
        }
        .formStyle(.grouped)
    }

    private func volume(_ title: String, note: String, value: Binding<Double>, range: ClosedRange<Double>) -> some View {
        LabeledContent {
            HStack(spacing: 12) {
                Slider(value: value, in: range).frame(width: 180).accessibilityLabel(title)
                Text(value.wrappedValue, format: .percent.precision(.fractionLength(0)))
                    .monospacedDigit().frame(width: 44, alignment: .trailing)
            }
        } label: {
            Text(title)
            Text(note)
        }
    }

    private func saveAll() {
        saving = true
        Task {
            do {
                if apiLoaded {
                    if routing.usesSarashina && !sarashina.ready {
                        throw DesktopError.message("Sarashina翻訳を使うには、AIサービスの「Sarashina翻訳」でAPIキーを入力してください。")
                    }
                    let encoder = JSONEncoder()
                    encoder.keyEncodingStrategy = .convertToSnakeCase
                    let _: SarashinaSettings = try await store.request("/api/sarashina-translation", method: "PUT",
                                                                      body: encoder.encode(sarashina))
                    let _: TaskRoutingSettings = try await store.request("/api/task-routing", method: "PUT",
                                                                        body: encoder.encode(routing))
                }
                try save(preferences)
                dismissWindow(id: "settings")
            } catch { self.error = error.localizedDescription }
            saving = false
        }
    }

    private func loadAPISettings() async {
        guard !loadingAPI else { return }
        loadingAPI = true
        defer { loadingAPI = false }
        do {
            sarashina = try await store.request("/api/sarashina-translation")
            var loaded: TaskRoutingSettings = try await store.request("/api/task-routing")
            loaded.useExplicitOrder()
            routing = loaded
            apiLoaded = true
            loadError = nil
        } catch { loadError = error.localizedDescription }
    }

    private func chooseDirectory() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = "選択"
        panel.directoryURL = URL(fileURLWithPath: (preferences.outputDirectory as NSString).expandingTildeInPath)
        if panel.runModal() == .OK, let url = panel.url { preferences.outputDirectory = url.path }
    }
}

/// Shared spacing so list pages line up with the grouped forms' section edges.
enum SettingsLayout {
    static let inset: CGFloat = 20
}

struct SarashinaSettings: Codable {
    var apiKey: String? = nil
    var hasKey = false

    /// An empty string marks the saved key for deletion; nil keeps the saved one.
    var ready: Bool { apiKey.map { !$0.isEmpty } ?? hasKey }
}
