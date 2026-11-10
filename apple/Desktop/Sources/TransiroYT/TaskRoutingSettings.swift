import SwiftUI

struct EngineSettings: Codable {
    var plamo = true
    var gemma = true
    var sarashina = false
    var google = false
    var bing = false
}

struct TaskRouteSettings: Codable {
    var chain = ["gemma"]
}

struct TaskRoutingSettings: Codable {
    var enabled = false
    var presets: [APIPreset] = []
    var engines = EngineSettings()
    var language = TaskRouteSettings()
    var translation = TaskRouteSettings(chain: ["plamo", "gemma"])
    var shortening = TaskRouteSettings()

    var usesSarashina: Bool {
        enabled ? translation.chain.contains("sarashina") : engines.sarashina
    }

    /// The settings screen always edits explicit orders. Simple mode (still available from the CLI) becomes the
    /// order the backend derives in `TaskRoutingConfig.chain`, so saving keeps the same behaviour.
    mutating func useExplicitOrder() {
        guard !enabled else { return }
        let apis = presets.filter(\.enabled).map { "api:\($0.id)" }
        let on = ["plamo": engines.plamo, "gemma": engines.gemma,
                  "sarashina": engines.sarashina, "google": engines.google, "bing": engines.bing]
        let pick = { (order: [String]) in order.filter { $0.hasPrefix("api:") || on[$0] == true } }
        translation.chain = pick(["plamo"] + apis + ["sarashina", "google", "bing", "gemma"])
        language.chain = pick(apis + ["gemma"])
        shortening.chain = language.chain
        enabled = true
    }

    func chain(_ task: RoutingTask) -> [String] { self[keyPath: task.route].chain }

    mutating func setUse(_ provider: String, for task: RoutingTask, _ use: Bool) {
        self[keyPath: task.route].chain.removeAll { $0 == provider }
        if use { self[keyPath: task.route].chain.append(provider) }
    }

    /// Translation-only engines are rejected by the backend for the other tasks.
    func candidates(_ task: RoutingTask) -> [String] {
        presets.map { "api:\($0.id)" }
            + (task == .translation ? ["plamo", "gemma", "sarashina", "google", "bing"] : ["gemma"])
    }

    func preset(_ provider: String) -> APIPreset? { presets.first { "api:\($0.id)" == provider } }

    func name(_ provider: String) -> String {
        preset(provider)?.name ?? BuiltInEngine.named(provider).name
    }

    func detail(_ provider: String) -> String {
        guard let preset = preset(provider) else { return BuiltInEngine.named(provider).place }
        return "API · \(preset.host) · \(preset.model.isEmpty ? "モデル未指定" : preset.model)"
    }

    func setupIssue(_ provider: String, sarashinaReady: Bool) -> String? {
        if let preset = preset(provider) { return preset.model.trimmingCharacters(in: .whitespaces).isEmpty ? "モデル未指定" : nil }
        return provider == "sarashina" && !sarashinaReady ? "APIキー未設定" : nil
    }

    /// The backend refuses to save without these, so the screen explains it before saving.
    var problem: String? {
        if language.chain.isEmpty { return "字幕の言語判定に使うAIを1つ以上選んでください。" }
        if translation.chain.isEmpty { return "翻訳に使うAIを1つ以上選んでください。" }
        return nil
    }

    mutating func removePreset(_ id: String) {
        presets.removeAll { $0.id == id }
        for task in RoutingTask.allCases {
            self[keyPath: task.route].chain.removeAll { $0 == "api:\(id)" }
        }
    }
}

enum RoutingTask: String, CaseIterable, Identifiable {
    case translation, language, shortening
    var id: Self { self }

    var route: WritableKeyPath<TaskRoutingSettings, TaskRouteSettings> {
        switch self {
        case .translation: \.translation
        case .language: \.language
        case .shortening: \.shortening
        }
    }
    var title: String {
        switch self {
        case .translation: "翻訳"
        case .language: "字幕の言語判定"
        case .shortening: "訳文の短縮"
        }
    }
    var icon: String {
        switch self {
        case .translation: "character.bubble"
        case .language: "text.magnifyingglass"
        case .shortening: "arrow.down.right.and.arrow.up.left"
        }
    }
    var summary: String {
        switch self {
        case .translation:
            "字幕を日本語に翻訳します。上から順に試し、訳せなかった区間だけを次のAIに送ります。PLaMoとSarashinaは英語の字幕だけに使い、それ以外の言語では飛ばします。"
        case .language:
            "字幕の言語が「自動」のとき、字幕の文字から言語を判定します。上から順に試し、失敗したら次へ進みます。音声からの判定はWhisperが行います。"
        case .shortening:
            "長すぎる訳文を話す時間に合わせて短くし、生成後に早口になった区間を訳し直します。精度の高いモデルを推奨します。何も選ばなければ短縮しません。"
        }
    }
    var lead: String {
        switch self {
        case .translation: "字幕を日本語にします。上から順に試します。"
        case .language: "字幕の言語が「自動」のとき、字幕から言語を判定します。"
        case .shortening: "長い訳文を、話す時間に合わせて短くします。"
        }
    }
    var emptyNote: String {
        self == .shortening ? "何も選んでいません。短縮と訳し直しはしません。" : "何も選んでいません。下の一覧から1つ以上選んでください。"
    }
}

struct BuiltInEngine {
    let name: String
    let place: String
    let summary: String
    let icon: String

    static func named(_ id: String) -> Self {
        switch id {
        case "plamo": Self(name: "PLaMo 2B", place: "このMac · 英語のみ",
                           summary: "このMacで動く英語→日本語の翻訳モデル（Q4_K_M）です。英語以外の字幕では使いません。", icon: "laptopcomputer")
        case "gemma": Self(name: "Gemma", place: "このMac",
                           summary: "このMacで動く汎用モデルです。翻訳・言語判定・短縮のすべてに使えます。", icon: "laptopcomputer")
        case "sarashina": Self(name: "Sarashina翻訳", place: "クラウド · 英語のみ · APIキーが必要",
                               summary: "英語→日本語の翻訳APIです。字幕の本文だけを1区間ずつ送り、前後の文脈は送りません。モデルの指定は不要です。", icon: "globe")
        case "google": Self(name: "Google翻訳", place: "クラウド · キー不要",
                            summary: "字幕の本文だけを1区間ずつ送ります。前後の文脈は送らず、短縮には使えません。提供元の制限や仕様変更で使えなくなる場合があります。", icon: "globe")
        case "bing": Self(name: "Bing翻訳", place: "クラウド · キー不要",
                          summary: "字幕の本文だけを1区間ずつ送ります。前後の文脈は送らず、短縮には使えません。提供元の制限や仕様変更で使えなくなる場合があります。", icon: "globe")
        default: Self(name: id, place: "", summary: "", icon: "questionmark")
        }
    }
}

/// One task's order: the chosen AIs on top (drag to reorder), the rest below with an unchecked box.
struct TaskOrderPage: View {
    let task: RoutingTask
    @Binding var config: TaskRoutingSettings
    let sarashinaReady: Bool
    let openService: (String) -> Void

    var body: some View {
        let chosen = config.chain(task)
        let others = config.candidates(task).filter { !chosen.contains($0) }
        List {
                Section {
                    ForEach(Array(chosen.enumerated()), id: \.element) { index, provider in
                        row(provider, position: index + 1)
                    }
                    .onMove { move($0, to: $1) }
                    if chosen.isEmpty {
                        Text(task.emptyNote).foregroundStyle(task == .shortening ? Color.secondary : Color.orange).padding(.vertical, 6)
                    }
                } header: {
                    Text("使う順番")
                }
                if !others.isEmpty {
                    Section("使わない") {
                        ForEach(others, id: \.self) { row($0, position: nil) }
                    }
                }
            }
            .listStyle(.bordered(alternatesRowBackgrounds: false))
            .padding(.horizontal, SettingsLayout.inset).padding(.top, 12).padding(.bottom, SettingsLayout.inset)
    }

    private func move(_ offsets: IndexSet, to destination: Int) {
        config[keyPath: task.route].chain.move(fromOffsets: offsets, toOffset: destination)
    }

    private func row(_ provider: String, position: Int?) -> some View {
        let issue = config.setupIssue(provider, sarashinaReady: sarashinaReady)
        return HStack(spacing: 12) {
            Toggle("\(config.name(provider))を使う", isOn: Binding(get: { position != nil },
                                                               set: { config.setUse(provider, for: task, $0) }))
                .toggleStyle(.checkbox).labelsHidden()
                .disabled(position == nil && issue != nil)
            Text(position.map(String.init) ?? "")
                .font(.headline).monospacedDigit().foregroundStyle(.secondary).frame(width: 18)
            VStack(alignment: .leading, spacing: 2) {
                Text(config.name(provider)).fontWeight(.medium)
                Text(config.detail(provider)).font(.caption).foregroundStyle(.secondary).lineLimit(1)
            }
            Spacer()
            if let issue {
                Label(issue, systemImage: "exclamationmark.triangle.fill").font(.caption).foregroundStyle(.orange)
                Button("設定…") { openService(provider) }.controlSize(.small)
            }
            if position != nil {
                Image(systemName: "line.3.horizontal").foregroundStyle(.tertiary).accessibilityHidden(true)
            }
        }
        .padding(.vertical, 5)
        .opacity(position == nil && issue != nil ? 0.7 : 1)
        .contextMenu {
            // Keyboard and VoiceOver users cannot drag; the menu gives them the same reordering.
            if let position {
                Button("上へ") { move([position - 1], to: position - 2) }.disabled(position == 1)
                Button("下へ") { move([position - 1], to: position + 1) }.disabled(position == config.chain(task).count)
            }
        }
    }
}
