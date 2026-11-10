import Foundation
import SwiftUI

struct APIPreset: Codable, Identifiable {
    var id = UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased()
    var name = "API"
    var enabled = false
    var `protocol` = "chat"
    var baseUrl = "https://api.openai.com/v1"
    var inferenceUrl = ""
    var modelsUrl = ""
    var model = ""
    var apiKey: String? = nil
    var hasKey = false
    var chatTokenParameter = "max_completion_tokens"
    var reasoningEffort = ""
    var reasoningTokenBudget = 4096

    var host: String { URLComponents(string: baseUrl)?.host ?? baseUrl }
    var keyStatus: String {
        if apiKey == "" { return "削除予定" }
        if let apiKey, !apiKey.isEmpty { return "保存時に更新" }
        return hasKey ? "保存済み" : "未設定・認証なし"
    }
    var credentialScope: [String] {
        [self.protocol] + [baseUrl, inferenceUrl.isEmpty ? baseUrl : inferenceUrl, modelsUrl.isEmpty ? baseUrl : modelsUrl].map {
            guard let url = URLComponents(string: $0) else { return $0 }
            return "\(url.scheme ?? "")://\(url.host ?? ""):\(url.port ?? (url.scheme == "https" ? 443 : 80))"
        }
    }
    func duplicate() -> Self {
        var copy = self
        copy.id = UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased()
        copy.name += " のコピー"
        copy.enabled = false
        copy.apiKey = nil
        copy.hasKey = false
        return copy
    }
}

enum APIConnectionTemplate: String, CaseIterable, Identifiable {
    case openAI = "OpenAI", anthropic = "Anthropic", gemini = "Gemini", compatible = "互換API"
    var id: Self { self }
    var preset: APIPreset {
        switch self {
        case .openAI: APIPreset(name: rawValue)
        case .anthropic: APIPreset(name: rawValue, protocol: "anthropic", baseUrl: "https://api.anthropic.com/v1")
        case .gemini: APIPreset(name: rawValue, protocol: "gemini", baseUrl: "https://generativelanguage.googleapis.com/v1beta")
        case .compatible: APIPreset(name: rawValue, baseUrl: "https://example.com/v1")
        }
    }
    var summary: String {
        switch self {
        case .openAI: "OpenAIのAPIキーで使います。"
        case .anthropic: "AnthropicのAPIキーで使います（Messages API）。"
        case .gemini: "Google AI StudioのAPIキーで使います。"
        case .compatible: "Chat Completions・Responses・Anthropic Messages・Geminiのどれかの形式に対応したAPIに接続します。LM StudioなどこのMacで動くサーバーにも使えます。形式とURLは次の画面で設定します。"
        }
    }
    var icon: String { "cloud" }
}

struct CustomAPIModels: Decodable {
    let models: [String]
}

struct APIConnectionTest: Decodable {
    let ok: Bool
    let message: String
    let latencyMs: Int
}

/// The connection form, shown when adding an API and on its page in AIサービス.
struct PresetEditor: View {
    @ObservedObject var store: BackendStore
    @Binding var config: APIPreset
    @State private var models: [String] = []
    @State private var modelSearch = ""
    @State private var working = false
    @State private var status: String?
    @State private var failed = false
    @State private var originalScope: [String]?
    private var needsKey: Bool { config.hasKey && config.apiKey == nil && originalScope != config.credentialScope }
    private var matchingModels: [String] {
        models.filter { modelSearch.isEmpty || $0.localizedCaseInsensitiveContains(modelSearch) }
    }

    var body: some View {
        Group {
            Section("接続先") {
                FieldRow("名前") { TextField("名前", text: $config.name, prompt: Text("例：OpenAI")) }
                Picker("API形式", selection: $config.protocol) {
                    Text("Chat Completions").tag("chat")
                    Text("Responses").tag("responses")
                    Text("Anthropic Messages").tag("anthropic")
                    Text("Gemini").tag("gemini")
                }
                FieldRow("ベースURL") { TextField("ベースURL", text: $config.baseUrl, prompt: Text("https://…/v1")) }
            }
            Section {
                FieldRow("APIキー") {
                    SecureField("APIキー", text: Binding(get: { config.apiKey ?? "" }, set: { config.apiKey = $0.isEmpty ? nil : $0 }),
                                prompt: Text(needsKey ? "再入力してください" : config.hasKey ? "保存済み（変更するときだけ入力）" : "認証がなければ空欄"))
                        .help("キーはこのMacのキーチェーンに保存します。")
                }
                LabeledContent("状態", value: config.keyStatus)
                if needsKey {
                    Label("接続先が変わったため、保存済みのキーは使いません。キーを入力し直すか削除してください。", systemImage: "exclamationmark.triangle.fill")
                        .foregroundStyle(.orange)
                }
                if config.hasKey || config.apiKey != nil {
                    Button("保存済みのキーを削除", role: .destructive) { config.apiKey = "" }
                }
            } header: {
                Text("APIキー")
            }
            Section("モデル") {
                FieldRow("モデルID") { TextField("モデルID", text: $config.model, prompt: Text("例：gpt-4.1-mini")) }
                HStack {
                    Button("モデル一覧を取得") { perform(test: false) }
                    Button("接続テスト") { perform(test: true) }
                        .help("短い文を1回送ります。利用料金がかかる場合があります。")
                        .disabled(config.model.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    if working { ProgressView().controlSize(.small) }
                    Spacer()
                }.disabled(working || needsKey)
                if !models.isEmpty {
                    FieldRow("絞り込み") { TextField("絞り込み", text: $modelSearch, prompt: Text("モデル名の一部")) }
                    Picker("取得したモデル（\(matchingModels.count)件）", selection: $config.model) {
                        Text("選択してください").tag("")
                        if !config.model.isEmpty && !matchingModels.contains(config.model) { Text(config.model).tag(config.model) }
                        ForEach(matchingModels, id: \.self) { Text($0).tag($0) }
                    }
                }
                if let status {
                    Label(status, systemImage: failed ? "xmark.octagon.fill" : "checkmark.circle.fill")
                        .foregroundStyle(failed ? Color.red : Color.green).textSelection(.enabled)
                }
            }
            Section {
                DisclosureGroup("詳細設定") {
                    if config.protocol == "chat" {
                        Picker("トークン上限の指定", selection: $config.chatTokenParameter) {
                            Text("max_completion_tokens").tag("max_completion_tokens")
                            Text("max_tokens（互換サーバー用）").tag("max_tokens")
                        }
                    }
                    if config.protocol == "chat" || config.protocol == "responses" {
                        Picker("推論の強さ（effort）", selection: $config.reasoningEffort) {
                            Text("APIの既定値").tag("")
                            Text("なし（none）").tag("none")
                            Text("低（low）").tag("low")
                            Text("中（medium）").tag("medium")
                            Text("高（high）").tag("high")
                        }
                        if !config.reasoningEffort.isEmpty && config.reasoningEffort != "none" {
                            FieldRow("推論用の追加トークン") {
                                TextField("推論用の追加トークン", value: $config.reasoningTokenBudget, format: .number)
                            }
                            Text("思考を含む出力上限に加算します。モデルが対応するeffortとトークン数を指定してください。")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    FieldRow("送信URL") { TextField("送信URL", text: $config.inferenceUrl, prompt: Text("空欄ならベースURLから決定")) }
                    FieldRow("モデル一覧URL") { TextField("モデル一覧URL", text: $config.modelsUrl, prompt: Text("空欄ならベースURLから決定")) }
                    Text("外部APIはHTTPSが必要です。HTTPはlocalhost・127.0.0.1・::1でだけ使えます。キーはURLに含めず、APIキー欄に入力してください。")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
        }
        .onAppear { originalScope = config.credentialScope }
        .onChange(of: config.protocol) { _, value in
            if value != "chat" && value != "responses" { config.reasoningEffort = "" }
        }
        .onChange(of: config.credentialScope) { _, _ in status = nil; models = [] }
        .onChange(of: config.model) { _, _ in status = nil }
        .onChange(of: config.apiKey) { _, _ in status = nil }
        .onChange(of: config.reasoningEffort) { _, _ in status = nil }
        .onChange(of: config.reasoningTokenBudget) { _, _ in status = nil }
    }

    private func perform(test: Bool) {
        working = true
        status = nil
        var draft = config
        draft.enabled = false
        Task {
            do {
                let encoder = JSONEncoder()
                encoder.keyEncodingStrategy = .convertToSnakeCase
                let body = try encoder.encode(draft)
                if test {
                    let result: APIConnectionTest = try await store.request("/api/task-routing/presets/test", method: "POST", body: body)
                    status = "\(result.message)（\(result.latencyMs) ms）"
                } else {
                    let result: CustomAPIModels = try await store.request("/api/task-routing/presets/models", method: "POST", body: body)
                    models = result.models
                    status = models.isEmpty ? "モデル一覧は空です。モデルIDを手入力できます。" : "\(models.count)件のモデルを取得しました。"
                }
                failed = false
            } catch {
                failed = true
                status = error.localizedDescription
            }
            working = false
        }
    }
}

/// A labelled text field with a visible border; grouped forms otherwise hide empty fields.
struct FieldRow<Field: View>: View {
    let title: String
    @ViewBuilder let field: Field
    init(_ title: String, @ViewBuilder field: () -> Field) {
        self.title = title
        self.field = field()
    }
    var body: some View {
        LabeledContent(title) {
            field.labelsHidden().textFieldStyle(.roundedBorder).frame(maxWidth: 320)
        }
    }
}
