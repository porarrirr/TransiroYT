import SwiftUI

/// Every AI the tasks can use, in one place: what it is, whether it is ready, and where it is used.
struct AIServicesPage: View {
    @ObservedObject var store: BackendStore
    @Binding var config: TaskRoutingSettings
    @Binding var sarashina: SarashinaSettings
    /// The service whose page is open (`api:<id>` or an engine name); nil shows the list.
    @Binding var selected: String?
    @State private var adding = false

    var body: some View {
        Group {
            if let selected { ServiceDetail(store: store, provider: selected, config: $config, sarashina: $sarashina) { self.selected = nil } }
            else { list }
        }
        .sheet(isPresented: $adding) { AddAPISheet(store: store, config: $config) }
    }

    private var list: some View {
        Form {
            Section {
                ForEach(config.presets) { preset in row("api:\(preset.id)", icon: ["localhost", "127.0.0.1", "::1", "[::1]"].contains(preset.host) ? "laptopcomputer" : "cloud") }
                HStack {
                    Button("APIを追加…", systemImage: "plus") { adding = true }
                        .disabled(config.presets.count >= 16)
                    Spacer()
                    Text("\(config.presets.count) / 16件").font(.caption).monospacedDigit().foregroundStyle(.secondary)
                }
            } header: {
                Text("自分のAPI")
            }
            Section("翻訳サービス") {
                ForEach(["sarashina", "google", "bing"], id: \.self) { row($0, icon: BuiltInEngine.named($0).icon) }
            }
            Section("このMac・Apple") {
                ForEach(["plamo", "gemma"], id: \.self) { row($0, icon: BuiltInEngine.named($0).icon) }
            }
        }
        .formStyle(.grouped)
    }

    private func row(_ provider: String, icon: String) -> some View {
        let issue = config.setupIssue(provider, sarashinaReady: sarashina.ready)
        let used = RoutingTask.allCases.filter { config.chain($0).contains(provider) }.map(\.title)
        return Button { selected = provider } label: {
            HStack(spacing: 12) {
                Image(systemName: icon).frame(width: 22).foregroundStyle(.secondary)
                VStack(alignment: .leading, spacing: 2) {
                    Text(config.name(provider)).fontWeight(.medium)
                    Text(config.detail(provider)).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                }
                Spacer()
                if let issue {
                    Label(issue, systemImage: "exclamationmark.triangle.fill").font(.caption).foregroundStyle(.orange)
                } else {
                    Text(used.isEmpty ? "未使用" : used.joined(separator: "・")).font(.caption).foregroundStyle(.secondary)
                }
                Image(systemName: "chevron.right").font(.caption.weight(.semibold)).foregroundStyle(.tertiary)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel("\(config.name(provider))の設定")
    }
}

struct ServiceDetail: View {
    @ObservedObject var store: BackendStore
    let provider: String
    @Binding var config: TaskRoutingSettings
    @Binding var sarashina: SarashinaSettings
    let back: () -> Void
    @State private var removing = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Button(action: back) { Label("AIサービス", systemImage: "chevron.left") }
                .buttonStyle(.borderless).padding(.horizontal, SettingsLayout.inset).padding(.top, 8)
            Form {
                if let index = config.presets.firstIndex(where: { "api:\($0.id)" == provider }) {
                    PresetEditor(store: store, config: $config.presets[index])
                    usage
                    Section {
                        HStack {
                            Button("複製") { config.presets.append(config.presets[index].duplicate()) }
                            Text("キーは複製しません。").font(.caption).foregroundStyle(.secondary)
                            Spacer()
                            Button("このAPIを削除…", role: .destructive) { removing = true }
                        }
                    }
                } else {
                    let engine = BuiltInEngine.named(provider)
                    Section {
                        LabeledContent("処理する場所", value: engine.place)
                        Text(engine.summary).fixedSize(horizontal: false, vertical: true)
                    }
                    if provider == "sarashina" { sarashinaKey }
                    usage
                }
            }
            .formStyle(.grouped)
        }
        .confirmationDialog("「\(config.name(provider))」を削除しますか？", isPresented: $removing) {
            Button("削除", role: .destructive) {
                back()
                config.removePreset(String(provider.dropFirst(4)))
            }
        } message: {
            Text("使っている作業からも外れます。「保存」すると、このAPIのキーをキーチェーンから削除します。")
        }
    }

    private var usage: some View {
        let issue = config.setupIssue(provider, sarashinaReady: sarashina.ready)
        return Section {
            ForEach(RoutingTask.allCases.filter { config.candidates($0).contains(provider) }) { task in
                let used = config.chain(task).contains(provider)
                Toggle(task.title, isOn: Binding(get: { used }, set: { config.setUse(provider, for: task, $0) }))
                    .disabled(!used && issue != nil)
            }
        } header: {
            Text("使う作業")
        } footer: {
            Text(issue.map { "\($0)のため、まだ使えません。" } ?? "オンにすると順番の最後に追加します。")
                .frame(maxWidth: .infinity, alignment: .leading).multilineTextAlignment(.leading)
        }
    }

    private var sarashinaKey: some View {
        Section("APIキー") {
            FieldRow("APIキー") {
                SecureField("APIキー", text: Binding(get: { sarashina.apiKey ?? "" }, set: { sarashina.apiKey = $0.isEmpty ? nil : $0 }),
                            prompt: Text(sarashina.hasKey ? "保存済み（変更するときだけ入力）" : "キーを入力"))
            }
            LabeledContent("状態", value: sarashina.apiKey == "" ? "保存時に削除" : sarashina.apiKey != nil ? "保存時に更新"
                                        : sarashina.hasKey ? "保存済み" : "未設定")
            if sarashina.hasKey && sarashina.apiKey != "" {
                Button("保存済みのキーを削除", role: .destructive) { sarashina.apiKey = "" }
            }
        }
    }
}

/// Adding an API: pick the kind of service, then fill in the key and model and choose whether to translate with it.
struct AddAPISheet: View {
    @Environment(\.dismiss) private var dismiss
    @ObservedObject var store: BackendStore
    @Binding var config: TaskRoutingSettings
    @State private var draft = APIPreset()
    @State private var choosing = true
    @State private var translate = true

    private var hasModel: Bool { !draft.model.trimmingCharacters(in: .whitespaces).isEmpty }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(choosing ? "追加するAPIを選びます" : "\(draft.name)を設定します")
                .font(.title3.weight(.semibold)).padding(.horizontal, SettingsLayout.inset).padding(.top, 20).padding(.bottom, 4)
            if choosing { chooser } else { editor }
            Divider()
            HStack {
                if !choosing { Button("戻る") { choosing = true } }
                if !choosing && translate && !hasModel {
                    Text("翻訳に使うにはモデルIDを入力してください。").font(.callout).foregroundStyle(.orange)
                }
                Spacer()
                Button("キャンセル") { dismiss() }.keyboardShortcut(.cancelAction)
                if !choosing {
                    Button("追加") {
                        config.presets.append(draft)
                        if translate { config.setUse("api:\(draft.id)", for: .translation, true) }
                        dismiss()
                    }
                    .buttonStyle(.borderedProminent).tint(.yellow).keyboardShortcut(.defaultAction)
                    .disabled(translate && !hasModel)
                }
            }.padding(.horizontal, SettingsLayout.inset).padding(.vertical, 12)
        }
        .frame(width: 600, height: 600)
    }

    private var chooser: some View {
        List(APIConnectionTemplate.allCases) { template in
            Button {
                draft = template.preset
                choosing = false
            } label: {
                HStack(spacing: 14) {
                    Image(systemName: template.icon).font(.title2).frame(width: 32).foregroundStyle(.secondary)
                    VStack(alignment: .leading, spacing: 3) {
                        Text(template.rawValue).font(.headline)
                        Text(template.summary).foregroundStyle(.secondary)
                    }
                    Spacer()
                    Image(systemName: "chevron.right").foregroundStyle(.tertiary)
                }
                .padding(.vertical, 8).contentShape(Rectangle())
            }
            .buttonStyle(.plain)
        }
        .listStyle(.bordered(alternatesRowBackgrounds: false))
        .padding(.horizontal, SettingsLayout.inset).padding(.vertical, 12)
    }

    private var editor: some View {
        Form {
            PresetEditor(store: store, config: $draft)
            Section {
                Toggle("翻訳に使う（翻訳の順番の最後に追加）", isOn: $translate)
            }
        }
        .formStyle(.grouped)
    }
}
