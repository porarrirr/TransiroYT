import SwiftUI

struct MobileDashboard: View {
    @ObservedObject var worker: MobileWorker
    @Binding var scanning: Bool
    @State private var showGuide = false
    @State private var confirmDisconnect = false

    private var stateTitle: String {
        if worker.connecting { return "接続を確認中" }
        if !worker.connected { return "Macに接続" }
        if !worker.reachable { return worker.linkProblem == nil ? "Macに接続しています" : "Macに接続できません" }
        if worker.preparing { return "モデルを準備中" }
        return worker.busy ? "処理中" : "Macと連携中"
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 28) {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("モバイル連携")
                            .font(.largeTitle.weight(.semibold)).tracking(-1)
                    }.padding(.top, 24)

                    connectionCard

                    if let error = worker.error {
                        VStack(alignment: .leading, spacing: 12) {
                            Label("操作を確認してください", systemImage: "exclamationmark.triangle")
                                .font(.headline).foregroundStyle(.red)
                            Text(error).font(.callout).textSelection(.enabled)
                            if worker.connected && !worker.preparing {
                                Button("モデルの準備を再試行") { worker.retryPreparation() }
                                    .buttonStyle(.bordered).disabled(worker.busy)
                            }
                        }.dashboardCard()
                    }

                    if worker.connected || worker.completed > 0 {
                        activityCard
                    }


                }
                .frame(maxWidth: 620).padding(.horizontal, 24).padding(.bottom, 32)
                .frame(maxWidth: .infinity)
            }
            .background(Color(uiColor: .systemGroupedBackground))
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    HStack(spacing: 8) {
                        Image("BrandIcon").resizable().frame(width: 28, height: 28)
                            .clipShape(RoundedRectangle(cornerRadius: 7)).accessibilityHidden(true)
                        Text("TransiroYT").font(.headline)
                    }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Menu {
                        Button("使い方と利用条件", systemImage: "questionmark.circle") { showGuide = true }
                        if worker.connected {
                            Button("接続を解除", systemImage: "wifi.slash", role: .destructive) { confirmDisconnect = true }
                        }
                    } label: { Image(systemName: "ellipsis") }
                    .accessibilityLabel("その他")
                }
            }
            .confirmationDialog("Macとの接続を解除しますか？", isPresented: $confirmDisconnect, titleVisibility: .visible) {
                Button("接続を解除", role: .destructive) { worker.disconnect() }
            } message: { Text("処理はMacに戻ります。再び連携するにはQRコードを読み取ってください。") }
            .sheet(isPresented: $showGuide) { guide }
        }.tint(.yellow)
    }

    private var connectionCard: some View {
        VStack(alignment: .leading, spacing: 20) {
            HStack(alignment: .top, spacing: 16) {
                Image(systemName: worker.connected ? "desktopcomputer" : "qrcode.viewfinder")
                    .font(.system(size: 28)).foregroundStyle(.primary)
                    .frame(width: 56, height: 56)
                    .background(.yellow.opacity(0.18), in: RoundedRectangle(cornerRadius: 14))
                VStack(alignment: .leading, spacing: 6) {
                    Text(stateTitle).font(.title2.weight(.semibold))
                    if worker.connected || worker.connecting {
                        Text(worker.reachable ? worker.status : worker.linkProblem ?? "Macの応答を待っています")
                            .font(.callout).foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
            if worker.preparing || worker.connecting {
                HStack(spacing: 10) {
                    ProgressView().tint(.primary)
                    Text(worker.preparing ? "準備中" : "接続中")
                        .font(.footnote).foregroundStyle(.secondary)
                }
            } else if !worker.connected {
                Button { scanning = true } label: {
                    Label("QRコードを読み取る", systemImage: "qrcode.viewfinder")
                        .font(.headline).frame(maxWidth: .infinity).padding(.vertical, 6)
                }
                .buttonStyle(.borderedProminent).foregroundStyle(.black).controlSize(.large)
                .accessibilityIdentifier("scanPairingQRCode")
            } else if worker.reachable {
                Label("接続済み", systemImage: "checkmark.circle.fill")
                    .font(.footnote).foregroundStyle(.secondary)
            } else {
                Label("Macと通信できていません", systemImage: "exclamationmark.triangle")
                    .font(.footnote).foregroundStyle(.secondary)
            }
        }.dashboardCard()
    }

    private var activityCard: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack {
                Text("この端末の作業").font(.headline)
                Spacer()
                Text("Macと接続してから").font(.caption).foregroundStyle(.secondary)
            }
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Text(worker.completed, format: .number).font(.system(.largeTitle, design: .rounded).weight(.semibold))
                    .contentTransition(.numericText())
                Text("件完了").font(.callout).foregroundStyle(.secondary)
            }.accessibilityElement(children: .combine)
            Divider()
            LabeledContent("音声生成", value: worker.capabilities["tts"] != nil ? "準備完了" : "未準備")
            if worker.connected && !worker.preparing && worker.capabilities.isEmpty && worker.error == nil {
                Button("モデルの準備を再試行") { worker.retryPreparation() }.disabled(worker.busy)
            }
        }.font(.callout).dashboardCard()
    }

    private var guide: some View {
        NavigationStack {
            List {
                Section("接続方法") {
                    Text("Macと同じWi-Fi、またはUSBケーブルで接続し、Macの「… → モバイル連携 → 端末を追加」に表示されるQRコードを読み取ります。")
                }
                Section("初回の準備") {
                    Text("音声モデル約2GBを取得します。取得済みのモデルは再利用します。")
                }
                Section("連携中の動作") {
                    Text("接続中は画面の自動ロックを止めます。低電力モード・画面を閉じた場合はMacに処理を戻します。端末が熱くなっても処理は続けます。")
                    Text("動画の追加と吹き替えの開始はMacで行います。端末が担当する処理はMacが自動で決めます。")
                }
                Section("データと利用条件") {
                    Text("モデルは端末内に保存します。処理対象の音声・字幕は、接続先のMacとだけ送受信します。")
                    Link("音声モデルの利用条件", destination: URL(string: "https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small-MF#-license--ethical-restrictions")!)
                }
            }
            .navigationTitle("使い方と利用条件").navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("閉じる") { showGuide = false } } }
        }
    }
}

private extension View {
    func dashboardCard() -> some View {
        self.padding(20).frame(maxWidth: .infinity, alignment: .leading)
            .background(Color(uiColor: .secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 16))
            .overlay(RoundedRectangle(cornerRadius: 16).strokeBorder(.primary.opacity(0.06)))
    }
}
