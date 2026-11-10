import SwiftUI
import AppKit
import CoreImage.CIFilterBuiltins

struct MobileWorkersStatus: Decodable {
    let enabled: Bool
    let pairingUrl: String?
    let pairingTransport: String?
    let pairingSeconds: Int
    let devices: [MobileWorkerDevice]
}

struct MobileWorkerDevice: Decodable, Identifiable {
    let id: String
    let name: String
    let online: Bool
    let state: String
    let task: String?
    let completed: Int
    let failures: Int
    let capabilities: [String]

    var displayState: String {
        if let task {
            return task == "tts" ? "音声を生成中" : "翻訳・音声分析中"
        }
        return state
    }
}

struct MobileWorkersView: View {
    @ObservedObject var store: BackendStore
    @Environment(\.dismiss) private var dismiss
    @State private var status: MobileWorkersStatus?
    @State private var error: String?
    @State private var busy = false
    @State private var transport = "wifi"

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack {
                Label("モバイル連携", systemImage: "iphone.and.ipad").font(.title2.bold())
                Spacer()
                Text("\(status?.devices.count ?? 0) / 2台").foregroundStyle(.secondary)
            }
            Text("iPhone・iPadで連携アプリを開いてください。準備ができた端末に、Macが処理を自動で割り振ります。")
            Picker("接続方法", selection: $transport) {
                Text("Wi-Fi").tag("wifi")
                Text("USBケーブル").tag("usb")
            }.pickerStyle(.segmented).disabled(busy)
            Text(transport == "usb"
                 ? "USBで接続してiPhoneの「信頼」を許可します。Wi-Fi・インターネット共有は不要です。接続方法を変えたらQRコードを再発行してください。"
                 : "Macと端末を同じWi-Fiに接続してください。接続方法を変えたらQRコードを再発行してください。")
                .font(.callout).foregroundStyle(.secondary)
            if let status {
                if status.enabled, let current = status.pairingTransport {
                    Text("連携中の接続方法：\(current == "usb" ? "USBケーブル" : "Wi-Fi")")
                        .font(.callout)
                }
                ForEach(status.devices) { device in
                    HStack(alignment: .top) {
                        Image(systemName: device.online ? "checkmark.circle.fill" : "wifi.slash")
                            .foregroundStyle(device.online ? Color.green : Color.secondary)
                        VStack(alignment: .leading, spacing: 4) {
                            Text(device.name).font(.headline)
                            Text(device.online ? device.displayState : "応答なし・Macで処理を続けます")
                                .font(.callout).foregroundStyle(.secondary)
                            Text("完了 \(device.completed)件・Macへ回収 \(device.failures)件")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        Spacer()
                        Button("解除") { change("DELETE", path: "/api/mobile/\(device.id)") }
                            .disabled(busy)
                    }.padding(12).background(.quaternary, in: RoundedRectangle(cornerRadius: 8))
                }
                if let url = status.pairingUrl, status.pairingTransport == transport {
                    Text("スマホの「QRコードを読み取る」で接続（あと\(status.pairingSeconds)秒）").font(.callout)
                    if let image = pairingQRCode(url) {
                        Image(nsImage: image).interpolation(.none).resizable()
                            .frame(width: 220, height: 220)
                            .padding(12).background(.white)
                            .accessibilityLabel("スマホ連携用QRコード")
                    }
                }
                HStack {
                    Button(status.enabled ? "QRコードを再発行" : "端末を追加") { change("POST", path: "/api/mobile?transport=\(transport)") }
                        .buttonStyle(.borderedProminent).disabled(busy || status.devices.count >= 2)
                    if status.enabled {
                        Button("連携を停止") { change("DELETE") }.disabled(busy)
                    }
                }
            } else { ProgressView() }
            Text("端末はアプリを前面にしたまま使います。初回は音声生成モデル（約2GB）を取得します。端末は音声生成だけを担当します。画面ロック・低電力モード・切断時はMacに戻します。")
                .font(.caption).foregroundStyle(.secondary)
            if let error { Text(error).foregroundStyle(.red).font(.callout) }
            Spacer(minLength: 0)
            HStack { Spacer(); Button("閉じる") { dismiss() }.keyboardShortcut(.cancelAction) }
        }.padding(24).frame(width: 580, height: 820).tint(.yellow)
        .task {
            while !Task.isCancelled {
                do {
                    let fetched: MobileWorkersStatus = try await store.request("/api/mobile")
                    // The picker starts at the method the saved pairing uses.
                    if status == nil, let current = fetched.pairingTransport { transport = current }
                    status = fetched
                    error = nil
                } catch { self.error = error.localizedDescription }
                try? await Task.sleep(for: .seconds(2))
            }
        }
    }

    private func change(_ method: String, path: String = "/api/mobile") {
        busy = true
        Task {
            defer { busy = false }
            do {
                status = try await store.request(path, method: method)
                error = nil
            } catch { self.error = error.localizedDescription }
        }
    }
}

private func pairingQRCode(_ value: String) -> NSImage? {
    let filter = CIFilter.qrCodeGenerator()
    filter.message = Data(value.utf8)
    filter.correctionLevel = "M"
    guard let output = filter.outputImage,
          let image = CIContext().createCGImage(output, from: output.extent) else { return nil }
    return NSImage(cgImage: image, size: NSSize(width: image.width, height: image.height))
}
