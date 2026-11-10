import SwiftUI

struct DubbingProgressView: View {
    @ObservedObject var store: BackendStore
    @State private var workers: MobileWorkersStatus?
    @State private var workersUnavailable = false
    let job: FileDubbingJob
    let cancel: () -> Void
    private let stages = [("取得", "arrow.down"), ("音声の分析", "waveform.path"), ("話者の判定", "person.wave.2"), ("翻訳・音声生成", "waveform"), ("動画の書き出し", "film.stack")]
    private var current: Int {
        switch job.stage {
        case "download", "model_download": 0
        case "prepare": 1
        case "speakers", "reference_conversion": 2
        case "translation", "synthesis": 3
        default: 4
        }
    }
    private var progress: Double? {
        if job.stage == "download" { return job.downloads?["youtube"]?.fraction }
        if current == 3 { return job.activities?["synthesis"]?.fraction }
        if job.stage == "model_download" || job.stage == "mix" { return nil }
        return job.fraction
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 30) {
            header
            stageOverview
            activityProgress
            workerProgress
            downloadProgress
            Divider()
            HStack {
                Image(systemName: "film").font(.system(size: 12)).foregroundStyle(.tertiary)
                Text(URL(fileURLWithPath: job.output).lastPathComponent).font(.system(size: 11)).foregroundStyle(.tertiary).lineLimit(1)
                Spacer()
                Button("中止", action: cancel).buttonStyle(.plain).font(.system(size: 12)).foregroundStyle(.secondary)
            }
        }
        .task {
            while !Task.isCancelled {
                do {
                    workers = try await store.request("/api/mobile")
                    workersUnavailable = false
                } catch { workersUnavailable = true }
                do { try await Task.sleep(for: .seconds(2)) } catch { break }
            }
        }
    }

    private var header: some View {
        HStack {
            HStack(spacing: 8) {
                Circle().fill(Color.yellow).frame(width: 5, height: 5)
                Text("吹き替えを作成中").font(.system(size: 12, weight: .medium))
            }.foregroundStyle(.secondary)
            Spacer()
            if let started = job.startedAt {
                TimelineView(.periodic(from: .now, by: 1)) { context in
                    let seconds = max(0, Int(context.date.timeIntervalSince1970 - started))
                    Text(String(format: "%02d:%02d", seconds / 60, seconds % 60))
                        .monospacedDigit().font(.system(size: 12, design: .monospaced)).foregroundStyle(.secondary)
                }
            }
        }
    }

    private var stageOverview: some View {
        HStack(alignment: .top, spacing: 44) {
            VStack(alignment: .leading, spacing: 24) {
                VStack(alignment: .leading, spacing: 10) {
                    Text("\(String(format: "%02d", current + 1)) / 05")
                        .font(.system(size: 12, weight: .medium, design: .monospaced)).foregroundStyle(.secondary)
                    Text(stages[current].0).font(.system(size: 28, weight: .semibold)).tracking(-1)
                }
                Text(current == 3 ? "音声生成の進捗" : "現在の工程の進捗").font(.caption).foregroundStyle(.secondary)
                HStack(alignment: .firstTextBaseline, spacing: 4) {
                    if let progress {
                        Text("\(Int(progress * 100))").font(.system(size: 76, weight: .ultraLight, design: .rounded))
                        Text("%").font(.system(size: 22, weight: .light)).foregroundStyle(.secondary)
                    } else {
                        Text("処理中").font(.system(size: 42, weight: .ultraLight))
                    }
                }.monospacedDigit().frame(height: 88, alignment: .bottomLeading)
                VStack(alignment: .leading, spacing: 16) {
                    ProgressView(value: progress).progressViewStyle(DubbingBarStyle())
                    Text(job.message).font(.system(size: 12)).foregroundStyle(.secondary)
                        .lineSpacing(4).textSelection(.enabled).frame(minHeight: 36, alignment: .topLeading)
                }
            }.frame(maxWidth: .infinity, alignment: .leading)
            VStack(alignment: .leading, spacing: 0) {
                ForEach(stages.indices, id: \.self) { index in
                    let parallel = index == 2 && current == 3 && (job.activities?["speakers"]?.fraction ?? 1) < 1
                    let active = index == current || parallel
                    HStack(alignment: .top, spacing: 14) {
                        VStack(spacing: 6) {
                            ZStack {
                                Circle().strokeBorder(active ? Color.yellow : Color.primary.opacity(0.15), lineWidth: 1).frame(width: 24, height: 24)
                                if index < current && !parallel {
                                    Image(systemName: "checkmark").font(.system(size: 9, weight: .medium)).foregroundStyle(.secondary)
                                } else if active {
                                    Circle().fill(Color.yellow).frame(width: 5, height: 5)
                                }
                            }
                            if index < stages.count - 1 {
                                Rectangle().fill(Color.primary.opacity(0.09)).frame(width: 1, height: 28)
                            }
                        }
                        VStack(alignment: .leading, spacing: 4) {
                            Text(stages[index].0).font(.system(size: 13, weight: active ? .semibold : .regular))
                                .foregroundStyle(active ? Color.primary : .secondary)
                            if active {
                                Text(parallel ? "並行処理中" : "処理中").font(.system(size: 10, weight: .medium)).foregroundStyle(.yellow)
                            }
                        }.padding(.top, 4)
                    }.frame(height: 64, alignment: .top)
                }
            }.frame(width: 192, alignment: .leading)
        }
    }

    private var activityProgress: some View {
        GroupBox("工程別の進み具合") {
            VStack(alignment: .leading, spacing: 16) {
                ForEach([("speakers", "話者の判定"), ("reference_conversion", "参照音声の調整"), ("translation", "翻訳"), ("synthesis", "音声生成")], id: \.0) { stage, title in
                    let activity = job.activities?[stage]
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text(title).font(.headline)
                            Spacer()
                            if let activity {
                                Text(activity.fraction >= 1 ? "完了" : activity.fraction.formatted(.percent.precision(.fractionLength(0))))
                                    .monospacedDigit()
                            } else {
                                Text("待機中").foregroundStyle(.secondary)
                            }
                        }
                        ProgressView(value: activity?.fraction ?? 0).tint(.yellow)
                        if let activity {
                            Text(activity.message).font(.callout).foregroundStyle(.secondary).textSelection(.enabled)
                            if activity.fraction < 1 {
                                TimelineView(.periodic(from: .now, by: 1)) { context in
                                    let seconds = max(0, Int(context.date.timeIntervalSince1970 - activity.updatedAt))
                                    Text("進捗更新から\(seconds)秒")
                                        .font(.caption).foregroundStyle(.secondary).monospacedDigit()
                                }
                            }
                        }
                    }
                }
            }.padding(8)
        }
    }

    @ViewBuilder private var workerProgress: some View {
        if let workers, !workers.devices.isEmpty {
            GroupBox {
                VStack(alignment: .leading, spacing: 12) {
                    Label("Macとモバイルの分担", systemImage: "iphone.and.ipad").font(.headline)
                    Text("Macが翻訳・音声生成を自動で割り振ります。完了件数は接続してからの累計です。")
                        .font(.caption).foregroundStyle(.secondary)
                    ForEach(workers.devices) { device in
                        HStack {
                            Image(systemName: device.online ? "iphone" : "wifi.slash")
                            VStack(alignment: .leading, spacing: 4) {
                                Text(device.name).font(.callout.weight(.medium))
                                Text(device.online ? device.displayState : "切断・Macで処理を続けます")
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                            Spacer()
                            Text("完了 \(device.completed)件").monospacedDigit()
                        }
                    }
                    if workersUnavailable {
                        Text("端末の状態を更新できません。前回の情報を表示しています。")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }.padding(8)
            }
        } else if workersUnavailable {
            Text("モバイル連携の状態を取得できません。")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    @ViewBuilder private var downloadProgress: some View {
        if let downloads = job.downloads, !downloads.isEmpty {
            Divider()
            VStack(alignment: .leading, spacing: 14) {
                ForEach(downloads.sorted(by: { $0.key < $1.key }), id: \.key) { key, download in
                    HStack(spacing: 16) {
                        Image(systemName: ["complete", "ready"].contains(download.state) ? "checkmark" : "arrow.down").font(.system(size: 10)).foregroundStyle(.secondary).frame(width: 16)
                        Text(["coreml": "Core ML音声モデル", "gemma": "Gemmaローカル代替モデル", "plamo": "PLaMo 2B翻訳モデル", "speaker_model": "話者照合モデル", "diarization_model": "話者区間モデル（Nemotron）", "separation_model": "音源分離モデル", "speech_check_model": "音声確認モデル", "indextts_model": "参照音声モデル（IndexTTS-2.5）", "youtube": "動画・字幕"][key] ?? key)
                            .font(.system(size: 12)).foregroundStyle(.secondary)
                        Spacer()
                        if download.state != "downloading" {
                            Text(download.detail).font(.system(size: 11)).monospacedDigit().foregroundStyle(.secondary)
                        }
                    }
                    if download.state == "downloading" {
                        DownloadActivityView(downloaded: Double(download.downloaded), total: download.total.map(Double.init),
                                             speed: download.speed, eta: download.eta, updatedAt: download.updatedAt)
                            .id(job.id + key)
                    }
                    if !["complete", "ready", "downloading"].contains(download.state) {
                        ProgressView(value: download.state == "preparing" ? nil : download.fraction).progressViewStyle(DubbingBarStyle())
                    }
                }
            }
        }
    }
}

private struct DubbingBarStyle: ProgressViewStyle {
    @State private var moving = false
    func makeBody(configuration: Configuration) -> some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                Capsule().fill(Color.primary.opacity(0.08))
                if let fraction = configuration.fractionCompleted {
                    Capsule().fill(Color.yellow.gradient).frame(width: geometry.size.width * fraction)
                } else {
                    Capsule().fill(Color.yellow.opacity(0.8)).frame(width: geometry.size.width * 0.2)
                        .offset(x: moving ? geometry.size.width * 0.8 : 0)
                        .animation(.easeInOut(duration: 1.8).repeatForever(autoreverses: true), value: moving)
                        .onAppear { moving = true }
                }
            }
        }.frame(height: 3)
    }
}
