import AppKit
import SwiftUI
import UniformTypeIdentifiers

struct FileDubbingView: View {
    @ObservedObject var store: BackendStore
    var preferencesRevision = 0
    @State private var youtubeURL = ""
    @State private var youtubeQuality = "best"
    @State private var video: URL?
    @State private var subtitles: URL?
    @State private var useFile = false
    @State private var submitting = false
    @State private var showQueueInput = false
    @State private var dropTarget = false
    private var job: FileDubbingJob? { store.fileDubbing?.job }
    private var running: Bool { job?.running == true }
    private var inputReady: Bool { useFile ? video != nil : !youtubeURL.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 28) {
                if let job, running {
                    DubbingProgressView(store: store, job: job) { Task { await store.cancelFileDubbing(job.id) } }
                }
                if running {
                    Button(showQueueInput ? "追加を閉じる" : "追加", systemImage: "plus") { showQueueInput.toggle() }
                }
                if !running || showQueueInput { input }
                if !store.dubbingQueue.isEmpty {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("待機中（\(store.dubbingQueue.count)件）").font(.headline)
                        Text("追加した順に処理します。失敗・中止すると停止します。アプリを終了すると待機中の項目は解除されます。")
                            .font(.caption).foregroundStyle(.secondary)
                        ForEach(store.dubbingQueue) { item in
                            HStack {
                                Text(item.title).lineLimit(2).textSelection(.enabled)
                                Spacer()
                                Button("削除", systemImage: "minus.circle") { store.dubbingQueue.removeAll { $0.id == item.id } }
                            }
                        }
                        if !running {
                            Button("次の動画を開始") { Task { await store.startNextDubbing() } }
                                .disabled(!store.canStartFileDubbing || store.submittingDubbing)
                        }
                    }
                }
                if let job, !running { completed(job) }
            }.frame(maxWidth: 860).padding(.horizontal, 44).padding(.top, running ? 36 : 72).padding(.bottom, 40)
                .frame(maxWidth: .infinity)
        }
        .task(id: preferencesRevision) {
            do { youtubeQuality = try ProductPreferences.load().youtubeQuality }
            catch { store.errorMessage = error.localizedDescription }
        }
    }

    private var input: some View {
        VStack(alignment: .leading, spacing: 32) {
            Text(running ? "キューに追加" : "日本語吹き替え")
                .font(.system(size: 40, weight: .semibold, design: .default)).tracking(-1.5)
                .padding(.bottom, 10)
            HStack(spacing: 24) {
                sourceButton("URL", symbol: "link", file: false)
                sourceButton("ファイル", symbol: "film", file: true)
            }
            if useFile {
                VStack(spacing: 20) {
                    HStack(spacing: 20) {
                        Image(systemName: video == nil ? "arrow.down.document" : "film")
                            .font(.system(size: 30, weight: .ultraLight)).foregroundStyle(.secondary)
                        VStack(alignment: .leading, spacing: 8) {
                            Text(video?.lastPathComponent ?? "動画をドロップ")
                                .font(.system(size: 18, weight: .medium)).lineLimit(2).truncationMode(.middle)
                            Button(video == nil ? "ファイルを選択" : "変更") { selectVideo() }.buttonStyle(.link)
                        }
                        Spacer()
                    }.padding(28)
                        .frame(maxWidth: .infinity, minHeight: 130, alignment: .leading)
                        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
                        .overlay(RoundedRectangle(cornerRadius: 14).strokeBorder(dropTarget ? Color.yellow : Color.primary.opacity(0.1), style: StrokeStyle(lineWidth: 1, dash: [5, 5])))
                        .dropDestination(for: URL.self) { urls, _ in
                            for url in urls where url.isFileURL {
                                if url.pathExtension.lowercased() == "srt" { subtitles = url }
                                else { video = url; useFile = true }
                            }
                            return !urls.isEmpty
                        } isTargeted: { dropTarget = $0 }
                    HStack {
                        if let subtitles {
                            Label(subtitles.lastPathComponent, systemImage: "captions.bubble").lineLimit(1)
                            Button { self.subtitles = nil } label: { Image(systemName: "xmark") }
                                .buttonStyle(.plain).accessibilityLabel("外部字幕を解除")
                        } else {
                            VStack(alignment: .leading, spacing: 4) {
                                Button("SRTを追加", systemImage: "plus") { selectSubtitles() }.buttonStyle(.link)
                                Text("SRTも動画内の字幕もない場合は、音声から字幕を作成します。")
                                    .font(.caption).foregroundStyle(.tertiary)
                            }
                        }
                        Spacer()
                        startButton
                    }.font(.callout).foregroundStyle(.secondary)
                }
            } else {
                VStack(alignment: .leading, spacing: 16) {
                    HStack(spacing: 16) {
                        HStack(spacing: 14) {
                            Image(systemName: "link").font(.system(size: 18, weight: .light)).foregroundStyle(.secondary)
                            TextField("YouTube URL", text: $youtubeURL, prompt: Text("YouTubeのURLを貼り付け"))
                                .textFieldStyle(.plain).font(.system(size: 18)).onSubmit { start() }
                                .accessibilityLabel("YouTube URL")
                            if !youtubeURL.isEmpty {
                                Button { youtubeURL = "" } label: { Image(systemName: "xmark.circle.fill") }
                                    .buttonStyle(.plain).foregroundStyle(.tertiary).accessibilityLabel("URLを消去")
                            }
                        }.padding(.horizontal, 20).frame(height: 62)
                            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
                            .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(Color.primary.opacity(0.12)))
                        startButton
                    }
                    VStack(alignment: .leading, spacing: 6) {
                        Picker("ダウンロード画質", selection: Binding(get: { youtubeQuality }, set: { quality in
                            do {
                                var preferences = try ProductPreferences.load()
                                preferences.youtubeQuality = quality
                                try preferences.save()
                                youtubeQuality = quality
                            } catch { store.errorMessage = error.localizedDescription }
                        })) {
                            Text("最高画質（自動）").tag("best")
                            Text("2160p（4K）").tag("2160")
                            Text("1440p").tag("1440")
                            Text("1080p").tag("1080")
                            Text("720p").tag("720")
                            Text("480p").tag("480")
                            Text("360p").tag("360")
                        }
                        .pickerStyle(.menu).fixedSize().font(.callout)
                        Text(youtubeQuality == "best" ? "動画で利用できる最高画質を取得します。" : "指定した画質以下で、利用できる最高画質を取得します。")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
            }
            if !store.connected {
                Label(store.starting ? "接続中" : "未接続", systemImage: "circle.dotted")
                    .font(.caption).foregroundStyle(.tertiary)
            }
            Text("モデルは初回に自動取得します（約21 GB）。")
                .font(.caption).foregroundStyle(.secondary)
            Button("初回の準備と利用条件") { store.showGuide = true }
                .buttonStyle(.link).font(.caption)
            if store.installation?.isOnDiskImage == true {
                Text("アプリを「アプリケーション」フォルダへ移してから開いてください。")
                    .font(.callout).foregroundStyle(.secondary)
            }
        }
    }

    private var startButton: some View {
        Button(action: start) {
            HStack(spacing: 10) {
                if submitting { ProgressView().controlSize(.small) }
                else { Image(systemName: "arrow.right") }
                Text(submitting ? "開始中" : running || !store.dubbingQueue.isEmpty ? "追加" : "開始").fontWeight(.semibold)
            }.font(.system(size: 16)).frame(width: 104, height: 44)
        }.buttonStyle(.borderedProminent).tint(.yellow).controlSize(.large)
            .disabled(!inputReady || submitting || store.submittingDubbing || !store.canStartFileDubbing).keyboardShortcut(.defaultAction)
    }

    private func sourceButton(_ title: String, symbol: String, file: Bool) -> some View {
        Button { useFile = file } label: {
            HStack(spacing: 8) {
                Image(systemName: symbol).font(.system(size: 13))
                Text(title).font(.system(size: 14, weight: .medium))
            }.foregroundStyle(useFile == file ? Color.primary : .secondary)
                .padding(.bottom, 10)
                .overlay(alignment: .bottom) {
                    if useFile == file { Capsule().fill(Color.yellow).frame(height: 2) }
                }
        }.buttonStyle(.plain)
    }

    @ViewBuilder private func completed(_ job: FileDubbingJob) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            Divider()
            Label(job.state == "done" ? "完了" : job.state == "error" ? "作成失敗" : "中止", systemImage: job.state == "done" ? "checkmark.circle" : "exclamationmark.circle")
                .font(.title2.weight(.semibold))
            if job.state == "done" {
                Text(URL(fileURLWithPath: job.output).lastPathComponent).foregroundStyle(.secondary)
                if let summary = job.summary {
                    HStack(spacing: 24) {
                        Text("吹き替え \(summary.ready)区間")
                        if summary.skipped > 0 { Text("元音声 \(summary.skipped)区間") }
                    }.font(.callout).monospacedDigit()
                }
                HStack {
                    Button("再生", systemImage: "play.fill") { NSWorkspace.shared.open(URL(fileURLWithPath: job.output)) }
                    Button("Finderで表示", systemImage: "folder") { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: job.output)]) }
                }
            } else if let error = job.error { Text(error).textSelection(.enabled).foregroundStyle(.secondary) }
        }
    }

    private func selectVideo() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.movie, UTType(filenameExtension: "mkv") ?? .data]
        if panel.runModal() == .OK { video = panel.url }
    }

    private func selectSubtitles() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [UTType(filenameExtension: "srt") ?? .plainText]
        if panel.runModal() == .OK { subtitles = panel.url }
    }

    private func start() {
        guard inputReady, !submitting, !store.submittingDubbing, store.canStartFileDubbing else { return }
        do {
            let preferences = try ProductPreferences.load()
            let directory = URL(fileURLWithPath: (preferences.outputDirectory as NSString).expandingTildeInPath, isDirectory: true)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let name = useFile ? video!.deletingPathExtension().lastPathComponent : "YouTube"
            let output = directory.appendingPathComponent("\(name)-\(UUID().uuidString.prefix(8)).ja.mp4")
            let request = FileDubbingRequest(video: useFile ? video?.path : nil, subtitles: useFile ? subtitles?.path : nil,
                                            output: output.path, originalVolume: preferences.originalVolume, dubVolume: preferences.dubVolume,
                                            speakers: preferences.speakers, steps: preferences.synthesisSteps,
                                            allowDownload: true, overwrite: false,
                                            youtubeUrl: useFile ? nil : youtubeURL.trimmingCharacters(in: .whitespacesAndNewlines),
                                            youtubeQuality: useFile ? nil : preferences.youtubeQuality)
            if running || !store.dubbingQueue.isEmpty {
                store.enqueueDubbing(request, title: useFile ? video!.lastPathComponent : youtubeURL.trimmingCharacters(in: .whitespacesAndNewlines))
                youtubeURL = ""
                video = nil
                subtitles = nil
                showQueueInput = false
            } else {
                submitting = true
                Task {
                    if await store.startFileDubbing(request) {
                        youtubeURL = ""
                        video = nil
                        subtitles = nil
                    }
                    submitting = false
                }
            }
        } catch { store.errorMessage = error.localizedDescription }
    }
}
