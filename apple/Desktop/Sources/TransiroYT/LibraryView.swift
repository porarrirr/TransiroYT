import AppKit
import SwiftUI

struct DubbingLibraryStatus: Decodable {
    let entries: [DubbingLibraryEntry]
    let totalBytes: Int64
    let warnings: [String]
}

struct DubbingLibraryEntry: Decodable, Identifiable {
    let id: String
    let title: String
    let output: String
    let createdAt: Double
    let duration: Double
    let available: Bool
    let sizeBytes: Int64
    let uploader: String?
    let thumbnailUrl: String?
}

struct DubbingCacheStatus: Decodable {
    let categories: [DubbingCacheCategory]
    let totalBytes: Int64
}

struct DubbingCacheCategory: Decodable, Identifiable {
    let id: String
    let title: String
    let sizeBytes: Int64
}

func storageGB(_ bytes: Int64) -> String {
    if bytes > 0 && bytes < 1_000_000 { return "0.001 GB未満" }
    return String(format: "%.3f GB", Double(bytes) / 1_000_000_000)
}

@MainActor
func libraryRequestPath() throws -> String {
    let preferences = try ProductPreferences.load()
    var components = URLComponents()
    components.path = "/api/library"
    components.queryItems = [URLQueryItem(name: "directory", value: (preferences.outputDirectory as NSString).expandingTildeInPath)]
    return components.string!
}

struct LibraryView: View {
    @ObservedObject var store: BackendStore
    @State private var library: DubbingLibraryStatus?
    @State private var loading = false
    @State private var error: String?
    @State private var search = ""
    @State private var showStorage = false
    private var entries: [DubbingLibraryEntry] {
        library?.entries.filter { search.isEmpty || $0.title.localizedStandardContains(search) || $0.output.localizedStandardContains(search) || $0.uploader?.localizedStandardContains(search) == true } ?? []
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 28) {
                HStack(alignment: .top) {
                    VStack(alignment: .leading, spacing: 10) {
                        Text("ライブラリ").font(.system(size: 40, weight: .semibold)).tracking(-1.5)
                        if let library {
                            Text("\(library.entries.count)本 · \(storageGB(library.totalBytes))")
                                .font(.callout).foregroundStyle(.secondary).monospacedDigit()
                        }
                    }
                    Spacer()
                    Button("キャッシュを削除…", systemImage: "trash") { showStorage = true }
                        .padding(.top, 12)
                }
                HStack(spacing: 12) {
                    Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
                    TextField("タイトル・投稿者・ファイル名で検索", text: $search).textFieldStyle(.plain)
                    if loading { ProgressView().controlSize(.small) }
                    Button { Task { await refresh() } } label: { Image(systemName: "arrow.clockwise") }
                        .buttonStyle(.plain).accessibilityLabel("ライブラリを更新").disabled(loading || !store.connected)
                }.padding(14).background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
                if let error {
                    Label(error, systemImage: "exclamationmark.circle").foregroundStyle(.secondary).textSelection(.enabled)
                }
                if let library {
                    ForEach(library.warnings, id: \.self) { warning in
                        Text(warning).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                    }
                    if entries.isEmpty {
                        ContentUnavailableView(search.isEmpty ? "完成した動画がここに並びます" : "見つかりませんでした",
                                               systemImage: search.isEmpty ? "film.stack" : "magnifyingglass",
                                               description: Text(search.isEmpty ? "現在の保存先にある作成済みの動画も自動で表示します。" : "別のタイトルやファイル名で検索してください。"))
                            .frame(minHeight: 220)
                    } else {
                        LazyVStack(spacing: 0) {
                            ForEach(entries) { entry in
                                row(entry)
                                Divider()
                            }
                        }
                    }
                } else if !store.connected {
                    ContentUnavailableView("接続を待っています", systemImage: "circle.dotted",
                                           description: Text("アプリの接続後にライブラリを読み込みます。"))
                }
                Text("完成動画の容量です。キャッシュの容量と削除は「キャッシュを削除…」から確認できます。")
                    .font(.caption).foregroundStyle(.secondary)
            }.frame(maxWidth: 860).padding(.horizontal, 44).padding(.top, 52).padding(.bottom, 40)
                .frame(maxWidth: .infinity)
        }
        .task(id: store.connected) { if store.connected { await refresh() } }
        .onChange(of: store.fileDubbing?.job?.state) { _, _ in Task { await refresh() } }
        .sheet(isPresented: $showStorage) { CacheStorageView(store: store) }
    }

    private func row(_ entry: DubbingLibraryEntry) -> some View {
        HStack(spacing: 20) {
            LibraryThumbnail(entry: entry)
            VStack(alignment: .leading, spacing: 7) {
                Text(entry.title).font(.system(size: 16, weight: .medium)).lineLimit(2).textSelection(.enabled)
                if let uploader = entry.uploader, !uploader.isEmpty {
                    Text(uploader).font(.callout).foregroundStyle(.secondary).lineLimit(1).textSelection(.enabled)
                }
                HStack(spacing: 12) {
                    Text(Date(timeIntervalSince1970: entry.createdAt), format: .dateTime.year().month().day())
                    Text(Duration.seconds(entry.duration), format: .time(pattern: .hourMinuteSecond))
                    Text(storageGB(entry.sizeBytes)).monospacedDigit()
                }.font(.caption).foregroundStyle(.secondary)
                Text(entry.available ? entry.output : "ファイルが見つかりません · \(entry.output)")
                    .font(.caption).foregroundStyle(.tertiary).lineLimit(1).truncationMode(.middle).help(entry.output)
            }.frame(maxWidth: .infinity, alignment: .leading)
            Button { NSWorkspace.shared.open(URL(fileURLWithPath: entry.output)) } label: { Image(systemName: "play.fill") }
                .accessibilityLabel("\(entry.title)を再生").disabled(!entry.available)
            Button { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: entry.output)]) } label: { Image(systemName: "folder") }
                .accessibilityLabel("\(entry.title)をFinderで表示").disabled(!entry.available)
        }.padding(.vertical, 22)
    }

    private func refresh() async {
        guard store.connected, !loading else { return }
        loading = true
        defer { loading = false }
        do {
            library = try await store.request(try libraryRequestPath())
            error = nil
        } catch { self.error = error.localizedDescription }
    }
}

struct CacheStorageView: View {
    @ObservedObject var store: BackendStore
    @Environment(\.dismiss) private var dismiss
    @State private var cache: DubbingCacheStatus?
    @State private var loading = false
    @State private var clearing = false
    @State private var confirmClear = false
    @State private var error: String?
    @State private var notice: String?
    private var running: Bool { store.fileDubbing?.job?.running == true }

    var body: some View {
        VStack(alignment: .leading, spacing: 24) {
            HStack {
                Text("キャッシュの削除").font(.system(size: 28, weight: .semibold)).tracking(-0.5)
                Spacer()
                Button { Task { await refresh() } } label: { Image(systemName: "arrow.clockwise") }
                    .accessibilityLabel("容量を更新").disabled(loading || clearing || !store.connected)
            }
            VStack(alignment: .leading, spacing: 8) {
                Text("削除できるキャッシュ").font(.callout).foregroundStyle(.secondary)
                if let cache {
                    Text(storageGB(cache.totalBytes)).font(.system(size: 42, weight: .light, design: .rounded)).monospacedDigit()
                } else if store.connected { ProgressView().controlSize(.small) }
                else { Text("接続を待っています").foregroundStyle(.secondary) }
            }
            if let cache {
                VStack(spacing: 16) {
                    ForEach(cache.categories) { category in
                        HStack {
                            Text(category.title)
                            Spacer()
                            Text(storageGB(category.sizeBytes)).monospacedDigit().foregroundStyle(.secondary)
                        }
                    }
                }.padding(20).background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
            }
            Text("完成動画・レポート、利用者が選んだ元動画、ダウンロード済みモデル、設定は保持します。キャッシュを削除すると、次回の吹き替えで動画の取得や音声の分析・生成をやり直します。")
                .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            if running {
                Label("吹き替えの実行中は削除できません。", systemImage: "clock").font(.callout).foregroundStyle(.secondary)
            }
            if let error { Text(error).font(.callout).foregroundStyle(.red).textSelection(.enabled) }
            if let notice { Label(notice, systemImage: "checkmark.circle").font(.callout).foregroundStyle(.secondary) }
            Spacer(minLength: 0)
            Divider()
            HStack {
                Button(role: .destructive) { confirmClear = true } label: {
                    if clearing { ProgressView().controlSize(.small) }
                    Text(clearing ? "削除中…" : "キャッシュを削除…")
                }.disabled(cache == nil || cache?.totalBytes == 0 || running || clearing || loading || !store.connected)
                Spacer()
                Button("閉じる") { dismiss() }.keyboardShortcut(.cancelAction).disabled(clearing)
            }
        }.padding(28).frame(width: 600, height: 580)
            .background(Color(nsColor: .windowBackgroundColor)).tint(.yellow)
            .task(id: store.connected) { if store.connected { await refresh() } }
            .confirmationDialog("キャッシュを削除しますか？", isPresented: $confirmClear, titleVisibility: .visible) {
                Button("キャッシュを削除", role: .destructive) { Task { await clear() } }
                Button("キャンセル", role: .cancel) {}
            } message: {
                Text("\(storageGB(cache?.totalBytes ?? 0))の作業データを削除します。完成動画とモデルは保持します。")
            }
    }

    private func refresh() async {
        loading = true
        defer { loading = false }
        do {
            let _: DubbingLibraryStatus = try await store.request(try libraryRequestPath())
            cache = try await store.request("/api/cache")
            error = nil
        } catch { self.error = error.localizedDescription }
    }

    private func clear() async {
        clearing = true
        notice = nil
        defer { clearing = false }
        do {
            cache = try await store.request("/api/cache", method: "DELETE")
            error = nil
            notice = "キャッシュを削除しました。"
        } catch {
            self.error = error.localizedDescription
            cache = try? await store.request("/api/cache")
        }
    }
}
