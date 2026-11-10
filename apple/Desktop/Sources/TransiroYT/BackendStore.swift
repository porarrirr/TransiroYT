import AppKit
import Combine
import Foundation
import UniformTypeIdentifiers

@MainActor
final class BackendStore: ObservableObject {
    let installation: ConsumerInstallation?
    @Published var showGuide = false
    /// Bumped when the settings window saves product preferences, so open views reload them.
    @Published var preferencesRevision = 0
    var consumerInitialized = false
    @Published var projectURL: URL?
    @Published var status: ServerStatus?
    @Published var fileDubbing: FileDubbingStatus?
    @Published var dubbingQueue: [QueuedDubbing] = []
    @Published var submittingDubbing = false
    private var automaticNextJobID: String?
    @Published var connected = false
    @Published var serverDetected = false
    @Published var busy = false
    @Published var starting = false
    @Published var ownsServer = false
    @Published var connectionMessage = "接続を確認しています…"
    @Published var errorMessage: String?
    @Published var notice: String?
    @Published var logs = ""
    private var server: Process?
    private var logHandle: FileHandle?
    private var refreshing = false
    private var rejectedConnection: BackendConnectionIdentity?
    private let client: BackendClient

    var canStartFileDubbing: Bool { connected && installation?.isOnDiskImage != true }
    var canStart: Bool { projectURL != nil && installation?.isOnDiskImage != true && !serverDetected && !connected && !ownsServer && !starting }
    var token: String {
        guard let projectURL else { return "" }
        return (try? String(contentsOf: projectURL.appendingPathComponent(".local/token"), encoding: .utf8))?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    }

    init(session: URLSession? = nil, installation: ConsumerInstallation? = ConsumerInstallation.bundled()) {
        let sessionConfiguration = URLSessionConfiguration.ephemeral
        sessionConfiguration.timeoutIntervalForRequest = 8
        sessionConfiguration.timeoutIntervalForResource = 40
        self.client = BackendClient(session: session ?? URLSession(configuration: sessionConfiguration))
        self.installation = installation
        if let installation {
            projectURL = installation.support
            return
        }
        let paths: [String?] = [
            UserDefaults.standard.string(forKey: "projectPath"),
            Bundle.main.object(forInfoDictionaryKey: "TransiroYTProjectPath") as? String,
            FileManager.default.currentDirectoryPath,
            URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
                .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().path
        ]
        let candidates = paths.compactMap { $0 }
        projectURL = candidates.map { URL(fileURLWithPath: $0, isDirectory: true) }.first(where: Self.isProject)
    }

    static func isProject(_ url: URL) -> Bool {
        FileManager.default.fileExists(atPath: url.appendingPathComponent("backend/transiroyt/app.py").path)
            && FileManager.default.fileExists(atPath: url.appendingPathComponent("scripts/start.sh").path)
    }

    func chooseProject() {
        let panel = NSOpenPanel()
        panel.title = "TransiroYTのプロジェクトフォルダを選択"
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        guard Self.isProject(url) else {
            errorMessage = "backendとscriptsを含むTransiroYTのフォルダを選んでください。"
            return
        }
        projectURL = url
        UserDefaults.standard.set(url.path, forKey: "projectPath")
        status = nil
        connected = false
        Task { await refresh() }
    }

    func request<T: Decodable>(_ path: String, method: String = "GET", body: Data? = nil) async throws -> T {
        try await client.request(path, token: token, method: method, body: body,
                                 contentType: body == nil ? "application/octet-stream" : "application/json")
    }

    func refreshDownloadProgress() async {
        guard connected, let job = fileDubbing?.job, job.running,
              ["download", "model_download"].contains(job.stage) else { return }
        if let latest: FileDubbingStatus = try? await request("/api/file-dubbing") {
            fileDubbing = latest
        }
    }

    func refresh() async {
        guard !refreshing else { return }
        refreshing = true
        defer { refreshing = false; readLogs() }
        do {
            let (healthData, response) = try await client.healthResponse()
            serverDetected = true
            guard (response as? HTTPURLResponse)?.statusCode == 200 else {
                throw DesktopError.message("ポート8765のサービスが正常に応答していません。")
            }
            let decoder = JSONDecoder()
            decoder.keyDecodingStrategy = .convertFromSnakeCase
            let health = try decoder.decode(ServerHealth.self, from: healthData)
            guard let directory = projectURL?.appendingPathComponent(".local") else {
                throw DesktopError.message("設定で接続先のプロジェクトフォルダを選んでください。")
            }
            try health.validate(directory: directory, consumer: installation != nil)
            let identity = BackendConnectionIdentity(instanceID: health.instanceId, dataID: health.dataId, token: token)
            guard rejectedConnection != identity else { throw DesktopError.unauthorized }
            let serverStatus: ServerStatus
            do { serverStatus = try await request("/api/status") }
            catch DesktopError.unauthorized {
                rejectedConnection = identity
                throw DesktopError.unauthorized
            }
            rejectedConnection = nil
            fileDubbing = try await request("/api/file-dubbing")
            status = serverStatus
            connected = true
            starting = false
            connectionMessage = "127.0.0.1:8765 · v\(health.version)"
            // A completion observed during submission must remain pending until the queue item is removed.
            if !submittingDubbing, let job = fileDubbing?.job, job.id == automaticNextJobID, !job.running {
                automaticNextJobID = nil
                if job.state == "done" { await startNextDubbing() }
            }
        } catch {
            if let error = error as? URLError, [.cannotConnectToHost, .networkConnectionLost].contains(error.code) {
                serverDetected = false
                rejectedConnection = nil
            }
            connected = false
            status = nil
            if let urlError = error as? URLError {
                switch urlError.code {
                case .cannotConnectToHost, .networkConnectionLost:
                    connectionMessage = "サーバーを起動すると接続できます。"
                case .timedOut:
                    connectionMessage = "サーバーの応答を待っています。ログを確認してください。"
                default: connectionMessage = urlError.localizedDescription
                }
            } else { connectionMessage = error.localizedDescription }
        }
    }

    func startServer() async {
        guard canStart, let projectURL else { return }
        starting = true
        // Check again immediately before launching; an existing terminal server is shared.
        while refreshing { try? await Task.sleep(for: .milliseconds(50)) }
        await refresh()
        guard !connected && !serverDetected else { starting = false; return }
        do {
            let launch = try BackendServerLaunch(projectURL: projectURL, installation: installation)
            let process = launch.process
            process.terminationHandler = { [weak self] finished in
                let code = finished.terminationStatus
                Task { @MainActor in
                    guard let self else { return }
                    self.server = nil
                    self.ownsServer = false
                    self.starting = false
                    try? self.logHandle?.close()
                    self.logHandle = nil
                    if code != 0 { self.errorMessage = "サーバーが終了しました（コード \(code)）。ログを確認してください。" }
                    await self.refresh()
                }
            }
            try process.run()
            server = process
            logHandle = launch.output
            ownsServer = true
            starting = true
        } catch { starting = false; errorMessage = error.localizedDescription }
    }

    func shutdown() {
        if let server, server.isRunning { server.terminate() }
    }

    struct QueuedDubbing: Identifiable {
        let id = UUID()
        let title: String
        let request: FileDubbingRequest
    }

    func enqueueDubbing(_ body: FileDubbingRequest, title: String) {
        dubbingQueue.append(QueuedDubbing(title: title, request: body))
        if fileDubbing?.job?.running == true { automaticNextJobID = fileDubbing?.job?.id }
    }

    func startNextDubbing() async {
        guard !submittingDubbing, fileDubbing?.job?.running != true,
              canStartFileDubbing, let next = dubbingQueue.first else { return }
        if await startFileDubbing(next.request) {
            dubbingQueue.removeAll { $0.id == next.id }
        }
    }

    @discardableResult
    func startFileDubbing(_ body: FileDubbingRequest) async -> Bool {
        guard !submittingDubbing else { return false }
        submittingDubbing = true
        defer { submittingDubbing = false }
        guard installation?.isOnDiskImage != true else {
            errorMessage = "アプリを「アプリケーション」フォルダへ移してから開いてください。"
            return false
        }
        do {
            let encoder = JSONEncoder()
            encoder.keyEncodingStrategy = .convertToSnakeCase
            let reply: FileDubbingReply = try await client.request(
                "/api/file-dubbing", token: token, method: "POST", body: try encoder.encode(body),
                contentType: "application/json")
            automaticNextJobID = reply.job?.id
            if let job = reply.job, let models = fileDubbing?.models {
                fileDubbing = FileDubbingStatus(job: job, models: models)
            }
            await refresh()
            return true
        } catch { errorMessage = error.localizedDescription; return false }
    }

    func cancelFileDubbing(_ id: String) async {
        automaticNextJobID = nil
        do {
            let _: FileDubbingReply = try await request("/api/file-dubbing/\(id)", method: "DELETE")
            await refresh()
        } catch { errorMessage = error.localizedDescription }
    }

    func readLogs() {
        guard let directory = projectURL?.appendingPathComponent(".local") else { logs = ""; return }
        logs = readDesktopLogs(in: directory, token: token)
    }

    func reveal(_ path: String) {
        guard let projectURL else { return }
        NSWorkspace.shared.selectFile(nil, inFileViewerRootedAtPath: projectURL.appendingPathComponent(path).path)
    }
}
