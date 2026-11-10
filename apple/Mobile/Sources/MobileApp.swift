import CryptoKit
import SwiftUI
import TransiroInference
import os

@main struct TransiroMobileApp: App {
    @State private var scanning = false
    @StateObject private var worker = MobileWorker()
    @Environment(\.scenePhase) private var phase
    var body: some Scene {
        WindowGroup {
            MobileDashboard(worker: worker, scanning: $scanning)
            .sheet(isPresented: $scanning) {
                PairingScanner { value in
                    scanning = false
                    worker.link = value
                    worker.connect()
                }
            }
            .onChange(of: phase) { _, value in worker.setActive(value == .active) }
        }
    }
}

@MainActor final class MobileWorker: ObservableObject {
    @Published var link = ""
    @Published var connected = false
    /// True only after the Mac answered; a saved pairing is not proof that the Mac is reachable.
    @Published var reachable = false
    @Published var linkProblem: String?
    @Published var connecting = false
    @Published var preparing = false
    @Published var status = "接続を待っています"
    @Published var error: String?
    @Published var completed = 0
    @Published var capabilities: [String: String] = [:]
    private let speech = SpeechEngine()
    private var connection: WorkerConnection?
    private var token: String?
    private var connectionTask: Task<Void, Never>?
    private var preparationTask: Task<Void, Never>?
    private var job: Task<Void, Never>?
    private var pollDelay: Task<Void, Never>?
    var busy: Bool { taskID != nil }
    @Published private var taskID: String?
    /// Tasks received and waiting, in the Mac's order; their references are already saved.
    private var queue: [(id: String, payload: [String: Any])] = []
    /// Lines the Mac lets this app hold at once (generating and waiting); its heartbeat says how many.
    private var slots = 1
    private var received: [String] = []
    private var active = true
    /// Reference WAVs by SHA-256, kept for this connection so the Mac sends each once.
    private var references: [String: URL] = [:]
    private var referenceOrder: [String] = []
    private static let referenceLimit = 32
    private let referenceDirectory = FileManager.default.temporaryDirectory.appendingPathComponent("references")

    init() {
        if let saved = SavedPairing.load() {
            link = saved.link
            connect(saved: saved)
        }
    }

    /// The Mac shows this state.
    private let idleStatus = "音声生成の割り当てを待っています"

    /// What the app holds, the line being generated first.
    private var held: [String] { (taskID.map { [$0] } ?? []) + queue.map(\.id) }

    private var accepting: Bool {
        // Heat does not pause the work by design; iOS still throttles a hot device itself.
        active && !ProcessInfo.processInfo.isLowPowerModeEnabled
    }

    func setActive(_ active: Bool) {
        self.active = active
        UIApplication.shared.isIdleTimerDisabled = active && connected
        if !active {
            queue = []
            job?.cancel()
            if let connection, let token {
                Task { _ = try? await connection.request("heartbeat", method: "POST", token: token,
                    body: ["capabilities": capabilities, "state": "画面を閉じたため一時停止", "accepting": false]) }
            }
        }
    }

    func connect(saved: SavedPairing? = nil) {
        guard connectionTask == nil, !connecting else { return }
        connecting = true; error = nil
        try? FileManager.default.removeItem(at: referenceDirectory)
        connectionTask = Task {
            do {
                let connection = WorkerConnection(link: try PairingLink(link))
                self.connection = connection
                try await connection.open()
                let id = UserDefaults.standard.string(forKey: "deviceID") ?? UUID().uuidString
                UserDefaults.standard.set(id, forKey: "deviceID")
                let token: String
                if let saved {
                    token = saved.token
                } else {
                    let reply = try await connection.request("pair", method: "POST", body: [
                        "code": connection.link.code, "name": UIDevice.current.name, "device_id": id
                    ])
                    guard let paired = reply["token"] as? String else { throw WorkerError.message("Macの応答が不正です。") }
                    token = paired
                    try SavedPairing(link: link, token: token).save()
                }
                self.token = token
                reachable = saved == nil
                connected = true; connecting = false; link = ""
                UIApplication.shared.isIdleTimerDisabled = active
                preparationTask = Task { await prepareModels() }
                var failures = 0
                while !Task.isCancelled {
                    do {
                    let state = accepting ? status : "一時停止中（画面・低電力モードを確認）"
                    let heartbeat = try await connection.request("heartbeat", method: "POST", token: token,
                        body: ["capabilities": capabilities, "state": state, "accepting": accepting,
                               "tasks": held, "references": referenceOrder])
                    // A saved pairing skips /pair, so every heartbeat carries the Mac's model versions.
                    guard heartbeat["tts_identity"] as? String == SpeechEngine.identity else {
                        throw WorkerError.message("Macとモバイルアプリのバージョンが一致しません。")
                    }
                    if let count = heartbeat["completed"] as? Int { completed = count }
                    if let count = heartbeat["slots"] as? Int { slots = count }
                    reachable = true; linkProblem = nil
                    // Lines the Mac took back (cancelled or given elsewhere) are dropped.
                    let kept = Set(heartbeat["task_ids"] as? [String] ?? [])
                    queue.removeAll { !kept.contains($0.id) }
                    if let taskID, !kept.contains(taskID) { job?.cancel() }
                    if accepting && held.count < slots && !capabilities.isEmpty {
                        let response = try await connection.request("task", token: token, waitForTask: true)
                        for task in response["tasks"] as? [[String: Any]] ?? [] {
                            await receive(task, connection: connection, token: token)
                        }
                        startNext(connection: connection, token: token)
                    }
                    failures = 0
                    } catch {
                        guard !Task.isCancelled, WorkerError.isTransient(error) else { throw error }
                        failures += 1
                        reachable = false
                        linkProblem = unreachableMessage(connection.link, failures: failures, error: error)
                        try await Task.sleep(for: .seconds(min(10, failures * 2)))
                        continue
                    }
                    if held.count >= slots || !accepting || capabilities.isEmpty {
                        let delay = Task { _ = try? await Task.sleep(for: .seconds(2)) }
                        pollDelay = delay
                        await delay.value
                        pollDelay = nil
                    }
                }
            } catch {
                if !Task.isCancelled { self.error = error.localizedDescription }
            }
            disconnect()
        }
    }

    private func unreachableMessage(_ link: PairingLink, failures: Int, error: Error) -> String {
        let check = link.usb
            ? "USBケーブルの接続と「このコンピュータを信頼」、Macでモバイル連携が有効かを確認してください。"
            : "この端末はWi-Fiで連携しています。Macと同じWi-Fiか確認してください。USBケーブルで使う場合は、接続を解除し、Macで「USBケーブル」を選んで発行したQRコードを読み取ってください。"
        return "Macから応答がありません（再試行\(failures)回目：\(error.localizedDescription)）。\(check)"
    }

    func retryPreparation() {
        guard connected, !preparing, job == nil else { return }
        error = nil
        preparationTask = Task { await prepareModels() }
    }

    private func prepareModels() async {
        preparing = true
        defer { preparing = false }
        do {
            let root = try FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask,
                                                   appropriateFor: nil, create: true)
            status = "音声モデルを準備しています"
            try await speech.prepare(directory: root.appendingPathComponent("irodori-coreml-light-int8-v0.2.0"), download: true) { [weak self] done, total, file in
                Task { @MainActor in self?.status = "音声モデルを取得：\(done)/\(total)・\(file)" }
            }
            try Task.checkCancellation()
            capabilities["tts"] = SpeechEngine.identity
            status = idleStatus
            // Text work moved to the Mac; earlier versions kept LFM3 here (about 1.9 GB).
            try? FileManager.default.removeItem(at: root.appendingPathComponent("lfm3-1.7b"))
        } catch {
            if !Task.isCancelled { self.error = error.localizedDescription; status = capabilities.isEmpty ? "音声モデルの準備に失敗しました" : "利用可能なモデルで待機中" }
        }
    }

    /// Queue a received line; its reference is saved now, so the Mac sends it with one line only.
    private func receive(_ task: [String: Any], connection: WorkerConnection, token: String) async {
        guard let id = task["id"] as? String, !received.contains(id) else { return }
        received = Array((received + [id]).suffix(16))
        guard task["kind"] as? String == "tts", let payload = task["payload"] as? [String: Any] else {
            await report(id, WorkerError.message("Macからの処理要求を読み取れません。"), connection: connection, token: token)
            return
        }
        do {
            _ = try reference(for: payload)
            queue.append((id, payload))
        } catch {
            await report(id, error, connection: connection, token: token)
        }
    }

    private func report(_ id: String, _ error: Error, connection: WorkerConnection, token: String) async {
        self.error = error.localizedDescription
        _ = try? await connection.request("result", method: "POST", token: token,
            body: ["task_id": id, "error": String(error.localizedDescription.prefix(500))])
    }

    /// Generate the next queued line as soon as the previous one is done.
    private func startNext(connection: WorkerConnection, token: String) {
        guard job == nil, accepting, !queue.isEmpty else { return }
        let (id, payload) = queue.removeFirst()
        taskID = id
        job = Task {
            defer {
                job = nil; taskID = nil; pollDelay?.cancel()
                if !preparing { status = idleStatus }
                startNext(connection: connection, token: token)
            }
            do {
                guard let text = payload["text"] as? String, let voice = payload["voice"] as? String,
                      let seconds = payload["seconds"] as? Double else {
                    throw WorkerError.message("参照音声・台詞の形式が不正です。")
                }
                let source = try reference(for: payload)
                status = "音声を生成しています"
                let work = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
                try FileManager.default.createDirectory(at: work, withIntermediateDirectories: true)
                defer { try? FileManager.default.removeItem(at: work) }
                let output = work.appendingPathComponent("output.wav")
                let clock = ContinuousClock()
                let started = clock.now
                _ = try await speech.synthesize(text: text, voice: voice, reference: source, output: output,
                                                steps: payload["steps"] as? Int ?? 4, seconds: seconds)
                let elapsed = clock.now - started
                print("TRANSIRO-MEM tts-done available=\(os_proc_available_memory() >> 20)MiB")
                let result: [String: Any] = [
                    "task_id": id, "wav": try Data(contentsOf: output).base64EncodedString(),
                    "references": referenceOrder,
                    "elapsed_ms": Int(elapsed.components.seconds * 1000 + elapsed.components.attoseconds / 1_000_000_000_000_000)]
                try Task.checkCancellation()
                for attempt in 0..<3 {
                    do {
                        _ = try await connection.request("result", method: "POST", token: token, body: result)
                        break
                    } catch {
                        if case .http(409, _) = error as? WorkerError { return }
                        guard attempt < 2, WorkerError.isTransient(error) else { throw error }
                        try await Task.sleep(for: .seconds(attempt + 1))
                    }
                }
            } catch {
                if !Task.isCancelled {
                    if case .http(409, _) = error as? WorkerError { return }
                    await report(id, error, connection: connection, token: token)
                }
            }
        }
    }

    /// The Mac sends a reference's bytes only while this app does not list its SHA-256.
    private func reference(for payload: [String: Any]) throws -> URL {
        guard let digest = payload["reference_sha256"] as? String else {
            throw WorkerError.message("Macのアプリを更新してください（参照音声の形式が古い）。")
        }
        if let url = references[digest] { return url }
        guard let encoded = payload["reference"] as? String, let data = Data(base64Encoded: encoded),
              SHA256.hash(data: data).map({ String(format: "%02x", $0) }).joined() == digest else {
            throw WorkerError.message("参照音声を受信できませんでした。")
        }
        try FileManager.default.createDirectory(at: referenceDirectory, withIntermediateDirectories: true)
        let url = referenceDirectory.appendingPathComponent("\(digest).wav")
        try data.write(to: url, options: .atomic)
        if referenceOrder.count >= Self.referenceLimit {
            let oldest = referenceOrder.removeFirst()
            if let file = references.removeValue(forKey: oldest) { try? FileManager.default.removeItem(at: file) }
        }
        references[digest] = url
        referenceOrder.append(digest)
        return url
    }

    func disconnect() {
        guard connection != nil || connectionTask != nil else { return }
        let old = connection, oldToken = token
        let oldPreparation = preparationTask, oldJob = job
        connectionTask?.cancel(); connectionTask = nil
        preparationTask?.cancel(); preparationTask = nil
        queue = []
        job?.cancel()
        connection = nil; token = nil; connected = false; reachable = false; linkProblem = nil; connecting = true
        SavedPairing.delete()
        capabilities = [:]; status = "接続を終了しています"
        UIApplication.shared.isIdleTimerDisabled = false
        Task {
            if let old, let oldToken { _ = try? await old.request("connection", method: "DELETE", token: oldToken) }
            old?.close()
            await oldPreparation?.value
            await oldJob?.value
            await speech.release()
            job = nil; taskID = nil; queue = []; received = []
            references = [:]; referenceOrder = []
            try? FileManager.default.removeItem(at: referenceDirectory)
            connecting = false; status = "接続を待っています"
        }
    }
}
