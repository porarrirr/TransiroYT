import CryptoKit
import Foundation

struct APIError: Decodable { let detail: String }

struct ServerHealth: Decodable {
    let service: String
    let version: String
    let instanceId: String?
    let dataId: String?

    static func dataID(for directory: URL) -> String {
        let path = directory.standardizedFileURL.resolvingSymlinksInPath().path
        return SHA256.hash(data: Data(path.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    func validate(directory: URL, consumer: Bool) throws {
        guard service == "transiroyt" else {
            throw DesktopError.message("ポート8765で別のサービスが実行されています。")
        }
        if let dataId, dataId != Self.dataID(for: directory) {
            let action = consumer ? "開発版など、先に起動したTransiroYTを停止してから、このアプリで開始してください。"
                : "先に起動した配布版または別フォルダのTransiroYTを停止するか、設定で接続先のプロジェクトフォルダを選び直してください。"
            throw DesktopError.message("別の保存先を使うTransiroYTがポート8765で実行されています。" + action)
        }
    }
}

struct BackendConnectionIdentity: Equatable {
    let instanceID: String?
    let dataID: String?
    let token: String
}

struct ServerStatus: Decodable, Sendable {
    let ready: Bool
    let problems: [String]
}

struct FileDubbingStatus: Decodable, Sendable {
    struct Models: Decodable, Sendable, Equatable {
        let llamaServer: Bool
        let gemma: Bool
        let speakerModel: Bool
        let separationModel: Bool
        let indexttsModel: Bool
    }
    let job: FileDubbingJob?
    let models: Models
}

struct FileDubbingReply: Decodable, Sendable { let job: FileDubbingJob? }

struct FileDubbingJob: Decodable, Sendable, Equatable {
    struct Download: Decodable, Sendable, Equatable {
        let state: String
        let downloaded: Int64
        let total: Int64?
        let filename: String?
        let completedFiles: Int?
        let totalFiles: Int?
        var speed: Double? = nil
        var eta: Double? = nil
        var updatedAt: Double? = nil
        var fraction: Double? {
            if let total, total > 0 { return min(1, Double(downloaded) / Double(total)) }
            if let completedFiles, let totalFiles, totalFiles > 0 {
                return min(1, Double(completedFiles) / Double(totalFiles))
            }
            return nil
        }
        var detail: String {
            if state == "preparing" { return "モデルを読み込んでいます" }
            if state == "ready" { return "利用可能" }
            if state == "checking" { return "取得情報を確認しています" }
            if state == "processing" { return "動画と音声を結合しています" }
            if state == "complete" { return "取得完了" }
            if let completedFiles, let totalFiles { return "\(completedFiles) / \(totalFiles)ファイル" }
            let received = ByteCountFormatter.string(fromByteCount: downloaded, countStyle: .file)
            guard let total, let fraction else { return "\(received) 取得済み" }
            let size = ByteCountFormatter.string(fromByteCount: total, countStyle: .file)
            return "\(Int(fraction * 100))%（\(received) / \(size)）"
        }
    }
    struct Summary: Decodable, Sendable, Equatable { let segments: Int; let ready: Int; let skipped: Int }
    struct Speaker: Decodable, Sendable, Equatable { let referenceSeconds: Double; let reference: Bool; let error: String? }
    struct Activity: Decodable, Sendable, Equatable {
        let fraction: Double
        let message: String
        let updatedAt: Double
    }
    let id: String
    let state: String
    let stage: String
    let fraction: Double
    let message: String
    let video: String
    let subtitles: String
    let output: String
    let error: String?
    let summary: Summary?
    let speakers: [String: Speaker]?
    let report: String?
    var startedAt: Double? = nil
    var downloads: [String: Download]? = nil
    var activities: [String: Activity]? = nil
    var running: Bool { state == "running" }
}

struct FileDubbingRequest: Encodable, Equatable {
    var video: String?
    var subtitles: String?
    var output: String
    var originalVolume: Double
    var dubVolume: Double
    var speakers: Int?
    var steps: Int? = nil
    var allowDownload: Bool
    var overwrite: Bool
    var youtubeUrl: String? = nil
    var youtubeQuality: String? = nil
}

enum DesktopError: LocalizedError {
    case message(String)
    case unauthorized
    var errorDescription: String? {
        switch self {
        case .message(let message): return message
        case .unauthorized:
            return "接続トークンがサーバーと一致しません。先に起動したTransiroYTを停止して再起動してください。開発版ではプロジェクトフォルダも確認してください。"
        }
    }
}
