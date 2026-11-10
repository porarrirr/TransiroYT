import Foundation
import IrodoriTTS

/// Shared by the Mac file worker and the iPhone/iPad worker.
public actor SpeechEngine {
    public static let identity = "irodori-coreml-948f17da-light-int8-v0.2.0-unwatermarked-v2-timed"
    public static let manifest = ModelVariant.lightINT8.manifestURL
    private let engine = IrodoriEngine()
    private var prepared = false

    public init() {}

    public func prepare(directory: URL, download: Bool,
                        byteProgress: @escaping @Sendable (Int64, Int64, Int64, String) -> Void = { _, _, _, _ in },
                        progress: @escaping @Sendable (Int, Int, String) -> Void = { _, _, _ in }) async throws {
        if prepared { return }
        if !FileManager.default.fileExists(atPath: directory.path) {
            guard download else { throw IrodoriError.invalid("Core ML音声モデルが未取得です。モデルのダウンロードを許可してください。") }
            try await ModelDownloader().download(manifestURL: Self.manifest, to: directory, byteProgress: byteProgress, progress: progress)
        }
        do {
            try await engine.prepare(modelDirectory: directory, prepareWatermark: false)
        } catch {
            // Observed on the first multifunction-decoder load on macOS 27.
            // Retry the same verified files; preserve all downloaded models/caches.
            guard error.localizedDescription.contains("Incomplete Core ML-only model bundle") else { throw error }
            try await engine.prepare(modelDirectory: directory, prepareWatermark: false)
        }
        prepared = true
    }

    /// seconds is the line's speaking length; the model's own estimate ran long on short lines.
    public func synthesize(text: String, voice: String, reference: URL, output: URL, steps: Int, seconds: Double) async throws -> Double {
        try await engine.registerReference(reference)
        let audio = try await engine.synthesize(text, caption: voice, steps: steps, seconds: seconds,
                                                splitSentences: false, watermark: nil)
        try audio.writeWAV(to: output)
        return audio.audioSeconds
    }

    public func release() async {
        await engine.release()
        prepared = false
    }
}
