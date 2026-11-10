import Foundation

func readDesktopLogs(in directory: URL, token: String) -> String {
    let files = [("desktop-server.log", "アプリの動作"), ("coreml-tts.log", "Core ML音声生成"),
                 ("gemma.log", "Gemmaローカル代替"), ("whisper.log", "音声確認"),
                 ("indextts.log", "参照音声モデル")]
    var parts: [String] = []
    for (filename, label) in files {
        guard let handle = try? FileHandle(forReadingFrom: directory.appendingPathComponent(filename)) else { continue }
        defer { try? handle.close() }
        let size = (try? handle.seekToEnd()) ?? 0
        let limit: UInt64 = 32768
        try? handle.seek(toOffset: size > limit ? size - limit : 0)
        let text = String(decoding: (try? handle.readToEnd()) ?? Data(), as: UTF8.self)
        if !text.isEmpty { parts.append("--- \(label) ---\n" + text) }
    }
    let text = parts.joined(separator: "\n\n")
    return token.isEmpty ? text : text.replacingOccurrences(of: token, with: "[接続トークン]")
}
