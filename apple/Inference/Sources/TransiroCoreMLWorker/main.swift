import Foundation
import TransiroInference

struct Request: Decodable {
    let op: String
    let models: String
    let download: Bool?
    let text: String?
    let voice: String?
    let reference: String?
    let output: String?
    let steps: Int?
    let seconds: Double?
}

@main struct Worker {
    static func main() async {
        let engine = SpeechEngine()
        while let line = readLine() {
            do {
                let request = try JSONDecoder().decode(Request.self, from: Data(line.utf8))
                try await engine.prepare(directory: URL(fileURLWithPath: request.models), download: request.download ?? false,
                                         byteProgress: { received, total, transferred, file in
                    write(["event": "download", "downloaded": received, "total_bytes": total,
                           "transferred": transferred, "filename": file])
                }) { done, total, file in
                    write(["event": "download", "completed": done, "total": total, "filename": file])
                }
                var result: [String: Any] = ["ok": true, "model": SpeechEngine.identity]
                if request.op == "synthesize" {
                    guard let text = request.text, let reference = request.reference, let output = request.output,
                          let seconds = request.seconds else {
                        throw NSError(domain: "TransiroYT", code: 1, userInfo: [NSLocalizedDescriptionKey: "音声生成の入力が不足しています"])
                    }
                    result["duration"] = try await engine.synthesize(text: text, voice: request.voice ?? "",
                        reference: URL(fileURLWithPath: reference), output: URL(fileURLWithPath: output),
                        steps: request.steps ?? 4, seconds: seconds)
                } else if request.op != "prepare" {
                    throw NSError(domain: "TransiroYT", code: 2, userInfo: [NSLocalizedDescriptionKey: "未知の音声処理です"])
                }
                write(result)
            } catch { write(["ok": false, "error": error.localizedDescription]) }
        }
        await engine.release()
    }

    static func write(_ value: [String: Any]) {
        if let bytes = try? JSONSerialization.data(withJSONObject: value) {
            FileHandle.standardOutput.write(bytes + Data([10]))
        }
    }
}
