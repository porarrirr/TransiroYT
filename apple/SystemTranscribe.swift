import AVFoundation
import Foundation
import Speech

// On-device speech transcription of a whole file for videos without subtitles.
// Usage: system-transcribe <16 kHz mono WAV> <language>
// stdout is one JSON object: {"locale", "segments": [{"start", "end", "text",
// "words": [{"start", "end", "word", "confidence"}]}]}. Errors go to stderr; a
// language the transcriber does not offer exits with status 3.
@main struct SystemTranscribe {
    struct Word: Encodable { let start: Double; let end: Double; let word: String; let confidence: Double? }
    struct Segment: Encodable { let start: Double; let end: Double; let text: String; let words: [Word] }
    struct Response: Encodable { let locale: String; let segments: [Segment] }

    static func main() async {
        guard #available(macOS 26.0, *) else {
            FileHandle.standardError.write(Data("macOS 26以降が必要です。\n".utf8))
            exit(1)
        }
        if CommandLine.arguments.contains("--check") { print("ready"); return }
        guard CommandLine.arguments.count == 3 else {
            FileHandle.standardError.write(Data("usage: system-transcribe <wav> <language>\n".utf8))
            exit(2)
        }
        do {
            guard let response = try await transcribe(
                URL(fileURLWithPath: CommandLine.arguments[1]), language: CommandLine.arguments[2]) else {
                FileHandle.standardError.write(Data("\(CommandLine.arguments[2])はmacOS標準の文字起こしに対応していません。\n".utf8))
                exit(3)
            }
            FileHandle.standardOutput.write(try JSONEncoder().encode(response) + Data([10]))
        } catch {
            FileHandle.standardError.write(Data("\(error)\n".utf8))
            exit(1)
        }
    }

    @available(macOS 26.0, *)
    static func transcribe(_ url: URL, language: String) async throws -> Response? {
        guard let locale = await SpeechTranscriber.supportedLocale(equivalentTo: Locale(identifier: language)) else {
            return nil
        }
        let transcriber = SpeechTranscriber(locale: locale, transcriptionOptions: [], reportingOptions: [],
                                            attributeOptions: [.audioTimeRange, .transcriptionConfidence])
        // The language's model is a system asset, downloaded once by macOS.
        if let request = try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
            FileHandle.standardError.write(Data("installing speech assets for \(locale.identifier)\n".utf8))
            try await request.downloadAndInstall()
        }
        let analyzer = SpeechAnalyzer(modules: [transcriber])
        let results = Task {
            var segments: [Segment] = []
            for try await result in transcriber.results {
                let words = result.text.runs.compactMap { run -> Word? in
                    guard let range = run.audioTimeRange else { return nil }
                    return Word(start: range.start.seconds, end: range.end.seconds,
                                word: String(result.text[run.range].characters),
                                confidence: run.transcriptionConfidence)
                }
                segments.append(Segment(start: result.range.start.seconds, end: result.range.end.seconds,
                                        text: String(result.text.characters), words: words))
            }
            return segments
        }
        let file = try AVAudioFile(forReading: url)
        if let end = try await analyzer.analyzeSequence(from: file) {
            try await analyzer.finalizeAndFinish(through: end)
        } else {
            await analyzer.cancelAndFinishNow()
        }
        return Response(locale: locale.identifier(.bcp47), segments: try await results.value)
    }
}
