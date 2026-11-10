import Foundation
import CryptoKit
import Testing
@testable import TransiroYT

@Test func supportDirectoryKeepsPrivateTokenWithoutModifyingApp() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("TransiroYT test's \(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: root) }
    let runtime = root.appendingPathComponent("Read Only App")
    let support = root.appendingPathComponent("Support")
    let info = DistributionInfo(version: "0.5.0", channel: "release", minimumMacos: "14.0")
    let installation = ConsumerInstallation(runtime: runtime, support: support, distribution: info)
    try installation.prepareSupportDirectory()
    let tokenURL = installation.data.appendingPathComponent("token")
    let token = try String(contentsOf: tokenURL, encoding: .utf8)
    #expect(token.count == 64)
    let attrs = try FileManager.default.attributesOfItem(atPath: tokenURL.path)
    #expect((attrs[.posixPermissions] as? NSNumber)?.intValue == 0o600)
    try installation.prepareSupportDirectory()
    #expect(try String(contentsOf: tokenURL, encoding: .utf8) == token)
    #expect(!FileManager.default.fileExists(atPath: runtime.path))
    // Browser integration is not part of the file-dubbing app.
    #expect(try FileManager.default.contentsOfDirectory(atPath: support.path).sorted() == [".local"])
    let environment = installation.environment
    #expect(environment["PATH"] == runtime.appendingPathComponent("bin").path + ":/usr/bin:/bin")
    #expect(environment["TRANSIROYT_DATA_DIR"] == installation.data.path)
}

@Test func healthRejectsAnotherInstallationAndAcceptsSymlinksAndOlderServers() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    let directory = root.appendingPathComponent("Data")
    let alias = root.appendingPathComponent("Alias")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: directory)
    let health = ServerHealth(service: "transiroyt", version: "0.5.0", instanceId: "server-a",
                              dataId: ServerHealth.dataID(for: directory))
    try health.validate(directory: alias, consumer: false)
    #expect(throws: DesktopError.self) {
        try health.validate(directory: root.appendingPathComponent("Other"), consumer: true)
    }
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    let legacy = try decoder.decode(ServerHealth.self, from: Data(#"{"service":"transiroyt","version":"0.4.0"}"#.utf8))
    try legacy.validate(directory: directory, consumer: false)
    #expect(legacy.instanceId == nil)
    let unrelated = ServerHealth(service: "other", version: "1", instanceId: nil, dataId: nil)
    #expect(throws: DesktopError.self) { try unrelated.validate(directory: directory, consumer: false) }
}

@Test func modelLogsIncludeAllWorkersAndRedactTokensWithBoundedTails() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let files = ["desktop-server.log", "coreml-tts.log", "gemma.log", "whisper.log", "indextts.log"]
    for file in files {
        let text = "discarded-prefix\n" + String(repeating: "x", count: 40000) + "\n\(file) secret-token"
        try text.write(to: root.appendingPathComponent(file), atomically: true, encoding: .utf8)
    }
    try "unrelated".write(to: root.appendingPathComponent("irodori.log"), atomically: true, encoding: .utf8)
    let text = readDesktopLogs(in: root, token: "secret-token")
    for file in files {
        #expect(text.contains(file + " [接続トークン]"))
    }
    #expect(!text.contains("secret-token"))
    #expect(!text.contains("discarded-prefix"))
    #expect(!text.contains("unrelated"))
    #expect(text.utf8.count < 5 * 33000)
}

@Test func youtubeFileDubbingRequestUsesExclusiveInputAndSnakeCaseURL() throws {
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    let request = FileDubbingRequest(video: nil, subtitles: nil, output: "/tmp/dub.mp4",
        originalVolume: 0.25, dubVolume: 1, speakers: nil, allowDownload: false, overwrite: false,
        youtubeUrl: "https://youtu.be/abcdefghijk", youtubeQuality: "720")
    let body = try JSONSerialization.jsonObject(with: encoder.encode(request)) as! [String: Any]
    #expect(body["youtube_url"] as? String == "https://youtu.be/abcdefghijk")
    #expect(body["youtube_quality"] as? String == "720")
    #expect(body["video"] == nil && body["subtitles"] == nil)
    #expect(body["original_volume"] as? Double == 0.25)
}

@Test func stepSettingDefaultsToTenAndIsSentWithTheJob() throws {
    // product.json saved before the setting existed has no "steps".
    let old = try JSONDecoder().decode(ProductPreferences.self, from: Data(#"""
        {"outputDirectory":"~/Downloads/TransiroYT","originalVolume":0,"dubVolume":1,"youtubeQuality":"best","downloadModels":false}
        """#.utf8))
    #expect(old.steps == nil && old.synthesisSteps == 10)
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    let request = FileDubbingRequest(video: "/tmp/in.mp4", subtitles: nil, output: "/tmp/dub.mp4", originalVolume: 0,
        dubVolume: 1, speakers: nil, steps: 8, allowDownload: false, overwrite: false)
    let body = try JSONSerialization.jsonObject(with: encoder.encode(request)) as! [String: Any]
    #expect(body["steps"] as? Int == 8)
}

@Test func fileDownloadProgressDecodesBytesAndUnknownTotals() throws {
    let decoder = JSONDecoder()
    let known = try decoder.decode(FileDubbingJob.Download.self,
        from: Data(#"{"state":"downloading","downloaded":2500000000,"total":5000000000}"#.utf8))
    #expect(known.fraction == 0.5)
    #expect(known.detail.contains("50%"))
    let unknown = try decoder.decode(FileDubbingJob.Download.self,
        from: Data(#"{"state":"downloading","downloaded":12345,"total":null}"#.utf8))
    #expect(unknown.fraction == nil)
    #expect(unknown.detail.contains("取得済み"))
}

@Test func fileDubbingStatusDecodesTheSeparationModel() throws {
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    let status = try decoder.decode(FileDubbingStatus.self, from: Data(
        #"{"job":null,"models":{"llama_server":true,"gemma":true,"speaker_model":true,"separation_model":false,"indextts_model":false}}"#.utf8))
    #expect(status.models.separationModel == false && status.models.speakerModel)
}

@Test func downloadMetricsDecodeAndBytesTakePriorityOverFileCount() throws {
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    let metrics = try decoder.decode(FileDubbingJob.Download.self, from: Data(
        #"{"state":"downloading","downloaded":750,"total":1000,"completed_files":1,"total_files":10,"speed":125,"eta":2,"updated_at":12345}"#.utf8))
    #expect(metrics.fraction == 0.75)
    #expect(metrics.speed == 125 && metrics.eta == 2 && metrics.updatedAt == 12345)
}

@Test func libraryDecodesYouTubeMetadataAndOlderExports() throws {
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    let base = #"""
        {"id":"a","title":"Movie","output":"/tmp/a.mp4","created_at":1,"duration":10,"available":true,"size_bytes":100}
        """#
    let old = try decoder.decode(DubbingLibraryEntry.self, from: Data(base.utf8))
    #expect(old.uploader == nil && old.thumbnailUrl == nil)
    var fields = try JSONSerialization.jsonObject(with: Data(base.utf8)) as! [String: Any]
    fields["uploader"] = "投稿者"
    fields["thumbnail_url"] = "https://i.ytimg.com/vi/abcdefghijk/hqdefault.jpg"
    let entry = try decoder.decode(DubbingLibraryEntry.self, from: JSONSerialization.data(withJSONObject: fields))
    #expect(entry.uploader == "投稿者")
    #expect(entry.thumbnailUrl == fields["thumbnail_url"] as? String)
}
