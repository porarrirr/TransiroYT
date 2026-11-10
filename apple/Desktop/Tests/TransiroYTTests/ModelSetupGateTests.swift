import Foundation
import Testing
@testable import TransiroYT

private final class SetupGateProtocol: URLProtocol, @unchecked Sendable {
    nonisolated(unsafe) static var requests = 0
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        Self.requests += 1
        client?.urlProtocol(self, didFailWithError: URLError(.cannotConnectToHost))
    }
    override func stopLoading() {}
}

@MainActor @Test func fileDubbingStartsWithoutLegacyRealtimeModels() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root.appendingPathComponent(".local"), withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    try "test-token".write(to: root.appendingPathComponent(".local/token"), atomically: true, encoding: .utf8)
    let info = DistributionInfo(version: "0.5.0", channel: "preview", minimumMacos: "26.0")
    let installation = ConsumerInstallation(runtime: root.appendingPathComponent("Runtime"), support: root, distribution: info)
    let configuration = URLSessionConfiguration.ephemeral
    configuration.protocolClasses = [SetupGateProtocol.self]
    let session = URLSession(configuration: configuration)
    defer { session.invalidateAndCancel() }
    let store = BackendStore(session: session, installation: installation)
    store.connected = true
    SetupGateProtocol.requests = 0
    await store.startFileDubbing(FileDubbingRequest(video: "in.mp4", subtitles: "in.srt", output: "out.mp4",
                                                  originalVolume: 0, dubVolume: 1, speakers: nil,
                                                  allowDownload: true, overwrite: false))
    #expect(store.canStartFileDubbing)
    #expect(!store.showGuide)
    #expect(SetupGateProtocol.requests == 1)
    #expect(store.errorMessage != nil)
    #expect(store.fileDubbing == nil)
}
