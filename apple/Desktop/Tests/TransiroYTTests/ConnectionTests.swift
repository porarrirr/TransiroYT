import Foundation
import Testing
@testable import TransiroYT

private final class ConnectionResponses: @unchecked Sendable {
    let lock = NSLock()
    var instanceID = "server-a"
    var dataID: String
    var denied = true
    var statusRequests = 0

    init(directory: URL) { dataID = ServerHealth.dataID(for: directory) }

    func response(for path: String) -> (Int, Data) {
        lock.withLock {
            let status: Int
            let json: [String: Any]
            switch path {
            case "/health":
                status = 200
                json = ["service": "transiroyt", "version": "0.5.0", "instance_id": instanceID, "data_id": dataID]
            case "/api/status":
                statusRequests += 1
                status = denied ? 401 : 200
                json = denied ? ["detail": "Unauthorized"] : ["ready": true, "problems": []]
            default:
                status = 200
                json = ["job": NSNull(), "models": ["llama_server": true, "gemma": true, "speaker_model": true, "separation_model": true, "indextts_model": true]]
            }
            return (status, try! JSONSerialization.data(withJSONObject: json))
        }
    }
}

private final class ConnectionProtocol: URLProtocol, @unchecked Sendable {
    nonisolated(unsafe) static var responses: ConnectionResponses?
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        let (status, data) = Self.responses!.response(for: request.url!.path)
        let response = HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: "HTTP/1.1",
                                       headerFields: ["Content-Type": "application/json"])!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: data)
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}

@MainActor @Test func authenticationFailureWaitsForRestartOrChangedCredentialsAndRejectsOtherStorage() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    let data = root.appendingPathComponent(".local")
    try FileManager.default.createDirectory(at: data, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let token = data.appendingPathComponent("token")
    try "first-token".write(to: token, atomically: true, encoding: .utf8)
    let responses = ConnectionResponses(directory: data)
    ConnectionProtocol.responses = responses
    defer { ConnectionProtocol.responses = nil }
    let configuration = URLSessionConfiguration.ephemeral
    configuration.protocolClasses = [ConnectionProtocol.self]
    let session = URLSession(configuration: configuration)
    defer { session.invalidateAndCancel() }
    let store = BackendStore(session: session)
    store.projectURL = root
    await store.refresh()
    await store.refresh()
    #expect(responses.statusRequests == 1)
    #expect(store.serverDetected && !store.connected && !store.canStart)
    #expect(store.connectionMessage.contains("接続トークン"))

    try "new-token".write(to: token, atomically: true, encoding: .utf8)
    await store.refresh()
    #expect(responses.statusRequests == 2)
    responses.lock.withLock { responses.instanceID = "server-b"; responses.denied = false }
    await store.refresh()
    #expect(responses.statusRequests == 3)
    #expect(store.connected)

    responses.lock.withLock { responses.dataID = "another-installation" }
    await store.refresh()
    await store.refresh()
    #expect(responses.statusRequests == 3)
    #expect(!store.connected && store.connectionMessage.contains("別の保存先"))

    // A consumer app connects without any legacy setup marker or downloaded model.
    let installation = ConsumerInstallation(runtime: root.appendingPathComponent("Runtime"), support: root,
        distribution: DistributionInfo(version: "0.5.0", channel: "release", minimumMacos: "26.0"))
    let consumer = BackendStore(session: session, installation: installation)
    #expect(consumer.canStart)
    responses.lock.withLock { responses.dataID = ServerHealth.dataID(for: data) }
    await consumer.initializeConsumer()
    #expect(consumer.consumerInitialized && consumer.connected)
    #expect(consumer.status?.ready == true && consumer.canStartFileDubbing)
}
