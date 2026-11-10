import Foundation
import Testing
@testable import TransiroYT

private final class QueueProtocol: URLProtocol, @unchecked Sendable {
    nonisolated(unsafe) static var handler: ((URLRequest) -> (Int, [String: Any]))?
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        let (code, json) = Self.handler!(request)
        client?.urlProtocol(self, didReceive: HTTPURLResponse(url: request.url!, statusCode: code, httpVersion: nil, headerFields: nil)!, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: try! JSONSerialization.data(withJSONObject: json))
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}

@MainActor @Test func dubbingQueueAdvancesInOrderAndKeepsFailedRequests() async throws {
    let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(".local/work/dubbing-queue/test")
    let data = root.appendingPathComponent(".local")
    try FileManager.default.createDirectory(at: data, withIntermediateDirectories: true)
    try "queue-test".write(to: data.appendingPathComponent("token"), atomically: true, encoding: .utf8)
    let configuration = URLSessionConfiguration.ephemeral
    configuration.protocolClasses = [QueueProtocol.self]
    let session = URLSession(configuration: configuration)
    defer { session.invalidateAndCancel(); QueueProtocol.handler = nil }
    nonisolated(unsafe) var state = "running"
    nonisolated(unsafe) var posts = 0
    nonisolated(unsafe) var reject = false
    nonisolated(unsafe) var progressUnavailable = false
    nonisolated(unsafe) var finishImmediately = false
    QueueProtocol.handler = { request in
        switch request.url!.path {
        case "/health":
            return (200, ["service": "transiroyt", "version": "0.5.0", "instance_id": "queue", "data_id": ServerHealth.dataID(for: data)])
        case "/api/status": return (200, ["ready": true, "problems": []])
        default:
            if request.httpMethod == "POST" {
                if reject { return (409, ["detail": "test rejection"]) }
                posts += 1
                state = finishImmediately ? "done" : "running"
            } else if progressUnavailable {
                return (503, ["detail": "progress temporarily unavailable"])
            }
            return (200, ["job": ["id": "job-\(posts)", "state": state, "stage": "prepare", "fraction": 0, "message": "test", "video": "input", "subtitles": "", "output": "output"], "models": ["llama_server": true, "gemma": true, "speaker_model": true, "separation_model": true, "indextts_model": true]])
        }
    }
    let store = BackendStore(session: session)
    store.projectURL = root
    await store.refresh()
    let first = FileDubbingRequest(output: "first.mp4", originalVolume: 0, dubVolume: 1, allowDownload: true, overwrite: false)
    var second = first
    second.output = "second.mp4"
    store.enqueueDubbing(first, title: "first")
    store.enqueueDubbing(second, title: "second")
    await store.refresh()
    #expect(posts == 0)
    progressUnavailable = true
    await store.refresh()
    #expect(store.fileDubbing?.job?.id == "job-0")
    #expect(store.fileDubbing?.job?.running == true)
    #expect(!store.connected && !store.canStartFileDubbing)
    #expect(store.dubbingQueue.map(\.title) == ["first", "second"])
    await store.startNextDubbing()
    #expect(posts == 0)
    progressUnavailable = false
    state = "done"
    await store.refresh()
    #expect(posts == 1)
    #expect(store.dubbingQueue.map(\.title) == ["second"])
    state = "error"
    await store.refresh()
    #expect(posts == 1 && store.dubbingQueue.count == 1)
    reject = true
    await store.startNextDubbing()
    #expect(store.dubbingQueue.count == 1)
    reject = false
    await store.startNextDubbing()
    #expect(posts == 2 && store.dubbingQueue.isEmpty)
    state = "done"
    await store.refresh()
    await store.refresh()
    #expect(posts == 2)

    // Completion can arrive in the refresh performed before submission returns.
    finishImmediately = true
    store.enqueueDubbing(first, title: "quick first")
    store.enqueueDubbing(second, title: "quick second")
    await store.startNextDubbing()
    #expect(posts == 3)
    #expect(store.dubbingQueue.map(\.title) == ["quick second"])
    await store.refresh()
    await store.refresh()
    #expect(posts == 4 && store.dubbingQueue.isEmpty)
}
