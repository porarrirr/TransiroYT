import Foundation
import Testing
@testable import TransiroYT

@Test func developmentServerLaunchAppendsBothStreamsInProjectDirectory() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("TransiroYT launch's \(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: root) }
    let scripts = root.appendingPathComponent("scripts")
    let local = root.appendingPathComponent(".local")
    try FileManager.default.createDirectory(at: scripts, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: local, withIntermediateDirectories: true)
    let script = scripts.appendingPathComponent("start.sh")
    try "#!/bin/sh\nprintf 'stdout\\n'\nprintf 'started\\n' > started\nprintf 'stderr\\n' >&2\n".write(to: script, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: script.path)
    let log = local.appendingPathComponent("desktop-server.log")
    try "previous log\n".write(to: log, atomically: true, encoding: .utf8)

    let launch = try BackendServerLaunch(projectURL: root, installation: nil)
    defer { try? launch.output.close() }
    try launch.process.run()
    launch.process.waitUntilExit()

    #expect(launch.process.terminationStatus == 0)
    let text = try String(contentsOf: log, encoding: .utf8)
    #expect(text.hasPrefix("previous log\n"))
    #expect(text.contains("サーバー起動"))
    #expect(text.contains("stdout\n"))
    #expect(try String(contentsOf: root.appendingPathComponent("started"), encoding: .utf8) == "started\n")
    #expect(text.contains("stderr\n"))
}

@Test func consumerServerLaunchUsesBundledRuntimeAndPrivateLog() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let installation = ConsumerInstallation(runtime: root.appendingPathComponent("Runtime"), support: root,
        distribution: DistributionInfo(version: "0.5.0", channel: "preview", minimumMacos: "26.0"))
    try FileManager.default.createDirectory(at: installation.apiPython.deletingLastPathComponent(), withIntermediateDirectories: true)
    try """
        #!/bin/sh
        printf 'data:%s\\n' "$TRANSIROYT_DATA_DIR"
        printf 'backend:%s\\n' "$PYTHONPATH"
        printf 'arguments:%s\\n' "$*"
        """.write(to: installation.apiPython, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: installation.apiPython.path)

    let launch = try BackendServerLaunch(projectURL: root, installation: installation)
    defer { try? launch.output.close() }
    try launch.process.run()
    launch.process.waitUntilExit()

    #expect(launch.process.terminationStatus == 0)
    let log = installation.data.appendingPathComponent("desktop-server.log")
    let text = try String(contentsOf: log, encoding: .utf8)
    #expect(text.contains("data:\(installation.data.path)"))
    #expect(text.contains("backend:\(installation.runtime.appendingPathComponent("backend").path)"))
    #expect(text.contains("arguments:-s -m uvicorn transiroyt.app:create_app --factory --host 127.0.0.1 --port 8765"))
    let attributes = try FileManager.default.attributesOfItem(atPath: log.path)
    #expect((attributes[.posixPermissions] as? NSNumber)?.intValue == 0o600)
}
