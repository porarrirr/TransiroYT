import Foundation

struct BackendServerLaunch {
    let process: Process
    let output: FileHandle

    init(projectURL: URL, installation: ConsumerInstallation?) throws {
        let local = projectURL.appendingPathComponent(".local")
        try FileManager.default.createDirectory(at: local, withIntermediateDirectories: true)
        let logURL = local.appendingPathComponent("desktop-server.log")
        if !FileManager.default.fileExists(atPath: logURL.path) {
            FileManager.default.createFile(atPath: logURL.path, contents: nil, attributes: [.posixPermissions: 0o600])
        }
        output = try FileHandle(forWritingTo: logURL)
        try output.seekToEnd()
        try output.write(contentsOf: Data("\n--- \(Date().formatted()) サーバー起動 ---\n".utf8))
        process = Process()
        if let installation {
            process.executableURL = installation.apiPython
            process.arguments = ["-s", "-m", "uvicorn", "transiroyt.app:create_app", "--factory", "--host", "127.0.0.1", "--port", "8765"]
        } else {
            process.executableURL = URL(fileURLWithPath: "/bin/zsh")
            process.arguments = ["-l", "-c", "exec ./scripts/start.sh"]
        }
        process.currentDirectoryURL = projectURL
        var environment = ProcessInfo.processInfo.environment
        let paths = [NSHomeDirectory() + "/.local/bin", "/opt/homebrew/bin", "/usr/local/bin", environment["PATH"] ?? "/usr/bin:/bin"]
        environment["PATH"] = paths.joined(separator: ":")
        environment["PYTHONUNBUFFERED"] = "1"
        process.environment = installation?.environment ?? environment
        process.standardOutput = output
        process.standardError = output
        process.standardInput = FileHandle.nullDevice
    }
}
