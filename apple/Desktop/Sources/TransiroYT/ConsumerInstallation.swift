import AppKit
import CryptoKit
import Foundation
import Security

struct DistributionInfo: Decodable, Sendable {
    let version: String
    let channel: String
    let minimumMacos: String
}

struct ConsumerInstallation: Sendable {
    let runtime: URL
    let support: URL
    let distribution: DistributionInfo
    var data: URL { support.appendingPathComponent(".local", isDirectory: true) }
    var apiPython: URL { runtime.appendingPathComponent("python-api/bin/python3.12") }
    var ttsPython: URL { runtime.appendingPathComponent("python-tts/bin/python3.11") }
    var indexttsPython: URL { runtime.appendingPathComponent("python-indextts/bin/python3.11") }
    var isOnDiskImage: Bool { Bundle.main.bundleURL.path.hasPrefix("/Volumes/") }

    static func bundled() -> ConsumerInstallation? {
        guard let runtime = Bundle.main.resourceURL?.appendingPathComponent("Runtime"),
              let data = try? Data(contentsOf: runtime.appendingPathComponent("distribution.json")),
              let distribution = try? decoder().decode(DistributionInfo.self, from: data) else { return nil }
        let override = ProcessInfo.processInfo.environment["TRANSIROYT_TEST_SUPPORT_DIR"]
        let support = override.map { URL(fileURLWithPath: $0, isDirectory: true) }
            ?? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("TransiroYT", isDirectory: true)
        return ConsumerInstallation(runtime: runtime, support: support, distribution: distribution)
    }

    static func decoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return decoder
    }

    var environment: [String: String] {
        var result = ProcessInfo.processInfo.environment
        for key in result.keys where key.hasPrefix("TRANSIROYT_") || ["PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH"].contains(key) {
            result.removeValue(forKey: key)
        }
        result["PATH"] = runtime.appendingPathComponent("bin").path + ":/usr/bin:/bin"
        result["PYTHONPATH"] = runtime.appendingPathComponent("backend").path
        result["PYTHONNOUSERSITE"] = "1"
        result["PYTHONDONTWRITEBYTECODE"] = "1"
        result["PYTHONUNBUFFERED"] = "1"
        result["HF_HOME"] = data.appendingPathComponent("huggingface").path
        result["HF_HUB_CACHE"] = data.appendingPathComponent("huggingface/hub").path
        result["NUMBA_CACHE_DIR"] = data.appendingPathComponent("cache/numba").path
        result["TORCH_HOME"] = data.appendingPathComponent("cache/torch").path
        result["HF_HUB_DISABLE_TELEMETRY"] = "1"
        result["HF_HUB_DISABLE_XET"] = "1"
        result["XDG_CACHE_HOME"] = data.appendingPathComponent("cache").path
        result["MPLCONFIGDIR"] = data.appendingPathComponent("cache/matplotlib").path
        result["TRANSIROYT_RESOURCE_ROOT"] = runtime.path
        result["TRANSIROYT_CONFIG_FILE"] = support.appendingPathComponent(".env").path
        result["TRANSIROYT_DATA_DIR"] = data.path
        result["TRANSIROYT_TTS_PYTHON"] = ttsPython.path
        result["TRANSIROYT_TTS_SITE_PACKAGES"] = runtime.appendingPathComponent("python-tts/lib/python3.11/site-packages").path
        result["TRANSIROYT_INDEXTTS_PYTHON"] = indexttsPython.path
        result["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        return result
    }

    func prepareSupportDirectory() throws {
        let fm = FileManager.default
        for directory in [support, data] {
            try fm.createDirectory(at: directory, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
            try fm.setAttributes([.posixPermissions: 0o700], ofItemAtPath: directory.path)
        }
        let tokenURL = data.appendingPathComponent("token")
        if !fm.fileExists(atPath: tokenURL.path) {
            var bytes = [UInt8](repeating: 0, count: 32)
            guard SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes) == errSecSuccess else {
                throw DesktopError.message("接続の準備ができませんでした。アプリを開き直してください。")
            }
            let token = bytes.map { String(format: "%02x", $0) }.joined()
            // O_EXCL prevents a second app instance from changing an existing token.
            let fd = open(tokenURL.path, O_WRONLY | O_CREAT | O_EXCL, 0o600)
            if fd >= 0 {
                let handle = FileHandle(fileDescriptor: fd, closeOnDealloc: true)
                try handle.write(contentsOf: Data(token.utf8))
                try handle.close()
            } else if errno != EEXIST { throw DesktopError.message("接続設定を保存できませんでした。") }
        }
        try fm.setAttributes([.posixPermissions: 0o600], ofItemAtPath: tokenURL.path)
    }

    func installCLI(in directory: URL = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".local/bin")) throws {
        let fm = FileManager.default
        let launcher = runtime.appendingPathComponent("backend/transiroyt/cli-launcher")
        guard fm.isExecutableFile(atPath: launcher.path) else {
            throw DesktopError.message("CLIコマンドが見つかりません。")
        }
        try fm.createDirectory(at: directory, withIntermediateDirectories: true)
        let command = directory.appendingPathComponent("transiroyt")
        if let target = try? fm.destinationOfSymbolicLink(atPath: command.path) {
            if target == launcher.path { return }
            guard target.hasSuffix("/TransiroYT.app/Contents/Resources/Runtime/backend/transiroyt/cli-launcher") else {
                throw DesktopError.message("既存の\(command.path)を保持しました。CLIのインストール先を確認してください。")
            }
            try fm.removeItem(at: command)
        }
        // createSymbolicLink fails if an unrelated command already exists.
        try fm.createSymbolicLink(at: command, withDestinationURL: launcher)
    }
}
