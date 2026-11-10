// Modified by TransiroYT on 2026-10-27: report byte progress during model transfers.
import Foundation
import CryptoKit

/// Downloads to disk, retaining completed verified files for a subsequent retry.
/// Prefer an immutable commit SHA or a versioned release tag over a moving main branch.
public actor ModelDownloader {
    public init() {}

    public func download(manifestURL: URL, to destination: URL,
                         byteProgress: @escaping @Sendable (Int64, Int64, Int64, String) -> Void = { _, _, _, _ in },
                         progress: @Sendable (Int, Int, String) -> Void = { _, _, _ in }) async throws {
        guard manifestURL.scheme == "https" else { throw IrodoriError.invalid("Use an HTTPS manifest URL") }
        let files = FileManager.default
        guard !files.fileExists(atPath: destination.path) else {
            throw IrodoriError.invalid("Destination already exists. Select a new folder or verify the existing model.")
        }
        // Distinct manifests must not share a staging directory during retries.
        let key = SHA256.hash(data: Data(manifestURL.absoluteString.utf8)).prefix(8).map { String(format: "%02x", $0) }.joined()
        let staging = destination.deletingLastPathComponent().appendingPathComponent(".\(destination.lastPathComponent)-\(key).partial")
        try files.createDirectory(at: staging, withIntermediateDirectories: true)
        let (data, response) = try await URLSession.shared.data(from: manifestURL)
        guard (response as? HTTPURLResponse)?.statusCode == 200 else { throw IrodoriError.invalid("Manifest download failed") }
        let manifest = try ModelBundle.manifest(from: data)
        if manifest.format == "irodori-coreml-distribution-v2" {
            if #available(iOS 18, macOS 15, *) {} else {
                throw IrodoriError.invalid("This model requires iOS 18 or macOS 15")
            }
        }
        let base = manifestURL.deletingLastPathComponent()
        let totalBytes = manifest.files.reduce(Int64(0)) { $0 + $1.bytes }
        var finishedBytes: Int64 = 0
        var transferredBytes: Int64 = 0
        for (index, entry) in manifest.files.enumerated() {
            try Task.checkCancellation()
            let target = try ModelBundle.safeURL(entry.path, under: staging)
            progress(index, manifest.files.count, entry.path)
            if let size = try? files.attributesOfItem(atPath: target.path)[.size] as? NSNumber,
               size.int64Value == entry.bytes, (try? ModelBundle.checksum(target)) == entry.sha256 {
                finishedBytes += entry.bytes
                continue
            }
            try files.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            let baseBytes = finishedBytes
            let baseTransferred = transferredBytes
            byteProgress(baseBytes, totalBytes, baseTransferred, entry.path)
            let delegate = ModelDownloadDelegate { received in
                byteProgress(baseBytes + received, totalBytes, baseTransferred + received, entry.path)
            }
            let (temporary, downloadResponse) = try await URLSession.shared.download(
                from: base.appendingPathComponent(entry.path), delegate: delegate)
            defer { try? files.removeItem(at: temporary) }
            guard (downloadResponse as? HTTPURLResponse)?.statusCode == 200,
                  (try files.attributesOfItem(atPath: temporary.path)[.size] as? NSNumber)?.int64Value == entry.bytes,
                  try ModelBundle.checksum(temporary) == entry.sha256 else {
                throw IrodoriError.invalid("Downloaded file failed verification: \(entry.path)")
            }
            if files.fileExists(atPath: target.path) { try files.removeItem(at: target) }
            try files.moveItem(at: temporary, to: target)
            finishedBytes += entry.bytes
            transferredBytes += entry.bytes
            byteProgress(finishedBytes, totalBytes, transferredBytes, entry.path)
        }
        try Task.checkCancellation()
        try data.write(to: staging.appendingPathComponent("manifest.json"), options: .atomic)
        try ModelBundle.validate(at: staging)
        try files.moveItem(at: staging, to: destination)
        progress(manifest.files.count, manifest.files.count, "Complete")
    }
}


private final class ModelDownloadDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let report: @Sendable (Int64) -> Void
    private var lastReport = Date.distantPast

    init(report: @escaping @Sendable (Int64) -> Void) { self.report = report }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didWriteData bytesWritten: Int64, totalBytesWritten: Int64,
                    totalBytesExpectedToWrite: Int64) {
        guard Date().timeIntervalSince(lastReport) >= 0.2 else { return }
        lastReport = Date()
        report(totalBytesWritten)
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didFinishDownloadingTo location: URL) {}
}
