import Foundation
import Testing
@testable import TransiroYT

@Test func cliInstallationRefreshesOwnedSymlinkAndPreservesOtherCommands() throws {
    let fm = FileManager.default
    let root = fm.temporaryDirectory.appendingPathComponent("CLI \(UUID().uuidString)")
    defer { try? fm.removeItem(at: root) }
    let runtime = root.appendingPathComponent("TransiroYT.app/Contents/Resources/Runtime")
    let launcher = runtime.appendingPathComponent("backend/transiroyt/cli-launcher")
    try fm.createDirectory(at: launcher.deletingLastPathComponent(), withIntermediateDirectories: true)
    try "#!/bin/sh\n".write(to: launcher, atomically: true, encoding: .utf8)
    try fm.setAttributes([.posixPermissions: 0o755], ofItemAtPath: launcher.path)
    let installation = ConsumerInstallation(runtime: runtime, support: root,
        distribution: DistributionInfo(version: "0.5.0", channel: "preview", minimumMacos: "27.0"))
    let bin = root.appendingPathComponent("bin")
    let command = bin.appendingPathComponent("transiroyt")
    try installation.installCLI(in: bin)
    try installation.installCLI(in: bin)
    #expect(try fm.destinationOfSymbolicLink(atPath: command.path) == launcher.path)
    try fm.removeItem(at: command)
    try fm.createSymbolicLink(atPath: command.path, withDestinationPath: "/old/TransiroYT.app/Contents/Resources/Runtime/backend/transiroyt/cli-launcher")
    try installation.installCLI(in: bin)
    #expect(try fm.destinationOfSymbolicLink(atPath: command.path) == launcher.path)
    try fm.removeItem(at: command)
    try "keep".write(to: command, atomically: true, encoding: .utf8)
    #expect(throws: (any Error).self) { try installation.installCLI(in: bin) }
    #expect(try String(contentsOf: command, encoding: .utf8) == "keep")
    try fm.removeItem(at: command)
    try fm.createSymbolicLink(atPath: command.path, withDestinationPath: "/unrelated/missing")
    #expect(throws: (any Error).self) { try installation.installCLI(in: bin) }
    #expect(try fm.destinationOfSymbolicLink(atPath: command.path) == "/unrelated/missing")
}
