import AppKit
import Foundation

// Resolve the approved artwork relative to this script so both build paths
// use the same source, independent of the caller's working directory.
let source = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
    .deletingLastPathComponent().appendingPathComponent("assets/AppIcon.png")
guard let artwork = NSImage(contentsOf: source) else {
    fatalError("Cannot load app icon: \(source.path)")
}
let output = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
let iconset = output.appendingPathComponent("AppIcon.iconset", isDirectory: true)
try FileManager.default.createDirectory(at: iconset, withIntermediateDirectories: true)
func draw(_ size: Int) -> Data {
    // Explicit pixel dimensions keep build results identical on Retina and
    // non-Retina build Macs, including Chrome's required 128px store icon.
    let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: size, pixelsHigh: size,
        bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
        colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: bitmap)
    NSGraphicsContext.current?.imageInterpolation = .high
    artwork.draw(in: NSRect(x: 0, y: 0, width: size, height: size),
                 from: .zero, operation: .copy, fraction: 1)
    NSGraphicsContext.restoreGraphicsState()
    return bitmap.representation(using: .png, properties: [:])!
}
for size in [16, 32, 128, 256, 512] {
    for scale in [1, 2] {
        let name = "icon_\(size)x\(size)" + (scale == 2 ? "@2x" : "") + ".png"
        try draw(size * scale).write(to: iconset.appendingPathComponent(name))
    }
}
if CommandLine.arguments.count > 2 {
    let icons = URL(fileURLWithPath: CommandLine.arguments[2], isDirectory: true)
    try FileManager.default.createDirectory(at: icons, withIntermediateDirectories: true)
    for size in [16, 32, 48, 128] {
        try draw(size).write(to: icons.appendingPathComponent("icon\(size).png"))
    }
}
let process = Process()
process.executableURL = URL(fileURLWithPath: "/usr/bin/iconutil")
process.arguments = ["-c", "icns", iconset.path, "-o", output.appendingPathComponent("AppIcon.icns").path]
try process.run()
process.waitUntilExit()
guard process.terminationStatus == 0 else { fatalError("Icon generation failed") }
try FileManager.default.removeItem(at: iconset)
