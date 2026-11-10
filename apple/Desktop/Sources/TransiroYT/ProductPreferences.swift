import AppKit
import Foundation

struct ProductPreferences: Codable {
    var outputDirectory = "~/Downloads/TransiroYT"
    var originalVolume = 0.0
    var dubVolume = 1.0
    var speakers: Int? = nil
    var youtubeQuality = "best"
    // Optional so product.json written before this setting still loads.
    var steps: Int? = nil

    static let defaultSteps = 10
    static let stepRange = 4...16
    var synthesisSteps: Int { steps ?? Self.defaultSteps }

    static var file: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("TransiroYT/product.json")
    }

    static func load() throws -> Self {
        if !FileManager.default.fileExists(atPath: file.path) {
            try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Self().save()
        }
        let value = try JSONDecoder().decode(Self.self, from: Data(contentsOf: file))
        guard (0...2).contains(value.originalVolume), value.dubVolume > 0, value.dubVolume <= 2,
              value.speakers.map({ (1...20).contains($0) }) ?? true,
              ["best", "2160", "1440", "1080", "720", "480", "360"].contains(value.youtubeQuality),
              value.steps.map({ stepRange.contains($0) }) ?? true,
              !value.outputDirectory.isEmpty else {
            throw DesktopError.message("product.jsonの音量・話者数・画質・ステップ数・保存先を確認してください。")
        }
        return value
    }

    func save() throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(self).write(to: Self.file, options: .atomic)
    }
}
