import AppKit
import QuickLookThumbnailing
import SwiftUI

struct LibraryThumbnail: View {
    let entry: DubbingLibraryEntry
    @State private var image: NSImage?
    @Environment(\.displayScale) private var displayScale
    @MainActor private static let cache: NSCache<NSString, NSImage> = {
        let cache = NSCache<NSString, NSImage>()
        cache.countLimit = 100
        return cache
    }()
    private var cacheKey: String {
        "\(entry.output)|\(entry.createdAt)|\(entry.sizeBytes)|\(entry.thumbnailUrl ?? "")|\(entry.available)"
    }

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 8).fill(.quaternary)
            if let image {
                Image(nsImage: image).resizable().scaledToFill()
            } else {
                Image(systemName: entry.available ? "film" : "exclamationmark.triangle")
                    .font(.system(size: 26, weight: .ultraLight)).foregroundStyle(.secondary)
            }
        }
        .frame(width: 128, height: 72).clipShape(RoundedRectangle(cornerRadius: 8))
        .accessibilityHidden(true)
        .task(id: cacheKey) { await load() }
    }

    @MainActor
    private func load() async {
        image = nil
        let key = cacheKey as NSString
        if let cached = Self.cache.object(forKey: key) {
            image = cached
            return
        }
        if let source = entry.thumbnailUrl, let url = URL(string: source), url.scheme == "https" {
            let request = URLRequest(url: url, cachePolicy: .returnCacheDataElseLoad, timeoutInterval: 10)
            if let (data, response) = try? await URLSession.shared.data(for: request),
               let response = response as? HTTPURLResponse, (200..<300).contains(response.statusCode),
               let downloaded = NSImage(data: data), !Task.isCancelled {
                image = downloaded
                Self.cache.setObject(downloaded, forKey: key)
                return
            }
        }
        guard entry.available, !Task.isCancelled else { return }
        // Quick Look reuses macOS thumbnails when available and generates missing ones.
        let request = QLThumbnailGenerator.Request(fileAt: URL(fileURLWithPath: entry.output),
                                                   size: CGSize(width: 256, height: 144),
                                                   scale: displayScale, representationTypes: .thumbnail)
        defer { QLThumbnailGenerator.shared.cancel(request) }
        if let thumbnail = try? await QLThumbnailGenerator.shared.generateBestRepresentation(for: request),
           !Task.isCancelled {
            image = thumbnail.nsImage
            Self.cache.setObject(thumbnail.nsImage, forKey: key)
        }
    }
}
