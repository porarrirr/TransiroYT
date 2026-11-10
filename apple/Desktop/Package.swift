// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "TransiroYTDesktop",
    platforms: [.macOS(.v14)],
    products: [.executable(name: "TransiroYT", targets: ["TransiroYT"])],
    dependencies: [.package(url: "https://github.com/sparkle-project/Sparkle", exact: "2.10.1")],
    targets: [
        .executableTarget(name: "TransiroYT", dependencies: [.product(name: "Sparkle", package: "Sparkle")],
                          linkerSettings: [.unsafeFlags(["-Xlinker", "-rpath", "-Xlinker", "@executable_path/../Frameworks"])]),
        .testTarget(name: "TransiroYTTests", dependencies: ["TransiroYT"])
    ]
)
