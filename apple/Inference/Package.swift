// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "TransiroInference",
    platforms: [.macOS(.v15), .iOS(.v18)],
    products: [
        .library(name: "TransiroInference", targets: ["TransiroInference"]),
        .executable(name: "transiro-coreml", targets: ["TransiroCoreMLWorker"])
    ],
    targets: [
        .target(name: "IrodoriNative", path: "Vendor/IrodoriTTS/IrodoriNative", publicHeadersPath: "include",
                cxxSettings: [.define("IRODORI_COREML_ONLY", to: "1")],
                linkerSettings: [.linkedFramework("Foundation"), .linkedFramework("CoreML"), .linkedFramework("Accelerate")]),
        .target(name: "IrodoriTTS", dependencies: ["IrodoriNative"], path: "Vendor/IrodoriTTS/IrodoriTTS",
                linkerSettings: [.linkedFramework("AVFoundation")]),
        .target(name: "TransiroInference", dependencies: ["IrodoriTTS"]),
        .executableTarget(name: "TransiroCoreMLWorker", dependencies: ["TransiroInference"])
    ],
    cxxLanguageStandard: .cxx17
)
