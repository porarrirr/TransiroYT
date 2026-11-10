import AppKit
import Foundation

extension BackendStore {
    func initializeConsumer() async {
        guard !consumerInitialized, let installation else { return }
        consumerInitialized = true
        do {
            if installation.isOnDiskImage { return }
            try installation.prepareSupportDirectory()
            do { try installation.installCLI() }
            catch { NSLog("CLIのインストール: %@", error.localizedDescription) }
            // Connection and diagnostics must work before model downloads finish.
            await startServer()
        } catch { errorMessage = error.localizedDescription }
    }

    func openGuide() {
        guard let installation else { return }
        NSWorkspace.shared.open(installation.runtime.appendingPathComponent("help/index.html"))
    }
}
