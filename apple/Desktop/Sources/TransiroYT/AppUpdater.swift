import Combine
import Foundation
import Sparkle

@MainActor
final class AppUpdater: NSObject, ObservableObject, SPUUpdaterDelegate {
    @Published private(set) var canCheck = false
    weak var store: BackendStore?
    private var controller: SPUStandardUpdaterController!
    private var started = false
    private var pendingInstallation: (() -> Void)?

    override init() {
        super.init()
        controller = SPUStandardUpdaterController(startingUpdater: false, updaterDelegate: self, userDriverDelegate: nil)
        controller.updater.publisher(for: \.canCheckForUpdates).assign(to: &$canCheck)
    }

    var working: Bool { store?.fileDubbing?.job?.running == true }

    func start() {
        guard !started, Bundle.main.object(forInfoDictionaryKey: "SUFeedURL") != nil else { return }
        controller.startUpdater()
        started = true
    }

    func check() { controller.checkForUpdates(nil) }

    func updater(_ updater: SPUUpdater, mayPerform updateCheck: SPUUpdateCheck) throws {
        if working {
            throw NSError(domain: "TransiroYT.Updates", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "作成・ダウンロードの完了後にアップデートを確認します。"])
        }
    }

    func updater(_ updater: SPUUpdater, shouldPostponeRelaunchForUpdate item: SUAppcastItem,
                 untilInvokingBlock installHandler: @escaping () -> Void) -> Bool {
        deferInstallationIfWorking(installHandler)
    }

    func deferInstallationIfWorking(_ install: @escaping () -> Void) -> Bool {
        guard working else { return false }
        pendingInstallation = install
        return true
    }

    func resumeWhenIdle() {
        guard !working, let install = pendingInstallation else { return }
        pendingInstallation = nil
        install()
    }
}
