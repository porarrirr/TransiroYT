import Foundation
import Testing
@testable import TransiroYT

@MainActor @Test func updatesWaitForAnActiveDubbingJob() throws {
    let store = BackendStore(installation: nil)
    let updates = AppUpdater()
    updates.store = store
    let data = Data(#"{"job":{"id":"job","state":"running","stage":"synthesis","fraction":0.5,"message":"working","video":"in.mp4","subtitles":"in.srt","output":"out.mp4"},"models":{"llama_server":true,"gemma":true,"speaker_model":true,"separation_model":true,"indextts_model":true}}"#.utf8)
    store.fileDubbing = try ConsumerInstallation.decoder().decode(FileDubbingStatus.self, from: data)
    var installed = false
    #expect(updates.deferInstallationIfWorking { installed = true })
    updates.resumeWhenIdle()
    #expect(!installed)
    store.fileDubbing = nil
    updates.resumeWhenIdle()
    #expect(installed)
}
