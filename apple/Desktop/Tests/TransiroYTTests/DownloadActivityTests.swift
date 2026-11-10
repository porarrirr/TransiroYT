import Foundation
import Testing
@testable import TransiroYT

@Test func downloadSpeedUsesNumericZeroAndFileUnits() {
    let date = Date(timeIntervalSince1970: 100)
    let idle = DownloadActivityReading(downloaded: 660_600_000, total: 1_610_000_000,
                                       speed: 0, eta: nil, updatedAt: 100)
    #expect(idle.speedText(at: date) == "0 KB/s")
    #expect(idle.remainingText(at: date) == "—")
    #expect(idle.transferredText.contains("660.6 MB"))
    let active = DownloadActivityReading(downloaded: 660_600_000, total: 1_610_000_000,
                                         speed: 12_500_000, eta: 75.2, updatedAt: 100)
    #expect(active.speedText(at: date) == "12.5 MB/s")
    #expect(active.remainingText(at: date) == "約1分16秒")
}

@Test func delayedDownloadUpdatesDoNotBecomeMeasuredZero() {
    let reading = DownloadActivityReading(downloaded: 500, total: 1000, speed: 100, eta: 5, updatedAt: 100)
    #expect(reading.speedText(at: Date(timeIntervalSince1970: 103.5)) == "100 B/s")
    #expect(reading.speedText(at: Date(timeIntervalSince1970: 106)) == "更新待ち")
    #expect(reading.remainingText(at: Date(timeIntervalSince1970: 106)) == "—")
    let unmeasured = DownloadActivityReading(downloaded: 500, total: nil, speed: nil, eta: nil, updatedAt: nil)
    #expect(unmeasured.speedText(at: Date()) == "計測中…")
    #expect(unmeasured.fraction == nil)
    #expect(unmeasured.transferredText.contains("取得済み"))
}

@Test func downloadHistoryUsesReportTimesAndResetsForAnotherTransfer() {
    var history = DownloadActivityHistory()
    history.record(DownloadActivityReading(downloaded: 100, total: 1000, speed: 20, eta: 45, updatedAt: 100))
    history.record(DownloadActivityReading(downloaded: 100, total: 1000, speed: 25, eta: 36, updatedAt: 100))
    #expect(history.samples.count == 1)
    #expect(history.samples[0].speed == 25)
    history.record(DownloadActivityReading(downloaded: 200, total: 1000, speed: 100, eta: 8, updatedAt: 101))
    #expect(history.samples.count == 2)
    history.record(DownloadActivityReading(downloaded: 900, total: 1000, speed: 50, eta: 2, updatedAt: 132))
    #expect(history.samples.count == 1)
    #expect(history.samples[0].date == Date(timeIntervalSince1970: 132))
    history.record(DownloadActivityReading(downloaded: 0, total: 2000, speed: nil, eta: nil, updatedAt: 133))
    #expect(history.samples.isEmpty)
}
