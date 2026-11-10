import SwiftUI

struct DownloadActivityView: View {
    let downloaded: Double
    let total: Double?
    let speed: Double?
    let eta: Double?
    let updatedAt: Double?
    @State private var history = DownloadActivityHistory()

    private var reading: DownloadActivityReading {
        DownloadActivityReading(downloaded: downloaded, total: total, speed: speed, eta: eta, updatedAt: updatedAt)
    }

    var body: some View {
        TimelineView(.periodic(from: .now, by: 1)) { context in
            VStack(alignment: .leading, spacing: 12) {
                ViewThatFits(in: .horizontal) {
                    HStack(alignment: .firstTextBaseline, spacing: 20) {
                        speedLabel(at: context.date)
                        Spacer(minLength: 12)
                        remainingLabel(at: context.date)
                    }
                    VStack(alignment: .leading, spacing: 8) {
                        speedLabel(at: context.date)
                        remainingLabel(at: context.date)
                    }
                }
                graph(at: context.date)
                VStack(alignment: .leading, spacing: 8) {
                    ViewThatFits(in: .horizontal) {
                        HStack(spacing: 12) {
                            Text(reading.transferredText)
                            Spacer(minLength: 0)
                            if let fraction = reading.fraction {
                                Text(fraction, format: .percent.precision(.fractionLength(1)))
                            }
                        }
                        Text(reading.transferredText)
                    }
                    .font(.caption).monospacedDigit().foregroundStyle(.secondary)
                    ProgressView(value: reading.fraction).progressViewStyle(.linear).tint(.yellow)
                }
            }
            .padding(14)
            .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 10))
            .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(.primary.opacity(0.07)))
        }
        .onChange(of: reading, initial: true) { _, value in
            history.record(value)
        }
    }

    private func speedLabel(at date: Date) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("ダウンロード速度").font(.caption).foregroundStyle(.secondary)
            Text(reading.speedText(at: date))
                .font(.system(size: 22, weight: .semibold)).monospacedDigit()
                .fixedSize(horizontal: true, vertical: false)
        }
    }

    private func remainingLabel(at date: Date) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("残り時間（目安）").font(.caption).foregroundStyle(.secondary)
            Text(reading.remainingText(at: date)).font(.callout).monospacedDigit()
        }.fixedSize(horizontal: true, vertical: false)
    }

    private func graph(at date: Date) -> some View {
        let samples = history.samples.filter { $0.date >= date.addingTimeInterval(-30) && $0.date <= date }
        let ceiling = max(1000, (samples.map(\.speed).max() ?? 0) * 1.15)
        return VStack(spacing: 4) {
            HStack {
                Text("直近30秒")
                Spacer()
                Text(samples.isEmpty ? "データ待ち" : "上限 \(DownloadActivityReading.bytes(ceiling))/s")
            }.font(.caption2).monospacedDigit().foregroundStyle(.tertiary)
            Canvas { context, size in
                let bounds = CGRect(origin: .zero, size: size)
                context.clip(to: Path(bounds))
                for fraction in [0.0, 0.5, 1.0] {
                    let y = 1 + (size.height - 2) * fraction
                    var grid = Path()
                    grid.move(to: CGPoint(x: 0, y: y))
                    grid.addLine(to: CGPoint(x: size.width, y: y))
                    context.stroke(grid, with: .color(.primary.opacity(0.07)), lineWidth: 1)
                }
                var line = Path()
                var previous: Date?
                for sample in samples {
                    let point = CGPoint(x: size.width * (1 + sample.date.timeIntervalSince(date) / 30),
                                        y: 1 + (size.height - 2) * (1 - sample.speed / ceiling))
                    // Reports can arrive in bursts. Leave missing intervals empty instead of inventing a rate.
                    if let previous, sample.date.timeIntervalSince(previous) <= DownloadActivityReading.updateTimeout {
                        line.addLine(to: point)
                    } else {
                        line.move(to: point)
                    }
                    if sample.date == samples.last?.date {
                        context.fill(Path(ellipseIn: CGRect(x: point.x - 2, y: point.y - 2, width: 4, height: 4)),
                                     with: .color(.yellow))
                    }
                    previous = sample.date
                }
                context.stroke(line, with: .color(.yellow), style: StrokeStyle(lineWidth: 2, lineJoin: .round))
            }
            .frame(height: 60)
            .clipped()
            .accessibilityLabel("直近30秒のダウンロード速度。\(reading.speedText(at: date))")
            HStack {
                Text("30秒前")
                Spacer()
                Text("現在")
            }.font(.caption2).foregroundStyle(.tertiary)
        }
    }
}

struct DownloadActivityReading: Equatable {
    let downloaded: Double
    let total: Double?
    let speed: Double?
    let eta: Double?
    let updatedAt: Double?
    static let updateTimeout = 5.0

    var fraction: Double? {
        guard let total, total > 0 else { return nil }
        return min(1, max(0, downloaded / total))
    }

    var transferredText: String {
        total.map { "\(Self.bytes(downloaded)) / \(Self.bytes($0))" } ?? "\(Self.bytes(downloaded)) 取得済み"
    }

    func isWaiting(at date: Date) -> Bool {
        updatedAt.map { date.timeIntervalSince1970 - $0 > Self.updateTimeout } ?? false
    }

    func speedText(at date: Date) -> String {
        if isWaiting(at: date) { return "更新待ち" }
        guard let speed else { return "計測中…" }
        if speed > 0 && speed < 1000 {
            return "\(speed.formatted(.number.precision(.fractionLength(0...1)))) B/s"
        }
        return "\(Self.bytes(speed))/s"
    }

    func remainingText(at date: Date) -> String {
        guard !isWaiting(at: date), let speed, speed > 0, let eta else { return "—" }
        if eta < 1 { return "まもなく完了" }
        let seconds = Int(eta.rounded(.up))
        if seconds < 60 { return "約\(seconds)秒" }
        if seconds < 3600 { return "約\(seconds / 60)分\(seconds % 60)秒" }
        return "約\(seconds / 3600)時間\((seconds % 3600) / 60)分"
    }

    static func bytes(_ value: Double) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        formatter.allowedUnits = [.useKB, .useMB, .useGB, .useTB]
        formatter.allowsNonnumericFormatting = false
        return formatter.string(fromByteCount: Int64(value))
    }
}

struct DownloadActivityHistory {
    struct Sample {
        let date: Date
        let speed: Double
    }
    private(set) var samples: [Sample] = []
    private var downloaded: Double?

    mutating func record(_ reading: DownloadActivityReading) {
        if let downloaded, reading.downloaded < downloaded { samples.removeAll() }
        downloaded = reading.downloaded
        guard let timestamp = reading.updatedAt, let speed = reading.speed else { return }
        let date = Date(timeIntervalSince1970: timestamp)
        samples.removeAll { $0.date < date.addingTimeInterval(-30) }
        if samples.last?.date == date {
            samples[samples.count - 1] = Sample(date: date, speed: speed)
        } else if samples.last.map({ $0.date < date }) ?? true {
            samples.append(Sample(date: date, speed: speed))
        }
    }
}
