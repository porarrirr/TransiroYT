import SwiftUI

struct LogsView: View {
    @ObservedObject var store: BackendStore
    @State private var query = ""

    private var visibleLog: String {
        query.isEmpty ? store.logs : store.logs.components(separatedBy: .newlines)
            .filter { $0.localizedCaseInsensitiveContains(query) }.joined(separator: "\n")
    }

    var body: some View {
        VStack(spacing: 12) {
            HStack {
                Text("ログ").font(.title2.bold())
                Spacer()
                Button("フォルダを開く", systemImage: "folder") { store.reveal(".local") }
                    .disabled(store.projectURL == nil)
            }
            TextField("検索", text: $query).textFieldStyle(.roundedBorder)
            ScrollView([.horizontal, .vertical]) {
                Text(visibleLog.isEmpty ? "該当するログはありません" : visibleLog)
                    .font(.system(.caption, design: .monospaced)).textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .topLeading).padding(12)
            }.frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color(nsColor: .textBackgroundColor), in: RoundedRectangle(cornerRadius: 8))
        }.padding(20)
    }
}
