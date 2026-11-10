import Foundation

/// Local HTTP transport. The store owns UI state and installation identity checks.
@MainActor
struct BackendClient {
    let session: URLSession

    func healthResponse() async throws -> (Data, URLResponse) {
        try await session.data(from: URL(string: "http://127.0.0.1:8765/health")!)
    }

    func send(_ path: String, token: String, method: String = "GET", body: Data? = nil,
              contentType: String = "application/octet-stream") async throws -> (Data, HTTPURLResponse) {
        var request = URLRequest(url: URL(string: "http://127.0.0.1:8765\(path)")!)
        request.httpMethod = method
        if path.hasPrefix("/api/task-routing/presets/") { request.timeoutInterval = 35 }
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        request.httpBody = body
        if body != nil { request.setValue(contentType, forHTTPHeaderField: "Content-Type") }
        let (data, response) = try await session.data(for: request)
        guard let response = response as? HTTPURLResponse else {
            throw DesktopError.message("サーバーの応答を読み取れません。")
        }
        return (data, response)
    }

    func request<T: Decodable>(_ path: String, token: String, method: String = "GET", body: Data? = nil,
                               contentType: String = "application/octet-stream") async throws -> T {
        guard !token.isEmpty else { throw DesktopError.message("接続の準備が必要です。「はじめに」を開いてください。") }
        let (data, response) = try await send(path, token: token, method: method, body: body, contentType: contentType)
        if response.statusCode == 401 { throw DesktopError.unauthorized }
        guard (200..<300).contains(response.statusCode) else {
            let detail = (try? JSONDecoder().decode(APIError.self, from: data))?.detail
            throw DesktopError.message(detail ?? "APIエラー: HTTP \(response.statusCode)")
        }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(T.self, from: data)
    }

}
