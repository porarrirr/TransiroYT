import Foundation
import CryptoKit
import Security

struct PairingLink {
    let base: URL
    let pin: String
    let code: String
    let usb: Bool

    init(_ raw: String) throws {
        guard var parts = URLComponents(string: raw.trimmingCharacters(in: .whitespacesAndNewlines)),
              parts.scheme == "https", let host = parts.host, parts.port != nil,
              parts.user == nil, parts.password == nil,
              let fragment = parts.fragment,
              let fields = URLComponents(string: "https://pair/?\(fragment)")?.queryItems,
              let pin = fields.first(where: { $0.name == "pin" })?.value,
              pin.count == 64, pin.allSatisfy({ $0.isHexDigit }),
              let code = fields.first(where: { $0.name == "code" })?.value, !code.isEmpty else {
            throw WorkerError.message("Macの「モバイル連携」に表示されたQRコードを読み取ってください。")
        }
        let usb = fields.first(where: { $0.name == "transport" })?.value == "usb"
        let octets = host.split(separator: ".").compactMap { Int($0) }
        guard (usb && host == "127.0.0.1") || (!usb && octets.count == 4 && octets.allSatisfy({ (0...255).contains($0) }) && (
              octets[0] == 10 || (octets[0] == 192 && octets[1] == 168) ||
                (octets[0] == 172 && (16...31).contains(octets[1])))) else {
            throw WorkerError.message("Wi-FiまたはUSBで接続したMacに表示されたQRコードを読み取ってください。")
        }
        parts.fragment = nil; parts.path = ""; parts.query = nil
        guard let base = parts.url else { throw WorkerError.message("接続情報が不正です。MacでQRコードを再発行してください。") }
        self.base = base; self.pin = pin.lowercased(); self.code = code; self.usb = usb
    }
}

/// The pairing outlives app launches, so the device reconnects without a new QR code.
struct SavedPairing: Codable {
    let link: String
    let token: String

    private static let query: [String: Any] = [kSecClass as String: kSecClassGenericPassword,
                                               kSecAttrService as String: "com.porarri.transiroyt.pairing",
                                               kSecAttrAccount as String: "mac"]

    static func load() -> SavedPairing? {
        var query = query
        query[kSecReturnData as String] = true
        var item: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess, let data = item as? Data else { return nil }
        return try? JSONDecoder().decode(SavedPairing.self, from: data)
    }

    func save() throws {
        Self.delete()
        var item = Self.query
        item[kSecValueData as String] = try JSONEncoder().encode(self)
        item[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
        let status = SecItemAdd(item as CFDictionary, nil)
        guard status == errSecSuccess else { throw WorkerError.message("接続情報を保存できませんでした（\(status)）。") }
    }

    static func delete() { SecItemDelete(query as CFDictionary) }
}

enum WorkerError: LocalizedError {
    case message(String)
    case http(Int, String)
    var errorDescription: String? {
        switch self { case .message(let text), .http(_, let text): return text }
    }
    static func isTransient(_ error: Error) -> Bool {
        if case .http(let code, _) = error as? WorkerError { return code >= 500 || code == 429 }
        return error is URLError
    }
}

final class WorkerConnection: NSObject, URLSessionDelegate, URLSessionTaskDelegate, @unchecked Sendable {
    let link: PairingLink
    private var base: URL
    private var usb: USBTunnel?
    private lazy var session: URLSession = {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 15
        config.timeoutIntervalForResource = 60
        config.urlCache = nil
        return URLSession(configuration: config, delegate: self, delegateQueue: nil)
    }()

    init(link: PairingLink) { self.link = link; self.base = link.base }
    func open() async throws {
        if link.usb {
            let tunnel = USBTunnel()
            usb = tunnel
            base = try await tunnel.start()
        }
    }
    func close() { session.invalidateAndCancel(); usb?.close() }

    func request(_ path: String, method: String = "GET", token: String? = nil,
                 body: [String: Any]? = nil, waitForTask: Bool = false) async throws -> [String: Any] {
        var url = base.appendingPathComponent(path)
        if waitForTask { url.append(queryItems: [URLQueryItem(name: "wait", value: "true")]) }
        var request = URLRequest(url: url)
        request.httpMethod = method
        if let token { request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
        if let body {
            request.httpBody = try JSONSerialization.data(withJSONObject: body)
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw WorkerError.message("Macの応答が不正です。") }
        let result = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] ?? [:]
        guard (200..<300).contains(http.statusCode) else {
            throw WorkerError.http(http.statusCode, result["detail"] as? String ?? "Macとの通信エラー（\(http.statusCode)）")
        }
        return result
    }

    func urlSession(_ session: URLSession, didReceive challenge: URLAuthenticationChallenge,
                    completionHandler: @escaping (URLSession.AuthChallengeDisposition, URLCredential?) -> Void) {
        guard challenge.protectionSpace.authenticationMethod == NSURLAuthenticationMethodServerTrust,
              challenge.protectionSpace.host == link.base.host,
              let trust = challenge.protectionSpace.serverTrust,
              let certificates = SecTrustCopyCertificateChain(trust) as? [SecCertificate],
              let certificate = certificates.first else {
            completionHandler(.cancelAuthenticationChallenge, nil); return
        }
        let hash = SHA256.hash(data: SecCertificateCopyData(certificate) as Data)
            .map { String(format: "%02x", $0) }.joined()
        guard hash == link.pin else { completionHandler(.cancelAuthenticationChallenge, nil); return }
        completionHandler(.useCredential, URLCredential(trust: trust))
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
        completionHandler(nil)
    }
}
