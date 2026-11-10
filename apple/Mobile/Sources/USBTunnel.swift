import Foundation
import Network

/// Carries the existing pinned HTTPS connection through Mac-initiated USB sockets.
final class USBTunnel: @unchecked Sendable {
    private let queue = DispatchQueue(label: "TransiroYT.USB")
    private var listeners: [NWListener] = []
    private var local: [NWConnection] = []
    private var remote: [NWConnection] = []
    private var pairs: [UUID: (NWConnection, NWConnection)] = [:]
    private var closed = false

    func start() async throws -> URL {
        do {
            _ = try await listen(port: 48765, usb: true)
            let port = try await listen(port: .any, usb: false)
            return URL(string: "https://127.0.0.1:\(port.rawValue)")!
        } catch {
            close()
            throw error
        }
    }

    private func listen(port: NWEndpoint.Port, usb: Bool) async throws -> NWEndpoint.Port {
        try await withCheckedThrowingContinuation { continuation in
            queue.async { self.startListener(port: port, usb: usb) { continuation.resume(with: $0) } }
        }
    }

    private func startListener(port: NWEndpoint.Port, usb: Bool,
                               ready: ((Result<NWEndpoint.Port, Error>) -> Void)? = nil) {
        var ready = ready
        var bound: NWEndpoint.Port?
        do {
            let parameters = NWParameters.tcp
            parameters.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: port)
            let listener = try NWListener(using: parameters)
            listeners.append(listener)
            listener.stateUpdateHandler = { state in
                switch state {
                case .ready:
                    bound = listener.port
                    ready?(.success(listener.port!)); ready = nil
                case .failed(let error):
                    listener.cancel()
                    self.listeners.removeAll { $0 === listener }
                    if let pending = ready { pending(.failure(error)); ready = nil; return }
                    guard !self.closed else { return }
                    // iOS reclaims a suspended app's listening sockets (TN2277). The Mac
                    // and URLSession know only the bound port, so listen on it again.
                    let again = bound ?? port
                    print("TRANSIRO-USB listener port=\(again) failed: \(error); listening again")
                    self.queue.asyncAfter(deadline: .now() + 1) {
                        if !self.closed { self.startListener(port: again, usb: usb) }
                    }
                case .cancelled:
                    ready?(.failure(CancellationError())); ready = nil
                default: break
                }
            }
            listener.newConnectionHandler = { connection in
                connection.stateUpdateHandler = { state in
                    switch state {
                    case .failed, .cancelled: self.remove(connection)
                    default: break
                    }
                }
                connection.start(queue: self.queue)
                if usb { self.remote.append(connection) }
                else { self.local.append(connection) }
                self.pair()
            }
            listener.start(queue: queue)
        } catch {
            ready?(.failure(error))
        }
    }

    private func pair() {
        while !local.isEmpty && !remote.isEmpty {
            let client = local.removeFirst(), mac = remote.removeFirst()
            let id = UUID()
            pairs[id] = (client, mac)
            mac.send(content: Data([1]), completion: .contentProcessed { error in
                if error != nil { self.finish(id); return }
                self.pump(client, to: mac, id: id)
                self.pump(mac, to: client, id: id)
            })
        }
    }

    private func pump(_ source: NWConnection, to destination: NWConnection, id: UUID) {
        source.receive(minimumIncompleteLength: 1, maximumLength: 65536) { data, _, complete, error in
            guard let data, !data.isEmpty else { self.finish(id); return }
            destination.send(content: data, completion: .contentProcessed { sendError in
                if complete || error != nil || sendError != nil { self.finish(id) }
                else { self.pump(source, to: destination, id: id) }
            })
        }
    }

    private func remove(_ connection: NWConnection) {
        local.removeAll { $0 === connection }
        remote.removeAll { $0 === connection }
        let ids = pairs.compactMap { id, pair in
            pair.0 === connection || pair.1 === connection ? id : nil
        }
        for id in ids { finish(id) }
    }

    private func finish(_ id: UUID) {
        guard let pair = pairs.removeValue(forKey: id) else { return }
        pair.0.cancel(); pair.1.cancel()
    }

    func close() {
        queue.async {
            self.closed = true
            for listener in self.listeners { listener.cancel() }
            self.listeners.removeAll()
            for connection in self.local + self.remote { connection.cancel() }
            self.local.removeAll(); self.remote.removeAll()
            for id in Array(self.pairs.keys) { self.finish(id) }
        }
    }
}
