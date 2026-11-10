import SwiftUI
import AVFoundation

struct PairingScanner: View {
    let onScan: (String) -> Void
    @State private var cameraError: String?
    @State private var cameraAuthorized = false
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Group {
                if cameraAuthorized {
                    QRScannerCamera(onScan: onScan, onError: { cameraError = $0 })
                        .ignoresSafeArea(edges: .bottom)
                } else if cameraError != nil {
                    ContentUnavailableView("カメラを利用できません", systemImage: "camera",
                        description: Text("設定でカメラへのアクセスを許可し、カメラを使えるiPhone・iPadで再度お試しください。"))
                } else { ProgressView("カメラを準備中") }
            }
            .task {
                cameraAuthorized = await AVCaptureDevice.requestAccess(for: .video)
                if !cameraAuthorized {
                    cameraError = "設定アプリでTransiroYT Mobileのカメラへのアクセスを許可してください。"
                }
            }
            .alert("カメラを起動できません", isPresented: Binding(get: { cameraError != nil }, set: { if !$0 { cameraError = nil } })) {
                Button("閉じる") { dismiss() }
            } message: { Text(cameraError ?? "") }
            .navigationTitle("MacのQRコードを読み取る")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .cancellationAction) { Button("キャンセル") { dismiss() } } }
        }
    }
}

private struct QRScannerCamera: UIViewControllerRepresentable {
    let onScan: (String) -> Void

    let onError: (String) -> Void

    func makeUIViewController(context: Context) -> QRScannerViewController {
        QRScannerViewController(onScan: onScan, onError: onError)
    }

    func updateUIViewController(_ scanner: QRScannerViewController, context: Context) {}
}

private final class QRScannerViewController: UIViewController, AVCaptureMetadataOutputObjectsDelegate {
    private let session = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "com.porarri.transiroyt.qr-camera")
    private let onScan: (String) -> Void
    private let onError: (String) -> Void
    private lazy var preview = AVCaptureVideoPreviewLayer(session: session)
    private var configured = false
    private var completed = false

    init(onScan: @escaping (String) -> Void, onError: @escaping (String) -> Void) {
        self.onScan = onScan
        self.onError = onError
        super.init(nibName: nil, bundle: nil)
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black
        preview.videoGravity = .resizeAspectFill
        view.layer.addSublayer(preview)
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        preview.frame = view.bounds
        let angle: CGFloat
        switch view.window?.windowScene?.interfaceOrientation {
        case .landscapeLeft: angle = 0
        case .landscapeRight: angle = 180
        case .portraitUpsideDown: angle = 270
        default: angle = 90
        }
        if let connection = preview.connection, connection.isVideoRotationAngleSupported(angle) {
            connection.videoRotationAngle = angle
        }
    }

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        // Attach and lay out the preview before starting the camera; session work must not block the UI.
        sessionQueue.async { [self] in
            do {
                if !configured { try configureSession() }
                session.startRunning()
                DispatchQueue.main.async { self.view.setNeedsLayout() }
            } catch {
                DispatchQueue.main.async { self.onError(error.localizedDescription) }
            }
        }
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        sessionQueue.async { self.session.stopRunning() }
    }

    private func configureSession() throws {
        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            throw CameraError.unavailable
        }
        let input = try AVCaptureDeviceInput(device: camera)
        let output = AVCaptureMetadataOutput()
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        session.sessionPreset = .high
        guard session.canAddInput(input), session.canAddOutput(output) else {
            throw CameraError.unavailable
        }
        session.addInput(input)
        session.addOutput(output)
        output.setMetadataObjectsDelegate(self, queue: .main)
        output.metadataObjectTypes = [.qr]
        configured = true
    }

    func metadataOutput(_ output: AVCaptureMetadataOutput, didOutput metadataObjects: [AVMetadataObject],
                        from connection: AVCaptureConnection) {
        guard !completed else { return }
        for case let code as AVMetadataMachineReadableCodeObject in metadataObjects {
            guard let value = code.stringValue, (try? PairingLink(value)) != nil else { continue }
            completed = true
            sessionQueue.async { self.session.stopRunning() }
            onScan(value)
            return
        }
    }

    private enum CameraError: LocalizedError {
        case unavailable
        var errorDescription: String? { "背面カメラを起動できません。カメラを使用中のアプリを閉じて、もう一度お試しください。" }
    }
}
