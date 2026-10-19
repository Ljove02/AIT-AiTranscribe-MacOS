/*
 TranscriptionUploadSession.swift
 ================================

 Sends public-API transcription uploads on a fresh, single-use connection.

 Rolling quick-recording batches upload roughly every 30 s. Reusing
 URLSession.shared's pooled connection across those idle gaps produced
 "network connection lost" followed by TCP timeouts on every retry, while
 the first upload on a new connection always succeeded. Each upload therefore
 gets its own ephemeral session, invalidated as soon as the request finishes.
 The extra TLS handshake costs ~100 ms per upload.
 */

import Foundation

enum TranscriptionUploadSession {
    /// Upper bound for one upload attempt, including connect, send, and response.
    /// 180 s covers a 30 s batch many times over; large single-shot uploads get
    /// extra time assuming a slow ~100 KB/s uplink.
    static func resourceTimeout(forBodyBytes bytes: Int) -> TimeInterval {
        max(180, 120 + Double(bytes) / 100_000)
    }

    static func data(for request: URLRequest, label: String) async throws -> (Data, URLResponse) {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = request.timeoutInterval
        configuration.timeoutIntervalForResource = resourceTimeout(forBodyBytes: request.httpBody?.count ?? 0)
        configuration.waitsForConnectivity = false
        configuration.urlCache = nil

        let session = URLSession(
            configuration: configuration,
            delegate: UploadMetricsLogger(label: label, bytes: request.httpBody?.count ?? 0),
            delegateQueue: nil
        )
        defer { session.finishTasksAndInvalidate() }
        return try await session.data(for: request)
    }
}

private final class UploadMetricsLogger: NSObject, URLSessionTaskDelegate {
    private let label: String
    private let bytes: Int

    init(label: String, bytes: Int) {
        self.label = label
        self.bytes = bytes
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didFinishCollecting metrics: URLSessionTaskMetrics) {
        guard let transaction = metrics.transactionMetrics.last(where: { $0.networkProtocolName != nil })
            ?? metrics.transactionMetrics.last else { return }
        let status = (transaction.response as? HTTPURLResponse)?.statusCode ?? 0
        AppLogger.info(String(
            format: "TranscriptionUpload %@ status=%d proto=%@ reused=%@ remote=%@ bytes=%d duration=%.2fs",
            label,
            status,
            transaction.networkProtocolName ?? "none",
            transaction.isReusedConnection ? "true" : "false",
            transaction.remoteAddress ?? "none",
            bytes,
            metrics.taskInterval.duration
        ))
    }
}
