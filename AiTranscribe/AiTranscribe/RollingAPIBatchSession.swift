/*
 RollingAPIBatchSession.swift
 ============================

 Transcribes public/API quick-recording chunks while capture continues.
 API chunks can run concurrently because they do not share a local model process.
 */

import Foundation

@MainActor
final class RollingAPIBatchSession {
    private struct ChunkResult {
        let index: Int
        let text: String
        let processingTimeSeconds: Double
        let estimatedCostUSD: Double?
    }

    private struct RetryFailure: Error {
        let underlying: Error
        let attempts: Int
    }

    private struct ChunkTranscriptionFailure: LocalizedError {
        let providerName: String
        let index: Int
        let attempts: Int
        let completed: Int
        let queued: Int
        let underlying: Error

        var errorDescription: String? {
            let detail = RollingAPIBatchSession.describe(underlying)
            return "\(providerName) API batch \(index) failed after \(attempts) attempt\(attempts == 1 ? "" : "s"): \(detail). Completed \(completed) of \(queued) queued batches."
        }
    }

    private let provider: any TranscriptionProvider
    private let model: TranscriptionModelDescriptor
    private let maxConcurrentUploads: Int
    private let maxAttempts = 3
    private let startedAt = Date()

    private var pendingChunks: [AudioRecorderChunk] = []
    private var inFlightTasks: [Int: Task<Void, Never>] = [:]
    private var chunkResults: [Int: ChunkResult] = [:]
    private var processingTask: Task<Void, Never>?
    private var processingError: Error?
    private var isFinishing = false
    private var totalEnqueuedChunks = 0
    private var completedChunks = 0

    var onStatusUpdate: ((String) -> Void)?

    var hasStarted: Bool {
        totalEnqueuedChunks > 0
    }

    init(
        provider: any TranscriptionProvider,
        model: TranscriptionModelDescriptor,
        maxConcurrentUploads: Int = 4
    ) {
        self.provider = provider
        self.model = model
        self.maxConcurrentUploads = max(1, maxConcurrentUploads)
    }

    func enqueue(_ chunk: AudioRecorderChunk) {
        pendingChunks.append(chunk)
        totalEnqueuedChunks += 1
        AppLogger.info(String(
            format: "RollingAPI queued provider=%@ model=%@ index=%d duration=%.2fs bytes=%d pending=%d inFlight=%d totalQueued=%d completed=%d",
            provider.descriptor.displayName,
            model.id,
            chunk.index,
            chunk.durationSeconds,
            chunk.audioData.count,
            pendingChunks.count,
            inFlightTasks.count,
            totalEnqueuedChunks,
            completedChunks
        ))
        onStatusUpdate?("Recording... queued API batch \(chunk.index)")
        startProcessingIfNeeded()
    }

    func finish(finalChunk: AudioRecorderChunk?, totalDurationSeconds: Double) async throws -> TranscriptionProviderResult {
        AppLogger.info(String(
            format: "RollingAPI finish requested provider=%@ totalDuration=%.2fs queued=%d completed=%d pending=%d inFlight=%d hasFinalChunk=%@",
            provider.descriptor.displayName,
            totalDurationSeconds,
            totalEnqueuedChunks,
            completedChunks,
            pendingChunks.count,
            inFlightTasks.count,
            finalChunk == nil ? "false" : "true"
        ))

        if let finalChunk {
            enqueue(finalChunk)
        }

        isFinishing = true
        startProcessingIfNeeded()
        await processingTask?.value

        if let processingError {
            throw processingError
        }

        let orderedResults = chunkResults
            .values
            .sorted { $0.index < $1.index }

        guard !orderedResults.isEmpty else {
            throw NSError(
                domain: "AiTranscribe.RollingAPIBatchSession",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "No public API batch audio was transcribed"]
            )
        }

        let text = mergeChunkTexts(orderedResults.map(\.text))
        let providerTime = orderedResults.reduce(0) { $0 + $1.processingTimeSeconds }
        let costs = orderedResults.compactMap(\.estimatedCostUSD)
        let estimatedCost = costs.isEmpty ? nil : costs.reduce(0, +)
        let elapsed = Date().timeIntervalSince(startedAt)

        AppLogger.info(String(
            format: "RollingAPI merged provider=%@ chunks=%d providerTimeSum=%.2fs elapsedSinceStart=%.2fs audioDuration=%.2fs outputWords=%d estimatedCost=$%.6f",
            provider.descriptor.displayName,
            orderedResults.count,
            providerTime,
            elapsed,
            totalDurationSeconds,
            text.split(separator: " ").count,
            estimatedCost ?? 0
        ))

        return TranscriptionProviderResult(
            text: text,
            durationSeconds: totalDurationSeconds,
            processingTimeSeconds: providerTime,
            realtimeFactor: totalDurationSeconds > 0 ? providerTime / totalDurationSeconds : 0,
            provider: provider.descriptor,
            model: model,
            workflow: .basicBatch,
            estimatedCostUSD: estimatedCost
        )
    }

    func cancel() {
        processingTask?.cancel()
        processingTask = nil
        for task in inFlightTasks.values {
            task.cancel()
        }
        inFlightTasks.removeAll()
        pendingChunks.removeAll()
        isFinishing = true
        AppLogger.warning(String(
            format: "RollingAPI cancelled provider=%@ queued=%d completed=%d elapsed=%.2fs",
            provider.descriptor.displayName,
            totalEnqueuedChunks,
            completedChunks,
            Date().timeIntervalSince(startedAt)
        ))
    }

    private func startProcessingIfNeeded() {
        guard processingTask == nil else { return }

        AppLogger.debug(String(
            format: "RollingAPI processor started provider=%@ maxConcurrent=%d",
            provider.descriptor.displayName,
            maxConcurrentUploads
        ))
        processingTask = Task { [weak self] in
            await self?.processLoop()
        }
    }

    private func processLoop() async {
        while !Task.isCancelled {
            if let processingError {
                pendingChunks.removeAll()
                for task in inFlightTasks.values {
                    task.cancel()
                }
                inFlightTasks.removeAll()
                AppLogger.error("RollingAPI failed error=\(processingError.localizedDescription)")
                onStatusUpdate?("API batch failed: \(processingError.localizedDescription)")
                return
            }

            while inFlightTasks.count < maxConcurrentUploads, !pendingChunks.isEmpty {
                let chunk = pendingChunks.removeFirst()
                launch(chunk)
            }

            if isFinishing, pendingChunks.isEmpty, inFlightTasks.isEmpty {
                return
            }

            try? await Task.sleep(nanoseconds: 100_000_000)
        }
    }

    private func launch(_ chunk: AudioRecorderChunk) {
        AppLogger.info(String(
            format: "RollingAPI processing provider=%@ index=%d duration=%.2fs bytes=%d pendingAfterPop=%d inFlight=%d",
            provider.descriptor.displayName,
            chunk.index,
            chunk.durationSeconds,
            chunk.audioData.count,
            pendingChunks.count,
            inFlightTasks.count + 1
        ))
        onStatusUpdate?("Recording... API batch \(chunk.index) uploading")

        let task = Task { [weak self] in
            guard let self else { return }
            await self.process(chunk)
        }
        inFlightTasks[chunk.index] = task
    }

    private func process(_ chunk: AudioRecorderChunk) async {
        let startedAt = Date()
        defer {
            inFlightTasks.removeValue(forKey: chunk.index)
        }

        do {
            let request = TranscriptionRequest(
                audioData: chunk.audioData,
                recordedDurationSeconds: chunk.durationSeconds,
                workflow: .basicBatch,
                model: model
            )
            let result = try await transcribeWithRetry(request, chunk: chunk) { [weak self] update in
                guard let self else { return }
                if let message = update.message, !message.isEmpty {
                    self.onStatusUpdate?("Recording... \(message)")
                }
            }

            completedChunks += 1
            chunkResults[chunk.index] = ChunkResult(
                index: chunk.index,
                text: result.text,
                processingTimeSeconds: result.processingTimeSeconds,
                estimatedCostUSD: result.estimatedCostUSD
            )
            AppLogger.info(String(
                format: "RollingAPI ready provider=%@ index=%d providerTime=%.2fs wallTime=%.2fs words=%d completed=%d/%d pending=%d inFlight=%d estimatedCost=$%.6f",
                provider.descriptor.displayName,
                chunk.index,
                result.processingTimeSeconds,
                Date().timeIntervalSince(startedAt),
                result.text.split(separator: " ").count,
                completedChunks,
                totalEnqueuedChunks,
                pendingChunks.count,
                max(0, inFlightTasks.count - 1),
                result.estimatedCostUSD ?? 0
            ))
            onStatusUpdate?("Recording... API batch \(chunk.index) ready")
        } catch {
            let retryFailure = error as? RetryFailure
            let underlying = retryFailure?.underlying ?? error
            let attempts = retryFailure?.attempts ?? 1
            let failure = ChunkTranscriptionFailure(
                providerName: provider.descriptor.displayName,
                index: chunk.index,
                attempts: attempts,
                completed: completedChunks,
                queued: totalEnqueuedChunks,
                underlying: underlying
            )
            AppLogger.error(String(
                format: "RollingAPI batchFailed provider=%@ index=%d attempts=%d completed=%d/%d pending=%d inFlight=%d error=%@",
                provider.descriptor.displayName,
                chunk.index,
                attempts,
                completedChunks,
                totalEnqueuedChunks,
                pendingChunks.count,
                max(0, inFlightTasks.count - 1),
                Self.describe(underlying)
            ))
            processingError = failure
        }
    }

    private func transcribeWithRetry(
        _ request: TranscriptionRequest,
        chunk: AudioRecorderChunk,
        onProgress: @escaping (TranscriptionProgressUpdate) -> Void
    ) async throws -> TranscriptionProviderResult {
        var attempt = 1

        while true {
            do {
                if attempt > 1 {
                    onStatusUpdate?("Recording... API batch \(chunk.index) retry \(attempt)")
                }
                return try await provider.transcribe(request, onProgress: onProgress)
            } catch {
                guard !Task.isCancelled else { throw error }
                guard isRetryable(error), attempt < maxAttempts else {
                    throw RetryFailure(underlying: error, attempts: attempt)
                }

                let delayNanoseconds = retryDelayNanoseconds(forAttempt: attempt)
                AppLogger.warning(String(
                    format: "RollingAPI retry provider=%@ index=%d attempt=%d/%d delay=%.1fs error=%@",
                    provider.descriptor.displayName,
                    chunk.index,
                    attempt + 1,
                    maxAttempts,
                    Double(delayNanoseconds) / 1_000_000_000,
                    Self.describe(error)
                ))
                onStatusUpdate?("Recording... API batch \(chunk.index) retrying")
                try? await Task.sleep(nanoseconds: delayNanoseconds)
                attempt += 1
            }
        }
    }

    private func retryDelayNanoseconds(forAttempt attempt: Int) -> UInt64 {
        let seconds = min(8.0, pow(2.0, Double(attempt - 1)))
        return UInt64(seconds * 1_000_000_000)
    }

    private func isRetryable(_ error: Error) -> Bool {
        if error is CancellationError {
            return false
        }

        if let urlError = error as? URLError {
            switch urlError.code {
            case .timedOut, .networkConnectionLost, .cannotConnectToHost, .cannotFindHost, .dnsLookupFailed, .notConnectedToInternet:
                return true
            default:
                return false
            }
        }

        let nsError = error as NSError
        if nsError.domain == NSURLErrorDomain {
            return [
                NSURLErrorTimedOut,
                NSURLErrorNetworkConnectionLost,
                NSURLErrorCannotConnectToHost,
                NSURLErrorCannotFindHost,
                NSURLErrorDNSLookupFailed,
                NSURLErrorNotConnectedToInternet
            ].contains(nsError.code)
        }

        switch error {
        case OpenAITranscriptionError.apiError(let statusCode, _):
            return statusCode == 408 || statusCode == 429 || (500..<600).contains(statusCode)
        case ElevenLabsTranscriptionError.apiError(let statusCode, _):
            return statusCode == 408 || statusCode == 429 || (500..<600).contains(statusCode)
        case OpenRouterTranscriptionError.apiError(let statusCode, _):
            return statusCode == 408 || statusCode == 429 || (500..<600).contains(statusCode)
        default:
            return false
        }
    }

    nonisolated private static func describe(_ error: Error) -> String {
        if let description = (error as? LocalizedError)?.errorDescription, !description.isEmpty {
            return description
        }
        return error.localizedDescription
    }

    private func mergeChunkTexts(_ texts: [String]) -> String {
        var merged = ""
        for text in texts {
            let cleaned = text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !cleaned.isEmpty else { continue }
            let deduplicated = deduplicateOverlap(previous: merged, current: cleaned)
            if merged.isEmpty {
                merged = deduplicated
            } else if !deduplicated.isEmpty {
                if !merged.hasSuffix(" ") {
                    merged += " "
                }
                merged += deduplicated
            }
        }
        return merged.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func deduplicateOverlap(previous: String, current: String, window: Int = 20) -> String {
        guard !previous.isEmpty, !current.isEmpty else { return current }

        let previousWords = previous.split(separator: " ").map(String.init)
        let currentWords = current.split(separator: " ").map(String.init)
        guard !previousWords.isEmpty, !currentWords.isEmpty else { return current }

        let previousTail = Array(previousWords.suffix(window))
        let currentHead = Array(currentWords.prefix(window))
        var bestOverlap = 0

        let maxOverlap = min(previousTail.count, currentHead.count)
        if maxOverlap > 0 {
            for overlap in 1...maxOverlap {
                if Array(previousTail.suffix(overlap)) == Array(currentHead.prefix(overlap)) {
                    bestOverlap = overlap
                }
            }
        }

        guard bestOverlap > 0 else { return current }
        return currentWords.dropFirst(bestOverlap).joined(separator: " ")
    }
}
