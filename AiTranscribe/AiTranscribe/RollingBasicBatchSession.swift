/*
 RollingBasicBatchSession.swift
 ==============================

 Transcribes completed basic-recording chunks while capture continues.
 Local ASR chunks are processed sequentially to avoid concurrent access to the
 same loaded NeMo/Whisper model; API providers can later use a parallel variant.
 */

import Foundation

@MainActor
final class RollingBasicBatchSession {
    private struct ChunkResult {
        let index: Int
        let text: String
        let processingTimeSeconds: Double
    }

    private let apiClient: APIClient
    private let provider: TranscriptionProviderDescriptor
    private let modelProvider: () -> TranscriptionModelDescriptor
    private let isReadyToProcess: () -> Bool
    private let startedAt = Date()

    private var pendingChunks: [AudioRecorderChunk] = []
    private var chunkResults: [Int: ChunkResult] = [:]
    private var processingTask: Task<Void, Never>?
    private var processingError: Error?
    private var isFinishing = false
    private var totalEnqueuedChunks = 0
    private var completedChunks = 0
    private var didReportWaitingForModel = false

    var onStatusUpdate: ((String) -> Void)?

    var hasStarted: Bool {
        totalEnqueuedChunks > 0
    }

    init(
        apiClient: APIClient,
        provider: TranscriptionProviderDescriptor,
        modelProvider: @escaping () -> TranscriptionModelDescriptor,
        isReadyToProcess: @escaping () -> Bool
    ) {
        self.apiClient = apiClient
        self.provider = provider
        self.modelProvider = modelProvider
        self.isReadyToProcess = isReadyToProcess
    }

    func enqueue(_ chunk: AudioRecorderChunk) {
        pendingChunks.append(chunk)
        totalEnqueuedChunks += 1
        print(String(
            format: "AiTranscribe INFO RollingBatch queued index=%d duration=%.2fs bytes=%d pending=%d totalQueued=%d completed=%d",
            chunk.index,
            chunk.durationSeconds,
            chunk.audioData.count,
            pendingChunks.count,
            totalEnqueuedChunks,
            completedChunks
        ))
        onStatusUpdate?("Recording... queued batch \(chunk.index)")
        startProcessingIfNeeded()
    }

    func finish(finalChunk: AudioRecorderChunk?, totalDurationSeconds: Double) async throws -> TranscriptionProviderResult {
        print(String(
            format: "AiTranscribe INFO RollingBatch finish requested totalDuration=%.2fs queued=%d completed=%d pending=%d hasFinalChunk=%@",
            totalDurationSeconds,
            totalEnqueuedChunks,
            completedChunks,
            pendingChunks.count,
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
                domain: "AiTranscribe.RollingBasicBatchSession",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "No rolling batch audio was transcribed"]
            )
        }

        let text = mergeChunkTexts(orderedResults.map(\.text))
        let processingTime = orderedResults.reduce(0) { $0 + $1.processingTimeSeconds }
        let realtimeFactor = processingTime > 0 ? totalDurationSeconds / processingTime : 0
        print(String(
            format: "AiTranscribe INFO RollingBatch merged chunks=%d totalProviderTime=%.2fs audioDuration=%.2fs rtf=%.3f outputWords=%d elapsedSinceStart=%.2fs",
            orderedResults.count,
            processingTime,
            totalDurationSeconds,
            realtimeFactor,
            text.split(separator: " ").count,
            Date().timeIntervalSince(startedAt)
        ))

        return TranscriptionProviderResult(
            text: text,
            durationSeconds: totalDurationSeconds,
            processingTimeSeconds: processingTime,
            realtimeFactor: realtimeFactor,
            provider: provider,
            model: modelProvider(),
            workflow: .basicBatch
        )
    }

    func cancel() {
        processingTask?.cancel()
        processingTask = nil
        pendingChunks.removeAll()
        isFinishing = true
        print(String(
            format: "AiTranscribe WARN RollingBatch cancelled queued=%d completed=%d elapsed=%.2fs",
            totalEnqueuedChunks,
            completedChunks,
            Date().timeIntervalSince(startedAt)
        ))
    }

    private func startProcessingIfNeeded() {
        guard processingTask == nil else { return }

        print("AiTranscribe DEBUG RollingBatch processor started")
        processingTask = Task { [weak self] in
            await self?.processLoop()
        }
    }

    private func processLoop() async {
        while !Task.isCancelled {
            if let processingError {
                pendingChunks.removeAll()
                print("AiTranscribe ERROR RollingBatch failed error=\(processingError.localizedDescription)")
                onStatusUpdate?("Batch failed: \(processingError.localizedDescription)")
                return
            }

            guard !pendingChunks.isEmpty else {
                if isFinishing { return }
                try? await Task.sleep(nanoseconds: 100_000_000)
                continue
            }

            guard isReadyToProcess() else {
                if !didReportWaitingForModel {
                    print("AiTranscribe DEBUG RollingBatch waiting for loaded local model before processing queued batches")
                    onStatusUpdate?("Recording... waiting for model")
                    didReportWaitingForModel = true
                }
                try? await Task.sleep(nanoseconds: 500_000_000)
                continue
            }
            didReportWaitingForModel = false

            let chunk = pendingChunks.removeFirst()
            let startedAt = Date()
            print(String(
                format: "AiTranscribe INFO RollingBatch processing index=%d duration=%.2fs bytes=%d pendingAfterPop=%d",
                chunk.index,
                chunk.durationSeconds,
                chunk.audioData.count,
                pendingChunks.count
            ))
            onStatusUpdate?("Recording... processing batch \(chunk.index)")

            do {
                let response = try await apiClient.transcribeAudioDataWithProgress(chunk.audioData) { [weak self] progress, _ in
                    guard let self else { return }
                    let percent = Int(progress * 100)
                    if percent > 0 {
                        self.onStatusUpdate?("Recording... batch \(chunk.index) \(percent)%")
                    }
                }

                completedChunks += 1
                chunkResults[chunk.index] = ChunkResult(
                    index: chunk.index,
                    text: response.text,
                    processingTimeSeconds: response.processingTime
                )
                print(String(
                    format: "AiTranscribe INFO RollingBatch ready index=%d providerTime=%.2fs wallTime=%.2fs words=%d completed=%d/%d pending=%d",
                    chunk.index,
                    response.processingTime,
                    Date().timeIntervalSince(startedAt),
                    response.text.split(separator: " ").count,
                    completedChunks,
                    totalEnqueuedChunks,
                    pendingChunks.count
                ))
                onStatusUpdate?("Recording... batch \(chunk.index) ready")
            } catch {
                print("AiTranscribe ERROR RollingBatch batchFailed index=\(chunk.index) error=\(error.localizedDescription)")
                processingError = error
            }
        }
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
