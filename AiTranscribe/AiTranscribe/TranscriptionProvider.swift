/*
 TranscriptionProvider.swift
 ===========================

 Provider abstraction for transcription engines.
 The first implementation wraps the existing local backend so behavior stays the same
 while future API providers, batching, privacy mode, and dashboard usage can share one seam.
 */

import Foundation

public enum TranscriptionProviderSource: String, Codable, Equatable {
    case local
    case api

    var displayName: String {
        switch self {
        case .local: return "Local"
        case .api: return "API"
        }
    }
}

public enum TranscriptionWorkflow: String, Codable, Equatable {
    case basic
    case basicBatch = "basic_batch"
    case session
    case streaming
    case polishing
}

struct TranscriptionProviderCapabilities: Codable, Equatable {
    let supportsBasic: Bool
    let supportsSessions: Bool
    let supportsBatching: Bool
    let supportsParallelBatching: Bool
    let supportsPolishing: Bool
    let supportsMultilingual: Bool

    static let localBackend = TranscriptionProviderCapabilities(
        supportsBasic: true,
        supportsSessions: true,
        supportsBatching: true,
        supportsParallelBatching: false,
        supportsPolishing: false,
        supportsMultilingual: true
    )

    static let openAI = TranscriptionProviderCapabilities(
        supportsBasic: true,
        supportsSessions: true,
        supportsBatching: true,
        supportsParallelBatching: true,
        supportsPolishing: false,
        supportsMultilingual: true
    )

    static let openRouter = TranscriptionProviderCapabilities(
        supportsBasic: true,
        supportsSessions: true,
        supportsBatching: true,
        supportsParallelBatching: true,
        supportsPolishing: false,
        supportsMultilingual: true
    )

    static let elevenLabs = TranscriptionProviderCapabilities(
        supportsBasic: true,
        supportsSessions: true,
        supportsBatching: true,
        supportsParallelBatching: true,
        supportsPolishing: false,
        supportsMultilingual: true
    )
}

struct TranscriptionProviderDescriptor: Identifiable, Codable, Equatable {
    let id: String
    let displayName: String
    let source: TranscriptionProviderSource
    let capabilities: TranscriptionProviderCapabilities
    let privacyDescription: String

    static let localBackend = TranscriptionProviderDescriptor(
        id: "local",
        displayName: "Local Backend",
        source: .local,
        capabilities: .localBackend,
        privacyDescription: "Audio and text stay on this Mac."
    )

    static let openAI = TranscriptionProviderDescriptor(
        id: "openai",
        displayName: "OpenAI",
        source: .api,
        capabilities: .openAI,
        privacyDescription: "Audio is sent to OpenAI for transcription."
    )

    static let openRouter = TranscriptionProviderDescriptor(
        id: "openrouter",
        displayName: "OpenRouter",
        source: .api,
        capabilities: .openRouter,
        privacyDescription: "Audio is sent to OpenRouter for transcription."
    )

    static let elevenLabs = TranscriptionProviderDescriptor(
        id: "elevenlabs",
        displayName: "ElevenLabs",
        source: .api,
        capabilities: .elevenLabs,
        privacyDescription: "Audio is sent to ElevenLabs for transcription."
    )
}

struct TranscriptionModelDescriptor: Identifiable, Codable, Equatable {
    let id: String
    let displayName: String
    let providerId: String
    let source: TranscriptionProviderSource

    static let openAIModels: [TranscriptionModelDescriptor] = [
        TranscriptionModelDescriptor(
            id: "gpt-4o-transcribe",
            displayName: "GPT-4o Transcribe",
            providerId: TranscriptionProviderDescriptor.openAI.id,
            source: .api
        ),
        TranscriptionModelDescriptor(
            id: "gpt-4o-mini-transcribe",
            displayName: "GPT-4o Mini Transcribe",
            providerId: TranscriptionProviderDescriptor.openAI.id,
            source: .api
        ),
        TranscriptionModelDescriptor(
            id: "whisper-1",
            displayName: "Whisper API",
            providerId: TranscriptionProviderDescriptor.openAI.id,
            source: .api
        )
    ]

    static func openAIModel(id: String?) -> TranscriptionModelDescriptor {
        openAIModels.first { $0.id == id } ?? openAIModels[0]
    }
}

struct TranscriptionProgressUpdate: Equatable {
    let progress: Double
    let elapsedSeconds: Double
    let message: String?

    init(progress: Double, elapsedSeconds: Double, message: String? = nil) {
        self.progress = progress
        self.elapsedSeconds = elapsedSeconds
        self.message = message
    }
}

struct TranscriptionUsageMetadata: Codable, Equatable {
    let providerId: String
    let providerName: String
    let providerSource: TranscriptionProviderSource
    let modelId: String
    let modelName: String
    let workflow: TranscriptionWorkflow
    let audioDurationSeconds: Double
    let processingTimeSeconds: Double
    let realtimeFactor: Double
    let estimatedCostUSD: Double?
    let createdAt: Date
}

struct TranscriptionRequest {
    let audioData: Data
    let recordedDurationSeconds: Double
    let workflow: TranscriptionWorkflow
    let model: TranscriptionModelDescriptor
}

struct TranscriptionProviderResult {
    let text: String
    let durationSeconds: Double
    let processingTimeSeconds: Double
    let realtimeFactor: Double
    let provider: TranscriptionProviderDescriptor
    let model: TranscriptionModelDescriptor
    let workflow: TranscriptionWorkflow
    let estimatedCostUSD: Double?

    init(
        text: String,
        durationSeconds: Double,
        processingTimeSeconds: Double,
        realtimeFactor: Double,
        provider: TranscriptionProviderDescriptor,
        model: TranscriptionModelDescriptor,
        workflow: TranscriptionWorkflow,
        estimatedCostUSD: Double? = nil
    ) {
        self.text = text
        self.durationSeconds = durationSeconds
        self.processingTimeSeconds = processingTimeSeconds
        self.realtimeFactor = realtimeFactor
        self.provider = provider
        self.model = model
        self.workflow = workflow
        self.estimatedCostUSD = estimatedCostUSD
    }

    var usage: TranscriptionUsageMetadata {
        TranscriptionUsageMetadata(
            providerId: provider.id,
            providerName: provider.displayName,
            providerSource: provider.source,
            modelId: model.id,
            modelName: model.displayName,
            workflow: workflow,
            audioDurationSeconds: durationSeconds,
            processingTimeSeconds: processingTimeSeconds,
            realtimeFactor: realtimeFactor,
            estimatedCostUSD: estimatedCostUSD,
            createdAt: Date()
        )
    }
}

@MainActor
protocol TranscriptionProvider {
    var descriptor: TranscriptionProviderDescriptor { get }

    func transcribe(
        _ request: TranscriptionRequest,
        onProgress: @escaping (TranscriptionProgressUpdate) -> Void
    ) async throws -> TranscriptionProviderResult
}

@MainActor
final class LocalTranscriptionProvider: TranscriptionProvider {
    let descriptor = TranscriptionProviderDescriptor.localBackend

    private let apiClient: APIClient
    private let defaultBatchingThresholdSeconds: Double = 120
    private let defaultBatchRamBudgetMB: Int = 4096

    init(apiClient: APIClient) {
        self.apiClient = apiClient
    }

    func transcribe(
        _ request: TranscriptionRequest,
        onProgress: @escaping (TranscriptionProgressUpdate) -> Void
    ) async throws -> TranscriptionProviderResult {
        if shouldUseBasicBatching(for: request) {
            return try await transcribeBasicInBatches(request, onProgress: onProgress)
        }

        let response = try await apiClient.transcribeAudioDataWithProgress(request.audioData) { progress, elapsed in
            onProgress(TranscriptionProgressUpdate(progress: progress, elapsedSeconds: elapsed))
        }

        let duration = response.durationSeconds > 0
            ? response.durationSeconds
            : request.recordedDurationSeconds

        return TranscriptionProviderResult(
            text: response.text,
            durationSeconds: duration,
            processingTimeSeconds: response.processingTime,
            realtimeFactor: response.realtimeFactor,
            provider: descriptor,
            model: request.model,
            workflow: request.workflow
        )
    }

    private func transcribeBasicInBatches(
        _ request: TranscriptionRequest,
        onProgress: @escaping (TranscriptionProgressUpdate) -> Void
    ) async throws -> TranscriptionProviderResult {
        let startedAt = Date()
        let response = try await apiClient.transcribeAudioDataInBatches(
            request.audioData,
            modelId: request.model.id,
            ramBudgetMB: basicBatchRamBudgetMB
        ) { event in
            let elapsed = Date().timeIntervalSince(startedAt)
            let progress = max(0, min((event.percent ?? 0) / 100.0, 0.98))

            switch event.event {
            case "started":
                let total = event.totalBatches ?? 0
                onProgress(TranscriptionProgressUpdate(
                    progress: 0,
                    elapsedSeconds: elapsed,
                    message: total > 1 ? "Preparing \(total) batches..." : "Preparing batch..."
                ))

            case "batch_progress":
                let batch = event.batch ?? 0
                let total = event.total ?? event.totalBatches ?? 0
                onProgress(TranscriptionProgressUpdate(
                    progress: progress,
                    elapsedSeconds: elapsed,
                    message: total > 0 ? "Batch \(batch)/\(total)..." : "Batching..."
                ))

            case "batch_complete":
                let batch = event.batch ?? 0
                let total = event.total ?? event.totalBatches ?? 0
                let completedProgress = total > 0 ? Double(batch) / Double(total) : progress
                onProgress(TranscriptionProgressUpdate(
                    progress: min(completedProgress, 0.98),
                    elapsedSeconds: elapsed,
                    message: total > 0 ? "Completed batch \(batch)/\(total)" : "Batch complete"
                ))

            case "stats":
                if let eta = event.etaSeconds {
                    onProgress(TranscriptionProgressUpdate(
                        progress: progress,
                        elapsedSeconds: elapsed,
                        message: "Batching... \(Int(eta))s remaining"
                    ))
                }

            default:
                break
            }
        }

        let duration = response.durationSeconds > 0
            ? response.durationSeconds
            : request.recordedDurationSeconds

        return TranscriptionProviderResult(
            text: response.text,
            durationSeconds: duration,
            processingTimeSeconds: response.processingTime,
            realtimeFactor: response.realtimeFactor,
            provider: descriptor,
            model: request.model,
            workflow: .basicBatch
        )
    }

    private func shouldUseBasicBatching(for request: TranscriptionRequest) -> Bool {
        guard request.workflow == .basic else { return false }
        guard descriptor.capabilities.supportsBatching else { return false }

        if let storedValue = UserDefaults.standard.object(forKey: "basicTranscriptionBatchingEnabled") as? Bool,
           storedValue == false {
            return false
        }

        return request.recordedDurationSeconds >= basicBatchingThresholdSeconds
    }

    private var basicBatchingThresholdSeconds: Double {
        let value = UserDefaults.standard.double(forKey: "basicTranscriptionBatchingThresholdSeconds")
        return value > 0 ? value : defaultBatchingThresholdSeconds
    }

    private var basicBatchRamBudgetMB: Int {
        let value = UserDefaults.standard.integer(forKey: "basicTranscriptionBatchingRamBudgetMB")
        return value > 0 ? value : defaultBatchRamBudgetMB
    }
}
