import Foundation

@MainActor
final class OpenRouterTranscriptionProvider: TranscriptionProvider {
    let descriptor = TranscriptionProviderDescriptor.openRouter

    private let client: OpenRouterTranscriptionClient
    private let apiKeyProvider: @MainActor () throws -> String?

    init() {
        self.client = OpenRouterTranscriptionClient()
        self.apiKeyProvider = {
            try APIKeychainStore.apiKey(providerId: TranscriptionProviderDescriptor.openRouter.id)
        }
    }

    init(
        client: OpenRouterTranscriptionClient,
        apiKeyProvider: @escaping @MainActor () throws -> String?
    ) {
        self.client = client
        self.apiKeyProvider = apiKeyProvider
    }

    func transcribe(
        _ request: TranscriptionRequest,
        onProgress: @escaping (TranscriptionProgressUpdate) -> Void
    ) async throws -> TranscriptionProviderResult {
        guard let apiKey = try apiKeyProvider(), !apiKey.isEmpty else {
            throw OpenRouterTranscriptionError.missingAPIKey
        }
        guard APITranscriptionModelStore.isSupportedModel(request.model.id, providerId: descriptor.id) else {
            throw OpenRouterTranscriptionError.unsupportedModel(APITranscriptionModelStore.supportedModelHint(providerId: descriptor.id))
        }

        let startedAt = Date()
        onProgress(TranscriptionProgressUpdate(
            progress: 0.15,
            elapsedSeconds: 0,
            message: "Uploading to OpenRouter..."
        ))

        let response = try await client.transcribe(
            audioData: request.audioData,
            modelId: request.model.id,
            apiKey: apiKey
        )

        let processingTime = Date().timeIntervalSince(startedAt)
        onProgress(TranscriptionProgressUpdate(
            progress: 1,
            elapsedSeconds: processingTime,
            message: "OpenRouter transcription complete"
        ))

        return TranscriptionProviderResult(
            text: response.text,
            durationSeconds: request.recordedDurationSeconds,
            processingTimeSeconds: processingTime,
            realtimeFactor: request.recordedDurationSeconds > 0 ? processingTime / request.recordedDurationSeconds : 0,
            provider: descriptor,
            model: request.model,
            workflow: request.workflow,
            estimatedCostUSD: response.usage?.cost
                ?? APITranscriptionModelStore.load()
                    .first(where: { $0.providerId == descriptor.id && $0.modelId == request.model.id })?
                    .estimatedCost(durationSeconds: request.recordedDurationSeconds)
        )
    }
}
