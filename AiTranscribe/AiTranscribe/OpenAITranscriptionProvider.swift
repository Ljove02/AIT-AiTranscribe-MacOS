import Foundation

@MainActor
final class OpenAITranscriptionProvider: TranscriptionProvider {
    let descriptor = TranscriptionProviderDescriptor.openAI

    private let client: OpenAITranscriptionClient
    private let apiKeyProvider: @MainActor () throws -> String?

    init() {
        self.client = OpenAITranscriptionClient()
        self.apiKeyProvider = {
            try APIKeychainStore.apiKey(providerId: "openai")
        }
    }

    init(
        client: OpenAITranscriptionClient,
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
            throw OpenAITranscriptionError.missingAPIKey
        }
        guard APITranscriptionModelStore.isSupportedOpenAITranscriptionModel(request.model.id) else {
            throw OpenAITranscriptionError.unsupportedModel(APITranscriptionModelStore.supportedModelHint)
        }

        let startedAt = Date()
        onProgress(TranscriptionProgressUpdate(
            progress: 0.15,
            elapsedSeconds: 0,
            message: "Uploading to OpenAI..."
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
            message: "OpenAI transcription complete"
        ))

        return TranscriptionProviderResult(
            text: response.text,
            durationSeconds: request.recordedDurationSeconds,
            processingTimeSeconds: processingTime,
            realtimeFactor: request.recordedDurationSeconds > 0 ? processingTime / request.recordedDurationSeconds : 0,
            provider: descriptor,
            model: request.model,
            workflow: request.workflow,
            estimatedCostUSD: APITranscriptionModelStore.load()
                .first(where: { $0.providerId == descriptor.id && $0.modelId == request.model.id })?
                .estimatedCost(durationSeconds: request.recordedDurationSeconds)
        )
    }
}
