import Foundation

struct APITranscriptionModel: Identifiable, Codable, Equatable {
    let id: UUID
    let providerId: String
    let providerName: String
    let modelId: String
    let displayName: String
    let audioCostPerMinuteUSD: Double?
    let createdAt: Date

    init(
        id: UUID = UUID(),
        providerId: String,
        providerName: String,
        modelId: String,
        displayName: String,
        audioCostPerMinuteUSD: Double? = nil,
        createdAt: Date = Date()
    ) {
        self.id = id
        self.providerId = providerId
        self.providerName = providerName
        self.modelId = modelId
        self.displayName = displayName
        self.audioCostPerMinuteUSD = audioCostPerMinuteUSD
        self.createdAt = createdAt
    }

    func estimatedCost(durationSeconds: Double) -> Double? {
        guard let audioCostPerMinuteUSD = audioCostPerMinuteUSD ?? Self.defaultAudioCostPerMinuteUSD(modelId: modelId, providerId: providerId) else { return nil }
        return max(0, durationSeconds / 60.0) * audioCostPerMinuteUSD
    }

    static func defaultAudioCostPerMinuteUSD(modelId: String, providerId: String) -> Double? {
        guard providerId == TranscriptionProviderDescriptor.openAI.id else { return nil }
        switch modelId.trimmingCharacters(in: .whitespacesAndNewlines) {
        case "gpt-4o-transcribe", "gpt-4o-transcribe-diarize":
            return 0.006
        case "gpt-4o-mini-transcribe", "gpt-4o-mini-transcribe-2025-12-15":
            return 0.003
        default:
            return nil
        }
    }
}

enum APITranscriptionModelStore {
    private static let storageKey = "apiTranscriptionModels"
    static let supportedOpenAITranscriptionModelIds = [
        "gpt-4o-transcribe",
        "gpt-4o-mini-transcribe",
        "gpt-4o-mini-transcribe-2025-12-15",
        "whisper-1",
        "gpt-4o-transcribe-diarize"
    ]
    static let supportedElevenLabsTranscriptionModelIds = [
        "scribe_v2",
        "scribe_v1"
    ]
    static let supportedModelHint = supportedOpenAITranscriptionModelIds.joined(separator: ", ")

    static func isSupportedOpenAITranscriptionModel(_ modelId: String) -> Bool {
        supportedOpenAITranscriptionModelIds.contains(modelId.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    static func isSupportedElevenLabsTranscriptionModel(_ modelId: String) -> Bool {
        supportedElevenLabsTranscriptionModelIds.contains(modelId.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    static func isSupportedModel(_ modelId: String, providerId: String) -> Bool {
        let trimmed = modelId.trimmingCharacters(in: .whitespacesAndNewlines)
        switch providerId {
        case TranscriptionProviderDescriptor.openAI.id:
            return isSupportedOpenAITranscriptionModel(trimmed)
        case TranscriptionProviderDescriptor.openRouter.id:
            return trimmed.contains("/") && !trimmed.hasSuffix("/")
        case TranscriptionProviderDescriptor.elevenLabs.id:
            return isSupportedElevenLabsTranscriptionModel(trimmed)
        default:
            return false
        }
    }

    static func supportedModelHint(providerId: String) -> String {
        switch providerId {
        case TranscriptionProviderDescriptor.openAI.id:
            return supportedModelHint
        case TranscriptionProviderDescriptor.openRouter.id:
            return "Use an OpenRouter Speech-to-Text model from the transcription collection, for example openai/whisper-1. Chat/LLM slugs such as google/gemini-* belong in the LLM tab and cannot use the STT endpoint."
        case TranscriptionProviderDescriptor.elevenLabs.id:
            return "Use scribe_v2 or scribe_v1. scribe_v2_realtime is a WebSocket realtime model and is not supported by this batch model form yet."
        default:
            return "Choose a supported public Speech-to-Text provider."
        }
    }

    static func load() -> [APITranscriptionModel] {
        guard let data = UserDefaults.standard.data(forKey: storageKey),
              let models = try? JSONDecoder().decode([APITranscriptionModel].self, from: data) else {
            return []
        }
        return models
    }

    static func save(_ models: [APITranscriptionModel]) {
        guard let data = try? JSONEncoder().encode(models) else { return }
        UserDefaults.standard.set(data, forKey: storageKey)
    }

    static func upsert(_ model: APITranscriptionModel) -> [APITranscriptionModel] {
        var models = load()
        if let existingIndex = models.firstIndex(where: { $0.id == model.id }) {
            models[existingIndex] = model
        } else if let existingIndex = models.firstIndex(where: { $0.providerId == model.providerId && $0.modelId == model.modelId }) {
            models[existingIndex] = model
        } else {
            models.append(model)
        }
        save(models)
        return models
    }

    static func delete(id: UUID) -> [APITranscriptionModel] {
        let models = load().filter { $0.id != id }
        save(models)
        return models
    }
}
