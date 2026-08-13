import Foundation

struct APILanguageModel: Identifiable, Codable, Equatable {
    let id: UUID
    let providerId: String
    let providerName: String
    let modelId: String
    let displayName: String
    let inputCostPerMillionTokensUSD: Double?
    let outputCostPerMillionTokensUSD: Double?
    let openRouterProviderOnly: [String]?
    let supportsVision: Bool
    let createdAt: Date

    enum CodingKeys: String, CodingKey {
        case id
        case providerId
        case providerName
        case modelId
        case displayName
        case inputCostPerMillionTokensUSD
        case outputCostPerMillionTokensUSD
        case openRouterProviderOnly
        case supportsVision
        case createdAt
    }

    init(
        id: UUID = UUID(),
        providerId: String,
        providerName: String,
        modelId: String,
        displayName: String,
        inputCostPerMillionTokensUSD: Double? = nil,
        outputCostPerMillionTokensUSD: Double? = nil,
        openRouterProviderOnly: [String]? = nil,
        supportsVision: Bool = false,
        createdAt: Date = Date()
    ) {
        self.id = id
        self.providerId = providerId
        self.providerName = providerName
        self.modelId = modelId
        self.displayName = displayName
        self.inputCostPerMillionTokensUSD = inputCostPerMillionTokensUSD
        self.outputCostPerMillionTokensUSD = outputCostPerMillionTokensUSD
        self.openRouterProviderOnly = openRouterProviderOnly
        self.supportsVision = supportsVision
        self.createdAt = createdAt
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(UUID.self, forKey: .id)
        providerId = try container.decode(String.self, forKey: .providerId)
        providerName = try container.decode(String.self, forKey: .providerName)
        modelId = try container.decode(String.self, forKey: .modelId)
        displayName = try container.decode(String.self, forKey: .displayName)
        inputCostPerMillionTokensUSD = try container.decodeIfPresent(Double.self, forKey: .inputCostPerMillionTokensUSD)
        outputCostPerMillionTokensUSD = try container.decodeIfPresent(Double.self, forKey: .outputCostPerMillionTokensUSD)
        openRouterProviderOnly = try container.decodeIfPresent([String].self, forKey: .openRouterProviderOnly)
        supportsVision = try container.decodeIfPresent(Bool.self, forKey: .supportsVision) ?? false
        createdAt = try container.decode(Date.self, forKey: .createdAt)
    }

    func estimatedCost(inputTokens: Int?, outputTokens: Int?) -> Double? {
        let inputCost = inputCostPerMillionTokensUSD.map { Double(max(inputTokens ?? 0, 0)) / 1_000_000.0 * $0 }
        let outputCost = outputCostPerMillionTokensUSD.map { Double(max(outputTokens ?? 0, 0)) / 1_000_000.0 * $0 }
        guard inputCost != nil || outputCost != nil else { return nil }
        return (inputCost ?? 0) + (outputCost ?? 0)
    }
}

enum APILanguageModelStore {
    private static let storageKey = "apiLanguageModels"

    static func load() -> [APILanguageModel] {
        guard let data = UserDefaults.standard.data(forKey: storageKey),
              let models = try? JSONDecoder().decode([APILanguageModel].self, from: data) else {
            return defaultModels
        }
        return models
    }

    static func save(_ models: [APILanguageModel]) {
        guard let data = try? JSONEncoder().encode(models) else { return }
        UserDefaults.standard.set(data, forKey: storageKey)
    }

    static func upsert(_ model: APILanguageModel) -> [APILanguageModel] {
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

    static func delete(id: UUID) -> [APILanguageModel] {
        let models = load().filter { $0.id != id }
        save(models)
        return models
    }

    static let defaultModels: [APILanguageModel] = [
        APILanguageModel(
            providerId: TranscriptionProviderDescriptor.openAI.id,
            providerName: TranscriptionProviderDescriptor.openAI.displayName,
            modelId: "gpt-4o-mini",
            displayName: "GPT-4o Mini"
        )
    ]
}
