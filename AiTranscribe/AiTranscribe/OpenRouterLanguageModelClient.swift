import Foundation

struct OpenRouterLanguageModelClient {
    private let endpoint = URL(string: "https://openrouter.ai/api/v1/chat/completions")!
    private let session: URLSession

    init(session: URLSession = .shared) {
        self.session = session
    }

    func testModel(
        modelId: String,
        apiKey: String,
        providerOnly: [String]? = nil
    ) async throws -> OpenAILanguageModelResponse {
        try await generateResponse(
            systemInstruction: "You are running a short health check. Reply with exactly: AiTranscribe OK",
            userInput: "Confirm this model can respond.",
            modelId: modelId,
            apiKey: apiKey,
            providerOnly: providerOnly,
            maxOutputTokens: 32
        )
    }

    func generateSummary(
        transcript: String,
        preset: SummaryGenerationPresetRequest,
        modelId: String,
        apiKey: String,
        providerOnly: [String]? = nil
    ) async throws -> OpenAILanguageModelResponse {
        let instruction = """
        \(preset.systemPrompt)

        Target length: about \(preset.targetWords) words.
        Return only the finished summary. Do not include preambles.
        """
        let userInput = """
        Summarize this transcript:

        \(transcript)
        """
        return try await generateResponse(
            systemInstruction: instruction,
            userInput: userInput,
            modelId: modelId,
            apiKey: apiKey,
            providerOnly: providerOnly,
            maxOutputTokens: preset.maxOutputTokens
        )
    }

    func polishText(
        transcript: String,
        preset: PolishPreset,
        modelId: String,
        apiKey: String,
        providerOnly: [String]? = nil
    ) async throws -> OpenAILanguageModelResponse {
        let instruction = """
        You are polishing a voice transcript.

        Polish instruction:
        \(preset.trimmedPrompt)

        Rules:
        - Preserve the speaker's intent and language.
        - Do not add facts that are not present.
        - Return only the final polished text.
        """
        let userInput = """
        Transcript:

        \(transcript)
        """
        return try await generateResponse(
            systemInstruction: instruction,
            userInput: userInput,
            modelId: modelId,
            apiKey: apiKey,
            providerOnly: providerOnly,
            maxOutputTokens: max(512, min(8_000, Int(Double(transcript.split(separator: " ").count) * 2.4)))
        )
    }

    func generatePolishPresetDraft(
        request: String,
        modelId: String,
        apiKey: String,
        providerOnly: [String]? = nil
    ) async throws -> GeneratedPolishPresetDraft {
        let response = try await generateResponse(
            systemInstruction: polishPresetGeneratorInstruction,
            userInput: request,
            modelId: modelId,
            apiKey: apiKey,
            providerOnly: providerOnly,
            maxOutputTokens: 900
        )
        return try Self.decodePolishPresetDraft(from: response.text)
    }

    private func generateResponse(
        systemInstruction: String,
        userInput: String,
        modelId: String,
        apiKey: String,
        providerOnly: [String]?,
        maxOutputTokens: Int
    ) async throws -> OpenAILanguageModelResponse {
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("AiTranscribe", forHTTPHeaderField: "X-OpenRouter-Title")
        request.timeoutInterval = 7200

        let body = OpenRouterChatCompletionRequest(
            model: modelId,
            provider: OpenRouterProviderRouting(only: providerOnly),
            messages: [
                OpenRouterChatMessage(role: "system", content: systemInstruction),
                OpenRouterChatMessage(role: "user", content: userInput)
            ],
            maxTokens: maxOutputTokens
        )
        request.httpBody = try JSONEncoder().encode(body)

        let startedAt = Date()
        let (data, response) = try await session.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw OpenRouterLanguageModelError.invalidResponse
        }
        guard (200..<300).contains(httpResponse.statusCode) else {
            let message = decodeErrorMessage(from: data)
            throw OpenRouterLanguageModelError.apiError(statusCode: httpResponse.statusCode, message: message)
        }

        do {
            let decoded = try JSONDecoder().decode(OpenRouterChatCompletionResponse.self, from: data)
            let text = decoded.choices.first?.message.outputText.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            guard !text.isEmpty else { throw OpenRouterLanguageModelError.emptyOutput }
            return OpenAILanguageModelResponse(
                text: text,
                processingTimeSeconds: Date().timeIntervalSince(startedAt),
                inputTokens: decoded.usage?.promptTokens,
                outputTokens: decoded.usage?.completionTokens
            )
        } catch let error as OpenRouterLanguageModelError {
            throw error
        } catch {
            let rawPreview = String(data: data, encoding: .utf8)
                .map { String($0.prefix(500)) }
            throw OpenRouterLanguageModelError.decodingFailed(rawPreview)
        }
    }

    private func decodeErrorMessage(from data: Data) -> String? {
        try? JSONDecoder().decode(OpenRouterLanguageModelErrorEnvelope.self, from: data).error.message
    }
}

extension OpenRouterLanguageModelClient {
    static func decodePolishPresetDraft(from text: String) throws -> GeneratedPolishPresetDraft {
        let jsonText = text
            .replacingOccurrences(of: "```json", with: "")
            .replacingOccurrences(of: "```", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard let data = jsonText.data(using: .utf8),
              let draft = try? JSONDecoder().decode(GeneratedPolishPresetDraft.self, from: data),
              !draft.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw OpenRouterLanguageModelError.decodingFailed(String(jsonText.prefix(500)))
        }
        return draft
    }
}

let polishPresetGeneratorInstruction = """
Create one AiTranscribe Polish preset from the user's request.

A Polish preset is a reusable instruction that transforms a raw voice transcript after transcription. It must be safe for messy dictated input and must not invent missing facts.

Use this built-in "Prompt Polish" behavior as the quality bar:
- Convert messy spoken notes into a clear prompt for another AI model.
- Preserve the user's goal, constraints, context, examples, deadlines, preferred output format, and edge cases.
- Remove filler, repeated phrases, and self-corrections only when they do not change intent.
- Keep uncertainty explicit instead of pretending missing details are known.
- Return only the final usable result, with no preamble.

Return strict JSON only, with this exact shape:
{
  "display_name": "Short title",
  "symbol_name": "valid SF Symbol name",
  "prompt": "Detailed instruction used to polish transcripts"
}

Rules:
- Keep display_name under 24 characters.
- symbol_name must be a common SF Symbol, for example sparkles, wand.and.stars, text.badge.checkmark, note.text, checklist, doc.text, quote.bubble, brain.head.profile, envelope, lightbulb, target.
- prompt must be written as a direct instruction to the model applying the Polish.
- prompt must require preserving language, user intent, constraints, and important details.
- prompt must require returning only the final polished output.
- Include edge-case handling when relevant: unclear speech, fragments, repeated words, contradictions, tasks, formatting, code, URLs, names, dates, prices, or model IDs.
"""

private struct OpenRouterChatCompletionRequest: Encodable {
    let model: String
    let provider: OpenRouterProviderRouting?
    let messages: [OpenRouterChatMessage]
    let maxTokens: Int

    enum CodingKeys: String, CodingKey {
        case model
        case provider
        case messages
        case maxTokens = "max_tokens"
    }
}

private struct OpenRouterProviderRouting: Encodable {
    let only: [String]

    init?(only: [String]?) {
        let normalized = (only ?? [])
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        guard !normalized.isEmpty else { return nil }
        self.only = normalized
    }
}

private struct OpenRouterChatMessage: Encodable {
    let role: String
    let content: String
}

private struct OpenRouterChatCompletionResponse: Decodable {
    let choices: [OpenRouterChatChoice]
    let usage: OpenRouterChatUsage?
}

private struct OpenRouterChatChoice: Decodable {
    let message: OpenRouterChatResponseMessage
}

private struct OpenRouterChatResponseMessage: Decodable {
    let role: String?
    let content: OpenRouterResponseContent?
    let reasoning: String?

    var outputText: String {
        if let text = content?.text, !text.isEmpty {
            return text
        }
        return reasoning ?? ""
    }
}

private enum OpenRouterResponseContent: Decodable {
    case text(String)
    case parts([OpenRouterResponseContentPart])
    case empty

    var text: String {
        switch self {
        case .text(let value):
            return value
        case .parts(let parts):
            return parts.compactMap(\.text).joined(separator: "\n")
        case .empty:
            return ""
        }
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .empty
            return
        }
        if let text = try? container.decode(String.self) {
            self = .text(text)
            return
        }
        if let parts = try? container.decode([OpenRouterResponseContentPart].self) {
            self = .parts(parts)
            return
        }
        self = .empty
    }
}

private struct OpenRouterResponseContentPart: Decodable {
    let type: String?
    let text: String?
}

private struct OpenRouterChatUsage: Decodable {
    let promptTokens: Int?
    let completionTokens: Int?

    enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
    }
}

private struct OpenRouterLanguageModelErrorEnvelope: Decodable {
    let error: OpenRouterLanguageModelErrorBody
}

private struct OpenRouterLanguageModelErrorBody: Decodable {
    let message: String
}

enum OpenRouterLanguageModelError: LocalizedError {
    case missingAPIKey
    case invalidResponse
    case apiError(statusCode: Int, message: String?)
    case decodingFailed(String?)
    case emptyOutput

    var errorDescription: String? {
        switch self {
        case .missingAPIKey:
            return "OpenRouter API key is missing."
        case .invalidResponse:
            return "OpenRouter returned an invalid LLM response."
        case .apiError(let statusCode, let message):
            if let message, !message.isEmpty {
                return "OpenRouter LLM failed (\(statusCode)): \(message)"
            }
            return "OpenRouter LLM failed with status \(statusCode)."
        case .decodingFailed(let rawPreview):
            if let rawPreview, !rawPreview.isEmpty {
                return "OpenRouter LLM response could not be decoded. Preview: \(rawPreview)"
            }
            return "OpenRouter LLM response could not be decoded."
        case .emptyOutput:
            return "OpenRouter returned an empty LLM response."
        }
    }
}
