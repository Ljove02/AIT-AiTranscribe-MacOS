import Foundation

struct OpenAILanguageModelClient {
    private let endpoint = URL(string: "https://api.openai.com/v1/responses")!
    private let session: URLSession

    init(session: URLSession = .shared) {
        self.session = session
    }

    func generateSummary(
        transcript: String,
        preset: SummaryGenerationPresetRequest,
        modelId: String,
        apiKey: String
    ) async throws -> OpenAILanguageModelResponse {
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 7200

        let instruction = """
        \(preset.systemPrompt)

        Target length: about \(preset.targetWords) words.
        Return only the finished summary. Do not include preambles.
        """
        let userInput = """
        Summarize this transcript:

        \(transcript)
        """
        let body = OpenAIResponsesRequest(
            model: modelId,
            input: [
                OpenAIResponsesInput(role: "developer", content: instruction),
                OpenAIResponsesInput(role: "user", content: userInput)
            ],
            maxOutputTokens: preset.maxOutputTokens
        )
        request.httpBody = try JSONEncoder().encode(body)

        let startedAt = Date()
        let (data, response) = try await session.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw OpenAILanguageModelError.invalidResponse
        }
        guard (200..<300).contains(httpResponse.statusCode) else {
            let message = decodeErrorMessage(from: data)
            throw OpenAILanguageModelError.apiError(statusCode: httpResponse.statusCode, message: message)
        }

        do {
            let decoded = try JSONDecoder().decode(OpenAIResponsesResponse.self, from: data)
            let text = decoded.outputText.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { throw OpenAILanguageModelError.emptyOutput }
            return OpenAILanguageModelResponse(
                text: text,
                processingTimeSeconds: Date().timeIntervalSince(startedAt),
                inputTokens: decoded.usage?.inputTokens,
                outputTokens: decoded.usage?.outputTokens
            )
        } catch let error as OpenAILanguageModelError {
            throw error
        } catch {
            throw OpenAILanguageModelError.decodingFailed
        }
    }

    private func decodeErrorMessage(from data: Data) -> String? {
        try? JSONDecoder().decode(OpenAITextErrorEnvelope.self, from: data).error.message
    }
}

struct OpenAILanguageModelResponse {
    let text: String
    let processingTimeSeconds: Double
    let inputTokens: Int?
    let outputTokens: Int?
}

private struct OpenAIResponsesRequest: Encodable {
    let model: String
    let input: [OpenAIResponsesInput]
    let maxOutputTokens: Int

    enum CodingKeys: String, CodingKey {
        case model
        case input
        case maxOutputTokens = "max_output_tokens"
    }
}

private struct OpenAIResponsesInput: Encodable {
    let role: String
    let content: String
}

private struct OpenAIResponsesResponse: Decodable {
    let output: [OpenAIResponsesOutput]?
    let usage: OpenAIResponsesUsage?

    var outputText: String {
        output?
            .flatMap { $0.content ?? [] }
            .compactMap(\.text)
            .joined(separator: "\n")
            ?? ""
    }
}

private struct OpenAIResponsesUsage: Decodable {
    let inputTokens: Int?
    let outputTokens: Int?
    let totalTokens: Int?

    enum CodingKeys: String, CodingKey {
        case inputTokens = "input_tokens"
        case outputTokens = "output_tokens"
        case totalTokens = "total_tokens"
    }
}

private struct OpenAIResponsesOutput: Decodable {
    let content: [OpenAIResponsesContent]?
}

private struct OpenAIResponsesContent: Decodable {
    let text: String?
    let type: String?
}

private struct OpenAITextErrorEnvelope: Decodable {
    let error: OpenAITextErrorBody
}

private struct OpenAITextErrorBody: Decodable {
    let message: String
}

enum OpenAILanguageModelError: LocalizedError {
    case missingAPIKey
    case invalidResponse
    case apiError(statusCode: Int, message: String?)
    case decodingFailed
    case emptyOutput

    var errorDescription: String? {
        switch self {
        case .missingAPIKey:
            return "OpenAI API key is missing."
        case .invalidResponse:
            return "OpenAI returned an invalid response."
        case .apiError(let statusCode, let message):
            if let message, !message.isEmpty {
                return "OpenAI summary failed (\(statusCode)): \(message)"
            }
            return "OpenAI summary failed with status \(statusCode)."
        case .decodingFailed:
            return "OpenAI summary response could not be decoded."
        case .emptyOutput:
        return "OpenAI returned an empty summary."
        }
    }
}

extension OpenAILanguageModelClient {
    func testModel(
        modelId: String,
        apiKey: String
    ) async throws -> OpenAILanguageModelResponse {
        try await generateResponse(
            developerInstruction: "You are running a short health check. Reply with exactly: AiTranscribe OK",
            userInput: "Confirm this model can respond.",
            modelId: modelId,
            apiKey: apiKey,
            maxOutputTokens: 32
        )
    }

    func polishText(
        transcript: String,
        preset: PolishPreset,
        modelId: String,
        apiKey: String
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
            developerInstruction: instruction,
            userInput: userInput,
            modelId: modelId,
            apiKey: apiKey,
            maxOutputTokens: max(512, min(8_000, Int(Double(transcript.split(separator: " ").count) * 2.4)))
        )
    }

    func generatePolishPresetDraft(
        request: String,
        modelId: String,
        apiKey: String
    ) async throws -> GeneratedPolishPresetDraft {
        let response = try await generateResponse(
            developerInstruction: polishPresetGeneratorInstruction,
            userInput: request,
            modelId: modelId,
            apiKey: apiKey,
            maxOutputTokens: 900
        )
        let jsonText = response.text
            .replacingOccurrences(of: "```json", with: "")
            .replacingOccurrences(of: "```", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard let data = jsonText.data(using: .utf8),
              let draft = try? JSONDecoder().decode(GeneratedPolishPresetDraft.self, from: data),
              !draft.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw OpenAILanguageModelError.decodingFailed
        }
        return draft
    }

    private func generateResponse(
        developerInstruction: String,
        userInput: String,
        modelId: String,
        apiKey: String,
        maxOutputTokens: Int
    ) async throws -> OpenAILanguageModelResponse {
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 7200

        let body = OpenAIResponsesRequest(
            model: modelId,
            input: [
                OpenAIResponsesInput(role: "developer", content: developerInstruction),
                OpenAIResponsesInput(role: "user", content: userInput)
            ],
            maxOutputTokens: maxOutputTokens
        )
        request.httpBody = try JSONEncoder().encode(body)

        let startedAt = Date()
        let (data, response) = try await session.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw OpenAILanguageModelError.invalidResponse
        }
        guard (200..<300).contains(httpResponse.statusCode) else {
            let message = decodeErrorMessage(from: data)
            throw OpenAILanguageModelError.apiError(statusCode: httpResponse.statusCode, message: message)
        }

        do {
            let decoded = try JSONDecoder().decode(OpenAIResponsesResponse.self, from: data)
            let text = decoded.outputText.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { throw OpenAILanguageModelError.emptyOutput }
            return OpenAILanguageModelResponse(
                text: text,
                processingTimeSeconds: Date().timeIntervalSince(startedAt),
                inputTokens: decoded.usage?.inputTokens,
                outputTokens: decoded.usage?.outputTokens
            )
        } catch let error as OpenAILanguageModelError {
            throw error
        } catch {
            throw OpenAILanguageModelError.decodingFailed
        }
    }
}
