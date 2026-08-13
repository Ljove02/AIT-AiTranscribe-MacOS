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
    let stream: Bool?

    enum CodingKeys: String, CodingKey {
        case model
        case input
        case maxOutputTokens = "max_output_tokens"
        case stream
    }

    init(
        model: String,
        input: [OpenAIResponsesInput],
        maxOutputTokens: Int,
        stream: Bool? = nil
    ) {
        self.model = model
        self.input = input
        self.maxOutputTokens = maxOutputTokens
        self.stream = stream
    }
}

private struct OpenAIResponsesInput: Encodable {
    let role: String
    let content: OpenAIResponsesInputContent

    init(role: String, content: String) {
        self.role = role
        self.content = .text(content)
    }

    init(role: String, contentParts: [OpenAIResponsesContentPart]) {
        self.role = role
        self.content = .parts(contentParts)
    }
}

private enum OpenAIResponsesInputContent: Encodable {
    case text(String)
    case parts([OpenAIResponsesContentPart])

    func encode(to encoder: Encoder) throws {
        switch self {
        case .text(let text):
            var container = encoder.singleValueContainer()
            try container.encode(text)
        case .parts(let parts):
            var container = encoder.singleValueContainer()
            try container.encode(parts)
        }
    }
}

private struct OpenAIResponsesContentPart: Encodable {
    let type: String
    let text: String?
    let imageURL: String?

    enum CodingKeys: String, CodingKey {
        case type
        case text
        case imageURL = "image_url"
    }

    static func inputText(_ text: String) -> OpenAIResponsesContentPart {
        OpenAIResponsesContentPart(type: "input_text", text: text, imageURL: nil)
    }

    static func inputImage(_ imageURL: String) -> OpenAIResponsesContentPart {
        OpenAIResponsesContentPart(type: "input_image", text: nil, imageURL: imageURL)
    }
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

private struct OpenAIResponsesStreamEvent: Decodable {
    let type: String?
    let delta: String?
    let response: OpenAIResponsesResponse?
    let error: OpenAITextErrorBody?
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
        apiKey: String,
        screenshotDataURL: String? = nil
    ) async throws -> OpenAILanguageModelResponse {
        let instruction = """
        You are polishing a voice transcript.

        Polish instruction:
        \(preset.trimmedPrompt)

        Rules:
        - Preserve the speaker's intent and language.
        - If clipboard text or active app context is provided, use it only as supporting context for the voice instruction.
        - If a screenshot image is attached, use it only as supporting visual context for the voice instruction.
        - If the instruction asks to include, rewrite, reply to, comment on, or preserve clipboard context, inline the relevant context exactly in the final answer.
        - Never replace clipboard content with phrases like "as shown above" or "from the clipboard" when exact content is required.
        - Do not add facts that are not present.
        - For formulas, equations, mathematical symbols, derivations, and matrices, always use LaTeX.
        - Use $...$ for inline math and $$...$$ for display equations.
        - Use LaTeX matrix environments such as \\begin{bmatrix} ... \\end{bmatrix} or \\begin{pmatrix} ... \\end{pmatrix} inside display math for matrices.
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
            imageDataURL: screenshotDataURL,
            maxOutputTokens: max(1_500, min(16_000, Int(Double(transcript.split(separator: " ").count) * 5.0) + 1_000))
        )
    }

    func polishTextStreamed(
        transcript: String,
        preset: PolishPreset,
        modelId: String,
        apiKey: String,
        screenshotDataURL: String? = nil,
        onDelta: @escaping (String) async -> Void
    ) async throws -> OpenAILanguageModelResponse {
        let instruction = """
        You are polishing a voice transcript.

        Polish instruction:
        \(preset.trimmedPrompt)

        Rules:
        - Preserve the speaker's intent and language.
        - If clipboard text or active app context is provided, use it only as supporting context for the voice instruction.
        - If a screenshot image is attached, use it only as supporting visual context for the voice instruction.
        - If the instruction asks to include, rewrite, reply to, comment on, or preserve clipboard context, inline the relevant context exactly in the final answer.
        - Never replace clipboard content with phrases like "as shown above" or "from the clipboard" when exact content is required.
        - Do not add facts that are not present.
        - For formulas, equations, mathematical symbols, derivations, and matrices, always use LaTeX.
        - Use $...$ for inline math and $$...$$ for display equations.
        - Use LaTeX matrix environments such as \\begin{bmatrix} ... \\end{bmatrix} or \\begin{pmatrix} ... \\end{pmatrix} inside display math for matrices.
        - Return only the final polished text.
        """
        let userInput = """
        Transcript:

        \(transcript)
        """
        return try await streamResponse(
            developerInstruction: instruction,
            userInput: userInput,
            modelId: modelId,
            apiKey: apiKey,
            imageDataURL: screenshotDataURL,
            maxOutputTokens: max(1_500, min(16_000, Int(Double(transcript.split(separator: " ").count) * 5.0) + 1_000)),
            onDelta: onDelta
        )
    }

    func generatePolishPresetDraft(
        request: String,
        modelId: String,
        apiKey: String
    ) async throws -> GeneratedPolishGenerationResult {
        let response = try await generateResponse(
            developerInstruction: polishPresetGeneratorInstruction,
            userInput: request,
            modelId: modelId,
            apiKey: apiKey,
            maxOutputTokens: 2_200
        )
        let jsonText = response.text
            .replacingOccurrences(of: "```json", with: "")
            .replacingOccurrences(of: "```", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard let data = jsonText.data(using: .utf8) else {
            throw OpenAILanguageModelError.decodingFailed
        }
        if let clarification = try? JSONDecoder().decode(GeneratedPolishClarificationDraft.self, from: data),
           !clarification.questions.isEmpty {
            return .clarification(clarification)
        }
        if let draft = try? JSONDecoder().decode(GeneratedPolishPresetDraft.self, from: data),
           !draft.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return .preset(draft)
        }
        throw OpenAILanguageModelError.decodingFailed
    }

    func filterRoutableApplications(
        applications: [PolishRoutableApplication],
        modelId: String,
        apiKey: String
    ) async throws -> Set<String> {
        let payload = applications.map { app in
            [
                "id": app.id,
                "name": app.displayName,
                "bundleIdentifier": app.bundleIdentifier ?? "",
                "isRunning": app.isRunning ? "true" : "false"
            ]
        }
        let payloadData = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        let payloadText = String(data: payloadData, encoding: .utf8) ?? "[]"
        let response = try await generateResponse(
            developerInstruction: routableApplicationFilterInstruction,
            userInput: payloadText,
            modelId: modelId,
            apiKey: apiKey,
            maxOutputTokens: 2_000
        )
        let jsonText = response.text
            .replacingOccurrences(of: "```json", with: "")
            .replacingOccurrences(of: "```", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard let data = jsonText.data(using: .utf8),
              let result = try? JSONDecoder().decode(PolishRouteApplicationFilterResult.self, from: data) else {
            throw OpenAILanguageModelError.decodingFailed
        }
        return Set(result.hiddenApplicationIds)
    }

    private func generateResponse(
        developerInstruction: String,
        userInput: String,
        modelId: String,
        apiKey: String,
        imageDataURL: String? = nil,
        maxOutputTokens: Int
    ) async throws -> OpenAILanguageModelResponse {
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 7200

        let userContent: OpenAIResponsesInput
        if let imageDataURL {
            userContent = OpenAIResponsesInput(
                role: "user",
                contentParts: [
                    .inputText(userInput),
                    .inputImage(imageDataURL)
                ]
            )
        } else {
            userContent = OpenAIResponsesInput(role: "user", content: userInput)
        }

        let body = OpenAIResponsesRequest(
            model: modelId,
            input: [
                OpenAIResponsesInput(role: "developer", content: developerInstruction),
                userContent
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

    private func streamResponse(
        developerInstruction: String,
        userInput: String,
        modelId: String,
        apiKey: String,
        imageDataURL: String? = nil,
        maxOutputTokens: Int,
        onDelta: @escaping (String) async -> Void
    ) async throws -> OpenAILanguageModelResponse {
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 7200

        let userContent: OpenAIResponsesInput
        if let imageDataURL {
            userContent = OpenAIResponsesInput(
                role: "user",
                contentParts: [
                    .inputText(userInput),
                    .inputImage(imageDataURL)
                ]
            )
        } else {
            userContent = OpenAIResponsesInput(role: "user", content: userInput)
        }

        let body = OpenAIResponsesRequest(
            model: modelId,
            input: [
                OpenAIResponsesInput(role: "developer", content: developerInstruction),
                userContent
            ],
            maxOutputTokens: maxOutputTokens,
            stream: true
        )
        request.httpBody = try JSONEncoder().encode(body)

        let startedAt = Date()
        let (bytes, response) = try await session.bytes(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw OpenAILanguageModelError.invalidResponse
        }
        guard (200..<300).contains(httpResponse.statusCode) else {
            var errorData = Data()
            for try await byte in bytes {
                errorData.append(byte)
            }
            let message = decodeErrorMessage(from: errorData)
            throw OpenAILanguageModelError.apiError(statusCode: httpResponse.statusCode, message: message)
        }

        let decoder = JSONDecoder()
        var accumulatedText = ""
        var completedText: String?
        var usage: OpenAIResponsesUsage?

        for try await line in bytes.lines {
            guard let payload = Self.ssePayload(from: line) else { continue }
            if payload == "[DONE]" { break }
            guard let data = payload.data(using: .utf8) else { continue }
            guard let event = try? decoder.decode(OpenAIResponsesStreamEvent.self, from: data) else {
                continue
            }
            if let error = event.error {
                throw OpenAILanguageModelError.apiError(statusCode: httpResponse.statusCode, message: error.message)
            }
            if event.type == "response.output_text.delta", let delta = event.delta, !delta.isEmpty {
                accumulatedText += delta
                await onDelta(delta)
            }
            if let response = event.response {
                let responseText = response.outputText.trimmingCharacters(in: .whitespacesAndNewlines)
                if !responseText.isEmpty {
                    completedText = responseText
                }
                usage = response.usage ?? usage
            }
        }

        let text = (completedText ?? accumulatedText).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { throw OpenAILanguageModelError.emptyOutput }
        return OpenAILanguageModelResponse(
            text: text,
            processingTimeSeconds: Date().timeIntervalSince(startedAt),
            inputTokens: usage?.inputTokens,
            outputTokens: usage?.outputTokens
        )
    }

    private static func ssePayload(from line: String) -> String? {
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("data:") else { return nil }
        return String(trimmed.dropFirst(5)).trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

private let routableApplicationFilterInstruction = """
You are cleaning a macOS application picker for a user-facing routing feature.

The user wants to route dictation output by real apps they intentionally use, such as Codex, Claude, ChatGPT, Terminal, Zed, Obsidian, Notes, Mail, Slack, Discord, Linear, GitHub Desktop, browsers, editors, and other obvious front-facing products.

Hide helper processes, renderers, updaters, launchers, accessibility helpers, graphics/web containers, login items, plugins, extensions, daemons, background services, crash reporters, and anything named like a helper for another app.

Return JSON only:
{
  "hiddenApplicationIds": ["id-to-hide"]
}
"""
