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
        providerOnly: [String]? = nil,
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
            systemInstruction: instruction,
            userInput: userInput,
            modelId: modelId,
            apiKey: apiKey,
            providerOnly: providerOnly,
            imageDataURL: screenshotDataURL,
            maxOutputTokens: max(1_500, min(16_000, Int(Double(transcript.split(separator: " ").count) * 5.0) + 1_000))
        )
    }

    func polishTextStreamed(
        transcript: String,
        preset: PolishPreset,
        modelId: String,
        apiKey: String,
        providerOnly: [String]? = nil,
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
            systemInstruction: instruction,
            userInput: userInput,
            modelId: modelId,
            apiKey: apiKey,
            providerOnly: providerOnly,
            imageDataURL: screenshotDataURL,
            maxOutputTokens: max(1_500, min(16_000, Int(Double(transcript.split(separator: " ").count) * 5.0) + 1_000)),
            onDelta: onDelta
        )
    }

    func generatePolishPresetDraft(
        request: String,
        modelId: String,
        apiKey: String,
        providerOnly: [String]? = nil
    ) async throws -> GeneratedPolishGenerationResult {
        let response = try await generateResponse(
            systemInstruction: polishPresetGeneratorInstruction,
            userInput: request,
            modelId: modelId,
            apiKey: apiKey,
            providerOnly: providerOnly,
            maxOutputTokens: 2_200
        )
        return try Self.decodePolishPresetGeneration(from: response.text)
    }

    func filterRoutableApplications(
        applications: [PolishRoutableApplication],
        modelId: String,
        apiKey: String,
        providerOnly: [String]? = nil
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
            systemInstruction: openRouterRoutableApplicationFilterInstruction,
            userInput: payloadText,
            modelId: modelId,
            apiKey: apiKey,
            providerOnly: providerOnly,
            maxOutputTokens: 2_000
        )
        let jsonText = response.text
            .replacingOccurrences(of: "```json", with: "")
            .replacingOccurrences(of: "```", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard let data = jsonText.data(using: .utf8),
              let result = try? JSONDecoder().decode(PolishRouteApplicationFilterResult.self, from: data) else {
            throw OpenRouterLanguageModelError.decodingFailed(String(jsonText.prefix(500)))
        }
        return Set(result.hiddenApplicationIds)
    }

    private func generateResponse(
        systemInstruction: String,
        userInput: String,
        modelId: String,
        apiKey: String,
        providerOnly: [String]?,
        imageDataURL: String? = nil,
        maxOutputTokens: Int
    ) async throws -> OpenAILanguageModelResponse {
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("AiTranscribe", forHTTPHeaderField: "X-OpenRouter-Title")
        request.timeoutInterval = 7200

        let userMessage: OpenRouterChatMessage
        if let imageDataURL {
            userMessage = OpenRouterChatMessage(
                role: "user",
                contentParts: [
                    .text(userInput),
                    .imageURL(imageDataURL)
                ]
            )
        } else {
            userMessage = OpenRouterChatMessage(role: "user", content: userInput)
        }

        let body = OpenRouterChatCompletionRequest(
            model: modelId,
            provider: OpenRouterProviderRouting(only: providerOnly),
            messages: [
                OpenRouterChatMessage(role: "system", content: systemInstruction),
                userMessage
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

    private func streamResponse(
        systemInstruction: String,
        userInput: String,
        modelId: String,
        apiKey: String,
        providerOnly: [String]?,
        imageDataURL: String? = nil,
        maxOutputTokens: Int,
        onDelta: @escaping (String) async -> Void
    ) async throws -> OpenAILanguageModelResponse {
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("AiTranscribe", forHTTPHeaderField: "X-OpenRouter-Title")
        request.timeoutInterval = 7200

        let userMessage: OpenRouterChatMessage
        if let imageDataURL {
            userMessage = OpenRouterChatMessage(
                role: "user",
                contentParts: [
                    .text(userInput),
                    .imageURL(imageDataURL)
                ]
            )
        } else {
            userMessage = OpenRouterChatMessage(role: "user", content: userInput)
        }

        let body = OpenRouterChatCompletionRequest(
            model: modelId,
            provider: OpenRouterProviderRouting(only: providerOnly),
            messages: [
                OpenRouterChatMessage(role: "system", content: systemInstruction),
                userMessage
            ],
            maxTokens: maxOutputTokens,
            stream: true
        )
        request.httpBody = try JSONEncoder().encode(body)

        let startedAt = Date()
        let (bytes, response) = try await session.bytes(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw OpenRouterLanguageModelError.invalidResponse
        }
        guard (200..<300).contains(httpResponse.statusCode) else {
            var errorData = Data()
            for try await byte in bytes {
                errorData.append(byte)
            }
            let message = decodeErrorMessage(from: errorData)
            throw OpenRouterLanguageModelError.apiError(statusCode: httpResponse.statusCode, message: message)
        }

        let decoder = JSONDecoder()
        var accumulatedText = ""
        var usage: OpenRouterChatUsage?

        for try await line in bytes.lines {
            guard let payload = Self.ssePayload(from: line) else { continue }
            if payload == "[DONE]" { break }
            guard let data = payload.data(using: .utf8) else { continue }
            guard let chunk = try? decoder.decode(OpenRouterChatStreamChunk.self, from: data) else {
                continue
            }
            if let error = chunk.error {
                throw OpenRouterLanguageModelError.apiError(statusCode: httpResponse.statusCode, message: error.message)
            }
            usage = chunk.usage ?? usage
            let delta = chunk.choices?
                .compactMap { $0.delta?.outputText }
                .joined()
                ?? ""
            if !delta.isEmpty {
                accumulatedText += delta
                await onDelta(delta)
            }
        }

        let text = accumulatedText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { throw OpenRouterLanguageModelError.emptyOutput }
        return OpenAILanguageModelResponse(
            text: text,
            processingTimeSeconds: Date().timeIntervalSince(startedAt),
            inputTokens: usage?.promptTokens,
            outputTokens: usage?.completionTokens
        )
    }

    private func decodeErrorMessage(from data: Data) -> String? {
        try? JSONDecoder().decode(OpenRouterLanguageModelErrorEnvelope.self, from: data).error.message
    }

    private static func ssePayload(from line: String) -> String? {
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("data:") else { return nil }
        return String(trimmed.dropFirst(5)).trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

private let openRouterRoutableApplicationFilterInstruction = """
You are cleaning a macOS application picker for a user-facing routing feature.

The user wants to route dictation output by real apps they intentionally use, such as Codex, Claude, ChatGPT, Terminal, Zed, Obsidian, Notes, Mail, Slack, Discord, Linear, GitHub Desktop, browsers, editors, and other obvious front-facing products.

Hide helper processes, renderers, updaters, launchers, accessibility helpers, graphics/web containers, login items, plugins, extensions, daemons, background services, crash reporters, and anything named like a helper for another app.

Return JSON only:
{
  "hiddenApplicationIds": ["id-to-hide"]
}
"""

extension OpenRouterLanguageModelClient {
    static func decodePolishPresetGeneration(from text: String) throws -> GeneratedPolishGenerationResult {
        let jsonText = text
            .replacingOccurrences(of: "```json", with: "")
            .replacingOccurrences(of: "```", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard let data = jsonText.data(using: .utf8) else {
            throw OpenRouterLanguageModelError.decodingFailed(String(jsonText.prefix(500)))
        }
        if let clarification = try? JSONDecoder().decode(GeneratedPolishClarificationDraft.self, from: data),
           !clarification.questions.isEmpty {
            return .clarification(clarification)
        }
        if let draft = try? JSONDecoder().decode(GeneratedPolishPresetDraft.self, from: data),
           !draft.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return .preset(draft)
        }
        throw OpenRouterLanguageModelError.decodingFailed(String(jsonText.prefix(500)))
    }
}

let polishPresetGeneratorInstruction = """
Create one AiTranscribe Polish preset from the user's request.

You are not only a prompt writer. You are the Polish workflow builder. A Polish preset has:
- a display name
- an SF Symbol icon
- a runtime prompt used by the selected LLM
- input modules
- output modules
- paste behavior

Available input_modules, use raw values exactly:
- voiceTranscript: microphone speech from the user. Usually enabled unless the workflow is clearly context-only.
- desktopAudio: system/desktop audio transcript, separately tagged from the user's voice.
- clipboardText: exact text currently copied by the user.
- screenSnapshot: selected screenshot/image region. Requires a vision-capable LLM.
- activeApp: name of the focused app, used for formatting and destination-specific behavior.

Available output_modules, use raw values exactly:
- copyToClipboard: copy final text output only.
- copyTextAndScreenshot: paste screenshot first, then final text. Use only when screenSnapshot and activeApp are enabled, paste_behavior is pasteIntoApp, and the user wants image+text inserted into a markdown-friendly app.
- floatingWindow: show the final response in the floating response window. Use for explainers, Q&A, diagnostics, reading help, and outputs the user wants to inspect before pasting.

Available paste_behavior values:
- copyOnly: leave the result on the clipboard or in the floating window. Default and safest.
- pasteIntoApp: automatically paste into the focused app after the result is ready. Use only when the user explicitly asks for auto-paste, insertion, or image+text paste into the current app.

Critical mental model:
- The generated prompt is executed later on real runtime context. It is NOT a prompt about creating a Polish.
- Build the final workflow the user asked for. If they ask for advanced/multi-step/context/screenshot/clipboard/floating window, select the modules yourself.
- If they ask for an explainer, Q&A, paper explanation, post explanation, screenshot analysis, or “tell me what this means”, make an informative output workflow, not a prompt-polish workflow.
- If they ask for prompt polish, social comment polish, email reply polish, task extraction, notes, rewrite, grammar, or formatting, make a transformative output workflow.
- If the final output is meant to be pasted into another AI model, the prompt should generate a clean final instruction/prompt.
- If the final output is meant for the user to read, the prompt should generate the final answer/explanation directly.
- If the user explicitly requests a shortcut, include the shortcut intent in the generated prompt only as implementation context; the app UI will handle actual shortcut capture separately.

Clarification rule:
- If the request does not contain enough detail to build a reliable Polish, do not invent workflow details.
- Ask targeted clarification questions instead of returning a weak preset.
- Ask at most 8 questions. Prefer fewer.
- Questions should resolve concrete implementation choices: purpose, intended output, required inputs, whether screenshot/clipboard/desktop audio are needed, floating window vs copy, auto-paste, target app/format, model quality/vision needs, and edge cases.
- Each question may include 2-4 short suggested answers when helpful.
- Use clarification only when the missing detail materially changes the workflow or prompt. Do not ask questions for obvious defaults.

Runtime prompt quality rules:
- Write the prompt as a direct instruction to the runtime LLM applying the Polish.
- Require using all available context sections: voice transcript, desktop/system audio transcript, exact clipboard context, screenshot context, and active app.
- Some context sections may be missing; instruct the runtime LLM to not invent missing context.
- Prioritize the user's voice transcript as intent when present.
- Treat clipboard as exact source context, not as a vague “above/below” reference.
- Treat desktop audio as separate from user voice.
- If screenshot context is enabled, instruct the runtime LLM to inspect the actual visible image and connect it to the user request.
- If activeApp is enabled, instruct the runtime LLM to adapt formatting to the focused app.

Formatting guidance for activeApp-aware prompts:
- For Obsidian, Markdown editors, VS Code, Cursor, Codex, Typora, or markdown-friendly apps: use Markdown with blank lines between headings, bullets, and paragraphs.
- For Apple Notes, Messages, Mail, browser text fields, and plain/rich-text apps: use clean plain text with short paragraphs and readable spacing.
- If active app is unknown, default to Markdown-friendly formatting unless the user asks for plain text.
- Always preserve whitespace/readability. Never output one giant paragraph when the task is an explanation, notes, or analysis.

Use this built-in Prompt Polish behavior as the quality bar for prompt-producing workflows:
- Convert messy spoken notes into a clear prompt for another AI model.
- Preserve the user's goal, constraints, context, examples, deadlines, preferred output format, and edge cases.
- Remove filler, repeated phrases, and self-corrections only when they do not change intent.
- Keep uncertainty explicit instead of pretending missing details are known.
- Return only the final usable prompt/result, with no preamble.

For explainer workflows, the generated prompt should usually require:
- Start with the direct answer or main idea.
- Explain in simple terms first.
- Add deeper technical detail only when helpful.
- If relevant, include “why it matters” or a clear verdict.
- For papers: explain intuition, method, contribution, limitations/risks if visible, and practical meaning.
- For posts/tweets/blogs: explain what it says, whether it seems important, and what the user should take away.

If clarification is needed, return strict JSON only in this exact shape:
{
  "reason": "Short reason questions are needed",
  "questions": [
    {
      "id": "q1",
      "question": "What should this Polish output?",
      "suggested_answers": ["Clean final text", "A prompt for another AI", "A readable explanation", "Structured notes"]
    }
  ]
}

If enough detail is present, return strict JSON only, with this exact shape:
{
  "display_name": "Short title",
  "symbol_name": "valid SF Symbol name",
  "prompt": "Detailed runtime instruction used to transform or answer from context",
  "input_modules": ["voiceTranscript"],
  "output_modules": ["copyToClipboard"],
  "paste_behavior": "copyOnly"
}

Rules:
- Keep display_name under 24 characters.
- symbol_name must be a common SF Symbol, for example sparkles, wand.and.stars, text.badge.checkmark, note.text, checklist, doc.text, quote.bubble, brain.head.profile, envelope, lightbulb, target, questionmark.bubble, photo.on.rectangle.
- input_modules must only use the raw values listed above.
- output_modules must only use the raw values listed above.
- If output_modules includes copyTextAndScreenshot, input_modules must include screenSnapshot and activeApp, and paste_behavior must be pasteIntoApp.
- Do not include both copyToClipboard and copyTextAndScreenshot; choose one.
- For advanced explainers, prefer input_modules ["voiceTranscript", "clipboardText", "screenSnapshot", "activeApp"] and output_modules ["floatingWindow"] unless the user asks to copy the image too.
- For desktop-audio explainers, include desktopAudio.
- For prompt-polish or rewrite workflows, usually use ["voiceTranscript", "clipboardText", "activeApp"] and ["copyToClipboard"].
- paste_behavior should be "copyOnly" unless the user explicitly asks to paste/insert automatically into the focused app. For image+text paste, use pasteIntoApp only when the target is expected to be markdown-friendly.
- The prompt must explicitly say not to output analysis of the instruction, meta-commentary, or a restatement of the instruction.
- The prompt must include enough edge-case handling for unclear speech, fragments, repeated words, contradictions, tasks, formatting, code, URLs, names, dates, prices, model IDs, screenshots, and app-specific formatting when relevant.
"""

private struct OpenRouterChatCompletionRequest: Encodable {
    let model: String
    let provider: OpenRouterProviderRouting?
    let messages: [OpenRouterChatMessage]
    let maxTokens: Int
    let stream: Bool?

    enum CodingKeys: String, CodingKey {
        case model
        case provider
        case messages
        case maxTokens = "max_tokens"
        case stream
    }

    init(
        model: String,
        provider: OpenRouterProviderRouting?,
        messages: [OpenRouterChatMessage],
        maxTokens: Int,
        stream: Bool? = nil
    ) {
        self.model = model
        self.provider = provider
        self.messages = messages
        self.maxTokens = maxTokens
        self.stream = stream
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
    let content: OpenRouterMessageContent

    init(role: String, content: String) {
        self.role = role
        self.content = .text(content)
    }

    init(role: String, contentParts: [OpenRouterContentPart]) {
        self.role = role
        self.content = .parts(contentParts)
    }
}

private enum OpenRouterMessageContent: Encodable {
    case text(String)
    case parts([OpenRouterContentPart])

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

private struct OpenRouterContentPart: Encodable {
    let type: String
    let text: String?
    let imageURL: OpenRouterImageURL?

    enum CodingKeys: String, CodingKey {
        case type
        case text
        case imageURL = "image_url"
    }

    static func text(_ text: String) -> OpenRouterContentPart {
        OpenRouterContentPart(type: "text", text: text, imageURL: nil)
    }

    static func imageURL(_ url: String) -> OpenRouterContentPart {
        OpenRouterContentPart(type: "image_url", text: nil, imageURL: OpenRouterImageURL(url: url))
    }
}

private struct OpenRouterImageURL: Encodable {
    let url: String
}

private struct OpenRouterChatCompletionResponse: Decodable {
    let choices: [OpenRouterChatChoice]
    let usage: OpenRouterChatUsage?
}

private struct OpenRouterChatStreamChunk: Decodable {
    let choices: [OpenRouterChatStreamChoice]?
    let usage: OpenRouterChatUsage?
    let error: OpenRouterLanguageModelErrorBody?
}

private struct OpenRouterChatStreamChoice: Decodable {
    let delta: OpenRouterChatStreamDelta?
}

private struct OpenRouterChatStreamDelta: Decodable {
    let content: OpenRouterResponseContent?
    let reasoning: String?

    var outputText: String {
        if let text = content?.text, !text.isEmpty {
            return text
        }
        return reasoning ?? ""
    }
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
