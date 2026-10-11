import Foundation

struct OpenRouterTranscriptionClient {
    private let endpoint = URL(string: "https://openrouter.ai/api/v1/audio/transcriptions")!
    private let session: URLSession?

    init(session: URLSession? = nil) {
        self.session = session
    }

    func transcribe(
        audioData: Data,
        modelId: String,
        apiKey: String,
        format: String = "wav"
    ) async throws -> OpenRouterTranscriptionResponse {
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.timeoutInterval = 90
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("AiTranscribe", forHTTPHeaderField: "X-OpenRouter-Title")

        let body = OpenRouterTranscriptionRequest(
            model: modelId,
            inputAudio: OpenRouterInputAudio(
                data: audioData.base64EncodedString(),
                format: format
            )
        )
        request.httpBody = try JSONEncoder().encode(body)

        let (data, response) = try await upload(request, modelId: modelId)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw OpenRouterTranscriptionError.invalidResponse
        }

        guard (200..<300).contains(httpResponse.statusCode) else {
            let message = decodeErrorMessage(from: data)
            if httpResponse.statusCode == 400,
               let message,
               message.localizedCaseInsensitiveContains("does not exist") {
                throw OpenRouterTranscriptionError.modelNotAvailableForTranscription(modelId)
            }
            throw OpenRouterTranscriptionError.apiError(statusCode: httpResponse.statusCode, message: message)
        }

        do {
            let decoded = try JSONDecoder().decode(OpenRouterTranscriptionResponse.self, from: data)
            return decoded
        } catch let error as OpenRouterTranscriptionError {
            throw error
        } catch {
            throw OpenRouterTranscriptionError.decodingFailed
        }
    }

    private func upload(_ request: URLRequest, modelId: String) async throws -> (Data, URLResponse) {
        if let session {
            return try await session.data(for: request)
        }
        return try await TranscriptionUploadSession.data(for: request, label: "provider=openrouter model=\(modelId)")
    }

    private func decodeErrorMessage(from data: Data) -> String? {
        try? JSONDecoder().decode(OpenRouterErrorEnvelope.self, from: data).error.message
    }
}

private struct OpenRouterTranscriptionRequest: Encodable {
    let model: String
    let inputAudio: OpenRouterInputAudio

    enum CodingKeys: String, CodingKey {
        case model
        case inputAudio = "input_audio"
    }
}

private struct OpenRouterInputAudio: Encodable {
    let data: String
    let format: String
}

struct OpenRouterTranscriptionResponse: Decodable {
    let text: String
    let usage: OpenRouterTranscriptionUsage?
}

struct OpenRouterTranscriptionUsage: Decodable {
    let seconds: Double?
    let totalTokens: Int?
    let inputTokens: Int?
    let outputTokens: Int?
    let cost: Double?

    enum CodingKeys: String, CodingKey {
        case seconds
        case totalTokens = "total_tokens"
        case inputTokens = "input_tokens"
        case outputTokens = "output_tokens"
        case cost
    }
}

private struct OpenRouterErrorEnvelope: Decodable {
    let error: OpenRouterErrorBody
}

private struct OpenRouterErrorBody: Decodable {
    let message: String
}

enum OpenRouterTranscriptionError: LocalizedError {
    case missingAPIKey
    case unsupportedModel(String)
    case modelNotAvailableForTranscription(String)
    case invalidResponse
    case apiError(statusCode: Int, message: String?)
    case decodingFailed

    var errorDescription: String? {
        switch self {
        case .missingAPIKey:
            return "OpenRouter API key is missing."
        case .unsupportedModel(let supportedModels):
            return "This is not an OpenRouter Speech-to-Text model. \(supportedModels)"
        case .modelNotAvailableForTranscription(let modelId):
            return "\(modelId) is an OpenRouter LLM/chat model or is not enabled for Speech-to-Text. Add it under LLM, or use an OpenRouter transcription model such as openai/whisper-1."
        case .invalidResponse:
            return "OpenRouter returned an invalid response."
        case .apiError(let statusCode, let message):
            if let message, !message.isEmpty {
                return "OpenRouter request failed (\(statusCode)): \(message)"
            }
            return "OpenRouter request failed with status \(statusCode)."
        case .decodingFailed:
            return "OpenRouter transcription response could not be decoded."
        }
    }
}
