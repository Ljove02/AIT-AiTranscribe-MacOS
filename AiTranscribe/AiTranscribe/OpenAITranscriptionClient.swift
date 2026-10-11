import Foundation

struct OpenAITranscriptionClient {
    private let endpoint = URL(string: "https://api.openai.com/v1/audio/transcriptions")!
    private let session: URLSession?

    init(session: URLSession? = nil) {
        self.session = session
    }

    func transcribe(
        audioData: Data,
        modelId: String,
        apiKey: String,
        fileName: String = "recording.wav",
        mimeType: String = "audio/wav"
    ) async throws -> OpenAITranscriptionResponse {
        let boundary = "Boundary-\(UUID().uuidString)"
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.timeoutInterval = 90
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        request.httpBody = multipartBody(
            boundary: boundary,
            fields: [
                "model": modelId,
                "response_format": "json"
            ],
            fileField: "file",
            fileName: fileName,
            mimeType: mimeType,
            fileData: audioData
        )

        let (data, response) = try await upload(request, modelId: modelId)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw OpenAITranscriptionError.invalidResponse
        }

        guard (200..<300).contains(httpResponse.statusCode) else {
            let message = decodeErrorMessage(from: data)
            throw OpenAITranscriptionError.apiError(statusCode: httpResponse.statusCode, message: message)
        }

        do {
            return try JSONDecoder().decode(OpenAITranscriptionResponse.self, from: data)
        } catch {
            throw OpenAITranscriptionError.decodingFailed
        }
    }

    private func multipartBody(
        boundary: String,
        fields: [String: String],
        fileField: String,
        fileName: String,
        mimeType: String,
        fileData: Data
    ) -> Data {
        var body = Data()

        for (name, value) in fields {
            body.appendString("--\(boundary)\r\n")
            body.appendString("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n")
            body.appendString("\(value)\r\n")
        }

        body.appendString("--\(boundary)\r\n")
        body.appendString("Content-Disposition: form-data; name=\"\(fileField)\"; filename=\"\(fileName)\"\r\n")
        body.appendString("Content-Type: \(mimeType)\r\n\r\n")
        body.append(fileData)
        body.appendString("\r\n")
        body.appendString("--\(boundary)--\r\n")

        return body
    }

    private func upload(_ request: URLRequest, modelId: String) async throws -> (Data, URLResponse) {
        if let session {
            return try await session.data(for: request)
        }
        return try await TranscriptionUploadSession.data(for: request, label: "provider=openai model=\(modelId)")
    }

    private func decodeErrorMessage(from data: Data) -> String? {
        try? JSONDecoder().decode(OpenAIErrorEnvelope.self, from: data).error.message
    }
}

struct OpenAITranscriptionResponse: Decodable {
    let text: String
}

private struct OpenAIErrorEnvelope: Decodable {
    let error: OpenAIErrorBody
}

private struct OpenAIErrorBody: Decodable {
    let message: String
}

enum OpenAITranscriptionError: LocalizedError {
    case missingAPIKey
    case unsupportedModel(String)
    case invalidResponse
    case apiError(statusCode: Int, message: String?)
    case decodingFailed

    var errorDescription: String? {
        switch self {
        case .missingAPIKey:
            return "OpenAI API key is missing."
        case .unsupportedModel(let supportedModels):
            return "This is not a Speech-to-Text model. Use one of: \(supportedModels)."
        case .invalidResponse:
            return "OpenAI returned an invalid response."
        case .apiError(let statusCode, let message):
            if let message, !message.isEmpty {
                return "OpenAI request failed (\(statusCode)): \(message)"
            }
            return "OpenAI request failed with status \(statusCode)."
        case .decodingFailed:
            return "OpenAI transcription response could not be decoded."
        }
    }
}

private extension Data {
    mutating func appendString(_ string: String) {
        append(Data(string.utf8))
    }
}
