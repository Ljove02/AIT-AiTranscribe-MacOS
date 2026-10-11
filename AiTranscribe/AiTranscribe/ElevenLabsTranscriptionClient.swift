import Foundation

struct ElevenLabsTranscriptionClient {
    private let endpoint = URL(string: "https://api.elevenlabs.io/v1/speech-to-text")!
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
    ) async throws -> ElevenLabsTranscriptionResponse {
        let boundary = "Boundary-\(UUID().uuidString)"
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.timeoutInterval = 90
        request.setValue(apiKey, forHTTPHeaderField: "xi-api-key")
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        request.httpBody = multipartBody(
            boundary: boundary,
            fields: [
                "model_id": modelId,
                "tag_audio_events": "false",
                "diarize": "false",
                "timestamps_granularity": "none"
            ],
            fileField: "file",
            fileName: fileName,
            mimeType: mimeType,
            fileData: audioData
        )

        let (data, response) = try await upload(request, modelId: modelId)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw ElevenLabsTranscriptionError.invalidResponse
        }

        guard (200..<300).contains(httpResponse.statusCode) else {
            let message = decodeErrorMessage(from: data)
            throw ElevenLabsTranscriptionError.apiError(statusCode: httpResponse.statusCode, message: message)
        }

        do {
            return try JSONDecoder().decode(ElevenLabsTranscriptionResponse.self, from: data)
        } catch {
            throw ElevenLabsTranscriptionError.decodingFailed
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
        return try await TranscriptionUploadSession.data(for: request, label: "provider=elevenlabs model=\(modelId)")
    }

    private func decodeErrorMessage(from data: Data) -> String? {
        guard !data.isEmpty else { return nil }

        if let json = try? JSONSerialization.jsonObject(with: data),
           let object = json as? [String: Any] {
            if let message = object["message"] as? String {
                return message
            }
            if let detail = object["detail"] as? String {
                return detail
            }
            if let detail = object["detail"] {
                return String(describing: detail)
            }
        }

        return String(data: data, encoding: .utf8)
    }
}

struct ElevenLabsTranscriptionResponse: Decodable {
    let languageCode: String?
    let languageProbability: Double?
    let text: String

    enum CodingKeys: String, CodingKey {
        case languageCode = "language_code"
        case languageProbability = "language_probability"
        case text
    }
}

enum ElevenLabsTranscriptionError: LocalizedError {
    case missingAPIKey
    case unsupportedModel(String)
    case invalidResponse
    case apiError(statusCode: Int, message: String?)
    case decodingFailed

    var errorDescription: String? {
        switch self {
        case .missingAPIKey:
            return "ElevenLabs API key is missing."
        case .unsupportedModel(let supportedModels):
            return "This is not an ElevenLabs batch Speech-to-Text model. \(supportedModels)"
        case .invalidResponse:
            return "ElevenLabs returned an invalid response."
        case .apiError(let statusCode, let message):
            if let message, !message.isEmpty {
                return "ElevenLabs request failed (\(statusCode)): \(message)"
            }
            return "ElevenLabs request failed with status \(statusCode)."
        case .decodingFailed:
            return "ElevenLabs transcription response could not be decoded."
        }
    }
}

private extension Data {
    mutating func appendString(_ string: String) {
        append(Data(string.utf8))
    }
}
