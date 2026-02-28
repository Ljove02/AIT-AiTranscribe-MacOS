/*
 APIClient.swift
 ===============

 This file handles HTTP communication with our Python backend.

 WHAT IS HTTP?
 -------------
 HTTP (HyperText Transfer Protocol) is how computers talk over the internet.

 Key concepts:
 - URL: The address (like http://127.0.0.1:8765/status)
 - Methods: GET (read), POST (create/action), PUT (update), DELETE (remove)
 - Headers: Metadata about the request
 - Body: The actual data being sent

 WHAT IS async/await?
 --------------------
 Network requests take time. We don't want to freeze the app while waiting.

 Old way (callbacks):
   fetchData { result in
       // handle result
   }

 New way (async/await):
   let result = await fetchData()  // Waits without blocking

 async = "this function might pause"
 await = "pause here until done"

 WHAT IS Codable?
 ----------------
 Codable is Swift's way to convert between:
 - Swift objects <-> JSON

 Example:
   struct Person: Codable {
       let name: String
       let age: Int
   }

   // JSON: {"name": "John", "age": 30}
   // Swift: Person(name: "John", age: 30)

 The compiler auto-generates conversion code!
 */

import Foundation

/// Errors that can occur when talking to the API
enum APIError: Error, LocalizedError {
    case serverNotRunning
    case invalidResponse
    case decodingError(Error)
    case serverError(String)

    var errorDescription: String? {
        switch self {
        case .serverNotRunning:
            return "Backend server is not running"
        case .invalidResponse:
            return "Invalid response from server"
        case .decodingError(let error):
            return "Failed to decode response: \(error.localizedDescription)"
        case .serverError(let message):
            return message
        }
    }
}

/// Client for communicating with the AiTranscribe Python backend
class APIClient {

    // =========================================================================
    // CONFIGURATION
    // =========================================================================

    /// Base URL for the API
    private let baseURL: URL

    /// Shared URLSession for making requests
    private let session: URLSession

    /// JSON decoder configured for our API
    private let decoder: JSONDecoder

    /// JSON encoder for sending data
    private let encoder: JSONEncoder

    // =========================================================================
    // INITIALIZATION
    // =========================================================================

    init(host: String = "127.0.0.1", port: Int = 8765) {
        /*
         URL(string:) can fail, so it returns Optional.
         We use ! here because we know this URL is valid.
         In production code, you'd handle this more gracefully.
         */
        self.baseURL = URL(string: "http://\(host):\(port)")!

        /*
         URLSession is Apple's networking API.
         .shared is a pre-configured singleton.
         */
        self.session = URLSession.shared

        self.decoder = JSONDecoder()
        self.encoder = JSONEncoder()
    }


    // =========================================================================
    // HELPER METHODS
    // =========================================================================

    /// Make a GET request and decode the response
    private func get<T: Decodable>(_ endpoint: String) async throws -> T {
        let url = baseURL.appendingPathComponent(endpoint)

        /*
         session.data(from:) makes the HTTP request.
         It returns a tuple: (data, response)
         - data = the response body (JSON)
         - response = metadata (status code, headers, etc.)

         'try await' because:
         - 'try' = this can throw an error (network failed, etc.)
         - 'await' = this is async, wait for it
         */
        let (data, response) = try await session.data(from: url)

        // Check for HTTP errors
        try checkResponse(response)

        // Decode JSON to Swift object
        do {
            return try decoder.decode(T.self, from: data)
        } catch {
            throw APIError.decodingError(error)
        }
    }

    /// Make a POST request and decode the response
    private func post<T: Decodable>(_ endpoint: String, body: Data? = nil) async throws -> T {
        let url = baseURL.appendingPathComponent(endpoint)

        /*
         URLRequest lets us customize the request.
         For POST, we need to set:
         - HTTP method
         - Content-Type header
         - Body (if any)
         */
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = body

        let (data, response) = try await session.data(for: request)

        try checkResponse(response)

        do {
            return try decoder.decode(T.self, from: data)
        } catch {
            throw APIError.decodingError(error)
        }
    }

    /// Check if the HTTP response indicates success
    private func checkResponse(_ response: URLResponse) throws {
        /*
         URLResponse is generic. HTTPURLResponse has HTTP-specific info.
         We cast to access the statusCode.
         */
        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.invalidResponse
        }

        /*
         HTTP status codes:
         - 200-299 = Success
         - 400-499 = Client error (your fault)
         - 500-599 = Server error (server's fault)
         */
        guard (200...299).contains(httpResponse.statusCode) else {
            throw APIError.serverError("Server returned status \(httpResponse.statusCode)")
        }
    }


    // =========================================================================
    // API ENDPOINTS
    // =========================================================================

    // MARK: - Health & Status

    /// Check if server is running
    func healthCheck() async throws -> Bool {
        let _: HealthResponse = try await get("/health")
        return true
    }

    /// Get server status
    func getStatus() async throws -> StatusResponse {
        return try await get("/status")
    }


    // MARK: - Model Management

    /// List all available ASR models
    func listModels() async throws -> [ModelInfoResponse] {
        return try await get("/models")
    }

    /// Get info about a specific model
    func getModelInfo(modelId: String) async throws -> ModelInfoResponse {
        return try await get("/models/\(modelId)")
    }

    /// Load a specific ASR model
    func loadModel(modelId: String = "parakeet-v2") async throws -> MessageResponse {
        let request = LoadModelRequest(modelId: modelId)
        let body = try encoder.encode(request)
        return try await post("/load", body: body)
    }

    /// Unload the ASR model
    func unloadModel() async throws -> MessageResponse {
        return try await post("/unload")
    }


    // MARK: - Storage & Downloads

    /// Get storage information
    func getStorageInfo() async throws -> StorageInfoResponse {
        return try await get("/storage")
    }

    /// Download a model with progress updates via SSE
    func downloadModel(modelId: String, onProgress: @escaping (DownloadProgressEvent) -> Void) async throws {
        let url = baseURL.appendingPathComponent("/models/\(modelId)/download")

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        request.timeoutInterval = 3600  // 1 hour max for large downloads

        let (asyncBytes, response) = try await session.bytes(for: request)

        try checkResponse(response)

        // Process SSE stream
        for try await line in asyncBytes.lines {
            if line.hasPrefix("data: ") {
                let jsonString = String(line.dropFirst(6))
                if let data = jsonString.data(using: .utf8),
                   let event = try? decoder.decode(DownloadProgressEvent.self, from: data) {
                    await MainActor.run {
                        onProgress(event)
                    }
                }
            }
        }
    }

    /// Delete a downloaded model
    func deleteModel(modelId: String) async throws -> MessageResponse {
        let url = baseURL.appendingPathComponent("/models/\(modelId)")

        var request = URLRequest(url: url)
        request.httpMethod = "DELETE"

        let (data, response) = try await session.data(for: request)

        try checkResponse(response)

        do {
            return try decoder.decode(MessageResponse.self, from: data)
        } catch {
            throw APIError.decodingError(error)
        }
    }


    // MARK: - Audio Devices

    /// Get available audio input devices
    func getDevices() async throws -> [AudioDevice] {
        return try await get("/devices")
    }


    // MARK: - Recording

    /// Start recording
    func startRecording() async throws -> MessageResponse {
        return try await post("/recording/start")
    }

    /// Stop recording and get transcription
    func stopRecording() async throws -> TranscriptionResponse {
        return try await post("/recording/stop")
    }

    /// Cancel recording
    func cancelRecording() async throws -> MessageResponse {
        return try await post("/recording/cancel")
    }

    /// Get current recording status
    func getRecordingStatus() async throws -> RecordingStatusResponse {
        return try await get("/recording/status")
    }

    /// Transcribe audio data (WAV format)
    /// Used when Swift records audio directly and sends to backend
    func transcribeAudioData(_ wavData: Data) async throws -> TranscriptionResponse {
        let url = baseURL.appendingPathComponent("/transcribe")

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = 600  // 10 minutes safety net for long transcriptions

        // Create multipart form data for file upload
        let boundary = UUID().uuidString
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        var body = Data()

        // Add audio file - the /transcribe endpoint expects a file field named "file"
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: audio/wav\r\n\r\n".data(using: .utf8)!)
        body.append(wavData)
        body.append("\r\n".data(using: .utf8)!)

        body.append("--\(boundary)--\r\n".data(using: .utf8)!)

        request.httpBody = body

        let (data, response) = try await session.data(for: request)

        try checkResponse(response)

        do {
            return try decoder.decode(TranscriptionResponse.self, from: data)
        } catch {
            throw APIError.decodingError(error)
        }
    }


    /// Transcribe audio data with SSE progress updates
    /// Sends audio to /transcribe-stream and reads heartbeat events until completion
    func transcribeAudioDataWithProgress(
        _ wavData: Data,
        onProgress: @escaping (Double, Double) -> Void  // (progress, elapsed)
    ) async throws -> TranscriptionResponse {
        let url = baseURL.appendingPathComponent("/transcribe-stream")

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        request.timeoutInterval = 600  // 10 minutes safety net

        // Build multipart body (same format as transcribeAudioData)
        let boundary = UUID().uuidString
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        var body = Data()
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: audio/wav\r\n\r\n".data(using: .utf8)!)
        body.append(wavData)
        body.append("\r\n".data(using: .utf8)!)
        body.append("--\(boundary)--\r\n".data(using: .utf8)!)

        request.httpBody = body

        let (asyncBytes, response) = try await session.bytes(for: request)
        try checkResponse(response)

        // Process SSE stream (same pattern as downloadModel and startStreamingRecording)
        for try await line in asyncBytes.lines {
            if line.hasPrefix("data: ") {
                let jsonString = String(line.dropFirst(6))
                if let data = jsonString.data(using: .utf8),
                   let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let type = json["type"] as? String {

                    switch type {
                    case "heartbeat":
                        let progress = json["progress"] as? Double ?? 0.0
                        let elapsed = json["elapsed"] as? Double ?? 0.0
                        await MainActor.run {
                            onProgress(progress, elapsed)
                        }

                    case "complete":
                        let text = json["text"] as? String ?? ""
                        let duration = json["duration_seconds"] as? Double ?? 0.0
                        let processingTime = json["processing_time"] as? Double ?? 0.0
                        let realtimeFactor = json["realtime_factor"] as? Double ?? 0.0
                        return TranscriptionResponse(
                            text: text,
                            durationSeconds: duration,
                            processingTime: processingTime,
                            realtimeFactor: realtimeFactor
                        )

                    case "error":
                        let message = json["message"] as? String ?? "Unknown error"
                        throw APIError.serverError(message)

                    default:
                        break
                    }
                }
            }
        }

        // If stream ends without a complete event
        throw APIError.serverError("Transcription stream ended unexpectedly")
    }


    // MARK: - Streaming Transcription

    /// Start streaming transcription session
    /// Returns an AsyncThrowingStream that yields transcription events
    func startStreamingRecording(onPartial: @escaping (String) -> Void, onFinal: @escaping (String) -> Void) async throws {
        let url = baseURL.appendingPathComponent("/recording/stream")

        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        request.timeoutInterval = 300  // 5 minutes max for streaming

        let (asyncBytes, response) = try await session.bytes(for: request)

        try checkResponse(response)

        // Process SSE stream
        for try await line in asyncBytes.lines {
            // SSE format: "data: {...json...}"
            if line.hasPrefix("data: ") {
                let jsonString = String(line.dropFirst(6))
                if let data = jsonString.data(using: .utf8),
                   let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let type = json["type"] as? String,
                   let text = json["text"] as? String {

                    await MainActor.run {
                        if type == "partial" {
                            onPartial(text)
                        } else if type == "final" {
                            onFinal(text)
                        }
                    }
                }
            }
        }
    }

    /// Stop the streaming transcription session
    func stopStreamingRecording() async throws -> MessageResponse {
        return try await post("/recording/stream/stop")
    }


    // MARK: - NeMo Status

    /// Get NeMo availability status
    func getNemoStatus() async throws -> NemoStatusResponse {
        return try await get("/nemo/status")
    }
}


// =============================================================================
// RESPONSE MODELS
// =============================================================================
/*
 These structs match the JSON responses from our Python API.
 Codable auto-generates the JSON parsing code.

 CodingKeys let us map Python's snake_case to Swift's camelCase:
   Python: "model_loaded" -> Swift: modelLoaded
 */

struct HealthResponse: Codable {
    let status: String
    let service: String
}

struct StatusResponse: Codable {
    let status: String
    let modelLoaded: Bool
    let modelId: String?
    let modelName: String?
    let device: String?

    enum CodingKeys: String, CodingKey {
        case status
        case modelLoaded = "model_loaded"
        case modelId = "model_id"
        case modelName = "model_name"
        case device
    }
}

struct ModelInfoResponse: Codable, Identifiable {
    let id: String
    let name: String
    let displayName: String
    let author: String
    let type: String           // "nemo" or "whisper"
    let languages: [String]
    let languageNames: [String]
    let description: String
    let multilingual: Bool
    let sizeMB: Int            // Download size in MB
    let ramMB: Int             // RAM usage in MB
    let streamingNative: Bool  // Is model optimized for streaming?
    let downloaded: Bool       // Is model downloaded?
    let downloadUrl: String?   // URL for download (Whisper only)
    let path: String?          // Path to model file
    let nemoRequired: Bool     // Does this model require NeMo?

    enum CodingKeys: String, CodingKey {
        case id, name, languages, description, multilingual, author, type, downloaded, path
        case displayName = "display_name"
        case languageNames = "language_names"
        case sizeMB = "size_mb"
        case ramMB = "ram_mb"
        case streamingNative = "streaming_native"
        case downloadUrl = "download_url"
        case nemoRequired = "nemo_required"
    }

    /// Size in GB for display
    var sizeGB: Double {
        return Double(sizeMB) / 1024.0
    }

    /// RAM in GB for display
    var ramGB: Double {
        return Double(ramMB) / 1024.0
    }
}

/// Storage information response
struct StorageInfoResponse: Codable {
    let storagePath: String
    let whisperPath: String
    let huggingfaceCache: String
    let totalSizeMB: Int
    let modelCount: Int

    enum CodingKeys: String, CodingKey {
        case storagePath = "storage_path"
        case whisperPath = "whisper_path"
        case huggingfaceCache = "huggingface_cache"
        case totalSizeMB = "total_size_mb"
        case modelCount = "model_count"
    }
}

/// Download progress event
struct DownloadProgressEvent: Codable {
    let status: String         // "downloading", "verifying", "complete", "error"
    let progress: Double?
    let downloadedMB: Int?
    let totalMB: Int?
    let path: String?
    let message: String?

    enum CodingKeys: String, CodingKey {
        case status, progress, path, message
        case downloadedMB = "downloaded_mb"
        case totalMB = "total_mb"
    }
}

struct LoadModelRequest: Codable {
    let modelId: String

    enum CodingKeys: String, CodingKey {
        case modelId = "model_id"
    }
}

struct MessageResponse: Codable {
    let message: String
    let success: Bool
}

struct TranscriptionResponse: Codable {
    let text: String
    let durationSeconds: Double
    let processingTime: Double
    let realtimeFactor: Double

    enum CodingKeys: String, CodingKey {
        case text
        case durationSeconds = "duration_seconds"
        case processingTime = "processing_time"
        case realtimeFactor = "realtime_factor"
    }
}

struct RecordingStatusResponse: Codable {
    let isRecording: Bool
    let durationSeconds: Double
    let volume: Double

    enum CodingKeys: String, CodingKey {
        case isRecording = "is_recording"
        case durationSeconds = "duration_seconds"
        case volume
    }
}

/// NeMo availability status response
struct NemoStatusResponse: Codable {
    let nemoAvailable: Bool
    let nemoVersion: String?
    let torchVersion: String?
    let device: String
    let backendMode: String

    enum CodingKeys: String, CodingKey {
        case device
        case nemoAvailable = "nemo_available"
        case nemoVersion = "nemo_version"
        case torchVersion = "torch_version"
        case backendMode = "backend_mode"
    }
}
