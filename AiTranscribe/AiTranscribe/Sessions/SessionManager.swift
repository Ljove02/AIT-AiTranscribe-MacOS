/*
 SessionManager.swift
 ====================

 Manages session recordings — long-form audio capture with batch transcription.

 This is the central state manager for the Sessions feature:
 - Creates/loads/deletes session directories
 - Tracks recording state and duration
 - Coordinates transcription progress

 Data Location:
 ~/Library/Application Support/AiTranscribe/Sessions/
 Each session has its own directory with audio.m4a, transcription.txt, and metadata.json
 */

import Foundation
import SwiftUI
import Combine
import CoreAudio

// MARK: - Session Status

/// Represents the current state of a session
enum SessionStatus: Codable, Equatable {
    case idle
    case recording
    case transcribing
    case completed
    case failed(String)

    // Custom Codable since enums with associated values need manual handling
    private enum CodingKeys: String, CodingKey {
        case type, message
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .idle:
            try container.encode("idle", forKey: .type)
        case .recording:
            try container.encode("recording", forKey: .type)
        case .transcribing:
            try container.encode("transcribing", forKey: .type)
        case .completed:
            try container.encode("completed", forKey: .type)
        case .failed(let message):
            try container.encode("failed", forKey: .type)
            try container.encode(message, forKey: .message)
        }
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)
        switch type {
        case "idle": self = .idle
        case "recording": self = .recording
        case "transcribing": self = .transcribing
        case "completed": self = .completed
        case "failed":
            let message = try container.decodeIfPresent(String.self, forKey: .message) ?? "Unknown error"
            self = .failed(message)
        default: self = .idle
        }
    }
}

// MARK: - Session Model

/// A single recording session with its metadata
struct Session: Identifiable, Codable, Equatable {
    let id: UUID
    var name: String
    let createdAt: Date
    var duration: TimeInterval
    var fileSize: Int64
    var hasAudio: Bool
    var hasTranscription: Bool
    var transcriptionText: String?
    var modelUsed: String?
    var ramBudgetMB: Int?
    var batchCount: Int?
    var transcriptionTime: TimeInterval?
    var wordCount: Int?
    var status: SessionStatus

    /// The directory name used for storage (based on date)
    var directoryName: String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd_HHmmss"
        return "session_\(formatter.string(from: createdAt))_\(id.uuidString.prefix(8))"
    }

    init(
        id: UUID = UUID(),
        name: String? = nil,
        createdAt: Date = Date()
    ) {
        self.id = id
        self.createdAt = createdAt
        self.duration = 0
        self.fileSize = 0
        self.hasAudio = false
        self.hasTranscription = false
        self.status = .idle

        // Auto-generate name from date if not provided
        if let name = name {
            self.name = name
        } else {
            let formatter = DateFormatter()
            formatter.dateFormat = "MMM d, h:mm a"
            self.name = "Session \(formatter.string(from: createdAt))"
        }
    }
}

// MARK: - Batch Progress

/// Tracks progress of batch transcription
struct BatchProgress: Equatable {
    var batch: Int
    var totalBatches: Int
    var progress: Double
    var cpuPercent: Double?
    var memoryMB: Double?
    var etaSeconds: Double?
    var textSoFar: String?
}

// MARK: - Session Metadata (for JSON persistence)

/// The metadata.json structure stored in each session directory
private struct SessionMetadata: Codable {
    let id: String
    var name: String
    let createdAt: String
    var durationSeconds: Double
    var fileSizeMB: Double
    var hasAudio: Bool
    var hasTranscription: Bool
    var modelUsed: String?
    var ramBudgetMB: Int?
    var batchCount: Int?
    var transcriptionTimeSeconds: Double?
    var wordCount: Int?
    var status: String
    var statusMessage: String?

    init(from session: Session) {
        self.id = session.id.uuidString
        self.name = session.name
        let formatter = ISO8601DateFormatter()
        self.createdAt = formatter.string(from: session.createdAt)
        self.durationSeconds = session.duration
        self.fileSizeMB = Double(session.fileSize) / 1_000_000.0
        self.hasAudio = session.hasAudio
        self.hasTranscription = session.hasTranscription
        self.modelUsed = session.modelUsed
        self.ramBudgetMB = session.ramBudgetMB
        self.batchCount = session.batchCount
        self.transcriptionTimeSeconds = session.transcriptionTime
        self.wordCount = session.wordCount

        switch session.status {
        case .idle: self.status = "idle"; self.statusMessage = nil
        case .recording: self.status = "recording"; self.statusMessage = nil
        case .transcribing: self.status = "transcribing"; self.statusMessage = nil
        case .completed: self.status = "completed"; self.statusMessage = nil
        case .failed(let msg): self.status = "failed"; self.statusMessage = msg
        }
    }

    func toSession() -> Session? {
        guard let uuid = UUID(uuidString: id) else { return nil }
        let formatter = ISO8601DateFormatter()
        guard let date = formatter.date(from: createdAt) else { return nil }

        var session = Session(id: uuid, name: name, createdAt: date)
        session.duration = durationSeconds
        session.fileSize = Int64(fileSizeMB * 1_000_000)
        session.hasAudio = hasAudio
        session.hasTranscription = hasTranscription
        session.modelUsed = modelUsed
        session.ramBudgetMB = ramBudgetMB
        session.batchCount = batchCount
        session.transcriptionTime = transcriptionTimeSeconds
        session.wordCount = wordCount

        switch status {
        case "recording": session.status = .recording
        case "transcribing": session.status = .transcribing
        case "completed": session.status = .completed
        case "failed": session.status = .failed(statusMessage ?? "Unknown error")
        default: session.status = .idle
        }

        return session
    }
}

// MARK: - Session Manager

@MainActor
class SessionManager: ObservableObject {

    // MARK: - Published State

    /// All sessions, sorted newest first
    @Published var sessions: [Session] = []

    /// Whether a session is currently being recorded
    @Published var isSessionRecording: Bool = false

    /// Duration of the current recording in seconds
    @Published var sessionDuration: TimeInterval = 0

    /// Current batch transcription progress (nil if not transcribing)
    @Published var transcriptionProgress: BatchProgress? = nil

    /// ID of the session currently being recorded
    @Published var currentRecordingSessionId: UUID? = nil

    /// ID of the session currently being transcribed
    @Published var currentTranscribingSessionId: UUID? = nil

    // MARK: - Recording

    /// The session recorder that handles mic + system audio capture
    let recorder = SessionRecorder()

    /// Floating indicator shown during session recording
    let indicatorController = SessionIndicatorController()

    /// Weak reference to AppState for mutual exclusion
    weak var appState: AppState?

    // MARK: - Private Properties

    private let fileManager = FileManager.default
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()
    private var durationTimer: Timer?
    private var durationObserver: AnyCancellable?

    /// Base directory for all sessions
    var sessionsDirectory: URL {
        let appSupport = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let dir = appSupport
            .appendingPathComponent("AiTranscribe", isDirectory: true)
            .appendingPathComponent("Sessions", isDirectory: true)
        try? fileManager.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    // MARK: - Initialization

    init() {
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    }

    // MARK: - Session CRUD

    /// Load all sessions from disk
    func loadSessions() {
        var loaded: [Session] = []

        guard let contents = try? fileManager.contentsOfDirectory(
            at: sessionsDirectory,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else {
            sessions = []
            return
        }

        for dirURL in contents {
            var isDir: ObjCBool = false
            guard fileManager.fileExists(atPath: dirURL.path, isDirectory: &isDir), isDir.boolValue else {
                continue
            }

            let metadataURL = dirURL.appendingPathComponent("metadata.json")
            guard let data = try? Data(contentsOf: metadataURL),
                  let metadata = try? decoder.decode(SessionMetadata.self, from: data),
                  var session = metadata.toSession() else {
                continue
            }

            // Check actual file state (audio might have been deleted externally)
            let audioURL = dirURL.appendingPathComponent("audio.m4a")
            session.hasAudio = fileManager.fileExists(atPath: audioURL.path)
            if session.hasAudio, let attrs = try? fileManager.attributesOfItem(atPath: audioURL.path) {
                session.fileSize = (attrs[.size] as? Int64) ?? 0
            }

            let transcriptionURL = dirURL.appendingPathComponent("transcription.txt")
            session.hasTranscription = fileManager.fileExists(atPath: transcriptionURL.path)
            if session.hasTranscription, let text = try? String(contentsOf: transcriptionURL, encoding: .utf8) {
                session.transcriptionText = text
                session.wordCount = text.split(separator: " ").count
            }

            // Reset stale recording/transcribing states (app may have crashed)
            if session.status == .recording || session.status == .transcribing {
                session.status = session.hasTranscription ? .completed : .idle
            }

            loaded.append(session)
        }

        // Sort newest first
        sessions = loaded.sorted { $0.createdAt > $1.createdAt }
        print("SessionManager: Loaded \(sessions.count) sessions")
    }

    /// Create a new session and its directory
    func createSession(name: String? = nil) -> Session {
        let session = Session(name: name)
        let sessionDir = getSessionDirectory(for: session)

        try? fileManager.createDirectory(at: sessionDir, withIntermediateDirectories: true)
        saveMetadata(for: session)

        sessions.insert(session, at: 0)
        print("SessionManager: Created session '\(session.name)' at \(sessionDir.path)")
        return session
    }

    /// Delete a session entirely (audio + transcription + metadata + directory)
    func deleteSession(id: UUID) {
        guard let session = sessions.first(where: { $0.id == id }) else { return }

        let sessionDir = getSessionDirectory(for: session)
        try? fileManager.removeItem(at: sessionDir)

        sessions.removeAll { $0.id == id }
        print("SessionManager: Deleted session '\(session.name)'")
    }

    /// Delete only the audio file for a session (keeps transcription)
    func deleteSessionAudio(id: UUID) {
        guard let index = sessions.firstIndex(where: { $0.id == id }) else { return }

        let audioURL = getSessionDirectory(for: sessions[index]).appendingPathComponent("audio.m4a")
        try? fileManager.removeItem(at: audioURL)

        sessions[index].hasAudio = false
        sessions[index].fileSize = 0
        saveMetadata(for: sessions[index])
        print("SessionManager: Deleted audio for session '\(sessions[index].name)'")
    }

    /// Bulk delete audio files
    func bulkDeleteAudio(transcribedOnly: Bool) {
        for i in sessions.indices {
            if sessions[i].hasAudio && (!transcribedOnly || sessions[i].hasTranscription) {
                let audioURL = getSessionDirectory(for: sessions[i]).appendingPathComponent("audio.m4a")
                try? fileManager.removeItem(at: audioURL)
                sessions[i].hasAudio = false
                sessions[i].fileSize = 0
                saveMetadata(for: sessions[i])
            }
        }
        print("SessionManager: Bulk deleted audio (transcribedOnly: \(transcribedOnly))")
    }

    /// Update a session's metadata on disk
    func updateSession(_ session: Session) {
        guard let index = sessions.firstIndex(where: { $0.id == session.id }) else { return }
        sessions[index] = session
        saveMetadata(for: session)
    }

    // MARK: - Storage Info

    /// Calculate total storage used by all session audio files
    func getTotalStorageSize() -> Int64 {
        return sessions.reduce(0) { $0 + $1.fileSize }
    }

    /// Format bytes as human-readable string
    static func formatFileSize(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        return formatter.string(fromByteCount: bytes)
    }

    /// Format duration as human-readable string (e.g., "47 min", "1h 12min")
    static func formatDuration(_ seconds: TimeInterval) -> String {
        let hours = Int(seconds) / 3600
        let minutes = Int(seconds) % 3600 / 60

        if hours > 0 {
            return "\(hours)h \(minutes)min"
        } else if minutes > 0 {
            return "\(minutes) min"
        } else {
            return "\(Int(seconds))s"
        }
    }

    /// Format duration as HH:MM:SS for the recording indicator
    static func formatDurationHHMMSS(_ seconds: TimeInterval) -> String {
        let hours = Int(seconds) / 3600
        let minutes = Int(seconds) % 3600 / 60
        let secs = Int(seconds) % 60
        return String(format: "%02d:%02d:%02d", hours, minutes, secs)
    }

    // MARK: - Directory Helpers

    /// Get the directory URL for a session
    func getSessionDirectory(for session: Session) -> URL {
        return sessionsDirectory.appendingPathComponent(session.directoryName, isDirectory: true)
    }

    /// Get the audio file URL for a session
    func getAudioURL(for session: Session) -> URL {
        return getSessionDirectory(for: session).appendingPathComponent("audio.m4a")
    }

    /// Get the transcription file URL for a session
    func getTranscriptionURL(for session: Session) -> URL {
        return getSessionDirectory(for: session).appendingPathComponent("transcription.txt")
    }

    // MARK: - Session Recording

    /// Start a new session recording.
    /// Creates a session, starts mic + system audio capture, and begins writing to M4A.
    func startSessionRecording(micDeviceId: AudioDeviceID? = nil) async -> Bool {
        guard !isSessionRecording else {
            print("SessionManager: Already recording a session")
            return false
        }

        // Mutual exclusion: don't start if quick-transcribe is active
        if let appState, appState.isRecording {
            print("SessionManager: Cannot start session — quick-transcribe is active")
            return false
        }

        // Create the session
        let session = createSession()
        let sessionDir = getSessionDirectory(for: session)

        // Start the recorder
        let started = await recorder.startRecording(sessionDir: sessionDir, micDeviceId: micDeviceId)

        if started {
            currentRecordingSessionId = session.id
            isSessionRecording = true
            appState?.isSessionRecordingActive = true

            // Update session status
            if let index = sessions.firstIndex(where: { $0.id == session.id }) {
                sessions[index].status = .recording
                saveMetadata(for: sessions[index])
            }

            // Forward recorder duration to our published property
            durationObserver = recorder.$duration
                .receive(on: RunLoop.main)
                .sink { [weak self] dur in
                    self?.sessionDuration = dur
                }

            // Show floating indicator
            indicatorController.show()

            print("SessionManager: Session recording started — '\(session.name)'")
        } else {
            // Recording failed — clean up the session
            deleteSession(id: session.id)
            print("SessionManager: Failed to start session recording")
        }

        return started
    }

    /// Stop the current session recording.
    /// Finalizes the audio file and updates session metadata.
    func stopSessionRecording() async {
        guard isSessionRecording, let sessionId = currentRecordingSessionId else { return }

        // Transition indicator to processing state while mixing/converting audio
        indicatorController.showProcessing()

        let audioURL = await recorder.stopRecording()

        // Hide the indicator now that conversion is complete
        indicatorController.hide()
        durationObserver?.cancel()
        durationObserver = nil

        // Update session metadata
        if let index = sessions.firstIndex(where: { $0.id == sessionId }) {
            sessions[index].duration = sessionDuration
            sessions[index].status = .idle
            sessions[index].hasAudio = audioURL != nil

            if let audioURL, let attrs = try? fileManager.attributesOfItem(atPath: audioURL.path) {
                sessions[index].fileSize = (attrs[.size] as? Int64) ?? 0
            }

            saveMetadata(for: sessions[index])
            print("SessionManager: Session recording stopped — '\(sessions[index].name)', " +
                  "duration: \(SessionManager.formatDuration(sessionDuration)), " +
                  "size: \(SessionManager.formatFileSize(sessions[index].fileSize))")
        }

        isSessionRecording = false
        currentRecordingSessionId = nil
        sessionDuration = 0
        appState?.isSessionRecordingActive = false
    }

    // MARK: - Session Rename

    /// Rename a session and persist to metadata.
    func renameSession(id: UUID, newName: String) {
        let trimmed = newName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, let index = sessions.firstIndex(where: { $0.id == id }) else { return }
        sessions[index].name = trimmed
        saveMetadata(for: sessions[index])
    }

    // MARK: - Session Transcription

    /// Start batch transcription for a session.
    /// Calls the backend /session/transcribe endpoint and streams progress via SSE.
    func startTranscription(sessionId: UUID, modelId: String, ramBudgetMB: Int, apiClient: APIClient) {
        guard let index = sessions.firstIndex(where: { $0.id == sessionId }) else { return }
        guard sessions[index].hasAudio else {
            print("SessionManager: No audio file for session")
            return
        }

        let session = sessions[index]
        let sessionDir = session.directoryName

        // Update state
        sessions[index].status = .transcribing
        sessions[index].modelUsed = modelId
        sessions[index].ramBudgetMB = ramBudgetMB
        currentTranscribingSessionId = sessionId
        transcriptionProgress = BatchProgress(batch: 0, totalBatches: 0, progress: 0)
        saveMetadata(for: sessions[index])

        Task {
            do {
                try await apiClient.transcribeSession(
                    sessionDir: sessionDir,
                    modelId: modelId,
                    ramBudgetMB: ramBudgetMB
                ) { [weak self] event in
                    self?.handleTranscriptionEvent(event, sessionId: sessionId)
                }
            } catch {
                print("SessionManager: Transcription error: \(error)")
                if let idx = sessions.firstIndex(where: { $0.id == sessionId }) {
                    sessions[idx].status = .failed(error.localizedDescription)
                    saveMetadata(for: sessions[idx])
                }
                currentTranscribingSessionId = nil
                transcriptionProgress = nil
            }
        }
    }

    /// Cancel the current batch transcription.
    func cancelTranscription(apiClient: APIClient) {
        guard let sessionId = currentTranscribingSessionId else { return }

        // Update UI immediately so user sees feedback
        if let index = sessions.firstIndex(where: { $0.id == sessionId }) {
            sessions[index].status = .failed("Cancelling...")
            saveMetadata(for: sessions[index])
        }
        currentTranscribingSessionId = nil
        transcriptionProgress = nil

        Task {
            let response = try? await apiClient.cancelSessionTranscription()
            await MainActor.run {
                if let index = sessions.firstIndex(where: { $0.id == sessionId }) {
                    sessions[index].status = .failed(response?.success == true
                        ? "Cancelled by user"
                        : "Cancel request failed")
                    saveMetadata(for: sessions[index])
                }
            }
        }
    }

    /// Handle an SSE event from the batch transcription backend.
    private func handleTranscriptionEvent(_ event: SessionTranscriptionEvent, sessionId: UUID) {
        guard let index = sessions.firstIndex(where: { $0.id == sessionId }) else { return }

        switch event.event {
        case "started":
            transcriptionProgress = BatchProgress(
                batch: 0,
                totalBatches: event.totalBatches ?? 0,
                progress: 0
            )

        case "batch_progress":
            transcriptionProgress?.batch = event.batch ?? 0
            transcriptionProgress?.totalBatches = event.total ?? transcriptionProgress?.totalBatches ?? 0
            transcriptionProgress?.progress = (event.percent ?? 0) / 100.0

        case "batch_complete":
            transcriptionProgress?.batch = event.batch ?? 0
            transcriptionProgress?.textSoFar = event.batchText

        case "stats":
            transcriptionProgress?.cpuPercent = event.cpuPercent
            transcriptionProgress?.memoryMB = event.memoryMb
            transcriptionProgress?.etaSeconds = event.etaSeconds

        case "done":
            sessions[index].status = .completed
            sessions[index].hasTranscription = true
            sessions[index].transcriptionText = event.fullText
            sessions[index].wordCount = event.wordCount
            sessions[index].batchCount = event.totalBatches
            sessions[index].transcriptionTime = event.totalTime
            saveMetadata(for: sessions[index])

            currentTranscribingSessionId = nil
            transcriptionProgress = nil
            print("SessionManager: Transcription complete — \(event.wordCount ?? 0) words")

        case "error":
            sessions[index].status = .failed(event.message ?? "Unknown error")
            saveMetadata(for: sessions[index])
            currentTranscribingSessionId = nil
            transcriptionProgress = nil
            print("SessionManager: Transcription error — \(event.message ?? "unknown")")

        case "cancelled":
            sessions[index].status = .idle
            sessions[index].transcriptionText = event.partialText
            if let partial = event.partialText, !partial.isEmpty {
                sessions[index].hasTranscription = true
                sessions[index].wordCount = partial.split(separator: " ").count
            }
            saveMetadata(for: sessions[index])
            currentTranscribingSessionId = nil
            transcriptionProgress = nil

        default:
            break
        }
    }

    // MARK: - Private Helpers

    private func saveMetadata(for session: Session) {
        let metadata = SessionMetadata(from: session)
        let sessionDir = getSessionDirectory(for: session)
        let metadataURL = sessionDir.appendingPathComponent("metadata.json")

        do {
            let data = try encoder.encode(metadata)
            try data.write(to: metadataURL, options: .atomic)
        } catch {
            print("SessionManager: Error saving metadata for '\(session.name)': \(error)")
        }
    }
}
