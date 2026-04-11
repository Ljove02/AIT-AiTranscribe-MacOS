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

enum SessionSummaryPreset: String, CaseIterable, Codable, Identifiable {
    case general
    case meetingNotes = "meeting_notes"
    case actionItems = "action_items"
    case technical

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .general: return "General Summary"
        case .meetingNotes: return "Meeting Notes"
        case .actionItems: return "Action Items"
        case .technical: return "Technical Summary"
        }
    }

    var fileName: String {
        switch self {
        case .general: return "summary-general.md"
        case .meetingNotes: return "summary-meeting-notes.md"
        case .actionItems: return "summary-action-items.md"
        case .technical: return "summary-technical.md"
        }
    }

    var storageKey: String {
        switch self {
        case .general: return "general"
        case .meetingNotes: return "meeting-notes"
        case .actionItems: return "action-items"
        case .technical: return "technical"
        }
    }

    var symbolName: String {
        switch self {
        case .general: return "text.alignleft"
        case .meetingNotes: return "person.3.sequence"
        case .actionItems: return "checklist"
        case .technical: return "desktopcomputer"
        }
    }

    var defaultInstructions: String {
        switch self {
        case .general:
            return """
            You are a precise summarization assistant.
            Summarize the transcript in the same language as the transcript.
            Prefer depth over brevity and write a readable, informative summary instead of a terse recap.
            Return Markdown with these sections:
            Overview
            Key points
            Notable details
            """
        case .meetingNotes:
            return """
            You are a meeting-notes assistant.
            Turn the transcript into polished meeting notes in the same language as the transcript.
            Keep decisions, open questions, and next steps explicit.
            Return Markdown with these sections:
            Overview
            Decisions
            Open questions
            Next steps
            """
        case .actionItems:
            return """
            You extract execution-ready action items from transcripts.
            Write in the same language as the transcript.
            Keep owners, deadlines, dependencies, and blockers explicit when the transcript gives enough signal.
            Return Markdown with these sections:
            Action items
            Follow-ups
            Risks or blockers
            """
        case .technical:
            return """
            You are a technical summarization assistant.
            Focus on architecture, implementation details, APIs, systems behavior, technical tradeoffs, bugs, and follow-up engineering work.
            Write in the same language as the transcript and keep technical terminology intact.
            Return Markdown with these sections:
            Technical overview
            Key implementation details
            Risks or edge cases
            Recommended follow-up
            """
        }
    }

    var defaultDefinition: SummaryPresetDefinition {
        SummaryPresetDefinition(
            id: rawValue,
            displayName: displayName,
            instructions: defaultInstructions,
            storageKey: storageKey,
            symbolName: symbolName,
            isBuiltIn: true
        )
    }
}

enum SummaryLengthOption: String, CaseIterable, Codable, Identifiable {
    case short
    case medium
    case long
    case custom

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .short: return "Short"
        case .medium: return "Medium"
        case .long: return "Long"
        case .custom: return "Custom"
        }
    }

    var defaultWordTarget: Int {
        switch self {
        case .short: return 240
        case .medium: return 520
        case .long: return 900
        case .custom: return 1100
        }
    }
}

struct SummaryPresetDefinition: Codable, Equatable, Identifiable {
    var id: String
    var displayName: String
    var instructions: String
    var storageKey: String
    var symbolName: String
    var isBuiltIn: Bool

    var fileName: String {
        "summary-\(storageKey).md"
    }
}

enum SummaryPresetLibrary {
    static let storageKey = "summaryPresetDefinitionsV1"

    static var builtIns: [SummaryPresetDefinition] {
        SessionSummaryPreset.allCases.map(\.defaultDefinition)
    }

    static func load() -> [SummaryPresetDefinition] {
        guard let data = UserDefaults.standard.data(forKey: storageKey),
              let presets = try? JSONDecoder().decode([SummaryPresetDefinition].self, from: data),
              !presets.isEmpty else {
            return builtIns
        }
        return mergeWithBuiltIns(presets)
    }

    static func save(_ presets: [SummaryPresetDefinition]) {
        guard let data = try? JSONEncoder().encode(mergeWithBuiltIns(presets)) else { return }
        UserDefaults.standard.set(data, forKey: storageKey)
    }

    static func mergeWithBuiltIns(_ presets: [SummaryPresetDefinition]) -> [SummaryPresetDefinition] {
        var byId = Dictionary(uniqueKeysWithValues: presets.map { ($0.id, $0) })
        for preset in builtIns where byId[preset.id] == nil {
            byId[preset.id] = preset
        }

        return byId.values.sorted { lhs, rhs in
            if lhs.isBuiltIn != rhs.isBuiltIn {
                return lhs.isBuiltIn && !rhs.isBuiltIn
            }
            return lhs.displayName.localizedCaseInsensitiveCompare(rhs.displayName) == .orderedAscending
        }
    }

    static func defaultDefinition(for id: String) -> SummaryPresetDefinition? {
        builtIns.first(where: { $0.id == id })
    }

    static func makeCustomPreset(name: String, instructions: String) -> SummaryPresetDefinition {
        let trimmedName = name.trimmingCharacters(in: .whitespacesAndNewlines)
        let displayName = trimmedName.isEmpty ? "Custom Summary" : trimmedName
        let normalizedInstructions = instructions.trimmingCharacters(in: .whitespacesAndNewlines)
        return SummaryPresetDefinition(
            id: "custom-\(UUID().uuidString.lowercased())",
            displayName: displayName,
            instructions: normalizedInstructions,
            storageKey: "custom-\(UUID().uuidString.prefix(8).lowercased())",
            symbolName: "slider.horizontal.3",
            isBuiltIn: false
        )
    }
}

struct SummaryGenerationPresetRequest: Codable, Equatable, Identifiable {
    var id: String { presetId }
    let presetId: String
    let displayName: String
    let fileName: String
    let systemPrompt: String
    let targetWords: Int
    let maxOutputTokens: Int

    enum CodingKeys: String, CodingKey {
        case presetId = "preset_id"
        case displayName = "display_name"
        case fileName = "file_name"
        case systemPrompt = "system_prompt"
        case targetWords = "target_words"
        case maxOutputTokens = "max_output_tokens"
    }

    static func make(from preset: SummaryPresetDefinition, targetWords: Int) -> SummaryGenerationPresetRequest {
        let clampedTarget = max(120, min(targetWords, 4000))
        let tokenBudget = max(256, Int(ceil(Double(clampedTarget) * 1.45)))
        return SummaryGenerationPresetRequest(
            presetId: preset.id,
            displayName: preset.displayName,
            fileName: preset.fileName,
            systemPrompt: preset.instructions,
            targetWords: clampedTarget,
            maxOutputTokens: tokenBudget
        )
    }
}

struct SessionSummaryMetadata: Codable, Equatable {
    var hasSummary: Bool
    var status: String
    var statusMessage: String?
    var fileName: String
    var modelId: String?
    var modelName: String?
    var presetDisplayName: String?
    var wordCount: Int?
    var targetWordCount: Int?
    var maxOutputTokens: Int?
    var processingTimeSeconds: Double?
    var generatedAt: Date?
    var text: String?

    enum CodingKeys: String, CodingKey {
        case hasSummary = "has_summary"
        case status
        case statusMessage = "status_message"
        case fileName = "file_name"
        case modelId = "model_id"
        case modelName = "model_name"
        case presetDisplayName = "preset_display_name"
        case wordCount = "word_count"
        case targetWordCount = "target_word_count"
        case maxOutputTokens = "max_output_tokens"
        case processingTimeSeconds = "processing_time_seconds"
        case generatedAt = "generated_at"
    }

    static func empty(for preset: SessionSummaryPreset) -> SessionSummaryMetadata {
        empty(
            displayName: preset.displayName,
            fileName: preset.fileName
        )
    }

    static func empty(
        displayName: String,
        fileName: String
    ) -> SessionSummaryMetadata {
        SessionSummaryMetadata(
            hasSummary: false,
            status: "idle",
            statusMessage: nil,
            fileName: fileName,
            modelId: nil,
            modelName: nil,
            presetDisplayName: displayName,
            wordCount: nil,
            targetWordCount: nil,
            maxOutputTokens: nil,
            processingTimeSeconds: nil,
            generatedAt: nil,
            text: nil
        )
    }

    static func defaultMap() -> [String: SessionSummaryMetadata] {
        Dictionary(uniqueKeysWithValues: SummaryPresetLibrary.builtIns.map {
            (
                $0.id,
                .empty(
                    displayName: $0.displayName,
                    fileName: $0.fileName
                )
            )
        })
    }
}

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
    var summaries: [String: SessionSummaryMetadata]

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
        self.summaries = SessionSummaryMetadata.defaultMap()

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

struct SessionSummaryProgress: Equatable {
    var activePresetId: String
    var selectedPresetIds: [String]
    var completedPresetIds: [String]
    var stage: String
    var partialText: String?
    var memoryEstimate: SummaryMemoryEstimateResponse?
    var kvQuantized: Bool?
    var currentIndex: Int
    var totalPresets: Int
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
    var summaries: [String: SessionSummaryMetadata]?

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
        self.summaries = session.summaries

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
        session.summaries = summaries ?? SessionSummaryMetadata.defaultMap()

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

    /// Shared singleton so AppDelegate can access it at launch
    static let shared = SessionManager()

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

    /// ID of the session currently being summarized
    @Published var currentSummarizingSessionId: UUID? = nil

    /// Current summary progress
    @Published var summaryProgress: SessionSummaryProgress? = nil

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
        encoder.dateEncodingStrategy = .iso8601
        decoder.dateDecodingStrategy = .iso8601
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

            session.summaries = SessionSummaryMetadata.defaultMap().merging(session.summaries) { _, persisted in
                persisted
            }

            for (presetId, metadata) in session.summaries {
                let summaryURL = dirURL.appendingPathComponent(metadata.fileName)
                guard fileManager.fileExists(atPath: summaryURL.path),
                      let text = try? String(contentsOf: summaryURL, encoding: .utf8) else {
                    continue
                }

                var loadedMetadata = metadata
                loadedMetadata.hasSummary = true
                if loadedMetadata.presetDisplayName == nil {
                    loadedMetadata.presetDisplayName = SummaryPresetLibrary.defaultDefinition(for: presetId)?.displayName
                        ?? presetId.replacingOccurrences(of: "_", with: " ").capitalized
                }
                loadedMetadata.text = text
                session.summaries[presetId] = loadedMetadata
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

    /// Delete multiple sessions entirely.
    func bulkDeleteSessions(ids: Set<UUID>) {
        guard !ids.isEmpty else { return }
        for id in ids {
            deleteSession(id: id)
        }
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

    /// Bulk clear transcriptions for selected sessions
    func bulkResetTranscription(ids: Set<UUID>) {
        guard !ids.isEmpty else { return }
        for id in ids {
            resetTranscription(id: id)
        }
        print("SessionManager: Bulk cleared transcription for \(ids.count) sessions")
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

    /// Get the summary file URL for a session summary file
    func getSummaryURL(for session: Session, fileName: String) -> URL {
        return getSessionDirectory(for: session).appendingPathComponent(fileName)
    }

    /// Update the text of an existing summary (user edit).
    func updateSummaryText(sessionId: UUID, presetId: String, text: String) {
        guard let index = sessions.firstIndex(where: { $0.id == sessionId }),
              var meta = sessions[index].summaries[presetId] else { return }
        meta.text = text
        meta.wordCount = text.split(separator: " ").count
        sessions[index].summaries[presetId] = meta
        saveSummaryText(for: sessions[index], fileName: meta.fileName, text: text)
        saveMetadata(for: sessions[index])
    }

    /// Delete stored transcription and all related summaries.
    func resetTranscription(id: UUID) {
        guard let index = sessions.firstIndex(where: { $0.id == id }) else { return }

        let transcriptionURL = getTranscriptionURL(for: sessions[index])
        try? fileManager.removeItem(at: transcriptionURL)

        sessions[index].hasTranscription = false
        sessions[index].transcriptionText = nil
        sessions[index].wordCount = nil
        sessions[index].batchCount = nil
        sessions[index].transcriptionTime = nil
        sessions[index].status = .idle

        clearSummaries(sessionIndex: index)
        saveMetadata(for: sessions[index])
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

        clearSummaries(sessionIndex: index)

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

    func startSummary(
        sessionId: UUID,
        modelId: String,
        presetRequests: [SummaryGenerationPresetRequest],
        apiClient: APIClient
    ) {
        guard let index = sessions.firstIndex(where: { $0.id == sessionId }) else { return }
        guard sessions[index].hasTranscription, sessions[index].transcriptionText?.isEmpty == false else {
            return
        }
        guard let firstPreset = presetRequests.first else { return }

        let modelName = appState?.availableSummaryModels.first(where: { $0.id == modelId })?.displayName
        let selectedPresetIds = presetRequests.map(\.presetId)
        for request in presetRequests {
            var metadata = sessions[index].summaries[request.presetId]
                ?? .empty(displayName: request.displayName, fileName: request.fileName)
            metadata.status = request.presetId == firstPreset.presetId ? "preparing" : "queued"
            metadata.statusMessage = nil
            metadata.fileName = request.fileName
            metadata.modelId = modelId
            metadata.modelName = modelName
            metadata.presetDisplayName = request.displayName
            metadata.targetWordCount = request.targetWords
            metadata.maxOutputTokens = request.maxOutputTokens
            sessions[index].summaries[request.presetId] = metadata
        }

        currentSummarizingSessionId = sessionId
        summaryProgress = SessionSummaryProgress(
            activePresetId: firstPreset.presetId,
            selectedPresetIds: selectedPresetIds,
            completedPresetIds: [],
            stage: "Preparing runtime",
            partialText: sessions[index].summaries[firstPreset.presetId]?.text,
            memoryEstimate: nil,
            kvQuantized: nil,
            currentIndex: 1,
            totalPresets: presetRequests.count
        )
        saveMetadata(for: sessions[index])

        let sessionDir = sessions[index].directoryName
        Task {
            do {
                try await apiClient.summarizeSession(
                    sessionDir: sessionDir,
                    modelId: modelId,
                    presets: presetRequests
                ) { [weak self] event in
                    self?.handleSummaryEvent(event, sessionId: sessionId)
                }
            } catch {
                markSummaryRunFailed(sessionId: sessionId, message: error.localizedDescription)
                currentSummarizingSessionId = nil
                summaryProgress = nil
            }
        }
    }

    func cancelSummary(apiClient: APIClient) {
        guard let sessionId = currentSummarizingSessionId else { return }

        Task {
            _ = try? await apiClient.cancelSessionSummary()
            await MainActor.run {
                markSummaryRunCancelled(sessionId: sessionId, message: "Cancelled by user")
                currentSummarizingSessionId = nil
                summaryProgress = nil
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

    private func handleSummaryEvent(_ event: SessionSummaryEvent, sessionId: UUID) {
        guard let index = sessions.firstIndex(where: { $0.id == sessionId }) else { return }

        let activePresetId = event.presetId ?? summaryProgress?.activePresetId
        var summary = activePresetId.flatMap { sessions[index].summaries[$0] }

        switch event.event {
        case "preparing_runtime":
            summaryProgress?.stage = "Preparing runtime"
            summaryProgress?.memoryEstimate = event.memoryEstimate
            summaryProgress?.kvQuantized = event.memoryEstimate?.willQuantizeKV

        case "batch_started":
            if let presetIds = event.presetIds, !presetIds.isEmpty {
                summaryProgress?.selectedPresetIds = presetIds
            }
            if let totalPresets = event.totalPresets {
                summaryProgress?.totalPresets = totalPresets
            }

        case "preset_started":
            guard let presetId = activePresetId else { return }
            let displayName = event.displayName
                ?? sessions[index].summaries[presetId]?.presetDisplayName
                ?? SummaryPresetLibrary.defaultDefinition(for: presetId)?.displayName
                ?? presetId
            let fileName = event.fileName
                ?? sessions[index].summaries[presetId]?.fileName
                ?? "summary-\(presetId).md"

            var startedSummary = sessions[index].summaries[presetId]
                ?? .empty(displayName: displayName, fileName: fileName)
            startedSummary.status = "preparing"
            startedSummary.statusMessage = nil
            startedSummary.fileName = fileName
            startedSummary.presetDisplayName = displayName
            startedSummary.targetWordCount = event.targetWords ?? startedSummary.targetWordCount
            startedSummary.maxOutputTokens = event.maxOutputTokens ?? startedSummary.maxOutputTokens
            sessions[index].summaries[presetId] = startedSummary

            summaryProgress?.activePresetId = presetId
            summaryProgress?.stage = "Preparing \(displayName)"
            summaryProgress?.partialText = startedSummary.text
            summaryProgress?.currentIndex = event.batchIndex ?? summaryProgress?.currentIndex ?? 1
            summaryProgress?.totalPresets = event.totalPresets ?? summaryProgress?.totalPresets ?? 1

        case "loading_model":
            summaryProgress?.stage = "Loading model"

        case "generating":
            guard let presetId = activePresetId else { return }
            var generatingSummary = summary
                ?? .empty(
                    displayName: event.displayName
                        ?? SummaryPresetLibrary.defaultDefinition(for: presetId)?.displayName
                        ?? presetId,
                    fileName: event.fileName ?? "summary-\(presetId).md"
                )
            generatingSummary.status = "generating"
            generatingSummary.statusMessage = nil
            sessions[index].summaries[presetId] = generatingSummary
            summaryProgress?.stage = "Generating \(generatingSummary.presetDisplayName ?? "summary")"
            summaryProgress?.activePresetId = presetId
            summaryProgress?.currentIndex = event.batchIndex ?? summaryProgress?.currentIndex ?? 1
            summaryProgress?.totalPresets = event.totalPresets ?? summaryProgress?.totalPresets ?? 1
            summaryProgress?.kvQuantized = event.kvQuantized

        case "partial":
            guard let presetId = activePresetId else { return }
            var partialSummary = summary
                ?? .empty(
                    displayName: event.displayName
                        ?? SummaryPresetLibrary.defaultDefinition(for: presetId)?.displayName
                        ?? presetId,
                    fileName: event.fileName ?? "summary-\(presetId).md"
                )
            partialSummary.text = event.text
            partialSummary.status = "generating"
            sessions[index].summaries[presetId] = partialSummary
            summaryProgress?.partialText = event.text
            summaryProgress?.activePresetId = presetId

        case "done":
            guard let presetId = activePresetId else { return }
            var doneSummary = summary
                ?? .empty(
                    displayName: event.displayName
                        ?? SummaryPresetLibrary.defaultDefinition(for: presetId)?.displayName
                        ?? presetId,
                    fileName: event.fileName ?? "summary-\(presetId).md"
                )
            doneSummary.hasSummary = true
            doneSummary.status = "completed"
            doneSummary.statusMessage = nil
            doneSummary.modelId = event.modelId ?? doneSummary.modelId
            doneSummary.modelName = event.modelName ?? doneSummary.modelName
            doneSummary.presetDisplayName = event.displayName ?? doneSummary.presetDisplayName
            doneSummary.wordCount = event.outputWordCount ?? event.text?.split(separator: " ").count
            doneSummary.targetWordCount = event.targetWords ?? doneSummary.targetWordCount
            doneSummary.maxOutputTokens = event.maxOutputTokens ?? doneSummary.maxOutputTokens
            doneSummary.processingTimeSeconds = event.processingTimeSeconds
            doneSummary.generatedAt = Date()
            doneSummary.text = event.text
            sessions[index].summaries[presetId] = doneSummary
            saveSummaryText(for: sessions[index], fileName: doneSummary.fileName, text: event.text ?? "")
            saveMetadata(for: sessions[index])
            if summaryProgress?.completedPresetIds.contains(presetId) == false {
                summaryProgress?.completedPresetIds.append(presetId)
            }
            summaryProgress?.partialText = event.text
            summaryProgress?.activePresetId = presetId
            summaryProgress?.currentIndex = event.batchIndex ?? summaryProgress?.currentIndex ?? 1
            summaryProgress?.totalPresets = event.totalPresets ?? summaryProgress?.totalPresets ?? 1
            summaryProgress?.stage = "Saved \(doneSummary.presetDisplayName ?? "summary")"

        case "error":
            if let presetId = activePresetId {
                var failedSummary = summary
                    ?? .empty(
                        displayName: event.displayName
                            ?? SummaryPresetLibrary.defaultDefinition(for: presetId)?.displayName
                            ?? presetId,
                        fileName: event.fileName ?? "summary-\(presetId).md"
                    )
                failedSummary.status = "failed"
                failedSummary.statusMessage = event.message ?? "Unknown error"
                sessions[index].summaries[presetId] = failedSummary
                saveMetadata(for: sessions[index])
            }
            currentSummarizingSessionId = nil
            summaryProgress = nil

        case "cancelled":
            markSummaryRunCancelled(sessionId: sessionId, message: "Cancelled by user")
            currentSummarizingSessionId = nil
            summaryProgress = nil

        case "batch_complete":
            currentSummarizingSessionId = nil
            summaryProgress = nil

        default:
            break
        }
    }

    // MARK: - Private Helpers

    private func saveSummaryText(for session: Session, fileName: String, text: String) {
        let summaryURL = getSummaryURL(for: session, fileName: fileName)
        try? text.write(to: summaryURL, atomically: true, encoding: .utf8)
    }

    private func clearSummaries(sessionIndex: Int) {
        let session = sessions[sessionIndex]
        for metadata in sessions[sessionIndex].summaries.values {
            let summaryURL = getSummaryURL(for: session, fileName: metadata.fileName)
            try? fileManager.removeItem(at: summaryURL)
        }
        sessions[sessionIndex].summaries = SessionSummaryMetadata.defaultMap()
    }

    private func markSummaryRunFailed(sessionId: UUID, message: String) {
        guard let index = sessions.firstIndex(where: { $0.id == sessionId }) else { return }
        let selectedPresetIds = summaryProgress?.selectedPresetIds ?? []
        let completedPresetIds = Set(summaryProgress?.completedPresetIds ?? [])

        for presetId in selectedPresetIds where !completedPresetIds.contains(presetId) {
            var metadata = sessions[index].summaries[presetId]
                ?? .empty(
                    displayName: SummaryPresetLibrary.defaultDefinition(for: presetId)?.displayName
                        ?? presetId.replacingOccurrences(of: "_", with: " ").capitalized,
                    fileName: SummaryPresetLibrary.defaultDefinition(for: presetId)?.fileName
                        ?? "summary-\(presetId).md"
                )
            metadata.status = "failed"
            metadata.statusMessage = message
            sessions[index].summaries[presetId] = metadata
        }
        saveMetadata(for: sessions[index])
    }

    private func markSummaryRunCancelled(sessionId: UUID, message: String) {
        guard let index = sessions.firstIndex(where: { $0.id == sessionId }) else { return }
        let selectedPresetIds = summaryProgress?.selectedPresetIds ?? []
        let completedPresetIds = Set(summaryProgress?.completedPresetIds ?? [])

        for presetId in selectedPresetIds where !completedPresetIds.contains(presetId) {
            var metadata = sessions[index].summaries[presetId]
                ?? .empty(
                    displayName: SummaryPresetLibrary.defaultDefinition(for: presetId)?.displayName
                        ?? presetId.replacingOccurrences(of: "_", with: " ").capitalized,
                    fileName: SummaryPresetLibrary.defaultDefinition(for: presetId)?.fileName
                        ?? "summary-\(presetId).md"
                )
            metadata.status = "cancelled"
            metadata.statusMessage = message
            sessions[index].summaries[presetId] = metadata
        }
        saveMetadata(for: sessions[index])
    }

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
