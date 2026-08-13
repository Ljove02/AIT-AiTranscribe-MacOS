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
import AVFoundation

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

enum SessionAutomationSource: String, CaseIterable, Codable, Identifiable {
    case local
    case `public`

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .local: return "Local"
        case .public: return "Public API"
        }
    }
}

enum SessionAutomationStatus: String, Codable, Equatable {
    case queued
    case transcribing
    case transcriptionFailed = "transcription_failed"
    case summarizing
    case summaryFailed = "summary_failed"
    case completed
    case skipped

    var displayName: String {
        switch self {
        case .queued: return "Queued"
        case .transcribing: return "Auto-transcribing"
        case .transcriptionFailed: return "Auto-transcription failed"
        case .summarizing: return "Auto-summarizing"
        case .summaryFailed: return "Auto-summary failed"
        case .completed: return "Auto-processed"
        case .skipped: return "Skipped"
        }
    }
}

struct SessionAutomationMetadata: Codable, Equatable {
    var status: SessionAutomationStatus
    var message: String?
    var startedAt: Date
    var updatedAt: Date
    var completedAt: Date?
    var transcribeEnabled: Bool
    var summarizeEnabled: Bool
    var transcriptionSource: SessionAutomationSource
    var transcriptionProviderId: String?
    var transcriptionModelId: String?
    var summarySource: SessionAutomationSource?
    var summaryProviderId: String?
    var summaryModelId: String?
    var summaryPresetIds: [String]

    var isActive: Bool {
        status == .queued || status == .transcribing || status == .summarizing
    }

    static func queued(settings: SessionAutomationPreferences) -> SessionAutomationMetadata {
        let now = Date()
        return SessionAutomationMetadata(
            status: .queued,
            message: settings.autoSummarize ? "Waiting to transcribe, then summarize." : "Waiting to transcribe.",
            startedAt: now,
            updatedAt: now,
            completedAt: nil,
            transcribeEnabled: settings.autoTranscribe,
            summarizeEnabled: settings.autoSummarize,
            transcriptionSource: settings.transcriptionSource,
            transcriptionProviderId: settings.transcriptionProviderId,
            transcriptionModelId: settings.transcriptionSource == .local
                ? settings.localTranscriptionModelId
                : settings.transcriptionModelId,
            summarySource: settings.summarySource,
            summaryProviderId: settings.summaryProviderId,
            summaryModelId: settings.summarySource == .local
                ? settings.localSummaryModelId
                : settings.summaryModelId,
            summaryPresetIds: settings.summaryPresetIds
        )
    }
}

struct SessionAutomationPreferences {
    static let autoTranscribeKey = "sessionAutomationAutoTranscribe"
    static let transcriptionSourceKey = "sessionAutomationTranscriptionSource"
    static let localTranscriptionModelKey = "sessionAutomationLocalTranscriptionModelId"
    static let transcriptionProviderKey = "sessionAutomationTranscriptionProviderId"
    static let transcriptionModelKey = "sessionAutomationTranscriptionModelId"
    static let localRamBudgetGBKey = "sessionAutomationLocalRamBudgetGB"
    static let publicBatchTargetKey = "sessionAutomationPublicBatchTarget"
    static let autoSummarizeKey = "sessionAutomationAutoSummarize"
    static let summarySourceKey = "sessionAutomationSummarySource"
    static let localSummaryModelKey = "sessionAutomationLocalSummaryModelId"
    static let summaryProviderKey = "sessionAutomationSummaryProviderId"
    static let summaryModelKey = "sessionAutomationSummaryModelId"
    static let summaryPresetIdsKey = "sessionAutomationSummaryPresetIds"
    static let summaryLengthOptionKey = "sessionAutomationSummaryLengthOption"
    static let summaryCustomWordsKey = "sessionAutomationSummaryCustomWords"

    var autoTranscribe: Bool
    var transcriptionSource: SessionAutomationSource
    var localTranscriptionModelId: String
    var transcriptionProviderId: String
    var transcriptionModelId: String
    var localRamBudgetGB: Double
    var publicBatchTarget: Int
    var autoSummarize: Bool
    var summarySource: SessionAutomationSource
    var localSummaryModelId: String
    var summaryProviderId: String
    var summaryModelId: String
    var summaryPresetIds: [String]
    var summaryLengthOption: SummaryLengthOption
    var summaryCustomWords: Int

    var targetSummaryWords: Int {
        summaryLengthOption == .custom ? max(120, summaryCustomWords) : summaryLengthOption.defaultWordTarget
    }

    static func load() -> SessionAutomationPreferences {
        let defaults = UserDefaults.standard
        let apiTranscriptionModels = APITranscriptionModelStore.load()
        let firstPublicTranscription = apiTranscriptionModels.first {
            APITranscriptionModelStore.isSupportedModel($0.modelId, providerId: $0.providerId)
        }
        let apiLanguageModels = APILanguageModelStore.load()
        let firstLanguageModel = apiLanguageModels.first

        let transcriptionSource = SessionAutomationSource(
            rawValue: defaults.string(forKey: transcriptionSourceKey) ?? SessionAutomationSource.local.rawValue
        ) ?? .local
        let summarySource = SessionAutomationSource(
            rawValue: defaults.string(forKey: summarySourceKey) ?? SessionAutomationSource.local.rawValue
        ) ?? .local
        let presetIds = defaults.stringArray(forKey: summaryPresetIdsKey)
            ?? [SessionSummaryPreset.meetingNotes.rawValue]

        return SessionAutomationPreferences(
            autoTranscribe: defaults.bool(forKey: autoTranscribeKey),
            transcriptionSource: transcriptionSource,
            localTranscriptionModelId: defaults.string(forKey: localTranscriptionModelKey)
                ?? defaults.string(forKey: "preferredModelId")
                ?? "parakeet-v2",
            transcriptionProviderId: defaults.string(forKey: transcriptionProviderKey)
                ?? firstPublicTranscription?.providerId
                ?? TranscriptionProviderDescriptor.openAI.id,
            transcriptionModelId: defaults.string(forKey: transcriptionModelKey)
                ?? firstPublicTranscription?.modelId
                ?? TranscriptionModelDescriptor.openAIModels[0].id,
            localRamBudgetGB: defaults.object(forKey: localRamBudgetGBKey) == nil
                ? 4.0
                : defaults.double(forKey: localRamBudgetGBKey),
            publicBatchTarget: defaults.integer(forKey: publicBatchTargetKey),
            autoSummarize: defaults.bool(forKey: autoSummarizeKey),
            summarySource: summarySource,
            localSummaryModelId: defaults.string(forKey: localSummaryModelKey)
                ?? defaults.string(forKey: "preferredSummaryModelId")
                ?? "gemma-4-e2b-it-4bit",
            summaryProviderId: defaults.string(forKey: summaryProviderKey)
                ?? firstLanguageModel?.providerId
                ?? TranscriptionProviderDescriptor.openAI.id,
            summaryModelId: defaults.string(forKey: summaryModelKey)
                ?? firstLanguageModel?.modelId
                ?? APILanguageModelStore.defaultModels[0].modelId,
            summaryPresetIds: presetIds.isEmpty ? [SessionSummaryPreset.meetingNotes.rawValue] : presetIds,
            summaryLengthOption: SummaryLengthOption(
                rawValue: defaults.string(forKey: summaryLengthOptionKey) ?? SummaryLengthOption.medium.rawValue
            ) ?? .medium,
            summaryCustomWords: defaults.object(forKey: summaryCustomWordsKey) == nil
                ? SummaryLengthOption.custom.defaultWordTarget
                : defaults.integer(forKey: summaryCustomWordsKey)
        )
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
    var inputTokens: Int?
    var outputTokens: Int?
    var estimatedCostUSD: Double?
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
        case inputTokens = "input_tokens"
        case outputTokens = "output_tokens"
        case estimatedCostUSD = "estimated_cost_usd"
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
            inputTokens: nil,
            outputTokens: nil,
            estimatedCostUSD: nil,
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
    var transcriptionEstimatedCostUSD: Double?
    var status: SessionStatus
    var summaries: [String: SessionSummaryMetadata]
    var automation: SessionAutomationMetadata?

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
        self.transcriptionEstimatedCostUSD = nil
        self.status = .idle
        self.summaries = SessionSummaryMetadata.defaultMap()
        self.automation = nil

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

private struct PublicSessionAudioBatch: Sendable {
    let index: Int
    let total: Int
    let url: URL
    let durationSeconds: Double
    let fileName: String
    let mimeType: String
}

private struct PublicSessionBatchPlan: Sendable {
    let batches: [PublicSessionAudioBatch]
    let temporaryDirectory: URL?
}

private struct PublicSessionBatchResult: Sendable {
    let index: Int
    let text: String
    let processingTimeSeconds: Double
    let estimatedCostUSD: Double?
}

private struct PublicSessionTranscriptionResult: Sendable {
    let text: String
    let processingTimeSeconds: Double
    let estimatedCostUSD: Double?
    let batchCount: Int
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
    var transcriptionEstimatedCostUSD: Double?
    var status: String
    var statusMessage: String?
    var summaries: [String: SessionSummaryMetadata]?
    var automation: SessionAutomationMetadata?

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
        self.transcriptionEstimatedCostUSD = session.transcriptionEstimatedCostUSD
        self.summaries = session.summaries
        self.automation = session.automation

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
        session.transcriptionEstimatedCostUSD = transcriptionEstimatedCostUSD
        session.summaries = summaries ?? SessionSummaryMetadata.defaultMap()
        session.automation = automation

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

    /// Sessions waiting for automatic post-processing.
    @Published var automationQueue: [UUID] = []

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
    private var isAutomationPumpScheduled = false

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
        sessions[index].transcriptionEstimatedCostUSD = nil
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

        enqueueAutomationIfNeeded(for: sessionId)
    }

    // MARK: - Session Automation

    private func enqueueAutomationIfNeeded(for sessionId: UUID) {
        let settings = SessionAutomationPreferences.load()
        guard settings.autoTranscribe else { return }
        guard let index = sessions.firstIndex(where: { $0.id == sessionId }),
              sessions[index].hasAudio else { return }

        sessions[index].automation = .queued(settings: settings)
        saveMetadata(for: sessions[index])

        if !automationQueue.contains(sessionId) {
            automationQueue.append(sessionId)
        }
        pumpAutomationQueue()
    }

    private func pumpAutomationQueue() {
        guard !isAutomationPumpScheduled else { return }
        isAutomationPumpScheduled = true
        Task { @MainActor in
            defer { isAutomationPumpScheduled = false }
            guard currentTranscribingSessionId == nil,
                  currentSummarizingSessionId == nil,
                  !isSessionRecording else { return }

            while let sessionId = automationQueue.first {
                automationQueue.removeFirst()
                guard let index = sessions.firstIndex(where: { $0.id == sessionId }),
                      sessions[index].hasAudio,
                      sessions[index].automation?.isActive == true else {
                    continue
                }

                beginAutomationTranscription(sessionId: sessionId)
                return
            }
        }
    }

    private func beginAutomationTranscription(sessionId: UUID) {
        let settings = SessionAutomationPreferences.load()
        guard sessions.contains(where: { $0.id == sessionId }) else { return }

        updateAutomation(
            sessionId: sessionId,
            status: .transcribing,
            message: "Starting automatic transcription."
        )

        switch settings.transcriptionSource {
        case .public:
            let model = APITranscriptionModelStore.load().first {
                $0.providerId == settings.transcriptionProviderId &&
                    $0.modelId == settings.transcriptionModelId &&
                    APITranscriptionModelStore.isSupportedModel($0.modelId, providerId: $0.providerId)
            } ?? APITranscriptionModelStore.load().first {
                APITranscriptionModelStore.isSupportedModel($0.modelId, providerId: $0.providerId)
            }

            guard let model else {
                markAutomationFailed(
                    sessionId: sessionId,
                    status: .transcriptionFailed,
                    message: "No supported public transcription model is configured."
                )
                return
            }

            guard let apiKey = try? APIKeychainStore.apiKey(providerId: model.providerId),
                  !apiKey.isEmpty else {
                markAutomationFailed(
                    sessionId: sessionId,
                    status: .transcriptionFailed,
                    message: "Add \(model.providerName) API key first."
                )
                return
            }

            startPublicTranscription(
                sessionId: sessionId,
                providerId: model.providerId,
                modelId: model.modelId,
                displayName: model.displayName,
                targetBatchCount: settings.publicBatchTarget > 0 ? settings.publicBatchTarget : nil
            )

        case .local:
            guard let appState else {
                markAutomationFailed(
                    sessionId: sessionId,
                    status: .transcriptionFailed,
                    message: "App state is unavailable."
                )
                return
            }

            guard appState.isServerConnected else {
                markAutomationFailed(
                    sessionId: sessionId,
                    status: .transcriptionFailed,
                    message: "Backend is not connected."
                )
                return
            }

            let modelId = settings.localTranscriptionModelId
            guard appState.availableModels.contains(where: { $0.id == modelId && $0.downloaded == true && $0.sessionCompatible }) else {
                markAutomationFailed(
                    sessionId: sessionId,
                    status: .transcriptionFailed,
                    message: "Download the selected session transcription model first."
                )
                return
            }

            Task { @MainActor in
                if appState.isModelLoaded && appState.loadedModelId != modelId {
                    await appState.unloadModel()
                }
                if !appState.isModelLoaded || appState.loadedModelId != modelId {
                    updateAutomation(
                        sessionId: sessionId,
                        status: .transcribing,
                        message: "Loading \(modelId)."
                    )
                    await appState.loadModel(modelId: modelId)
                }

                guard appState.isModelLoaded && appState.loadedModelId == modelId else {
                    markAutomationFailed(
                        sessionId: sessionId,
                        status: .transcriptionFailed,
                        message: "Could not load \(modelId)."
                    )
                    return
                }

                startTranscription(
                    sessionId: sessionId,
                    modelId: modelId,
                    ramBudgetMB: Int(max(1.0, settings.localRamBudgetGB) * 1024),
                    apiClient: appState.apiClient
                )
            }
        }
    }

    private func continueAutomationAfterTranscription(sessionId: UUID) {
        let settings = SessionAutomationPreferences.load()
        guard settings.autoSummarize else {
            completeAutomation(sessionId: sessionId, message: "Automatic transcription complete.")
            pumpAutomationQueue()
            return
        }

        guard let index = sessions.firstIndex(where: { $0.id == sessionId }),
              sessions[index].hasTranscription else {
            markAutomationFailed(
                sessionId: sessionId,
                status: .summaryFailed,
                message: "Summary skipped because transcription did not complete."
            )
            return
        }

        let presetRequests = automationSummaryPresetRequests(settings: settings)
        guard !presetRequests.isEmpty else {
            completeAutomation(sessionId: sessionId, message: "Automatic transcription complete. No summary presets selected.")
            pumpAutomationQueue()
            return
        }

        updateAutomation(
            sessionId: sessionId,
            status: .summarizing,
            message: "Generating \(presetRequests.count) automatic summar\(presetRequests.count == 1 ? "y" : "ies")."
        )

        switch settings.summarySource {
        case .public:
            let model = APILanguageModelStore.load().first {
                $0.providerId == settings.summaryProviderId &&
                    $0.modelId == settings.summaryModelId
            } ?? APILanguageModelStore.load().first

            guard let model else {
                markAutomationFailed(
                    sessionId: sessionId,
                    status: .summaryFailed,
                    message: "No public language model is configured."
                )
                return
            }

            guard let apiKey = try? APIKeychainStore.apiKey(providerId: model.providerId),
                  !apiKey.isEmpty else {
                markAutomationFailed(
                    sessionId: sessionId,
                    status: .summaryFailed,
                    message: "Add \(model.providerName) API key first."
                )
                return
            }

            startPublicSummary(
                sessionId: sessionId,
                providerId: model.providerId,
                modelId: model.modelId,
                modelName: model.displayName,
                presetRequests: presetRequests
            )

        case .local:
            guard let appState else {
                markAutomationFailed(
                    sessionId: sessionId,
                    status: .summaryFailed,
                    message: "App state is unavailable."
                )
                return
            }

            guard appState.availableSummaryModels.contains(where: { $0.id == settings.localSummaryModelId && $0.downloaded }) else {
                markAutomationFailed(
                    sessionId: sessionId,
                    status: .summaryFailed,
                    message: "Download the selected local summary model first."
                )
                return
            }

            startSummary(
                sessionId: sessionId,
                modelId: settings.localSummaryModelId,
                presetRequests: presetRequests,
                apiClient: appState.apiClient
            )
        }
    }

    private func automationSummaryPresetRequests(settings: SessionAutomationPreferences) -> [SummaryGenerationPresetRequest] {
        let targetWords = settings.targetSummaryWords
        let selectedIds = Set(settings.summaryPresetIds)
        return SummaryPresetLibrary.load()
            .filter {
                selectedIds.contains($0.id) &&
                    !$0.instructions.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            }
            .map { SummaryGenerationPresetRequest.make(from: $0, targetWords: targetWords) }
    }

    private func updateAutomation(sessionId: UUID, status: SessionAutomationStatus, message: String?) {
        guard let index = sessions.firstIndex(where: { $0.id == sessionId }) else { return }
        var metadata = sessions[index].automation ?? .queued(settings: SessionAutomationPreferences.load())
        metadata.status = status
        metadata.message = message
        metadata.updatedAt = Date()
        if status == .completed || status == .skipped || status == .summaryFailed || status == .transcriptionFailed {
            metadata.completedAt = Date()
        }
        sessions[index].automation = metadata
        saveMetadata(for: sessions[index])
    }

    private func completeAutomation(sessionId: UUID, message: String?) {
        updateAutomation(sessionId: sessionId, status: .completed, message: message)
    }

    private func markAutomationFailed(sessionId: UUID, status: SessionAutomationStatus, message: String) {
        updateAutomation(sessionId: sessionId, status: status, message: message)
        pumpAutomationQueue()
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
                if sessions.first(where: { $0.id == sessionId })?.automation?.status == .transcribing {
                    markAutomationFailed(
                        sessionId: sessionId,
                        status: .transcriptionFailed,
                        message: error.localizedDescription
                    )
                }
            }
        }
    }

    /// Start public API transcription for a session with the selected public provider model.
    /// Long recordings are exported into ordered chunks and transcribed concurrently.
    func startPublicTranscription(
        sessionId: UUID,
        providerId: String,
        modelId: String,
        displayName: String,
        targetBatchCount: Int? = nil
    ) {
        guard let index = sessions.firstIndex(where: { $0.id == sessionId }) else { return }
        guard sessions[index].hasAudio else {
            print("SessionManager: No audio file for session")
            return
        }

        let session = sessions[index]
        let audioURL = getAudioURL(for: session)

        guard let attrs = try? fileManager.attributesOfItem(atPath: audioURL.path),
              let fileSize = attrs[.size] as? Int64 else {
            sessions[index].status = .failed("Could not read session audio file.")
            saveMetadata(for: sessions[index])
            return
        }

        guard let apiKey = try? APIKeychainStore.apiKey(providerId: providerId),
              !apiKey.isEmpty else {
            sessions[index].status = .failed("Add \(publicTranscriptionProviderName(for: providerId)) API key first.")
            saveMetadata(for: sessions[index])
            return
        }

        clearSummaries(sessionIndex: index)

        sessions[index].status = .transcribing
        sessions[index].modelUsed = modelId
        sessions[index].ramBudgetMB = nil
        currentTranscribingSessionId = sessionId
        transcriptionProgress = BatchProgress(
            batch: 0,
            totalBatches: 0,
            progress: 0.02
        )
        saveMetadata(for: sessions[index])

        Task {
            let startedAt = Date()
            do {
                let plan = try await Self.makePublicSessionBatchPlan(
                    audioURL: audioURL,
                    durationSeconds: session.duration,
                    targetBatchCount: targetBatchCount
                )
                defer {
                    if let temporaryDirectory = plan.temporaryDirectory {
                        try? FileManager.default.removeItem(at: temporaryDirectory)
                    }
                }

                await MainActor.run {
                    transcriptionProgress = BatchProgress(
                        batch: 0,
                        totalBatches: plan.batches.count,
                        progress: 0.08,
                        textSoFar: "Prepared \(plan.batches.count) API batch\(plan.batches.count == 1 ? "" : "es")"
                    )
                    print("SessionManager: public session batching route=\(providerId):\(modelId) batches=\(plan.batches.count) targetBatches=\(targetBatchCount ?? 0) fileSizeMB=\(Double(fileSize) / 1_000_000.0)")
                }

                let result = try await Self.transcribePublicSessionBatches(
                    plan.batches,
                    providerId: providerId,
                    modelId: modelId,
                    apiKey: apiKey
                ) { completed, total, preview in
                    await MainActor.run {
                        let progress = total > 0 ? 0.08 + (Double(completed) / Double(total) * 0.90) : 0.08
                        self.transcriptionProgress = BatchProgress(
                            batch: completed,
                            totalBatches: total,
                            progress: min(progress, 0.98),
                            textSoFar: preview
                        )
                    }
                }

                await MainActor.run {
                    guard let idx = sessions.firstIndex(where: { $0.id == sessionId }) else { return }
                    let text = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
                    sessions[idx].status = .completed
                    sessions[idx].hasTranscription = !text.isEmpty
                    sessions[idx].transcriptionText = text
                    sessions[idx].wordCount = text.split(separator: " ").count
                    sessions[idx].batchCount = result.batchCount
                    sessions[idx].transcriptionTime = Date().timeIntervalSince(startedAt)
                    sessions[idx].modelUsed = displayName
                    sessions[idx].transcriptionEstimatedCostUSD = result.estimatedCostUSD
                        ?? APITranscriptionModelStore.load()
                            .first(where: { $0.providerId == providerId && $0.modelId == modelId })?
                            .estimatedCost(durationSeconds: sessions[idx].duration)
                    saveTranscriptionText(for: sessions[idx], text: text)
                    saveMetadata(for: sessions[idx])
                    currentTranscribingSessionId = nil
                    transcriptionProgress = nil
                    print(String(
                        format: "SessionManager: public session transcription complete route=%@ batches=%d elapsed=%.2fs requestElapsed=%.2fs words=%d estimatedCost=$%.6f",
                        "\(providerId):\(modelId)",
                        result.batchCount,
                        Date().timeIntervalSince(startedAt),
                        result.processingTimeSeconds,
                        sessions[idx].wordCount ?? 0,
                        sessions[idx].transcriptionEstimatedCostUSD ?? 0
                    ))
                    if sessions[idx].automation?.status == .transcribing {
                        continueAutomationAfterTranscription(sessionId: sessionId)
                    }
                }
            } catch {
                await MainActor.run {
                    if let idx = sessions.firstIndex(where: { $0.id == sessionId }) {
                        sessions[idx].status = .failed(error.localizedDescription)
                        saveMetadata(for: sessions[idx])
                    }
                    currentTranscribingSessionId = nil
                    transcriptionProgress = nil
                    if sessions.first(where: { $0.id == sessionId })?.automation?.status == .transcribing {
                        markAutomationFailed(
                            sessionId: sessionId,
                            status: .transcriptionFailed,
                            message: error.localizedDescription
                        )
                    }
                }
            }
        }
    }

    private func publicTranscriptionProviderName(for providerId: String) -> String {
        switch providerId {
        case TranscriptionProviderDescriptor.openRouter.id:
            return TranscriptionProviderDescriptor.openRouter.displayName
        case TranscriptionProviderDescriptor.elevenLabs.id:
            return TranscriptionProviderDescriptor.elevenLabs.displayName
        default:
            return TranscriptionProviderDescriptor.openAI.displayName
        }
    }

    private nonisolated static func makePublicSessionBatchPlan(
        audioURL: URL,
        durationSeconds: Double,
        targetBatchCount: Int?
    ) async throws -> PublicSessionBatchPlan {
        let asset = AVURLAsset(url: audioURL)
        let assetDuration = CMTimeGetSeconds(asset.duration)
        let effectiveDuration = durationSeconds > 0
            ? durationSeconds
            : (assetDuration.isFinite && assetDuration > 0 ? assetDuration : 0)
        let requestedBatchCount = Self.normalizedPublicSessionBatchCount(
            targetBatchCount: targetBatchCount,
            durationSeconds: effectiveDuration
        )
        let batchDuration = publicSessionBatchDurationSeconds(
            targetBatchCount: requestedBatchCount,
            durationSeconds: effectiveDuration
        )

        guard effectiveDuration > batchDuration else {
            return PublicSessionBatchPlan(
                batches: [
                    PublicSessionAudioBatch(
                        index: 0,
                        total: 1,
                        url: audioURL,
                        durationSeconds: max(effectiveDuration, durationSeconds),
                        fileName: "session-batch-001.m4a",
                        mimeType: "audio/mp4"
                    )
                ],
                temporaryDirectory: nil
            )
        }

        let temporaryDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("AiTranscribePublicSessionBatches-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: temporaryDirectory, withIntermediateDirectories: true)

        let totalBatches = requestedBatchCount ?? Int(ceil(effectiveDuration / batchDuration))
        var batches: [PublicSessionAudioBatch] = []
        batches.reserveCapacity(totalBatches)

        for index in 0..<totalBatches {
            let startSeconds = Double(index) * batchDuration
            let remaining = max(0, effectiveDuration - startSeconds)
            let segmentDuration = min(batchDuration, remaining)
            guard segmentDuration > 0 else { continue }

            let outputURL = temporaryDirectory
                .appendingPathComponent(String(format: "session-batch-%03d.m4a", index + 1))
            try await exportPublicSessionSegment(
                asset: asset,
                startSeconds: startSeconds,
                durationSeconds: segmentDuration,
                outputURL: outputURL
            )
            batches.append(
                PublicSessionAudioBatch(
                    index: index,
                    total: totalBatches,
                    url: outputURL,
                    durationSeconds: segmentDuration,
                    fileName: outputURL.lastPathComponent,
                    mimeType: "audio/mp4"
                )
            )
        }

        guard !batches.isEmpty else {
            throw NSError(
                domain: "AiTranscribe.SessionManager",
                code: 2001,
                userInfo: [NSLocalizedDescriptionKey: "Could not create API session batches."]
            )
        }

        return PublicSessionBatchPlan(batches: batches, temporaryDirectory: temporaryDirectory)
    }

    private nonisolated static func exportPublicSessionSegment(
        asset: AVAsset,
        startSeconds: Double,
        durationSeconds: Double,
        outputURL: URL
    ) async throws {
        try? FileManager.default.removeItem(at: outputURL)

        guard let exportSession = AVAssetExportSession(asset: asset, presetName: AVAssetExportPresetAppleM4A) else {
            throw NSError(
                domain: "AiTranscribe.SessionManager",
                code: 2002,
                userInfo: [NSLocalizedDescriptionKey: "Could not create audio exporter for API session batch."]
            )
        }

        exportSession.outputURL = outputURL
        exportSession.outputFileType = .m4a
        exportSession.timeRange = CMTimeRange(
            start: CMTime(seconds: startSeconds, preferredTimescale: 600),
            duration: CMTime(seconds: durationSeconds, preferredTimescale: 600)
        )

        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            exportSession.exportAsynchronously {
                switch exportSession.status {
                case .completed:
                    continuation.resume()
                case .failed, .cancelled:
                    continuation.resume(throwing: exportSession.error ?? NSError(
                        domain: "AiTranscribe.SessionManager",
                        code: 2003,
                        userInfo: [NSLocalizedDescriptionKey: "Audio batch export failed."]
                    ))
                default:
                    continuation.resume(throwing: NSError(
                        domain: "AiTranscribe.SessionManager",
                        code: 2004,
                        userInfo: [NSLocalizedDescriptionKey: "Audio batch export ended unexpectedly."]
                    ))
                }
            }
        }
    }

    private nonisolated static func transcribePublicSessionBatches(
        _ batches: [PublicSessionAudioBatch],
        providerId: String,
        modelId: String,
        apiKey: String,
        onBatchComplete: @escaping @Sendable (_ completed: Int, _ total: Int, _ preview: String?) async -> Void
    ) async throws -> PublicSessionTranscriptionResult {
        let startedAt = Date()
        let maxConcurrent = maxPublicSessionConcurrentUploads()
        var results = Array<PublicSessionBatchResult?>(repeating: nil, count: batches.count)
        var nextIndex = 0
        var completed = 0

        try await withThrowingTaskGroup(of: PublicSessionBatchResult.self) { group in
            func scheduleNext() {
                guard nextIndex < batches.count else { return }

                let batch = batches[nextIndex]
                nextIndex += 1
                group.addTask {
                    try await Self.transcribePublicSessionBatch(
                        batch,
                        providerId: providerId,
                        modelId: modelId,
                        apiKey: apiKey
                    )
                }
            }

            for _ in 0..<min(maxConcurrent, batches.count) {
                scheduleNext()
            }

            while let result = try await group.next() {
                completed += 1
                results[result.index] = result
                let preview = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
                await onBatchComplete(completed, batches.count, preview.isEmpty ? nil : preview)
                scheduleNext()
            }
        }

        let orderedResults = try results.enumerated().map { index, result in
            guard let result else {
                throw NSError(
                    domain: "AiTranscribe.SessionManager",
                    code: 2005,
                    userInfo: [NSLocalizedDescriptionKey: "API session batch \(index + 1) did not complete."]
                )
            }
            return result
        }

        let combinedText = orderedResults
            .map { $0.text.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .joined(separator: "\n\n")
        let requestCost = orderedResults.reduce(Double?.none) { partial, result in
            guard partial != nil || result.estimatedCostUSD != nil else { return nil }
            return (partial ?? 0) + (result.estimatedCostUSD ?? 0)
        }

        return PublicSessionTranscriptionResult(
            text: combinedText,
            processingTimeSeconds: Date().timeIntervalSince(startedAt),
            estimatedCostUSD: requestCost,
            batchCount: orderedResults.count
        )
    }

    private nonisolated static func transcribePublicSessionBatch(
        _ batch: PublicSessionAudioBatch,
        providerId: String,
        modelId: String,
        apiKey: String
    ) async throws -> PublicSessionBatchResult {
        let audioData = try Data(contentsOf: batch.url)
        let startedAt = Date()

        if providerId == "openrouter" {
            let response = try await OpenRouterTranscriptionClient().transcribe(
                audioData: audioData,
                modelId: modelId,
                apiKey: apiKey,
                format: "m4a"
            )
            return PublicSessionBatchResult(
                index: batch.index,
                text: response.text,
                processingTimeSeconds: Date().timeIntervalSince(startedAt),
                estimatedCostUSD: response.usage?.cost
            )
        }

        if providerId == "elevenlabs" {
            let response = try await ElevenLabsTranscriptionClient().transcribe(
                audioData: audioData,
                modelId: modelId,
                apiKey: apiKey,
                fileName: batch.fileName,
                mimeType: batch.mimeType
            )
            return PublicSessionBatchResult(
                index: batch.index,
                text: response.text,
                processingTimeSeconds: Date().timeIntervalSince(startedAt),
                estimatedCostUSD: nil
            )
        }

        let response = try await OpenAITranscriptionClient().transcribe(
            audioData: audioData,
            modelId: modelId,
            apiKey: apiKey,
            fileName: batch.fileName,
            mimeType: batch.mimeType
        )
        return PublicSessionBatchResult(
            index: batch.index,
            text: response.text,
            processingTimeSeconds: Date().timeIntervalSince(startedAt),
            estimatedCostUSD: nil
        )
    }

    private nonisolated static func publicSessionBatchDurationSeconds(
        targetBatchCount: Int?,
        durationSeconds: Double
    ) -> Double {
        if let targetBatchCount, targetBatchCount > 0, durationSeconds > 0 {
            return max(1, durationSeconds / Double(targetBatchCount))
        }
        let storedValue = UserDefaults.standard.double(forKey: "publicSessionBatchDurationSeconds")
        return storedValue > 0 ? storedValue : 120
    }

    private nonisolated static func normalizedPublicSessionBatchCount(
        targetBatchCount: Int?,
        durationSeconds: Double
    ) -> Int? {
        guard durationSeconds > 0 else { return targetBatchCount.map { max(1, min($0, 30)) } }

        if let targetBatchCount, targetBatchCount > 0 {
            return max(6, min(nearestMultipleOfSix(targetBatchCount), 30))
        }

        let targetMinutesPerBatch = 15.0
        let rawCount = Int(ceil((durationSeconds / 60.0) / targetMinutesPerBatch))
        return max(6, min(roundUpToMultipleOfSix(rawCount), 30))
    }

    private nonisolated static func roundUpToMultipleOfSix(_ value: Int) -> Int {
        guard value > 0 else { return 6 }
        return Int(ceil(Double(value) / 6.0)) * 6
    }

    private nonisolated static func nearestMultipleOfSix(_ value: Int) -> Int {
        guard value > 0 else { return 6 }
        return Int(round(Double(value) / 6.0)) * 6
    }

    private nonisolated static func maxPublicSessionConcurrentUploads() -> Int {
        let storedValue = UserDefaults.standard.integer(forKey: "publicSessionConcurrentUploads")
        return max(1, min(storedValue > 0 ? storedValue : 6, 6))
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
                if sessions.first(where: { $0.id == sessionId })?.automation?.status == .summarizing {
                    markAutomationFailed(
                        sessionId: sessionId,
                        status: .summaryFailed,
                        message: error.localizedDescription
                    )
                }
            }
        }
    }

    func startPublicSummary(
        sessionId: UUID,
        providerId: String,
        modelId: String,
        modelName: String,
        presetRequests: [SummaryGenerationPresetRequest]
    ) {
        guard let index = sessions.firstIndex(where: { $0.id == sessionId }) else { return }
        guard let transcript = sessions[index].transcriptionText,
              !transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return
        }
        guard let firstPreset = presetRequests.first else { return }
        guard let apiKey = try? APIKeychainStore.apiKey(providerId: providerId),
              !apiKey.isEmpty else {
            markSummaryRunFailed(sessionId: sessionId, message: "Add \(publicSummaryProviderName(for: providerId)) API key first.")
            return
        }
        let pricingModel = APILanguageModelStore.load()
            .first(where: { $0.providerId == providerId && $0.modelId == modelId })

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
            stage: "Sending to API",
            partialText: sessions[index].summaries[firstPreset.presetId]?.text,
            memoryEstimate: nil,
            kvQuantized: nil,
            currentIndex: 1,
            totalPresets: presetRequests.count
        )
        saveMetadata(for: sessions[index])

        Task {
            var completedPresetIds: [String] = []
            do {
                for (offset, request) in presetRequests.enumerated() {
                    guard let idx = sessions.firstIndex(where: { $0.id == sessionId }) else { return }
                    var activeMetadata = sessions[idx].summaries[request.presetId]
                        ?? .empty(displayName: request.displayName, fileName: request.fileName)
                    activeMetadata.status = "generating"
                    activeMetadata.statusMessage = nil
                    sessions[idx].summaries[request.presetId] = activeMetadata
                    summaryProgress = SessionSummaryProgress(
                        activePresetId: request.presetId,
                        selectedPresetIds: selectedPresetIds,
                        completedPresetIds: completedPresetIds,
                        stage: "Generating with \(publicSummaryProviderName(for: providerId))",
                        partialText: activeMetadata.text,
                        memoryEstimate: nil,
                        kvQuantized: nil,
                        currentIndex: offset + 1,
                        totalPresets: presetRequests.count
                    )
                    saveMetadata(for: sessions[idx])

                    let result: OpenAILanguageModelResponse
                    if providerId == TranscriptionProviderDescriptor.openRouter.id {
                        result = try await OpenRouterLanguageModelClient().generateSummary(
                            transcript: transcript,
                            preset: request,
                            modelId: modelId,
                            apiKey: apiKey,
                            providerOnly: pricingModel?.openRouterProviderOnly
                        )
                    } else {
                        result = try await OpenAILanguageModelClient().generateSummary(
                            transcript: transcript,
                            preset: request,
                            modelId: modelId,
                            apiKey: apiKey
                        )
                    }

                    summaryProgress?.stage = "Streaming \(request.displayName)"
                    await streamPublicSummaryPreview(
                        result.text,
                        sessionId: sessionId,
                        presetId: request.presetId
                    )

                    guard let doneIndex = sessions.firstIndex(where: { $0.id == sessionId }) else { return }
                    var doneMetadata = sessions[doneIndex].summaries[request.presetId]
                        ?? .empty(displayName: request.displayName, fileName: request.fileName)
                    doneMetadata.hasSummary = true
                    doneMetadata.status = "complete"
                    doneMetadata.statusMessage = nil
                    doneMetadata.fileName = request.fileName
                    doneMetadata.modelId = modelId
                    doneMetadata.modelName = modelName
                    doneMetadata.presetDisplayName = request.displayName
                    doneMetadata.wordCount = result.text.split(separator: " ").count
                    doneMetadata.targetWordCount = request.targetWords
                    doneMetadata.maxOutputTokens = request.maxOutputTokens
                    doneMetadata.processingTimeSeconds = result.processingTimeSeconds
                    doneMetadata.inputTokens = result.inputTokens
                    doneMetadata.outputTokens = result.outputTokens
                    doneMetadata.estimatedCostUSD = pricingModel?.estimatedCost(
                        inputTokens: result.inputTokens,
                        outputTokens: result.outputTokens
                    )
                    doneMetadata.generatedAt = Date()
                    doneMetadata.text = result.text
                    sessions[doneIndex].summaries[request.presetId] = doneMetadata
                    saveSummaryText(for: sessions[doneIndex], fileName: request.fileName, text: result.text)
                    saveMetadata(for: sessions[doneIndex])
                    completedPresetIds.append(request.presetId)
                }

                currentSummarizingSessionId = nil
                summaryProgress = nil
                if sessions.first(where: { $0.id == sessionId })?.automation?.status == .summarizing {
                    completeAutomation(sessionId: sessionId, message: "Automatic transcription and summary complete.")
                    pumpAutomationQueue()
                }
            } catch {
                markSummaryRunFailed(sessionId: sessionId, message: error.localizedDescription)
                currentSummarizingSessionId = nil
                summaryProgress = nil
                if sessions.first(where: { $0.id == sessionId })?.automation?.status == .summarizing {
                    markAutomationFailed(
                        sessionId: sessionId,
                        status: .summaryFailed,
                        message: error.localizedDescription
                    )
                }
            }
        }
    }

    private func streamPublicSummaryPreview(
        _ text: String,
        sessionId: UUID,
        presetId: String
    ) async {
        let characters = Array(text)
        guard !characters.isEmpty else { return }

        let chunkSize = max(1, characters.count / 140)
        var index = 0
        while index < characters.count {
            guard currentSummarizingSessionId == sessionId,
                  summaryProgress?.activePresetId == presetId else {
                return
            }

            index = min(characters.count, index + chunkSize)
            let partial = String(characters.prefix(index))

            if let sessionIndex = sessions.firstIndex(where: { $0.id == sessionId }),
               var metadata = sessions[sessionIndex].summaries[presetId] {
                metadata.text = partial
                metadata.status = "generating"
                sessions[sessionIndex].summaries[presetId] = metadata
            }
            summaryProgress?.partialText = partial

            try? await Task.sleep(nanoseconds: 9_000_000)
        }
    }

    private func publicSummaryProviderName(for providerId: String) -> String {
        providerId == TranscriptionProviderDescriptor.openRouter.id
            ? TranscriptionProviderDescriptor.openRouter.displayName
            : TranscriptionProviderDescriptor.openAI.displayName
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
            if sessions[index].automation?.status == .transcribing {
                continueAutomationAfterTranscription(sessionId: sessionId)
            }

        case "error":
            sessions[index].status = .failed(event.message ?? "Unknown error")
            saveMetadata(for: sessions[index])
            currentTranscribingSessionId = nil
            transcriptionProgress = nil
            if sessions[index].automation?.status == .transcribing {
                markAutomationFailed(
                    sessionId: sessionId,
                    status: .transcriptionFailed,
                    message: event.message ?? "Unknown error"
                )
            }
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
            if sessions[index].automation?.status == .transcribing {
                markAutomationFailed(
                    sessionId: sessionId,
                    status: .transcriptionFailed,
                    message: "Automatic transcription was cancelled."
                )
            }

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
            if sessions[index].automation?.status == .summarizing {
                markAutomationFailed(
                    sessionId: sessionId,
                    status: .summaryFailed,
                    message: event.message ?? "Unknown error"
                )
            }

        case "cancelled":
            markSummaryRunCancelled(sessionId: sessionId, message: "Cancelled by user")
            currentSummarizingSessionId = nil
            summaryProgress = nil
            if sessions[index].automation?.status == .summarizing {
                markAutomationFailed(
                    sessionId: sessionId,
                    status: .summaryFailed,
                    message: "Automatic summary was cancelled."
                )
            }

        case "batch_complete":
            currentSummarizingSessionId = nil
            summaryProgress = nil
            if sessions[index].automation?.status == .summarizing {
                completeAutomation(sessionId: sessionId, message: "Automatic transcription and summary complete.")
                pumpAutomationQueue()
            }

        default:
            break
        }
    }

    // MARK: - Private Helpers

    private func saveTranscriptionText(for session: Session, text: String) {
        let transcriptionURL = getTranscriptionURL(for: session)
        try? text.write(to: transcriptionURL, atomically: true, encoding: .utf8)
    }

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
