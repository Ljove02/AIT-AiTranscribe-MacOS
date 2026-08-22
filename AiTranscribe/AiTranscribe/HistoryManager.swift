/*
 HistoryManager.swift
 ====================

 Manages transcription history storage in JSON format with pagination.

 Data Location:
 ~/Library/Application Support/AiTranscribe/transcriptions.json

 Features:
 - JSON file storage (not UserDefaults)
 - Pagination (load 20 entries at a time)
 - Extra fields for future analysis (modelId, wordCount, wordsPerMinute)
 - Thread-safe operations
 */

import Foundation
import NaturalLanguage

/// Manages transcription history storage in JSON format
final class HistoryManager {
    
    // MARK: - Constants
    
    private let fileName = "transcriptions.json"
    private let pageSize = 20
    
    // MARK: - Properties
    
    private let fileURL: URL
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()
    
    // MARK: - Initialization
    
    init() {
        // Get the application support directory
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let appFolder = appSupport.appendingPathComponent("AiTranscribe", isDirectory: true)
        
        // Create directory if it doesn't exist
        try? FileManager.default.createDirectory(at: appFolder, withIntermediateDirectories: true)
        
        self.fileURL = appFolder.appendingPathComponent(fileName)
        
        // Configure encoder/decoder
        encoder.dateEncodingStrategy = .iso8601
        decoder.dateDecodingStrategy = .iso8601
    }
    
    // MARK: - In-Memory Store

    /// Canonical in-memory copy of all entries (newest first).
    /// Loaded from disk exactly once per session; all reads and writes go through it.
    private var cachedEntries: [TranscriptionEntry]?
    private var loadTask: Task<[TranscriptionEntry], Never>?

    /// Serial chain of background writes — each write awaits the previous one so
    /// writes can never interleave or reorder.
    private var pendingWrite: Task<Void, Never>?
    /// Whether the one-time session backup has been scheduled.
    private var didEnsureBackup = false

    // MARK: - Public Methods

    /// Load a specific page of transcriptions (newest first)
    func loadPage(page: Int) async -> [TranscriptionEntry] {
        let allEntries = await loadAllEntries()
        
        let startIndex = page * pageSize
        let endIndex = min(startIndex + pageSize, allEntries.count)
        
        guard startIndex < allEntries.count else {
            return []
        }
        
        return Array(allEntries[startIndex..<endIndex])
    }
    
    /// All entries, served from the in-memory store.
    /// The first call decodes the file once on a detached task; later calls are free.
    func loadAllEntries() async -> [TranscriptionEntry] {
        if let cachedEntries {
            return cachedEntries
        }

        if loadTask == nil {
            let fileURL = fileURL
            loadTask = Task.detached(priority: .userInitiated) {
                guard FileManager.default.fileExists(atPath: fileURL.path) else {
                    return []
                }

                do {
                    let decoder = JSONDecoder()
                    decoder.dateDecodingStrategy = .iso8601
                    let data = try Data(contentsOf: fileURL)
                    return try decoder.decode([TranscriptionEntry].self, from: data)
                } catch {
                    print("Error loading history: \(error)")
                    return []
                }
            }
        }

        let loaded = await loadTask!.value
        // A mutation may have raced ahead of us (e.g. clearAll) — never clobber it.
        if cachedEntries == nil {
            cachedEntries = loaded
        }
        return cachedEntries!
    }
    
    /// Save a new entry (adds to beginning)
    func save(_ entry: TranscriptionEntry) async {
        var entries = await loadAllEntries()
        entries.insert(entry, at: 0)
        saveEntries(entries)
    }

    /// Delete an entry by ID
    func delete(id: UUID) async {
        var entries = await loadAllEntries()
        entries.removeAll { $0.id == id }
        saveEntries(entries)
    }

    /// Delete multiple entries by IDs
    func delete(ids: Set<UUID>) async {
        var entries = await loadAllEntries()
        entries.removeAll { ids.contains($0.id) }
        saveEntries(entries)
    }

    /// Clear all history
    func clearAll() async {
        saveEntries([])
    }

    /// Wait for any in-flight background writes to finish (call before app termination).
    func flushPendingWrites() async {
        await pendingWrite?.value
    }
    
    /// Get total count of entries
    func getTotalCount() async -> Int {
        let entries = await loadAllEntries()
        return entries.count
    }
    
    /// Check if file exists
    func hasEntries() -> Bool {
        return FileManager.default.fileExists(atPath: fileURL.path)
    }
    
    /// Get all entries as data (for backup/export)
    func exportAsJSON() async -> Data? {
        let entries = await loadAllEntries()
        do {
            return try encoder.encode(entries)
        } catch {
            print("Error exporting history: \(error)")
            return nil
        }
    }
    
    // MARK: - Semantic Search (Cached Index)

    /// Pre-computed embedding vectors keyed by entry ID.
    /// Built on user request, updated incrementally on save.
    private var embeddingIndex: [UUID: [Double]] = [:]
    private var indexedEntryCache: [UUID: TranscriptionEntry] = [:]
    private(set) var isIndexed = false
    private let sentenceEmbedding = NLEmbedding.sentenceEmbedding(for: .english)

    /// Build or refresh the embedding index from all entries on disk.
    /// Call explicitly when user requests indexing.
    func buildSearchIndex() async {
        let allEntries = await loadAllEntries()

        var newCache: [UUID: TranscriptionEntry] = [:]
        for entry in allEntries { newCache[entry.id] = entry }
        indexedEntryCache = newCache

        guard sentenceEmbedding != nil else {
            // No embedding model — semantic search won't work but text search will
            isIndexed = true
            return
        }

        // Compute missing vectors off the main actor; existing vectors are reused.
        let samples: [(id: UUID, text: String)] = allEntries
            .filter { embeddingIndex[$0.id] == nil }
            .map { ($0.id, String($0.text.prefix(500))) }

        let computed = await Task.detached(priority: .userInitiated) {
            var vectors: [UUID: [Double]] = [:]
            guard let embedding = NLEmbedding.sentenceEmbedding(for: .english) else {
                return vectors
            }
            for sample in samples {
                if let vector = embedding.vector(for: sample.text) {
                    vectors[sample.id] = vector
                }
            }
            return vectors
        }.value

        // Back on main: merge against the live cache so entries indexed or removed
        // while we were computing are respected.
        var newIndex = embeddingIndex.filter { indexedEntryCache[$0.key] != nil }
        for (id, vector) in computed where indexedEntryCache[id] != nil {
            newIndex[id] = vector
        }

        embeddingIndex = newIndex
        isIndexed = true
    }

    /// Add a single entry to the index (call after saving a new entry).
    func indexEntry(_ entry: TranscriptionEntry) {
        indexedEntryCache[entry.id] = entry
        guard let embedding = sentenceEmbedding else { return }
        let sample = String(entry.text.prefix(500))
        if let vector = embedding.vector(for: sample) {
            embeddingIndex[entry.id] = vector
        }
    }

    /// Remove entries from the index.
    func removeFromIndex(ids: Set<UUID>) {
        for id in ids {
            embeddingIndex.removeValue(forKey: id)
            indexedEntryCache.removeValue(forKey: id)
        }
    }

    /// Clear the entire index.
    func clearIndex() {
        embeddingIndex.removeAll()
        indexedEntryCache.removeAll()
        isIndexed = false
    }

    /// Basic keyword search across ALL entries (no embedding needed).
    func textSearch(query: String, limit: Int = 50) async -> [TranscriptionEntry] {
        let trimmed = query.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return [] }

        let allEntries = await loadAllEntries()
        return Array(allEntries.filter {
            $0.text.localizedCaseInsensitiveContains(trimmed)
        }.prefix(limit))
    }

    /// Semantic search using the cached index. Requires `buildSearchIndex()` first.
    /// Text matches appear first, then semantic matches, deduplicated.
    func semanticSearch(query: String, limit: Int = 50) -> [TranscriptionEntry] {
        let trimmed = query.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty, isIndexed else { return [] }

        let allEntries = Array(indexedEntryCache.values)
        guard !allEntries.isEmpty else { return [] }

        // 1. Text matches (always first)
        let textMatches = allEntries.filter {
            $0.text.localizedCaseInsensitiveContains(trimmed)
        }
        let textMatchIds = Set(textMatches.map(\.id))

        // 2. Semantic matches from cached vectors — only the query vector is computed
        var semanticMatches: [TranscriptionEntry] = []

        if let embedding = sentenceEmbedding,
           let queryVector = embedding.vector(for: trimmed) {

            var scored: [(entry: TranscriptionEntry, score: Double)] = []

            for (id, entryVector) in embeddingIndex where !textMatchIds.contains(id) {
                let similarity = Self.cosineSimilarity(queryVector, entryVector)
                if similarity > 0.3, let entry = indexedEntryCache[id] {
                    scored.append((entry, similarity))
                }
            }

            scored.sort { $0.score > $1.score }
            semanticMatches = scored.prefix(limit).map(\.entry)
        }

        return Array((textMatches + semanticMatches).prefix(limit))
    }

    private static func cosineSimilarity(_ a: [Double], _ b: [Double]) -> Double {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        var dot = 0.0, magA = 0.0, magB = 0.0
        for i in a.indices {
            dot += a[i] * b[i]
            magA += a[i] * a[i]
            magB += b[i] * b[i]
        }
        let denom = sqrt(magA) * sqrt(magB)
        return denom > 0 ? dot / denom : 0
    }

    // MARK: - Private Methods

    /// Update the in-memory store and persist it to disk off the main thread.
    /// Writes are chained (each awaits the previous) so they cannot interleave.
    /// If a write fails we log and keep the in-memory state — the next successful
    /// write persists the full, current array.
    private func saveEntries(_ entries: [TranscriptionEntry]) {
        cachedEntries = entries

        let snapshot = entries
        let fileURL = fileURL
        let needsBackup = !didEnsureBackup
        didEnsureBackup = true
        let previous = pendingWrite

        pendingWrite = Task.detached(priority: .utility) {
            await previous?.value

            // One-time safety net: before the first overwrite of the session,
            // keep a copy of the current file if no backup exists yet.
            if needsBackup {
                let backupURL = fileURL.deletingLastPathComponent()
                    .appendingPathComponent("transcriptions.backup.json")
                if !FileManager.default.fileExists(atPath: backupURL.path),
                   FileManager.default.fileExists(atPath: fileURL.path) {
                    try? FileManager.default.copyItem(at: fileURL, to: backupURL)
                }
            }

            do {
                let encoder = JSONEncoder()
                encoder.dateEncodingStrategy = .iso8601
                let data = try encoder.encode(snapshot)
                try data.write(to: fileURL, options: .atomic)
            } catch {
                print("Error saving history: \(error)")
            }
        }
    }
    
    // MARK: - Migration from UserDefaults
    
    /// Migrate existing history from UserDefaults to JSON
    /// Returns true if migration was performed
    func migrateFromUserDefaults(userDefaultsKey: String = "transcriptionHistory") async -> Bool {
        guard let data = UserDefaults.standard.data(forKey: userDefaultsKey),
              let oldEntries = try? JSONDecoder().decode([TranscriptionEntry].self, from: data),
              !oldEntries.isEmpty else {
            return false
        }
        
        // Check if we already have JSON entries
        let existingEntries = await loadAllEntries()
        guard existingEntries.isEmpty else {
            return false
        }
        
        // Migrate entries
        saveEntries(oldEntries)
        
        // Remove from UserDefaults
        UserDefaults.standard.removeObject(forKey: userDefaultsKey)
        
        print("Migrated \(oldEntries.count) entries from UserDefaults to JSON")
        return true
    }
}

// MARK: - Transcription Entry

/// A single transcription entry for history
public struct TranscriptionEntry: Identifiable, Codable, Equatable {
    public let id: UUID
    let text: String
    let rawText: String?
    let polishedText: String?
    let polishPresetId: String?
    let polishPresetName: String?
    let duration: Double  // Recording duration in seconds
    let timestamp: Date
    
    // Extra fields for future analysis
    let modelId: String
    let modelName: String
    let providerId: String?
    let providerName: String?
    let providerSource: TranscriptionProviderSource?
    let workflow: TranscriptionWorkflow?
    let processingTimeSeconds: Double?
    let realtimeFactor: Double?
    let estimatedCostUSD: Double?
    let wordCount: Int
    let wordsPerMinute: Double
    let phrasesCount: Int
    let charCount: Int
    let language: String
    
    public init(
        id: UUID = UUID(),
        text: String,
        rawText: String? = nil,
        polishedText: String? = nil,
        polishPresetId: String? = nil,
        polishPresetName: String? = nil,
        duration: Double,
        timestamp: Date = Date(),
        modelId: String = "unknown",
        modelName: String = "Unknown",
        providerId: String? = nil,
        providerName: String? = nil,
        providerSource: TranscriptionProviderSource? = nil,
        workflow: TranscriptionWorkflow? = nil,
        processingTimeSeconds: Double? = nil,
        realtimeFactor: Double? = nil,
        estimatedCostUSD: Double? = nil
    ) {
        self.id = id
        self.text = text
        self.rawText = rawText
        self.polishedText = polishedText
        self.polishPresetId = polishPresetId
        self.polishPresetName = polishPresetName
        self.duration = duration
        self.timestamp = timestamp
        self.modelId = modelId
        self.modelName = modelName
        self.providerId = providerId
        self.providerName = providerName
        self.providerSource = providerSource
        self.workflow = workflow
        self.processingTimeSeconds = processingTimeSeconds
        self.realtimeFactor = realtimeFactor
        self.estimatedCostUSD = estimatedCostUSD

        // Calculate extra fields
        self.wordCount = text.split(separator: " ").count
        self.phrasesCount = text.split(whereSeparator: { $0 == "." || $0 == "!" || $0 == "?" }).count
        self.charCount = text.count
        self.wordsPerMinute = duration > 0 ? Double(self.wordCount) / (duration / 60) : 0
        self.language = TranscriptionEntry.detectLanguage(from: text)
    }
    
    private static func detectLanguage(from text: String) -> String {
        // Simple language detection based on common words
        let englishCommon = ["the", "is", "are", "was", "were", "have", "has", "been"]
        let words = text.lowercased().split(separator: " ").map(String.init)
        
        let englishCount = words.filter { englishCommon.contains($0) }.count
        let totalWords = words.count
        
        guard totalWords > 0 else { return "unknown" }
        
        let englishRatio = Double(englishCount) / Double(totalWords)
        return englishRatio > 0.1 ? "en" : "unknown"
    }
}
