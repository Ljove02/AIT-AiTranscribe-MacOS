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
    
    /// Load all entries (use sparingly - for migration only)
    func loadAllEntries() async -> [TranscriptionEntry] {
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            return []
        }
        
        do {
            let data = try Data(contentsOf: fileURL)
            let entries = try decoder.decode([TranscriptionEntry].self, from: data)
            return entries
        } catch {
            print("Error loading history: \(error)")
            return []
        }
    }
    
    /// Save a new entry (adds to beginning)
    func save(_ entry: TranscriptionEntry) async {
        var entries = await loadAllEntries()
        entries.insert(entry, at: 0)
        await saveEntries(entries)
    }
    
    /// Delete an entry by ID
    func delete(id: UUID) async {
        var entries = await loadAllEntries()
        entries.removeAll { $0.id == id }
        await saveEntries(entries)
    }
    
    /// Delete multiple entries by IDs
    func delete(ids: Set<UUID>) async {
        var entries = await loadAllEntries()
        entries.removeAll { ids.contains($0.id) }
        await saveEntries(entries)
    }
    
    /// Clear all history
    func clearAll() async {
        await saveEntries([])
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
    
    // MARK: - Private Methods
    
    private func saveEntries(_ entries: [TranscriptionEntry]) async {
        do {
            let data = try encoder.encode(entries)
            try data.write(to: fileURL, options: .atomic)
        } catch {
            print("Error saving history: \(error)")
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
        await saveEntries(oldEntries)
        
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
    let duration: Double  // Recording duration in seconds
    let timestamp: Date
    
    // Extra fields for future analysis
    let modelId: String
    let modelName: String
    let wordCount: Int
    let wordsPerMinute: Double
    let phrasesCount: Int
    let charCount: Int
    let language: String
    
    public init(
        id: UUID = UUID(),
        text: String,
        duration: Double,
        timestamp: Date = Date(),
        modelId: String = "unknown",
        modelName: String = "Unknown"
    ) {
        self.id = id
        self.text = text
        self.duration = duration
        self.timestamp = timestamp
        self.modelId = modelId
        self.modelName = modelName
        
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
