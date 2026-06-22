import Foundation

enum PolishOutputBehavior: String, Codable, CaseIterable, Identifiable {
    case copyPolished
    case copyRaw

    var id: String { rawValue }

    var label: String {
        switch self {
        case .copyPolished: return "Copy polished"
        case .copyRaw: return "Copy raw"
        }
    }
}

struct PolishPreset: Identifiable, Codable, Equatable {
    var id: String
    var displayName: String
    var symbolName: String
    var prompt: String
    var modelProviderId: String
    var modelId: String
    var isEnabled: Bool
    var showInMenu: Bool
    var outputBehavior: PolishOutputBehavior
    var isBuiltIn: Bool
    var createdAt: Date

    var trimmedPrompt: String {
        prompt.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

struct GeneratedPolishPresetDraft: Decodable {
    let displayName: String
    let symbolName: String
    let prompt: String

    enum CodingKeys: String, CodingKey {
        case displayName = "display_name"
        case symbolName = "symbol_name"
        case prompt
    }
}

enum PolishPresetStore {
    private static let storageKey = "polishPresetsV1"

    static var builtIns: [PolishPreset] {
        [
            makeBuiltIn(
                id: "clean-grammar",
                displayName: "Clean Grammar",
                symbolName: "text.badge.checkmark",
                prompt: "Clean grammar, punctuation, and wording while preserving the speaker's intent and language. Return only the polished text."
            ),
            makeBuiltIn(
                id: "notes",
                displayName: "Notes",
                symbolName: "note.text",
                prompt: "Turn the transcript into concise, structured notes with clear headings and bullets. Preserve important details. Return only the notes."
            ),
            makeBuiltIn(
                id: "tasks",
                displayName: "Tasks",
                symbolName: "checklist",
                prompt: "Extract actionable tasks from the transcript. Include owner or deadline only if stated. Return only the task list."
            ),
            makeBuiltIn(
                id: "prompt-polish",
                displayName: "Prompt Polish",
                symbolName: "sparkles",
                prompt: "Rewrite the transcript into a clear, effective AI prompt. Preserve constraints, goals, context, and requested output format. Return only the prompt."
            )
        ]
    }

    static func load() -> [PolishPreset] {
        guard let data = UserDefaults.standard.data(forKey: storageKey),
              let presets = try? JSONDecoder().decode([PolishPreset].self, from: data),
              !presets.isEmpty else {
            return builtIns
        }
        return mergeWithBuiltIns(presets)
    }

    static func save(_ presets: [PolishPreset]) {
        guard let data = try? JSONEncoder().encode(mergeWithBuiltIns(presets)) else { return }
        UserDefaults.standard.set(data, forKey: storageKey)
    }

    static func upsert(_ preset: PolishPreset, in presets: [PolishPreset]) -> [PolishPreset] {
        var updated = presets
        if let index = updated.firstIndex(where: { $0.id == preset.id }) {
            updated[index] = preset
        } else {
            updated.append(preset)
        }
        let merged = mergeWithBuiltIns(updated)
        save(merged)
        return merged
    }

    static func delete(id: String, from presets: [PolishPreset]) -> [PolishPreset] {
        let updated = presets.filter { $0.id != id || $0.isBuiltIn }
        save(updated)
        return load()
    }

    static func makeCustom(
        displayName: String,
        symbolName: String,
        prompt: String,
        modelProviderId: String = TranscriptionProviderDescriptor.openAI.id,
        modelId: String,
        outputBehavior: PolishOutputBehavior = .copyPolished
    ) -> PolishPreset {
        let trimmedName = displayName.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedSymbol = symbolName.trimmingCharacters(in: .whitespacesAndNewlines)
        return PolishPreset(
            id: "polish-\(UUID().uuidString.lowercased())",
            displayName: trimmedName.isEmpty ? "Custom Polish" : trimmedName,
            symbolName: trimmedSymbol.isEmpty ? "sparkles" : trimmedSymbol,
            prompt: prompt.trimmingCharacters(in: .whitespacesAndNewlines),
            modelProviderId: modelProviderId,
            modelId: modelId,
            isEnabled: true,
            showInMenu: true,
            outputBehavior: outputBehavior,
            isBuiltIn: false,
            createdAt: Date()
        )
    }

    private static func mergeWithBuiltIns(_ presets: [PolishPreset]) -> [PolishPreset] {
        let builtInOrder = Dictionary(uniqueKeysWithValues: builtIns.enumerated().map { ($0.element.id, $0.offset) })
        var byId = Dictionary(uniqueKeysWithValues: presets.map { ($0.id, $0) })
        for builtIn in builtIns where byId[builtIn.id] == nil {
            byId[builtIn.id] = builtIn
        }
        return byId.values.sorted { lhs, rhs in
            if lhs.isBuiltIn != rhs.isBuiltIn {
                return !lhs.isBuiltIn && rhs.isBuiltIn
            }
            if lhs.isBuiltIn {
                return (builtInOrder[lhs.id] ?? Int.max) < (builtInOrder[rhs.id] ?? Int.max)
            }
            return lhs.createdAt > rhs.createdAt
        }
    }

    private static func makeBuiltIn(
        id: String,
        displayName: String,
        symbolName: String,
        prompt: String
    ) -> PolishPreset {
        PolishPreset(
            id: id,
            displayName: displayName,
            symbolName: symbolName,
            prompt: prompt,
            modelProviderId: TranscriptionProviderDescriptor.openAI.id,
            modelId: UserDefaults.standard.string(forKey: "selectedOpenAILanguageModel") ?? APILanguageModelStore.defaultModels[0].modelId,
            isEnabled: true,
            showInMenu: true,
            outputBehavior: .copyPolished,
            isBuiltIn: true,
            createdAt: Date(timeIntervalSince1970: 0)
        )
    }
}
