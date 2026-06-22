import SwiftUI

struct PolishSettingsView: View {
    @EnvironmentObject var appState: AppState
    let hasAnimated: Bool
    let onAnimated: () -> Void

    @AppStorage("selectedOpenAILanguageModel") private var selectedOpenAILanguageModel = APILanguageModelStore.defaultModels[0].modelId
    @State private var appeared: Bool
    @State private var apiLanguageModels: [APILanguageModel] = APILanguageModelStore.load()
    @State private var editingPreset: PolishPreset?
    @State private var isCreatingPreset = false
    @State private var generatorRequest = ""
    @State private var isGeneratingPreset = false
    @State private var generationMessage: String?
    @State private var generationIsError = false
    @State private var testingPresetId: String?
    @State private var polishTestMessage: String?
    @State private var polishTestIsError = false

    init(hasAnimated: Bool, onAnimated: @escaping () -> Void) {
        self.hasAnimated = hasAnimated
        self.onAnimated = onAnimated
        _appeared = State(initialValue: hasAnimated)
    }

    var body: some View {
        ScrollView {
            VStack(spacing: 0) {
                header
                    .staggerIn(index: 0, appeared: appeared)

                generatorCard
                    .staggerIn(index: 1, appeared: appeared)

                SettingsDivider()
                    .staggerIn(index: 2, appeared: appeared)

                Group {
                    SettingsSectionHeader(title: "Polishes")

                    if appState.polishPresets.isEmpty {
                        emptyState
                    } else {
                        LazyVGrid(columns: [GridItem(.adaptive(minimum: 240), spacing: 12)], spacing: 12) {
                            ForEach(appState.polishPresets) { preset in
                                PolishPresetCard(
                                    preset: preset,
                                    isSelected: appState.selectedPolishPresetId == preset.id,
                                    isTesting: testingPresetId == preset.id,
                                    onSelect: {
                                        appState.setSelectedPolishPreset(appState.selectedPolishPresetId == preset.id ? nil : preset.id)
                                    },
                                    onToggleMenu: {
                                        updatePreset(preset) { $0.showInMenu.toggle() }
                                    },
                                    onToggleEnabled: {
                                        updatePreset(preset) { $0.isEnabled.toggle() }
                                    },
                                    onEdit: {
                                        editingPreset = preset
                                    },
                                    onDelete: {
                                        appState.deletePolishPreset(id: preset.id)
                                    },
                                    onTest: {
                                        Task { await testPreset(preset) }
                                    }
                                )
                            }
                        }
                    }

                    if let polishTestMessage {
                        HStack(alignment: .top, spacing: 8) {
                            Image(systemName: polishTestIsError ? "exclamationmark.triangle.fill" : "checkmark.circle.fill")
                                .foregroundStyle(polishTestIsError ? .red : .green)
                            Text(polishTestMessage)
                                .font(.system(size: 11, weight: .medium))
                                .foregroundStyle(polishTestIsError ? .red : .secondary)
                                .lineLimit(3)
                            Spacer()
                        }
                        .padding(10)
                        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                        .padding(.top, 10)
                    }
                }
                .staggerIn(index: 3, appeared: appeared)

                Spacer(minLength: 24)
            }
            .padding(.horizontal, 28)
            .padding(.top, 8)
        }
        .scrollIndicators(.automatic)
        .onAppear {
            apiLanguageModels = APILanguageModelStore.load()
            appState.reloadPolishPresets()
        }
        .task(id: "stagger") {
            guard !hasAnimated else { return }
            try? await Task.sleep(for: .milliseconds(80))
            appeared = true
            onAnimated()
        }
        .sheet(isPresented: $isCreatingPreset) {
            PolishPresetEditorSheet(
                preset: nil,
                apiLanguageModels: apiLanguageModels,
                selectedModelId: selectedOpenAILanguageModel
            ) { preset in
                appState.upsertPolishPreset(preset)
            }
        }
        .sheet(item: $editingPreset) { preset in
            PolishPresetEditorSheet(
                preset: preset,
                apiLanguageModels: apiLanguageModels,
                selectedModelId: selectedOpenAILanguageModel
            ) { updated in
                appState.upsertPolishPreset(updated)
            }
        }
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 3) {
                Text("Polish")
                    .font(.system(size: 22, weight: .bold, design: .rounded))
                Text("\(appState.polishPresets.count) presets")
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
            }

            Spacer()

            Button {
                isCreatingPreset = true
            } label: {
                Label("New Polish", systemImage: "plus")
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
        }
    }

    private var generatorCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: "wand.and.stars")
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(.blue)
                    .frame(width: 28, height: 28)
                    .background(Color.blue.opacity(0.12), in: Circle())

                VStack(alignment: .leading, spacing: 4) {
                    Text("Generate a Polish with LLM")
                        .font(.system(size: 14, weight: .semibold))
                    Text("Describe what the polish should do. AiTranscribe will draft the name, icon, and prompt using your selected public LLM.")
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                }

                Spacer()
            }

            TextField("Example: Make this transcript into a clear prompt for another AI model", text: $generatorRequest, axis: .vertical)
                .textFieldStyle(.plain)
                .font(.system(size: 12))
                .lineLimit(2...4)
                .padding(10)
                .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 10, style: .continuous))

            HStack(spacing: 10) {
                if let selectedModel = selectedLanguageModel {
                    Label(selectedModel.displayName, systemImage: "brain.head.profile")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(.secondary)
                } else {
                    Label("Add a public LLM first", systemImage: "exclamationmark.triangle")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(.orange)
                }

                Spacer()

                if let generationMessage {
                    Text(generationMessage)
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(generationIsError ? .red : .secondary)
                        .lineLimit(1)
                }

                Button {
                    Task { await generatePreset() }
                } label: {
                    if isGeneratingPreset {
                        ProgressView()
                            .scaleEffect(0.7)
                            .frame(width: 88, height: 26)
                    } else {
                        Text("Generate")
                            .font(.system(size: 11, weight: .semibold))
                            .frame(width: 88, height: 26)
                    }
                }
                .buttonStyle(.plain)
                .foregroundStyle(.white)
                .background(canGeneratePreset ? Color.blue : Color.gray.opacity(0.35), in: Capsule())
                .overlay(
                    Capsule()
                        .strokeBorder(.white.opacity(canGeneratePreset ? 0.16 : 0.06), lineWidth: 0.8)
                )
                .disabled(!canGeneratePreset)
            }
        }
        .padding(14)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(.white.opacity(0.08), lineWidth: 0.8)
        )
        .padding(.top, 18)
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "sparkles")
                .font(.system(size: 22, weight: .semibold))
                .foregroundStyle(.secondary)
            Text("No polish presets yet")
                .font(.system(size: 13, weight: .semibold))
            Text("Create one manually or generate it with a public LLM.")
                .font(.system(size: 11))
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(30)
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
    }

    private var canGeneratePreset: Bool {
        guard !generatorRequest.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              let selectedLanguageModel,
              !isGeneratingPreset else {
            return false
        }
        return APIKeychainStore.hasAPIKey(providerId: selectedLanguageModel.providerId)
    }

    private var selectedLanguageModel: APILanguageModel? {
        let models = apiLanguageModels.isEmpty ? APILanguageModelStore.load() : apiLanguageModels
        let selected = models.first(where: { $0.modelId == selectedOpenAILanguageModel }) ?? models.first
        return selected.map(routedLanguageModel)
    }

    private func routedLanguageModel(_ model: APILanguageModel) -> APILanguageModel {
        guard model.providerId == TranscriptionProviderDescriptor.openAI.id,
              model.modelId.contains("/"),
              APIKeychainStore.hasAPIKey(providerId: TranscriptionProviderDescriptor.openRouter.id) else {
            return model
        }

        return APILanguageModel(
            id: model.id,
            providerId: TranscriptionProviderDescriptor.openRouter.id,
            providerName: TranscriptionProviderDescriptor.openRouter.displayName,
            modelId: model.modelId,
            displayName: model.displayName,
            inputCostPerMillionTokensUSD: model.inputCostPerMillionTokensUSD,
            outputCostPerMillionTokensUSD: model.outputCostPerMillionTokensUSD,
            openRouterProviderOnly: model.openRouterProviderOnly,
            createdAt: model.createdAt
        )
    }

    private func updatePreset(_ preset: PolishPreset, mutate: (inout PolishPreset) -> Void) {
        var copy = preset
        mutate(&copy)
        appState.upsertPolishPreset(copy)
    }

    private func generatePreset() async {
        let request = generatorRequest.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !request.isEmpty else { return }
        isGeneratingPreset = true
        generationMessage = nil
        defer { isGeneratingPreset = false }

        do {
            guard let selectedLanguageModel else {
                throw OpenAILanguageModelError.missingAPIKey
            }
            guard let apiKey = try APIKeychainStore.apiKey(providerId: selectedLanguageModel.providerId),
                  !apiKey.isEmpty else {
                if selectedLanguageModel.providerId == TranscriptionProviderDescriptor.openRouter.id {
                    throw OpenRouterLanguageModelError.missingAPIKey
                }
                throw OpenAILanguageModelError.missingAPIKey
            }
            let draft: GeneratedPolishPresetDraft
            if selectedLanguageModel.providerId == TranscriptionProviderDescriptor.openRouter.id {
                draft = try await OpenRouterLanguageModelClient().generatePolishPresetDraft(
                    request: request,
                    modelId: selectedLanguageModel.modelId,
                    apiKey: apiKey,
                    providerOnly: selectedLanguageModel.openRouterProviderOnly
                )
            } else {
                draft = try await OpenAILanguageModelClient().generatePolishPresetDraft(
                    request: request,
                    modelId: selectedLanguageModel.modelId,
                    apiKey: apiKey
                )
            }
            let preset = PolishPresetStore.makeCustom(
                displayName: draft.displayName,
                symbolName: draft.symbolName,
                prompt: draft.prompt,
                modelProviderId: selectedLanguageModel.providerId,
                modelId: selectedLanguageModel.modelId
            )
            appState.upsertPolishPreset(preset)
            appState.setSelectedPolishPreset(preset.id)
            generatorRequest = ""
            generationMessage = "Created \(preset.displayName)"
            generationIsError = false
        } catch {
            generationMessage = error.localizedDescription
            generationIsError = true
        }
    }

    private func testPreset(_ preset: PolishPreset) async {
        testingPresetId = preset.id
        polishTestMessage = nil
        defer { testingPresetId = nil }

        do {
            let output = try await appState.testPolishPreset(preset)
            polishTestMessage = "\(preset.displayName) works. Sample output: \(output)"
            polishTestIsError = false
        } catch {
            polishTestMessage = "\(preset.displayName) failed: \(error.localizedDescription)"
            polishTestIsError = true
        }
    }
}

private struct PolishPresetCard: View {
    let preset: PolishPreset
    let isSelected: Bool
    let isTesting: Bool
    let onSelect: () -> Void
    let onToggleMenu: () -> Void
    let onToggleEnabled: () -> Void
    let onEdit: () -> Void
    let onDelete: () -> Void
    let onTest: () -> Void

    @State private var showingDeleteConfirmation = false
    @State private var isHovered = false

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: preset.symbolName)
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(isSelected ? .white : .blue)
                    .frame(width: 32, height: 32)
                    .background(isSelected ? Color.blue : Color.blue.opacity(0.12), in: RoundedRectangle(cornerRadius: 9, style: .continuous))

                VStack(alignment: .leading, spacing: 3) {
                    Text(preset.displayName)
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(.primary)
                        .lineLimit(1)
                    Text(preset.modelId)
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.tertiary)
                        .lineLimit(1)
                }

                Spacer()

                if isSelected {
                    Text("Selected")
                        .font(.system(size: 9, weight: .bold))
                        .foregroundStyle(.green)
                        .padding(.horizontal, 7)
                        .padding(.vertical, 3)
                        .background(Color.green.opacity(0.12), in: Capsule())
                }
            }

            Text(preset.prompt)
                .font(.system(size: 11))
                .foregroundStyle(.secondary)
                .lineLimit(3)
                .frame(maxWidth: .infinity, alignment: .leading)

            HStack(spacing: 8) {
                Button(isSelected ? "Use Raw" : "Use Polish") {
                    onSelect()
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)

                Button {
                    onToggleMenu()
                } label: {
                    Image(systemName: preset.showInMenu ? "menubar.rectangle" : "menubar.dock.rectangle")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .help(preset.showInMenu ? "Shown in menu" : "Hidden from menu")

                Button {
                    onToggleEnabled()
                } label: {
                    Image(systemName: preset.isEnabled ? "checkmark.circle" : "pause.circle")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .help(preset.isEnabled ? "Enabled" : "Disabled")

                Spacer()

                Button {
                    onTest()
                } label: {
                    if isTesting {
                        ProgressView()
                            .scaleEffect(0.6)
                            .frame(width: 14, height: 14)
                    } else {
                        Image(systemName: "play.circle")
                    }
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(!preset.isEnabled || isTesting)
                .help("Test this Polish with a sample transcript")

                Button {
                    onEdit()
                } label: {
                    Image(systemName: "pencil")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)

                Button {
                    showingDeleteConfirmation = true
                } label: {
                    Image(systemName: "trash")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(preset.isBuiltIn)
            }
        }
        .padding(14)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(isSelected ? Color.green.opacity(0.45) : Color.white.opacity(isHovered ? 0.14 : 0.06), lineWidth: 0.8)
        )
        .scaleEffect(isHovered ? 1.01 : 1)
        .animation(.easeOut(duration: 0.16), value: isHovered)
        .onHover { isHovered = $0 }
        .confirmationDialog("Delete Polish?", isPresented: $showingDeleteConfirmation) {
            Button("Delete", role: .destructive) { onDelete() }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Built-in presets cannot be deleted. Custom presets are removed from the menu and settings.")
        }
    }
}

private struct PolishPresetEditorSheet: View {
    let preset: PolishPreset?
    let apiLanguageModels: [APILanguageModel]
    let selectedModelId: String
    let onSave: (PolishPreset) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var displayName: String
    @State private var symbolName: String
    @State private var prompt: String
    @State private var modelId: String
    @State private var showInMenu: Bool
    @State private var isEnabled: Bool
    @State private var outputBehavior: PolishOutputBehavior
    @State private var showingIconPicker = false

    init(
        preset: PolishPreset?,
        apiLanguageModels: [APILanguageModel],
        selectedModelId: String,
        onSave: @escaping (PolishPreset) -> Void
    ) {
        self.preset = preset
        self.apiLanguageModels = apiLanguageModels
        self.selectedModelId = selectedModelId
        self.onSave = onSave
        _displayName = State(initialValue: preset?.displayName ?? "")
        _symbolName = State(initialValue: preset?.symbolName ?? "sparkles")
        _prompt = State(initialValue: preset?.prompt ?? "")
        _modelId = State(initialValue: preset?.modelId ?? selectedModelId)
        _showInMenu = State(initialValue: preset?.showInMenu ?? true)
        _isEnabled = State(initialValue: preset?.isEnabled ?? true)
        _outputBehavior = State(initialValue: preset?.outputBehavior ?? .copyPolished)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            VStack(alignment: .leading, spacing: 4) {
                Text(preset == nil ? "New Polish" : "Edit Polish")
                    .font(.system(size: 20, weight: .bold, design: .rounded))
                Text("A Polish transforms a finished transcript using the selected public LLM.")
                    .font(.system(size: 12))
                    .foregroundStyle(.secondary)
            }

            formRow(label: "Name") {
                TextField("Prompt Polish", text: $displayName)
                    .textFieldStyle(.roundedBorder)
            }

            formRow(label: "Icon") {
                Button {
                    showingIconPicker.toggle()
                } label: {
                    HStack(spacing: 10) {
                        Image(systemName: symbolName.isEmpty ? "sparkles" : symbolName)
                            .font(.system(size: 18, weight: .semibold))
                            .foregroundStyle(.blue)
                            .frame(width: 42, height: 42)
                            .background(Color.blue.opacity(0.12), in: RoundedRectangle(cornerRadius: 12, style: .continuous))

                        VStack(alignment: .leading, spacing: 2) {
                            Text(symbolName.isEmpty ? "sparkles" : symbolName)
                                .font(.system(size: 12, weight: .semibold, design: .monospaced))
                                .foregroundStyle(.primary)
                            Text("Click to choose from icon grid")
                                .font(.system(size: 10))
                                .foregroundStyle(.secondary)
                        }

                        Spacer()

                        Image(systemName: "chevron.down")
                            .font(.system(size: 10, weight: .semibold))
                            .foregroundStyle(.secondary)
                    }
                    .padding(8)
                    .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                    .overlay(
                        RoundedRectangle(cornerRadius: 12, style: .continuous)
                            .strokeBorder(.white.opacity(0.08), lineWidth: 0.8)
                    )
                }
                .buttonStyle(.plain)
                .popover(isPresented: $showingIconPicker, arrowEdge: .bottom) {
                    PolishIconGridPicker(selectedSymbol: $symbolName) {
                        showingIconPicker = false
                    }
                }
            }

            formRow(label: "LLM") {
                Picker("", selection: $modelId) {
                    ForEach(apiLanguageModels) { model in
                        Text("\(model.displayName) · \(model.providerName)").tag(model.modelId)
                    }
                }
                .labelsHidden()
                .frame(maxWidth: .infinity, alignment: .leading)
            }

            formRow(label: "Prompt") {
                TextEditor(text: $prompt)
                    .font(.system(size: 12))
                    .frame(height: 150)
                    .scrollContentBackground(.hidden)
                    .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
            }

            HStack(spacing: 18) {
                Toggle("Show in menu", isOn: $showInMenu)
                    .toggleStyle(.switch)
                Toggle("Enabled", isOn: $isEnabled)
                    .toggleStyle(.switch)
                Picker("", selection: $outputBehavior) {
                    ForEach(PolishOutputBehavior.allCases) { behavior in
                        Text(behavior.label).tag(behavior)
                    }
                }
                .labelsHidden()
            }
            .font(.system(size: 12))

            HStack {
                Button("Cancel") { dismiss() }
                    .buttonStyle(.bordered)

                Spacer()

                Button("Save") {
                    save()
                }
                .buttonStyle(.borderedProminent)
                .disabled(!canSave)
            }
        }
        .padding(24)
        .frame(width: 540)
    }

    private var canSave: Bool {
        !displayName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty &&
        !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty &&
        !modelId.isEmpty
    }

    private func formRow<Content: View>(label: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(label.uppercased())
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(.tertiary)
                .tracking(0.7)
            content()
        }
    }

    private func save() {
        let selectedModel = apiLanguageModels.first { $0.modelId == modelId }
        if var preset {
            preset.displayName = displayName.trimmingCharacters(in: .whitespacesAndNewlines)
            preset.symbolName = symbolName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "sparkles" : symbolName.trimmingCharacters(in: .whitespacesAndNewlines)
            preset.prompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
            preset.modelId = modelId
            preset.modelProviderId = selectedModel?.providerId ?? preset.modelProviderId
            preset.showInMenu = showInMenu
            preset.isEnabled = isEnabled
            preset.outputBehavior = outputBehavior
            onSave(preset)
        } else {
            var preset = PolishPresetStore.makeCustom(
                displayName: displayName,
                symbolName: symbolName,
                prompt: prompt,
                modelId: modelId,
                outputBehavior: outputBehavior
            )
            preset.modelProviderId = selectedModel?.providerId ?? preset.modelProviderId
            preset.showInMenu = showInMenu
            preset.isEnabled = isEnabled
            onSave(preset)
        }
        dismiss()
    }
}

private struct PolishIconGridPicker: View {
    @Binding var selectedSymbol: String
    let onPick: () -> Void

    private let columns = Array(repeating: GridItem(.fixed(58), spacing: 8), count: 5)

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Choose Icon")
                        .font(.system(size: 14, weight: .semibold, design: .rounded))
                    Text("Used in Settings and the menu Polish buttons.")
                        .font(.system(size: 10))
                        .foregroundStyle(.secondary)
                }

                Spacer()

                Button {
                    onPick()
                } label: {
                    Image(systemName: "xmark")
                        .font(.system(size: 11, weight: .semibold))
                        .frame(width: 24, height: 24)
                }
                .buttonStyle(.plain)
            }

            ScrollView {
                LazyVGrid(columns: columns, spacing: 8) {
                    ForEach(Self.symbols, id: \.self) { symbol in
                        Button {
                            selectedSymbol = symbol
                            onPick()
                        } label: {
                            VStack(spacing: 5) {
                                Image(systemName: symbol)
                                    .font(.system(size: 20, weight: .medium))
                                    .foregroundStyle(selectedSymbol == symbol ? .white : .primary)
                                    .frame(width: 46, height: 34)

                                Text(symbol)
                                    .font(.system(size: 7, weight: .medium, design: .rounded))
                                    .foregroundStyle(selectedSymbol == symbol ? .white.opacity(0.85) : .secondary)
                                    .lineLimit(2)
                                    .multilineTextAlignment(.center)
                                    .frame(width: 52, height: 18, alignment: .top)
                            }
                            .padding(.vertical, 7)
                            .frame(width: 58, height: 72)
                            .background(
                                selectedSymbol == symbol ? Color.blue : Color.primary.opacity(0.045),
                                in: RoundedRectangle(cornerRadius: 10, style: .continuous)
                            )
                            .overlay(
                                RoundedRectangle(cornerRadius: 10, style: .continuous)
                                    .strokeBorder(selectedSymbol == symbol ? Color.blue.opacity(0.7) : Color.white.opacity(0.08), lineWidth: 0.8)
                            )
                        }
                        .buttonStyle(.plain)
                        .help(symbol)
                    }
                }
                .padding(.vertical, 2)
            }
            .frame(height: 380)
        }
        .padding(14)
        .frame(width: 340)
    }

    private static let symbols: [String] = [
        "sparkles", "wand.and.stars", "text.badge.checkmark", "text.alignleft", "text.quote",
        "note.text", "list.bullet.rectangle", "checklist", "checkmark.seal", "target",
        "brain.head.profile", "lightbulb", "bolt", "flame", "leaf",
        "pencil", "pencil.line", "pencil.and.outline", "highlighter", "eraser",
        "doc.text", "doc.richtext", "doc.badge.gearshape", "doc.on.doc", "clipboard",
        "quote.bubble", "bubble.left.and.text.bubble.right", "message", "envelope", "paperplane",
        "tray.full", "archivebox", "folder", "tag", "bookmark",
        "slider.horizontal.3", "switch.2", "gearshape", "wrench.and.screwdriver", "hammer",
        "clock", "timer", "calendar", "flag", "pin",
        "person.text.rectangle", "person.crop.circle.badge.checkmark", "signature", "building.columns", "briefcase",
        "chart.bar", "chart.line.uptrend.xyaxis", "sum", "number", "percent",
        "globe", "translate", "character.book.closed", "book", "graduationcap",
        "lock.shield", "shield", "eye", "eye.slash", "key",
        "mic", "waveform", "waveform.badge.magnifyingglass", "headphones", "speaker.wave.2",
        "arrow.triangle.2.circlepath", "arrow.up.doc", "arrow.down.doc", "square.and.arrow.up", "square.and.arrow.down"
    ]
}
