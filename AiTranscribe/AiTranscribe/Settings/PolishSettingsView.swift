import SwiftUI

struct PolishSettingsView: View {
    @EnvironmentObject var appState: AppState
    let hasAnimated: Bool
    let onAnimated: () -> Void

    @AppStorage("selectedOpenAILanguageModel") private var selectedOpenAILanguageModel = APILanguageModelStore.defaultModels[0].modelId
    @AppStorage("selectedTranscriptionProviderId") private var selectedTranscriptionProviderId = TranscriptionProviderDescriptor.localBackend.id
    @State private var appeared: Bool
    @State private var apiLanguageModels: [APILanguageModel] = APILanguageModelStore.load()
    @State private var editingPreset: PolishPreset?
    @State private var isCreatingPreset = false
    @State private var isImportingPreset = false
    @State private var generatorRequest = ""
    @State private var isGeneratingPreset = false
    @State private var generationMessage: String?
    @State private var generationIsError = false
    @State private var clarificationBaseRequest = ""
    @State private var clarificationReason: String?
    @State private var clarificationQuestions: [GeneratedPolishClarificationQuestion] = []
    @State private var clarificationAnswers: [String: String] = [:]
    @State private var generatorPhaseText: String?
    @State private var reviewingDraft: PolishGeneratorDraftPackage?
    @State private var aiEditTargetPresetId: String?
    @AppStorage("polishGeneratorModelId") private var polishGeneratorModelId = ""
    @AppStorage("polishGeneratorLocalModelId") private var polishGeneratorLocalModelId = ""
    @State private var testingPresetId: String?
    @State private var polishTestMessage: String?
    @State private var polishTestIsError = false
    @State private var isManagingRoutes = false

    init(hasAnimated: Bool, onAnimated: @escaping () -> Void) {
        self.hasAnimated = hasAnimated
        self.onAnimated = onAnimated
        _appeared = State(initialValue: hasAnimated)
    }

    var body: some View {
        VStack(spacing: 0) {
            if isCreatingPreset {
                PolishPresetDetailView(
                    preset: nil,
                    apiLanguageModels: apiLanguageModels,
                    localLanguageModels: appState.availableSummaryModels,
                    selectedModelId: selectedOpenAILanguageModel,
                    selectedLocalModelId: appState.preferredSummaryModelId,
                    activeModelSource: activePolishModelSource,
                    onBack: {
                        isCreatingPreset = false
                    },
                    onSave: { preset in
                        appState.upsertPolishPreset(preset)
                        isCreatingPreset = false
                    }
                )
            } else if let reviewingDraft {
                PolishDraftReviewView(
                    package: reviewingDraft,
                    targetModelName: generatorUsesLocalMode
                        ? generatorLocalModel.map { "\($0.displayName) · Local" }
                        : generatorLanguageModel?.displayName,
                    onDiscard: {
                        self.reviewingDraft = nil
                        aiEditTargetPresetId = nil
                        generationMessage = "Draft discarded"
                        generationIsError = false
                    },
                    onSave: { finalPackage in
                        if let targetId = aiEditTargetPresetId,
                           var updated = appState.polishPresets.first(where: { $0.id == targetId }) {
                            // AI edit of an existing Polish: apply the drafted
                            // fields, keep everything else (shortcut, model,
                            // enabled state) as it was.
                            updated.displayName = finalPackage.displayName
                            updated.symbolName = finalPackage.symbolName
                            updated.prompt = finalPackage.prompt
                            updated.inputModules = finalPackage.inputModules
                            updated.outputModules = finalPackage.outputModules
                            updated.pasteBehavior = finalPackage.pasteBehavior
                            appState.upsertPolishPreset(updated)
                            generationMessage = "Updated \(updated.displayName)"
                        } else {
                            var preset = PolishPresetStore.makeCustom(
                                displayName: finalPackage.displayName,
                                symbolName: finalPackage.symbolName,
                                prompt: finalPackage.prompt,
                                modelProviderId: generatorLanguageModel?.providerId ?? TranscriptionProviderDescriptor.openAI.id,
                                modelId: generatorLanguageModel?.modelId ?? selectedOpenAILanguageModel,
                                outputBehavior: .copyPolished,
                                pasteBehavior: finalPackage.pasteBehavior,
                                inputModules: finalPackage.inputModules,
                                outputModules: finalPackage.outputModules
                            )
                            if generatorUsesLocalMode, let localModel = generatorLocalModel {
                                preset.localModelId = localModel.id
                            }
                            appState.upsertPolishPreset(preset)
                            appState.setSelectedPolishPreset(preset.id)
                            generatorRequest = ""
                            clearClarification()
                            generationMessage = "Created \(preset.displayName)"
                        }
                        self.reviewingDraft = nil
                        aiEditTargetPresetId = nil
                        generationIsError = false
                    }
                )
            } else if isImportingPreset {
                PolishImportView(
                    apiLanguageModels: apiLanguageModels,
                    selectedModelId: selectedOpenAILanguageModel,
                    onBack: {
                        isImportingPreset = false
                    },
                    onCreate: { preset in
                        appState.upsertPolishPreset(preset)
                        appState.setSelectedPolishPreset(preset.id)
                        isImportingPreset = false
                    }
                )
            } else if let editingPreset,
                      appState.polishPresets.contains(where: { $0.id == editingPreset.id }) {
                PolishPresetDetailView(
                    preset: appState.polishPresets.first(where: { $0.id == editingPreset.id }) ?? editingPreset,
                    apiLanguageModels: apiLanguageModels,
                    localLanguageModels: appState.availableSummaryModels,
                    selectedModelId: selectedOpenAILanguageModel,
                    selectedLocalModelId: appState.preferredSummaryModelId,
                    activeModelSource: activePolishModelSource,
                    onBack: {
                        self.editingPreset = nil
                    },
                    onSave: { updated in
                        appState.upsertPolishPreset(updated)
                        self.editingPreset = nil
                    },
                    onAIDraft: { package in
                        aiEditTargetPresetId = editingPreset.id
                        reviewingDraft = package
                        self.editingPreset = nil
                    }
                )
            } else if isManagingRoutes {
                PolishRoutesSettingsView(
                    activeModelSource: activePolishModelSource,
                    onBack: {
                        isManagingRoutes = false
                    }
                )
            } else {
                polishListView
            }
        }
        .onAppear {
            apiLanguageModels = APILanguageModelStore.load()
            appState.reloadPolishPresets()
            Task { await appState.fetchSummaryModels() }
        }
        .task(id: "stagger") {
            guard !hasAnimated else { return }
            try? await Task.sleep(for: .milliseconds(80))
            appeared = true
            onAnimated()
        }
    }

    private var activePolishModelSource: PolishModelSource {
        selectedTranscriptionProviderId == TranscriptionProviderDescriptor.localBackend.id ? .local : .public
    }

    private var polishListView: some View {
        ScrollView {
            VStack(spacing: 0) {
                header
                    .staggerIn(index: 0, appeared: appeared)

                generatorCard
                    .staggerIn(index: 1, appeared: appeared)

                Group {
                    SettingsSectionHeader(title: "Polishes")

                    if appState.polishPresets.isEmpty {
                        emptyState
                    } else {
                        LazyVGrid(columns: [GridItem(.flexible(), spacing: 14), GridItem(.flexible(), spacing: 14)], spacing: 14) {
                            ForEach(appState.polishPresets) { preset in
                                PolishPresetCard(
                                    preset: preset,
                                    isSelected: appState.selectedPolishPresetId == preset.id,
                                    isTesting: testingPresetId == preset.id,
                                    activeModelSource: activePolishModelSource,
                                    localLanguageModels: appState.availableSummaryModels,
                                    onSelect: {
                                        appState.setSelectedPolishPreset(appState.selectedPolishPresetId == preset.id ? nil : preset.id)
                                    },
                                    onToggleMenu: {
                                        updatePreset(preset) { $0.showInMenu.toggle() }
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
                .staggerIn(index: 2, appeared: appeared)

                Spacer(minLength: 24)
            }
            .padding(.horizontal, 28)
            .padding(.top, 8)
        }
        .scrollIndicators(.automatic)
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
                isManagingRoutes = true
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: "point.topleft.down.curvedto.point.bottomright.up")
                        .font(.system(size: 10, weight: .bold))
                    Text("Routes")
                        .font(.system(size: 11, weight: .semibold))
                }
                .foregroundStyle(.green)
                .padding(.horizontal, 13)
                .padding(.vertical, 8)
                .background(Color.green.opacity(0.12), in: Capsule())
                .overlay(
                    Capsule()
                        .strokeBorder(Color.green.opacity(0.22), lineWidth: 0.8)
                )
            }
            .buttonStyle(.plain)

            Button {
                isImportingPreset = true
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: "square.and.arrow.down")
                        .font(.system(size: 10, weight: .bold))
                    Text("Import Polish")
                        .font(.system(size: 11, weight: .semibold))
                }
                .foregroundStyle(.blue)
                .padding(.horizontal, 13)
                .padding(.vertical, 8)
                .background(Color.blue.opacity(0.12), in: Capsule())
                .overlay(
                    Capsule()
                        .strokeBorder(Color.blue.opacity(0.22), lineWidth: 0.8)
                )
            }
            .buttonStyle(.plain)

            Button {
                isCreatingPreset = true
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: "plus")
                        .font(.system(size: 10, weight: .bold))
                    Text("New Polish")
                        .font(.system(size: 11, weight: .semibold))
                }
                .foregroundStyle(.white)
                .padding(.horizontal, 13)
                .padding(.vertical, 8)
                .background(Color.accentColor, in: Capsule())
                .overlay(
                    Capsule()
                        .strokeBorder(.white.opacity(0.16), lineWidth: 0.8)
                )
            }
            .buttonStyle(.plain)
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

            if !clarificationQuestions.isEmpty {
                clarificationCard
            }

            HStack(spacing: 10) {
                if generatorUsesLocalMode {
                    if let localModel = generatorLocalModel {
                        Menu {
                            ForEach(generatorLocalModelChoices) { model in
                                Button {
                                    polishGeneratorLocalModelId = model.id
                                } label: {
                                    if model.id == localModel.id {
                                        Label("\(model.displayName) · Local", systemImage: "checkmark")
                                    } else {
                                        Text("\(model.displayName) · Local")
                                    }
                                }
                            }
                        } label: {
                            generatorModelPill(icon: "memorychip", title: localModel.displayName, tint: .green)
                        }
                        .menuStyle(.button)
                        .buttonStyle(.plain)
                        .menuIndicator(.hidden)
                        .fixedSize()
                        .help("The local LLM used only for generating Polishes. Does not change your app-wide model.")
                    } else {
                        Label("Download a local LLM first", systemImage: "exclamationmark.triangle")
                            .font(.system(size: 11, weight: .medium))
                            .foregroundStyle(.orange)
                    }
                } else if let generatorModel = generatorLanguageModel {
                    Menu {
                        ForEach(generatorModelChoices) { model in
                            Button {
                                polishGeneratorModelId = model.modelId
                            } label: {
                                if model.modelId == generatorModel.modelId {
                                    Label("\(model.displayName) · \(model.providerName)", systemImage: "checkmark")
                                } else {
                                    Text("\(model.displayName) · \(model.providerName)")
                                }
                            }
                        }
                    } label: {
                        generatorModelPill(icon: "brain.head.profile", title: generatorModel.displayName, tint: .blue)
                    }
                    .menuStyle(.button)
                    .buttonStyle(.plain)
                    .menuIndicator(.hidden)
                    .fixedSize()
                    .help("The LLM used only for generating Polishes. Does not change your app-wide LLM.")
                } else {
                    Label("Add a public LLM first", systemImage: "exclamationmark.triangle")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(.orange)
                }

                Spacer()

                if isGeneratingPreset, let generatorPhaseText {
                    HStack(spacing: 6) {
                        Image(systemName: "sparkles")
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundStyle(.blue)
                            .symbolEffect(.variableColor.iterative, options: .repeating)

                        Text(generatorPhaseText)
                            .font(.system(size: 11, weight: .medium))
                            .foregroundStyle(.secondary)
                            .id(generatorPhaseText)
                            .transition(.push(from: .bottom).combined(with: .opacity))
                    }
                    .animation(.easeInOut(duration: 0.3), value: generatorPhaseText)
                } else if let generationMessage {
                    Text(generationMessage)
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(generationIsError ? .red : .secondary)
                        .lineLimit(3)
                        .multilineTextAlignment(.trailing)
                        .fixedSize(horizontal: false, vertical: true)
                }

                Button {
                    Task { await generatePreset() }
                } label: {
                    if isGeneratingPreset {
                        ProgressView()
                            .scaleEffect(0.7)
                            .frame(width: 88, height: 26)
                    } else {
                        Text(clarificationQuestions.isEmpty ? "Generate" : "Continue")
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
        .padding(.bottom, 8)
    }

    private var clarificationCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                Image(systemName: "questionmark.bubble")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(.orange)
                Text("More detail needed")
                    .font(.system(size: 12, weight: .semibold))
                Spacer()
                Button {
                    clearClarification()
                } label: {
                    Image(systemName: "xmark")
                        .font(.system(size: 9, weight: .bold))
                        .foregroundStyle(.secondary)
                        .frame(width: 22, height: 22)
                        .background(Color.primary.opacity(0.06), in: Circle())
                }
                .buttonStyle(.plain)
            }

            if let clarificationReason, !clarificationReason.isEmpty {
                Text(clarificationReason)
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
            }

            ForEach(clarificationQuestions) { question in
                VStack(alignment: .leading, spacing: 7) {
                    Text(question.question)
                        .font(.system(size: 11, weight: .semibold))

                    if !question.suggestedAnswers.isEmpty {
                        FlowLayout(spacing: 6) {
                            ForEach(question.suggestedAnswers, id: \.self) { answer in
                                Button {
                                    clarificationAnswers[question.id] = answer
                                } label: {
                                    Text(answer)
                                        .font(.system(size: 10, weight: .semibold))
                                        .foregroundStyle(clarificationAnswers[question.id] == answer ? .white : .secondary)
                                        .padding(.horizontal, 8)
                                        .padding(.vertical, 5)
                                        .background(
                                            clarificationAnswers[question.id] == answer ? Color.orange.opacity(0.85) : Color.primary.opacity(0.06),
                                            in: Capsule()
                                        )
                                }
                                .buttonStyle(.plain)
                            }
                        }
                    }

                    TextField("Custom answer", text: Binding(
                        get: { clarificationAnswers[question.id] ?? "" },
                        set: { clarificationAnswers[question.id] = $0 }
                    ))
                    .textFieldStyle(.plain)
                    .font(.system(size: 11))
                    .padding(8)
                    .background(Color.primary.opacity(0.045), in: RoundedRectangle(cornerRadius: 9, style: .continuous))
                }
            }
        }
        .padding(12)
        .background(Color.orange.opacity(0.08), in: RoundedRectangle(cornerRadius: 13, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 13, style: .continuous)
                .strokeBorder(Color.orange.opacity(0.18), lineWidth: 0.8)
        )
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
              !isGeneratingPreset else {
            return false
        }
        if generatorUsesLocalMode {
            return generatorLocalModel != nil && appState.isServerConnected
        }
        guard let generatorLanguageModel else { return false }
        return APIKeychainStore.hasAPIKey(providerId: generatorLanguageModel.providerId)
    }

    private var selectedLanguageModel: APILanguageModel? {
        let models = apiLanguageModels.isEmpty ? APILanguageModelStore.load() : apiLanguageModels
        let selected = models.first(where: { $0.modelId == selectedOpenAILanguageModel }) ?? models.first
        return selected.map(routedLanguageModel)
    }

    /// Models the generator picker offers.
    private var generatorModelChoices: [APILanguageModel] {
        apiLanguageModels.isEmpty ? APILanguageModelStore.load() : apiLanguageModels
    }

    /// The LLM the Polish generator uses. Its own choice, scoped to this card;
    /// falls back to the app-wide selected LLM until one is picked here.
    private var generatorLanguageModel: APILanguageModel? {
        let models = generatorModelChoices
        let chosen = models.first(where: { $0.modelId == polishGeneratorModelId })
            ?? models.first(where: { $0.modelId == selectedOpenAILanguageModel })
            ?? models.first
        return chosen.map(routedLanguageModel)
    }

    /// Downloaded local models the generator can use in Local mode.
    private var generatorLocalModelChoices: [SummaryModelInfoResponse] {
        appState.availableSummaryModels.filter(\.downloaded)
    }

    /// The local LLM the generator uses in Local mode, with the same scoping:
    /// its own pick first, then the preferred summary model, then any download.
    private var generatorLocalModel: SummaryModelInfoResponse? {
        let models = generatorLocalModelChoices
        return models.first(where: { $0.id == polishGeneratorLocalModelId })
            ?? models.first(where: { $0.id == appState.preferredSummaryModelId })
            ?? models.first
    }

    private var generatorUsesLocalMode: Bool {
        activePolishModelSource == .local
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
        let request = generationRequestWithClarifications.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !request.isEmpty else { return }
        isGeneratingPreset = true
        generationMessage = nil
        generatorPhaseText = PolishGeneratorPhase.understanding.displayText
        defer {
            isGeneratingPreset = false
            generatorPhaseText = nil
        }

        do {
            let engine = try makeGeneratorEngine()
            let outcome = try await engine.run(request: request) { phase in
                generatorPhaseText = phase.displayText
            }

            switch outcome {
            case .clarification(let clarification):
                clarificationBaseRequest = clarificationBaseRequest.isEmpty ? generatorRequest.trimmingCharacters(in: .whitespacesAndNewlines) : clarificationBaseRequest
                clarificationReason = clarification.reason
                clarificationQuestions = clarification.questions
                var preservedAnswers: [String: String] = [:]
                for question in clarification.questions {
                    preservedAnswers[question.id] = clarificationAnswers[question.id] ?? ""
                }
                clarificationAnswers = preservedAnswers
                generationMessage = "Answer the questions"
                generationIsError = false
            case .draft(let package):
                reviewingDraft = package
                generationMessage = "Draft ready, \(package.passedCount)/\(package.checks.count) checks passed"
                generationIsError = false
            }
        } catch {
            generationMessage = error.localizedDescription
            generationIsError = true
        }
    }

    private func generatorModelPill(icon: String, title: String, tint: Color) -> some View {
        HStack(spacing: 7) {
            Image(systemName: icon)
                .font(.system(size: 10, weight: .semibold))
            Text(title)
                .font(.system(size: 11, weight: .semibold))
                .lineLimit(1)
            Image(systemName: "chevron.down")
                .font(.system(size: 8, weight: .bold))
        }
        .foregroundStyle(tint)
        .padding(.horizontal, 12)
        .padding(.vertical, 7)
        .background(tint.opacity(0.12), in: Capsule())
        .overlay(
            Capsule()
                .strokeBorder(tint.opacity(0.28), lineWidth: 0.8)
        )
        .contentShape(Capsule())
    }

    /// Builds the engine for whichever mode the app is in: local models through
    /// the bundled backend, API models through their clients.
    private func makeGeneratorEngine() throws -> PolishGeneratorEngine {
        if generatorUsesLocalMode {
            guard let localModel = generatorLocalModel else {
                throw APIError.serverError("Download a local LLM in Models before generating in Local mode.")
            }
            return PolishGeneratorEngine(
                llm: .local(modelId: localModel.id),
                targetModelName: localModel.displayName,
                targetSupportsVision: localModel.supportsVision
            )
        }
        guard let generatorLanguageModel else {
            throw OpenAILanguageModelError.missingAPIKey
        }
        guard let apiKey = try APIKeychainStore.apiKey(providerId: generatorLanguageModel.providerId),
              !apiKey.isEmpty else {
            if generatorLanguageModel.providerId == TranscriptionProviderDescriptor.openRouter.id {
                throw OpenRouterLanguageModelError.missingAPIKey
            }
            throw OpenAILanguageModelError.missingAPIKey
        }
        return PolishGeneratorEngine(
            llm: .api(model: generatorLanguageModel, apiKey: apiKey),
            targetModelName: generatorLanguageModel.displayName,
            targetSupportsVision: generatorLanguageModel.supportsVision
        )
    }

    private var generationRequestWithClarifications: String {
        guard !clarificationQuestions.isEmpty else {
            return generatorRequest
        }
        let base = clarificationBaseRequest.isEmpty ? generatorRequest : clarificationBaseRequest
        let answers = clarificationQuestions.map { question in
            let answer = clarificationAnswers[question.id]?.trimmingCharacters(in: .whitespacesAndNewlines)
            let resolvedAnswer = (answer?.isEmpty == false) ? (answer ?? "") : "<unspecified>"
            return """
            Q: \(question.question)
            A: \(resolvedAnswer)
            """
        }.joined(separator: "\n\n")
        return """
        Original request:
        \(base)

        Clarification answers:
        \(answers)

        Create the final Polish preset now. Do not ask more questions unless the answers still make the workflow impossible to define.
        """
    }

    private func clearClarification() {
        clarificationBaseRequest = ""
        clarificationReason = nil
        clarificationQuestions = []
        clarificationAnswers = [:]
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

private struct PolishRouteGroup: Identifiable, Equatable {
    let id: String
    var title: String
    var description: String
    var presetId: String
    var isEnabled: Bool
    var priority: Int
    var rules: [PolishRoutingRule]

    var appRules: [PolishRoutingRule] {
        rules.filter { !$0.trimmedAppName.isEmpty }
            .sorted { lhs, rhs in
                if lhs.priority != rhs.priority { return lhs.priority < rhs.priority }
                return lhs.displayAppName.localizedCaseInsensitiveCompare(rhs.displayAppName) == .orderedAscending
            }
    }
}

private struct PolishRoutesSettingsView: View {
    @EnvironmentObject private var appState: AppState

    let activeModelSource: PolishModelSource
    let onBack: () -> Void

    @AppStorage("selectedOpenAILanguageModel") private var selectedOpenAILanguageModel = APILanguageModelStore.defaultModels[0].modelId
    @State private var selectedRouteId: String?
    @State private var draftTitle = ""
    @State private var draftPresetId = ""
    @State private var draftIsEnabled = true
    @State private var appSearch = ""
    @State private var manualAppName = ""
    @State private var scannedApps: [PolishRoutableApplication] = []
    @State private var hiddenAppIds: Set<String> = []
    @State private var showHiddenApps = false
    @State private var showInstalledApps = false
    @State private var appDisplayLimit = 24
    @State private var isScanningApps = false
    @State private var isCleaningApps = false
    @State private var appScanMessage: String?
    @State private var apiLanguageModels: [APILanguageModel] = APILanguageModelStore.load()
    @State private var routeToDelete: PolishRouteGroup?

    var body: some View {
        VStack(spacing: 0) {
            header
                .padding(.horizontal, 28)
                .padding(.top, 8)
                .padding(.bottom, 12)

            Divider()
                .opacity(0.5)

            HStack(spacing: 0) {
                sidebar
                    .frame(width: 224)
                    .padding(.vertical, 14)
                    .padding(.leading, 22)
                    .padding(.trailing, 12)

                Divider()
                    .opacity(0.5)

                routeDetail
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .padding(.horizontal, 20)
                    .padding(.vertical, 16)
            }
        }
        .task {
            apiLanguageModels = APILanguageModelStore.load()
            hiddenAppIds = loadHiddenAppIds()
            if appState.polishRoutingFallback != .selectedPolish {
                appState.setPolishRoutingFallback(.selectedPolish)
            }
            syncSelectionIfNeeded()
            await rescanApplications(includeInstalled: false)
        }
        .onChange(of: appState.polishRoutingRules) {
            syncSelectionIfNeeded()
        }
        .confirmationDialog("Delete route?", isPresented: deleteDialogBinding) {
            Button("Delete Route", role: .destructive) {
                if let routeToDelete {
                    appState.deletePolishRoute(routeId: routeToDelete.id)
                    self.routeToDelete = nil
                }
            }
            Button("Cancel", role: .cancel) {
                routeToDelete = nil
            }
        } message: {
            Text("This removes the route and all apps assigned to it.")
        }
    }

    private var header: some View {
        HStack(alignment: .center, spacing: 12) {
            Button(action: onBack) {
                HStack(spacing: 6) {
                    Image(systemName: "chevron.left")
                        .font(.system(size: 10, weight: .bold))
                    Text("Polish")
                        .font(.system(size: 11, weight: .semibold))
                }
                .foregroundStyle(.secondary)
                .padding(.horizontal, 11)
                .padding(.vertical, 8)
                .background(Color.primary.opacity(0.06), in: Capsule())
            }
            .buttonStyle(.plain)

            VStack(alignment: .leading, spacing: 3) {
                Text("Routes")
                    .font(.system(size: 22, weight: .bold, design: .rounded))
                Text("Choose which Polish runs for each app.")
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
            }

            Spacer()

            routeStatePill

            Toggle("", isOn: Binding(
                get: { appState.polishRoutingEnabled },
                set: { appState.setPolishRoutingEnabled($0) }
            ))
            .toggleStyle(.switch)
            .labelsHidden()
            .controlSize(.small)

            Button {
                createRoute()
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: "plus")
                        .font(.system(size: 10, weight: .bold))
                    Text("New Route")
                        .font(.system(size: 11, weight: .semibold))
                }
                .foregroundStyle(.white)
                .padding(.horizontal, 13)
                .padding(.vertical, 8)
                .background(Color.accentColor, in: Capsule())
                .overlay(Capsule().strokeBorder(.white.opacity(0.14), lineWidth: 0.8))
            }
            .buttonStyle(.plain)
        }
    }

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Route list")
                    .font(.system(size: 11, weight: .bold))
                    .foregroundStyle(.secondary)
                Spacer()
                Text("\(routeGroups.count)")
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .foregroundStyle(.tertiary)
            }

            if routeGroups.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    Image(systemName: "point.topleft.down.curvedto.point.bottomright.up")
                        .font(.system(size: 18, weight: .semibold))
                        .foregroundStyle(.secondary)
                    Text("No routes yet")
                        .font(.system(size: 13, weight: .semibold))
                    Text("Create a route, pick a Polish, then add apps from the scan.")
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(14)
                .background(Color.primary.opacity(0.045), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            } else {
                ScrollView {
                    VStack(spacing: 8) {
                        ForEach(routeGroups) { route in
                            routeRow(route)
                        }
                    }
                }
                .scrollIndicators(.automatic)
            }

            Spacer(minLength: 0)
        }
    }

    private var routeDetail: some View {
        Group {
            if let selectedGroup {
                ScrollView {
                    VStack(alignment: .leading, spacing: 14) {
                        routeForm(selectedGroup)
                        selectedAppsSection(selectedGroup)
                        appScannerSection(selectedGroup)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .scrollIndicators(.automatic)
            } else {
                VStack(spacing: 10) {
                    Image(systemName: "point.topleft.down.curvedto.point.bottomright.up")
                        .font(.system(size: 28, weight: .semibold))
                        .foregroundStyle(.secondary)
                    Text("Select a route")
                        .font(.system(size: 15, weight: .semibold))
                    Text("Routes connect apps like Codex, Notes, Mail, or Discord to the Polish that should run automatically.")
                        .font(.system(size: 12))
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                        .frame(maxWidth: 360)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
    }

    private func routeRow(_ route: PolishRouteGroup) -> some View {
        let selected = selectedRouteId == route.id
        let preset = appState.polishPresets.first(where: { $0.id == route.presetId })

        return Button {
            saveDraftIfPossible()
            selectedRouteId = route.id
            loadDraft(from: route)
        } label: {
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 8) {
                    Circle()
                        .fill(route.isEnabled ? Color.green : Color.secondary.opacity(0.45))
                        .frame(width: 7, height: 7)
                    Text(route.title)
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(.primary)
                        .lineLimit(1)
                    Spacer()
                    Text("\(route.appRules.count)")
                        .font(.system(size: 10, weight: .bold, design: .monospaced))
                        .foregroundStyle(.secondary)
                }

                Text(preset?.displayName ?? "Choose Polish")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(selected ? .green : .secondary)
                    .lineLimit(1)
            }
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(selected ? Color.green.opacity(0.12) : Color.primary.opacity(0.045), in: RoundedRectangle(cornerRadius: 13, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 13, style: .continuous)
                    .strokeBorder(selected ? Color.green.opacity(0.28) : Color.white.opacity(0.06), lineWidth: 0.8)
            )
        }
        .buttonStyle(.plain)
        .contextMenu {
            Button("Delete Route", role: .destructive) {
                routeToDelete = route
            }
        }
    }

    private func routeForm(_ route: PolishRouteGroup) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top, spacing: 12) {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Route")
                        .font(.system(size: 15, weight: .bold))
                    Text("Name the context and choose the Polish it should use.")
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Toggle("Enabled", isOn: Binding(
                    get: { draftIsEnabled },
                    set: {
                        draftIsEnabled = $0
                        saveDraft(routeId: route.id)
                    }
                ))
                .toggleStyle(.switch)
                .font(.system(size: 11, weight: .semibold))
                .controlSize(.small)
            }

            VStack(alignment: .leading, spacing: 6) {
                Text("Title")
                    .font(.system(size: 10, weight: .bold))
                    .foregroundStyle(.secondary)
                TextField("Prompting", text: $draftTitle)
                    .textFieldStyle(.plain)
                    .font(.system(size: 14, weight: .semibold))
                    .padding(10)
                    .background(Color.primary.opacity(0.055), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                    .onSubmit { saveDraft(routeId: route.id) }
            }

            HStack(spacing: 10) {
                Menu {
                    ForEach(enabledPolishPresets) { preset in
                        Button(preset.displayName) {
                            draftPresetId = preset.id
                            saveDraft(routeId: route.id)
                        }
                    }
                } label: {
                    routeSelectPill(
                        title: selectedPresetName,
                        icon: "sparkles",
                        tint: .blue
                    )
                }
                .menuStyle(.button)
                .buttonStyle(.plain)
                .menuIndicator(.hidden)

                Button {
                    saveDraft(routeId: route.id)
                } label: {
                    routeSmallButton("Save", icon: "checkmark", tint: .green)
                }
                .buttonStyle(.plain)

                Spacer()

                Button(role: .destructive) {
                    routeToDelete = route
                } label: {
                    routeSmallButton("Delete Route", icon: "trash", tint: .red)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(16)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(.white.opacity(0.08), lineWidth: 0.8)
        )
    }

    private func selectedAppsSection(_ route: PolishRouteGroup) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Apps in this route")
                        .font(.system(size: 13, weight: .bold))
                    Text("When one of these apps is frontmost, AiTranscribe uses \(selectedPresetName).")
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }

            if route.appRules.isEmpty {
                Text("No apps assigned yet. Add apps from the scan below.")
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
                    .padding(12)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color.primary.opacity(0.04), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
            } else {
                FlowLayout(spacing: 8) {
                    ForEach(route.appRules) { rule in
                        selectedAppChip(rule, route: route)
                    }
                }
            }
        }
    }

    private func appScannerSection(_ route: PolishRouteGroup) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 10) {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Add apps")
                        .font(.system(size: 13, weight: .bold))
                    Text(appScannerSubtitle)
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                }

                Spacer()

                Button {
                    Task { await rescanApplications() }
                } label: {
                    if isScanningApps {
                        ProgressView()
                            .scaleEffect(0.62)
                            .frame(width: 68, height: 28)
                    } else {
                        routeSmallButton("Rescan", icon: "arrow.clockwise", tint: .secondary)
                    }
                }
                .buttonStyle(.plain)
                .disabled(isScanningApps)

                Button {
                    let includeInstalled = !showInstalledApps
                    showInstalledApps = includeInstalled
                    Task { await rescanApplications(includeInstalled: includeInstalled) }
                } label: {
                    routeSmallButton(showInstalledApps ? "Active Apps" : "All Apps", icon: showInstalledApps ? "bolt.fill" : "square.grid.2x2", tint: .secondary)
                }
                .buttonStyle(.plain)
                .disabled(isScanningApps)

                Button {
                    showHiddenApps.toggle()
                } label: {
                    routeSmallButton(showHiddenApps ? "Hide Hidden" : "Show Hidden", icon: showHiddenApps ? "eye.slash" : "eye", tint: .secondary)
                }
                .buttonStyle(.plain)

                if !hiddenAppIds.isEmpty {
                    Button {
                        let restoredCount = hiddenAppIds.count
                        hiddenAppIds.removeAll()
                        saveHiddenAppIds(hiddenAppIds)
                        appScanMessage = "Restored \(restoredCount) hidden \(restoredCount == 1 ? "app" : "apps")."
                    } label: {
                        routeSmallButton("Unhide \(hiddenAppIds.count)", icon: "eye.fill", tint: .orange)
                    }
                    .buttonStyle(.plain)
                    .help("Make every hidden app visible again")
                }

                Button {
                    Task { await cleanAppsWithAI() }
                } label: {
                    if isCleaningApps {
                        ProgressView()
                            .scaleEffect(0.62)
                            .frame(width: 72, height: 28)
                    } else {
                        routeSmallButton("AI Clean", icon: "wand.and.stars", tint: .blue)
                    }
                }
                .buttonStyle(.plain)
                .disabled(isCleaningApps || scannedApps.isEmpty)
            }

            TextField("Search apps", text: $appSearch)
                .textFieldStyle(.plain)
                .font(.system(size: 12))
                .padding(10)
                .background(Color.primary.opacity(0.055), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                .onChange(of: appSearch) {
                    appDisplayLimit = showInstalledApps ? 30 : 24
                }

            if let appScanMessage {
                Text(appScanMessage)
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }

            let visibleApps = visibleApps(for: route)
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 172), spacing: 8)], spacing: 8) {
                ForEach(visibleApps) { app in
                    scannedAppButton(app, route: route)
                }
            }

            let remainingApps = hiddenRemainingAppCount(for: route)
            if remainingApps > 0 {
                Button {
                    appDisplayLimit += 24
                } label: {
                    routeSmallButton("Show \(min(24, remainingApps)) More", icon: "chevron.down", tint: .secondary)
                }
                .buttonStyle(.plain)
                .frame(maxWidth: .infinity, alignment: .center)
            }

            Divider()
                .opacity(0.45)
                .padding(.vertical, 2)

            HStack(spacing: 8) {
                TextField("Manual app name", text: $manualAppName)
                    .textFieldStyle(.plain)
                    .font(.system(size: 12))
                    .padding(.horizontal, 10)
                    .frame(height: 32)
                    .background(Color.primary.opacity(0.045), in: Capsule())

                Button {
                    addManualApp(to: route)
                } label: {
                    routeSmallButton("Add Name", icon: "plus", tint: .blue)
                }
                .buttonStyle(.plain)
                .disabled(manualAppName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .padding(16)
        .background(Color.primary.opacity(0.035), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
    }

    private func selectedAppChip(_ rule: PolishRoutingRule, route: PolishRouteGroup) -> some View {
        HStack(spacing: 7) {
            Button {
                var copy = rule
                copy.isEnabled.toggle()
                appState.updatePolishRoutingRule(copy)
            } label: {
                Image(systemName: rule.isEnabled ? "checkmark.circle.fill" : "circle")
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(rule.isEnabled ? .green : .secondary)
            }
            .buttonStyle(.plain)

            VStack(alignment: .leading, spacing: 1) {
                Text(rule.displayAppName)
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(rule.isEnabled ? .primary : .secondary)
                if let bundleIdentifier = rule.bundleIdentifier {
                    Text(bundleIdentifier)
                        .font(.system(size: 8, design: .monospaced))
                        .foregroundStyle(.tertiary)
                        .lineLimit(1)
                }
            }

            Button {
                removeApp(rule, from: route)
            } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 8, weight: .bold))
                    .foregroundStyle(.tertiary)
            }
            .buttonStyle(.plain)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 7)
        .background(Color.primary.opacity(rule.isEnabled ? 0.07 : 0.035), in: Capsule())
        .overlay(Capsule().strokeBorder(.white.opacity(0.06), lineWidth: 0.5))
    }

    private func scannedAppButton(_ app: PolishRoutableApplication, route: PolishRouteGroup) -> some View {
        let isHidden = hiddenAppIds.contains(app.id)

        return HStack(spacing: 8) {
            Button {
                saveDraft(routeId: route.id)
                appState.addPolishRoutingApp(
                    routeId: route.id,
                    appName: app.displayName,
                    bundleIdentifier: app.bundleIdentifier
                )
            } label: {
                HStack(spacing: 9) {
                    Image(systemName: app.isRunning ? "circle.fill" : "app.dashed")
                        .font(.system(size: app.isRunning ? 7 : 11, weight: .semibold))
                        .foregroundStyle(app.isRunning ? .green : .secondary)
                        .frame(width: 16)

                    VStack(alignment: .leading, spacing: 1) {
                        Text(app.displayName)
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundStyle(isHidden ? .secondary : .primary)
                            .lineLimit(1)
                        Text(app.bundleIdentifier ?? "No bundle id")
                            .font(.system(size: 8, design: .monospaced))
                            .foregroundStyle(.tertiary)
                            .lineLimit(1)
                    }

                    Spacer(minLength: 4)

                    Image(systemName: "plus")
                        .font(.system(size: 9, weight: .bold))
                        .foregroundStyle(.secondary)
                }
            }
            .buttonStyle(.plain)
            .disabled(isHidden)

            Button {
                toggleHiddenApp(app)
            } label: {
                Image(systemName: isHidden ? "eye" : "eye.slash")
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(isHidden ? Color.green : Color.secondary.opacity(0.55))
                    .frame(width: 24, height: 24)
                    .background(Color.primary.opacity(0.045), in: Circle())
            }
            .buttonStyle(.plain)
        }
        .padding(8)
        .background(Color.primary.opacity(isHidden ? 0.025 : 0.045), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(app.isRunning ? Color.green.opacity(isHidden ? 0.10 : 0.20) : Color.white.opacity(0.06), lineWidth: 0.8)
        )
        .opacity(isHidden ? 0.62 : 1)
    }

    private var routeStatePill: some View {
        let tint: Color = appState.polishRoutingEnabled ? .green : .secondary
        return HStack(spacing: 7) {
            Image(systemName: appState.polishRoutingEnabled ? "checkmark.circle.fill" : "pause.circle")
                .font(.system(size: 10, weight: .semibold))
            Text(appState.polishRoutingEnabled ? "Routes on" : "Routes off")
                .font(.system(size: 10, weight: .semibold))
                .lineLimit(1)
            Text("\(routeGroups.count)")
                .font(.system(size: 10, weight: .bold, design: .monospaced))
                .foregroundStyle(.tertiary)
        }
        .foregroundStyle(tint)
        .padding(.horizontal, 10)
        .padding(.vertical, 7)
        .background(tint.opacity(0.10), in: Capsule())
    }

    private var appScannerSubtitle: String {
        if showHiddenApps {
            return "Hidden apps are visible. Use the eye button to restore them."
        }
        if showInstalledApps {
            return "All installed apps are shown gradually. Use Show More when you need deeper results."
        }
        return "Active apps only. Open an app, then rescan if it does not appear."
    }

    private func routeSelectPill(title: String, icon: String, tint: Color) -> some View {
        HStack(spacing: 7) {
            Image(systemName: icon)
                .font(.system(size: 10, weight: .semibold))
            Text(title)
                .font(.system(size: 11, weight: .semibold))
                .lineLimit(1)
            Image(systemName: "chevron.down")
                .font(.system(size: 8, weight: .bold))
                .foregroundStyle(.tertiary)
        }
        .foregroundStyle(tint)
        .padding(.horizontal, 11)
        .frame(height: 30)
        .background(tint.opacity(0.10), in: Capsule())
        .overlay(Capsule().strokeBorder(tint.opacity(0.18), lineWidth: 0.7))
    }

    private func routeSmallButton(_ title: String, icon: String, tint: Color) -> some View {
        HStack(spacing: 6) {
            Image(systemName: icon)
                .font(.system(size: 9, weight: .bold))
            Text(title)
                .font(.system(size: 10, weight: .semibold))
        }
        .foregroundStyle(tint)
        .padding(.horizontal, 10)
        .frame(height: 28)
        .background(tint.opacity(0.10), in: Capsule())
        .overlay(Capsule().strokeBorder(tint.opacity(0.16), lineWidth: 0.6))
    }

    private var routeGroups: [PolishRouteGroup] {
        let grouped = Dictionary(grouping: appState.polishRoutingRules, by: \.effectiveRouteId)
        return grouped.compactMap { routeId, rules -> PolishRouteGroup? in
            guard let first = rules.sorted(by: { lhs, rhs in
                if lhs.priority != rhs.priority { return lhs.priority < rhs.priority }
                return lhs.createdAt < rhs.createdAt
            }).first else {
                return nil
            }
            return PolishRouteGroup(
                id: routeId,
                title: first.routeDisplayName,
                description: first.routeDetails,
                presetId: first.presetId,
                isEnabled: first.isEnabled,
                priority: first.priority,
                rules: rules
            )
        }
        .sorted { lhs, rhs in
            if lhs.priority != rhs.priority { return lhs.priority < rhs.priority }
            return lhs.title.localizedCaseInsensitiveCompare(rhs.title) == .orderedAscending
        }
    }

    private var selectedGroup: PolishRouteGroup? {
        guard let selectedRouteId else { return nil }
        return routeGroups.first { $0.id == selectedRouteId }
    }

    private var enabledPolishPresets: [PolishPreset] {
        appState.polishPresets.filter(\.isEnabled)
    }

    private var selectedPresetName: String {
        appState.polishPresets.first(where: { $0.id == draftPresetId })?.displayName ?? "Choose Polish"
    }

    private var deleteDialogBinding: Binding<Bool> {
        Binding(
            get: { routeToDelete != nil },
            set: { if !$0 { routeToDelete = nil } }
        )
    }

    private func syncSelectionIfNeeded() {
        if let selectedRouteId,
           let route = routeGroups.first(where: { $0.id == selectedRouteId }) {
            if draftTitle.isEmpty {
                loadDraft(from: route)
            }
            return
        }
        selectedRouteId = routeGroups.first?.id
        if let route = routeGroups.first {
            loadDraft(from: route)
        }
    }

    private func loadDraft(from route: PolishRouteGroup) {
        draftTitle = route.title
        draftPresetId = route.presetId
        draftIsEnabled = route.isEnabled
        manualAppName = ""
    }

    private func saveDraftIfPossible() {
        guard let selectedRouteId else { return }
        saveDraft(routeId: selectedRouteId)
    }

    private func saveDraft(routeId: String) {
        let fallbackPresetId = enabledPolishPresets.first?.id ?? draftPresetId
        let presetId = draftPresetId.isEmpty ? fallbackPresetId : draftPresetId
        appState.updatePolishRoute(
            routeId: routeId,
            title: draftTitle,
            description: "",
            presetId: presetId,
            isEnabled: draftIsEnabled
        )
    }

    private func createRoute() {
        saveDraftIfPossible()
        let presetId = enabledPolishPresets.first?.id
        guard let routeId = appState.addPolishRoute(
            title: "New route",
            description: "",
            presetId: presetId
        ) else {
            return
        }
        selectedRouteId = routeId
        if let route = routeGroups.first(where: { $0.id == routeId }) {
            loadDraft(from: route)
        }
    }

    private func rescanApplications(includeInstalled: Bool? = nil) async {
        guard !isScanningApps else { return }
        let includeInstalled = includeInstalled ?? showInstalledApps
        isScanningApps = true
        appScanMessage = nil
        appDisplayLimit = includeInstalled ? 30 : 24
        defer { isScanningApps = false }

        let apps = await PolishContextReader.scanRoutableApplications(includeInstalled: includeInstalled)
        scannedApps = apps
        let visibleCount = apps.filter { app in
            (includeInstalled || app.isRunning) && !hiddenAppIds.contains(app.id)
        }.count
        let hiddenCount = apps.filter { app in
            (includeInstalled || app.isRunning) && hiddenAppIds.contains(app.id)
        }.count
        let hiddenSuffix = hiddenCount == 0 ? "." : " and \(hiddenCount) hidden apps."
        if includeInstalled {
            appScanMessage = "Found \(visibleCount) visible apps\(hiddenSuffix)"
        } else {
            appScanMessage = "Found \(visibleCount) active apps\(hiddenSuffix)"
        }
    }

    private func toggleHiddenApp(_ app: PolishRoutableApplication) {
        if hiddenAppIds.contains(app.id) {
            hiddenAppIds.remove(app.id)
        } else {
            hiddenAppIds.insert(app.id)
        }
        saveHiddenAppIds(hiddenAppIds)
    }

    private func cleanAppsWithAI() async {
        guard !isCleaningApps else { return }
        if activeModelSource == .local {
            await cleanAppsWithLocalAI()
            return
        }
        guard let model = selectedLanguageModel else {
            appScanMessage = "Add a public LLM before using AI Clean."
            return
        }
        guard APIKeychainStore.hasAPIKey(providerId: model.providerId) else {
            appScanMessage = "Add an API key for \(model.providerName) before using AI Clean."
            return
        }

        isCleaningApps = true
        appScanMessage = "AI is reviewing \(scannedApps.count) apps..."
        defer { isCleaningApps = false }

        do {
            guard let apiKey = try APIKeychainStore.apiKey(providerId: model.providerId),
                  !apiKey.isEmpty else {
                appScanMessage = "Add an API key for \(model.providerName) before using AI Clean."
                return
            }
            let hiddenIds: Set<String>
            if model.providerId == TranscriptionProviderDescriptor.openRouter.id {
                hiddenIds = try await OpenRouterLanguageModelClient().filterRoutableApplications(
                    applications: scannedApps,
                    modelId: model.modelId,
                    apiKey: apiKey,
                    providerOnly: model.openRouterProviderOnly
                )
            } else {
                hiddenIds = try await OpenAILanguageModelClient().filterRoutableApplications(
                    applications: scannedApps,
                    modelId: model.modelId,
                    apiKey: apiKey
                )
            }

            let scannedIds = Set(scannedApps.map(\.id))
            let validHiddenIds = hiddenIds.intersection(scannedIds)
            hiddenAppIds.formUnion(validHiddenIds)
            saveHiddenAppIds(hiddenAppIds)
            appScanMessage = "AI hid \(validHiddenIds.count) helper apps."
        } catch {
            appScanMessage = "AI Clean failed: \(error.localizedDescription)"
        }
    }

    /// AI Clean through the bundled backend when the app is in Local mode, so
    /// the button works without any API key.
    private func cleanAppsWithLocalAI() async {
        let localModels = appState.availableSummaryModels.filter(\.downloaded)
        guard let localModel = localModels.first(where: { $0.id == appState.preferredSummaryModelId }) ?? localModels.first else {
            appScanMessage = "Download a local LLM in Models before using AI Clean."
            return
        }
        guard appState.isServerConnected else {
            appScanMessage = "The local backend is offline. Start it before using AI Clean."
            return
        }

        isCleaningApps = true
        appScanMessage = "\(localModel.displayName) is reviewing \(scannedApps.count) apps..."
        defer { isCleaningApps = false }

        do {
            let payload = scannedApps.map { app in
                [
                    "id": app.id,
                    "name": app.displayName,
                    "bundleIdentifier": app.bundleIdentifier ?? "",
                    "isRunning": app.isRunning ? "true" : "false"
                ]
            }
            let payloadData = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
            let payloadText = String(data: payloadData, encoding: .utf8) ?? "[]"

            let responseText = try await PolishGeneratorLLM.local(modelId: localModel.id).complete(
                system: routableApplicationFilterInstruction,
                user: payloadText,
                maxTokens: 2_000
            )
            let jsonText = responseText
                .replacingOccurrences(of: "```json", with: "")
                .replacingOccurrences(of: "```", with: "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard let data = jsonText.data(using: .utf8),
                  let result = try? JSONDecoder().decode(PolishRouteApplicationFilterResult.self, from: data) else {
                appScanMessage = "AI Clean failed: the local model returned unusable output."
                return
            }

            let scannedIds = Set(scannedApps.map(\.id))
            let validHiddenIds = Set(result.hiddenApplicationIds).intersection(scannedIds)
            hiddenAppIds.formUnion(validHiddenIds)
            saveHiddenAppIds(hiddenAppIds)
            appScanMessage = "AI hid \(validHiddenIds.count) helper apps."
        } catch {
            appScanMessage = "AI Clean failed: \(error.localizedDescription)"
        }
    }

    private func removeApp(_ rule: PolishRoutingRule, from route: PolishRouteGroup) {
        if route.rules.count == 1 {
            var copy = rule
            copy.appName = ""
            copy.bundleIdentifier = nil
            appState.updatePolishRoutingRule(copy)
        } else {
            appState.deletePolishRoutingRule(id: rule.id)
        }
    }

    private func appCandidates(for route: PolishRouteGroup) -> [PolishRoutableApplication] {
        let query = appSearch.trimmingCharacters(in: .whitespacesAndNewlines)
        let existing = Set(route.appRules.map { rule in
            (rule.bundleIdentifier ?? rule.displayAppName).lowercased()
        })

        return scannedApps.filter { app in
            let key = (app.bundleIdentifier ?? app.displayName).lowercased()
            guard showInstalledApps || app.isRunning else { return false }
            guard !existing.contains(key) else { return false }
            guard showHiddenApps || !hiddenAppIds.contains(app.id) else { return false }
            guard !query.isEmpty else { return true }
            return app.displayName.localizedCaseInsensitiveContains(query) ||
                (app.bundleIdentifier?.localizedCaseInsensitiveContains(query) ?? false)
        }
    }

    private func visibleApps(for route: PolishRouteGroup) -> [PolishRoutableApplication] {
        Array(appCandidates(for: route).prefix(appDisplayLimit))
    }

    private func hiddenRemainingAppCount(for route: PolishRouteGroup) -> Int {
        max(appCandidates(for: route).count - appDisplayLimit, 0)
    }

    private func addManualApp(to route: PolishRouteGroup) {
        let appName = manualAppName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !appName.isEmpty else { return }
        saveDraft(routeId: route.id)
        appState.addPolishRoutingApp(routeId: route.id, appName: appName)
        manualAppName = ""
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

    private func loadHiddenAppIds() -> Set<String> {
        guard let data = UserDefaults.standard.data(forKey: "polishRouteHiddenAppIds"),
              let ids = try? JSONDecoder().decode([String].self, from: data) else {
            return []
        }
        return Set(ids)
    }

    private func saveHiddenAppIds(_ ids: Set<String>) {
        guard let data = try? JSONEncoder().encode(Array(ids).sorted()) else { return }
        UserDefaults.standard.set(data, forKey: "polishRouteHiddenAppIds")
    }
}

private struct PolishPresetCard: View {
    let preset: PolishPreset
    let isSelected: Bool
    let isTesting: Bool
    let activeModelSource: PolishModelSource
    let localLanguageModels: [SummaryModelInfoResponse]
    let onSelect: () -> Void
    let onToggleMenu: () -> Void
    let onEdit: () -> Void
    let onDelete: () -> Void
    let onTest: () -> Void

    @State private var showingDeleteConfirmation = false
    @State private var isHovered = false

    var body: some View {
        VStack(alignment: .leading, spacing: 11) {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: preset.symbolName)
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(isSelected ? .white : .blue)
                    .frame(width: 32, height: 32)
                    .background(isSelected ? Color.blue : Color.blue.opacity(0.12), in: RoundedRectangle(cornerRadius: 9, style: .continuous))

                VStack(alignment: .leading, spacing: 3) {
                    HStack(spacing: 6) {
                        Text(preset.displayName)
                            .font(.system(size: 13, weight: .semibold))
                            .foregroundStyle(.primary)
                            .lineLimit(1)

                        if preset.isMultiStep {
                            Text("Multi-step")
                                .font(.system(size: 8, weight: .bold))
                                .foregroundStyle(.purple)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(Color.purple.opacity(0.12), in: Capsule())
                        }
                    }

                    Text(modelLabel)
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.tertiary)
                        .lineLimit(1)
                }

                Spacer()

                cardStatusPill
            }

            Text(preset.prompt)
                .font(.system(size: 11))
                .foregroundStyle(.secondary)
                .lineLimit(2)
                .frame(maxWidth: .infinity, alignment: .leading)
                .frame(height: 34, alignment: .topLeading)

            moduleBand

            HStack(spacing: 9) {
                Button {
                    onSelect()
                } label: {
                    HStack(spacing: 5) {
                        Image(systemName: isSelected ? "xmark" : "checkmark")
                            .font(.system(size: 9, weight: .bold))
                        Text(isSelected ? "Raw" : "Use")
                            .font(.system(size: 10, weight: .bold))
                    }
                    .foregroundStyle(.white)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(isSelected ? Color.secondary.opacity(0.55) : Color.accentColor, in: Capsule())
                }
                .buttonStyle(.plain)
                .disabled(needsLocalLanguageModel)
                .opacity(needsLocalLanguageModel ? 0.45 : 1)
                .help(needsLocalLanguageModel ? "Download or pick a local LLM before using this Polish in Local mode" : "Use this Polish")

                HStack(spacing: 5) {
                    iconAction(
                        icon: preset.showInMenu ? "menubar.rectangle" : "menubar.dock.rectangle",
                        isActive: preset.showInMenu,
                        help: preset.showInMenu ? "Shown in menu" : "Hidden from menu",
                        action: onToggleMenu
                    )
                }

                Spacer()

                HStack(spacing: 5) {
                    Button {
                        onTest()
                    } label: {
                        HStack(spacing: 5) {
                            if isTesting {
                                ProgressView()
                                    .scaleEffect(0.58)
                                    .frame(width: 12, height: 12)
                            } else {
                                Image(systemName: "play.circle")
                                    .font(.system(size: 10, weight: .semibold))
                                    .frame(width: 12, height: 12)
                            }

                            Text("Test")
                                .font(.system(size: 10, weight: .semibold))
                        }
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 9)
                        .padding(.vertical, 6)
                        .background(Color.primary.opacity(0.06), in: Capsule())
                    }
                    .buttonStyle(.plain)
                    .disabled(!preset.isEnabled || isTesting || needsLocalLanguageModel)
                    .opacity(needsLocalLanguageModel ? 0.45 : 1)
                    .help(needsLocalLanguageModel ? "Download or pick a local LLM before testing this Polish in Local mode" : "Test this Polish with a sample transcript")

                    iconAction(
                        icon: "trash",
                        isActive: false,
                        tint: .red,
                        help: "Delete Polish",
                        action: { showingDeleteConfirmation = true }
                    )
                }
            }
        }
        .padding(15)
        .frame(height: 184, alignment: .top)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(isSelected ? Color.green.opacity(0.45) : Color.white.opacity(isHovered ? 0.14 : 0.06), lineWidth: 0.8)
        )
        .scaleEffect(isHovered ? 1.01 : 1)
        .contentShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
        .onTapGesture { onEdit() }
        .animation(.easeOut(duration: 0.16), value: isHovered)
        .onHover { isHovered = $0 }
        .confirmationDialog("Delete Polish?", isPresented: $showingDeleteConfirmation) {
            Button("Delete", role: .destructive) { onDelete() }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This removes the Polish from Settings and the menu. You can recreate it later if needed.")
        }
    }

    private var modelLabel: String {
        if activeModelSource == .local {
            if let model = selectedLocalModel {
                return model.displayName
            }
            return "Pick local LLM"
        }
        return preset.modelId
    }

    private var selectedLocalModel: SummaryModelInfoResponse? {
        let downloaded = localLanguageModels.filter(\.downloaded)
        if let localModelId = preset.localModelId,
           let model = downloaded.first(where: { $0.id == localModelId }) {
            return model
        }
        if let preferred = UserDefaults.standard.string(forKey: "preferredSummaryModelId"),
           let model = downloaded.first(where: { $0.id == preferred }) {
            return model
        }
        return downloaded.first
    }

    @ViewBuilder
    private var cardStatusPill: some View {
        if needsLocalLanguageModel {
            Text("Needs LLM")
                .font(.system(size: 9, weight: .bold))
                .foregroundStyle(.orange)
                .padding(.horizontal, 7)
                .padding(.vertical, 3)
                .background(Color.orange.opacity(0.12), in: Capsule())
        } else if isSelected {
            Text("Selected")
                .font(.system(size: 9, weight: .bold))
                .foregroundStyle(.green)
                .padding(.horizontal, 7)
                .padding(.vertical, 3)
                .background(Color.green.opacity(0.12), in: Capsule())
        } else if !preset.isEnabled {
            Text("Paused")
                .font(.system(size: 9, weight: .bold))
                .foregroundStyle(.secondary)
                .padding(.horizontal, 7)
                .padding(.vertical, 3)
                .background(Color.primary.opacity(0.06), in: Capsule())
        }
    }

    private var needsLocalLanguageModel: Bool {
        activeModelSource == .local && selectedLocalModel == nil
    }

    private var moduleBand: some View {
        HStack(spacing: 6) {
            if preset.isMultiStep {
                let visibleModules = Array((preset.inputModules.map { ($0.symbolName, $0.label, Color.blue) } +
                                            preset.outputModules.map { ($0.symbolName, $0.label, Color.purple) }).prefix(4))
                ForEach(visibleModules.indices, id: \.self) { index in
                    PolishModuleChip(
                        icon: visibleModules[index].0,
                        text: visibleModules[index].1,
                        tint: visibleModules[index].2
                    )
                }
            } else {
                PolishModuleChip(icon: "waveform", text: "Voice", tint: .blue)
                PolishModuleChip(icon: "doc.on.doc", text: "Copy", tint: .purple)
            }

            Spacer(minLength: 0)
        }
        .frame(height: 25, alignment: .leading)
        .clipped()
    }

    private func iconAction(
        icon: String,
        isActive: Bool,
        tint: Color = .blue,
        help: String,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            Image(systemName: icon)
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(isActive ? .white : tint == .red ? .red : .secondary)
                .frame(width: 24, height: 24)
                .background(
                    isActive
                        ? tint.opacity(0.75)
                        : tint == .red ? Color.red.opacity(0.10) : Color.primary.opacity(0.06),
                    in: Circle()
                )
        }
        .buttonStyle(.plain)
        .help(help)
    }
}

private struct PolishModuleChip: View {
    let icon: String
    let text: String
    let tint: Color

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: icon)
                .font(.system(size: 8, weight: .semibold))
            Text(text)
                .font(.system(size: 9, weight: .semibold))
                .lineLimit(1)
        }
        .foregroundStyle(tint)
        .padding(.horizontal, 7)
        .padding(.vertical, 4)
        .background(tint.opacity(0.10), in: Capsule())
        .overlay(
            Capsule()
                .strokeBorder(tint.opacity(0.18), lineWidth: 0.6)
        )
    }
}

private struct FlowLayout: Layout {
    var spacing: CGFloat = 8

    func sizeThatFits(
        proposal: ProposedViewSize,
        subviews: Subviews,
        cache: inout ()
    ) -> CGSize {
        let maxWidth = proposal.width ?? 320
        let rows = layoutRows(for: subviews, maxWidth: maxWidth)
        return CGSize(
            width: maxWidth,
            height: rows.reduce(CGFloat.zero) { $0 + $1.height } + CGFloat(max(rows.count - 1, 0)) * spacing
        )
    }

    func placeSubviews(
        in bounds: CGRect,
        proposal: ProposedViewSize,
        subviews: Subviews,
        cache: inout ()
    ) {
        let rows = layoutRows(for: subviews, maxWidth: bounds.width)
        var y = bounds.minY

        for row in rows {
            var x = bounds.minX
            for item in row.items {
                item.subview.place(
                    at: CGPoint(x: x, y: y),
                    anchor: .topLeading,
                    proposal: ProposedViewSize(item.size)
                )
                x += item.size.width + spacing
            }
            y += row.height + spacing
        }
    }

    private func layoutRows(for subviews: Subviews, maxWidth: CGFloat) -> [FlowRow] {
        var rows: [FlowRow] = []
        var currentItems: [FlowItem] = []
        var currentWidth: CGFloat = 0
        var currentHeight: CGFloat = 0

        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            let proposedWidth = currentItems.isEmpty ? size.width : currentWidth + spacing + size.width

            if proposedWidth > maxWidth, !currentItems.isEmpty {
                rows.append(FlowRow(items: currentItems, height: currentHeight))
                currentItems = [FlowItem(subview: subview, size: size)]
                currentWidth = size.width
                currentHeight = size.height
            } else {
                currentItems.append(FlowItem(subview: subview, size: size))
                currentWidth = proposedWidth
                currentHeight = max(currentHeight, size.height)
            }
        }

        if !currentItems.isEmpty {
            rows.append(FlowRow(items: currentItems, height: currentHeight))
        }

        return rows
    }

    private struct FlowRow {
        let items: [FlowItem]
        let height: CGFloat
    }

    private struct FlowItem {
        let subview: LayoutSubview
        let size: CGSize
    }
}

private struct PolishPresetDetailView: View {
    let preset: PolishPreset?
    let apiLanguageModels: [APILanguageModel]
    let localLanguageModels: [SummaryModelInfoResponse]
    let selectedModelId: String
    let selectedLocalModelId: String?
    let activeModelSource: PolishModelSource
    let onBack: () -> Void
    let onSave: (PolishPreset) -> Void
    let onAIDraft: ((PolishGeneratorDraftPackage) -> Void)?

    @State private var displayName: String
    @State private var symbolName: String
    @State private var prompt: String
    @State private var bypassLLM: Bool
    @State private var modelId: String
    @State private var localModelId: String
    @State private var showInMenu: Bool
    @State private var isEnabled: Bool
    @State private var pasteBehavior: PolishPasteBehavior
    @State private var shortcut: String
    @State private var isRecordingShortcut = false
    @State private var showingIconPicker = false
    @State private var showAdvancedModules: Bool
    @State private var renderAdvancedModules: Bool
    @State private var inputModules: Set<PolishInputModule>
    @State private var outputModules: Set<PolishOutputModule>
    @State private var isEditingName = false
    @State private var showingAIEditPopover = false
    @State private var aiEditRequest = ""
    @State private var isAIEditing = false
    @State private var aiEditPhaseText: String?
    @State private var aiEditError: String?
    @AppStorage("polishGeneratorModelId") private var polishGeneratorModelId = ""

    init(
        preset: PolishPreset?,
        apiLanguageModels: [APILanguageModel],
        localLanguageModels: [SummaryModelInfoResponse],
        selectedModelId: String,
        selectedLocalModelId: String?,
        activeModelSource: PolishModelSource,
        onBack: @escaping () -> Void,
        onSave: @escaping (PolishPreset) -> Void,
        onAIDraft: ((PolishGeneratorDraftPackage) -> Void)? = nil
    ) {
        self.preset = preset
        self.apiLanguageModels = apiLanguageModels
        self.selectedModelId = selectedModelId
        self.localLanguageModels = localLanguageModels
        self.selectedLocalModelId = selectedLocalModelId
        self.activeModelSource = activeModelSource
        self.onBack = onBack
        self.onSave = onSave
        self.onAIDraft = onAIDraft
        _displayName = State(initialValue: preset?.displayName ?? "")
        _symbolName = State(initialValue: preset?.symbolName ?? "sparkles")
        _prompt = State(initialValue: preset?.prompt ?? "")
        _bypassLLM = State(initialValue: preset?.bypassLLM ?? false)
        _modelId = State(initialValue: preset?.modelId ?? selectedModelId)
        let fallbackLocalModelId = preset?.localModelId ?? selectedLocalModelId ?? localLanguageModels.first(where: { $0.downloaded })?.id ?? ""
        _localModelId = State(initialValue: fallbackLocalModelId)
        _showInMenu = State(initialValue: preset?.showInMenu ?? true)
        _isEnabled = State(initialValue: preset?.isEnabled ?? true)
        _pasteBehavior = State(initialValue: preset?.pasteBehavior ?? .copyOnly)
        _shortcut = State(initialValue: preset?.shortcut ?? "")
        _showAdvancedModules = State(initialValue: preset?.isMultiStep ?? false)
        _renderAdvancedModules = State(initialValue: preset?.isMultiStep ?? false)
        _inputModules = State(initialValue: Set(preset?.inputModules ?? PolishPreset.defaultInputModules))
        _outputModules = State(initialValue: Set(preset?.outputModules ?? PolishPreset.defaultOutputModules))
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                detailHeader

                configurationStrip

                formRow(label: "LLM") {
                    VStack(alignment: .leading, spacing: 8) {
                        llmSelectionRow
                            .opacity(bypassLLM ? 0.45 : 1)
                            .disabled(bypassLLM)

                        if bypassLLM {
                            Text("Bypass LLM is on. Raw transcription goes straight to outputs.")
                                .font(.system(size: 10, weight: .medium))
                                .foregroundStyle(.secondary)
                        }
                    }
                }

                if renderAdvancedModules {
                    VStack(alignment: .leading, spacing: 10) {
                        if preset != nil, onAIDraft != nil {
                            HStack {
                                Spacer()
                                aiEditButton
                            }
                        }

                        advancedModulePipeline
                    }
                    .transition(
                        .asymmetric(
                            insertion: .opacity.combined(with: .move(edge: .top)),
                            removal: .opacity
                        )
                    )
                }

                formRow(label: "Prompt") {
                    VStack(alignment: .leading, spacing: 6) {
                        TextEditor(text: $prompt)
                            .font(.system(size: 12))
                            .frame(height: promptEditorHeight)
                            .scrollContentBackground(.hidden)
                            .padding(10)
                            .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                            .overlay(
                                RoundedRectangle(cornerRadius: 10, style: .continuous)
                                    .strokeBorder(.white.opacity(0.08), lineWidth: 0.8)
                            )
                            .opacity(bypassLLM ? 0.45 : 1)
                            .disabled(bypassLLM)

                        if bypassLLM {
                            Text("Bypass LLM is on. The prompt is not used.")
                                .font(.system(size: 10))
                                .foregroundStyle(.tertiary)
                        }
                    }
                }

                Spacer(minLength: 12)
            }
            .padding(.horizontal, 28)
            .padding(.top, 8)
            .padding(.bottom, 92)
        }
        .scrollIndicators(.automatic)
        .safeAreaInset(edge: .bottom) {
            detailActionBar
        }
    }

    private var detailHeader: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Button {
                    withAnimation(.easeInOut(duration: 0.15)) { onBack() }
                } label: {
                    HStack(spacing: 5) {
                        Image(systemName: "chevron.left")
                            .font(.system(size: 10, weight: .semibold))
                        Text("Polishes")
                            .font(.system(size: 12, weight: .medium))
                    }
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(.quaternary, in: Capsule())
                }
                .buttonStyle(.plain)

                Spacer()
            }

            HStack(alignment: .center, spacing: 12) {
                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 8) {
                        Text(preset == nil ? "New" : "Edit")
                            .font(.system(size: 22, weight: .bold, design: .rounded))
                            .foregroundStyle(.secondary)

                        if isEditingName {
                            TextField("Polish name", text: $displayName)
                                .textFieldStyle(.plain)
                                .font(.system(size: 22, weight: .bold, design: .rounded))
                                .onSubmit {
                                    displayName = normalizedDisplayName
                                    isEditingName = false
                                }
                                .onExitCommand {
                                    displayName = normalizedDisplayName
                                    isEditingName = false
                                }
                                .frame(maxWidth: 360)
                        } else {
                            Text(normalizedDisplayName)
                                .font(.system(size: 22, weight: .bold, design: .rounded))
                                .lineLimit(1)
                                .onTapGesture(count: 2) {
                                    isEditingName = true
                                }
                        }

                        Button {
                            if isEditingName {
                                displayName = normalizedDisplayName
                            }
                            isEditingName.toggle()
                        } label: {
                            Image(systemName: isEditingName ? "checkmark.circle.fill" : "pencil")
                                .font(.system(size: 11, weight: .semibold))
                                .foregroundStyle(.secondary)
                                .frame(width: 24, height: 24)
                                .background(Color.primary.opacity(0.055), in: Circle())
                        }
                        .buttonStyle(.plain)
                    }

                    Text("Configure prompt, LLM, and optional context modules.")
                        .font(.system(size: 12))
                        .foregroundStyle(.secondary)
                }

                Spacer()

                Button {
                    toggleAdvancedModules()
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: showAdvancedModules ? "slider.horizontal.below.rectangle" : "point.3.connected.trianglepath.dotted")
                            .font(.system(size: 10, weight: .semibold))
                        Text(showAdvancedModules ? "Simple" : "Advanced")
                            .font(.system(size: 11, weight: .semibold))
                    }
                    .foregroundStyle(showAdvancedModules ? .white : .blue)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 7)
                    .background(showAdvancedModules ? Color.purple : Color.blue.opacity(0.13), in: Capsule())
                    .overlay(
                        Capsule()
                            .strokeBorder(showAdvancedModules ? Color.purple.opacity(0.35) : Color.blue.opacity(0.24), lineWidth: 0.7)
                    )
                }
                .buttonStyle(.plain)
            }

            SettingsDivider()
        }
    }

    private var configurationStrip: some View {
        HStack(spacing: 10) {
            Button {
                showingIconPicker.toggle()
            } label: {
                HStack(spacing: 9) {
                    Image(systemName: symbolName.isEmpty ? "sparkles" : symbolName)
                        .font(.system(size: 15, weight: .semibold))
                        .foregroundStyle(.blue)
                        .frame(width: 34, height: 34)
                        .background(Color.blue.opacity(0.12), in: Circle())

                    VStack(alignment: .leading, spacing: 2) {
                        Text("Icon")
                            .font(.system(size: 10, weight: .bold))
                            .foregroundStyle(.tertiary)
                            .tracking(0.7)
                        Text(symbolName.isEmpty ? "sparkles" : symbolName)
                            .font(.system(size: 11, weight: .semibold, design: .monospaced))
                            .foregroundStyle(.primary)
                            .lineLimit(1)
                    }

                    Image(systemName: "square.grid.2x2")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(.secondary)
                }
                .padding(.horizontal, 11)
                .frame(height: 50)
                .background(Color.primary.opacity(0.075), in: Capsule())
                .overlay(
                    Capsule()
                        .strokeBorder(.white.opacity(0.06), lineWidth: 0.5)
                )
            }
            .buttonStyle(.plain)
            .popover(isPresented: $showingIconPicker, arrowEdge: .bottom) {
                PolishIconGridPicker(selectedSymbol: $symbolName) {
                    showingIconPicker = false
                }
            }

            Spacer(minLength: 0)

            settingsPillToggle(label: "Menu", icon: "menubar.rectangle", isOn: $showInMenu, tint: .blue)
            settingsPillToggle(label: "Enabled", icon: "checkmark.circle", isOn: $isEnabled, tint: .green)

            pasteBehaviorMenu

            ShortcutPill(
                shortcut: $shortcut,
                isRecording: $isRecordingShortcut,
                onChanged: { HotkeyManager.shared.refreshHotkeys() }
            )
            .help("Shortcut for this Polish")

            if !shortcut.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                Button {
                    shortcut = ""
                    HotkeyManager.shared.refreshHotkeys()
                } label: {
                    Image(systemName: "xmark")
                        .font(.system(size: 9, weight: .bold))
                        .foregroundStyle(.secondary)
                        .frame(width: 24, height: 24)
                        .background(Color.primary.opacity(0.045), in: Circle())
                }
                .buttonStyle(.plain)
                .help("Clear shortcut")
            }
        }
        .padding(.vertical, 2)
    }

    private func settingsPillToggle(
        label: String,
        icon: String,
        isOn: Binding<Bool>,
        tint: Color
    ) -> some View {
        Button {
            isOn.wrappedValue.toggle()
        } label: {
            HStack(spacing: 7) {
                Image(systemName: icon)
                    .font(.system(size: 10, weight: .semibold))
                Text(label)
                    .font(.system(size: 11, weight: .semibold))
                Circle()
                    .fill(isOn.wrappedValue ? tint : Color.secondary.opacity(0.35))
                    .frame(width: 7, height: 7)
            }
            .foregroundStyle(isOn.wrappedValue ? tint : .secondary)
            .padding(.horizontal, 11)
            .padding(.vertical, 7)
            .background(isOn.wrappedValue ? tint.opacity(0.11) : Color.primary.opacity(0.045), in: Capsule())
            .overlay(
                Capsule()
                    .strokeBorder(isOn.wrappedValue ? tint.opacity(0.24) : Color.white.opacity(0.06), lineWidth: 0.7)
            )
        }
        .buttonStyle(.plain)
    }

    private var detailActionBar: some View {
        HStack(spacing: 10) {
            Spacer()

            Button {
                save()
            } label: {
                Text("Save Polish")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 15)
                    .padding(.vertical, 8)
                    .background(canSave ? Color.accentColor : Color.gray.opacity(0.35), in: Capsule())
            }
            .buttonStyle(.plain)
            .disabled(!canSave)
        }
        .padding(.horizontal, 28)
        .padding(.vertical, 12)
        .background(.regularMaterial)
        .overlay(alignment: .top) {
            Rectangle()
                .fill(.white.opacity(0.06))
                .frame(height: 0.7)
        }
    }

    @ViewBuilder
    private var llmSelectionRow: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 7) {
                Image(systemName: activeModelSource == .local ? "memorychip" : "network")
                    .font(.system(size: 10, weight: .semibold))
                Text(activeModelSource.label)
                    .font(.system(size: 11, weight: .semibold))
                Text(activeModelSource == .local ? "Uses downloaded Gemma models" : "Uses API LLMs")
                    .font(.system(size: 10))
                    .foregroundStyle(.secondary)
            }
            .foregroundStyle(activeModelSource == .local ? .green : .blue)

            if activeModelSource == .local {
                if downloadedLocalModels.isEmpty {
                    HStack(alignment: .top, spacing: 7) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .font(.system(size: 11, weight: .semibold))
                        Text("Download a local LLM in Models > LLM before using Polishes in Local mode.")
                            .font(.system(size: 11, weight: .medium))
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .foregroundStyle(.orange)
                    .padding(10)
                    .background(Color.orange.opacity(0.10), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                } else {
                    localModelMenu
                }
            } else {
                apiModelMenu
            }
        }
    }

    private var pasteBehaviorMenu: some View {
        Menu {
            ForEach(PolishPasteBehavior.allCases) { behavior in
                Button(behavior.label) {
                    pasteBehavior = behavior
                    if behavior == .copyOnly {
                        outputModules.remove(.copyTextAndScreenshot)
                    }
                }
            }
        } label: {
            polishSelectPill(
                title: pasteBehavior.label,
                icon: "keyboard",
                width: 154
            )
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
    }

    private var localModelMenu: some View {
        Menu {
            ForEach(downloadedLocalModels) { model in
                Button("\(model.displayName) · Local") {
                    localModelId = model.id
                }
            }
        } label: {
            polishSelectPill(
                title: selectedLocalModel.map { "\($0.displayName) · Local" } ?? "Choose local LLM",
                icon: "memorychip",
                width: 330
            )
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
    }

    private var apiModelMenu: some View {
        Menu {
            ForEach(apiLanguageModels) { model in
                Button("\(model.displayName) · \(model.providerName)") {
                    modelId = model.modelId
                }
            }
        } label: {
            polishSelectPill(
                title: selectedAPIModel.map { "\($0.displayName) · \($0.providerName)" } ?? "Choose API LLM",
                icon: "network",
                width: 330
            )
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
    }

    private func polishSelectPill(title: String, icon: String, width: CGFloat) -> some View {
        HStack(spacing: 7) {
            Image(systemName: icon)
                .font(.system(size: 9, weight: .semibold))
                .foregroundStyle(.secondary)

            Text(title)
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(.primary)
                .lineLimit(1)
                .truncationMode(.tail)

            Spacer(minLength: 8)

            Image(systemName: "chevron.down")
                .font(.system(size: 8, weight: .bold))
                .foregroundStyle(.tertiary)
        }
        .padding(.horizontal, 11)
        .frame(width: width, height: 30)
        .background(Color.primary.opacity(0.075), in: Capsule())
        .overlay(Capsule().strokeBorder(.white.opacity(0.06), lineWidth: 0.5))
        .contentShape(Capsule())
    }

    private var canSave: Bool {
        !displayName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty &&
        (bypassLLM || (
            !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty &&
            !modelId.isEmpty
        )) &&
        !requiresVisionModelSelection &&
        !requiresLocalModelSelection &&
        !requiresLocalVisionSelection
    }

    private var requiresVisionModelSelection: Bool {
        !bypassLLM &&
        activeModelSource == .public &&
        inputModules.contains(.screenSnapshot) &&
        !(selectedAPIModel?.supportsVision ?? false)
    }

    private var requiresLocalModelSelection: Bool {
        !bypassLLM &&
        activeModelSource == .local && downloadedLocalModels.isEmpty
    }

    private var requiresLocalVisionSelection: Bool {
        !bypassLLM &&
        activeModelSource == .local &&
        inputModules.contains(.screenSnapshot) &&
        !(selectedLocalModel?.supportsVision ?? false)
    }

    private var downloadedLocalModels: [SummaryModelInfoResponse] {
        localLanguageModels.filter(\.downloaded)
    }

    private var normalizedDisplayName: String {
        let trimmed = displayName.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? "Untitled Polish" : trimmed
    }

    private var promptEditorHeight: CGFloat {
        let measuredPrompt = prompt.trimmingCharacters(in: .newlines)
        let lines = measuredPrompt.components(separatedBy: .newlines)
        let wrappedLineCount = lines.reduce(0) { total, line in
            total + max(1, Int(ceil(Double(line.count) / 92.0)))
        }
        let visibleLines = min(18, max(6, wrappedLineCount))
        return CGFloat(visibleLines) * 17 + 42
    }

    private func toggleAdvancedModules() {
        let willShowAdvanced = !showAdvancedModules
        showAdvancedModules = willShowAdvanced

        if willShowAdvanced {
            Task { @MainActor in
                try? await Task.sleep(for: .milliseconds(95))
                guard showAdvancedModules else { return }
                withAnimation(.easeOut(duration: 0.28)) {
                    renderAdvancedModules = true
                }
            }
        } else {
            withAnimation(.easeOut(duration: 0.11)) {
                renderAdvancedModules = false
            }
        }
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

    // MARK: Edit with AI

    /// The generator's own model choice, same key the generator card uses.
    private var aiEditModel: APILanguageModel? {
        let chosen = apiLanguageModels.first(where: { $0.modelId == polishGeneratorModelId })
            ?? apiLanguageModels.first(where: { $0.modelId == selectedModelId })
            ?? apiLanguageModels.first
        guard let chosen else { return nil }
        guard chosen.providerId == TranscriptionProviderDescriptor.openAI.id,
              chosen.modelId.contains("/"),
              APIKeychainStore.hasAPIKey(providerId: TranscriptionProviderDescriptor.openRouter.id) else {
            return chosen
        }
        return APILanguageModel(
            id: chosen.id,
            providerId: TranscriptionProviderDescriptor.openRouter.id,
            providerName: TranscriptionProviderDescriptor.openRouter.displayName,
            modelId: chosen.modelId,
            displayName: chosen.displayName,
            inputCostPerMillionTokensUSD: chosen.inputCostPerMillionTokensUSD,
            outputCostPerMillionTokensUSD: chosen.outputCostPerMillionTokensUSD,
            openRouterProviderOnly: chosen.openRouterProviderOnly,
            createdAt: chosen.createdAt
        )
    }

    /// Sized like the configuration strip's pills (Enabled, Menu) so it reads
    /// as a first-class action, not a footnote.
    private var aiEditButton: some View {
        Button {
            showingAIEditPopover = true
        } label: {
            HStack(spacing: 7) {
                Image(systemName: "sparkles")
                    .font(.system(size: 10, weight: .semibold))
                Text("Edit with AI")
                    .font(.system(size: 11, weight: .semibold))
            }
            .foregroundStyle(.blue)
            .padding(.horizontal, 11)
            .padding(.vertical, 7)
            .background(Color.blue.opacity(0.11), in: Capsule())
            .overlay(
                Capsule()
                    .strokeBorder(Color.blue.opacity(0.24), lineWidth: 0.7)
            )
        }
        .buttonStyle(.plain)
        .disabled(isAIEditing)
        .popover(isPresented: $showingAIEditPopover, arrowEdge: .bottom) {
            aiEditPopover
        }
    }

    private var aiEditPopover: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 6) {
                Image(systemName: "sparkles")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(.blue)
                Text("Edit with AI")
                    .font(.system(size: 13, weight: .semibold))
            }

            Text("Describe the change. The current Polish is the starting point; everything you do not mention stays as it is.")
                .font(.system(size: 11))
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            TextField(
                "Example: Also use clipboard text so I can copy crash logs as extra context",
                text: $aiEditRequest,
                axis: .vertical
            )
            .textFieldStyle(.plain)
            .font(.system(size: 12))
            .lineLimit(2...5)
            .padding(9)
            .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 9, style: .continuous))

            if let aiEditError {
                Text(aiEditError)
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(.red)
                    .fixedSize(horizontal: false, vertical: true)
            }

            HStack(spacing: 8) {
                if isAIEditing, let aiEditPhaseText {
                    HStack(spacing: 5) {
                        Image(systemName: "sparkles")
                            .font(.system(size: 10, weight: .semibold))
                            .foregroundStyle(.blue)
                            .symbolEffect(.variableColor.iterative, options: .repeating)
                        Text(aiEditPhaseText)
                            .font(.system(size: 10, weight: .medium))
                            .foregroundStyle(.secondary)
                            .id(aiEditPhaseText)
                            .transition(.push(from: .bottom).combined(with: .opacity))
                    }
                    .animation(.easeInOut(duration: 0.3), value: aiEditPhaseText)
                } else if let aiEditModelDisplayName {
                    Label(aiEditModelDisplayName, systemImage: aiEditUsesLocalMode ? "memorychip" : "brain.head.profile")
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(.tertiary)
                        .lineLimit(1)
                }

                Spacer()

                Button {
                    Task { await runAIEdit() }
                } label: {
                    if isAIEditing {
                        ProgressView()
                            .scaleEffect(0.55)
                            .frame(width: 60, height: 24)
                    } else {
                        Text("Apply")
                            .font(.system(size: 11, weight: .semibold))
                            .frame(width: 60, height: 24)
                    }
                }
                .buttonStyle(.plain)
                .foregroundStyle(.white)
                .background(
                    canRunAIEdit ? Color.blue : Color.gray.opacity(0.35),
                    in: Capsule()
                )
                .disabled(!canRunAIEdit)
            }
        }
        .padding(14)
        .frame(width: 340)
    }

    /// The local model Edit with AI uses in Local mode: the preset's own local
    /// model choice first, then any downloaded model.
    private var aiEditLocalModel: SummaryModelInfoResponse? {
        downloadedLocalModels.first(where: { $0.id == localModelId })
            ?? downloadedLocalModels.first
    }

    private var aiEditUsesLocalMode: Bool {
        activeModelSource == .local
    }

    private var aiEditModelDisplayName: String? {
        aiEditUsesLocalMode
            ? aiEditLocalModel.map { "\($0.displayName) · Local" }
            : aiEditModel?.displayName
    }

    private var canRunAIEdit: Bool {
        guard !isAIEditing,
              !aiEditRequest.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return false
        }
        if aiEditUsesLocalMode {
            return aiEditLocalModel != nil
        }
        guard let aiEditModel else { return false }
        return APIKeychainStore.hasAPIKey(providerId: aiEditModel.providerId)
    }

    private func makeAIEditEngine() throws -> PolishGeneratorEngine {
        if aiEditUsesLocalMode {
            guard let localModel = aiEditLocalModel else {
                throw APIError.serverError("Download a local LLM in Models before editing with AI in Local mode.")
            }
            return PolishGeneratorEngine(
                llm: .local(modelId: localModel.id),
                targetModelName: localModel.displayName,
                targetSupportsVision: localModel.supportsVision
            )
        }
        guard let model = aiEditModel else {
            throw OpenAILanguageModelError.missingAPIKey
        }
        let storedKey = (try? APIKeychainStore.apiKey(providerId: model.providerId)) ?? nil
        guard let apiKey = storedKey, !apiKey.isEmpty else {
            throw APIError.serverError("No API key found for \(model.providerName).")
        }
        return PolishGeneratorEngine(
            llm: .api(model: model, apiKey: apiKey),
            targetModelName: model.displayName,
            targetSupportsVision: model.supportsVision
        )
    }

    private func runAIEdit() async {
        let instruction = aiEditRequest.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !instruction.isEmpty else { return }

        isAIEditing = true
        aiEditError = nil
        aiEditPhaseText = PolishGeneratorPhase.planning.displayText
        defer {
            isAIEditing = false
            aiEditPhaseText = nil
        }

        let snapshot = PolishEditSnapshot(
            displayName: displayName,
            symbolName: symbolName,
            prompt: prompt,
            inputModules: PolishInputModule.allCases.filter { inputModules.contains($0) },
            outputModules: PolishOutputModule.allCases.filter { outputModules.contains($0) },
            pasteBehavior: pasteBehavior
        )

        do {
            let engine = try makeAIEditEngine()
            let package = try await engine.runEdit(current: snapshot, instruction: instruction) { phase in
                aiEditPhaseText = phase.displayText
            }
            showingAIEditPopover = false
            aiEditRequest = ""
            onAIDraft?(package)
        } catch {
            aiEditError = error.localizedDescription
        }
    }

    private var advancedModulePipeline: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Multi-step modules")
                        .font(.system(size: 13, weight: .semibold))
                    Text("Build the context pipeline. Voice and desktop audio can be combined or used separately; clipboard, screenshot, and active app add context.")
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                }

                Spacer()

                if isMultiStepDraft {
                    Text("Multi-step")
                        .font(.system(size: 9, weight: .bold))
                        .foregroundStyle(.purple)
                        .padding(.horizontal, 7)
                        .padding(.vertical, 3)
                        .background(Color.purple.opacity(0.12), in: Capsule())
                }
            }

            HStack(alignment: .top, spacing: 10) {
                moduleStage(
                    title: "Inputs",
                    tint: .blue,
                    modules: PolishInputModule.allCases.map { module in
                        let ignoredUnderBypass = bypassLLM && bypassIgnoredInputModules.contains(module)
                        return ModuleToggleItem(
                            id: module.id,
                            icon: module.symbolName,
                            title: module.label,
                            description: ignoredUnderBypass ? "Ignored while Bypass LLM is on." : module.description,
                            isEnabled: inputModules.contains(module),
                            isLocked: ignoredUnderBypass,
                            action: { toggleInputModule(module) }
                        )
                    }
                )

                pipelineArrow

                VStack(alignment: .leading, spacing: 8) {
                    moduleStage(
                        title: "LLM",
                        tint: .green,
                        modules: [
                            ModuleToggleItem(
                                id: "selected-llm",
                                icon: bypassLLM ? "arrow.right.to.line" : "brain.head.profile",
                                title: bypassLLM ? "Skipped" : selectedModelName,
                                description: bypassLLM
                                    ? "Bypass LLM is on. Raw transcription goes straight to outputs."
                                    : (activeModelSource == .local ? "Uses the selected local LLM above." : "Uses the selected public LLM above."),
                                isEnabled: !bypassLLM,
                                isLocked: true,
                                action: {}
                            )
                        ]
                    )

                    settingsPillToggle(label: "Bypass LLM", icon: "arrow.right.to.line", isOn: $bypassLLM, tint: .orange)
                }
                .frame(maxWidth: .infinity, alignment: .topLeading)

                pipelineArrow

                moduleStage(
                    title: "Outputs",
                    tint: .purple,
                    modules: PolishOutputModule.editableCases.map { module in
                        let requiresScreenshot = module == .copyTextAndScreenshot
                        let canUseModule = !requiresScreenshot ||
                            (inputModules.contains(.screenSnapshot) &&
                             inputModules.contains(.activeApp) &&
                             pasteBehavior == .pasteIntoApp)
                        let lockedDescription = lockedOutputModuleDescription(for: module)
                        return ModuleToggleItem(
                            id: module.id,
                            icon: module.symbolName,
                            title: module.label,
                            description: canUseModule ? module.description : lockedDescription,
                            isEnabled: outputModules.contains(module),
                            isLocked: !canUseModule,
                            action: { toggleOutputModule(module) }
                        )
                    }
                )
            }

            if requiresVisionModelSelection {
                HStack(alignment: .top, spacing: 7) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.system(size: 11, weight: .semibold))
                    Text("Screenshot context requires a Vision-enabled LLM. Edit the LLM card and enable Vision, or choose a different model.")
                        .font(.system(size: 10, weight: .medium))
                        .fixedSize(horizontal: false, vertical: true)
                }
                .foregroundStyle(.orange)
                .padding(10)
                .background(Color.orange.opacity(0.10), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
            }

            if requiresLocalVisionSelection {
                HStack(alignment: .top, spacing: 7) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.system(size: 11, weight: .semibold))
                    Text("Screenshot context needs a vision-capable local LLM. Download or choose a Gemma local LLM with Vision support, or disable Screenshot.")
                        .font(.system(size: 10, weight: .medium))
                        .fixedSize(horizontal: false, vertical: true)
                }
                .foregroundStyle(.orange)
                .padding(10)
                .background(Color.orange.opacity(0.10), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
            }
        }
        .padding(14)
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(Color.purple.opacity(0.16), lineWidth: 0.8)
        )
    }

    private var pipelineArrow: some View {
        Image(systemName: "arrow.right")
            .font(.system(size: 11, weight: .bold))
            .foregroundStyle(.tertiary)
            .frame(width: 18)
            .padding(.top, 40)
    }

    private func moduleStage(
        title: String,
        tint: Color,
        modules: [ModuleToggleItem]
    ) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title.uppercased())
                .font(.system(size: 9, weight: .bold))
                .foregroundStyle(tint)
                .tracking(0.8)

            ForEach(modules) { module in
                Button {
                    module.action()
                } label: {
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: module.icon)
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundStyle(module.isEnabled ? .white : tint)
                            .frame(width: 24, height: 24)
                            .background(module.isEnabled ? tint : tint.opacity(0.12), in: RoundedRectangle(cornerRadius: 7, style: .continuous))

                        VStack(alignment: .leading, spacing: 2) {
                            HStack(spacing: 5) {
                                Text(module.title)
                                    .font(.system(size: 11, weight: .semibold))
                                    .foregroundStyle(.primary)
                                    .lineLimit(1)

                                if module.isLocked {
                                    Image(systemName: "lock.fill")
                                        .font(.system(size: 7, weight: .semibold))
                                        .foregroundStyle(.tertiary)
                                }
                            }

                            Text(module.description)
                                .font(.system(size: 9))
                                .foregroundStyle(.secondary)
                                .lineLimit(2)
                                .multilineTextAlignment(.leading)
                        }

                        Spacer(minLength: 0)
                    }
                    .padding(8)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(module.isEnabled ? tint.opacity(0.10) : Color.primary.opacity(0.035), in: RoundedRectangle(cornerRadius: 11, style: .continuous))
                    .overlay(
                        RoundedRectangle(cornerRadius: 11, style: .continuous)
                            .strokeBorder(module.isEnabled ? tint.opacity(0.35) : Color.white.opacity(0.06), lineWidth: 0.8)
                    )
                }
                .buttonStyle(.plain)
                .disabled(module.isLocked)
            }
        }
        .frame(maxWidth: .infinity, alignment: .topLeading)
    }

    /// Input modules that carry no data into a bypass run: only voice and
    /// desktop audio survive when the LLM step is skipped.
    private let bypassIgnoredInputModules: Set<PolishInputModule> = [.clipboardText, .screenSnapshot, .activeApp]

    private var isMultiStepDraft: Bool {
        Set(orderedInputModules) != Set(PolishPreset.defaultInputModules) ||
        Set(orderedOutputModules) != Set(PolishPreset.defaultOutputModules)
    }

    private var orderedInputModules: [PolishInputModule] {
        PolishInputModule.allCases.filter { module in
            inputModules.contains(module)
        }
    }

    private var orderedOutputModules: [PolishOutputModule] {
        let modules = PolishOutputModule.editableCases.filter { module in
            guard outputModules.contains(module) else { return false }
            if module == .copyToClipboard && outputModules.contains(.copyTextAndScreenshot) {
                return false
            }
            if module == .copyTextAndScreenshot {
                return inputModules.contains(.screenSnapshot)
            }
            return true
        }
        return modules.isEmpty ? PolishPreset.defaultOutputModules : modules
    }

    private var selectedModelName: String {
        if activeModelSource == .local {
            return downloadedLocalModels.first(where: { $0.id == localModelId })?.displayName ?? "Local LLM"
        }
        return selectedAPIModel?.displayName ?? modelId
    }

    private var selectedAPIModel: APILanguageModel? {
        apiLanguageModels.first(where: { $0.modelId == modelId })
    }

    private var selectedLocalModel: SummaryModelInfoResponse? {
        downloadedLocalModels.first(where: { $0.id == localModelId })
    }

    private var normalizedLocalModelId: String? {
        let trimmed = localModelId.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    private func toggleInputModule(_ module: PolishInputModule) {
        if inputModules.contains(module) {
            inputModules.remove(module)
            if module == .screenSnapshot || module == .activeApp {
                outputModules.remove(.copyTextAndScreenshot)
            }
        } else {
            inputModules.insert(module)
        }
        if inputModules.isEmpty {
            inputModules.insert(.voiceTranscript)
        }
    }

    private func toggleOutputModule(_ module: PolishOutputModule) {
        if module == .copyTextAndScreenshot &&
           (!inputModules.contains(.screenSnapshot) ||
            !inputModules.contains(.activeApp) ||
            pasteBehavior != .pasteIntoApp) {
            return
        }
        if outputModules.contains(module) {
            outputModules.remove(module)
        } else {
            if module == .copyTextAndScreenshot {
                outputModules.remove(.copyToClipboard)
            } else if module == .copyToClipboard {
                outputModules.remove(.copyTextAndScreenshot)
            }
            outputModules.insert(module)
        }
        if outputModules.isEmpty {
            outputModules = Set(PolishPreset.defaultOutputModules)
        }
    }

    private func lockedOutputModuleDescription(for module: PolishOutputModule) -> String {
        guard module == .copyTextAndScreenshot else { return module.description }
        if !inputModules.contains(.screenSnapshot) {
            return "Enable Screenshot input first."
        }
        if !inputModules.contains(.activeApp) {
            return "Enable Active app so markdown targets can be detected."
        }
        if pasteBehavior != .pasteIntoApp {
            return "Set delivery to Paste into app first."
        }
        return "Markdown-friendly target required."
    }

    private func save() {
        let selectedModel = apiLanguageModels.first { $0.modelId == modelId }
        if var preset {
            preset.displayName = displayName.trimmingCharacters(in: .whitespacesAndNewlines)
            preset.symbolName = symbolName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "sparkles" : symbolName.trimmingCharacters(in: .whitespacesAndNewlines)
            preset.prompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
            preset.bypassLLM = bypassLLM
            preset.modelId = modelId
            preset.modelProviderId = selectedModel?.providerId ?? preset.modelProviderId
            preset.localModelId = normalizedLocalModelId
            preset.showInMenu = showInMenu
            preset.isEnabled = isEnabled
            preset.outputBehavior = .copyPolished
            preset.pasteBehavior = pasteBehavior
            preset.shortcut = normalizedShortcut
            preset.inputModules = orderedInputModules
            preset.outputModules = orderedOutputModules
            onSave(preset)
        } else {
            var preset = PolishPresetStore.makeCustom(
                displayName: displayName,
                symbolName: symbolName,
                prompt: prompt,
                bypassLLM: bypassLLM,
                modelId: modelId,
                outputBehavior: .copyPolished,
                pasteBehavior: pasteBehavior,
                shortcut: normalizedShortcut
            )
            preset.modelProviderId = selectedModel?.providerId ?? preset.modelProviderId
            preset.localModelId = normalizedLocalModelId
            preset.showInMenu = showInMenu
            preset.isEnabled = isEnabled
            preset.inputModules = orderedInputModules
            preset.outputModules = orderedOutputModules
            onSave(preset)
        }
        onBack()
    }

    private var normalizedShortcut: String? {
        normalizedPolishShortcut(shortcut)
    }
}

/// Shared normalization for Polish shortcut strings: trims whitespace, and an
/// empty result means "no shortcut" (nil). Used by the preset editor here and
/// by the Shortcuts tab registry — keep a single normalizer.
func normalizedPolishShortcut(_ raw: String) -> String? {
    let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
    return trimmed.isEmpty ? nil : trimmed
}

private struct ModuleToggleItem: Identifiable {
    let id: String
    let icon: String
    let title: String
    let description: String
    let isEnabled: Bool
    let isLocked: Bool
    let action: () -> Void
}

private struct PolishImportView: View {
    let apiLanguageModels: [APILanguageModel]
    let selectedModelId: String
    let onBack: () -> Void
    let onCreate: (PolishPreset) -> Void

    @State private var jsonText = ""
    @State private var statusMessage: String?
    @State private var statusIsError = false
    @State private var copiedPrompt = false

    var body: some View {
        VStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 14) {
                header
                    .fixedSize(horizontal: false, vertical: true)

                qualityNotice
                    .fixedSize(horizontal: false, vertical: true)

                HStack(alignment: .top, spacing: 16) {
                    builderPromptCard
                    jsonImportCard
                }
                .frame(maxHeight: .infinity)
            }
            .padding(.horizontal, 28)
            .padding(.top, 8)
            .padding(.bottom, 12)
        }
        .safeAreaInset(edge: .bottom) {
            actionBar
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Button {
                    withAnimation(.easeInOut(duration: 0.15)) { onBack() }
                } label: {
                    HStack(spacing: 5) {
                        Image(systemName: "chevron.left")
                            .font(.system(size: 10, weight: .semibold))
                        Text("Polishes")
                            .font(.system(size: 12, weight: .medium))
                    }
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(.quaternary, in: Capsule())
                }
                .buttonStyle(.plain)

                Spacer()
            }

            HStack(alignment: .center, spacing: 12) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Import Polish")
                        .font(.system(size: 22, weight: .bold, design: .rounded))
                    Text("Use a better external model or AI client to build a strict JSON Polish, then import it here.")
                        .font(.system(size: 12))
                        .foregroundStyle(.secondary)
                }

                Spacer()
            }

            SettingsDivider()
        }
    }

    private var qualityNotice: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "brain.head.profile")
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(.orange)
                .frame(width: 28, height: 28)
                .background(Color.orange.opacity(0.12), in: Circle())

            VStack(alignment: .leading, spacing: 3) {
                Text("Complex Polishes need a strong reasoning model")
                    .font(.system(size: 13, weight: .semibold))
                Text("For advanced workflows, generate JSON with a high-quality external model. Small or fast models can miss module choices, paste constraints, or edge cases.")
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Spacer()
        }
        .padding(13)
        .background(Color.orange.opacity(0.08), in: RoundedRectangle(cornerRadius: 15, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 15, style: .continuous)
                .strokeBorder(Color.orange.opacity(0.18), lineWidth: 0.8)
        )
    }

    private var builderPromptCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: "doc.on.doc")
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(.blue)
                    .frame(width: 30, height: 30)
                    .background(Color.blue.opacity(0.12), in: RoundedRectangle(cornerRadius: 9, style: .continuous))

                VStack(alignment: .leading, spacing: 3) {
                    Text("1. Copy builder prompt")
                        .font(.system(size: 14, weight: .semibold))
                    Text("Paste this into a high-quality AI tool or client, then add your request after the final `User request:` line.")
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }

                Spacer()
            }

            ScrollView {
                Text(builderPromptPreview)
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .topLeading)
                    .padding(12)
                    .textSelection(.enabled)
            }
            .scrollIndicators(.automatic)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Color.primary.opacity(0.045), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .strokeBorder(.white.opacity(0.08), lineWidth: 0.8)
            )

            Color.clear
                .frame(height: 42)

            Button {
                copyBuilderPrompt()
            } label: {
                HStack(spacing: 7) {
                    Image(systemName: copiedPrompt ? "checkmark.circle.fill" : "doc.on.doc")
                        .font(.system(size: 11, weight: .semibold))
                    Text(copiedPrompt ? "Copied" : "Copy Builder Prompt")
                        .font(.system(size: 11, weight: .semibold))
                }
                .foregroundStyle(.white)
                .padding(.horizontal, 13)
                .padding(.vertical, 8)
                .background(copiedPrompt ? Color.green : Color.blue, in: Capsule())
            }
            .buttonStyle(.plain)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    private var jsonImportCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: "curlybraces.square")
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(.green)
                    .frame(width: 30, height: 30)
                    .background(Color.green.opacity(0.12), in: RoundedRectangle(cornerRadius: 9, style: .continuous))

                VStack(alignment: .leading, spacing: 3) {
                    Text("2. Paste JSON")
                        .font(.system(size: 14, weight: .semibold))
                    Text("Paste your crafted JSON here. It will be validated before the Polish is created.")
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }

                Spacer()

                Button {
                    pasteJSONFromClipboard()
                } label: {
                    Text("Paste")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(.green)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(Color.green.opacity(0.10), in: Capsule())
                }
                .buttonStyle(.plain)
            }

            TextEditor(text: $jsonText)
                .font(.system(size: 11, design: .monospaced))
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .scrollContentBackground(.hidden)
                .padding(10)
                .background(Color.primary.opacity(0.045), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .strokeBorder(.white.opacity(0.08), lineWidth: 0.8)
                )
                .onChange(of: jsonText) { _, _ in
                    if statusMessage != nil {
                        statusMessage = nil
                        statusIsError = false
                    }
                }

            Group {
                if let statusMessage {
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: statusIsError ? "exclamationmark.triangle.fill" : "checkmark.circle.fill")
                            .font(.system(size: 11, weight: .semibold))
                        Text(statusMessage)
                            .font(.system(size: 11, weight: .medium))
                            .lineLimit(2)
                            .fixedSize(horizontal: false, vertical: true)
                        Spacer()
                    }
                    .foregroundStyle(statusIsError ? .red : .green)
                    .padding(10)
                    .background((statusIsError ? Color.red : Color.green).opacity(0.08), in: RoundedRectangle(cornerRadius: 11, style: .continuous))
                } else {
                    Color.clear
                }
            }
            .frame(height: 42)

            HStack(spacing: 8) {
                Button {
                    validateOnly()
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: validationSucceeded ? "checkmark.circle.fill" : "checkmark.seal")
                            .font(.system(size: 10, weight: .semibold))
                        Text(validationSucceeded ? "Validated" : "Validate")
                            .font(.system(size: 11, weight: .semibold))
                    }
                    .foregroundStyle(.white)
                    .padding(.horizontal, 13)
                    .padding(.vertical, 8)
                    .background(validateButtonTint, in: Capsule())
                }
                .buttonStyle(.plain)
                .disabled(!canAttemptImport)

                Spacer()
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    private var validationSucceeded: Bool {
        statusMessage != nil && !statusIsError
    }

    private var validateButtonTint: Color {
        if !canAttemptImport {
            return Color.gray.opacity(0.35)
        }
        return validationSucceeded ? Color.green : Color.blue
    }

    private var actionBar: some View {
        HStack(spacing: 10) {
            Spacer()

            Button {
                createPreset()
            } label: {
                Text("Create Polish")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 15)
                    .padding(.vertical, 8)
                    .background(canAttemptImport ? Color.accentColor : Color.gray.opacity(0.35), in: Capsule())
            }
            .buttonStyle(.plain)
            .disabled(!canAttemptImport)
        }
        .padding(.horizontal, 28)
        .padding(.vertical, 12)
        .background(.regularMaterial)
        .overlay(alignment: .top) {
            Rectangle()
                .fill(.white.opacity(0.06))
                .frame(height: 0.7)
        }
    }

    private var canAttemptImport: Bool {
        !jsonText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private var builderPromptPreview: String {
        polishImportBuilderPrompt(models: apiLanguageModels)
    }

    private func copyBuilderPrompt() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(builderPromptPreview, forType: .string)
        copiedPrompt = true
        Task { @MainActor in
            try? await Task.sleep(for: .seconds(1.1))
            copiedPrompt = false
        }
    }

    private func pasteJSONFromClipboard() {
        if let text = NSPasteboard.general.string(forType: .string) {
            jsonText = text
            statusMessage = nil
        }
    }

    private func validateOnly() {
        do {
            _ = try makePresetFromJSON()
            statusMessage = "JSON is valid and ready to import."
            statusIsError = false
        } catch {
            statusMessage = error.localizedDescription
            statusIsError = true
        }
    }

    private func createPreset() {
        do {
            let preset = try makePresetFromJSON()
            onCreate(preset)
        } catch {
            statusMessage = error.localizedDescription
            statusIsError = true
        }
    }

    private func makePresetFromJSON() throws -> PolishPreset {
        let draft = try PolishImportDraft.decode(from: jsonText)
        let selectedModel = apiLanguageModels.first(where: { $0.modelId == selectedModelId }) ?? apiLanguageModels.first
        let model = draft.resolvedModel(from: apiLanguageModels, fallback: selectedModel)
        return try draft.makePreset(model: model)
    }
}

private struct PolishImportDraft: Decodable {
    let displayName: String
    let symbolName: String
    let prompt: String
    let modelProviderId: String?
    let modelId: String?
    let showInMenu: Bool?
    let isEnabled: Bool?
    let pasteBehavior: String?
    let shortcut: String?
    let inputModules: [String]?
    let outputModules: [String]?

    enum CodingKeys: String, CodingKey {
        case displayName = "display_name"
        case symbolName = "symbol_name"
        case prompt
        case modelProviderId = "model_provider_id"
        case modelId = "model_id"
        case showInMenu = "show_in_menu"
        case isEnabled = "is_enabled"
        case pasteBehavior = "paste_behavior"
        case shortcut
        case inputModules = "input_modules"
        case outputModules = "output_modules"
    }

    static func decode(from rawText: String) throws -> PolishImportDraft {
        var text = rawText.trimmingCharacters(in: .whitespacesAndNewlines)
        if text.hasPrefix("```") {
            text = text
                .replacingOccurrences(of: "```json", with: "")
                .replacingOccurrences(of: "```JSON", with: "")
                .replacingOccurrences(of: "```", with: "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }
        guard let data = text.data(using: .utf8) else {
            throw PolishImportError.invalidJSON("Could not read JSON text.")
        }
        do {
            return try JSONDecoder().decode(PolishImportDraft.self, from: data)
        } catch {
            throw PolishImportError.invalidJSON("Invalid JSON or schema mismatch: \(error.localizedDescription)")
        }
    }

    func resolvedModel(from models: [APILanguageModel], fallback: APILanguageModel?) -> APILanguageModel? {
        if let modelId,
           let model = models.first(where: { $0.modelId == modelId && ($0.providerId == modelProviderId || modelProviderId == nil) }) {
            return model
        }
        return fallback
    }

    func makePreset(model: APILanguageModel?) throws -> PolishPreset {
        let name = displayName.trimmingCharacters(in: .whitespacesAndNewlines)
        let cleanedPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else {
            throw PolishImportError.invalidJSON("display_name is required.")
        }
        guard !cleanedPrompt.isEmpty else {
            throw PolishImportError.invalidJSON("prompt is required.")
        }

        let inputs = try decodeInputModules()
        let outputs = try decodeOutputModules()
        let paste = try decodePasteBehavior()

        if outputs.contains(.copyTextAndScreenshot) {
            guard inputs.contains(.screenSnapshot) else {
                throw PolishImportError.invalidJSON("copyTextAndScreenshot requires input_modules to include screenSnapshot.")
            }
            guard inputs.contains(.activeApp) else {
                throw PolishImportError.invalidJSON("copyTextAndScreenshot requires input_modules to include activeApp.")
            }
            guard paste == .pasteIntoApp else {
                throw PolishImportError.invalidJSON("copyTextAndScreenshot requires paste_behavior to be pasteIntoApp.")
            }
        }

        if inputs.contains(.screenSnapshot), let model, !model.supportsVision {
            throw PolishImportError.invalidJSON("This Polish uses screenSnapshot, but the selected/imported LLM is not marked as Vision-capable.")
        }

        let resolvedProviderId = model?.providerId ?? modelProviderId ?? TranscriptionProviderDescriptor.openAI.id
        let resolvedModelId = model?.modelId ?? modelId ?? APILanguageModelStore.defaultModels[0].modelId
        var preset = PolishPresetStore.makeCustom(
            displayName: name,
            symbolName: normalizedSymbolName,
            prompt: cleanedPrompt,
            modelProviderId: resolvedProviderId,
            modelId: resolvedModelId,
            outputBehavior: .copyPolished,
            pasteBehavior: paste,
            shortcut: shortcut,
            inputModules: inputs,
            outputModules: outputs
        )
        preset.showInMenu = showInMenu ?? true
        preset.isEnabled = isEnabled ?? true
        return preset
    }

    private var normalizedSymbolName: String {
        let symbol = symbolName.trimmingCharacters(in: .whitespacesAndNewlines)
        return polishBuilderSymbolNames.contains(symbol) ? symbol : "sparkles"
    }

    private func decodeInputModules() throws -> [PolishInputModule] {
        try decodeModules(
            inputModules ?? PolishPreset.defaultInputModules.map(\.rawValue),
            validValues: PolishInputModule.allCases.map(\.rawValue),
            label: "input_modules",
            transform: PolishInputModule.init(rawValue:)
        )
    }

    private func decodeOutputModules() throws -> [PolishOutputModule] {
        try decodeModules(
            outputModules ?? PolishPreset.defaultOutputModules.map(\.rawValue),
            validValues: PolishOutputModule.editableCases.map(\.rawValue),
            label: "output_modules",
            transform: PolishOutputModule.init(rawValue:)
        )
    }

    private func decodePasteBehavior() throws -> PolishPasteBehavior {
        let value = pasteBehavior ?? PolishPasteBehavior.copyOnly.rawValue
        guard let behavior = PolishPasteBehavior(rawValue: value) else {
            throw PolishImportError.invalidJSON("Unknown paste_behavior `\(value)`. Use copyOnly or pasteIntoApp.")
        }
        return behavior
    }

    private func decodeModules<T>(
        _ rawValues: [String],
        validValues: [String],
        label: String,
        transform: (String) -> T?
    ) throws -> [T] {
        let unknown = rawValues.filter { !validValues.contains($0) }
        guard unknown.isEmpty else {
            throw PolishImportError.invalidJSON("Unknown \(label): \(unknown.joined(separator: ", ")).")
        }
        let modules = rawValues.compactMap(transform)
        guard !modules.isEmpty else {
            throw PolishImportError.invalidJSON("\(label) cannot be empty.")
        }
        return modules
    }
}

private enum PolishImportError: LocalizedError {
    case invalidJSON(String)

    var errorDescription: String? {
        switch self {
        case .invalidJSON(let message):
            return message
        }
    }
}

private func polishImportBuilderPrompt(models: [APILanguageModel]) -> String {
    let modelLines = models.map { model in
        "- provider: \(model.providerId), model_id: \(model.modelId), display: \(model.displayName), vision: \(model.supportsVision ? "true" : "false")"
    }.joined(separator: "\n")
    let icons = polishBuilderSymbolNames.joined(separator: ", ")
    return """
    You are creating an AiTranscribe Polish preset as strict JSON.

    AiTranscribe Polish environment:
    - A Polish runs after transcription and transforms or answers from runtime context.
    - Runtime context can include voice transcript, desktop audio transcript, exact clipboard text, screenshot/image region, and active app name.
    - The prompt you write will be executed later. Do not write a prompt about creating a Polish; write the runtime instruction that the selected LLM should follow.
    - Complex Polishes need strong reasoning models. Ask clarifying questions in normal chat before final JSON if the user request is underspecified.

    Available input_modules:
    - voiceTranscript: microphone/user speech transcript.
    - desktopAudio: system audio transcript, separate from user voice.
    - clipboardText: exact copied text from clipboard.
    - screenSnapshot: selected screenshot/image region; requires a vision-capable LLM.
    - activeApp: focused app name for formatting and destination behavior.

    Available output_modules:
    - copyToClipboard: copy final text output only.
    - copyTextAndScreenshot: paste screenshot first, then final text. Use only with screenSnapshot + activeApp + pasteIntoApp and markdown-friendly target apps.
    - floatingWindow: show final answer in the floating response window for reading, explanation, diagnostics, or review.

    paste_behavior:
    - copyOnly: default. Result stays in clipboard or floating window.
    - pasteIntoApp: only when user explicitly wants automatic paste/insert, or when using copyTextAndScreenshot.

    Available LLMs in this AiTranscribe install:
    \(modelLines.isEmpty ? "- No public LLMs listed. Omit model_provider_id and model_id to use the current app default." : modelLines)

    Common SF Symbol icons you may use:
    \(icons)

    Strict JSON schema:
    {
      "display_name": "Short name under 24 chars",
      "symbol_name": "valid SF Symbol from the list",
      "prompt": "Detailed runtime instruction. It must say how to use available context and what to output.",
      "model_provider_id": "optional provider id from available LLMs",
      "model_id": "optional model id from available LLMs",
      "input_modules": ["voiceTranscript"],
      "output_modules": ["copyToClipboard"],
      "paste_behavior": "copyOnly",
      "show_in_menu": true,
      "is_enabled": true,
      "shortcut": null
    }

    Hard rules:
    - Return only JSON. No markdown fences and no explanation.
    - input_modules may only use: voiceTranscript, desktopAudio, clipboardText, screenSnapshot, activeApp.
    - output_modules may only use: copyToClipboard, copyTextAndScreenshot, floatingWindow.
    - Do not include both copyToClipboard and copyTextAndScreenshot.
    - If output_modules includes copyTextAndScreenshot, input_modules must include screenSnapshot and activeApp, and paste_behavior must be pasteIntoApp.
    - If input_modules includes screenSnapshot, select a vision-capable model when one is available.
    - If the user asks for an explainer/Q&A/diagnostic, usually use floatingWindow.
    - If the user asks for prompt polish, grammar, rewrite, notes, task extraction, or generated text for another app, usually use copyToClipboard.
    - If context may be missing at runtime, the prompt must say not to invent missing context.
    - The prompt must output only the final result, with no meta-commentary.

    User request:
    """
}

private let polishBuilderSymbolNames: [String] = [
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

private struct PolishIconGridPicker: View {
    @Binding var selectedSymbol: String
    let onPick: () -> Void

    private let columns = Array(repeating: GridItem(.fixed(64), spacing: 12), count: 5)

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Choose Icon")
                        .font(.system(size: 14, weight: .semibold, design: .rounded))
                    Text("Used in Settings and the menu Polish buttons.")
                        .font(.system(size: 10))
                        .foregroundStyle(.secondary)
                }

                Spacer()
            }
            .padding(.horizontal, 4)
            .padding(.top, 2)

            ScrollView {
                LazyVGrid(columns: columns, spacing: 12) {
                    ForEach(polishBuilderSymbolNames, id: \.self) { symbol in
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
                            .frame(width: 62, height: 74)
                            .background(
                                selectedSymbol == symbol ? Color.blue : Color.primary.opacity(0.045),
                                in: RoundedRectangle(cornerRadius: 16, style: .continuous)
                            )
                            .overlay(
                                RoundedRectangle(cornerRadius: 16, style: .continuous)
                                    .strokeBorder(selectedSymbol == symbol ? Color.blue.opacity(0.7) : Color.white.opacity(0.08), lineWidth: 0.8)
                            )
                        }
                        .buttonStyle(.plain)
                        .help(symbol)
                    }
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 14)
            }
            .frame(height: 380)
            .background(Color.primary.opacity(0.025), in: RoundedRectangle(cornerRadius: 18, style: .continuous))
        }
        .padding(26)
        .frame(width: 438)
    }
}

// MARK: - Generated Draft Review

private struct PolishDraftReviewView: View {
    let package: PolishGeneratorDraftPackage
    let targetModelName: String?
    let onDiscard: () -> Void
    let onSave: (PolishGeneratorDraftPackage) -> Void

    @State private var displayName: String
    @State private var prompt: String

    init(
        package: PolishGeneratorDraftPackage,
        targetModelName: String?,
        onDiscard: @escaping () -> Void,
        onSave: @escaping (PolishGeneratorDraftPackage) -> Void
    ) {
        self.package = package
        self.targetModelName = targetModelName
        self.onDiscard = onDiscard
        self.onSave = onSave
        _displayName = State(initialValue: package.displayName)
        _prompt = State(initialValue: package.prompt)
    }

    private var canSave: Bool {
        !displayName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty &&
        !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                header
                scoreCard
                pipelineCard
                promptCard
            }
            .padding(.horizontal, 28)
            .padding(.top, 8)
            .padding(.bottom, 12)
        }
        .safeAreaInset(edge: .bottom) {
            actionBar
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Button {
                    withAnimation(.easeInOut(duration: 0.15)) { onDiscard() }
                } label: {
                    HStack(spacing: 5) {
                        Image(systemName: "chevron.left")
                            .font(.system(size: 10, weight: .semibold))
                        Text("Polishes")
                            .font(.system(size: 12, weight: .medium))
                    }
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(.quaternary, in: Capsule())
                }
                .buttonStyle(.plain)

                Spacer()
            }

            HStack(alignment: .center, spacing: 12) {
                Image(systemName: package.symbolName)
                    .font(.system(size: 19, weight: .semibold))
                    .foregroundStyle(.blue)
                    .frame(width: 42, height: 42)
                    .background(Color.blue.opacity(0.12), in: RoundedRectangle(cornerRadius: 12, style: .continuous))

                VStack(alignment: .leading, spacing: 4) {
                    TextField("Polish name", text: $displayName)
                        .textFieldStyle(.plain)
                        .font(.system(size: 22, weight: .bold, design: .rounded))
                    Text("Review the generated draft. Nothing is saved until you keep it.")
                        .font(.system(size: 12))
                        .foregroundStyle(.secondary)
                }

                Spacer()
            }

            if let goalSummary = package.goalSummary, !goalSummary.isEmpty {
                Text(goalSummary)
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(11)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
            }

            SettingsDivider()
        }
    }

    private var scoreCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Eval scorecard")
                    .font(.system(size: 13, weight: .semibold))

                Spacer()

                if package.repairRounds > 0 {
                    Text("\(package.repairRounds) repair \(package.repairRounds == 1 ? "round" : "rounds")")
                        .font(.system(size: 9, weight: .semibold))
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 7)
                        .padding(.vertical, 3)
                        .background(.quaternary, in: Capsule())
                }

                Text("\(package.passedCount)/\(package.checks.count) passed")
                    .font(.system(size: 10, weight: .bold))
                    .foregroundStyle(package.checks.allSatisfy(\.pass) ? Color.green : Color.orange)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(
                        (package.checks.allSatisfy(\.pass) ? Color.green : Color.orange).opacity(0.12),
                        in: Capsule()
                    )
            }

            VStack(alignment: .leading, spacing: 7) {
                ForEach(package.checks) { check in
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: check.pass ? "checkmark.circle.fill" : "xmark.circle.fill")
                            .font(.system(size: 12, weight: .semibold))
                            .foregroundStyle(check.pass ? Color.green : Color.red)
                            .padding(.top, 1)

                        VStack(alignment: .leading, spacing: 1) {
                            Text(check.criterion)
                                .font(.system(size: 11, weight: .medium))
                                .fixedSize(horizontal: false, vertical: true)
                            Text(check.reason)
                                .font(.system(size: 9))
                                .foregroundStyle(.secondary)
                                .fixedSize(horizontal: false, vertical: true)
                        }

                        Spacer(minLength: 0)
                    }
                }

                ForEach(package.deterministicWarnings, id: \.self) { warning in
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundStyle(.orange)
                            .padding(.top, 1)

                        Text(warning)
                            .font(.system(size: 10, weight: .medium))
                            .foregroundStyle(.orange)
                            .fixedSize(horizontal: false, vertical: true)

                        Spacer(minLength: 0)
                    }
                }
            }
        }
        .padding(14)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(.white.opacity(0.08), lineWidth: 0.8)
        )
    }

    private var pipelineCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Workflow")
                .font(.system(size: 13, weight: .semibold))

            pipelineRow(title: "Inputs", tint: .blue, labels: package.inputModules.map(\.label))
            pipelineRow(title: "Outputs", tint: .purple, labels: package.outputModules.map(\.label))

            HStack(spacing: 14) {
                Label(package.pasteBehavior == .pasteIntoApp ? "Auto-pastes into the focused app" : "Copy only",
                      systemImage: package.pasteBehavior == .pasteIntoApp ? "arrow.down.doc" : "doc.on.clipboard")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(.secondary)

                if let targetModelName {
                    Label(targetModelName, systemImage: "brain.head.profile")
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(.secondary)
                }
            }

            if let rationale = package.rationale, !rationale.isEmpty {
                Text(rationale)
                    .font(.system(size: 10))
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(14)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(.white.opacity(0.08), lineWidth: 0.8)
        )
    }

    private func pipelineRow(title: String, tint: Color, labels: [String]) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(title)
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(.secondary)
                .frame(width: 52, alignment: .leading)

            HStack(spacing: 5) {
                ForEach(labels, id: \.self) { label in
                    Text(label)
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(tint)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3)
                        .background(tint.opacity(0.11), in: Capsule())
                }
            }
        }
    }

    private var promptCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Runtime prompt")
                    .font(.system(size: 13, weight: .semibold))
                Spacer()
                Text("Editable")
                    .font(.system(size: 9, weight: .semibold))
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 7)
                    .padding(.vertical, 3)
                    .background(.quaternary, in: Capsule())
            }

            TextEditor(text: $prompt)
                .font(.system(size: 12))
                .frame(height: 220)
                .scrollContentBackground(.hidden)
                .padding(10)
                .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .strokeBorder(.white.opacity(0.08), lineWidth: 0.8)
                )
        }
        .padding(14)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(.white.opacity(0.08), lineWidth: 0.8)
        )
    }

    private var actionBar: some View {
        HStack(spacing: 10) {
            Button {
                withAnimation(.easeInOut(duration: 0.15)) { onDiscard() }
            } label: {
                Text("Discard")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 18)
                    .frame(height: 30)
                    .background(.quaternary, in: Capsule())
            }
            .buttonStyle(.plain)

            Spacer()

            Button {
                var finalPackage = package
                finalPackage.displayName = displayName
                finalPackage.prompt = prompt
                withAnimation(.easeInOut(duration: 0.15)) { onSave(finalPackage) }
            } label: {
                Text("Save Polish")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 20)
                    .frame(height: 30)
                    .background(canSave ? Color.blue : Color.gray.opacity(0.35), in: Capsule())
            }
            .buttonStyle(.plain)
            .disabled(!canSave)
        }
        .padding(.horizontal, 28)
        .padding(.vertical, 12)
        .background(.ultraThinMaterial)
    }
}
