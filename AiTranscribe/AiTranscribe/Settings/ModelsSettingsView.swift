import SwiftUI

// MARK: - Models Settings View

struct ModelsSettingsView: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var backendManager: BackendManager

    let hasAnimated: Bool
    let onAnimated: () -> Void
    let initialMode: ModelMode

    /// Real-time streaming toggle
    @AppStorage("realTimeStreaming") private var realTimeStreaming = false

    /// Selected quick-transcription provider
    @AppStorage("selectedTranscriptionProviderId") private var selectedTranscriptionProviderId = TranscriptionProviderDescriptor.localBackend.id

    /// Selected public speech-to-text model. Kept under the original key for migration safety.
    @AppStorage("selectedOpenAITranscriptionModel") private var selectedOpenAITranscriptionModel = TranscriptionModelDescriptor.openAIModels[0].id

    /// Selected OpenAI language model for public summaries / LLM workflows
    @AppStorage("selectedOpenAILanguageModel") private var selectedOpenAILanguageModel = APILanguageModelStore.defaultModels[0].modelId

    /// Current mode: speech-to-text or summarization
    @State private var selectedMode: ModelMode = .speechToText


    /// Filter for model list
    @State private var selectedFilter: ModelFilter = .all

    /// Search query
    @State private var searchQuery = ""

    /// Show NeMo setup sheet
    @State private var showingNemoSetup = false

    /// Show Summary runtime setup sheet
    @State private var showingSummarySetup = false

    /// Whether we're still checking NeMo status
    @State private var isCheckingNemoStatus = true

    /// Animation state — init from hasAnimated so first frame is correct
    @State private var appeared: Bool

    @State private var showingAddAPIModel = false
    @State private var showingAddAPILanguageModel = false
    @State private var editingAPIModel: APITranscriptionModel?
    @State private var editingAPILanguageModel: APILanguageModel?
    @State private var apiModels: [APITranscriptionModel] = APITranscriptionModelStore.load()
    @State private var apiLanguageModels: [APILanguageModel] = APILanguageModelStore.load()

    init(hasAnimated: Bool, onAnimated: @escaping () -> Void, initialMode: ModelMode = .speechToText) {
        self.hasAnimated = hasAnimated
        self.onAnimated = onAnimated
        self.initialMode = initialMode
        _appeared = State(initialValue: hasAnimated)
    }

    // MARK: - Computed

    private var effectiveNemoAvailable: Bool {
        // Trust the backend's nemoAvailable flag (it knows if NeMo actually works)
        // OR check if the venv exists locally (production install)
        // OR if a NeMo model is currently loaded (proof it works)
        appState.nemoAvailable || appState.nemoVenvExists || nemoModelIsLoaded
    }

    /// If a NeMo model is loaded, NeMo is definitively available
    private var nemoModelIsLoaded: Bool {
        guard let loadedId = appState.loadedModelId else { return false }
        return appState.availableModels.first(where: { $0.id == loadedId })?.nemoRequired == true
    }

    private var downloadedSTTModels: [ModelInfoResponse] {
        appState.availableModels.filter { $0.downloaded }
    }

    private var totalSTTSizeMB: Int {
        downloadedSTTModels.reduce(0) { $0 + $1.sizeMB }
    }

    private var localModelsSizeString: String {
        if totalSTTSizeMB >= 1024 {
            return String(format: "%.1f GB", Double(totalSTTSizeMB) / 1024.0)
        } else if totalSTTSizeMB > 0 {
            return "\(totalSTTSizeMB) MB"
        }
        return "Zero KB"
    }

    /// Filter + search speech-to-text models
    private var filteredModels: [ModelInfoResponse] {
        var models = appState.availableModels

        // Apply filter
        switch selectedFilter {
        case .all:
            break
        case .recommended:
            models = models.filter { isRecommended($0) }
        case .downloaded:
            models = models.filter { $0.downloaded }
        }

        // Apply search
        if !searchQuery.isEmpty {
            models = models.filter {
                $0.displayName.localizedCaseInsensitiveContains(searchQuery) ||
                $0.author.localizedCaseInsensitiveContains(searchQuery) ||
                $0.description.localizedCaseInsensitiveContains(searchQuery)
            }
        }

        return models
    }

    /// Group filtered models by category
    private var streamingModels: [ModelInfoResponse] {
        filteredModels.filter { $0.streamingNative }
    }

    private var nvidiaModels: [ModelInfoResponse] {
        filteredModels.filter { $0.author == "NVIDIA" && !$0.streamingNative }
    }

    private var openaiModels: [ModelInfoResponse] {
        filteredModels.filter { $0.author == "OpenAI" }
    }

    private var summaryModels: [SummaryModelInfoResponse] {
        appState.availableSummaryModels
    }

    private var isPublicMode: Bool {
        selectedTranscriptionProviderId != TranscriptionProviderDescriptor.localBackend.id
    }

    private var selectedPublicProviderName: String {
        switch selectedTranscriptionProviderId {
        case TranscriptionProviderDescriptor.openAI.id:
            return TranscriptionProviderDescriptor.openAI.displayName
        case TranscriptionProviderDescriptor.openRouter.id:
            return TranscriptionProviderDescriptor.openRouter.displayName
        case TranscriptionProviderDescriptor.elevenLabs.id:
            return TranscriptionProviderDescriptor.elevenLabs.displayName
        default:
            return "Public"
        }
    }

    private var selectedPublicContextName: String {
        if selectedMode == .speechToText {
            return selectedPublicProviderName
        }
        return apiLanguageModels.first(where: { $0.modelId == selectedOpenAILanguageModel })?.providerName ?? "Public LLM"
    }

    private func isRecommended(_ model: ModelInfoResponse) -> Bool {
        ["parakeet-tdt-v2", "parakeet-v2", "whisper-large-v3-turbo", "large-v3-turbo", "nemotron-streaming"].contains(model.id)
    }

    // MARK: - Body

    var body: some View {
        ScrollView {
            VStack(spacing: 0) {
                // Top: mode-aware models header
                ModelsHeaderBar(
                    title: isPublicMode ? "API Models" : "Local Models",
                    subtitle: isPublicMode
                        ? "\(selectedMode == .speechToText ? apiModels.count : apiLanguageModels.count) configured · \(selectedPublicContextName) active"
                        : "\(downloadedSTTModels.count) downloaded · \(localModelsSizeString)",
                    isPublicMode: isPublicMode,
                    downloadedCount: downloadedSTTModels.count,
                    totalCount: appState.availableModels.count,
                    totalSizeMB: totalSTTSizeMB,
                    apiModelCount: apiModels.count,
                    addModelTitle: selectedMode == .speechToText ? "Add Model" : "Add LLM",
                    onAddAPIModel: {
                        if selectedMode == .speechToText {
                            showingAddAPIModel = true
                        } else {
                            showingAddAPILanguageModel = true
                        }
                    },
                    onRefresh: {
                        Task {
                            await appState.fetchAvailableModels()
                            await appState.fetchSummaryModels()
                        }
                    }
                )
                .staggerIn(index: 0, appeared: appeared)

                // Mode switcher
                ModeSwitcher(selectedMode: $selectedMode)
                    .padding(.top, 16)
                    .staggerIn(index: 1, appeared: appeared)

                SettingsDivider()
                    .padding(.top, 12)
                    .staggerIn(index: 2, appeared: appeared)

                // Content based on selected mode
                if selectedMode == .speechToText {
                    speechToTextContent
                } else {
                    summarizationContent
                }

                Spacer(minLength: 20)
            }
            .padding(.horizontal, 28)
            .padding(.top, 8)
        }
        .scrollIndicators(.automatic)
        .onAppear {
            selectedMode = initialMode
            apiModels = APITranscriptionModelStore.load()
            apiLanguageModels = APILanguageModelStore.load()
            reconcilePublicSelections()
            Task { await appState.fetchAvailableModels() }
            Task { await appState.fetchSummaryModels() }
            Task { await appState.fetchSummaryRuntimeStatus() }
            Task {
                // checkNemoStatus calls refreshNemoVenvState internally
                // and updates nemoAvailable + nemoVenvExists
                await appState.checkNemoStatus()
                // Only clear checking state AFTER the backend responded
                isCheckingNemoStatus = false
            }
        }
        .onChange(of: appState.isServerConnected) { _, isConnected in
            if isConnected && appState.availableModels.isEmpty {
                Task { await appState.fetchAvailableModels() }
            }
            if isConnected && appState.availableSummaryModels.isEmpty {
                Task { await appState.fetchSummaryModels() }
                Task { await appState.fetchSummaryRuntimeStatus() }
            }
        }
        .sheet(isPresented: $showingNemoSetup) {
            NemoSetupView(
                setupManager: appState.nemoSetupManager,
                onComplete: {
                    Task {
                        await appState.checkNemoStatus()
                        await appState.fetchAvailableModels()
                    }
                }
            )
            .environmentObject(appState)
            .environmentObject(backendManager)
        }
        .sheet(isPresented: $showingSummarySetup) {
            SummarySetupView(setupManager: appState.summarySetupManager)
                .environmentObject(appState)
        }
        .sheet(isPresented: $showingAddAPIModel) {
            AddAPIModelSheet { model in
                apiModels = APITranscriptionModelStore.upsert(model)
                selectedTranscriptionProviderId = model.providerId
                selectedOpenAITranscriptionModel = model.modelId
                reconcilePublicSelections()
            }
        }
        .sheet(item: $editingAPIModel) { model in
            AddAPIModelSheet(model: model) { updated in
                apiModels = APITranscriptionModelStore.upsert(updated)
                selectedTranscriptionProviderId = updated.providerId
                selectedOpenAITranscriptionModel = updated.modelId
                reconcilePublicSelections()
            }
        }
        .sheet(isPresented: $showingAddAPILanguageModel) {
            AddAPILanguageModelSheet { model in
                apiLanguageModels = APILanguageModelStore.upsert(model)
                selectedOpenAILanguageModel = model.modelId
                reconcilePublicSelections()
            }
        }
        .sheet(item: $editingAPILanguageModel) { model in
            AddAPILanguageModelSheet(model: model) { updated in
                apiLanguageModels = APILanguageModelStore.upsert(updated)
                selectedOpenAILanguageModel = updated.modelId
                reconcilePublicSelections()
            }
        }
        .task(id: "stagger") {
            guard !hasAnimated else { return }
            try? await Task.sleep(for: .milliseconds(80))
            appeared = true
            onAnimated()
        }
        .onChange(of: selectedMode) { _, _ in
            reconcilePublicSelections()
        }
        .onChange(of: selectedTranscriptionProviderId) { _, _ in
            reconcilePublicSelections()
        }
    }

    private func reconcilePublicSelections() {
        guard isPublicMode else { return }

        if selectedMode == .speechToText {
            let supportedModels = apiModels.filter {
                APITranscriptionModelStore.isSupportedModel($0.modelId, providerId: $0.providerId)
            }
            let providerModels = supportedModels.filter { $0.providerId == selectedTranscriptionProviderId }
            if providerModels.contains(where: { $0.modelId == selectedOpenAITranscriptionModel }) {
                return
            }
            if let fallback = providerModels.first ?? supportedModels.first {
                selectedTranscriptionProviderId = fallback.providerId
                selectedOpenAITranscriptionModel = fallback.modelId
            }
            return
        }

        guard !apiLanguageModels.contains(where: { $0.modelId == selectedOpenAILanguageModel }) else {
            return
        }
        if let fallback = apiLanguageModels.first {
            selectedOpenAILanguageModel = fallback.modelId
        }
    }

    // MARK: - Speech-to-Text Content

    @ViewBuilder
    private var speechToTextContent: some View {
        if isPublicMode {
            apiModelsContent
        } else {
            localModelsContent
        }
    }

    @ViewBuilder
    private var localModelsContent: some View {
        // Search + filter row
        SearchAndFilterBar(
            searchQuery: $searchQuery,
            selectedFilter: $selectedFilter,
            allCount: appState.availableModels.count,
            recommendedCount: appState.availableModels.filter { isRecommended($0) }.count,
            downloadedCount: downloadedSTTModels.count
        )
        .padding(.top, 12)
        .staggerIn(index: 4, appeared: appeared)

        if appState.availableModels.isEmpty {
            modelLoadingState
                .staggerIn(index: 5, appeared: appeared)
        } else if filteredModels.isEmpty {
            emptySearchState
                .staggerIn(index: 5, appeared: appeared)
        } else {
            // Model grid sections
            VStack(spacing: 16) {
                if !streamingModels.isEmpty {
                    ModelSectionView(
                        title: "Real-Time Streaming",
                        icon: "waveform",
                        tint: .orange,
                        columns: 1
                    ) {
                        ForEach(streamingModels) { model in
                            compactModelCard(model)
                        }
                    }
                }

                if !nvidiaModels.isEmpty {
                    ModelSectionView(
                        title: "NVIDIA Parakeet",
                        icon: "cpu",
                        tint: .green,
                        columns: 2
                    ) {
                        ForEach(nvidiaModels) { model in
                            compactModelCard(model)
                        }
                    }
                }

                if !openaiModels.isEmpty {
                    ModelSectionView(
                        title: "OpenAI Whisper",
                        icon: "globe",
                        tint: .blue,
                        columns: 2
                    ) {
                        ForEach(openaiModels) { model in
                            compactModelCard(model)
                        }
                    }
                }
            }
            .padding(.top, 12)
            .staggerIn(index: 5, appeared: appeared)
        }

        // Storage footer
        storageFooter
            .padding(.top, 12)
            .staggerIn(index: 6, appeared: appeared)
    }

    @ViewBuilder
    private var apiModelsContent: some View {
        if apiModels.isEmpty {
            apiModelsEmptyState
                .padding(.top, 18)
                .staggerIn(index: 3, appeared: appeared)
        } else {
            ModelSectionView(
                title: "Public Speech-to-Text",
                icon: "network",
                tint: .blue,
                columns: 2
            ) {
                ForEach(apiModels) { model in
                    APIModelCardView(
                        model: model,
                        isSelected: isPublicMode &&
                            selectedTranscriptionProviderId == model.providerId &&
                            selectedOpenAITranscriptionModel == model.modelId,
                        onUse: {
                            selectedTranscriptionProviderId = model.providerId
                            selectedOpenAITranscriptionModel = model.modelId
                        },
                        onEdit: {
                            editingAPIModel = model
                        },
                        onDelete: {
                            apiModels = APITranscriptionModelStore.delete(id: model.id)
                            if selectedTranscriptionProviderId == model.providerId &&
                                selectedOpenAITranscriptionModel == model.modelId {
                                if let fallback = apiModels.first(where: {
                                    APITranscriptionModelStore.isSupportedModel($0.modelId, providerId: $0.providerId)
                                }) {
                                    selectedTranscriptionProviderId = fallback.providerId
                                    selectedOpenAITranscriptionModel = fallback.modelId
                                } else {
                                    selectedTranscriptionProviderId = TranscriptionProviderDescriptor.localBackend.id
                                    selectedOpenAITranscriptionModel = TranscriptionModelDescriptor.openAIModels[0].id
                                }
                            }
                        }
                    )
                }
            }
            .padding(.top, 16)
            .staggerIn(index: 3, appeared: appeared)
        }
    }

    private var apiModelsEmptyState: some View {
        VStack(spacing: 14) {
            Image(systemName: "network.badge.shield.half.filled")
                .font(.system(size: 30, weight: .medium))
                .foregroundStyle(.secondary)

            VStack(spacing: 4) {
                Text("No API models configured")
                    .font(.system(size: 15, weight: .semibold))
                Text("Add an OpenAI or OpenRouter transcription model to use Public mode for quick recordings.")
                    .font(.system(size: 12))
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }

            Text("Use Add Model to connect a public Speech-to-Text provider.")
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(.tertiary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 42)
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
    }

    // MARK: - Summarization Content

    @ViewBuilder
    private var summarizationContent: some View {
        if isPublicMode {
            apiSummarizationContent
                .padding(.top, 18)
                .staggerIn(index: 3, appeared: appeared)
        } else if !summaryModels.isEmpty {
            VStack(spacing: 12) {
                HStack {
                    SettingsSectionHeader(title: "Local LLM")
                    Spacer()
                    Button {
                        Task { await appState.fetchSummaryModels() }
                    } label: {
                        Image(systemName: "arrow.clockwise")
                            .font(.system(size: 12))
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                }

                ForEach(summaryModels) { model in
                    SummaryModelCard(
                        model: model,
                        isDownloading: appState.downloadingSummaryModelId == model.id,
                        downloadProgress: appState.summaryDownloadProgress,
                        runtimeInstalled: appState.summaryRuntimeInstalled,
                        onDownload: {
                            Task { _ = await appState.downloadSummaryModel(modelId: model.id) }
                        },
                        onDelete: {
                            Task { await appState.deleteSummaryModel(modelId: model.id) }
                        }
                    )
                }
            }
            .padding(.top, 12)
            .staggerIn(index: 3, appeared: appeared)
        } else if !appState.summaryRuntimeInstalled {
            // No models and no runtime — prompt to install via pill
            VStack(spacing: 8) {
                Image(systemName: "sparkles")
                    .font(.system(size: 24))
                    .foregroundStyle(.tertiary)
                Text("Summary runtime not installed")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(.secondary)
                Text("Click the LLM pill above to set up")
                    .font(.system(size: 11))
                    .foregroundStyle(.tertiary)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 40)
            .staggerIn(index: 3, appeared: appeared)
        }
    }

    private var apiSummarizationContent: some View {
        Group {
            if apiLanguageModels.isEmpty {
                VStack(spacing: 14) {
                    Image(systemName: "sparkles.rectangle.stack")
                        .font(.system(size: 30, weight: .medium))
                        .foregroundStyle(.secondary)

                    VStack(spacing: 4) {
                        Text("No API LLM configured")
                            .font(.system(size: 15, weight: .semibold))
                        Text("Add a public LLM for API summaries and future polishing. Local LLMs stay hidden in Public mode.")
                            .font(.system(size: 12))
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.center)
                    }

                    Text("Use the Add LLM button in the header to connect OpenAI.")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(.tertiary)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 42)
                .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
            } else {
                ModelSectionView(
                    title: "OpenAI LLM",
                    icon: "sparkles",
                    tint: .orange,
                    columns: 2
                ) {
                    ForEach(apiLanguageModels) { model in
                        APILanguageModelCardView(
                            model: model,
                            isSelected: isPublicMode && selectedOpenAILanguageModel == model.modelId,
                            onUse: {
                                selectedOpenAILanguageModel = model.modelId
                            },
                            onEdit: {
                                editingAPILanguageModel = model
                            },
                            onDelete: {
                                apiLanguageModels = APILanguageModelStore.delete(id: model.id)
                                if selectedOpenAILanguageModel == model.modelId {
                                    selectedOpenAILanguageModel = apiLanguageModels.first?.modelId ?? APILanguageModelStore.defaultModels[0].modelId
                                }
                                reconcilePublicSelections()
                            }
                        )
                    }
                }
            }
        }
    }

    // MARK: - Loading / Empty States

    private var modelLoadingState: some View {
        VStack(spacing: 12) {
            ProgressView()
                .scaleEffect(1.1)
            Text(appState.isServerConnected ? "Loading models..." : "Server Starting...")
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(.primary)
            Text(appState.isServerConnected ? "Fetching available models" : "Models will appear once the backend is ready.")
                .font(.system(size: 11))
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 40)
    }

    private var emptySearchState: some View {
        VStack(spacing: 8) {
            Image(systemName: "magnifyingglass")
                .font(.system(size: 24))
                .foregroundStyle(.tertiary)
            Text("No models match your search")
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 40)
    }

    // MARK: - Compact Model Card

    @ViewBuilder
    private func compactModelCard(_ model: ModelInfoResponse) -> some View {
        CompactModelCardView(
            model: model,
            isLoaded: !isPublicMode && appState.loadedModelId == model.id,
            isLoading: appState.loadingModelId == model.id,
            isDownloading: appState.downloadingModelId == model.id,
            nemoAvailable: effectiveNemoAvailable,
            isCheckingNemoStatus: isCheckingNemoStatus,
            realTimeStreaming: $realTimeStreaming,
            onLoad: {
                selectedTranscriptionProviderId = TranscriptionProviderDescriptor.localBackend.id
                appState.setPreferredModel(model.id)
                await appState.loadModel(modelId: model.id)
            },
            onUnload: { await appState.unloadModel() },
            onDownload: { _ = await appState.downloadModel(modelId: model.id) },
            onDelete: { await appState.deleteModel(modelId: model.id) },
            onStreamingChanged: { enabled in
                if enabled {
                    appState.checkAccessibilityPermissionsIfNeeded()
                }
            },
            onInstallNemo: { showingNemoSetup = true }
        )
    }

    // MARK: - Storage Footer

    private var storageFooter: some View {
        HStack(spacing: 8) {
            Image(systemName: "info.circle")
                .font(.system(size: 11))
                .foregroundStyle(.tertiary)
            Text("Only one model can be loaded at a time")
                .font(.system(size: 11))
                .foregroundStyle(.tertiary)
            Spacer()
            Button {
                let path = NSString(string: "~/.cache/huggingface/hub").expandingTildeInPath
                let url = URL(fileURLWithPath: path)
                try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
                NSWorkspace.shared.selectFile(nil, inFileViewerRootedAtPath: url.path)
            } label: {
                HStack(spacing: 4) {
                    Image(systemName: "folder")
                    Text("Open Cache")
                }
                .font(.system(size: 10, weight: .medium))
                .foregroundStyle(.secondary)
                .padding(.horizontal, 10)
                .padding(.vertical, 5)
                .background(.thinMaterial, in: Capsule())
            }
            .buttonStyle(.plain)
        }
        .padding(.vertical, 8)
    }
}

// MARK: - Enums

enum ModelMode: String, CaseIterable {
    case speechToText = "Speech-to-Text"
    case summarization = "LLM"

    var icon: String {
        switch self {
        case .speechToText: return "waveform"
        case .summarization: return "sparkles"
        }
    }
}

enum ModelFilter: String, CaseIterable {
    case all = "All"
    case recommended = "Recommended"
    case downloaded = "Downloaded"
}

// MARK: - Models Header Bar

/// Top bar with download stats, storage info, support pills, and refresh
struct ModelsHeaderBar: View {
    let title: String
    let subtitle: String
    let isPublicMode: Bool
    let downloadedCount: Int
    let totalCount: Int
    let totalSizeMB: Int
    let apiModelCount: Int
    let addModelTitle: String
    let onAddAPIModel: () -> Void
    let onRefresh: () -> Void

    var body: some View {
        VStack(spacing: 12) {
            // Title row
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 3) {
                    Text(title)
                        .font(.system(size: 22, weight: .bold, design: .rounded))
                    Text(subtitle)
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                }

                Spacer()

                if isPublicMode {
                    Button(action: onAddAPIModel) {
                        HStack(spacing: 5) {
                            Image(systemName: "plus")
                            Text(addModelTitle)
                        }
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 11)
                        .frame(height: 28)
                        .background(Color.blue, in: Capsule())
                    }
                    .buttonStyle(.plain)
                    .padding(.trailing, 8)
                }

                Button(action: onRefresh) {
                    Image(systemName: "arrow.clockwise")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(.secondary)
                        .frame(width: 28, height: 28)
                        .background(.quaternary, in: .circle)
                }
                .buttonStyle(.plain)
                .help("Refresh models")
            }
        }
    }
}

// MARK: - Support Pill

enum SupportStatus {
    case checking, installed, warning, notInstalled

    var dotColor: Color {
        switch self {
        case .checking: return .secondary
        case .installed: return .green
        case .warning: return .orange
        case .notInstalled: return .secondary
        }
    }
}

struct SupportPill: View {
    let label: String
    let icon: String
    let status: SupportStatus
    let tint: Color
    let action: () -> Void

    @State private var isHovered = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 6) {
                Image(systemName: icon)
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(tint)

                Text(label)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.primary)

                Circle()
                    .fill(status.dotColor)
                    .frame(width: 6, height: 6)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 7)
            .background(.thinMaterial, in: Capsule())
            .overlay(
                Capsule()
                    .strokeBorder(.white.opacity(isHovered ? 0.12 : 0.06), lineWidth: 0.5)
            )
            .scaleEffect(isHovered ? 1.02 : 1.0)
            .animation(.easeOut(duration: 0.15), value: isHovered)
        }
        .buttonStyle(.plain)
        .onHover { isHovered = $0 }
    }
}

// MARK: - Mode Switcher

struct ModeSwitcher: View {
    @Binding var selectedMode: ModelMode

    var body: some View {
        HStack(spacing: 3) {
            ForEach(ModelMode.allCases, id: \.self) { mode in
                Button {
                    withAnimation(.easeInOut(duration: 0.2)) {
                        selectedMode = mode
                    }
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: mode.icon)
                            .font(.system(size: 11, weight: .medium))
                        Text(mode.rawValue)
                            .font(.system(size: 12, weight: .medium))
                    }
                    .foregroundStyle(selectedMode == mode ? .white : .secondary)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 7)
                    .background(
                        selectedMode == mode
                            ? AnyShapeStyle(Color.accentColor)
                            : AnyShapeStyle(Color.primary.opacity(0.06))
                    )
                    .clipShape(Capsule())
                }
                .buttonStyle(.plain)
            }

            Spacer()
        }
    }
}

// MARK: - Search and Filter Bar

struct SearchAndFilterBar: View {
    @Binding var searchQuery: String
    @Binding var selectedFilter: ModelFilter
    let allCount: Int
    let recommendedCount: Int
    let downloadedCount: Int

    var body: some View {
        VStack(spacing: 10) {
            // Filter tabs
            HStack(spacing: 3) {
                FilterTab(label: "All", count: allCount, isSelected: selectedFilter == .all) {
                    selectedFilter = .all
                }
                FilterTab(label: "Recommended", count: recommendedCount, isSelected: selectedFilter == .recommended) {
                    selectedFilter = .recommended
                }
                FilterTab(label: "Downloads", count: downloadedCount, isSelected: selectedFilter == .downloaded) {
                    selectedFilter = .downloaded
                }

                Spacer()

                // Search field
                HStack(spacing: 6) {
                    Image(systemName: "magnifyingglass")
                        .font(.system(size: 11))
                        .foregroundStyle(.tertiary)
                    TextField("Search models", text: $searchQuery)
                        .textFieldStyle(.plain)
                        .font(.system(size: 12))
                        .frame(width: 120)
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(.thinMaterial, in: Capsule())
            }
        }
    }
}

struct FilterTab: View {
    let label: String
    let count: Int
    let isSelected: Bool
    let action: () -> Void

    @State private var isHovered = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 4) {
                Text(label)
                    .font(.system(size: 11, weight: isSelected ? .semibold : .regular))
                Text("(\(count))")
                    .font(.system(size: 10))
                    .foregroundStyle(.tertiary)
            }
            .foregroundStyle(isSelected ? .primary : .secondary)
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(
                isSelected
                    ? AnyShapeStyle(Color.primary.opacity(0.08))
                    : (isHovered ? AnyShapeStyle(Color.primary.opacity(0.04)) : AnyShapeStyle(.clear))
            )
            .clipShape(Capsule())
            .animation(.easeOut(duration: 0.15), value: isSelected)
        }
        .buttonStyle(.plain)
        .onHover { isHovered = $0 }
    }
}

// MARK: - API Model Card

struct APIModelCardView: View {
    let model: APITranscriptionModel
    let isSelected: Bool
    let onUse: () -> Void
    let onEdit: () -> Void
    let onDelete: () -> Void

    @State private var isHovered = false
    @State private var showingDeleteConfirmation = false
    private var isSupported: Bool {
        APITranscriptionModelStore.isSupportedModel(model.modelId, providerId: model.providerId)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 3) {
                    Text(model.displayName)
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(.primary)
                        .lineLimit(1)
                    Text(model.modelId)
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.tertiary)
                        .lineLimit(1)
                }

                Spacer()

                statusDot(
                    color: !isSupported ? .orange : (isSelected ? .green : .blue),
                    label: !isSupported ? "Needs Fix" : (isSelected ? "Public Active" : "API")
                )
            }

            HStack(spacing: 4) {
                badge(model.providerName, color: .blue)
                badge("Public", color: .orange)
                badge("Keychain", color: .green)
                if !isSupported {
                    badge("Invalid STT ID", color: .orange)
                }
                if let cost = model.audioCostPerMinuteUSD {
                    badge(String(format: "$%.4f/min", cost), color: .mint)
                }
            }

            HStack(spacing: 8) {
                Button {
                    isSupported ? onUse() : onEdit()
                } label: {
                    Text(!isSupported ? "Fix ID" : (isSelected ? "Public Active" : "Use"))
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(isSelected ? .green : .white)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 7)
                        .background(
                            isSelected
                                ? Color.green.opacity(0.12)
                                : (!isSupported ? Color.orange : Color.blue),
                            in: Capsule()
                        )
                }
                .buttonStyle(.plain)

                Button {
                    onEdit()
                } label: {
                    Image(systemName: "pencil")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(.blue)
                        .frame(width: 30, height: 30)
                        .background(Color.blue.opacity(0.10), in: Circle())
                }
                .buttonStyle(.plain)

                Button {
                    showingDeleteConfirmation = true
                } label: {
                    Image(systemName: "trash")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(.red)
                        .frame(width: 30, height: 30)
                        .background(Color.red.opacity(0.10), in: Circle())
                }
                .buttonStyle(.plain)
            }
        }
        .padding(14)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(isSelected ? Color.green.opacity(0.4) : Color.white.opacity(isHovered ? 0.14 : 0.06), lineWidth: 0.8)
        )
        .scaleEffect(isHovered ? 1.01 : 1.0)
        .animation(.easeOut(duration: 0.16), value: isHovered)
        .onHover { isHovered = $0 }
        .confirmationDialog("Remove API model?", isPresented: $showingDeleteConfirmation) {
            Button("Remove", role: .destructive) { onDelete() }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This removes the model card only. It does not remove your saved API key.")
        }
    }

    private func statusDot(color: Color, label: String) -> some View {
        HStack(spacing: 4) {
            Circle()
                .fill(color)
                .frame(width: 6, height: 6)
            Text(label)
                .font(.system(size: 9, weight: .medium))
                .foregroundStyle(color)
        }
        .padding(.horizontal, 7)
        .padding(.vertical, 3)
        .background(color.opacity(0.12), in: Capsule())
    }

    private func badge(_ text: String, color: Color) -> some View {
        Text(text)
            .font(.system(size: 9, weight: .medium))
            .foregroundStyle(color)
            .padding(.horizontal, 6)
            .padding(.vertical, 3)
            .background(color.opacity(0.10), in: Capsule())
    }
}

// MARK: - API Language Model Card

struct APILanguageModelCardView: View {
    let model: APILanguageModel
    let isSelected: Bool
    let onUse: () -> Void
    let onEdit: () -> Void
    let onDelete: () -> Void

    @State private var isHovered = false
    @State private var showingDeleteConfirmation = false

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 3) {
                    Text(model.displayName)
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(.primary)
                        .lineLimit(1)
                    Text(model.modelId)
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.tertiary)
                        .lineLimit(1)
                }

                Spacer()

                statusDot(color: isSelected ? .green : .orange, label: isSelected ? "Public Active" : "LLM")
            }

            HStack(spacing: 4) {
                badge(model.providerName, color: .blue)
                badge("Public", color: .orange)
                badge("Keychain", color: .green)
                if model.providerId == TranscriptionProviderDescriptor.openRouter.id,
                   let providerOnly = model.openRouterProviderOnly,
                   !providerOnly.isEmpty {
                    badge("Only \(providerOnly.joined(separator: ","))", color: .purple)
                }
                if let input = model.inputCostPerMillionTokensUSD,
                   let output = model.outputCostPerMillionTokensUSD {
                    badge(String(format: "$%.2f/$%.2f", input, output), color: .mint)
                }
            }

            HStack(spacing: 8) {
                Button {
                    onUse()
                } label: {
                    Text(isSelected ? "Public Active" : "Use")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(isSelected ? .green : .white)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 7)
                        .background(isSelected ? Color.green.opacity(0.12) : Color.orange, in: Capsule())
                }
                .buttonStyle(.plain)

                Button {
                    onEdit()
                } label: {
                    Image(systemName: "pencil")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(.blue)
                        .frame(width: 30, height: 30)
                        .background(Color.blue.opacity(0.10), in: Circle())
                }
                .buttonStyle(.plain)

                Button {
                    showingDeleteConfirmation = true
                } label: {
                    Image(systemName: "trash")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(.red)
                        .frame(width: 30, height: 30)
                        .background(Color.red.opacity(0.10), in: Circle())
                }
                .buttonStyle(.plain)
            }
        }
        .padding(14)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(isSelected ? Color.green.opacity(0.4) : Color.white.opacity(isHovered ? 0.14 : 0.06), lineWidth: 0.8)
        )
        .scaleEffect(isHovered ? 1.01 : 1.0)
        .animation(.easeOut(duration: 0.16), value: isHovered)
        .onHover { isHovered = $0 }
        .confirmationDialog("Remove API LLM?", isPresented: $showingDeleteConfirmation) {
            Button("Remove", role: .destructive) { onDelete() }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This removes the LLM card only. It does not remove your saved API key.")
        }
    }

    private func statusDot(color: Color, label: String) -> some View {
        HStack(spacing: 4) {
            Circle()
                .fill(color)
                .frame(width: 6, height: 6)
            Text(label)
                .font(.system(size: 9, weight: .medium))
                .foregroundStyle(color)
        }
        .padding(.horizontal, 7)
        .padding(.vertical, 3)
        .background(color.opacity(0.12), in: Capsule())
    }

    private func badge(_ text: String, color: Color) -> some View {
        Text(text)
            .font(.system(size: 9, weight: .medium))
            .foregroundStyle(color)
            .padding(.horizontal, 6)
            .padding(.vertical, 3)
            .background(color.opacity(0.10), in: Capsule())
    }
}

// MARK: - Add API Model Sheet

struct AddAPIModelSheet: View {
    let model: APITranscriptionModel?
    let onSave: (APITranscriptionModel) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var providerId: String
    @State private var displayName = "GPT-4o Transcribe"
    @State private var modelId = "gpt-4o-transcribe"
    @State private var audioCostPerMinute = ""
    @State private var apiKeyInput = ""
    @State private var statusMessage: String?
    @State private var statusIsError = false
    @State private var isTesting = false

    private var canSave: Bool {
        APITranscriptionModelStore.isSupportedModel(modelId, providerId: providerId)
    }

    private var providerDescriptor: TranscriptionProviderDescriptor {
        switch providerId {
        case TranscriptionProviderDescriptor.openRouter.id:
            return .openRouter
        case TranscriptionProviderDescriptor.elevenLabs.id:
            return .elevenLabs
        default:
            return .openAI
        }
    }

    private var modelPlaceholder: String {
        defaultModelId(for: providerId)
    }

    private var keyPlaceholder: String {
        switch providerId {
        case TranscriptionProviderDescriptor.openRouter.id:
            return "sk-or-..."
        case TranscriptionProviderDescriptor.elevenLabs.id:
            return "xi-api-key..."
        default:
            return "sk-..."
        }
    }

    private var providerInfoText: String {
        switch providerId {
        case TranscriptionProviderDescriptor.openRouter.id:
            return "OpenRouter STT model IDs must include the provider prefix, for example openai/whisper-1. Chat/LLM models like google/gemini-* go under LLM, not Speech-to-Text."
        case TranscriptionProviderDescriptor.elevenLabs.id:
            return "ElevenLabs batch Speech-to-Text supports scribe_v2 and scribe_v1 here. The scribe_v2_realtime model uses WebSockets and will be handled as a separate realtime feature."
        default:
            return "OpenAI Speech-to-Text accepts only known transcription model IDs. Pricing is optional and used only for dashboard estimates."
        }
    }

    init(model: APITranscriptionModel? = nil, onSave: @escaping (APITranscriptionModel) -> Void) {
        self.model = model
        self.onSave = onSave
        _providerId = State(initialValue: model?.providerId ?? TranscriptionProviderDescriptor.openAI.id)
        _displayName = State(initialValue: model?.displayName ?? "GPT-4o Transcribe")
        _modelId = State(initialValue: model?.modelId ?? "gpt-4o-transcribe")
        _audioCostPerMinute = State(initialValue: model?.audioCostPerMinuteUSD.map { String(format: "%.6g", $0) } ?? "")
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            VStack(alignment: .leading, spacing: 4) {
                Text(model == nil ? "Add API Model" : "Edit API Model")
                    .font(.system(size: 20, weight: .bold, design: .rounded))
                Text(model == nil ? "Connect a public Speech-to-Text provider." : "Update provider, model name, model ID, pricing, or API key.")
                    .font(.system(size: 12))
                    .foregroundStyle(.secondary)
            }

            VStack(alignment: .leading, spacing: 12) {
                formRow(label: "Provider") {
                    Picker("Provider", selection: $providerId) {
                        Text("OpenAI API").tag(TranscriptionProviderDescriptor.openAI.id)
                        Text("OpenRouter").tag(TranscriptionProviderDescriptor.openRouter.id)
                        Text("ElevenLabs").tag(TranscriptionProviderDescriptor.elevenLabs.id)
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    .onChange(of: providerId) { _, newProviderId in
                        statusMessage = nil
                        statusIsError = false
                        if !APITranscriptionModelStore.isSupportedModel(modelId, providerId: newProviderId) {
                            displayName = defaultDisplayName(for: newProviderId)
                            modelId = defaultModelId(for: newProviderId)
                        }
                    }
                }

                formRow(label: "Display name") {
                    TextField(defaultDisplayName(for: providerId), text: $displayName)
                        .textFieldStyle(.roundedBorder)
                }

                formRow(label: "Model ID") {
                    TextField(modelPlaceholder, text: $modelId)
                        .textFieldStyle(.roundedBorder)
                        .font(.system(size: 12, design: .monospaced))
                }

                if !modelId.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                   !APITranscriptionModelStore.isSupportedModel(modelId, providerId: providerId) {
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundStyle(.orange)
                        Text("This is not a supported Speech-to-Text model ID. \(APITranscriptionModelStore.supportedModelHint(providerId: providerId))")
                            .font(.system(size: 11, weight: .medium))
                            .foregroundStyle(.orange)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .padding(10)
                    .background(Color.orange.opacity(0.12), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                }

                formRow(label: "Audio cost / minute") {
                    TextField(audioCostPlaceholder, text: $audioCostPerMinute)
                        .textFieldStyle(.roundedBorder)
                        .font(.system(size: 12, design: .monospaced))
                }

                formRow(label: "API key") {
                    VStack(alignment: .leading, spacing: 6) {
                        SecureField(keyPlaceholder, text: $apiKeyInput)
                            .textFieldStyle(.roundedBorder)
                        savedKeyIndicator(providerId: providerId, isOverriding: !apiKeyInput.isEmpty)
                    }
                }
            }

            HStack(alignment: .top, spacing: 8) {
                Image(systemName: "info.circle")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(.secondary)
                    .padding(.top, 2)
                Text(providerInfoText)
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(12)
            .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 12, style: .continuous))

            if let statusMessage {
                Text(statusMessage)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(statusIsError ? .red : .secondary)
            }

            HStack {
                Button("Cancel") {
                    dismiss()
                }
                .buttonStyle(.bordered)

                Spacer()

                Button {
                    saveModel()
                } label: {
                    Text("Save")
                }
                .buttonStyle(.bordered)
                .disabled(!canSave || isTesting)

                Button {
                    Task { await testAndSaveModel() }
                } label: {
                    if isTesting {
                        ProgressView()
                            .scaleEffect(0.7)
                            .frame(width: 72)
                    } else {
                        Text("Test & Save")
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(!canSave || isTesting)
            }
        }
        .padding(24)
        .frame(width: 480)
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

    private func saveModel() {
        do {
            try validateModelId()
            try saveAPIKeyIfNeeded()
            onSave(makeModel())
            dismiss()
        } catch {
            statusMessage = error.localizedDescription
            statusIsError = true
        }
    }

    private func testAndSaveModel() async {
        isTesting = true
        defer { isTesting = false }

        do {
            try validateModelId()
            let apiKey = try resolvedAPIKey()
            try await testSelectedProvider(apiKey: apiKey)
            try saveAPIKeyIfNeeded()
            onSave(makeModel())
            dismiss()
        } catch {
            statusMessage = error.localizedDescription
            statusIsError = true
        }
    }

    private func testSelectedProvider(apiKey: String) async throws {
        let trimmedModelId = modelId.trimmingCharacters(in: .whitespacesAndNewlines)
        if providerId == TranscriptionProviderDescriptor.openRouter.id {
            _ = try await OpenRouterTranscriptionClient().transcribe(
                audioData: Self.generatedSilentWAV(),
                modelId: trimmedModelId,
                apiKey: apiKey
            )
            return
        }
        if providerId == TranscriptionProviderDescriptor.elevenLabs.id {
            _ = try await ElevenLabsTranscriptionClient().transcribe(
                audioData: Self.generatedSilentWAV(),
                modelId: trimmedModelId,
                apiKey: apiKey
            )
            return
        }

        _ = try await OpenAITranscriptionClient().transcribe(
            audioData: Self.generatedSilentWAV(),
            modelId: trimmedModelId,
            apiKey: apiKey
        )
    }

    private func validateModelId() throws {
        guard APITranscriptionModelStore.isSupportedModel(modelId, providerId: providerId) else {
            let hint = APITranscriptionModelStore.supportedModelHint(providerId: providerId)
            if providerId == TranscriptionProviderDescriptor.openRouter.id {
                throw OpenRouterTranscriptionError.unsupportedModel(hint)
            }
            if providerId == TranscriptionProviderDescriptor.elevenLabs.id {
                throw ElevenLabsTranscriptionError.unsupportedModel(hint)
            }
            throw OpenAITranscriptionError.unsupportedModel(hint)
        }
    }

    private func makeModel() -> APITranscriptionModel {
        let trimmedModelId = modelId.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedName = displayName.trimmingCharacters(in: .whitespacesAndNewlines)
        return APITranscriptionModel(
            id: model?.id ?? UUID(),
            providerId: providerDescriptor.id,
            providerName: providerDescriptor.displayName,
            modelId: trimmedModelId,
            displayName: trimmedName.isEmpty ? trimmedModelId : trimmedName,
            audioCostPerMinuteUSD: resolvedAudioCostPerMinute(for: trimmedModelId, providerId: providerDescriptor.id),
            createdAt: model?.createdAt ?? Date()
        )
    }

    private var audioCostPlaceholder: String {
        if let defaultCost = APITranscriptionModel.defaultAudioCostPerMinuteUSD(modelId: modelId, providerId: providerId) {
            return String(format: "%.3f", defaultCost)
        }
        return providerId == TranscriptionProviderDescriptor.elevenLabs.id ? "Manual estimate" : "0.006"
    }

    private func resolvedAudioCostPerMinute(for modelId: String, providerId: String) -> Double? {
        let manualCost = Double(audioCostPerMinute.trimmingCharacters(in: .whitespacesAndNewlines))
        return manualCost ?? APITranscriptionModel.defaultAudioCostPerMinuteUSD(modelId: modelId, providerId: providerId)
    }

    private func savedKeyIndicator(providerId: String, isOverriding: Bool) -> some View {
        Group {
            if APIKeychainStore.hasAPIKey(providerId: providerId) {
                HStack(spacing: 6) {
                    Image(systemName: isOverriding ? "key.fill" : "lock.fill")
                        .font(.system(size: 10, weight: .semibold))
                    Text(isOverriding ? "New key will replace saved key" : "Saved key: ••••••••••••")
                        .font(.system(size: 10, weight: .medium))
                }
                .foregroundStyle(isOverriding ? .orange : .secondary)
            }
        }
    }

    private func saveAPIKeyIfNeeded() throws {
        let trimmedKey = apiKeyInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedKey.isEmpty else { return }
        try APIKeychainStore.saveAPIKey(trimmedKey, providerId: providerId)
    }

    private func resolvedAPIKey() throws -> String {
        let trimmedKey = apiKeyInput.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmedKey.isEmpty {
            return trimmedKey
        }
        if let savedKey = try APIKeychainStore.apiKey(providerId: providerId),
           !savedKey.isEmpty {
            return savedKey
        }
        if providerId == TranscriptionProviderDescriptor.openRouter.id {
            throw OpenRouterTranscriptionError.missingAPIKey
        }
        if providerId == TranscriptionProviderDescriptor.elevenLabs.id {
            throw ElevenLabsTranscriptionError.missingAPIKey
        }
        throw OpenAITranscriptionError.missingAPIKey
    }

    private func defaultDisplayName(for providerId: String) -> String {
        switch providerId {
        case TranscriptionProviderDescriptor.openRouter.id:
            return "OpenRouter Whisper"
        case TranscriptionProviderDescriptor.elevenLabs.id:
            return "ElevenLabs Scribe v2"
        default:
            return "GPT-4o Transcribe"
        }
    }

    private func defaultModelId(for providerId: String) -> String {
        switch providerId {
        case TranscriptionProviderDescriptor.openRouter.id:
            return "openai/whisper-1"
        case TranscriptionProviderDescriptor.elevenLabs.id:
            return "scribe_v2"
        default:
            return "gpt-4o-transcribe"
        }
    }

    private static func generatedSilentWAV() -> Data {
        let sampleRate = 16_000
        let durationSeconds = 1
        let channelCount = 1
        let bitsPerSample = 16
        let byteRate = sampleRate * channelCount * bitsPerSample / 8
        let blockAlign = channelCount * bitsPerSample / 8
        let dataSize = sampleRate * durationSeconds * blockAlign

        var data = Data()
        data.appendASCII("RIFF")
        data.appendUInt32LE(UInt32(36 + dataSize))
        data.appendASCII("WAVE")
        data.appendASCII("fmt ")
        data.appendUInt32LE(16)
        data.appendUInt16LE(1)
        data.appendUInt16LE(UInt16(channelCount))
        data.appendUInt32LE(UInt32(sampleRate))
        data.appendUInt32LE(UInt32(byteRate))
        data.appendUInt16LE(UInt16(blockAlign))
        data.appendUInt16LE(UInt16(bitsPerSample))
        data.appendASCII("data")
        data.appendUInt32LE(UInt32(dataSize))
        data.append(Data(repeating: 0, count: dataSize))
        return data
    }
}

// MARK: - Add API Language Model Sheet

struct AddAPILanguageModelSheet: View {
    let model: APILanguageModel?
    let onSave: (APILanguageModel) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var providerId: String
    @State private var displayName = "GPT-4o Mini"
    @State private var modelId = "gpt-4o-mini"
    @State private var openRouterProviderOnly = ""
    @State private var inputCostPerMillionTokens = ""
    @State private var outputCostPerMillionTokens = ""
    @State private var apiKeyInput = ""
    @State private var statusMessage: String?
    @State private var statusIsError = false
    @State private var isTesting = false

    private var canSave: Bool {
        !modelId.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private var providerDescriptor: TranscriptionProviderDescriptor {
        providerId == TranscriptionProviderDescriptor.openRouter.id ? .openRouter : .openAI
    }

    private var keyPlaceholder: String {
        return providerId == TranscriptionProviderDescriptor.openRouter.id ? "sk-or-..." : "sk-..."
    }

    init(model: APILanguageModel? = nil, onSave: @escaping (APILanguageModel) -> Void) {
        self.model = model
        self.onSave = onSave
        _providerId = State(initialValue: model?.providerId ?? TranscriptionProviderDescriptor.openAI.id)
        _displayName = State(initialValue: model?.displayName ?? "GPT-4o Mini")
        _modelId = State(initialValue: model?.modelId ?? "gpt-4o-mini")
        _openRouterProviderOnly = State(initialValue: model?.openRouterProviderOnly?.joined(separator: ", ") ?? "")
        _inputCostPerMillionTokens = State(initialValue: model?.inputCostPerMillionTokensUSD.map { String(format: "%.6g", $0) } ?? "")
        _outputCostPerMillionTokens = State(initialValue: model?.outputCostPerMillionTokensUSD.map { String(format: "%.6g", $0) } ?? "")
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            VStack(alignment: .leading, spacing: 4) {
                Text(model == nil ? "Add API LLM" : "Edit API LLM")
                    .font(.system(size: 20, weight: .bold, design: .rounded))
                Text(model == nil ? "Use public language models for summaries first, then polishing workflows later." : "Update LLM name, model ID, pricing, or API key.")
                    .font(.system(size: 12))
                    .foregroundStyle(.secondary)
            }

            VStack(alignment: .leading, spacing: 12) {
                formRow(label: "Provider") {
                    Picker("Provider", selection: $providerId) {
                        Text("OpenAI API").tag(TranscriptionProviderDescriptor.openAI.id)
                        Text("OpenRouter").tag(TranscriptionProviderDescriptor.openRouter.id)
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    .onChange(of: providerId) { _, newProviderId in
                        statusMessage = nil
                        statusIsError = false
                        if model == nil {
                            displayName = defaultDisplayName(for: newProviderId)
                            modelId = defaultModelId(for: newProviderId)
                        }
                    }
                }

                formRow(label: "Display name") {
                    TextField(defaultDisplayName(for: providerId), text: $displayName)
                        .textFieldStyle(.roundedBorder)
                }

                formRow(label: "Model ID") {
                    TextField(defaultModelId(for: providerId), text: $modelId)
                        .textFieldStyle(.roundedBorder)
                        .font(.system(size: 12, design: .monospaced))
                }

                if providerId == TranscriptionProviderDescriptor.openRouter.id {
                    formRow(label: "Only provider") {
                        TextField("cerebras", text: $openRouterProviderOnly)
                            .textFieldStyle(.roundedBorder)
                            .font(.system(size: 12, design: .monospaced))
                    }
                }

                HStack(spacing: 10) {
                    formRow(label: "Input $ / 1M tokens") {
                        TextField("1.25", text: $inputCostPerMillionTokens)
                            .textFieldStyle(.roundedBorder)
                            .font(.system(size: 12, design: .monospaced))
                    }

                    formRow(label: "Output $ / 1M tokens") {
                        TextField("5.00", text: $outputCostPerMillionTokens)
                            .textFieldStyle(.roundedBorder)
                            .font(.system(size: 12, design: .monospaced))
                    }
                }

                formRow(label: "API key") {
                    VStack(alignment: .leading, spacing: 6) {
                        SecureField(keyPlaceholder, text: $apiKeyInput)
                            .textFieldStyle(.roundedBorder)
                        savedKeyIndicator(providerId: providerId, isOverriding: !apiKeyInput.isEmpty)
                    }
                }
            }

            HStack(alignment: .top, spacing: 8) {
                Image(systemName: "info.circle")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(.secondary)
                    .padding(.top, 2)
                Text(providerId == TranscriptionProviderDescriptor.openRouter.id
                    ? "Use OpenRouter LLM slugs such as openai/gpt-oss-120b. Leave Only provider empty to let OpenRouter choose, or enter cerebras to pin fastest routing."
                    : "Pricing is optional. Public summaries and polishing use actual input/output token counts when available.")
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(12)
            .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 12, style: .continuous))

            if let statusMessage {
                Text(statusMessage)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(statusIsError ? .red : .secondary)
            }

            HStack {
                Button("Cancel") {
                    dismiss()
                }
                .buttonStyle(.bordered)

                Spacer()

                Button {
                    saveModel()
                } label: {
                    Text("Save")
                }
                .buttonStyle(.bordered)
                .disabled(!canSave || isTesting)

                Button {
                    Task { await testAndSaveModel() }
                } label: {
                    if isTesting {
                        ProgressView()
                            .scaleEffect(0.7)
                            .frame(width: 72)
                    } else {
                        Text("Test & Save")
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(!canSave || isTesting)
            }
        }
        .padding(24)
        .frame(width: 480)
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

    private func saveModel() {
        do {
            try saveAPIKeyIfNeeded()
            onSave(makeModel())
            dismiss()
        } catch {
            statusMessage = error.localizedDescription
            statusIsError = true
        }
    }

    private func testAndSaveModel() async {
        isTesting = true
        defer { isTesting = false }

        do {
            let apiKey = try resolvedAPIKey()
            if providerId == TranscriptionProviderDescriptor.openRouter.id {
                _ = try await OpenRouterLanguageModelClient().testModel(
                    modelId: modelId.trimmingCharacters(in: .whitespacesAndNewlines),
                    apiKey: apiKey,
                    providerOnly: parsedOpenRouterProviderOnly()
                )
            } else {
                _ = try await OpenAILanguageModelClient().testModel(
                    modelId: modelId.trimmingCharacters(in: .whitespacesAndNewlines),
                    apiKey: apiKey
                )
            }
            try saveAPIKeyIfNeeded()
            onSave(makeModel())
            dismiss()
        } catch {
            statusMessage = error.localizedDescription
            statusIsError = true
        }
    }

    private func makeModel() -> APILanguageModel {
        let trimmedModelId = modelId.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedName = displayName.trimmingCharacters(in: .whitespacesAndNewlines)
        return APILanguageModel(
            id: model?.id ?? UUID(),
            providerId: providerDescriptor.id,
            providerName: providerDescriptor.displayName,
            modelId: trimmedModelId,
            displayName: trimmedName.isEmpty ? trimmedModelId : trimmedName,
            inputCostPerMillionTokensUSD: Double(inputCostPerMillionTokens.trimmingCharacters(in: .whitespacesAndNewlines)),
            outputCostPerMillionTokensUSD: Double(outputCostPerMillionTokens.trimmingCharacters(in: .whitespacesAndNewlines)),
            openRouterProviderOnly: providerId == TranscriptionProviderDescriptor.openRouter.id ? parsedOpenRouterProviderOnly() : nil,
            createdAt: model?.createdAt ?? Date()
        )
    }

    private func parsedOpenRouterProviderOnly() -> [String]? {
        let providers = openRouterProviderOnly
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        return providers.isEmpty ? nil : providers
    }

    private func saveAPIKeyIfNeeded() throws {
        let trimmedKey = apiKeyInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedKey.isEmpty else { return }
        try APIKeychainStore.saveAPIKey(trimmedKey, providerId: providerId)
    }

    private func resolvedAPIKey() throws -> String {
        let trimmedKey = apiKeyInput.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmedKey.isEmpty {
            return trimmedKey
        }
        if let savedKey = try APIKeychainStore.apiKey(providerId: providerId),
           !savedKey.isEmpty {
            return savedKey
        }
        if providerId == TranscriptionProviderDescriptor.openRouter.id {
            throw OpenRouterLanguageModelError.missingAPIKey
        }
        throw OpenAILanguageModelError.missingAPIKey
    }

    private func defaultDisplayName(for providerId: String) -> String {
        providerId == TranscriptionProviderDescriptor.openRouter.id ? "OpenRouter LLM" : "GPT-4o Mini"
    }

    private func defaultModelId(for providerId: String) -> String {
        providerId == TranscriptionProviderDescriptor.openRouter.id ? "openai/gpt-oss-120b" : "gpt-4o-mini"
    }

    private func savedKeyIndicator(providerId: String, isOverriding: Bool) -> some View {
        Group {
            if APIKeychainStore.hasAPIKey(providerId: providerId) {
                HStack(spacing: 6) {
                    Image(systemName: isOverriding ? "key.fill" : "lock.fill")
                        .font(.system(size: 10, weight: .semibold))
                    Text(isOverriding ? "New key will replace saved key" : "Saved key: ••••••••••••")
                        .font(.system(size: 10, weight: .medium))
                }
                .foregroundStyle(isOverriding ? .orange : .secondary)
            }
        }
    }
}

private extension Data {
    mutating func appendASCII(_ string: String) {
        append(Data(string.utf8))
    }

    mutating func appendUInt16LE(_ value: UInt16) {
        var littleEndianValue = value.littleEndian
        Swift.withUnsafeBytes(of: &littleEndianValue) { append(contentsOf: $0) }
    }

    mutating func appendUInt32LE(_ value: UInt32) {
        var littleEndianValue = value.littleEndian
        Swift.withUnsafeBytes(of: &littleEndianValue) { append(contentsOf: $0) }
    }
}

// MARK: - Model Section View (Grid)

struct ModelSectionView<Content: View>: View {
    let title: String
    let icon: String
    let tint: Color
    let columns: Int
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            // Section label
            HStack(spacing: 6) {
                Image(systemName: icon)
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(tint)
                Text(title.uppercased())
                    .font(.system(size: 11, weight: .semibold, design: .rounded))
                    .foregroundStyle(.secondary)
                    .tracking(0.8)
            }

            // Grid layout
            let gridColumns = Array(repeating: GridItem(.flexible(), spacing: 10), count: columns)
            LazyVGrid(columns: gridColumns, spacing: 10) {
                content
            }
        }
    }
}

// MARK: - Compact Model Card

struct CompactModelCardView: View {
    let model: ModelInfoResponse
    let isLoaded: Bool
    let isLoading: Bool
    let isDownloading: Bool
    let nemoAvailable: Bool
    let isCheckingNemoStatus: Bool
    @Binding var realTimeStreaming: Bool
    let onLoad: () async -> Void
    let onUnload: () async -> Void
    let onDownload: () async -> Void
    let onDelete: () async -> Void
    let onStreamingChanged: (Bool) -> Void
    let onInstallNemo: () -> Void

    @State private var isHovered = false
    @State private var showingDeleteConfirmation = false
    @State private var cardAppeared = false

    private var canUseModel: Bool {
        if model.nemoRequired {
            return isCheckingNemoStatus || nemoAvailable
        }
        return true
    }

    private var showNemoRequired: Bool {
        model.nemoRequired && !nemoAvailable && !isCheckingNemoStatus
    }

    private func formatSize(_ mb: Int) -> String {
        if mb >= 1024 { return String(format: "%.1f GB", Double(mb) / 1024.0) }
        return "\(mb) MB"
    }

    private func getPerformanceMetrics() -> (speed: Double, accuracy: Double) {
        switch model.id {
        case "parakeet-tdt-v2", "parakeet-v2": return (0.95, 0.85)
        case "parakeet-tdt-v3", "parakeet-v3": return (0.85, 0.85)
        case "nemotron-streaming": return (1.00, 0.70)
        case "whisper-base-en", "base.en": return (1.00, 0.60)
        case "whisper-small-en", "small.en": return (0.92, 0.65)
        case "whisper-large-v3-turbo", "large-v3-turbo": return (0.85, 0.90)
        case "whisper-large-v3", "large-v3": return (0.80, 0.98)
        default: return (0.70, 0.75)
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            // Top: name + status
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 3) {
                    Text(model.displayName)
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(.primary)
                        .lineLimit(1)

                    Text("by \(model.author)")
                        .font(.system(size: 10))
                        .foregroundStyle(.tertiary)
                }

                Spacer()

                // Status indicator
                if isLoaded {
                    statusDot(color: .green, label: "Local Active")
                } else if model.downloaded {
                    statusDot(color: .blue, label: "Ready")
                } else if showNemoRequired {
                    statusDot(color: .orange, label: "NeMo")
                }
            }

            // Badges row
            HStack(spacing: 4) {
                if model.streamingNative {
                    badge("Real-Time", color: .orange)
                }
                if model.multilingual {
                    badge("Multilingual", color: .purple)
                }
                if model.sessionCompatible {
                    badge("Sessions", color: .teal)
                }
            }

            // Specs
            HStack(spacing: 12) {
                specLabel(icon: "arrow.down.circle", text: formatSize(model.sizeMB))
                specLabel(icon: "memorychip", text: formatSize(model.ramMB))
                if model.multilingual && model.languageNames.count > 1 {
                    specLabel(icon: "globe", text: model.languageNames.first == "99+ languages" ? "99+ langs" : "\(model.languageNames.count) langs")
                } else {
                    specLabel(icon: "globe", text: model.languageNames.first ?? "English")
                }
            }

            // Performance bars
            let metrics = getPerformanceMetrics()
            VStack(spacing: 5) {
                miniBar(label: "Speed", value: metrics.speed, color: .blue)
                miniBar(label: "Accuracy", value: metrics.accuracy, color: .purple)
            }

            // Divider
            Rectangle()
                .fill(.white.opacity(0.06))
                .frame(height: 0.5)

            // Action row
            HStack(spacing: 8) {
                if showNemoRequired {
                    actionButton(label: "Install NeMo", icon: "cpu", style: .prominent, tint: .orange) {
                        onInstallNemo()
                    }
                } else if !model.downloaded {
                    actionButton(label: "Download", icon: "arrow.down.circle", style: .prominent) {
                        Task { await onDownload() }
                    }
                    .disabled(isDownloading || !canUseModel)
                } else if isLoaded {
                    actionButton(label: "Unload", icon: "eject", style: .regular) {
                        Task { await onUnload() }
                    }

                    if model.streamingNative {
                        Toggle("Live", isOn: $realTimeStreaming)
                            .toggleStyle(.switch)
                            .controlSize(.mini)
                            .font(.system(size: 10))
                            .onChange(of: realTimeStreaming) { _, newValue in
                                onStreamingChanged(newValue)
                            }
                    }
                } else {
                    actionButton(label: "Load", icon: "play.fill", style: .prominent) {
                        Task { await onLoad() }
                    }
                    .disabled(isLoading || !model.downloaded || !canUseModel)
                }

                Spacer()

                // Loading/downloading indicators
                if isLoading {
                    ProgressView()
                        .scaleEffect(0.6)
                }

                if isDownloading {
                    AnimatedDownloadingBar()
                        .frame(width: 40, height: 4)
                }

                // Delete button
                if model.downloaded && !isLoaded {
                    Button {
                        showingDeleteConfirmation = true
                    } label: {
                        Image(systemName: "trash")
                            .font(.system(size: 10))
                            .foregroundStyle(.red.opacity(0.7))
                    }
                    .buttonStyle(.plain)
                }
            }

            // Streaming info
            if isLoaded && realTimeStreaming {
                HStack(spacing: 4) {
                    Image(systemName: model.streamingNative ? "info.circle" : "exclamationmark.triangle.fill")
                        .font(.system(size: 9))
                        .foregroundStyle(model.streamingNative ? .blue : .orange)
                    Text(model.streamingNative
                        ? "Cache-aware streaming active"
                        : "Simulated streaming — use Nemotron for better performance")
                        .font(.system(size: 9))
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }
            }
        }
        .padding(14)
        .background(.thinMaterial, in: .rect(cornerRadius: 14, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(
                    isLoaded ? Color.green.opacity(0.4) : .white.opacity(0.06),
                    lineWidth: isLoaded ? 1.5 : 0.5
                )
        )
        .scaleEffect(isHovered ? 1.01 : 1.0)
        .scaleEffect(cardAppeared ? 1.0 : 0.95)
        .offset(y: cardAppeared ? 0 : 8)
        .animation(.easeOut(duration: 0.15), value: isHovered)
        .animation(.spring(duration: 0.4, bounce: 0.15), value: cardAppeared)
        .onHover { isHovered = $0 }
        .onAppear {
            // Pop-in animation
            if !cardAppeared {
                cardAppeared = true
            }
        }
        .confirmationDialog("Delete Model", isPresented: $showingDeleteConfirmation) {
            Button("Delete \(model.displayName)", role: .destructive) {
                Task { await onDelete() }
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This will permanently delete the downloaded model files.")
        }
    }

    // MARK: - Sub-components

    private func statusDot(color: Color, label: String) -> some View {
        HStack(spacing: 4) {
            Circle()
                .fill(color)
                .frame(width: 6, height: 6)
            Text(label)
                .font(.system(size: 9, weight: .medium))
                .foregroundStyle(color)
        }
    }

    private func badge(_ text: String, color: Color) -> some View {
        Text(text)
            .font(.system(size: 9, weight: .medium))
            .foregroundStyle(color)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(color.opacity(0.12), in: Capsule())
    }

    private func specLabel(icon: String, text: String) -> some View {
        HStack(spacing: 3) {
            Image(systemName: icon)
                .font(.system(size: 9))
            Text(text)
                .font(.system(size: 10))
        }
        .foregroundStyle(.secondary)
    }

    private func miniBar(label: String, value: Double, color: Color) -> some View {
        HStack(spacing: 6) {
            Text(label)
                .font(.system(size: 9))
                .foregroundStyle(.tertiary)
                .frame(width: 46, alignment: .leading)

            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 2)
                        .fill(Color.secondary.opacity(0.12))
                        .frame(height: 3)
                    RoundedRectangle(cornerRadius: 2)
                        .fill(color.opacity(0.7))
                        .frame(width: geometry.size.width * value, height: 3)
                }
            }
            .frame(height: 3)
        }
    }

    private enum ActionStyle { case prominent, regular }

    private func actionButton(label: String, icon: String, style: ActionStyle, tint: Color = .accentColor, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.system(size: 9, weight: .medium))
                Text(label)
                    .font(.system(size: 10, weight: .medium))
            }
            .foregroundStyle(style == .prominent ? .white : .primary)
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .background(
                style == .prominent
                    ? AnyShapeStyle(tint)
                    : AnyShapeStyle(.quaternary)
            )
            .clipShape(Capsule())
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Summary Runtime Card

struct SummaryRuntimeCard: View {
    let installed: Bool
    let ready: Bool
    let isInstalling: Bool
    let progress: Double
    let statusMessage: String
    let onInstall: () -> Void
    let onRemove: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                Image(systemName: "sparkles")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(.orange)

                Text("Summary Runtime")
                    .font(.system(size: 13, weight: .medium))

                Spacer()

                if installed {
                    HStack(spacing: 4) {
                        Circle()
                            .fill(ready ? .green : .orange)
                            .frame(width: 6, height: 6)
                        Text(ready ? "Ready" : "Not Ready")
                            .font(.system(size: 10, weight: .medium))
                            .foregroundStyle(ready ? .green : .orange)
                    }
                }
            }

            if isInstalling {
                VStack(spacing: 6) {
                    ProgressView(value: progress)
                        .tint(.orange)
                    Text(statusMessage)
                        .font(.system(size: 10))
                        .foregroundStyle(.secondary)
                }
            } else if installed {
                HStack {
                    Text("Runs Gemma summaries in a separate worker process.")
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button("Remove") {
                        onRemove()
                    }
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(.red.opacity(0.8))
                    .buttonStyle(.plain)
                }
            } else {
                HStack {
                    Text("Required for Gemma session summaries (~3 GB)")
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button(action: onInstall) {
                        HStack(spacing: 4) {
                            Image(systemName: "arrow.down.circle")
                                .font(.system(size: 10))
                            Text("Install")
                                .font(.system(size: 10, weight: .medium))
                        }
                        .foregroundStyle(.white)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 5)
                        .background(.orange, in: Capsule())
                    }
                    .buttonStyle(.plain)
                }
            }
        }
        .padding(14)
        .background(.thinMaterial, in: .rect(cornerRadius: 12, style: .continuous))
    }
}

// MARK: - Summary Model Card

struct SummaryModelCard: View {
    let model: SummaryModelInfoResponse
    let isDownloading: Bool
    let downloadProgress: Double
    let runtimeInstalled: Bool
    let onDownload: () -> Void
    let onDelete: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 6) {
                        Text(model.displayName)
                            .font(.system(size: 13, weight: .semibold))
                        if model.recommended {
                            Text("Recommended")
                                .font(.system(size: 9, weight: .semibold))
                                .foregroundStyle(.orange)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(Color.orange.opacity(0.12), in: Capsule())
                        }
                    }
                    Text(model.provider)
                        .font(.system(size: 10))
                        .foregroundStyle(.tertiary)
                }
                Spacer()

                if model.downloaded {
                    Button(action: onDelete) {
                        Image(systemName: "trash")
                            .font(.system(size: 10))
                            .foregroundStyle(.red.opacity(0.7))
                    }
                    .buttonStyle(.plain)
                } else {
                    Button(action: onDownload) {
                        HStack(spacing: 4) {
                            Image(systemName: "arrow.down.circle")
                                .font(.system(size: 9))
                            Text(isDownloading ? "Downloading..." : "Download")
                                .font(.system(size: 10, weight: .medium))
                        }
                        .foregroundStyle(.white)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 5)
                        .background(Color.accentColor, in: Capsule())
                    }
                    .buttonStyle(.plain)
                    .disabled(!runtimeInstalled || isDownloading)
                }
            }

            // Specs row
            HStack(spacing: 12) {
                specLabel(icon: "text.word.spacing", text: "\(model.contextTokens.formatted()) ctx")
                specLabel(icon: "memorychip", text: "\(String(format: "%.1f", Double(model.residentModelRamMB) / 1024.0)) GB RAM")
                specLabel(icon: "arrow.down.circle", text: "\(String(format: "%.1f", Double(model.downloadSizeMB) / 1024.0)) GB")
            }

            if isDownloading {
                ProgressView(value: downloadProgress)
                    .tint(.accentColor)
            }
        }
        .padding(14)
        .background(.thinMaterial, in: .rect(cornerRadius: 12, style: .continuous))
    }

    private func specLabel(icon: String, text: String) -> some View {
        HStack(spacing: 3) {
            Image(systemName: icon)
                .font(.system(size: 9))
            Text(text)
                .font(.system(size: 10))
        }
        .foregroundStyle(.secondary)
    }
}

// MARK: - Language Grid View

struct LanguageGridView: View {
    let languages: [String]

    let columns = [
        GridItem(.flexible()),
        GridItem(.flexible()),
        GridItem(.flexible()),
        GridItem(.flexible()),
        GridItem(.flexible())
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Supported Languages")
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(.secondary)

            LazyVGrid(columns: columns, spacing: 6) {
                ForEach(languages, id: \.self) { language in
                    Text(language)
                        .font(.system(size: 10))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(.quaternary, in: .rect(cornerRadius: 4))
                }
            }
        }
        .padding(10)
        .background(.thinMaterial, in: .rect(cornerRadius: 8, style: .continuous))
    }
}
