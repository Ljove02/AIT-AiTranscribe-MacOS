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

    private func isRecommended(_ model: ModelInfoResponse) -> Bool {
        ["parakeet-tdt-v2", "parakeet-v2", "whisper-large-v3-turbo", "large-v3-turbo", "nemotron-streaming"].contains(model.id)
    }

    // MARK: - Body

    var body: some View {
        ScrollView {
            VStack(spacing: 0) {
                // Top: storage bar + support pills
                ModelsHeaderBar(
                    downloadedCount: downloadedSTTModels.count,
                    totalCount: appState.availableModels.count,
                    totalSizeMB: totalSTTSizeMB,
                    nemoAvailable: effectiveNemoAvailable,
                    isCheckingNemo: isCheckingNemoStatus,
                    summaryInstalled: appState.summaryRuntimeInstalled,
                    summaryReady: appState.summaryRuntimeReady,
                    onNemoTap: { showingNemoSetup = true },
                    onSummaryTap: { showingSummarySetup = true },
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
        .task(id: "stagger") {
            guard !hasAnimated else { return }
            try? await Task.sleep(for: .milliseconds(80))
            appeared = true
            onAnimated()
        }
    }

    // MARK: - Speech-to-Text Content

    @ViewBuilder
    private var speechToTextContent: some View {
        // Search + filter row
        SearchAndFilterBar(
            searchQuery: $searchQuery,
            selectedFilter: $selectedFilter,
            allCount: appState.availableModels.count,
            recommendedCount: appState.availableModels.filter { isRecommended($0) }.count,
            downloadedCount: downloadedSTTModels.count
        )
        .padding(.top, 12)
        .staggerIn(index: 3, appeared: appeared)

        if appState.availableModels.isEmpty {
            modelLoadingState
                .staggerIn(index: 4, appeared: appeared)
        } else if filteredModels.isEmpty {
            emptySearchState
                .staggerIn(index: 4, appeared: appeared)
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
            .staggerIn(index: 4, appeared: appeared)
        }

        // Storage footer
        storageFooter
            .padding(.top, 12)
            .staggerIn(index: 5, appeared: appeared)
    }

    // MARK: - Summarization Content

    @ViewBuilder
    private var summarizationContent: some View {
        if !summaryModels.isEmpty {
            VStack(spacing: 12) {
                HStack {
                    SettingsSectionHeader(title: "Summarization Models")
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
                Text("Click the Summarization pill above to set up")
                    .font(.system(size: 11))
                    .foregroundStyle(.tertiary)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 40)
            .staggerIn(index: 3, appeared: appeared)
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
            isLoaded: appState.loadedModelId == model.id,
            isLoading: appState.loadingModelId == model.id,
            isDownloading: appState.downloadingModelId == model.id,
            nemoAvailable: effectiveNemoAvailable,
            isCheckingNemoStatus: isCheckingNemoStatus,
            realTimeStreaming: $realTimeStreaming,
            onLoad: {
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
    case summarization = "Summarization"

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
    let downloadedCount: Int
    let totalCount: Int
    let totalSizeMB: Int
    let nemoAvailable: Bool
    let isCheckingNemo: Bool
    let summaryInstalled: Bool
    let summaryReady: Bool
    let onNemoTap: () -> Void
    let onSummaryTap: () -> Void
    let onRefresh: () -> Void

    private var sizeString: String {
        if totalSizeMB >= 1024 {
            return String(format: "%.1f GB", Double(totalSizeMB) / 1024.0)
        } else if totalSizeMB > 0 {
            return "\(totalSizeMB) MB"
        }
        return "Zero KB"
    }

    var body: some View {
        VStack(spacing: 12) {
            // Title row
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Models")
                        .font(.system(size: 22, weight: .bold, design: .rounded))
                    Text("\(downloadedCount) downloaded · \(sizeString)")
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                }

                Spacer()

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

            // Support pills row
            HStack(spacing: 8) {
                // NeMo pill
                SupportPill(
                    label: "NeMo",
                    icon: "cpu",
                    status: isCheckingNemo ? .checking : (nemoAvailable ? .installed : .notInstalled),
                    tint: .purple,
                    action: onNemoTap
                )

                // Summarization pill
                SupportPill(
                    label: "Summarization",
                    icon: "sparkles",
                    status: summaryInstalled ? (summaryReady ? .installed : .warning) : .notInstalled,
                    tint: .orange,
                    action: onSummaryTap
                )

                Spacer()
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
                    statusDot(color: .green, label: "Active")
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
