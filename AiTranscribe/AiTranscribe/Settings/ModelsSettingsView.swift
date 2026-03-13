import SwiftUI

// MARK: - Models Settings View

struct ModelsSettingsView: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var backendManager: BackendManager

    /// Real-time streaming - output text at cursor position as you speak
    @AppStorage("realTimeStreaming") private var realTimeStreaming = false

    /// Sort option for models
    @State private var sortBy: ModelSortOption = .none

    /// Show NeMo setup sheet
    @State private var showingNemoSetup = false

    /// Whether we're still checking NeMo status (prevents false "Install NeMo" display)
    @State private var isCheckingNemoStatus = true

    /// Computed property for effective NeMo availability
    /// - In development mode: trust backend's nemoAvailable (devs may have NeMo in main env)
    /// - In production mode: require user to install via UI (nemoVenvExists)
    private var effectiveNemoAvailable: Bool {
        appState.nemoVenvExists || (appState.backendMode == "development" && appState.nemoAvailable)
    }

    /// Sort options for models
    enum ModelSortOption: String, CaseIterable {
        case none = "None"
        case speed = "Speed"
        case accuracy = "Accuracy"
    }

    /// Helper to get performance metrics for sorting
    private func getPerformanceMetrics(for model: ModelInfoResponse) -> (speed: Double, accuracy: Double) {
        switch model.id {
        case "parakeet-tdt-v2", "parakeet-v2":
            return (speed: 0.95, accuracy: 0.85)
        case "parakeet-tdt-v3", "parakeet-v3":
            return (speed: 0.85, accuracy: 0.85)
        case "nemotron-streaming":
            return (speed: 1.00, accuracy: 0.70)
        case "whisper-base-en", "base.en":
            return (speed: 1.00, accuracy: 0.60)
        case "whisper-small-en", "small.en":
            return (speed: 0.92, accuracy: 0.65)
        case "whisper-large-v3-turbo", "large-v3-turbo":
            return (speed: 0.85, accuracy: 0.90)
        case "whisper-large-v3", "large-v3":
            return (speed: 0.80, accuracy: 0.98)
        default:
            return (speed: 0.70, accuracy: 0.75)
        }
    }

    /// Sort models based on selected option
    private func sortedModels(_ models: [ModelInfoResponse]) -> [ModelInfoResponse] {
        switch sortBy {
        case .none:
            return models
        case .speed:
            return models.sorted { getPerformanceMetrics(for: $0).speed > getPerformanceMetrics(for: $1).speed }
        case .accuracy:
            return models.sorted { getPerformanceMetrics(for: $0).accuracy > getPerformanceMetrics(for: $1).accuracy }
        }
    }

    /// Group models by streaming capability and author
    var streamingModels: [ModelInfoResponse] {
        sortedModels(appState.availableModels.filter { $0.streamingNative })
    }

    var nvidiaModels: [ModelInfoResponse] {
        sortedModels(appState.availableModels.filter { $0.author == "NVIDIA" && !$0.streamingNative })
    }

    var openaiModels: [ModelInfoResponse] {
        sortedModels(appState.availableModels.filter { $0.author == "OpenAI" })
    }

    /// All models sorted when sort is active
    var allSortedModels: [ModelInfoResponse] {
        sortedModels(appState.availableModels)
    }

    var body: some View {
        ScrollView {
            VStack(spacing: 0) {
                // Header
                HStack {
                    Text("Speech-to-Text Models")
                        .font(.headline)
                    Spacer()

                    // Sort picker
                    HStack(spacing: 6) {
                        Text("Sort by:")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Picker("", selection: $sortBy) {
                            ForEach(ModelSortOption.allCases, id: \.self) { option in
                                Text(option.rawValue).tag(option)
                            }
                        }
                        .pickerStyle(.menu)
                        .frame(width: 100)
                    }

                    Button {
                        Task { await appState.fetchAvailableModels() }
                    } label: {
                        Image(systemName: "arrow.clockwise")
                    }
                    .buttonStyle(.plain)
                    .help("Refresh models")
                }
                .padding()

                Divider()

                // Show loading indicator only if models are empty AND server is connected
                if appState.availableModels.isEmpty {
                    if appState.isServerConnected {
                        VStack(spacing: 12) {
                            ProgressView()
                            Text("Loading models...")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 40)
                    } else {
                        VStack(spacing: 12) {
                            Image(systemName: "exclamationmark.triangle")
                                .font(.largeTitle)
                                .foregroundColor(.orange)
                            Text("Backend not connected")
                                .font(.headline)
                            Text("Waiting for server to start...")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 40)
                    }
                } else {
                    // Model cards - either grouped or flat sorted list
                    VStack(spacing: 16) {
                        if sortBy == .none {
                            // Grouped sections (original behavior)
                            if !streamingModels.isEmpty {
                                modelSection(
                                    title: "Real-Time Streaming",
                                    subtitle: "Optimized for live transcription with sub-100ms latency",
                                    models: streamingModels
                                )
                            }

                            if !nvidiaModels.isEmpty {
                                modelSection(
                                    title: "NVIDIA Parakeet",
                                    subtitle: "High accuracy batch transcription",
                                    models: nvidiaModels
                                )
                            }

                            if !openaiModels.isEmpty {
                                modelSection(
                                    title: "OpenAI Whisper",
                                    subtitle: "Versatile multilingual transcription",
                                    models: openaiModels
                                )
                            }
                        } else {
                            // Flat sorted list when sorting is active
                            modelSection(
                                title: sortBy == .speed ? "All Models (Sorted by Speed)" : "All Models (Sorted by Accuracy)",
                                subtitle: sortBy == .speed ? "Fastest to slowest" : "Highest to lowest accuracy",
                                models: allSortedModels
                            )
                        }
                    }
                    .padding(.horizontal, 20)
                    .padding(.vertical, 12)
                    .padding(.trailing, 4)
                }

                Divider()

                // NeMo status section
                nemoStatusSection

                Divider()

                // Storage info footer
                storageFooter
            }
        }
        .onAppear {
            // Fetch models - this will be quick if already cached
            Task { await appState.fetchAvailableModels() }

            // Check NeMo status in background (don't block model display)
            Task {
                // Quick local check first (instant)
                appState.refreshNemoVenvState()
                isCheckingNemoStatus = false

                // Then do the backend check asynchronously
                await appState.checkNemoStatus()
            }
        }
        // Re-fetch models when server connection changes
        .onChange(of: appState.isServerConnected) { _, isConnected in
            if isConnected && appState.availableModels.isEmpty {
                Task { await appState.fetchAvailableModels() }
            }
        }
        .sheet(isPresented: $showingNemoSetup) {
            NemoSetupView(
                setupManager: appState.nemoSetupManager,
                onComplete: {
                    // Refresh state after installation
                    Task {
                        await appState.checkNemoStatus()
                        await appState.fetchAvailableModels()
                    }
                }
            )
            .environmentObject(appState)
            .environmentObject(backendManager)
        }
    }

    // MARK: - NeMo Status Section

    private var nemoStatusSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "cpu")
                    .foregroundColor(.purple)
                Text("NeMo Support")
                    .font(.headline)
                Spacer()
            }

            // Show loading state while checking NeMo status
            if isCheckingNemoStatus {
                HStack(spacing: 8) {
                    ProgressView()
                        .scaleEffect(0.8)
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Checking NeMo status...")
                            .font(.subheadline)
                        Text("Please wait")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    Spacer()
                }
            }
            // Use effectiveNemoAvailable which considers both backend mode and venv
            else if effectiveNemoAvailable {
                // NeMo is available (either via venv or dev mode)
                HStack(spacing: 8) {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(.green)
                    VStack(alignment: .leading, spacing: 2) {
                        if appState.nemoVenvExists {
                            Text("NeMo is installed")
                                .font(.subheadline)
                        } else {
                            Text("NeMo available (development)")
                                .font(.subheadline)
                        }
                        if let version = appState.nemoVersion {
                            Text("Version: \(version)")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                    Spacer()
                }
            } else if appState.nemoVenvExists && !appState.nemoAvailable {
                // Venv exists but NeMo not available in current backend (needs restart)
                HStack(spacing: 8) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundColor(.orange)
                    VStack(alignment: .leading, spacing: 2) {
                        Text("NeMo installed but not active")
                            .font(.subheadline)
                        Text("Restart the app to use NeMo models")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    Spacer()
                    Button("Restart Backend") {
                        Task {
                            await backendManager.switchToNemoMode()
                            await appState.checkNemoStatus()
                        }
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }
            } else {
                // NeMo not available
                HStack(spacing: 8) {
                    Image(systemName: "xmark.circle")
                        .foregroundColor(.secondary)
                    VStack(alignment: .leading, spacing: 2) {
                        Text("NeMo not installed")
                            .font(.subheadline)
                        Text("Required for NVIDIA Parakeet & Nemotron models (~3GB)")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    Spacer()
                    Button("Install NeMo") {
                        showingNemoSetup = true
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
                }
            }

            // Remove NeMo button (when installed via UI and not checking)
            if appState.nemoVenvExists && !isCheckingNemoStatus {
                HStack {
                    Spacer()
                    Button(role: .destructive) {
                        Task {
                            try? await appState.nemoSetupManager.removeNemoVenv()
                            await backendManager.switchToWhisperMode()
                            await appState.checkNemoStatus()
                        }
                    } label: {
                        Label("Remove NeMo", systemImage: "trash")
                            .font(.caption)
                    }
                    .buttonStyle(.borderless)
                    .foregroundColor(.red)
                }
            }
        }
        .padding()
        .background(Color(NSColor.controlBackgroundColor))
        .cornerRadius(8)
        .padding(.horizontal, 20)
        .padding(.vertical, 8)
    }

    /// Section for a group of models
    @ViewBuilder
    private func modelSection(title: String, subtitle: String, models: [ModelInfoResponse]) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            // Section header
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.headline)
                Text(subtitle)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            .padding(.bottom, 4)

            // Model cards
            ForEach(models) { model in
                ModelCardView(
                    model: model,
                    isLoaded: appState.loadedModelId == model.id,
                    isLoading: appState.loadingModelId == model.id,
                    isDownloading: appState.downloadingModelId == model.id,
                    // Use effectiveNemoAvailable which considers both backend mode and venv existence
                    nemoAvailable: effectiveNemoAvailable,
                    isCheckingNemoStatus: isCheckingNemoStatus,
                    realTimeStreaming: $realTimeStreaming,
                    onLoad: {
                        // Save as preferred model
                        appState.setPreferredModel(model.id)
                        // Load the model
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
                    onInstallNemo: {
                        showingNemoSetup = true
                    }
                )
            }
        }
    }

    /// Storage information footer
    private var storageFooter: some View {
        HStack {
            Image(systemName: "info.circle")
                .foregroundColor(.secondary)
            Text("Only one model can be loaded at a time. Models are downloaded on first use.")
                .font(.caption)
                .foregroundColor(.secondary)
            Spacer()

            // Open storage folder button (HuggingFace cache)
            Button {
                openStorageFolder()
            } label: {
                HStack(spacing: 4) {
                    Image(systemName: "folder")
                    Text("Open Cache")
                }
                .font(.caption)
            }
            .buttonStyle(.plain)
            .foregroundColor(.accentColor)
            .help("Open HuggingFace model cache folder")
        }
        .padding(8)
    }

    /// Open the model storage folder in Finder (HuggingFace cache)
    private func openStorageFolder() {
        // Models are stored in HuggingFace cache
        let path = NSString(string: "~/.cache/huggingface/hub").expandingTildeInPath
        let url = URL(fileURLWithPath: path)

        // Create directory if it doesn't exist
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)

        NSWorkspace.shared.selectFile(nil, inFileViewerRootedAtPath: url.path)
    }
}

// MARK: - Model Card View

/// Reusable model card component
struct ModelCardView: View {
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

    @State private var languagesExpanded = false
    @State private var showingDeleteConfirmation = false

    /// Whether this model can be used (has required dependencies)
    private var canUseModel: Bool {
        if model.nemoRequired {
            // If still checking, assume it's available to avoid showing false "Install NeMo"
            if isCheckingNemoStatus {
                return true
            }
            return nemoAvailable
        }
        return true
    }

    /// Whether to show the "Requires NeMo" badge
    private var showNemoRequiredBadge: Bool {
        // Don't show while still checking - prevents flickering
        model.nemoRequired && !nemoAvailable && !isCheckingNemoStatus
    }

    /// Format size for display
    private func formatSize(_ mb: Int) -> String {
        if mb >= 1024 {
            return String(format: "%.1f GB", Double(mb) / 1024.0)
        }
        return "\(mb) MB"
    }

    /// URL to the model's page (HuggingFace or GitHub)
    private var huggingFaceURL: URL? {
        if let urlString = model.modelUrl {
            return URL(string: urlString)
        }
        // Fallback for NeMo models without model_url
        if model.type == "nemo" {
            return URL(string: "https://huggingface.co/\(model.name)")
        }
        return nil
    }

    /// Get performance metrics for a model (speed and accuracy on a 0-1 scale)
    private func getPerformanceMetrics() -> (speed: Double, accuracy: Double) {
        switch model.id {
        case "parakeet-tdt-v2", "parakeet-v2":
            return (speed: 0.95, accuracy: 0.85)
        case "parakeet-tdt-v3", "parakeet-v3":
            return (speed: 0.85, accuracy: 0.85)
        case "nemotron-streaming":
            return (speed: 1.00, accuracy: 0.70)
        case "whisper-base-en", "base.en":
            return (speed: 1.00, accuracy: 0.60)
        case "whisper-small-en", "small.en":
            return (speed: 0.92, accuracy: 0.65)
        case "whisper-large-v3-turbo", "large-v3-turbo":
            return (speed: 0.85, accuracy: 0.90)
        case "whisper-large-v3", "large-v3":
            return (speed: 0.80, accuracy: 0.98)
        default:
            print("⚠️ Unknown model ID for performance metrics: \(model.id)")
            return (speed: 0.70, accuracy: 0.75)
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header with badges
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 8) {
                        Text(model.displayName)
                            .font(.title3)
                            .fontWeight(.semibold)

                        // Downloaded badge (only for downloaded models)
                        if model.downloaded {
                            HStack(spacing: 4) {
                                Image(systemName: "arrow.down.circle.fill")
                                    .foregroundColor(.blue)
                                Text("Downloaded")
                                    .foregroundColor(.blue)
                            }
                            .font(.caption)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(Color.blue.opacity(0.15))
                            .cornerRadius(6)
                        }

                        // Active badge
                        if isLoaded {
                            HStack(spacing: 4) {
                                Image(systemName: "checkmark.circle.fill")
                                    .foregroundColor(.green)
                                Text("Active")
                                    .foregroundColor(.green)
                            }
                            .font(.caption)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(Color.green.opacity(0.15))
                            .cornerRadius(6)
                        }

                        // Real-Time Streaming badge (ONLY for Nemotron Streaming)
                        if model.streamingNative && model.id == "nemotron-streaming" {
                            HStack(spacing: 4) {
                                Image(systemName: "waveform")
                                    .foregroundColor(.orange)
                                Text("Real-Time")
                                    .foregroundColor(.orange)
                            }
                            .font(.caption)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(Color.orange.opacity(0.15))
                            .cornerRadius(6)
                        }

                        // Multilingual badge
                        if model.multilingual {
                            HStack(spacing: 4) {
                                Image(systemName: "globe")
                                    .foregroundColor(.purple)
                                Text("Multilingual")
                                    .foregroundColor(.purple)
                            }
                            .font(.caption)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(Color.purple.opacity(0.15))
                            .cornerRadius(6)
                        }

                        // Session-compatible badge
                        if model.sessionCompatible {
                            HStack(spacing: 4) {
                                Image(systemName: "waveform.circle")
                                    .foregroundColor(.teal)
                                Text("Sessions")
                                    .foregroundColor(.teal)
                            }
                            .font(.caption)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(Color.teal.opacity(0.15))
                            .cornerRadius(6)
                        }

                        // Requires NeMo badge (for NeMo models when NeMo is not available)
                        if showNemoRequiredBadge {
                            HStack(spacing: 4) {
                                Image(systemName: "exclamationmark.triangle.fill")
                                    .foregroundColor(.orange)
                                Text("Requires NeMo")
                                    .foregroundColor(.orange)
                            }
                            .font(.caption)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(Color.orange.opacity(0.15))
                            .cornerRadius(6)
                        }
                    }

                    Text("by \(model.author)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                Spacer()

                // View on HuggingFace link
                if let url = huggingFaceURL {
                    Link(destination: url) {
                        HStack(spacing: 4) {
                            Image(systemName: "arrow.up.right.square")
                            Text("View")
                        }
                        .font(.caption)
                    }
                }
            }

            // Description
            Text(model.description)
                .font(.subheadline)
                .foregroundColor(.secondary)

            // Model specs
            HStack(spacing: 16) {
                HStack(spacing: 4) {
                    Image(systemName: "arrow.down.circle")
                    Text(formatSize(model.sizeMB))
                }
                .help("Download size")

                HStack(spacing: 4) {
                    Image(systemName: "memorychip")
                    Text(formatSize(model.ramMB))
                }
                .help("RAM usage when loaded")

                // Language count
                if model.multilingual && model.languageNames.count > 1 {
                    Button {
                        withAnimation(.easeInOut(duration: 0.2)) {
                            languagesExpanded.toggle()
                        }
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: "globe")
                            if model.languageNames.first == "99+ languages" {
                                Text("99+ languages")
                            } else {
                                Text("\(model.languageNames.count) languages")
                            }
                            Image(systemName: languagesExpanded ? "chevron.up" : "chevron.down")
                                .font(.caption2)
                        }
                    }
                    .buttonStyle(.plain)
                    .foregroundColor(.secondary)
                    .help("Click to see supported languages")
                } else {
                    HStack(spacing: 4) {
                        Image(systemName: "globe")
                        Text(model.languageNames.first ?? "English")
                    }
                    .help("Supported language")
                }
            }
            .font(.caption)
            .foregroundColor(.secondary)

            // Performance bars
            VStack(spacing: 8) {
                let metrics = getPerformanceMetrics()

                // Speed bar
                HStack(spacing: 8) {
                    Image(systemName: "bolt.fill")
                        .foregroundColor(.purple)
                        .frame(width: 16)
                    Text("Speed")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .frame(width: 60, alignment: .leading)

                    GeometryReader { geometry in
                        ZStack(alignment: .leading) {
                            // Background
                            RoundedRectangle(cornerRadius: 3)
                                .fill(Color.secondary.opacity(0.2))
                                .frame(height: 6)

                            // Filled portion
                            RoundedRectangle(cornerRadius: 3)
                                .fill(Color.purple)
                                .frame(width: geometry.size.width * metrics.speed, height: 6)
                        }
                    }
                    .frame(height: 6)
                }

                // Accuracy bar
                HStack(spacing: 8) {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(.purple)
                        .frame(width: 16)
                    Text("Accuracy")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .frame(width: 60, alignment: .leading)

                    GeometryReader { geometry in
                        ZStack(alignment: .leading) {
                            // Background
                            RoundedRectangle(cornerRadius: 3)
                                .fill(Color.secondary.opacity(0.2))
                                .frame(height: 6)

                            // Filled portion
                            RoundedRectangle(cornerRadius: 3)
                                .fill(Color.purple)
                                .frame(width: geometry.size.width * metrics.accuracy, height: 6)
                        }
                    }
                    .frame(height: 6)
                }
            }
            .padding(.vertical, 8)

            // Expanded language grid (for multilingual models)
            if model.multilingual && languagesExpanded && model.languageNames.first != "99+ languages" {
                LanguageGridView(languages: model.languageNames)
                    .transition(.opacity.combined(with: .scale(scale: 0.95, anchor: .top)))
            }

            Divider()

            // Action buttons row
            HStack(spacing: 12) {
                // Left side: Action button
                if showNemoRequiredBadge {
                    Button {
                        onInstallNemo()
                    } label: {
                        HStack {
                            Image(systemName: "cpu")
                            Text("Install NeMo Support")
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.orange)
                }
                // Download button (all models must be downloaded first)
                else if !model.downloaded {
                    Button {
                        Task { await onDownload() }
                    } label: {
                        HStack {
                            Image(systemName: "arrow.down.circle")
                            Text("Download")
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(isDownloading || !canUseModel)
                } else if isLoaded {
                    // Unload button
                    Button {
                        Task { await onUnload() }
                    } label: {
                        HStack {
                            Image(systemName: "eject")
                            Text("Unload")
                        }
                    }
                    .buttonStyle(.bordered)

                    // Streaming toggle
                    if model.streamingNative {
                        Toggle("Live Streaming", isOn: $realTimeStreaming)
                            .toggleStyle(.switch)
                            .help("Stream transcription to cursor position in real-time")
                            .onChange(of: realTimeStreaming) { _, newValue in
                                onStreamingChanged(newValue)
                            }
                    } else {
                        Toggle("Streaming", isOn: $realTimeStreaming)
                            .toggleStyle(.switch)
                            .help("Not optimized - re-transcribes entire buffer each time")
                            .onChange(of: realTimeStreaming) { _, newValue in
                                onStreamingChanged(newValue)
                            }
                    }
                } else {
                    // Load button
                    Button {
                        Task { await onLoad() }
                    } label: {
                        HStack {
                            Image(systemName: "play.fill")
                            Text("Load Model")
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(isLoading || !model.downloaded || !canUseModel)
                }

                Spacer()

                // Right side: Loading/Downloading indicators and delete button
                HStack(spacing: 12) {
                    // Loading indicator
                    if isLoading {
                        HStack(spacing: 4) {
                            ProgressView()
                                .scaleEffect(0.7)
                            Text("Loading...")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }

                    // Downloading indicator with animated shimmer bar
                    if isDownloading {
                        HStack(spacing: 6) {
                            AnimatedDownloadingBar()
                                .frame(width: 60, height: 6)
                            Text("Downloading...")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }

                    // Delete button (only for downloaded, non-loaded models)
                    if model.downloaded && !isLoaded {
                        Button {
                            showingDeleteConfirmation = true
                        } label: {
                            Image(systemName: "trash")
                                .foregroundColor(.red)
                        }
                        .buttonStyle(.borderless)
                        .help("Delete downloaded model")
                    }
                }
            }

            // Streaming info/warning
            if isLoaded && realTimeStreaming {
                HStack(spacing: 6) {
                    if model.streamingNative {
                        // Streaming-native model - info message
                        Image(systemName: "info.circle.fill")
                            .foregroundColor(.blue)
                        Text("Cache-aware streaming: each audio chunk is processed once for optimal performance")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    } else {
                        // Non-streaming model - warning
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundColor(.orange)
                        Text("Simulated streaming: re-transcribes entire buffer each time. Use Nemotron Streaming for better performance.")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
                .padding(.top, 4)
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(Color(NSColor.controlBackgroundColor))
                .shadow(color: Color.black.opacity(0.1), radius: 2, x: 0, y: 1)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .strokeBorder(isLoaded ? Color.green.opacity(0.5) : Color.secondary.opacity(0.2), lineWidth: isLoaded ? 2 : 1)
        )
        .confirmationDialog("Delete Model", isPresented: $showingDeleteConfirmation) {
            Button("Delete \(model.displayName)", role: .destructive) {
                Task { await onDelete() }
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This will permanently delete the downloaded model files. You can download it again later.")
        }
    }
}

// MARK: - Language Grid View

/// Grid view for displaying supported languages
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
                .font(.caption)
                .fontWeight(.medium)
                .foregroundColor(.secondary)

            LazyVGrid(columns: columns, spacing: 6) {
                ForEach(languages, id: \.self) { language in
                    Text(language)
                        .font(.caption2)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Color.secondary.opacity(0.1))
                        .cornerRadius(4)
                }
            }
        }
        .padding(10)
        .background(Color(NSColor.windowBackgroundColor))
        .cornerRadius(8)
    }
}
