/*
 SettingsView.swift
 ==================

 The Settings window for configuring the app.

 This opens when:
 - User clicks "Settings..." in the menu
 - User presses Cmd+, (standard macOS shortcut)

 TABVIEW:
 --------
 Settings windows often have tabs (General, Shortcuts, Advanced, etc.)
 SwiftUI's TabView makes this easy.

 @AppStorage:
 ------------
 @AppStorage is SwiftUI's way to persist settings.
 It automatically saves to UserDefaults (macOS's built-in settings storage).

 Example:
   @AppStorage("showPreview") var showPreview = true

 This:
 1. Loads from UserDefaults on app start
 2. Saves to UserDefaults when changed
 3. Updates UI automatically when changed
 */

import SwiftUI
import Carbon.HIToolbox  // For key code constants
import AVFoundation  // For microphone permission

struct SettingsView: View {
    @State private var selectedTab = 0
    @EnvironmentObject var backendManager: BackendManager

    var body: some View {
        TabView(selection: $selectedTab) {
            DashboardView()
                .id("dashboard")
                .tabItem {
                    Label("Dashboard", systemImage: "square.grid.2x2")
                }
                .tag(0)

            GeneralSettingsView()
                .id("general")
                .tabItem {
                    Label("General", systemImage: "gear")
                }
                .tag(1)

            ModelsSettingsView()
                .id("models")
                .tabItem {
                    Label("Models", systemImage: "cube.box")
                }
                .tag(2)

            HistorySettingsView()
                .id("history")
                .tabItem {
                    Label("History", systemImage: "clock.arrow.circlepath")
                }
                .tag(3)

            ShortcutsSettingsView()
                .id("shortcuts")
                .tabItem {
                    Label("Shortcuts", systemImage: "keyboard")
                }
                .tag(4)

            AboutView()
                .id("about")
                .tabItem {
                    Label("About", systemImage: "info.circle")
                }
                .tag(5)
        }
        .frame(width: 550, height: 450)
        .onAppear {
            // Pause health check timer to prevent UI re-renders that cause tab icon shivering
            backendManager.pauseHealthCheck()
        }
        .onDisappear {
            // Resume health check timer when Settings window closes
            backendManager.resumeHealthCheck()
        }
    }
}

// =============================================================================
// DASHBOARD (MOCK-UP)
// =============================================================================

struct DashboardView: View {
    // NOTE: Removed unused @EnvironmentObject var appState to prevent unnecessary re-renders
    // that caused tab icon squishing when switching to this tab

    var body: some View {
        // Wrap in ScrollView to match layout behavior of Form-based tabs
        // This prevents layout differences that cause tab icon squishing
        ScrollView {
            VStack(spacing: 24) {
            // Welcome Header - centered layout
            VStack(spacing: 12) {
                // App Icon centered at top
                Group {
                    if let nsImage = loadDashboardIconFromBundle() {
                        Image(nsImage: nsImage)
                            .resizable()
                            .scaledToFit()
                            .frame(width: 72, height: 72)
                    } else {
                        // Fallback to SF Symbol
                        Image(systemName: "mic.circle.fill")
                            .resizable()
                            .scaledToFit()
                            .frame(width: 72, height: 72)
                            .foregroundStyle(.blue.gradient)
                    }
                }

                VStack(spacing: 4) {
                    Text("Welcome to AIT")
                        .font(.system(size: 28, weight: .bold))

                    Text("Your AI-powered transcription assistant")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
            }
            .frame(maxWidth: .infinity)
            .padding(.top, 24)

            Divider()
                .padding(.horizontal, 24)
            
            // Mock content area
            VStack(spacing: 16) {
                Text("Dashboard Coming Soon")
                    .font(.title2.weight(.semibold))
                    .foregroundColor(.secondary)
                
                Text("This is where you'll see your transcription stats, quick actions, and recent activity.")
                    .font(.body)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 40)
                
                // Placeholder cards
                HStack(spacing: 16) {
                    PlaceholderCard(
                        icon: "waveform",
                        title: "Quick Stats",
                        subtitle: "View your activity"
                    )
                    
                    PlaceholderCard(
                        icon: "clock.arrow.circlepath",
                        title: "Recent",
                        subtitle: "Latest transcriptions"
                    )
                    
                    PlaceholderCard(
                        icon: "bolt.fill",
                        title: "Quick Actions",
                        subtitle: "Common tasks"
                    )
                }
                .padding(.horizontal, 24)
            }

            // Spacer replaced with fixed padding (Spacer doesn't work in ScrollView)
            Spacer()
                .frame(minHeight: 20)

            // Hint to explore other tabs
            VStack(spacing: 8) {
                Text("Explore the tabs above to configure AIT")
                    .font(.footnote)
                    .foregroundColor(.secondary)
                
                HStack(spacing: 12) {
                    Label("General", systemImage: "gear")
                    Label("Models", systemImage: "cube.box")
                    Label("History", systemImage: "clock.arrow.circlepath")
                }
                .font(.caption)
                .foregroundColor(.secondary)
            }
            .padding(.bottom, 24)
            }
        }
    }
}

// MARK: - Dashboard Icon Loading Helper

/// Attempts to load the app icon from the bundle resources for Dashboard
private func loadDashboardIconFromBundle() -> NSImage? {
    // List of possible icon filenames to try
    let iconNames = [
        "Icon-iOS-Default-256x256@1x",
        "Icon-iOS-Default-128x128@1x",
        "Icon-iOS-Default-512x512@1x",
        "AppIcon",
        "Icon"
    ]
    
    for iconName in iconNames {
        // Try loading as PNG
        if let image = NSImage(named: iconName) {
            return image
        }
        
        // Try loading from bundle resources
        if let imagePath = Bundle.main.path(forResource: iconName, ofType: "png"),
           let image = NSImage(contentsOfFile: imagePath) {
            return image
        }
    }
    
    return nil
}

// Placeholder card for dashboard mock-up
struct PlaceholderCard: View {
    let icon: String
    let title: String
    let subtitle: String
    
    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: icon)
                .font(.system(size: 32))
                .foregroundColor(.accentColor.opacity(0.5))
            
            VStack(spacing: 4) {
                Text(title)
                    .font(.headline)
                    .foregroundColor(.primary.opacity(0.5))
                
                Text(subtitle)
                    .font(.caption)
                    .foregroundColor(.secondary.opacity(0.5))
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 24)
        .background(Color.gray.opacity(0.05))
        .cornerRadius(12)
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(Color.gray.opacity(0.2), style: StrokeStyle(lineWidth: 1, dash: [5, 5]))
        )
    }
}


// =============================================================================
// GENERAL SETTINGS
// =============================================================================

struct GeneralSettingsView: View {
    @EnvironmentObject var appState: AppState

    /// Auto-paste after transcription completes (simulates Cmd+V)
    @AppStorage("autoPasteAfterTranscription") private var autoPasteAfterTranscription = false

    /// Auto-start recording on app launch
    @AppStorage("autoLoadModel") private var autoLoadModel = true

    /// Play sound when recording starts/stops
    @AppStorage("playSounds") private var playSounds = true

    /// Recording indicator position
    @AppStorage("indicatorPosition") private var indicatorPosition = "topCenter"

    var body: some View {
        Form {
            Section("Transcription") {
                Text("Transcribed text is automatically copied to clipboard")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                Toggle("Auto-paste after transcription", isOn: $autoPasteAfterTranscription)
                    .help("Automatically paste (Cmd+V) after transcription completes")
                    .onChange(of: autoPasteAfterTranscription) { _, newValue in
                        if newValue {
                            appState.checkAccessibilityPermissionsIfNeeded()
                        }
                    }

                if autoPasteAfterTranscription {
                    Text("Text will be pasted at cursor position after copying to clipboard")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }

            Section("Recording Indicator") {
                Picker("Position", selection: $indicatorPosition) {
                    ForEach(RecordingIndicatorController.ScreenPosition.allCases, id: \.rawValue) { pos in
                        Text(pos.displayName).tag(pos.rawValue)
                    }
                }
                .onChange(of: indicatorPosition) { _, newValue in
                    if let pos = RecordingIndicatorController.ScreenPosition(rawValue: newValue) {
                        appState.recordingIndicator.setPosition(pos)
                    }
                }
            }

            Section("Startup") {
                Toggle("Load model on app launch", isOn: $autoLoadModel)
            }

            Section("Feedback") {
                Toggle("Play sounds", isOn: $playSounds)
            }

            Section("Audio") {
                AudioDuckingSettings()
            }

            Section("Permissions") {
                // Microphone permission status
                HStack {
                    Text("Microphone Access")
                    Spacer()
                    microphoneStatusView
                }

                // Request permission button (only show if not determined or denied)
                if microphoneStatus != .authorized {
                    Button("Request Microphone Permission") {
                        requestMicrophonePermission()
                    }
                    .buttonStyle(.borderedProminent)
                }

                // Open System Settings button
                Button("Open Microphone Settings...") {
                    openMicrophoneSettings()
                }

                Text("If the app doesn't appear in Microphone settings, click 'Request Microphone Permission' first, then check again.")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .formStyle(.grouped)
        .padding()
    }

    // MARK: - Microphone Permission

    /// Current microphone authorization status
    private var microphoneStatus: AVAuthorizationStatus {
        AVCaptureDevice.authorizationStatus(for: .audio)
    }

    /// View showing microphone permission status
    @ViewBuilder
    private var microphoneStatusView: some View {
        switch microphoneStatus {
        case .authorized:
            HStack(spacing: 4) {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundColor(.green)
                Text("Granted")
                    .foregroundColor(.green)
            }
        case .denied, .restricted:
            HStack(spacing: 4) {
                Image(systemName: "xmark.circle.fill")
                    .foregroundColor(.red)
                Text("Denied")
                    .foregroundColor(.red)
            }
        case .notDetermined:
            HStack(spacing: 4) {
                Image(systemName: "questionmark.circle.fill")
                    .foregroundColor(.orange)
                Text("Not Requested")
                    .foregroundColor(.orange)
            }
        @unknown default:
            Text("Unknown")
                .foregroundColor(.secondary)
        }
    }

    /// Request microphone permission - this triggers the system dialog
    private func requestMicrophonePermission() {
        AVCaptureDevice.requestAccess(for: .audio) { granted in
            DispatchQueue.main.async {
                if granted {
                    appState.statusMessage = "Microphone permission granted"
                } else {
                    appState.statusMessage = "Microphone permission denied - enable in System Settings"
                }
            }
        }
    }

    /// Open System Settings > Privacy & Security > Microphone
    private func openMicrophoneSettings() {
        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone") {
            NSWorkspace.shared.open(url)
        }
    }

    /// Open System Settings > Privacy & Security > Accessibility
    private func openAccessibilitySettings() {
        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility") {
            NSWorkspace.shared.open(url)
        }
    }
}


// =============================================================================
// MODELS SETTINGS
// =============================================================================

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
            return (speed: 0.98, accuracy: 0.70)
        case "whisper-base-en", "base.en":
            return (speed: 0.75, accuracy: 0.70)
        case "whisper-small-en", "small.en":
            return (speed: 0.75, accuracy: 0.70)
        case "whisper-large-v3-turbo", "large-v3-turbo":
            return (speed: 0.50, accuracy: 0.90)
        case "whisper-large-v3", "large-v3":
            return (speed: 0.35, accuracy: 0.98)
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

/// Reusable model card component
struct ModelCardView: View {
    let model: ModelInfoResponse
    let isLoaded: Bool
    let isLoading: Bool
    let isDownloading: Bool
    let nemoAvailable: Bool
    let isCheckingNemoStatus: Bool  // New: indicates if NeMo status is still being checked
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

    /// Compute the correct HuggingFace URL for this model
    private var huggingFaceURL: URL? {
        var repoPath: String

        if model.type == "nemo" {
            // NeMo models use their name directly (e.g., "nvidia/parakeet-tdt-0.6b-v2")
            repoPath = model.name
        } else {
            // Whisper models via faster-whisper
            if model.name.contains("/") {
                // Already a full path (e.g., "deepdml/faster-whisper-large-v3-turbo-ct2")
                repoPath = model.name
            } else {
                // Standard size name (e.g., "base.en") -> "Systran/faster-whisper-base.en"
                repoPath = "Systran/faster-whisper-\(model.name)"
            }
        }

        return URL(string: "https://huggingface.co/\(repoPath)")
    }
    
    /// Get performance metrics for a model (speed and accuracy on a 0-1 scale)
    private func getPerformanceMetrics() -> (speed: Double, accuracy: Double) {
        // Based on actual hardware testing on M4 24GB
        // NOTE: Last updated 2026-01-16 - v2 and v3 both at 95% speed, V3 at 35% speed
        switch model.id {
        case "parakeet-tdt-v2", "parakeet-v2":
            return (speed: 0.95, accuracy: 0.85)  // Very fast, great accuracy
        case "parakeet-tdt-v3", "parakeet-v3":
            return (speed: 0.85, accuracy: 0.85)  // Very fast, good accuracy (multilingual)
        case "nemotron-streaming":
            return (speed: 0.98, accuracy: 0.70)  // Fastest, lower accuracy
        case "whisper-base-en", "base.en":
            return (speed: 0.75, accuracy: 0.70)  // Fast, decent accuracy
        case "whisper-small-en", "small.en":
            return (speed: 0.75, accuracy: 0.70)  // Fast, good accuracy
        case "whisper-large-v3-turbo", "large-v3-turbo":
            return (speed: 0.50, accuracy: 0.90)  // Slower, excellent accuracy
        case "whisper-large-v3", "large-v3":
            return (speed: 0.35, accuracy: 0.98)  // Slow, exceptional accuracy
        default:
            // Debug: Print unknown model IDs
            print("⚠️ Unknown model ID for performance metrics: \(model.id)")
            return (speed: 0.70, accuracy: 0.75)  // Default values
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

                        // Requires NeMo badge (for NeMo models when NeMo is not available)
                        // Don't show while checking to prevent flickering
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
                // Show "Install NeMo" button for NeMo models when NeMo is not available
                // Don't show while still checking status to prevent flickering
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


// =============================================================================
// HISTORY VIEW
// =============================================================================

struct HistorySettingsView: View {
    @EnvironmentObject var appState: AppState
    @State private var searchText = ""
    
    var filteredHistory: [TranscriptionEntry] {
        if searchText.isEmpty {
            return appState.transcriptionHistory
        }
        return appState.transcriptionHistory.filter {
            $0.text.localizedCaseInsensitiveContains(searchText)
        }
    }
    
    var body: some View {
        VStack(spacing: 0) {
            // Search bar
            HStack {
                Image(systemName: "magnifyingglass")
                    .foregroundColor(.secondary)
                TextField("Search transcriptions...", text: $searchText)
                    .textFieldStyle(.plain)
                if !searchText.isEmpty {
                    Button {
                        searchText = ""
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundColor(.secondary)
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(8)
            .background(Color(NSColor.textBackgroundColor))

            Divider()

            // History list with pagination
            if filteredHistory.isEmpty {
                VStack(spacing: 12) {
                    Image(systemName: "clock")
                        .font(.system(size: 40))
                        .foregroundColor(.secondary)
                    Text(searchText.isEmpty ? "No transcriptions yet" : "No results found")
                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    LazyVStack(spacing: 0) {
                        ForEach(filteredHistory) { entry in
                            HistoryEntryRow(entry: entry)
                                .contextMenu {
                                    Button("Copy") {
                                        appState.copyToClipboard(entry.text)
                                    }
                                    Divider()
                                    Button("Delete", role: .destructive) {
                                        appState.deleteFromHistory(entry)
                                    }
                                }
                            
                            Divider()
                        }
                        
                        // Load more button - compact and centered
                        if !searchText.isEmpty {
                            // In search mode, show all results (no pagination)
                            Color.clear
                                .frame(height: 1)
                        } else if appState.hasMoreHistory {
                            HStack {
                                Spacer()
                                Button {
                                    Task {
                                        await appState.loadMoreHistory()
                                    }
                                } label: {
                                    if appState.isLoadingMoreHistory {
                                        HStack(spacing: 6) {
                                            ProgressView()
                                                .scaleEffect(0.7)
                                                .frame(width: 12, height: 12)
                                            Text("Loading...")
                                                .font(.caption)
                                        }
                                        .padding(.horizontal, 16)
                                        .padding(.vertical, 6)
                                    } else {
                                        HStack(spacing: 6) {
                                            Text("Load More")
                                                .font(.caption)
                                            Image(systemName: "chevron.down")
                                                .font(.caption2)
                                        }
                                        .padding(.horizontal, 16)
                                        .padding(.vertical, 6)
                                    }
                                }
                                .buttonStyle(.bordered)
                                .disabled(appState.isLoadingMoreHistory)
                                .padding(.vertical, 12)
                                Spacer()
                            }
                        } else {
                            Color.clear
                                .frame(height: 1)
                        }
                    }
                }
            }

            Divider()

            // Footer with count and clear button
            HStack {
                Text("\(filteredHistory.count) transcriptions")
                    .font(.caption)
                    .foregroundColor(.secondary)

                Spacer()

                Button("Clear All") {
                    appState.clearHistory()
                }
                .disabled(filteredHistory.isEmpty)
            }
            .padding(8)
        }
    }
}

struct HistoryEntryRow: View {
    let entry: TranscriptionEntry
    
    private var dateFormatter: DateFormatter {
        let formatter = DateFormatter()
        formatter.dateStyle = .short
        formatter.timeStyle = .short
        return formatter
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            // Main text - left aligned
            Text(entry.text)
                .font(.body)
                .lineLimit(3)
                .multilineTextAlignment(.leading)
                .frame(maxWidth: .infinity, alignment: .leading)
            
            // Metadata row - left aligned
            HStack(spacing: 8) {
                // Timestamp
                Text(dateFormatter.string(from: entry.timestamp))
                    .font(.caption2)
                    .foregroundColor(.secondary)
                
                Text("•")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                
                // Duration
                Text(String(format: "%.0fs", entry.duration))
                    .font(.caption2)
                    .foregroundColor(.secondary)
                
                // Word count
                Text("•")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                Text("\(entry.wordCount) words")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                
                // Model name
                Text("•")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                Text(entry.modelName)
                    .font(.caption2)
                    .foregroundColor(.accentColor)
                
                Spacer()
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.vertical, 8)
        .padding(.horizontal, 12)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}


// =============================================================================
// SHORTCUTS SETTINGS
// =============================================================================

struct ShortcutsSettingsView: View {
    /*
     Keyboard shortcuts are stored as a combination of:
     - Key code (which key)
     - Modifiers (Cmd, Shift, Option, Control)

     We'll store these as strings for simplicity.
     A more robust implementation would use proper key recording.
     */

    @AppStorage("toggleRecordingShortcut") private var toggleShortcut = "⌥Space"
    @AppStorage("cancelRecordingShortcut") private var cancelShortcut = "⌥Escape"

    @State private var isRecordingToggleShortcut = false
    @State private var isRecordingCancelShortcut = false

    var body: some View {
        Form {
            Section("Recording Shortcuts") {
                HStack {
                    Text("Toggle Recording:")
                    Spacer()
                    ShortcutButton(
                        shortcut: $toggleShortcut,
                        isRecording: $isRecordingToggleShortcut,
                        onShortcutChanged: {
                            HotkeyManager.shared.refreshHotkeys()
                        }
                    )
                }

                HStack {
                    Text("Cancel Recording:")
                    Spacer()
                    ShortcutButton(
                        shortcut: $cancelShortcut,
                        isRecording: $isRecordingCancelShortcut,
                        onShortcutChanged: {
                            HotkeyManager.shared.refreshHotkeys()
                        }
                    )
                }
            }

            Section {
                Text("Click a shortcut button and press your desired key combination.")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Text("Default: Option+Space to toggle, Option+Escape to cancel")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .formStyle(.grouped)
        .padding()
    }
}


/// A button that records keyboard shortcuts
struct ShortcutButton: View {
    @Binding var shortcut: String
    @Binding var isRecording: Bool
    var onShortcutChanged: (() -> Void)?

    var body: some View {
        Button {
            isRecording = true
        } label: {
            Text(isRecording ? "Press keys..." : shortcut)
                .frame(minWidth: 100)
        }
        .buttonStyle(.bordered)
        .background(
            /*
             This is a clever trick to capture key events.
             When isRecording is true, we add a hidden key listener.

             Note: For a production app, you'd want to use a proper
             global hotkey library like HotKey or MASShortcut.
             This is simplified for learning purposes.
             */
            KeyEventHandler(isActive: $isRecording, onKeyPress: { key in
                shortcut = key
                isRecording = false
                // Notify that shortcut changed so hotkeys can be refreshed
                onShortcutChanged?()
            })
        )
    }
}


/// Captures key events (simplified implementation)
struct KeyEventHandler: NSViewRepresentable {
    @Binding var isActive: Bool
    let onKeyPress: (String) -> Void

    func makeNSView(context: Context) -> NSView {
        let view = KeyCaptureView()
        view.onKeyPress = onKeyPress
        view.isActiveBinding = $isActive
        return view
    }

    func updateNSView(_ nsView: NSView, context: Context) {
        if let view = nsView as? KeyCaptureView {
            view.isActiveBinding = $isActive
            if isActive {
                nsView.window?.makeFirstResponder(nsView)
            }
        }
    }
}

class KeyCaptureView: NSView {
    var onKeyPress: ((String) -> Void)?
    var isActiveBinding: Binding<Bool>?

    override var acceptsFirstResponder: Bool { true }

    override func keyDown(with event: NSEvent) {
        guard isActiveBinding?.wrappedValue == true else {
            super.keyDown(with: event)
            return
        }

        var parts: [String] = []

        // Add modifier symbols
        if event.modifierFlags.contains(.control) { parts.append("⌃") }
        if event.modifierFlags.contains(.option) { parts.append("⌥") }
        if event.modifierFlags.contains(.shift) { parts.append("⇧") }
        if event.modifierFlags.contains(.command) { parts.append("⌘") }

        // Add the key character
        if let chars = event.charactersIgnoringModifiers?.uppercased() {
            // Map special keys
            let keyString: String
            switch event.keyCode {
            case 49: keyString = "Space"
            case 36: keyString = "Return"
            case 53: keyString = "Escape"
            case 51: keyString = "Delete"
            case 48: keyString = "Tab"
            default: keyString = chars
            }
            parts.append(keyString)
        }

        let shortcut = parts.joined()
        onKeyPress?(shortcut)
    }
}


// =============================================================================
// ABOUT VIEW
// =============================================================================

struct AboutView: View {
    @EnvironmentObject var backendManager: BackendManager

    var body: some View {
        VStack(spacing: 16) {
            // App Icon
            Group {
                if let nsImage = loadAboutIconFromBundle() {
                    Image(nsImage: nsImage)
                        .resizable()
                        .scaledToFit()
                        .frame(width: 96, height: 96)
                        .padding(.top, 20)
                } else {
                    // Fallback to SF Symbol
                    Image(systemName: "mic.circle.fill")
                        .resizable()
                        .scaledToFit()
                        .frame(width: 96, height: 96)
                        .foregroundStyle(.blue.gradient)
                        .padding(.top, 20)
                }
            }
            
            // App name
            Text("AIT")
                .font(.largeTitle)
                .fontWeight(.bold)

            // Version
            Text("Version 0.1.0")
                .font(.subheadline)
                .foregroundColor(.secondary)

            // Description
            Text("Open source picture text transcription")
                .font(.body)
                .multilineTextAlignment(.center)
                .foregroundColor(.secondary)
                .padding(.bottom, 8)

            // Links
            HStack(spacing: 24) {
                // GitHub link
                Link(destination: URL(string: "https://github.com/Ljove02")!) {
                    HStack(spacing: 6) {
                        Image(systemName: "chevron.left.forwardslash.chevron.right")
                            .font(.system(size: 14))
                        Text("GitHub")
                            .font(.caption)
                    }
                }
                .buttonStyle(.plain)
                .foregroundColor(.accentColor)

                // Buy me a coffee / Support
                Link(destination: URL(string: "https://ko-fi.com/yourusername")!) {
                    HStack(spacing: 6) {
                        Image(systemName: "cup.and.saucer.fill")
                            .font(.system(size: 14))
                        Text("Support")
                            .font(.caption)
                    }
                }
                .buttonStyle(.plain)
                .foregroundColor(.accentColor)
            }
            .padding(.vertical, 4)

            Divider()
                .padding(.vertical, 8)

            // Backend status with inline debug console button
            VStack(spacing: 8) {
                HStack(spacing: 8) {
                    Circle()
                        .fill(backendManager.isServerReady ? Color.green : (backendManager.isRunning ? Color.orange : Color.red))
                        .frame(width: 10, height: 10)
                    Text("Backend: \(backendManager.statusMessage)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Spacer()
                    
                    // Debug console button (compact, inline)
                    Button {
                        DebugConsoleWindowController.shared.showWindow(backendManager: backendManager)
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: "terminal")
                                .font(.system(size: 11))
                            Text("Debug Console")
                                .font(.caption2)
                        }
                    }
                    .buttonStyle(.link)
                }
            }
            .padding(.horizontal, 12)

            Divider()
                .padding(.vertical, 8)

            // Credits - more prominent with better spacing
            VStack(spacing: 8) {
                Text("Powered by")
                    .font(.caption)
                    .foregroundColor(.secondary)

                HStack(spacing: 16) {
                    VStack(spacing: 2) {
                        Text("NVIDIA")
                            .font(.caption)
                            .fontWeight(.semibold)
                        Text("Parakeet & Nemotron")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }

                    Text("&")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    VStack(spacing: 2) {
                        Text("OpenAI")
                            .font(.caption)
                            .fontWeight(.semibold)
                        Text("Whisper")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                }
            }
            .padding(.bottom, 12)
        }
        .padding()
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

// MARK: - About Icon Loading Helper

/// Attempts to load the app icon from the bundle resources for About view
private func loadAboutIconFromBundle() -> NSImage? {
    // List of possible icon filenames to try
    let iconNames = [
        "Icon-iOS-Default-256x256@1x",
        "Icon-iOS-Default-128x128@1x",
        "Icon-iOS-Default-512x512@1x",
        "AppIcon",
        "Icon"
    ]
    
    for iconName in iconNames {
        // Try loading as PNG
        if let image = NSImage(named: iconName) {
            return image
        }
        
        // Try loading from bundle resources
        if let imagePath = Bundle.main.path(forResource: iconName, ofType: "png"),
           let image = NSImage(contentsOfFile: imagePath) {
            return image
        }
    }
    
    return nil
}

// =============================================================================
// AUDIO DUCKING SETTINGS (Isolated Component)
// =============================================================================

/// Separate component to isolate audio ducking settings and prevent publishing warnings
struct AudioDuckingSettings: View {
    @EnvironmentObject var appState: AppState
    
    // Local state to buffer changes and prevent direct binding issues
    @State private var localMuteEnabled: Bool = false
    @State private var localMuteMode: String = "mute"
    @State private var localDuckVolume: Double = 10.0
    
    var body: some View {
        Group {
            Toggle("Mute playback during recording", isOn: $localMuteEnabled)
                .onChange(of: localMuteEnabled) { _, newValue in
                    // Use animation transaction to absorb the change
                    withAnimation(.easeInOut(duration: 0.2)) {
                        appState.mutePlaybackDuringRecording = newValue
                    }
                }

            if localMuteEnabled {
                VStack(alignment: .leading, spacing: 8) {
                    Picker("Mode", selection: $localMuteMode) {
                        Text("Mute completely").tag("mute")
                        Text("Lower volume").tag("lower")
                    }
                    .pickerStyle(.segmented)
                    .onChange(of: localMuteMode) { _, newValue in
                        // Defer the change to break the publish cycle
                        DispatchQueue.main.async {
                            appState.muteMode = newValue
                        }
                    }

                    if localMuteMode == "lower" {
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Text("Lower by:")
                                    .font(.caption)
                                Text("\(Int(localDuckVolume))%")
                                    .font(.caption)
                                    .fontWeight(.medium)
                                    .monospacedDigit()
                            }

                            Slider(value: $localDuckVolume, in: 10...99, step: 1)
                                .help("How much to reduce volume during recording")
                                .onChange(of: localDuckVolume) { _, newValue in
                                    // Debounce slider changes
                                    DispatchQueue.main.async {
                                        appState.duckVolume = newValue
                                    }
                                }

                            Text("Volume will be reduced by \(Int(localDuckVolume))% while recording. For example, if your volume is at 50%, it will drop to \(Int(Float(50) * (1.0 - Float(localDuckVolume)/100)))%.")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                        }
                        .transition(.opacity)
                    } else {
                        Text("All system audio will be muted during recording")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                            .transition(.opacity)
                    }
                }
                .transition(.opacity)
            }
        }
        .onAppear {
            // Initialize local state from appState
            localMuteEnabled = appState.mutePlaybackDuringRecording
            localMuteMode = appState.muteMode
            localDuckVolume = appState.duckVolume
        }
    }
}


// =============================================================================
// ANIMATED DOWNLOADING BAR
// =============================================================================

/// Animated indeterminate progress bar that moves left to right (shimmer effect)
struct AnimatedDownloadingBar: View {
    @State private var animationOffset: CGFloat = 0
    
    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                // Background track
                RoundedRectangle(cornerRadius: 3)
                    .fill(Color.secondary.opacity(0.2))
                
                // Animated shimmer/wave that moves left to right
                RoundedRectangle(cornerRadius: 3)
                    .fill(
                        LinearGradient(
                            colors: [
                                Color.accentColor.opacity(0.3),
                                Color.accentColor,
                                Color.accentColor.opacity(0.3)
                            ],
                            startPoint: .leading,
                            endPoint: .trailing
                        )
                    )
                    .frame(width: geometry.size.width * 0.4) // Wave is 40% of bar width
                    .offset(x: -geometry.size.width * 0.4 + (geometry.size.width * 1.4 * animationOffset))
            }
            .clipShape(RoundedRectangle(cornerRadius: 3))
            .onAppear {
                // Reset to start position
                animationOffset = 0
                // Start continuous animation
                withAnimation(
                    .linear(duration: 1.5)
                    .repeatForever(autoreverses: false)
                ) {
                    animationOffset = 1.0
                }
            }
        }
    }
}


// =============================================================================
// PREVIEW
// =============================================================================

#Preview {
    SettingsView()
        .environmentObject(AppState())
        .environmentObject(BackendManager())
}

