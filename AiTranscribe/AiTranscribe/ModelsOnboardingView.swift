/*
 ModelsOnboardingView.swift
 ==========================

 Model download screen (OPTIONAL).
 
 Features:
 - Show 2 recommended models (Parakeet TDT v2 + Whisper)
 - Display download size and RAM usage
 - Allow download or skip
 - Hint about more models in Settings
 */

import SwiftUI

struct ModelsOnboardingView: View {
    let onNext: () -> Void
    let onBack: () -> Void

    @EnvironmentObject var appState: AppState
    @EnvironmentObject var backendManager: BackendManager

    // State for tracking downloads
    @State private var downloadingModels: Set<String> = []
    @State private var loadingModels: Set<String> = []
    @State private var downloadedModels: Set<String> = []
    @State private var isDownloading = false
    @State private var downloadError: String?
    @State private var showError = false

    // State for NeMo setup
    @State private var showingNemoSetup = false
    @State private var pendingNemoModelId: String? = nil

    /// Computed property for effective NeMo availability
    /// - In development mode: trust backend's nemoAvailable (devs may have NeMo in main env)
    /// - In production mode: require user to install via UI (nemoVenvExists)
    private var effectiveNemoAvailable: Bool {
        appState.nemoVenvExists || (appState.backendMode == "development" && appState.nemoAvailable)
    }

    // Predefined recommended models (built from backend data when available)
    private var recommendedModels: [RecommendedModel] {
        func makeModel(id: String, fallbackDisplay: String, fallbackDescription: String, fallbackSizeGB: Double, fallbackRamGB: Double, tags: [String], requiresNemo: Bool = false) -> RecommendedModel {
            if let info = appState.availableModels.first(where: { $0.id == id }) {
                return RecommendedModel(
                    id: info.id,
                    displayName: info.displayName,
                    description: info.description,
                    downloadSizeGB: info.sizeGB,
                    ramSizeGB: info.ramGB,
                    tags: tags,
                    requiresNemo: info.nemoRequired
                )
            } else {
                return RecommendedModel(
                    id: id,
                    displayName: fallbackDisplay,
                    description: fallbackDescription,
                    downloadSizeGB: fallbackSizeGB,
                    ramSizeGB: fallbackRamGB,
                    tags: tags,
                    requiresNemo: requiresNemo
                )
            }
        }

        return [
            // NVIDIA Parakeet v2 (English, fast) - Requires NeMo
            makeModel(
                id: "parakeet-v2",
                fallbackDisplay: "Parakeet TDT v2",
                fallbackDescription: "Fast • English • Recommended",
                fallbackSizeGB: 2.0,   // ~2000 MB
                fallbackRamGB: 3.0,    // ~3000 MB
                tags: ["Fast", "English", "Recommended"],
                requiresNemo: true
            ),
            // Whisper Base (English) — whisper.cpp GGML
            makeModel(
                id: "whisper-base-en",
                fallbackDisplay: "Whisper Base (English)",
                fallbackDescription: "Fast English transcription with Metal GPU acceleration.",
                fallbackSizeGB: 0.148,  // 148 MB
                fallbackRamGB: 0.400,   // 400 MB
                tags: ["Compact", "English", "Metal GPU"]
            ),
            // Whisper Large v3 Turbo (multilingual) — whisper.cpp GGML
            makeModel(
                id: "whisper-large-v3-turbo",
                fallbackDisplay: "Whisper Large v3 Turbo",
                fallbackDescription: "Balanced multilingual transcription with Metal GPU acceleration.",
                fallbackSizeGB: 1.62,   // ~1620 MB
                fallbackRamGB: 1.7,     // ~1700 MB
                tags: ["Balanced", "Multilingual", "Metal GPU"]
            )
        ]
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            OnboardingHeader(
                eyebrow: "Local models",
                title: "Choose what runs on this Mac",
                subtitle: "Start with one private model. You can add the larger catalog later from Settings."
            )
            .padding(.bottom, 28)
            
            // Error alert
            .alert("Download Failed", isPresented: $showError) {
                Button("OK", role: .cancel) {}
            } message: {
                Text(downloadError ?? "An unknown error occurred")
            }
            
            // Backend status banner
            if !backendManager.isServerReady {
                HStack(spacing: 8) {
                    ProgressView()
                        .controlSize(.small)
                    Text("Backend is starting; downloads unlock shortly.")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(.secondary)
                    Spacer()
                }
                .padding(.bottom, 16)
            }

            VStack(spacing: 0) {
                ForEach(recommendedModels) { model in
                    OnboardingModelCard(
                        model: model,
                        isDownloading: downloadingModels.contains(model.id),
                        isLoading: loadingModels.contains(model.id),
                        isDownloaded: downloadedModels.contains(model.id) || isModelAlreadyDownloaded(model.id),
                        nemoAvailable: effectiveNemoAvailable,
                        backendReady: backendManager.isServerReady,
                        onDownload: {
                            if model.requiresNemo && !effectiveNemoAvailable {
                                pendingNemoModelId = model.id
                                showingNemoSetup = true
                            } else {
                                Task {
                                    await downloadModel(model.id)
                                }
                            }
                        },
                        onInstallNemo: {
                            pendingNemoModelId = model.id
                            showingNemoSetup = true
                        }
                    )
                }
            }
            .padding(.top, 2)

            localModelFootnote
                .padding(.top, 20)

            Spacer()
            
            // Action Buttons
            HStack {
                OnboardingSecondaryActionButton(
                    title: "Back",
                    systemImage: "arrow.left",
                    action: onBack
                )
                .disabled(isDownloading)

                Spacer()

                if isDownloading {
                    OnboardingPrimaryActionButton(
                        title: "Downloading...",
                        systemImage: "arrow.down",
                        isLoading: true,
                        action: {}
                    )
                } else if readyLocalModelCount > 0 {
                    OnboardingPrimaryActionButton(
                        title: "Continue",
                        systemImage: "arrow.right",
                        action: onNext
                    )
                } else {
                    OnboardingPrimaryActionButton(
                        title: "Skip Local",
                        systemImage: "arrow.right",
                        action: onNext
                    )
                }
            }
        }
        .padding(34)
        .onAppear {
            Task {
                // Fetch models - server is already ready from first screen
                if appState.availableModels.isEmpty {
                    await appState.fetchAvailableModels()
                }
                await appState.checkNemoStatus()
                checkAlreadyDownloadedModels()
            }
        }
        .sheet(isPresented: $showingNemoSetup) {
            NemoSetupView(
                setupManager: appState.nemoSetupManager,
                onComplete: {
                    // After NeMo is installed, just refresh the UI
                    // User will click download manually
                    Task {
                        await appState.checkNemoStatus()
                        await appState.fetchAvailableModels()
                        checkAlreadyDownloadedModels()
                    }
                    pendingNemoModelId = nil
                }
            )
            .environmentObject(appState)
            .environmentObject(backendManager)
        }
    }

    private var localModelFootnote: some View {
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            Text("\(readyLocalModelCount)")
                .font(.system(size: 24, weight: .bold, design: .rounded))
                .foregroundStyle(.primary)

            Text("local model\(readyLocalModelCount == 1 ? "" : "s") ready now. More speech-to-text and local LLM models live in Settings > Models.")
                .font(.system(size: 13, weight: .medium))
                .lineSpacing(2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var readyLocalModelCount: Int {
        downloadedModels.union(Set(appState.availableModels.filter(\.downloaded).map(\.id))).count
    }
    
    // MARK: - Helpers
    
    private func isModelAlreadyDownloaded(_ modelId: String) -> Bool {
        appState.availableModels.first(where: { $0.id == modelId })?.downloaded ?? false
    }
    
    private func checkAlreadyDownloadedModels() {
        // Check which models are already downloaded
        for model in recommendedModels {
            if isModelAlreadyDownloaded(model.id) {
                downloadedModels.insert(model.id)
            }
        }
    }
    
    // MARK: - Model Download (Simplified - Click to Download)
    
    /// Download a single model when user clicks it
    private func downloadModel(_ modelId: String) async {
        // Skip if already downloaded
        guard !isModelAlreadyDownloaded(modelId) else {
            downloadedModels.insert(modelId)
            return
        }
        
        isDownloading = true
        downloadingModels.insert(modelId)
        
        // Download the model
        let success = await appState.downloadModel(modelId: modelId)
        
        downloadingModels.remove(modelId)
        
        if success {
            downloadedModels.insert(modelId)
            // Refresh state from backend
            await appState.fetchAvailableModels()
            
            // Auto-load the model after download
            loadingModels.insert(modelId)
            await appState.loadModel(modelId: modelId)
            loadingModels.remove(modelId)
        } else {
            // Download failed - show alert
            downloadError = appState.statusMessage
            showError = true
        }
        
        isDownloading = false
    }
}

// MARK: - Recommended Model

struct RecommendedModel: Identifiable {
    let id: String
    let displayName: String
    let description: String
    let downloadSizeGB: Double
    let ramSizeGB: Double
    let tags: [String]
    let requiresNemo: Bool

    init(id: String, displayName: String, description: String, downloadSizeGB: Double, ramSizeGB: Double, tags: [String], requiresNemo: Bool = false) {
        self.id = id
        self.displayName = displayName
        self.description = description
        self.downloadSizeGB = downloadSizeGB
        self.ramSizeGB = ramSizeGB
        self.tags = tags
        self.requiresNemo = requiresNemo
    }
}

// MARK: - Animated Download Dots

/// Animated dots to show download progress (since we can't show actual progress)
struct AnimatedDownloadDots: View {
    @State private var animationPhase: Int = 0
    
    var body: some View {
        HStack(spacing: 6) {
            ForEach(0..<3, id: \.self) { index in
                Circle()
                    .fill(Color.accentColor)
                    .frame(width: 8, height: 8)
                    .scaleEffect(animationPhase == index ? 1.3 : 0.8)
                    .opacity(animationPhase == index ? 1.0 : 0.4)
                    .animation(.easeInOut(duration: 0.4), value: animationPhase)
            }
        }
        .onAppear {
            startAnimation()
        }
    }
    
    private func startAnimation() {
        Timer.scheduledTimer(withTimeInterval: 0.4, repeats: true) { _ in
            animationPhase = (animationPhase + 1) % 3
        }
    }
}

// MARK: - Onboarding Model Card

struct OnboardingModelCard: View {
    let model: RecommendedModel
    let isDownloading: Bool
    let isLoading: Bool
    let isDownloaded: Bool
    let nemoAvailable: Bool
    let backendReady: Bool
    let onDownload: () -> Void
    let onInstallNemo: () -> Void

    /// Whether this model can be used (has required dependencies)
    private var canUseModel: Bool {
        if model.requiresNemo {
            return nemoAvailable
        }
        return true
    }

    var body: some View {
        Button(action: {
            if !isDownloading && !isLoading && !isDownloaded {
                onDownload()
            }
        }) {
            VStack(spacing: 0) {
                HStack(alignment: .top, spacing: 18) {
                    statusGlyph
                        .padding(.top, 1)

                    VStack(alignment: .leading, spacing: 7) {
                        HStack(alignment: .firstTextBaseline, spacing: 10) {
                            Text(model.displayName)
                                .font(.system(size: 18, weight: .bold, design: .rounded))
                                .foregroundStyle(.primary)

                            Text(primaryTag)
                                .font(.system(size: 11, weight: .bold, design: .rounded))
                                .tracking(0.7)
                                .textCase(.uppercase)
                                .foregroundStyle(statusColor)
                        }

                        Text(model.description)
                            .font(.system(size: 13, weight: .medium))
                            .lineSpacing(2)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)

                        if model.requiresNemo && !nemoAvailable {
                            Label("Requires NeMo framework before download", systemImage: "cpu")
                                .font(.system(size: 12, weight: .semibold))
                                .foregroundStyle(.orange)
                        }
                    }

                    Spacer(minLength: 24)

                    VStack(alignment: .trailing, spacing: 8) {
                        Text(actionText)
                            .font(.system(size: 13, weight: .bold, design: .rounded))
                            .foregroundStyle(statusColor)

                        HStack(spacing: 12) {
                            Label("\(model.downloadSizeGB, specifier: "%.1f") GB", systemImage: "arrow.down.circle")
                            Label("\(model.ramSizeGB, specifier: "%.1f") GB RAM", systemImage: "memorychip")
                        }
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(.secondary)
                    }
                }

                Divider()
                    .overlay(.white.opacity(0.08))
                    .padding(.top, 18)
            }
            .padding(.vertical, 13)
            .frame(maxWidth: .infinity, alignment: .leading)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(isDownloading || isDownloaded || isLoading || !backendReady)
        .help(helpText)
    }

    @ViewBuilder
    private var statusGlyph: some View {
        ZStack {
            Circle()
                .fill(statusColor.opacity(0.14))
                .frame(width: 36, height: 36)

            if isDownloading || isLoading {
                ProgressView()
                    .scaleEffect(0.62)
                    .frame(width: 36, height: 36)
            } else {
                Image(systemName: statusIcon)
                    .font(.system(size: 15, weight: .bold))
                    .foregroundStyle(statusColor)
            }
        }
    }

    private var helpText: String {
        if isDownloaded {
            return "Model ready to use"
        } else if isDownloading {
            return "Downloading..."
        } else if model.requiresNemo && !nemoAvailable {
            return "Click to install NeMo and download this model"
        } else {
            return "Click to download this model"
        }
    }

    private var statusColor: Color {
        if isDownloaded {
            return .green
        } else if isDownloading || isLoading {
            return .accentColor
        } else if model.requiresNemo && !nemoAvailable {
            return .orange
        } else {
            return .gray
        }
    }

    private var statusIcon: String {
        if isDownloaded {
            return "checkmark"
        } else if model.requiresNemo && !nemoAvailable {
            return "cpu"
        } else if isDownloading || isLoading {
            return "arrow.triangle.2.circlepath"
        } else {
            return "arrow.down"
        }
    }

    private var primaryTag: String {
        if model.tags.contains("Recommended") {
            return "Recommended"
        }
        return model.tags.first ?? "Local"
    }

    private var actionText: String {
        if isDownloaded {
            return "Ready"
        } else if isLoading {
            return "Loading"
        } else if isDownloading {
            return "Downloading"
        } else if model.requiresNemo && !nemoAvailable {
            return "Install NeMo"
        } else {
            return "Download"
        }
    }
}

// MARK: - Preview

#Preview {
    ModelsOnboardingView(onNext: {}, onBack: {})
        .environmentObject(AppState())
        .environmentObject(BackendManager())
        .frame(width: 800, height: 750)
}
