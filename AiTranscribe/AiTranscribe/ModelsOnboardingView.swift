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
            // Whisper Base (English)
            makeModel(
                id: "whisper-base-en",
                fallbackDisplay: "Whisper Base (English)",
                fallbackDescription: "Fast, lightweight English transcription. Great for quick dictation.",
                fallbackSizeGB: 0.142,  // 142 MB
                fallbackRamGB: 0.150,   // 150 MB
                tags: ["Compact", "English"]
            ),
            // Whisper Large v3 Turbo (multilingual)
            makeModel(
                id: "whisper-large-v3-turbo",
                fallbackDisplay: "Whisper Large v3 Turbo",
                fallbackDescription: "Balanced • Multilingual",
                fallbackSizeGB: 1.5,   // ~1500 MB
                fallbackRamGB: 0.65,   // ~650 MB
                tags: ["Balanced", "Multilingual"]
            )
        ]
    }
    
    var body: some View {
        VStack(spacing: 24) {
            // Header
            VStack(spacing: 8) {
                Text("Download Models")
                    .font(.system(size: 32, weight: .bold))
                
                Text("Click a model to download and get started")
                    .font(.title3)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
                
                // Warning about large models
                HStack(spacing: 6) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundColor(.orange)
                    Text("Larger models take longer to download")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding(.top, 4)
            }
            .padding(.top, 30)
            
            // Error alert
            .alert("Download Failed", isPresented: $showError) {
                Button("OK", role: .cancel) {}
            } message: {
                Text(downloadError ?? "An unknown error occurred")
            }
            
            // Model List - SCROLLABLE to prevent pushing buttons
            ScrollView {
                VStack(spacing: 16) {
                    ForEach(recommendedModels) { model in
                        OnboardingModelCard(
                            model: model,
                            isDownloading: downloadingModels.contains(model.id),
                            isLoading: loadingModels.contains(model.id),
                            isDownloaded: downloadedModels.contains(model.id) || isModelAlreadyDownloaded(model.id),
                            nemoAvailable: effectiveNemoAvailable,
                            onDownload: {
                                // Check if NeMo is required but not available
                                if model.requiresNemo && !effectiveNemoAvailable {
                                    // Store the pending model and show NeMo setup
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
                .padding(.horizontal, 40)
            }

            // Hint about more models
            HStack(spacing: 6) {
                Image(systemName: "info.circle")
                    .foregroundColor(.secondary)
                Text("More models available in Settings → Models")
                    .font(.footnote)
                    .foregroundColor(.secondary)
            }

            Spacer()
            
            // Action Buttons
            VStack(spacing: 12) {
                if isDownloading {
                    // Downloading state
                    HStack {
                        ProgressView()
                            .controlSize(.small)
                        Text("Downloading...")
                            .font(.title3.weight(.semibold))
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
                    .background(Color.accentColor.opacity(0.1))
                    .cornerRadius(10)
                    .padding(.horizontal, 60)
                } else if downloadedModels.count > 0 {
                    // At least one model downloaded - can continue
                    Button(action: onNext) {
                        HStack {
                            Text("Continue")
                                .font(.title3.weight(.semibold))
                            Image(systemName: "arrow.right")
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 14)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .padding(.horizontal, 60)
                } else {
                    // No models downloaded yet - show skip
                    Button(action: onNext) {
                        HStack {
                            Text("Skip - Download Later")
                                .font(.title3.weight(.semibold))
                            Image(systemName: "arrow.right")
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 14)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.large)
                    .padding(.horizontal, 60)
                }
                
                // Back button
                Button(action: onBack) {
                    HStack {
                        Image(systemName: "chevron.left")
                        Text("Back")
                    }
                    .font(.body.weight(.medium))
                }
                .buttonStyle(.bordered)
                .controlSize(.large)
                .disabled(isDownloading)
                .padding(.bottom, 20)
            }
        }
        .padding()
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
            HStack(spacing: 16) {
                // Status indicator
                ZStack {
                    Circle()
                        .stroke(statusColor, lineWidth: 2)
                        .frame(width: 24, height: 24)

                    if isDownloading {
                        ProgressView()
                            .scaleEffect(0.5)
                            .frame(width: 24, height: 24)
                    } else if isLoading {
                        ProgressView()
                            .scaleEffect(0.5)
                            .frame(width: 24, height: 24)
                            .foregroundColor(.orange)
                    } else if isDownloaded {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.system(size: 20))
                            .foregroundColor(.green)
                    } else if model.requiresNemo && !nemoAvailable {
                        Image(systemName: "cpu")
                            .font(.system(size: 16))
                            .foregroundColor(.orange)
                    } else {
                        Image(systemName: "arrow.down.circle")
                            .font(.system(size: 18))
                            .foregroundColor(.accentColor)
                    }
                }

                // Model Info
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text(model.displayName)
                            .font(.headline)
                            .foregroundColor(.primary)

                        Spacer()

                        // Status badge
                        if isDownloaded {
                            HStack(spacing: 4) {
                                Image(systemName: "checkmark.circle.fill")
                                    .foregroundColor(.green)
                                Text("Ready")
                                    .font(.caption.weight(.medium))
                                    .foregroundColor(.green)
                            }
                        } else if isLoading {
                            HStack(spacing: 4) {
                                Text("Loading...")
                                    .font(.caption.weight(.medium))
                                    .foregroundColor(.orange)
                            }
                        } else if isDownloading {
                            HStack(spacing: 4) {
                                Text("Downloading...")
                                    .font(.caption.weight(.medium))
                                    .foregroundColor(.accentColor)
                            }
                        } else if model.requiresNemo && !nemoAvailable {
                            // Show NeMo required badge
                            HStack(spacing: 4) {
                                Image(systemName: "cpu")
                                    .foregroundColor(.orange)
                                Text("Requires NeMo")
                                    .font(.caption.weight(.medium))
                                    .foregroundColor(.orange)
                            }
                        } else {
                            // Show "Click to download" hint
                            Text("Click to download")
                                .font(.caption.weight(.medium))
                                .foregroundColor(.accentColor)
                        }
                    }

                    Text(model.description)
                        .font(.subheadline)
                        .foregroundColor(.secondary)

                    // NeMo info for Parakeet models
                    if model.requiresNemo && !nemoAvailable {
                        HStack(spacing: 6) {
                            Image(systemName: "info.circle.fill")
                                .foregroundColor(.orange)
                            Text("Requires NeMo framework (~3GB install)")
                                .font(.caption)
                                .foregroundColor(.orange)
                        }
                        .padding(.vertical, 4)
                    }

                    HStack(spacing: 12) {
                        Label("\(model.downloadSizeGB, specifier: "%.1f") GB", systemImage: "arrow.down.circle")
                        Label("\(model.ramSizeGB, specifier: "%.1f") GB RAM", systemImage: "memorychip")
                    }
                    .font(.caption)
                    .foregroundColor(.secondary)

                    // Tags
                    HStack(spacing: 6) {
                        ForEach(model.tags, id: \.self) { tag in
                            Text(tag)
                                .font(.caption2)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 3)
                                .background(Color.accentColor.opacity(0.1))
                                .foregroundColor(.accentColor)
                                .cornerRadius(6)
                        }

                        // Add NeMo tag for Parakeet models
                        if model.requiresNemo {
                            Text("NeMo")
                                .font(.caption2)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 3)
                                .background(Color.purple.opacity(0.1))
                                .foregroundColor(.purple)
                                .cornerRadius(6)
                        }
                    }
                    .padding(.top, 4)
                }

                Spacer()
            }
            .padding(16)
            .frame(maxWidth: .infinity, alignment: .leading)
            .contentShape(Rectangle())
            .background(backgroundColor)
            .cornerRadius(12)
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(borderColor, lineWidth: borderWidth)
            )
        }
        .buttonStyle(.plain)
        .disabled(isDownloading || isDownloaded || isLoading)
        .help(helpText)
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

    private var backgroundColor: Color {
        if isDownloaded {
            return Color.green.opacity(0.05)
        } else if isDownloading || isLoading {
            return Color.accentColor.opacity(0.05)
        } else if model.requiresNemo && !nemoAvailable {
            return Color.orange.opacity(0.03)
        } else {
            return Color.clear
        }
    }

    private var borderColor: Color {
        if isDownloaded {
            return .green.opacity(0.3)
        } else if isDownloading || isLoading {
            return .accentColor.opacity(0.3)
        } else if model.requiresNemo && !nemoAvailable {
            return .orange.opacity(0.3)
        } else {
            return .gray.opacity(0.2)
        }
    }

    private var borderWidth: CGFloat {
        (isDownloaded || isDownloading || isLoading || (model.requiresNemo && !nemoAvailable)) ? 2 : 1
    }
}

// MARK: - Preview

#Preview {
    ModelsOnboardingView(onNext: {}, onBack: {})
        .environmentObject(AppState())
        .environmentObject(BackendManager())
        .frame(width: 800, height: 750)
}

