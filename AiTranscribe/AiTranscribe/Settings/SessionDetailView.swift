import SwiftUI
import AppKit
import UniformTypeIdentifiers

// MARK: - Session Detail View

private enum SessionDetailTab: String, CaseIterable {
    case transcription = "Transcription"
    case summary = "Summary"
}

struct SessionDetailView: View {
    @EnvironmentObject var sessionManager: SessionManager
    @EnvironmentObject var appState: AppState
    @AppStorage("preferredSummaryLengthOption") private var preferredSummaryLengthOption = SummaryLengthOption.medium.rawValue
    @AppStorage("preferredSummaryCustomWords") private var preferredSummaryCustomWords = SummaryLengthOption.custom.defaultWordTarget
    @AppStorage("selectedTranscriptionProviderId") private var selectedTranscriptionProviderId = TranscriptionProviderDescriptor.localBackend.id
    @AppStorage("selectedOpenAITranscriptionModel") private var selectedOpenAITranscriptionModel = TranscriptionModelDescriptor.openAIModels[0].id
    @AppStorage("selectedOpenAILanguageModel") private var selectedOpenAILanguageModel = APILanguageModelStore.defaultModels[0].modelId
    @AppStorage("publicSessionTargetBatchCount") private var publicSessionTargetBatchCount = 0
    @AppStorage("publicSessionConcurrentUploads") private var publicSessionConcurrentUploads = 6
    let sessionId: UUID
    let onBack: () -> Void
    var onNavigateToModels: ((ModelMode) -> Void)? = nil

    @State private var selectedModelId: String = "parakeet-v2"
    @State private var ramBudgetGB: Double = 4.0
    @State private var showDeleteAudioAlert = false
    @State private var showDeleteAllAlert = false
    @State private var estimate: SessionEstimateResponse? = nil
    @State private var isEditingName = false
    @State private var editedName = ""
    @State private var isPreparingModel = false
    @State private var preparationMessage = ""
    @State private var selectedTab: SessionDetailTab = .transcription
    @State private var focusedSummaryPresetId: String = SessionSummaryPreset.general.rawValue
    @State private var selectedSummaryPresetIds: Set<String> = [SessionSummaryPreset.general.rawValue]
    @State private var selectedSummaryModelId: String = "gemma-4-e2b-it-4bit"
    @State private var showPresetManager = false
    @State private var isEditingSummary = false
    @State private var editableSummaryText = ""
    @State private var openedSummaryPresetId: String? = nil
    @State private var apiTranscriptionModels: [APITranscriptionModel] = APITranscriptionModelStore.load()
    @State private var apiLanguageModels: [APILanguageModel] = APILanguageModelStore.load()

    private var session: Session? {
        sessionManager.sessions.first(where: { $0.id == sessionId })
    }

    private var isTranscribing: Bool {
        sessionManager.currentTranscribingSessionId == sessionId
    }

    private var isSummarizing: Bool {
        sessionManager.currentSummarizingSessionId == sessionId
    }

    private var maxRamBudget: Double {
        let totalRAM = Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824
        return max(minRamBudget + 0.5, totalRAM - 4)
    }

    private var minRamBudget: Double {
        let modelRamGB = Double(selectedModel?.ramMB ?? 3000) / 1024.0
        return ceil(modelRamGB + 1.0)
    }

    private var selectedModel: ModelInfoResponse? {
        availableModels.first(where: { $0.id == selectedModelId })
    }

    private var availableModels: [ModelInfoResponse] {
        appState.availableModels.filter { $0.downloaded == true && $0.sessionCompatible }
    }

    private var availableSummaryModels: [SummaryModelInfoResponse] {
        appState.availableSummaryModels
    }

    private var selectedSummaryModel: SummaryModelInfoResponse? {
        availableSummaryModels.first(where: { $0.id == selectedSummaryModelId })
    }

    private var isPublicMode: Bool {
        selectedTranscriptionProviderId != TranscriptionProviderDescriptor.localBackend.id
    }

    private var publicTranscriptionModels: [APITranscriptionModel] {
        apiTranscriptionModels.filter {
            APITranscriptionModelStore.isSupportedModel($0.modelId, providerId: $0.providerId)
        }
    }

    private var selectedAPITranscriptionModel: APITranscriptionModel? {
        publicTranscriptionModels.first {
            $0.providerId == selectedTranscriptionProviderId &&
            $0.modelId == selectedOpenAITranscriptionModel
        }
        ?? publicTranscriptionModels.first { $0.providerId == selectedTranscriptionProviderId }
        ?? publicTranscriptionModels.first
    }

    private var selectedAPILanguageModel: APILanguageModel? {
        apiLanguageModels.first(where: { $0.modelId == selectedOpenAILanguageModel })
            ?? apiLanguageModels.first
    }

    private var currentLengthOption: SummaryLengthOption {
        get { SummaryLengthOption(rawValue: preferredSummaryLengthOption) ?? .medium }
        nonmutating set { preferredSummaryLengthOption = newValue.rawValue }
    }

    private var summaryPresetDefinitions: [SummaryPresetDefinition] {
        appState.summaryPresetDefinitions
    }

    private var visibleSummaryPresets: [SummaryPresetDefinition] {
        let knownIds = Set(summaryPresetDefinitions.map(\.id))
        var presets = summaryPresetDefinitions

        if let session {
            for (presetId, metadata) in session.summaries where !knownIds.contains(presetId) {
                presets.append(
                    SummaryPresetDefinition(
                        id: presetId,
                        displayName: metadata.presetDisplayName ?? presetId.replacingOccurrences(of: "_", with: " ").capitalized,
                        instructions: "",
                        storageKey: metadata.fileName
                            .replacingOccurrences(of: "summary-", with: "")
                            .replacingOccurrences(of: ".md", with: ""),
                        symbolName: "archivebox",
                        isBuiltIn: false
                    )
                )
            }
        }

        return presets.sorted { lhs, rhs in
            if lhs.isBuiltIn != rhs.isBuiltIn {
                return lhs.isBuiltIn && !rhs.isBuiltIn
            }
            return lhs.displayName.localizedCaseInsensitiveCompare(rhs.displayName) == .orderedAscending
        }
    }

    private var focusedSummaryPreset: SummaryPresetDefinition? {
        visibleSummaryPresets.first(where: { $0.id == focusedSummaryPresetId })
            ?? visibleSummaryPresets.first
    }

    private var focusedSummaryMetadata: SessionSummaryMetadata? {
        guard let focusedSummaryPreset else { return nil }
        return session?.summaries[focusedSummaryPreset.id]
    }

    private var targetSummaryWords: Int {
        currentLengthOption == .custom ? max(120, preferredSummaryCustomWords) : currentLengthOption.defaultWordTarget
    }

    private var targetSummaryTokens: Int {
        max(256, Int(ceil(Double(targetSummaryWords) * 1.45)))
    }

    private var selectedSummaryMetadata: SessionSummaryMetadata? {
        focusedSummaryMetadata
    }

    private var selectedPresetRequests: [SummaryGenerationPresetRequest] {
        visibleSummaryPresets
            .filter {
                selectedSummaryPresetIds.contains($0.id)
                    && !$0.instructions.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            }
            .map { SummaryGenerationPresetRequest.make(from: $0, targetWords: targetSummaryWords) }
    }

    private var summaryEstimate: SummaryMemoryEstimateResponse? {
        guard let session,
              let model = selectedSummaryModel,
              let wordCount = session.wordCount ?? session.transcriptionText?.split(separator: " ").count else {
            return nil
        }

        let transcriptTokens = Int(ceil(Double(wordCount) * 1.35))
        let promptTokens = transcriptTokens + 512
        let requiredContextTokens = promptTokens + targetSummaryTokens
        let kvTokens = transcriptTokens + targetSummaryTokens
        let defaultKVMB = Int(ceil(Double(model.kvBytesPerTokenDefault * kvTokens) / (1024 * 1024)))
        let quantizedKVMB = Int(ceil(Double(model.kvBytesPerTokenQuantized * kvTokens) / (1024 * 1024)))
        let physicalRamMB = Int(ProcessInfo.processInfo.physicalMemory / 1_048_576)
        let memoryBudgetMB = max(0, physicalRamMB - 4096)
        let defaultTotal = model.residentModelRamMB + defaultKVMB
        let quantized = defaultTotal > memoryBudgetMB && memoryBudgetMB > 0

        return SummaryMemoryEstimateResponse(
            estimatedPromptTokens: promptTokens,
            estimatedTranscriptTokens: transcriptTokens,
            reservedOutputTokens: targetSummaryTokens,
            requiredContextTokens: requiredContextTokens,
            contextUtilization: min(1.0, Double(requiredContextTokens) / Double(model.contextTokens)),
            residentModelRamMB: model.residentModelRamMB,
            estimatedKVCacheRamMB: quantized ? quantizedKVMB : defaultKVMB,
            estimatedTotalRamMB: model.residentModelRamMB + (quantized ? quantizedKVMB : defaultKVMB),
            physicalRamMB: physicalRamMB,
            memoryBudgetMB: memoryBudgetMB,
            willQuantizeKV: quantized,
            fitsMemoryBudget: (model.residentModelRamMB + (quantized ? quantizedKVMB : defaultKVMB)) <= memoryBudgetMB || memoryBudgetMB == 0
        )
    }

    // MARK: - Body

    var body: some View {
        VStack(spacing: 0) {
            // Back button header
            HStack {
                Button {
                    withAnimation(.easeInOut(duration: 0.15)) { onBack() }
                } label: {
                    HStack(spacing: 5) {
                        Image(systemName: "chevron.left")
                            .font(.system(size: 10, weight: .semibold))
                        Text("Sessions")
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
            .padding(.horizontal, 28)
            .padding(.top, 12)
            .padding(.bottom, 8)

            if let session {
                if let openedId = openedSummaryPresetId,
                   let preset = visibleSummaryPresets.first(where: { $0.id == openedId }) {
                    // Full-height reader — no outer scroll
                    summaryReaderView(session: session, preset: preset)
                        .padding(.horizontal, 28)
                } else {
                    ScrollView {
                        VStack(alignment: .leading, spacing: 16) {
                            // Audio file card
                            audioFileCard(session)

                            // Tab switcher
                            tabSwitcher

                            // Content — identity transition prevents system controls
                            // (Picker, Stepper) from flashing black during tab switch
                            switch selectedTab {
                            case .transcription:
                                transcriptionSection(session)
                                    .transition(.identity)
                            case .summary:
                                summarySection(session)
                                    .transition(.identity)
                            }

                            // Delete actions only on transcription tab
                            if selectedTab == .transcription {
                                deleteActions(session)
                            }
                        }
                        .padding(.horizontal, 28)
                        .padding(.bottom, 20)
                    }
                    .scrollIndicators(.automatic)
                }
            } else {
                VStack(spacing: 8) {
                    Image(systemName: "questionmark.circle")
                        .font(.system(size: 24))
                        .foregroundStyle(.tertiary)
                    Text("Session not found")
                        .font(.system(size: 13))
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .task(id: sessionId) {
            // Transcript and summary markdown are no longer loaded with the
            // session list — pull them in (off-main, cached) when a session opens.
            sessionManager.ensureSessionContentLoaded(sessionId: sessionId)
        }
    }

    // MARK: - Tab Switcher

    private var tabSwitcher: some View {
        HStack(spacing: 3) {
            ForEach(SessionDetailTab.allCases, id: \.self) { tab in
                Button {
                    withAnimation(.easeInOut(duration: 0.2)) { selectedTab = tab }
                } label: {
                    HStack(spacing: 5) {
                        Image(systemName: tab == .transcription ? "text.alignleft" : "sparkles")
                            .font(.system(size: 10, weight: .medium))
                        Text(tab.rawValue)
                            .font(.system(size: 12, weight: .medium))
                    }
                    .foregroundStyle(selectedTab == tab ? .white : .secondary)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 7)
                    .background(
                        selectedTab == tab
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

    // MARK: - Audio File Card

    private func audioFileCard(_ session: Session) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 14) {
                // Icon
                ZStack {
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .fill(Color.accentColor.opacity(0.1))
                        .frame(width: 44, height: 44)
                    Image(systemName: session.hasAudio ? "waveform" : "waveform.slash")
                        .font(.system(size: 18))
                        .foregroundStyle(session.hasAudio ? Color.accentColor : .secondary)
                }

                VStack(alignment: .leading, spacing: 4) {
                    // Editable name
                    HStack(spacing: 6) {
                        if isEditingName {
                            TextField("Session name", text: $editedName)
                                .textFieldStyle(.plain)
                                .font(.system(size: 15, weight: .semibold))
                                .onSubmit {
                                    sessionManager.renameSession(id: session.id, newName: editedName)
                                    isEditingName = false
                                }
                                .onExitCommand { isEditingName = false }
                        } else {
                            Text(session.name)
                                .font(.system(size: 15, weight: .semibold))
                                .onTapGesture(count: 2) {
                                    editedName = session.name
                                    isEditingName = true
                                }
                        }

                        Button {
                            if isEditingName {
                                sessionManager.renameSession(id: session.id, newName: editedName)
                                isEditingName = false
                            } else {
                                editedName = session.name
                                isEditingName = true
                            }
                        } label: {
                            Image(systemName: isEditingName ? "checkmark.circle.fill" : "pencil")
                                .font(.system(size: 11))
                                .foregroundStyle(.secondary)
                        }
                        .buttonStyle(.plain)
                    }

                    // Metadata pills
                    HStack(spacing: 6) {
                        detailDatePill(session.createdAt)

                        if session.duration > 0 {
                            metaPill(icon: "clock", text: SessionManager.formatDuration(session.duration))
                        }
                        if session.fileSize > 0 {
                            metaPill(icon: "doc", text: SessionManager.formatFileSize(session.fileSize))
                        }
                    }
                }

                Spacer()
            }

            if session.hasAudio {
                HStack(spacing: 8) {
                    capsuleButton(label: "Open in Finder", icon: "folder", style: .regular) {
                        let audioURL = sessionManager.getAudioURL(for: session)
                        NSWorkspace.shared.activateFileViewerSelecting([audioURL])
                    }
                    capsuleButton(label: "Play", icon: "play.fill", style: .regular) {
                        let audioURL = sessionManager.getAudioURL(for: session)
                        NSWorkspace.shared.open(audioURL)
                    }
                }
            } else {
                HStack(spacing: 4) {
                    Image(systemName: "exclamationmark.triangle")
                        .font(.system(size: 9))
                        .foregroundStyle(.orange)
                    Text("Audio file deleted")
                        .font(.system(size: 10))
                        .foregroundStyle(.tertiary)
                }
            }
        }
        .padding(16)
        .background(.thinMaterial, in: .rect(cornerRadius: 14, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(.white.opacity(0.06), lineWidth: 0.5)
        )
    }

    // MARK: - Transcription Section

    @ViewBuilder
    private func transcriptionSection(_ session: Session) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            SettingsSectionHeader(title: "Transcription")

            if isTranscribing {
                transcribingView()
            } else if session.hasTranscription {
                if let text = session.transcriptionText {
                    completedView(session, text: text)
                } else {
                    // Transcript exists on disk but is still loading on demand.
                    transcriptLoadingView()
                }
            } else {
                notTranscribedView(session)
            }
        }
    }

    // MARK: State A — Not Transcribed

    @ViewBuilder
    private func notTranscribedView(_ session: Session) -> some View {
        if !session.hasAudio {
            HStack(spacing: 6) {
                Image(systemName: "exclamationmark.triangle")
                    .font(.system(size: 10))
                    .foregroundStyle(.orange)
                Text("Cannot transcribe — audio file has been deleted.")
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
            }
        } else {
            VStack(alignment: .leading, spacing: 12) {
                detailTranscriptionModePills

                if !isPublicMode && !appState.isServerConnected {
                    VStack(spacing: 10) {
                        ProgressView().scaleEffect(0.8)
                        Text("Server Starting...")
                            .font(.system(size: 13, weight: .semibold))
                        Text("Transcription will be available once the backend is ready.")
                            .font(.system(size: 11))
                            .foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(20)
                    .background(.thinMaterial, in: .rect(cornerRadius: 12, style: .continuous))
                } else {
                    notTranscribedControls(session)
                }
            }
        }
    }

    @ViewBuilder
    private func notTranscribedControls(_ session: Session) -> some View {
        if isPublicMode {
            publicNotTranscribedControls(session)
        } else {
            localNotTranscribedControls(session)
        }
    }

    private func publicNotTranscribedControls(_ session: Session) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            if publicTranscriptionModels.isEmpty {
                VStack(spacing: 8) {
                    Image(systemName: "network.badge.shield.half.filled")
                        .font(.system(size: 20))
                        .foregroundStyle(.tertiary)
                    Text("No public transcription models configured")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(.secondary)
                    if let onNavigateToModels {
                        capsuleButton(label: "Go to API Models", icon: "arrow.right", style: .prominent) {
                            onNavigateToModels(.speechToText)
                        }
                    }
                }
                .frame(maxWidth: .infinity)
                .padding(20)
                .background(.thinMaterial, in: .rect(cornerRadius: 12, style: .continuous))
            } else {
                glassCard {
                    VStack(alignment: .leading, spacing: 12) {
                        VStack(alignment: .leading, spacing: 6) {
                            detailControlLabel("PUBLIC MODEL")
                            detailPublicTranscriptionModelMenu
                        }

                        HStack(spacing: 5) {
                            Image(systemName: "info.circle")
                                .font(.system(size: 9))
                                .foregroundStyle(.orange)
                            Text("Public session transcription uploads API batches in parallel and stitches them back in order.")
                                .font(.system(size: 10))
                                .foregroundStyle(.secondary)
                        }

                        VStack(alignment: .leading, spacing: 8) {
                            HStack(alignment: .center, spacing: 12) {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text("API Batch Target")
                                        .font(.system(size: 11, weight: .semibold))
                                        .foregroundStyle(.secondary)
                                    Text(publicBatchEstimateText(for: session))
                                        .font(.system(size: 10))
                                        .foregroundStyle(.tertiary)
                                }

                                Spacer()

                                detailIntegerStepperPill(
                                    label: "Batches",
                                    value: manualPublicBatchBinding,
                                    in: 0...30,
                                    step: 6,
                                    zeroLabel: "Auto"
                                )
                            }

                            HStack(spacing: 8) {
                                metaPill(
                                    icon: "square.stack.3d.up",
                                    text: "\(effectivePublicBatchCount(for: session)) batches"
                                )
                                metaPill(
                                    icon: "clock",
                                    text: "~\(publicBatchMinutesText(for: session))/batch"
                                )
                                metaPill(
                                    icon: "arrow.triangle.branch",
                                    text: "parallel \(max(1, min(publicSessionConcurrentUploads, 6)))"
                                )
                            }

                            if publicSessionTargetBatchCount == 0 {
                                Text("Auto targets ~15 min per batch, then rounds up to a multiple of 6 so the parallel wave stays balanced.")
                                    .font(.system(size: 10))
                                    .foregroundStyle(.secondary)
                            }
                        }

                    }
                }

                Button {
                    if let selectedAPITranscriptionModel {
                        sessionManager.startPublicTranscription(
                            sessionId: session.id,
                            providerId: selectedAPITranscriptionModel.providerId,
                            modelId: selectedAPITranscriptionModel.modelId,
                            displayName: selectedAPITranscriptionModel.displayName,
                            targetBatchCount: publicSessionTargetBatchCount > 0 ? normalizedManualPublicBatchTarget : nil
                        )
                    }
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: "network")
                            .font(.system(size: 10))
                        Text("Start Public Transcription")
                            .font(.system(size: 11, weight: .medium))
                    }
                    .foregroundStyle(.white)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 8)
                    .background(Color.accentColor, in: Capsule())
                }
                .buttonStyle(.plain)
                .disabled(apiTranscriptionModels.isEmpty || !session.hasAudio)
            }
        }
        .task {
            apiTranscriptionModels = APITranscriptionModelStore.load()
            normalizePublicBatchTarget()
            if !publicTranscriptionModels.contains(where: {
                $0.providerId == selectedTranscriptionProviderId &&
                $0.modelId == selectedOpenAITranscriptionModel
            }),
               let firstModel = publicTranscriptionModels.first {
                selectedTranscriptionProviderId = firstModel.providerId
                selectedOpenAITranscriptionModel = firstModel.modelId
            }
        }
    }

    private var normalizedManualPublicBatchTarget: Int {
        guard publicSessionTargetBatchCount > 0 else { return 0 }
        let rounded = Int(round(Double(publicSessionTargetBatchCount) / 6.0)) * 6
        return max(6, min(rounded, 30))
    }

    private func normalizePublicBatchTarget() {
        guard publicSessionTargetBatchCount > 0 else { return }
        let normalized = normalizedManualPublicBatchTarget
        if publicSessionTargetBatchCount != normalized {
            publicSessionTargetBatchCount = normalized
        }
    }

    private func effectivePublicBatchCount(for session: Session) -> Int {
        if publicSessionTargetBatchCount > 0 {
            return normalizedManualPublicBatchTarget
        }

        let minutes = max(session.duration / 60.0, 0.1)
        let rawCount = Int(ceil(minutes / 15.0))
        let roundedToWave = Int(ceil(Double(rawCount) / 6.0)) * 6
        return max(6, min(roundedToWave, 30))
    }

    private func publicBatchMinutesText(for session: Session) -> String {
        let minutes = max(session.duration / 60.0, 0)
        let perBatch = minutes / Double(max(effectivePublicBatchCount(for: session), 1))
        if perBatch >= 10 {
            return String(format: "%.1f min", perBatch)
        }
        return String(format: "%.2f min", perBatch)
    }

    private func publicBatchEstimateText(for session: Session) -> String {
        let minutes = max(session.duration / 60.0, 0)
        let batchCount = effectivePublicBatchCount(for: session)
        if publicSessionTargetBatchCount == 0 {
            return String(format: "Auto: %.0f min session / %d wave-aligned batches = %@.", minutes, batchCount, publicBatchMinutesText(for: session))
        }
        return String(format: "%.0f min session / %d batches = %@.", minutes, batchCount, publicBatchMinutesText(for: session))
    }

    private func localNotTranscribedControls(_ session: Session) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            // Model selector
            if availableModels.isEmpty {
                VStack(spacing: 8) {
                    Image(systemName: "cube.box")
                        .font(.system(size: 20))
                        .foregroundStyle(.tertiary)
                    Text("No compatible models installed")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(.secondary)
                    if let onNavigateToModels {
                        capsuleButton(label: "Go to Models", icon: "arrow.right", style: .prominent) {
                            onNavigateToModels(.speechToText)
                        }
                    }
                }
                .frame(maxWidth: .infinity)
                .padding(20)
                .background(.thinMaterial, in: .rect(cornerRadius: 12, style: .continuous))
            } else {
                glassCard {
                    VStack(alignment: .leading, spacing: 12) {
                        VStack(alignment: .leading, spacing: 6) {
                            detailControlLabel("LOCAL MODEL")
                            detailLocalTranscriptionModelMenu
                        }

                        if selectedModelId == "whisper-large-v3" {
                            HStack(spacing: 4) {
                                Image(systemName: "info.circle")
                                    .font(.system(size: 9))
                                    .foregroundStyle(.orange)
                                Text("Slower but supports 99+ languages with highest accuracy.")
                                    .font(.system(size: 10))
                                    .foregroundStyle(.secondary)
                            }
                        }

                        // RAM budget
                        VStack(alignment: .leading, spacing: 8) {
                            HStack(alignment: .center, spacing: 12) {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text("RAM Budget")
                                        .font(.system(size: 11, weight: .semibold))
                                        .foregroundStyle(.secondary)
                                    Text("Controls local batching and memory headroom.")
                                        .font(.system(size: 10))
                                        .foregroundStyle(.tertiary)
                                }
                                Spacer()
                                detailDoubleStepperPill(
                                    label: "RAM",
                                    value: ramBudgetBinding(for: session),
                                    in: minRamBudget...maxRamBudget,
                                    step: 0.5,
                                    suffix: "GB"
                                )
                            }

                            HStack(spacing: 8) {
                                if let est = estimate {
                                    metaPill(icon: "square.stack.3d.up", text: "~\(est.numBatches) batches")
                                    metaPill(icon: "clock", text: "~\(formatEstimatedTime(est.estimatedTimeSeconds))")
                                }
                            }
                        }
                    }
                }

                // Start button
                Button {
                    Task { await startTranscriptionWithModelCheck(session) }
                } label: {
                    HStack(spacing: 6) {
                        if isPreparingModel {
                            ProgressView().scaleEffect(0.6)
                            Text(preparationMessage)
                                .font(.system(size: 11, weight: .medium))
                        } else {
                            Image(systemName: "waveform.badge.magnifyingglass")
                                .font(.system(size: 10))
                            Text("Start Transcription")
                                .font(.system(size: 11, weight: .medium))
                        }
                    }
                    .foregroundStyle(.white)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 8)
                    .background(Color.accentColor, in: Capsule())
                }
                .buttonStyle(.plain)
                .disabled(availableModels.isEmpty || !session.hasAudio || isPreparingModel || appState.isModelLoading)
            }
        }
        .task {
            if ramBudgetGB < minRamBudget { ramBudgetGB = minRamBudget }
            updateEstimate(session)
        }
        .onChange(of: selectedModelId) { _, _ in
            if ramBudgetGB < minRamBudget { ramBudgetGB = minRamBudget }
            updateEstimate(session)
        }
    }

    // MARK: State B — Transcribing

    private func transcribingView() -> some View {
        glassCard {
            VStack(alignment: .leading, spacing: 12) {
                if let progress = sessionManager.transcriptionProgress {
                    HStack {
                        Text("Batch \(progress.batch) / \(progress.totalBatches)")
                            .font(.system(size: 12, weight: .semibold, design: .monospaced))
                        Spacer()
                        Text("\(Int(progress.progress * 100))%")
                            .font(.system(size: 12, weight: .bold, design: .monospaced))
                    }

                    ProgressView(value: progress.progress)
                        .progressViewStyle(.linear)
                        .tint(.accentColor)

                    HStack(spacing: 10) {
                        if let cpu = progress.cpuPercent {
                            metaPill(icon: "cpu", text: "\(Int(cpu))%")
                        }
                        if let mem = progress.memoryMB {
                            metaPill(icon: "memorychip", text: "\(Int(mem)) MB")
                        }
                        if let eta = progress.etaSeconds, eta > 0 {
                            metaPill(icon: "clock", text: "~\(formatEstimatedTime(eta))")
                        }
                    }
                } else {
                    HStack(spacing: 8) {
                        ProgressView().scaleEffect(0.7)
                        Text("Preparing transcription...")
                            .font(.system(size: 11))
                            .foregroundStyle(.secondary)
                    }
                }

                capsuleButton(label: "Cancel", icon: "xmark", style: .destructive) {
                    sessionManager.cancelTranscription(apiClient: appState.apiClient)
                }
            }
        }
    }

    // MARK: State C — Completed

    /// Placeholder while the transcript text loads on demand — brief, and never
    /// shows the transcribe controls for a session that already has a transcript.
    private func transcriptLoadingView() -> some View {
        glassCard {
            HStack(spacing: 8) {
                ProgressView().scaleEffect(0.7)
                Text("Loading transcript...")
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
            }
        }
    }

    private func completedView(_ session: Session, text: String) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            // Action buttons
            HStack(spacing: 8) {
                capsuleButton(label: "Copy", icon: "doc.on.doc", style: .regular) {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(text, forType: .string)
                }
                capsuleButton(label: "Export TXT", icon: "square.and.arrow.up", style: .regular) {
                    exportTranscription(text, sessionName: session.name)
                }
            }

            // Stats
            HStack(spacing: 6) {
                if let wc = session.wordCount, wc > 0 {
                    metaPill(icon: "text.word.spacing", text: "\(wc) words")
                }
                if let time = session.transcriptionTime, time > 0 {
                    metaPill(icon: "clock", text: formatEstimatedTime(time))
                }
                if let model = session.modelUsed {
                    metaPill(icon: "cpu", text: model)
                }
            }

            // Text preview
            ScrollView {
                Text(text)
                    .font(.system(.body, design: .default))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(14)
            }
            .frame(maxHeight: 300)
            .background(.thinMaterial, in: .rect(cornerRadius: 12, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .strokeBorder(.white.opacity(0.06), lineWidth: 0.5)
            )

            if session.hasAudio {
                Button("Re-transcribe with different settings") {
                    selectedTab = .transcription
                    sessionManager.resetTranscription(id: session.id)
                }
                .font(.system(size: 11))
                .foregroundStyle(.secondary)
                .buttonStyle(.plain)
            }
        }
    }

    // MARK: - Summary

    @ViewBuilder
    private func summarySection(_ session: Session) -> some View {
        Group {
            if let openedId = openedSummaryPresetId,
               let preset = visibleSummaryPresets.first(where: { $0.id == openedId }) {
                summaryReaderView(session: session, preset: preset)
            } else {
                summaryOverviewSection(session)
            }
        }
        .task {
            apiLanguageModels = APILanguageModelStore.load()
            await appState.fetchSummaryRuntimeStatus()
            await appState.fetchSummaryModels()
            if let preferred = appState.preferredSummaryModelId {
                selectedSummaryModelId = preferred
            }
            normalizeSummaryPresetSelection()
        }
        .onChange(of: appState.summaryPresetDefinitions) { _, _ in
            normalizeSummaryPresetSelection()
        }
        .onChange(of: focusedSummaryPresetId) { _, _ in
            isEditingSummary = false
        }
        .sheet(isPresented: $showPresetManager, onDismiss: normalizeSummaryPresetSelection) {
            SummaryPresetManagerSheet()
                .environmentObject(appState)
        }
    }

    private func summaryOverviewSection(_ session: Session) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            SettingsSectionHeader(title: "Summary Studio")

            if !session.hasTranscription {
                HStack(spacing: 8) {
                    Image(systemName: "sparkles")
                        .font(.system(size: 12))
                        .foregroundStyle(.orange)
                    Text("Summary becomes available after transcription.")
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                }
                .padding(14)
                .background(.thinMaterial, in: .rect(cornerRadius: 12, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).strokeBorder(.orange.opacity(0.15), lineWidth: 0.5))
            } else if isPublicMode && apiLanguageModels.isEmpty {
                summaryEmptyAPIModelsView()
            } else if !isPublicMode && !appState.summaryRuntimeInstalled {
                summaryRuntimeInstallView()
            } else if !isPublicMode && availableSummaryModels.isEmpty {
                summaryEmptyModelsView()
            } else {
                summaryOverview(session)
            }
        }
    }

    private func summaryRuntimeInstallView() -> some View {
        glassCard {
            VStack(alignment: .leading, spacing: 10) {
                Text("Install summary runtime")
                    .font(.system(size: 13, weight: .semibold))
                Text("Required for local Gemma summaries.")
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)

                if appState.summarySetupManager.isInstalling {
                    ProgressView(value: appState.summarySetupManager.progress)
                        .tint(.orange)
                    Text(appState.summarySetupManager.statusMessage)
                        .font(.system(size: 10))
                        .foregroundStyle(.secondary)
                } else {
                    HStack(spacing: 8) {
                        capsuleButton(label: "Install Runtime", icon: "arrow.down.circle", style: .prominent, tint: .orange) {
                            Task {
                                do {
                                    try await appState.summarySetupManager.installRuntime()
                                    await appState.fetchSummaryRuntimeStatus()
                                } catch {
                                    appState.statusMessage = error.localizedDescription
                                }
                            }
                        }
                        if let onNavigateToModels {
                            capsuleButton(label: "Open Models", icon: "arrow.right", style: .regular) {
                                onNavigateToModels(.summarization)
                            }
                        }
                    }
                }
            }
        }
    }

    private func summaryEmptyModelsView() -> some View {
        glassCard {
            VStack(alignment: .leading, spacing: 10) {
                Text("No local LLM available")
                    .font(.system(size: 13, weight: .semibold))
                Text("Download a Gemma model to start summarizing.")
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
                if let onNavigateToModels {
                    capsuleButton(label: "Open Models", icon: "arrow.right", style: .regular) {
                        onNavigateToModels(.summarization)
                    }
                }
            }
        }
    }

    private func summaryEmptyAPIModelsView() -> some View {
        glassCard {
            VStack(alignment: .leading, spacing: 10) {
                Text("No public LLM configured")
                    .font(.system(size: 13, weight: .semibold))
                Text("Add an API LLM to generate summaries in Public mode.")
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
                if let onNavigateToModels {
                    capsuleButton(label: "Open API LLM", icon: "arrow.right", style: .regular) {
                        onNavigateToModels(.summarization)
                    }
                }
            }
        }
    }

    // MARK: - Summary Overview

    private func summaryOverview(_ session: Session) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            summaryGenerationCard(session)
            summaryPresetGrid(session)
        }
    }

    private func summaryPresetGrid(_ session: Session) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("PRESETS")
                    .font(.system(size: 10, weight: .semibold, design: .rounded))
                    .foregroundStyle(.tertiary)
                    .tracking(0.6)
                Spacer()
                capsuleButton(label: "Manage", icon: "slider.horizontal.3", style: .regular) {
                    showPresetManager = true
                }
            }

            VStack(spacing: 6) {
                ForEach(visibleSummaryPresets, id: \.id) { preset in
                    summaryPresetCard(preset, session: session)
                }
            }
        }
    }

    private func summaryPresetCard(_ preset: SummaryPresetDefinition, session: Session) -> some View {
        let metadata = session.summaries[preset.id]
        let hasSummary = metadata?.hasSummary == true
        let isGenerating = sessionManager.summaryProgress?.activePresetId == preset.id && isSummarizing
        let isSelected = selectedSummaryPresetIds.contains(preset.id)
        let isGenerateable = !preset.instructions.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty

        return HStack(spacing: 12) {
            // Checkbox for generation selection
            if isGenerateable {
                Button {
                    toggleSummaryPresetSelection(preset.id)
                } label: {
                    Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                        .font(.system(size: 14))
                        .foregroundStyle(isSelected ? Color.accentColor : Color.secondary.opacity(0.5))
                }
                .buttonStyle(.plain)
            }

            // Icon
            ZStack {
                Circle()
                    .fill((hasSummary ? Color.green : isGenerating ? .orange : .secondary).opacity(0.12))
                    .frame(width: 32, height: 32)
                Image(systemName: preset.symbolName)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(hasSummary ? .green : isGenerating ? .orange : .secondary)
            }

            // Name + status
            VStack(alignment: .leading, spacing: 2) {
                Text(preset.displayName)
                    .font(.system(size: 12, weight: .semibold))

                if isGenerating {
                    HStack(spacing: 4) {
                        ProgressView().scaleEffect(0.5)
                        Text("Generating...")
                            .font(.system(size: 10))
                            .foregroundStyle(.orange)
                    }
                } else if hasSummary {
                    HStack(spacing: 4) {
                        Circle().fill(.green).frame(width: 5, height: 5)
                        Text("Generated")
                            .font(.system(size: 10))
                            .foregroundStyle(.green)
                        if let words = metadata?.wordCount {
                            Text("· \(words) words")
                                .font(.system(size: 10))
                                .foregroundStyle(.tertiary)
                        }
                    }
                } else {
                    Text("Not yet generated")
                        .font(.system(size: 10))
                        .foregroundStyle(.tertiary)
                }
            }

            Spacer()

            // Open button
            if hasSummary || isGenerating {
                Button {
                    focusedSummaryPresetId = preset.id
                    withAnimation(.easeInOut(duration: 0.2)) {
                        openedSummaryPresetId = preset.id
                    }
                } label: {
                    HStack(spacing: 4) {
                        Text("Open")
                            .font(.system(size: 10, weight: .medium))
                        Image(systemName: "chevron.right")
                            .font(.system(size: 8, weight: .semibold))
                    }
                    .foregroundStyle(Color.accentColor)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 5)
                    .background(Color.accentColor.opacity(0.1), in: Capsule())
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.vertical, 10)
        .padding(.horizontal, 14)
        .background(.thinMaterial, in: .rect(cornerRadius: 12, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(
                    isSelected ? Color.accentColor.opacity(0.25) : .white.opacity(0.06),
                    lineWidth: isSelected ? 1 : 0.5
                )
        )
    }

    private func summaryGenerationCard(_ session: Session) -> some View {
        glassCard {
            VStack(alignment: .leading, spacing: 14) {
                // Generate button row
                summaryGenerateRow(session)

                Divider().opacity(0.3)

                VStack(alignment: .leading, spacing: 12) {
                    HStack(alignment: .top, spacing: 18) {
                        VStack(alignment: .leading, spacing: 6) {
                            detailControlLabel("SOURCE")
                            detailSummaryModePills
                        }

                        VStack(alignment: .leading, spacing: 6) {
                            detailControlLabel(isPublicMode ? "PUBLIC LLM" : "LOCAL LLM")
                            HStack(spacing: 8) {
                                if isPublicMode {
                                    detailPublicSummaryModelMenu
                                } else {
                                    detailLocalSummaryModelMenu

                                    if let model = selectedSummaryModel, !model.downloaded, let onNavigateToModels {
                                        capsuleButton(label: "Download", icon: "arrow.down.circle", style: .regular) {
                                            onNavigateToModels(.summarization)
                                        }
                                    }
                                }
                            }
                        }

                        Spacer(minLength: 0)
                    }

                    VStack(alignment: .leading, spacing: 6) {
                        detailControlLabel("LENGTH")

                        HStack(spacing: 10) {
                            detailSummaryLengthPills

                            if currentLengthOption == .custom {
                                detailIntegerStepperPill(
                                    label: "Words",
                                    value: summaryCustomWordsBinding,
                                    in: 180...2200,
                                    step: 80,
                                    zeroLabel: nil
                                )
                            }
                        }
                    }
                }
            }
        }
    }

    private func summaryGenerateRow(_ session: Session) -> some View {
        Group {
            if isSummarizing {
                HStack(spacing: 10) {
                    ProgressView().scaleEffect(0.7)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(summaryProgressTitle)
                            .font(.system(size: 11, weight: .semibold))
                        Text(sessionManager.summaryProgress?.stage ?? "Generating")
                            .font(.system(size: 10))
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    capsuleButton(label: "Cancel", icon: "xmark", style: .destructive) {
                        sessionManager.cancelSummary(apiClient: appState.apiClient)
                    }
                }
            } else {
                HStack(spacing: 12) {
                    // Big generate button
                    Button {
                        startSelectedSummaries(session)
                    } label: {
                        HStack(spacing: 6) {
                            Image(systemName: "sparkles")
                                .font(.system(size: 12, weight: .medium))
                            Text(generateSummaryButtonTitle(session: session))
                                .font(.system(size: 12, weight: .semibold))
                        }
                        .foregroundStyle(.white)
                        .padding(.horizontal, 16)
                        .padding(.vertical, 9)
                        .background(Color.accentColor, in: Capsule())
                    }
                    .buttonStyle(.plain)
                    .disabled(
                        selectedPresetRequests.isEmpty
                            || (isPublicMode
                                ? selectedAPILanguageModel == nil
                                : (selectedSummaryModel == nil || !(selectedSummaryModel?.downloaded ?? false)))
                    )
                    .opacity(
                        (selectedPresetRequests.isEmpty
                            || (isPublicMode
                                ? selectedAPILanguageModel == nil
                                : (selectedSummaryModel == nil || !(selectedSummaryModel?.downloaded ?? false)))) ? 0.5 : 1
                    )

                    HStack(spacing: 6) {
                        metaPill(icon: "text.word.spacing", text: "~\(targetSummaryWords) words")
                        if let model = selectedAPILanguageModel, isPublicMode {
                            metaPill(icon: "network", text: model.displayName)
                        } else if let model = selectedSummaryModel {
                            metaPill(icon: "memorychip", text: formatMB(model.residentModelRamMB))
                        }
                    }

                    Spacer()
                }
            }
        }
    }

    // MARK: - Summary Reader View

    private func summaryReaderView(session: Session, preset: SummaryPresetDefinition) -> some View {
        let metadata = session.summaries[preset.id]
        let text = previewText(for: preset.id)
        let isPresetGenerating = sessionManager.summaryProgress?.activePresetId == preset.id && isSummarizing

        return VStack(alignment: .leading, spacing: 0) {
            // Compact header row: back + title + meta + actions
            HStack(spacing: 8) {
                Button {
                    isEditingSummary = false
                    withAnimation(.easeInOut(duration: 0.2)) {
                        openedSummaryPresetId = nil
                    }
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: "chevron.left")
                            .font(.system(size: 9, weight: .semibold))
                        Text("Back")
                            .font(.system(size: 11, weight: .medium))
                    }
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 5)
                    .background(.quaternary, in: Capsule())
                }
                .buttonStyle(.plain)

                Image(systemName: preset.symbolName)
                    .font(.system(size: 11))
                    .foregroundStyle(Color.accentColor)
                Text(metadata?.presetDisplayName ?? preset.displayName)
                    .font(.system(size: 14, weight: .bold, design: .rounded))

                if let modelName = metadata?.modelName ?? metadata?.modelId {
                    metaPill(icon: "cpu", text: modelName)
                }
                if let words = metadata?.wordCount {
                    metaPill(icon: "text.word.spacing", text: "\(words) words")
                }

                Spacer()

                if let text, !text.isEmpty {
                    Button {
                        if isEditingSummary {
                            sessionManager.updateSummaryText(
                                sessionId: session.id,
                                presetId: preset.id,
                                text: editableSummaryText
                            )
                            isEditingSummary = false
                        } else {
                            editableSummaryText = text
                            isEditingSummary = true
                        }
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: isEditingSummary ? "eye" : "pencil")
                                .font(.system(size: 9, weight: .medium))
                            Text(isEditingSummary ? "Preview" : "Edit")
                                .font(.system(size: 10, weight: .medium))
                        }
                        .foregroundStyle(isEditingSummary ? Color.accentColor : .secondary)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 5)
                        .background(
                            isEditingSummary
                                ? AnyShapeStyle(Color.accentColor.opacity(0.1))
                                : AnyShapeStyle(.quaternary.opacity(0.5))
                        )
                        .clipShape(Capsule())
                    }
                    .buttonStyle(.plain)

                    capsuleButton(label: "Copy", icon: "doc.on.doc", style: .regular) {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(text, forType: .string)
                    }
                    capsuleButton(label: "Export", icon: "square.and.arrow.up", style: .regular) {
                        exportSummary(text, sessionName: session.name, presetId: preset.id)
                    }
                }
            }
            .padding(.bottom, 10)

            // Generating progress bar
            if isPresetGenerating {
                HStack(spacing: 8) {
                    ProgressView().scaleEffect(0.6)
                    Text(sessionManager.summaryProgress?.stage ?? "Generating...")
                        .font(.system(size: 10))
                        .foregroundStyle(.secondary)
                    Spacer()
                    capsuleButton(label: "Cancel", icon: "xmark", style: .destructive) {
                        sessionManager.cancelSummary(apiClient: appState.apiClient)
                    }
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(.thinMaterial, in: .rect(cornerRadius: 8, style: .continuous))
                .padding(.bottom, 8)
            }

            // Content — takes all remaining height
            if isPresetGenerating && !isEditingSummary {
                // Live stream: only this subview observes the high-frequency
                // partial text, so ticks don't re-render the rest of the app.
                SummaryStreamingText(stream: sessionManager.summaryStream, fallbackText: text)
            } else if let text, !text.isEmpty {
                if isEditingSummary {
                    TextEditor(text: $editableSummaryText)
                        .font(.system(.body, design: .monospaced))
                        .scrollContentBackground(.hidden)
                        .padding(14)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .background(.thinMaterial, in: .rect(cornerRadius: 14, style: .continuous))
                        .overlay(
                            RoundedRectangle(cornerRadius: 14, style: .continuous)
                                .strokeBorder(Color.accentColor.opacity(0.15), lineWidth: 0.5)
                        )
                } else {
                    ScrollView {
                        MarkdownView(text: text)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .textSelection(.enabled)
                            .padding(16)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(.thinMaterial, in: .rect(cornerRadius: 14, style: .continuous))
                    .overlay(
                        RoundedRectangle(cornerRadius: 14, style: .continuous)
                            .strokeBorder(.white.opacity(0.06), lineWidth: 0.5)
                    )
                }
            } else {
                summaryEmptyState(preset: preset)
                    .frame(maxHeight: .infinity)
            }
        }
        .padding(.bottom, 12)
    }

    private func summaryEmptyState(preset: SummaryPresetDefinition) -> some View {
        VStack(spacing: 8) {
            Image(systemName: "doc.text")
                .font(.system(size: 20))
                .foregroundStyle(.tertiary)
            Text("No saved output yet")
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(.secondary)
            Text(selectedSummaryPresetIds.contains(preset.id)
                    ? "Will appear here after the next summary run."
                    : "Select this preset to generate it.")
                .font(.system(size: 10))
                .foregroundStyle(.tertiary)
        }
        .frame(maxWidth: .infinity, minHeight: 180)
        .background(.thinMaterial, in: .rect(cornerRadius: 14, style: .continuous))
    }

    private var summaryProgressTitle: String {
        guard let progress = sessionManager.summaryProgress else { return "Generating summaries" }
        let activeName = visibleSummaryPresets.first(where: { $0.id == progress.activePresetId })?.displayName
            ?? session?.summaries[progress.activePresetId]?.presetDisplayName
            ?? "summary"
        return "\(progress.currentIndex)/\(max(progress.totalPresets, 1)) · \(activeName)"
    }

    private func generateSummaryButtonTitle(session: Session) -> String {
        let hasExisting = selectedPresetRequests.contains { request in
            session.summaries[request.presetId]?.hasSummary == true
        }
        if selectedPresetRequests.count > 1 {
            return hasExisting ? "Regenerate \(selectedPresetRequests.count) summaries" : "Generate \(selectedPresetRequests.count) summaries"
        }
        return hasExisting ? "Regenerate summary" : "Generate summary"
    }

    private func toggleSummaryPresetSelection(_ presetId: String) {
        if selectedSummaryPresetIds.contains(presetId) {
            selectedSummaryPresetIds.remove(presetId)
        } else {
            selectedSummaryPresetIds.insert(presetId)
        }
        if selectedSummaryPresetIds.isEmpty, let fallback = visibleSummaryPresets.first?.id {
            selectedSummaryPresetIds = [fallback]
        }
        if !selectedSummaryPresetIds.contains(focusedSummaryPresetId), let firstSelected = selectedSummaryPresetIds.first {
            focusedSummaryPresetId = firstSelected
        }
    }

    private func normalizeSummaryPresetSelection() {
        let validIds = Set(visibleSummaryPresets.map(\.id))
        selectedSummaryPresetIds = selectedSummaryPresetIds.intersection(validIds)
        if selectedSummaryPresetIds.isEmpty {
            if validIds.contains(SessionSummaryPreset.general.rawValue) {
                selectedSummaryPresetIds = [SessionSummaryPreset.general.rawValue]
            } else if let first = visibleSummaryPresets.first?.id {
                selectedSummaryPresetIds = [first]
            }
        }
        if !validIds.contains(focusedSummaryPresetId) {
            focusedSummaryPresetId = selectedSummaryPresetIds.first
                ?? visibleSummaryPresets.first?.id
                ?? SessionSummaryPreset.general.rawValue
        }
    }

    private func startSelectedSummaries(_ session: Session) {
        guard !selectedPresetRequests.isEmpty else { return }
        focusedSummaryPresetId = selectedPresetRequests.first?.presetId ?? focusedSummaryPresetId
        if isPublicMode {
            guard let model = selectedAPILanguageModel else { return }
            sessionManager.startPublicSummary(
                sessionId: session.id,
                providerId: model.providerId,
                modelId: model.modelId,
                modelName: model.displayName,
                presetRequests: selectedPresetRequests
            )
        } else {
            guard let modelId = selectedSummaryModel?.id else { return }
            sessionManager.startSummary(
                sessionId: session.id,
                modelId: modelId,
                presetRequests: selectedPresetRequests,
                apiClient: appState.apiClient
            )
        }
    }

    /// Saved (or cached) summary text. Live streaming text is rendered by
    /// SummaryStreamingText, which observes SummaryStreamModel directly.
    private func previewText(for presetId: String) -> String? {
        session?.summaries[presetId]?.text
    }

    // MARK: - Delete Actions (inline)

    private func deleteActions(_ session: Session) -> some View {
        HStack(spacing: 8) {
            if session.hasAudio {
                capsuleButton(label: "Delete Audio", icon: "speaker.slash", style: .destructive) {
                    showDeleteAudioAlert = true
                }
                .alert("Delete Audio?", isPresented: $showDeleteAudioAlert) {
                    Button("Cancel", role: .cancel) {}
                    Button("Delete Audio", role: .destructive) {
                        sessionManager.deleteSessionAudio(id: session.id)
                    }
                } message: {
                    Text("The audio file will be deleted. Transcription is kept.")
                }
            }

            capsuleButton(label: "Delete Everything", icon: "trash", style: .destructive) {
                showDeleteAllAlert = true
            }
            .alert("Delete Session?", isPresented: $showDeleteAllAlert) {
                Button("Cancel", role: .cancel) {}
                Button("Delete Everything", role: .destructive) {
                    onBack()
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                        sessionManager.deleteSession(id: sessionId)
                    }
                }
            } message: {
                Text("The entire session including audio and transcription will be permanently deleted.")
            }

            Spacer()
        }
    }

    // MARK: - Model Loading & Transcription

    private func startTranscriptionWithModelCheck(_ session: Session) async {
        let targetModelId = selectedModelId

        if appState.isModelLoaded && appState.loadedModelId == targetModelId {
            sessionManager.startTranscription(
                sessionId: session.id,
                modelId: targetModelId,
                ramBudgetMB: Int(ramBudgetGB * 1024),
                apiClient: appState.apiClient
            )
            return
        }

        isPreparingModel = true

        if appState.isModelLoaded && appState.loadedModelId != targetModelId {
            preparationMessage = "Unloading current model..."
            await appState.unloadModel()
        }

        preparationMessage = "Loading model..."
        await appState.loadModel(modelId: targetModelId)

        isPreparingModel = false

        guard appState.isModelLoaded && appState.loadedModelId == targetModelId else { return }

        sessionManager.startTranscription(
            sessionId: session.id,
            modelId: targetModelId,
            ramBudgetMB: Int(ramBudgetGB * 1024),
            apiClient: appState.apiClient
        )
    }

    // MARK: - Reusable Components

    private var detailTranscriptionModePills: some View {
        detailSourcePills(
            localTitle: "Local",
            publicTitle: "Public API",
            localWidth: 78,
            publicWidth: 96,
            selection: detailPublicModeBinding
        )
    }

    private var detailSummaryModePills: some View {
        detailSourcePills(
            localTitle: "Local LLM",
            publicTitle: "Public LLM",
            localWidth: 92,
            publicWidth: 104,
            selection: detailPublicModeBinding
        )
    }

    private var detailPublicTranscriptionModelMenu: some View {
        Menu {
            ForEach(publicTranscriptionModels) { model in
                Button("\(model.displayName) · \(model.providerName)") {
                    selectedTranscriptionProviderId = model.providerId
                    selectedOpenAITranscriptionModel = model.modelId
                }
            }
        } label: {
            detailSelectPill(
                title: selectedAPITranscriptionModel.map { "\($0.displayName) · \($0.providerName)" } ?? "Choose API transcription model",
                icon: "waveform",
                width: 330
            )
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
    }

    private var detailLocalTranscriptionModelMenu: some View {
        Menu {
            ForEach(availableModels) { model in
                Button(model.displayName) {
                    selectedModelId = model.id
                    if ramBudgetGB < minRamBudget {
                        ramBudgetGB = minRamBudget
                    }
                }
            }
        } label: {
            detailSelectPill(
                title: selectedModel?.displayName ?? "Choose local transcription model",
                icon: "externaldrive.fill",
                width: 300
            )
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
    }

    private var detailPublicSummaryModelMenu: some View {
        Menu {
            ForEach(apiLanguageModels) { model in
                Button("\(model.displayName) · \(model.providerName)") {
                    selectedOpenAILanguageModel = model.modelId
                }
            }
        } label: {
            detailSelectPill(
                title: selectedAPILanguageModel.map { "\($0.displayName) · \($0.providerName)" } ?? "Choose API summary model",
                icon: "sparkles",
                width: 330
            )
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
    }

    private var detailLocalSummaryModelMenu: some View {
        Menu {
            ForEach(availableSummaryModels) { model in
                Button(model.displayName) {
                    selectedSummaryModelId = model.id
                }
            }
        } label: {
            detailSelectPill(
                title: selectedSummaryModel?.displayName ?? "Choose local summary model",
                icon: "sparkles",
                width: 300
            )
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
    }

    private var detailSummaryLengthPills: some View {
        HStack(spacing: 2) {
            ForEach(SummaryLengthOption.allCases) { option in
                let selected = currentLengthOption == option
                Button {
                    withAnimation(.easeOut(duration: 0.12)) {
                        currentLengthOption = option
                    }
                } label: {
                    Text(option.displayName)
                        .font(.system(size: 10, weight: selected ? .semibold : .medium))
                        .foregroundStyle(selected ? .white : .secondary)
                        .frame(width: option == .custom ? 70 : 58, height: 26)
                        .background(selected ? AnyShapeStyle(Color.accentColor) : AnyShapeStyle(.clear), in: Capsule())
                }
                .buttonStyle(.plain)
            }
        }
        .padding(3)
        .background(Color.primary.opacity(0.075), in: Capsule())
        .overlay(Capsule().strokeBorder(.white.opacity(0.06), lineWidth: 0.5))
    }

    private var detailPublicModeBinding: Binding<Bool> {
        Binding(
            get: { isPublicMode },
            set: { usePublic in
                if usePublic {
                    let model = selectedAPITranscriptionModel ?? publicTranscriptionModels.first
                    selectedTranscriptionProviderId = model?.providerId ?? TranscriptionProviderDescriptor.openAI.id
                    if let model {
                        selectedOpenAITranscriptionModel = model.modelId
                    }
                } else {
                    selectedTranscriptionProviderId = TranscriptionProviderDescriptor.localBackend.id
                }
            }
        )
    }

    private var manualPublicBatchBinding: Binding<Int> {
        Binding(
            get: { publicSessionTargetBatchCount },
            set: { newValue in
                publicSessionTargetBatchCount = newValue
                normalizePublicBatchTarget()
            }
        )
    }

    private var summaryCustomWordsBinding: Binding<Int> {
        Binding(
            get: { preferredSummaryCustomWords },
            set: { preferredSummaryCustomWords = $0 }
        )
    }

    private func ramBudgetBinding(for session: Session) -> Binding<Double> {
        Binding(
            get: { ramBudgetGB },
            set: { newValue in
                ramBudgetGB = min(max(newValue, minRamBudget), maxRamBudget)
                updateEstimate(session)
            }
        )
    }

    private func detailControlLabel(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 9, weight: .semibold, design: .rounded))
            .foregroundStyle(.tertiary)
            .tracking(0.5)
    }

    private func detailSourcePills(
        localTitle: String,
        publicTitle: String,
        localWidth: CGFloat,
        publicWidth: CGFloat,
        selection: Binding<Bool>
    ) -> some View {
        HStack(spacing: 2) {
            detailSourcePillButton(title: localTitle, width: localWidth, selected: !selection.wrappedValue) {
                withAnimation(.easeOut(duration: 0.12)) {
                    selection.wrappedValue = false
                }
            }
            detailSourcePillButton(title: publicTitle, width: publicWidth, selected: selection.wrappedValue) {
                withAnimation(.easeOut(duration: 0.12)) {
                    selection.wrappedValue = true
                }
            }
        }
        .padding(3)
        .background(Color.primary.opacity(0.075), in: Capsule())
        .overlay(Capsule().strokeBorder(.white.opacity(0.06), lineWidth: 0.5))
    }

    private func detailSourcePillButton(title: String, width: CGFloat, selected: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(selected ? .white : .secondary)
                .frame(width: width, height: 26)
                .background(selected ? AnyShapeStyle(Color.accentColor) : AnyShapeStyle(.clear), in: Capsule())
        }
        .buttonStyle(.plain)
    }

    private func detailSelectPill(title: String, icon: String, width: CGFloat) -> some View {
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

    private func detailIntegerStepperPill(
        label: String,
        value: Binding<Int>,
        in range: ClosedRange<Int>,
        step: Int,
        zeroLabel: String?
    ) -> some View {
        let displayValue = zeroLabel != nil && value.wrappedValue == 0 ? zeroLabel! : "\(value.wrappedValue)"
        return HStack(spacing: 6) {
            Text(label)
                .font(.system(size: 9, weight: .semibold))
                .foregroundStyle(.secondary)
                .frame(width: 48, alignment: .leading)

            Button {
                value.wrappedValue = max(range.lowerBound, value.wrappedValue - step)
            } label: {
                Image(systemName: "minus")
                    .font(.system(size: 8, weight: .bold))
                    .frame(width: 18, height: 18)
                    .background(Color.primary.opacity(0.08), in: Circle())
            }
            .buttonStyle(.plain)
            .disabled(value.wrappedValue <= range.lowerBound)

            Text(displayValue)
                .font(.system(size: 10, weight: .semibold, design: .monospaced))
                .foregroundStyle(.primary)
                .frame(width: zeroLabel == nil ? 42 : 44)

            Button {
                value.wrappedValue = min(range.upperBound, value.wrappedValue + step)
            } label: {
                Image(systemName: "plus")
                    .font(.system(size: 8, weight: .bold))
                    .frame(width: 18, height: 18)
                    .background(Color.primary.opacity(0.08), in: Circle())
            }
            .buttonStyle(.plain)
            .disabled(value.wrappedValue >= range.upperBound)
        }
        .foregroundStyle(.secondary)
        .padding(.horizontal, 9)
        .frame(height: 30)
        .background(Color.primary.opacity(0.075), in: Capsule())
        .overlay(Capsule().strokeBorder(.white.opacity(0.06), lineWidth: 0.5))
    }

    private func detailDoubleStepperPill(
        label: String,
        value: Binding<Double>,
        in range: ClosedRange<Double>,
        step: Double,
        suffix: String
    ) -> some View {
        HStack(spacing: 6) {
            Text(label)
                .font(.system(size: 9, weight: .semibold))
                .foregroundStyle(.secondary)
                .frame(width: 32, alignment: .leading)

            Button {
                value.wrappedValue = max(range.lowerBound, value.wrappedValue - step)
            } label: {
                Image(systemName: "minus")
                    .font(.system(size: 8, weight: .bold))
                    .frame(width: 18, height: 18)
                    .background(Color.primary.opacity(0.08), in: Circle())
            }
            .buttonStyle(.plain)
            .disabled(value.wrappedValue <= range.lowerBound)

            Text("\(String(format: "%.1f", value.wrappedValue)) \(suffix)")
                .font(.system(size: 10, weight: .semibold, design: .monospaced))
                .foregroundStyle(.primary)
                .frame(width: 66)

            Button {
                value.wrappedValue = min(range.upperBound, value.wrappedValue + step)
            } label: {
                Image(systemName: "plus")
                    .font(.system(size: 8, weight: .bold))
                    .frame(width: 18, height: 18)
                    .background(Color.primary.opacity(0.08), in: Circle())
            }
            .buttonStyle(.plain)
            .disabled(value.wrappedValue >= range.upperBound)
        }
        .foregroundStyle(.secondary)
        .padding(.horizontal, 9)
        .frame(height: 30)
        .background(Color.primary.opacity(0.075), in: Capsule())
        .overlay(Capsule().strokeBorder(.white.opacity(0.06), lineWidth: 0.5))
    }

    private func glassCard<Content: View>(@ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            content()
        }
        .padding(14)
        .background(.thinMaterial, in: .rect(cornerRadius: 14, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(.white.opacity(0.06), lineWidth: 0.5)
        )
    }

    private enum CapsuleStyle { case prominent, regular, destructive }

    private func capsuleButton(label: String, icon: String, style: CapsuleStyle, tint: Color = .accentColor, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.system(size: 9, weight: .medium))
                Text(label)
                    .font(.system(size: 10, weight: .medium))
            }
            .foregroundStyle(style == .prominent ? .white : (style == .destructive ? .red : .primary))
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .background(
                style == .prominent
                    ? AnyShapeStyle(tint)
                    : (style == .destructive ? AnyShapeStyle(Color.red.opacity(0.1)) : AnyShapeStyle(.quaternary))
            )
            .clipShape(Capsule())
        }
        .buttonStyle(.plain)
    }

    private func metaPill(icon: String, text: String) -> some View {
        HStack(spacing: 3) {
            Image(systemName: icon)
                .font(.system(size: 8))
            Text(text)
                .font(.system(size: 9))
        }
        .foregroundStyle(.secondary)
        .padding(.horizontal, 6)
        .padding(.vertical, 3)
        .background(.quaternary, in: Capsule())
    }

    private func detailDatePill(_ date: Date) -> some View {
        let formatter = DateFormatter()
        formatter.dateFormat = "MMM d, yyyy 'at' h:mm a"
        return Text(formatter.string(from: date))
            .font(.system(size: 9))
            .foregroundStyle(.tertiary)
            .padding(.horizontal, 6)
            .padding(.vertical, 3)
            .background(.quaternary, in: Capsule())
    }

    private func miniBadge(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 9, weight: .medium))
            .foregroundStyle(.secondary)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(.quaternary, in: Capsule())
    }

    // MARK: - Helpers

    private func updateEstimate(_ session: Session) {
        guard session.duration > 0 else { return }
        Task {
            do {
                estimate = try await appState.apiClient.estimateSessionBatches(
                    audioDuration: session.duration,
                    modelId: selectedModelId,
                    ramBudgetMB: Int(ramBudgetGB * 1024)
                )
            } catch {}
        }
    }

    private func formatEstimatedTime(_ seconds: Double) -> String {
        if seconds < 60 { return "\(Int(seconds))s" }
        else if seconds < 3600 { return "\(Int(seconds / 60)) min" }
        else {
            let hours = Int(seconds / 3600)
            let mins = Int(seconds.truncatingRemainder(dividingBy: 3600) / 60)
            return "\(hours)h \(mins)min"
        }
    }

    private func exportTranscription(_ text: String, sessionName: String) {
        let panel = NSSavePanel()
        panel.nameFieldStringValue = "\(sessionName).txt"
        panel.allowedContentTypes = [.plainText]
        panel.begin { response in
            if response == .OK, let url = panel.url {
                try? text.write(to: url, atomically: true, encoding: .utf8)
            }
        }
    }

    private func exportSummary(_ text: String, sessionName: String, presetId: String) {
        let panel = NSSavePanel()
        panel.nameFieldStringValue = "\(sessionName)-\(presetId).md"
        panel.allowedContentTypes = [.plainText]
        panel.begin { response in
            if response == .OK, let url = panel.url {
                try? text.write(to: url, atomically: true, encoding: .utf8)
            }
        }
    }

    private func formatMB(_ mb: Int) -> String {
        if mb >= 1024 { return String(format: "%.1f GB", Double(mb) / 1024.0) }
        return "\(mb) MB"
    }
}

// MARK: - Live Summary Stream View

/// The only view observing SummaryStreamModel's high-frequency partial text —
/// the same isolation pattern as RecordingActivityModel — so summary streaming
/// re-renders just this markdown pane instead of the whole settings scene.
private struct SummaryStreamingText: View {
    @ObservedObject var stream: SummaryStreamModel
    let fallbackText: String?

    var body: some View {
        ScrollView {
            MarkdownView(text: stream.partialText ?? fallbackText ?? "")
                .frame(maxWidth: .infinity, alignment: .leading)
                .textSelection(.enabled)
                .padding(16)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(.thinMaterial, in: .rect(cornerRadius: 14, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(.white.opacity(0.06), lineWidth: 0.5)
        )
    }
}

// MARK: - Markdown Rendering View

/// Renders Markdown text as styled SwiftUI views.
/// Supports: headings (#-###), bold (**), italic (*), bullet/numbered lists, inline code, code blocks.
private struct MarkdownView: View {
    let text: String

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(Array(parseLines().enumerated()), id: \.offset) { _, element in
                element
            }
        }
    }

    private func parseLines() -> [AnyView] {
        var views: [AnyView] = []
        let lines = text.components(separatedBy: "\n")
        var inCodeBlock = false
        var codeBlockLines: [String] = []

        for line in lines {
            if line.hasPrefix("```") {
                if inCodeBlock {
                    // End code block
                    views.append(AnyView(codeBlockView(codeBlockLines.joined(separator: "\n"))))
                    codeBlockLines = []
                    inCodeBlock = false
                } else {
                    inCodeBlock = true
                }
                continue
            }

            if inCodeBlock {
                codeBlockLines.append(line)
                continue
            }

            if line.hasPrefix("### ") {
                views.append(AnyView(heading(String(line.dropFirst(4)), level: 3)))
            } else if line.hasPrefix("## ") {
                views.append(AnyView(heading(String(line.dropFirst(3)), level: 2)))
            } else if line.hasPrefix("# ") {
                views.append(AnyView(heading(String(line.dropFirst(2)), level: 1)))
            } else if line.hasPrefix("- ") || line.hasPrefix("* ") {
                views.append(AnyView(bulletItem(String(line.dropFirst(2)))))
            } else if let match = line.range(of: #"^\d+\.\s"#, options: .regularExpression) {
                views.append(AnyView(numberedItem(prefix: String(line[match]), content: String(line[match.upperBound...]))))
            } else if line.trimmingCharacters(in: .whitespaces).isEmpty {
                views.append(AnyView(Spacer().frame(height: 4)))
            } else {
                views.append(AnyView(inlineMarkdown(line)))
            }
        }

        // Unclosed code block
        if inCodeBlock && !codeBlockLines.isEmpty {
            views.append(AnyView(codeBlockView(codeBlockLines.joined(separator: "\n"))))
        }

        return views
    }

    private func heading(_ text: String, level: Int) -> some View {
        let size: CGFloat = level == 1 ? 18 : level == 2 ? 15 : 13
        return inlineMarkdown(text)
            .font(.system(size: size, weight: .bold))
            .padding(.top, level == 1 ? 6 : 4)
            .padding(.bottom, 2)
    }

    private func bulletItem(_ text: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text("•")
                .foregroundStyle(.secondary)
            inlineMarkdown(text)
        }
        .padding(.leading, 8)
    }

    private func numberedItem(prefix: String, content: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 4) {
            Text(prefix)
                .foregroundStyle(.secondary)
                .font(.system(.body, design: .default))
            inlineMarkdown(content)
        }
        .padding(.leading, 8)
    }

    private func codeBlockView(_ code: String) -> some View {
        Text(code)
            .font(.system(size: 12, design: .monospaced))
            .foregroundStyle(.primary)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(12)
            .background(.quaternary.opacity(0.3), in: .rect(cornerRadius: 8, style: .continuous))
    }

    private func inlineMarkdown(_ text: String) -> Text {
        var result = Text("")
        var remaining = text[text.startIndex...]

        while !remaining.isEmpty {
            if remaining.hasPrefix("`"), let end = remaining.dropFirst().firstIndex(of: "`") {
                let code = remaining[remaining.index(after: remaining.startIndex)..<end]
                result = result + Text(String(code))
                    .font(.system(.body, design: .monospaced))
                    .foregroundColor(.orange)
                remaining = remaining[remaining.index(after: end)...]
            } else if remaining.hasPrefix("**"), let end = remaining.dropFirst(2).range(of: "**") {
                let bold = remaining[remaining.index(remaining.startIndex, offsetBy: 2)..<end.lowerBound]
                result = result + Text(String(bold)).bold()
                remaining = remaining[end.upperBound...]
            } else if remaining.hasPrefix("*"), let end = remaining.dropFirst().firstIndex(of: "*"),
                      !remaining.hasPrefix("**") {
                let italic = remaining[remaining.index(after: remaining.startIndex)..<end]
                result = result + Text(String(italic)).italic()
                remaining = remaining[remaining.index(after: end)...]
            } else {
                // Grab all plain text until the next special character
                let nextSpecial = remaining.dropFirst().firstIndex(where: { $0 == "*" || $0 == "`" })
                    ?? remaining.endIndex
                result = result + Text(String(remaining[remaining.startIndex..<nextSpecial]))
                remaining = remaining[nextSpecial...]
            }
        }

        return result
    }
}

// MARK: - Summary Preset Manager Sheet

private struct SummaryPresetManagerSheet: View {
    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject var appState: AppState

    @State private var selectedPresetId: String = SessionSummaryPreset.general.rawValue
    @State private var draftName: String = ""
    @State private var draftInstructions: String = ""

    private var presets: [SummaryPresetDefinition] {
        appState.summaryPresetDefinitions.sorted { lhs, rhs in
            if lhs.isBuiltIn != rhs.isBuiltIn { return lhs.isBuiltIn && !rhs.isBuiltIn }
            return lhs.displayName.localizedCaseInsensitiveCompare(rhs.displayName) == .orderedAscending
        }
    }

    private var selectedPreset: SummaryPresetDefinition? {
        presets.first(where: { $0.id == selectedPresetId }) ?? presets.first
    }

    var body: some View {
        VStack(spacing: 0) {
            sheetHeader
            Divider()
            HStack(spacing: 0) {
                presetSidebar
                Divider()
                presetEditor
            }
        }
        .frame(minWidth: 900, minHeight: 540)
        .onAppear {
            if let preset = selectedPreset {
                loadDraft(from: preset)
            } else if let first = presets.first {
                selectedPresetId = first.id
                loadDraft(from: first)
            }
        }
        .onChange(of: selectedPresetId) { _, newValue in
            if let preset = presets.first(where: { $0.id == newValue }) {
                loadDraft(from: preset)
            }
        }
    }

    private var sheetHeader: some View {
        HStack(spacing: 14) {
            ZStack {
                Circle()
                    .fill(Color.accentColor.opacity(0.14))
                    .frame(width: 42, height: 42)
                Image(systemName: "text.badge.star")
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(Color.accentColor)
            }

            VStack(alignment: .leading, spacing: 3) {
                Text("Summary Presets")
                    .font(.system(size: 17, weight: .semibold, design: .rounded))
                Text("Edit built-ins or add custom views.")
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
            }

            Spacer()

            Button("Done") { dismiss() }
                .buttonStyle(.plain)
                .modifier(PresetSheetActionPillStyle(tint: .accentColor, prominent: true))
                .keyboardShortcut(.defaultAction)
        }
        .padding(20)
    }

    private var presetSidebar: some View {
        VStack(alignment: .leading, spacing: 12) {
            Button {
                let preset = SummaryPresetLibrary.makeCustomPreset(
                    name: "Custom Summary",
                    instructions: """
                    Summarize the transcript in the same language as the transcript.
                    Focus on the perspective described by the preset title and keep the output concrete and detailed.
                    Return Markdown.
                    """
                )
                appState.upsertSummaryPreset(preset)
                selectedPresetId = preset.id
                loadDraft(from: preset)
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: "plus")
                        .font(.system(size: 9, weight: .bold))
                    Text("New custom preset")
                        .font(.system(size: 11, weight: .semibold))
                }
                .foregroundStyle(Color.accentColor)
                .frame(maxWidth: .infinity)
                .frame(height: 30)
                .background(Color.accentColor.opacity(0.10), in: Capsule())
                .overlay(Capsule().strokeBorder(Color.accentColor.opacity(0.18), lineWidth: 0.7))
                .contentShape(Capsule())
            }
            .buttonStyle(.plain)

            ScrollView {
                VStack(alignment: .leading, spacing: 6) {
                    ForEach(presets) { preset in
                        presetRow(preset)
                    }
                }
            }
        }
        .frame(width: 250, alignment: .top)
        .frame(maxHeight: .infinity, alignment: .top)
        .padding(20)
    }

    private func presetRow(_ preset: SummaryPresetDefinition) -> some View {
        Button {
            selectedPresetId = preset.id
            loadDraft(from: preset)
        } label: {
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text(preset.displayName)
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(.primary)
                        .lineLimit(1)
                    Text(preset.isBuiltIn ? "Built-in" : "Custom")
                        .font(.system(size: 9, weight: .medium))
                        .foregroundStyle(.tertiary)
                }
                Spacer()
                if selectedPresetId == preset.id {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.system(size: 12))
                        .foregroundStyle(Color.accentColor)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .background(
                selectedPresetId == preset.id
                    ? AnyShapeStyle(Color.accentColor.opacity(0.10))
                    : AnyShapeStyle(Color.primary.opacity(0.045)),
                in: RoundedRectangle(cornerRadius: 10, style: .continuous)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .strokeBorder(
                        selectedPresetId == preset.id ? Color.accentColor.opacity(0.35) : Color.white.opacity(0.06),
                        lineWidth: 0.7
                    )
            )
            .contentShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
        }
        .buttonStyle(.plain)
    }

    @ViewBuilder
    private var presetEditor: some View {
        VStack(alignment: .leading, spacing: 16) {
            if let preset = selectedPreset {
                VStack(alignment: .leading, spacing: 6) {
                    Text("PRESET NAME")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(.tertiary)
                        .tracking(0.7)
                    TextField("Preset name", text: $draftName)
                        .textFieldStyle(.plain)
                        .font(.system(size: 13, weight: .semibold))
                        .padding(.horizontal, 11)
                        .frame(height: 34)
                        .background(Color.primary.opacity(0.055), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                        .overlay(
                            RoundedRectangle(cornerRadius: 10, style: .continuous)
                                .strokeBorder(.white.opacity(0.07), lineWidth: 0.5)
                        )
                }

                VStack(alignment: .leading, spacing: 6) {
                    Text("INSTRUCTIONS")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(.tertiary)
                        .tracking(0.7)
                    TextEditor(text: $draftInstructions)
                        .font(.system(size: 13))
                        .scrollContentBackground(.hidden)
                        .frame(minHeight: 260)
                        .padding(10)
                        .background(Color.primary.opacity(0.055), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                        .overlay(
                            RoundedRectangle(cornerRadius: 12, style: .continuous)
                                .strokeBorder(.white.opacity(0.07), lineWidth: 0.5)
                        )
                }

                presetActionButtons(preset)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .padding(20)
    }

    private func presetActionButtons(_ preset: SummaryPresetDefinition) -> some View {
        HStack(spacing: 10) {
            if preset.isBuiltIn {
                Button {
                    appState.resetBuiltInSummaryPreset(id: preset.id)
                    if let resetPreset = appState.summaryPresetDefinitions.first(where: { $0.id == preset.id }) {
                        loadDraft(from: resetPreset)
                    }
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: "arrow.counterclockwise")
                            .font(.system(size: 9, weight: .bold))
                        Text("Reset to default")
                    }
                }
                .buttonStyle(.plain)
                .modifier(PresetSheetActionPillStyle(tint: .primary, prominent: false))
            } else {
                Button {
                    let deletedId = preset.id
                    appState.deleteSummaryPreset(id: deletedId)
                    selectedPresetId = appState.summaryPresetDefinitions.first?.id ?? SessionSummaryPreset.general.rawValue
                    if let nextPreset = appState.summaryPresetDefinitions.first(where: { $0.id == selectedPresetId }) {
                        loadDraft(from: nextPreset)
                    }
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: "trash")
                            .font(.system(size: 9, weight: .bold))
                        Text("Delete preset")
                    }
                }
                .buttonStyle(.plain)
                .modifier(PresetSheetActionPillStyle(tint: .red, prominent: false))
            }

            Spacer()

            Button("Save preset") {
                var updated = preset
                updated.displayName = draftName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                    ? preset.displayName
                    : draftName.trimmingCharacters(in: .whitespacesAndNewlines)
                updated.instructions = draftInstructions.trimmingCharacters(in: .whitespacesAndNewlines)
                appState.upsertSummaryPreset(updated)
                loadDraft(from: updated)
            }
            .buttonStyle(.plain)
            .modifier(PresetSheetActionPillStyle(tint: .accentColor, prominent: true))
            .disabled(draftInstructions.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            .opacity(draftInstructions.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? 0.45 : 1)
        }
    }

    private func loadDraft(from preset: SummaryPresetDefinition) {
        draftName = preset.displayName
        draftInstructions = preset.instructions
    }
}

// MARK: - Preset Sheet Styling

private struct PresetSheetActionPillStyle: ViewModifier {
    let tint: Color
    let prominent: Bool

    func body(content: Content) -> some View {
        content
            .font(.system(size: 11, weight: .semibold))
            .foregroundStyle(prominent ? Color.white : tint)
            .padding(.horizontal, 14)
            .frame(height: 30)
            .background(prominent ? AnyShapeStyle(tint) : AnyShapeStyle(Color.primary.opacity(0.075)), in: Capsule())
            .overlay(Capsule().strokeBorder(prominent ? tint.opacity(0.28) : .white.opacity(0.07), lineWidth: 0.6))
            .contentShape(Capsule())
    }
}
