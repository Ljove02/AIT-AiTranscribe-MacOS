import SwiftUI
import AppKit
import UniformTypeIdentifiers

// MARK: - Session Detail View

/// Detail view for a single session — shows audio info, transcription controls, and results.
struct SessionDetailView: View {
    @EnvironmentObject var sessionManager: SessionManager
    @EnvironmentObject var appState: AppState
    let sessionId: UUID
    let onBack: () -> Void
    var onNavigateToModels: (() -> Void)? = nil

    @State private var selectedModelId: String = "parakeet-v2"
    @State private var ramBudgetGB: Double = 4.0
    @State private var showDeleteAudioAlert = false
    @State private var showDeleteAllAlert = false
    @State private var estimate: SessionEstimateResponse? = nil
    @State private var isEditingName = false
    @State private var editedName = ""
    @State private var isPreparingModel = false
    @State private var preparationMessage = ""

    private var session: Session? {
        sessionManager.sessions.first(where: { $0.id == sessionId })
    }

    private var isTranscribing: Bool {
        sessionManager.currentTranscribingSessionId == sessionId
    }

    /// System RAM in GB (minus 4GB for OS)
    private var maxRamBudget: Double {
        let totalRAM = Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824
        return max(minRamBudget, totalRAM - 4)
    }

    /// Minimum RAM budget = model RAM + 1GB buffer (can't go below what the model needs)
    private var minRamBudget: Double {
        let modelRamGB = Double(selectedModel?.ramMB ?? 3000) / 1024.0
        return ceil(modelRamGB + 1.0)
    }

    /// Currently selected model info
    private var selectedModel: ModelInfoResponse? {
        availableModels.first(where: { $0.id == selectedModelId })
    }

    /// Available models that support batch transcription (downloaded + session-compatible)
    private var availableModels: [ModelInfoResponse] {
        appState.availableModels.filter { $0.downloaded == true && $0.sessionCompatible }
    }

    var body: some View {
        VStack(spacing: 0) {
            // Back button header
            HStack {
                Button {
                    withAnimation(.easeInOut(duration: 0.15)) {
                        onBack()
                    }
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: "chevron.left")
                        Text("Back to Sessions")
                    }
                    .font(.caption)
                }
                .buttonStyle(.plain)
                .foregroundColor(.accentColor)

                Spacer()
            }
            .padding(.horizontal, 16)
            .padding(.top, 12)
            .padding(.bottom, 8)

            Divider()

            if let session {
                ScrollView {
                    VStack(alignment: .leading, spacing: 20) {
                        // Audio file card
                        audioFileCard(session)

                        Divider()

                        // Transcription section
                        transcriptionSection(session)

                        Divider()

                        // Danger zone
                        dangerZone(session)
                    }
                    .padding(16)
                }
            } else {
                Text("Session not found")
                    .foregroundColor(.secondary)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
    }

    // MARK: - Audio File Card

    private func audioFileCard(_ session: Session) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 12) {
                // File icon
                Image(systemName: session.hasAudio ? "music.note" : "music.note.slash")
                    .font(.system(size: 24))
                    .foregroundColor(session.hasAudio ? .accentColor : .secondary)
                    .frame(width: 40, height: 40)
                    .background(
                        RoundedRectangle(cornerRadius: 8)
                            .fill(Color.accentColor.opacity(0.1))
                    )

                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 6) {
                        if isEditingName {
                            TextField("Session name", text: $editedName)
                                .font(.headline)
                                .textFieldStyle(.roundedBorder)
                                .frame(maxWidth: 250)
                                .onSubmit {
                                    sessionManager.renameSession(id: session.id, newName: editedName)
                                    isEditingName = false
                                }
                        } else {
                            Text(session.name)
                                .font(.headline)
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
                                .foregroundColor(.accentColor)
                                .font(.system(size: 14))
                        }
                        .buttonStyle(.plain)
                    }

                    HStack(spacing: 8) {
                        let dateFormatter = {
                            let f = DateFormatter()
                            f.dateFormat = "MMMM d, yyyy 'at' h:mm a"
                            return f
                        }()
                        Text(dateFormatter.string(from: session.createdAt))
                            .font(.caption)
                            .foregroundColor(.secondary)

                        if session.duration > 0 {
                            Text("•")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text(SessionManager.formatDuration(session.duration))
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }

                        if session.fileSize > 0 {
                            Text("•")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text(SessionManager.formatFileSize(session.fileSize))
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                }

                Spacer()
            }

            if session.hasAudio {
                HStack(spacing: 8) {
                    Button("Open in Finder") {
                        let audioURL = sessionManager.getAudioURL(for: session)
                        NSWorkspace.shared.activateFileViewerSelecting([audioURL])
                    }
                    .font(.caption)
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }
            } else {
                Text("Audio file deleted")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .italic()
            }
        }
    }

    // MARK: - Transcription Section

    @ViewBuilder
    private func transcriptionSection(_ session: Session) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Transcription")
                .font(.headline)

            if isTranscribing {
                // State B: Transcribing in progress
                transcribingView()
            } else if session.hasTranscription, let text = session.transcriptionText {
                // State C: Completed
                completedView(session, text: text)
            } else {
                // State A: Not yet transcribed
                notTranscribedView(session)
            }
        }
    }

    // MARK: State A — Not Transcribed

    @ViewBuilder
    private func notTranscribedView(_ session: Session) -> some View {
        if !session.hasAudio {
            Text("Cannot transcribe — audio file has been deleted.")
                .font(.caption)
                .foregroundColor(.secondary)
        } else {
            notTranscribedControls(session)
        }
    }

    private func notTranscribedControls(_ session: Session) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            // Model selector
            HStack {
                Text("Model:")
                    .font(.caption)
                    .foregroundColor(.secondary)

                if availableModels.isEmpty {
                    VStack(spacing: 8) {
                        Image(systemName: "cube.box")
                            .font(.title3)
                            .foregroundColor(.secondary)
                        Text("No compatible models installed")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Text("Download a session-compatible model to transcribe recordings.")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                        if let onNavigateToModels {
                            Button("Go to Models") {
                                onNavigateToModels()
                            }
                            .buttonStyle(.borderedProminent)
                            .controlSize(.small)
                        }
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(RoundedRectangle(cornerRadius: 8).fill(Color.red.opacity(0.05)))
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.red.opacity(0.2)))
                } else {
                    Picker("", selection: $selectedModelId) {
                        ForEach(availableModels) { model in
                            Text(model.displayName)
                                .tag(model.id)
                        }
                    }
                    .labelsHidden()
                    .frame(maxWidth: 250)
                }
            }

            if selectedModelId == "whisper-large-v3" {
                HStack(spacing: 4) {
                    Image(systemName: "info.circle")
                        .foregroundColor(.orange)
                    Text("Whisper Large v3 is slower but supports 99+ languages with highest accuracy.")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            }

            // RAM budget slider
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("RAM Budget:")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text("\(String(format: "%.1f", ramBudgetGB)) GB")
                        .font(.caption.bold())
                }

                Slider(value: $ramBudgetGB, in: minRamBudget...maxRamBudget, step: 0.5)
                    .onChange(of: ramBudgetGB) { _, _ in
                        updateEstimate(session)
                    }
                    .onChange(of: selectedModelId) { _, _ in
                        // Clamp budget when switching to a larger model
                        if ramBudgetGB < minRamBudget {
                            ramBudgetGB = minRamBudget
                        }
                    }

                if let est = estimate {
                    Text("~\(est.numBatches) batches, ~\(formatEstimatedTime(est.estimatedTimeSeconds))")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                Text("Higher budget = faster transcription. Lower = less RAM used.")
                    .font(.caption2)
                    .foregroundColor(.secondary.opacity(0.7))
            }

            // Start button
            Button {
                Task {
                    await startTranscriptionWithModelCheck(session)
                }
            } label: {
                HStack {
                    if isPreparingModel {
                        ProgressView()
                            .scaleEffect(0.7)
                        Text(preparationMessage)
                    } else {
                        Image(systemName: "waveform.badge.magnifyingglass")
                        Text("Start Transcription")
                    }
                }
            }
            .buttonStyle(.borderedProminent)
            .disabled(availableModels.isEmpty || !session.hasAudio || isPreparingModel || appState.isModelLoading)
        }
        .task {
            // Ensure RAM budget starts at a valid minimum for the selected model
            if ramBudgetGB < minRamBudget {
                ramBudgetGB = minRamBudget
            }
            updateEstimate(session)
        }
    }

    // MARK: State B — Transcribing

    private func transcribingView() -> some View {
        VStack(alignment: .leading, spacing: 12) {
            if let progress = sessionManager.transcriptionProgress {
                // Batch counter
                HStack {
                    Text("Transcribing batch \(progress.batch) of \(progress.totalBatches)")
                        .font(.system(.caption, design: .monospaced))
                    Spacer()
                    Text("\(Int(progress.progress * 100))%")
                        .font(.system(.caption, design: .monospaced).bold())
                }

                // Progress bar
                ProgressView(value: progress.progress)
                    .progressViewStyle(.linear)

                // Stats row
                HStack(spacing: 16) {
                    if let cpu = progress.cpuPercent {
                        Label("CPU: \(Int(cpu))%", systemImage: "cpu")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                    if let mem = progress.memoryMB {
                        Label("RAM: \(Int(mem)) MB", systemImage: "memorychip")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                    if let eta = progress.etaSeconds, eta > 0 {
                        Label("~\(formatEstimatedTime(eta)) left", systemImage: "clock")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                }
            } else {
                HStack(spacing: 8) {
                    ProgressView()
                        .scaleEffect(0.7)
                    Text("Preparing transcription...")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }

            // Cancel button
            Button("Cancel Transcription", role: .destructive) {
                sessionManager.cancelTranscription(apiClient: appState.apiClient)
            }
            .font(.caption)
            .buttonStyle(.bordered)
            .controlSize(.small)
        }
    }

    // MARK: State C — Completed

    private func completedView(_ session: Session, text: String) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            // Action buttons
            HStack(spacing: 8) {
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(text, forType: .string)
                } label: {
                    Label("Copy Transcription", systemImage: "doc.on.doc")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)

                Button {
                    exportTranscription(text, sessionName: session.name)
                } label: {
                    Label("Export as TXT", systemImage: "square.and.arrow.up")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }

            // Stats
            HStack(spacing: 12) {
                if let wc = session.wordCount, wc > 0 {
                    Text("\(wc) words")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                if let time = session.transcriptionTime, time > 0 {
                    Text("Transcribed in \(formatEstimatedTime(time))")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                if let model = session.modelUsed {
                    Text("Model: \(model)")
                        .font(.caption)
                        .foregroundColor(.accentColor)
                }
            }

            // Scrollable text preview
            ScrollView {
                Text(text)
                    .font(.system(.body, design: .default))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(12)
            }
            .frame(maxHeight: 300)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color(NSColor.textBackgroundColor))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(Color.secondary.opacity(0.2), lineWidth: 1)
            )

            // Re-transcribe option
            if session.hasAudio {
                Button("Re-transcribe with different settings") {
                    // Reset transcription state to show the "not transcribed" UI
                    if let index = sessionManager.sessions.firstIndex(where: { $0.id == session.id }) {
                        sessionManager.sessions[index].hasTranscription = false
                        sessionManager.sessions[index].transcriptionText = nil
                        sessionManager.sessions[index].status = .idle
                    }
                }
                .font(.caption)
                .buttonStyle(.plain)
                .foregroundColor(.accentColor)
            }
        }
    }

    // MARK: - Danger Zone

    private func dangerZone(_ session: Session) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Manage Session")
                .font(.caption.bold())
                .foregroundColor(.secondary)

            HStack(spacing: 8) {
                if session.hasAudio {
                    Button("Delete Audio", role: .destructive) {
                        showDeleteAudioAlert = true
                    }
                    .font(.caption)
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .alert("Delete Audio?", isPresented: $showDeleteAudioAlert) {
                        Button("Cancel", role: .cancel) {}
                        Button("Delete Audio", role: .destructive) {
                            sessionManager.deleteSessionAudio(id: session.id)
                        }
                    } message: {
                        Text("The audio file will be deleted. Your transcription will be kept, but you won't be able to re-transcribe.")
                    }
                }

                Button("Delete Everything", role: .destructive) {
                    showDeleteAllAlert = true
                }
                .font(.caption)
                .buttonStyle(.bordered)
                .controlSize(.small)
                .alert("Delete Session?", isPresented: $showDeleteAllAlert) {
                    Button("Cancel", role: .cancel) {}
                    Button("Delete Everything", role: .destructive) {
                        onBack()
                        // Slight delay so navigation completes before deletion
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                            sessionManager.deleteSession(id: sessionId)
                        }
                    }
                } message: {
                    Text("The entire session including audio and transcription will be permanently deleted.")
                }
            }
        }
    }

    // MARK: - Model Loading & Transcription

    private func startTranscriptionWithModelCheck(_ session: Session) async {
        let targetModelId = selectedModelId

        // Case 1: Correct model already loaded — transcribe immediately
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

        // Case 2: Different model loaded — unload first
        if appState.isModelLoaded && appState.loadedModelId != targetModelId {
            preparationMessage = "Unloading current model..."
            await appState.unloadModel()
        }

        // Case 3: Load the target model
        preparationMessage = "Loading model..."
        await appState.loadModel(modelId: targetModelId)

        isPreparingModel = false

        // Verify load succeeded
        guard appState.isModelLoaded && appState.loadedModelId == targetModelId else {
            return
        }

        // Start transcription
        sessionManager.startTranscription(
            sessionId: session.id,
            modelId: targetModelId,
            ramBudgetMB: Int(ramBudgetGB * 1024),
            apiClient: appState.apiClient
        )
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
            } catch {
                // Silently fail — estimates are optional UI polish
            }
        }
    }

    private func formatEstimatedTime(_ seconds: Double) -> String {
        if seconds < 60 {
            return "\(Int(seconds))s"
        } else if seconds < 3600 {
            return "\(Int(seconds / 60)) min"
        } else {
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
}
