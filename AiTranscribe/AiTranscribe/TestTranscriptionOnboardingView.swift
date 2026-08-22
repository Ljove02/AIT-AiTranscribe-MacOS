/*
 TestTranscriptionOnboardingView.swift
 =====================================

 Test transcription screen (OPTIONAL but RECOMMENDED).
 
 Features:
 - Show test sentence from a pool of 5
 - Auto-detect silence to stop recording (Option A)
 - Calculate accuracy with Levenshtein distance
 - Allow retry with new sentence
 - Option to go back and download more models
 */

import SwiftUI
import AVFoundation

struct TestTranscriptionOnboardingView: View {
    let onNext: () -> Void
    let onBack: () -> Void
    
    @EnvironmentObject var appState: AppState
    
    @State private var currentSentenceIndex = 0
    @State private var isRecording = false
    @State private var isTranscribing = false
    @State private var transcriptionResult: String?
    @State private var accuracy: Double?
    @State private var recordingDuration: TimeInterval = 0
    @State private var silenceTimer: Timer?
    @State private var recordingTimer: Timer?
    @State private var audioData: Data?
    @State private var silenceStartTime: Date?
    @State private var isLoadingModel = false
    @State private var selectedTestMode: TestTranscriptionMode = .publicAPI
    @State private var selectedLocalModelId = ""
    @State private var selectedAPIModelId = ""
    @State private var transcriptionError: String?
    @State private var providerProgressMessage: String?
    
    // API client for transcription
    private let apiClient = APIClient()
    
    // Silence detection constants
    private let silenceThreshold: TimeInterval = 2.0 // 2 seconds of silence
    
    // Test sentences
    private let testSentences = [
        "The quick brown fox jumps over the lazy dog.",
        "She sells seashells by the seashore.",
        "How much wood would a woodchuck chuck if a woodchuck could chuck wood.",
        "Peter Piper picked a peck of pickled peppers.",
        "I scream you scream we all scream for ice cream."
    ]
    
    private var currentSentence: String {
        testSentences[currentSentenceIndex]
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            OnboardingHeader(
                eyebrow: "Test",
                title: "Try one recording",
                subtitle: "Switch Local/Public, pick a model, then read one sentence."
            )
            .padding(.bottom, 28)

            HStack(alignment: .top, spacing: 44) {
                VStack(alignment: .leading, spacing: 20) {
                    modeSwitch
                    modelPicker
                    statusNote
                }
                .frame(width: 318, alignment: .topLeading)

                VStack(alignment: .leading, spacing: 22) {
                    sentencePanel
                    testStatePanel
                }
                .frame(maxWidth: .infinity, alignment: .topLeading)
            }

            Spacer()

            bottomActions
        }
        .padding(34)
        .onAppear(perform: prepareInitialSelection)
        .onChange(of: selectedTestMode) { _, _ in
            transcriptionError = nil
            transcriptionResult = nil
            accuracy = nil
            prepareInitialSelection()
        }
    }

    private var downloadedLocalModels: [ModelInfoResponse] {
        appState.availableModels.filter(\.downloaded)
    }

    private var usableAPIModels: [APITranscriptionModel] {
        APITranscriptionModelStore.load().filter { model in
            APITranscriptionModelStore.isSupportedModel(model.modelId, providerId: model.providerId) &&
            APIKeychainStore.hasAPIKey(providerId: model.providerId)
        }
    }

    private var hasLocalTestRoute: Bool {
        !downloadedLocalModels.isEmpty
    }

    private var hasPublicTestRoute: Bool {
        !usableAPIModels.isEmpty
    }

    private var isModeSwitchLocked: Bool {
        hasLocalTestRoute != hasPublicTestRoute
    }

    private var selectedAPIModel: APITranscriptionModel? {
        usableAPIModels.first { "\($0.providerId):\($0.modelId)" == selectedAPIModelId }
    }

    private var modeSwitch: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Mode")
                .font(.system(size: 13, weight: .bold, design: .rounded))
                .tracking(1.1)
                .textCase(.uppercase)
                .foregroundStyle(.secondary)

            HStack(spacing: 2) {
                TestProviderModeButton(
                    title: "Local",
                    isSelected: selectedTestMode == .local,
                    isEnabled: hasLocalTestRoute && !isModeSwitchLocked,
                    tint: .green
                ) {
                    withAnimation(.spring(duration: 0.24, bounce: 0.14)) {
                        selectedTestMode = .local
                    }
                }

                TestProviderModeButton(
                    title: "Public",
                    isSelected: selectedTestMode == .publicAPI,
                    isEnabled: hasPublicTestRoute && !isModeSwitchLocked,
                    tint: .blue
                ) {
                    withAnimation(.spring(duration: 0.24, bounce: 0.14)) {
                        selectedTestMode = .publicAPI
                    }
                }
            }
            .padding(3)
            .background(.white.opacity(0.06), in: Capsule())
            .overlay(Capsule().strokeBorder(.white.opacity(0.08), lineWidth: 0.5))

            Text(selectedTestMode.description)
                .font(.system(size: 13, weight: .medium))
                .lineSpacing(2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            if isModeSwitchLocked {
                Label("\(selectedTestMode.shortTitle) is the only configured test route.", systemImage: "lock.fill")
                    .font(.system(size: 12, weight: .semibold, design: .rounded))
                    .foregroundStyle(.secondary)
            }
        }
    }

    @ViewBuilder
    private var modelPicker: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Model")
                .font(.system(size: 13, weight: .bold, design: .rounded))
                .tracking(1.1)
                .textCase(.uppercase)
                .foregroundStyle(.secondary)

            if selectedTestMode == .local {
                if downloadedLocalModels.isEmpty {
                    EmptyTestModelState(
                        icon: "arrow.down.circle",
                        title: "No local model ready",
                        text: "Download a local model first, or switch to Public for this test."
                    )
                } else {
                    Menu {
                        ForEach(downloadedLocalModels) { model in
                            Button(model.displayName) {
                                selectedLocalModelId = model.id
                            }
                        }
                    } label: {
                        onboardingSelectPill(
                            title: downloadedLocalModels.first(where: { $0.id == selectedLocalModelId })?.displayName ?? "Choose local model",
                            icon: "externaldrive.fill",
                            width: 300
                        )
                    }
                    .menuStyle(.button)
                    .buttonStyle(.plain)
                    .menuIndicator(.hidden)
                    .disabled(isLoadingModel || isRecording || isTranscribing)
                    .opacity(isLoadingModel || isRecording || isTranscribing ? 0.55 : 1)
                    .onChange(of: selectedLocalModelId) { _, newValue in
                        loadLocalModel(newValue)
                    }

                    if isLoadingModel {
                        Label("Loading local model...", systemImage: "arrow.triangle.2.circlepath")
                            .font(.system(size: 12, weight: .medium))
                            .foregroundStyle(.secondary)
                    }
                }
            } else {
                if usableAPIModels.isEmpty {
                    EmptyTestModelState(
                        icon: "key.fill",
                        title: "No public model ready",
                        text: "Add an API key and model first, or switch to Local for this test."
                    )
                } else {
                    Menu {
                        ForEach(usableAPIModels) { model in
                            Button("\(model.providerName) · \(model.displayName)") {
                                selectedAPIModelId = "\(model.providerId):\(model.modelId)"
                            }
                        }
                    } label: {
                        onboardingSelectPill(
                            title: usableAPIModels.first(where: { "\($0.providerId):\($0.modelId)" == selectedAPIModelId })
                                .map { "\($0.providerName) · \($0.displayName)" } ?? "Choose public model",
                            icon: "network",
                            width: 300
                        )
                    }
                    .menuStyle(.button)
                    .buttonStyle(.plain)
                    .menuIndicator(.hidden)
                    .disabled(isRecording || isTranscribing)
                    .opacity(isRecording || isTranscribing ? 0.55 : 1)
                    .onChange(of: selectedAPIModelId) { _, _ in
                        persistSelectedAPIModel()
                    }
                }
            }
        }
    }

    private func onboardingSelectPill(title: String, icon: String, width: CGFloat) -> some View {
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
        .background(.white.opacity(0.06), in: Capsule())
        .overlay(Capsule().strokeBorder(.white.opacity(0.08), lineWidth: 0.5))
        .contentShape(Capsule())
    }

    private var statusNote: some View {
        HStack(alignment: .top, spacing: 9) {
            Image(systemName: validationState.icon)
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(validationState.color)
                .padding(.top, 1)

            Text(validationState.message)
                .font(.system(size: 13, weight: .medium))
                .lineSpacing(2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            Spacer()
        }
    }

    private var sentencePanel: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Read this")
                .font(.system(size: 13, weight: .bold, design: .rounded))
                .tracking(1.1)
                .textCase(.uppercase)
                .foregroundStyle(.secondary)

            Text(currentSentence)
                .font(.system(size: 29, weight: .bold, design: .rounded))
                .lineSpacing(5)
                .foregroundStyle(.primary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: 455, alignment: .leading)
    }

    @ViewBuilder
    private var testStatePanel: some View {
        if let result = transcriptionResult, let acc = accuracy {
            TestResultsView(
                originalText: currentSentence,
                transcribedText: result,
                accuracy: acc,
                onTryAgain: tryAgain
            )
        } else if isRecording {
            RecordingTestStateView(duration: recordingDuration)
        } else if isTranscribing {
            TranscribingTestStateView(message: providerProgressMessage)
        } else if let transcriptionError {
            InlineTestIssueView(
                icon: "exclamationmark.triangle.fill",
                title: "Transcription failed",
                text: transcriptionError,
                color: .orange
            )
        }
    }

    private var bottomActions: some View {
        HStack {
            OnboardingSecondaryActionButton(
                title: "Back",
                systemImage: "arrow.left",
                action: onBack
            )
            .disabled(isRecording || isTranscribing)

            Button {
                onNext()
            } label: {
                Label("Skip Test", systemImage: "arrow.right")
                    .font(.system(size: 14, weight: .semibold, design: .rounded))
                    .frame(minWidth: 128)
            }
            .buttonStyle(.bordered)
            .controlSize(.large)
            .disabled(isRecording || isTranscribing)

            Spacer()

            if isRecording {
                OnboardingPrimaryActionButton(
                    title: "Stop Recording",
                    systemImage: "stop.circle.fill",
                    action: stopRecording
                )
            } else if transcriptionResult == nil {
                OnboardingPrimaryActionButton(
                    title: isTranscribing ? "Transcribing" : "Start Recording",
                    systemImage: "mic.circle.fill",
                    isDisabled: !canStartRecording,
                    isLoading: isTranscribing,
                    action: startRecording
                )
            } else {
                OnboardingPrimaryActionButton(
                    title: "Continue",
                    systemImage: "arrow.right",
                    action: onNext
                )
            }
        }
    }

    private var canStartRecording: Bool {
        guard !isRecording, !isTranscribing, !isLoadingModel else { return false }
        switch selectedTestMode {
        case .local:
            return !selectedLocalModelId.isEmpty && appState.loadedModelId == selectedLocalModelId
        case .publicAPI:
            return selectedAPIModel != nil
        }
    }

    private var validationState: TestValidationState {
        if let transcriptionError {
            return TestValidationState(icon: "exclamationmark.triangle.fill", color: .orange, message: transcriptionError)
        }
        if isRecording {
            return TestValidationState(icon: "record.circle.fill", color: .red, message: "Recording. Silence auto-stops after a short pause.")
        }
        if isTranscribing {
            return TestValidationState(icon: "arrow.triangle.2.circlepath", color: .blue, message: providerProgressMessage ?? "Transcribing...")
        }
        switch selectedTestMode {
        case .local:
            if downloadedLocalModels.isEmpty {
                return TestValidationState(icon: "info.circle.fill", color: .orange, message: "No local model is downloaded yet.")
            }
            if selectedLocalModelId.isEmpty {
                return TestValidationState(icon: "cursorarrow.click.2", color: .orange, message: "Choose a local model before starting the test.")
            }
            if appState.loadedModelId != selectedLocalModelId {
                return TestValidationState(icon: "arrow.triangle.2.circlepath", color: .orange, message: "Load the selected local model before recording.")
            }
            return TestValidationState(icon: "checkmark.circle.fill", color: .green, message: "Local mode keeps this test on your Mac.")
        case .publicAPI:
            if usableAPIModels.isEmpty {
                return TestValidationState(icon: "key.fill", color: .orange, message: "Public mode needs a saved provider key and API model.")
            }
            return TestValidationState(icon: "network", color: .blue, message: "Public mode sends this test audio to \(selectedAPIModel?.providerName ?? "the provider").")
        }
    }
    
    // MARK: - Recording Control
    
    private func startRecording() {
        guard canStartRecording else { return }
        
        // Reset state
        transcriptionResult = nil
        accuracy = nil
        transcriptionError = nil
        providerProgressMessage = nil
        recordingDuration = 0
        isRecording = true
        
        Task { @MainActor in
            if await appState.audioRecorder.startRecording() {
                // Start duration timer
                recordingTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { _ in
                    recordingDuration = appState.audioRecorder.duration

                    // Check for silence (volume < 0.01 for 2 seconds)
                    if appState.audioRecorder.currentVolume < 0.01 {
                        self.checkForSilence()
                    } else {
                        // Reset silence tracking if sound detected
                        self.silenceStartTime = nil
                    }

                    // Safety timeout at 30 seconds
                    if recordingDuration >= 30.0 {
                        stopRecording()
                    }
                }
            } else {
                isRecording = false
            }
        }
    }
    
    private func checkForSilence() {
        let now = Date()
        
        if let startTime = silenceStartTime {
            // Already tracking silence
            if now.timeIntervalSince(startTime) >= silenceThreshold {
                // 2 seconds of silence detected - auto stop
                stopRecording()
            }
        } else {
            // Start tracking silence
            silenceStartTime = now
        }
    }
    
    private func stopRecording() {
        recordingTimer?.invalidate()
        recordingTimer = nil
        silenceStartTime = nil
        
        isRecording = false
        isTranscribing = true
        
        // Get audio data from recorder
        Task {
            guard let data = await appState.audioRecorder.stopRecording() else {
                await MainActor.run {
                    isTranscribing = false
                }
                return
            }

            await MainActor.run {
                audioData = data
            }

            // Transcribe
            await transcribeAudio(data)
        }
    }
    
    private func transcribeAudio(_ data: Data) async {
        do {
            let request = TranscriptionRequest(
                audioData: data,
                recordedDurationSeconds: max(recordingDuration, 0.1),
                workflow: .basic,
                model: currentTestModelDescriptor()
            )
            let provider = currentTestProvider()
            let result = try await provider.transcribe(request) { update in
                Task { @MainActor in
                    providerProgressMessage = update.message
                }
            }
            
            await MainActor.run {
                transcriptionResult = result.text
                accuracy = calculateAccuracy(original: currentSentence, transcribed: result.text)
                isTranscribing = false
            }
        } catch {
            print("Transcription error: \(error)")
            await MainActor.run {
                transcriptionError = error.localizedDescription
                isTranscribing = false
            }
        }
    }
    
    // MARK: - Accuracy Calculation
    
    private func calculateAccuracy(original: String, transcribed: String) -> Double {
        let distance = levenshteinDistance(original.lowercased(), transcribed.lowercased())
        let maxLength = max(original.count, transcribed.count)
        
        guard maxLength > 0 else { return 100.0 }
        
        let accuracy = (1.0 - Double(distance) / Double(maxLength)) * 100.0
        return max(0, min(100, accuracy))
    }
    
    /// Calculate Levenshtein distance (edit distance)
    private func levenshteinDistance(_ s1: String, _ s2: String) -> Int {
        let a = Array(s1)
        let b = Array(s2)
        
        var matrix = [[Int]](repeating: [Int](repeating: 0, count: b.count + 1), count: a.count + 1)
        
        for i in 0...a.count {
            matrix[i][0] = i
        }
        
        for j in 0...b.count {
            matrix[0][j] = j
        }
        
        for i in 1...a.count {
            for j in 1...b.count {
                let cost = a[i-1] == b[j-1] ? 0 : 1
                matrix[i][j] = min(
                    matrix[i-1][j] + 1,      // deletion
                    matrix[i][j-1] + 1,      // insertion
                    matrix[i-1][j-1] + cost  // substitution
                )
            }
        }
        
        return matrix[a.count][b.count]
    }
    
    // MARK: - Actions
    
    private func tryAgain() {
        // Reset and pick a new sentence
        transcriptionResult = nil
        accuracy = nil
        transcriptionError = nil
        providerProgressMessage = nil
        currentSentenceIndex = (currentSentenceIndex + 1) % testSentences.count
    }

    private func prepareInitialSelection() {
        if !hasLocalTestRoute, hasPublicTestRoute {
            if selectedTestMode != .publicAPI {
                selectedTestMode = .publicAPI
                return
            }
        }
        if !hasPublicTestRoute, hasLocalTestRoute {
            if selectedTestMode != .local {
                selectedTestMode = .local
                return
            }
        }
        if selectedLocalModelId.isEmpty {
            selectedLocalModelId = appState.loadedModelId ?? downloadedLocalModels.first?.id ?? ""
        }
        if selectedAPIModelId.isEmpty {
            selectedAPIModelId = usableAPIModels.first.map { "\($0.providerId):\($0.modelId)" } ?? ""
        }
        if selectedTestMode == .local, !selectedLocalModelId.isEmpty, appState.loadedModelId != selectedLocalModelId {
            loadLocalModel(selectedLocalModelId)
        }
        if selectedTestMode == .publicAPI {
            persistSelectedAPIModel()
        }
    }

    private func loadLocalModel(_ modelId: String) {
        guard !modelId.isEmpty, !isLoadingModel else { return }
        appState.setPreferredModel(modelId)

        Task {
            if appState.loadedModelId != modelId {
                isLoadingModel = true
                await appState.loadModel(modelId: modelId)
                isLoadingModel = false
            }
        }
    }

    private func persistSelectedAPIModel() {
        guard let selectedAPIModel else { return }
        UserDefaults.standard.set(selectedAPIModel.providerId, forKey: "selectedTranscriptionProviderId")
        UserDefaults.standard.set(selectedAPIModel.modelId, forKey: "selectedOpenAITranscriptionModel")
    }

    private func currentTestModelDescriptor() -> TranscriptionModelDescriptor {
        switch selectedTestMode {
        case .local:
            let model = downloadedLocalModels.first { $0.id == selectedLocalModelId }
            return TranscriptionModelDescriptor(
                id: selectedLocalModelId,
                displayName: model?.displayName ?? "Local model",
                providerId: TranscriptionProviderDescriptor.localBackend.id,
                source: .local
            )
        case .publicAPI:
            guard let model = selectedAPIModel else {
                return TranscriptionModelDescriptor.openAIModels[0]
            }
            return TranscriptionModelDescriptor(
                id: model.modelId,
                displayName: model.displayName,
                providerId: model.providerId,
                source: .api
            )
        }
    }

    private func currentTestProvider() -> any TranscriptionProvider {
        switch selectedTestMode {
        case .local:
            return LocalTranscriptionProvider(apiClient: apiClient)
        case .publicAPI:
            switch selectedAPIModel?.providerId {
            case TranscriptionProviderDescriptor.openRouter.id:
                return OpenRouterTranscriptionProvider()
            case TranscriptionProviderDescriptor.elevenLabs.id:
                return ElevenLabsTranscriptionProvider()
            default:
                return OpenAITranscriptionProvider()
            }
        }
    }
}

// MARK: - Test Results View

private enum TestTranscriptionMode: Equatable {
    case local
    case publicAPI

    var description: String {
        switch self {
        case .local:
            return "Local uses downloaded models. Audio stays on this Mac."
        case .publicAPI:
            return "Public sends audio to the selected provider for faster cloud transcription."
        }
    }

    var shortTitle: String {
        switch self {
        case .local:
            return "Local"
        case .publicAPI:
            return "Public"
        }
    }
}

private struct TestValidationState {
    let icon: String
    let color: Color
    let message: String
}

private struct TestProviderModeButton: View {
    let title: String
    let isSelected: Bool
    let isEnabled: Bool
    let tint: Color
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: 12, weight: .bold, design: .rounded))
                .foregroundStyle(isSelected ? .white : .secondary)
                .padding(.horizontal, 16)
                .frame(height: 30)
                .background(isSelected ? tint : .clear, in: Capsule())
        }
        .buttonStyle(.plain)
        .disabled(!isEnabled)
        .opacity(isEnabled || isSelected ? 1 : 0.42)
    }
}

private struct EmptyTestModelState: View {
    let icon: String
    let title: String
    let text: String

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label(title, systemImage: icon)
                .font(.system(size: 14, weight: .bold, design: .rounded))
                .foregroundStyle(.primary)

            Text(text)
                .font(.system(size: 12, weight: .medium))
                .lineSpacing(2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.white.opacity(0.045), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(.white.opacity(0.08), lineWidth: 1)
        )
    }
}

private struct InlineTestIssueView: View {
    let icon: String
    let title: String
    let text: String
    let color: Color

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: icon)
                .font(.system(size: 16, weight: .semibold))
                .foregroundStyle(color)
                .padding(.top, 1)

            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.system(size: 15, weight: .bold, design: .rounded))
                    .foregroundStyle(.primary)

                Text(text)
                    .font(.system(size: 13, weight: .medium))
                    .lineSpacing(2)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Spacer()
        }
        .padding(14)
        .background(color.opacity(0.08), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(color.opacity(0.18), lineWidth: 1)
        )
    }
}

private struct RecordingTestStateView: View {
    let duration: TimeInterval

    var body: some View {
        HStack(spacing: 12) {
            Circle()
                .fill(Color.red)
                .frame(width: 10, height: 10)
                .shadow(color: .red.opacity(0.55), radius: 6)

            Text("Recording")
                .font(.system(size: 15, weight: .bold, design: .rounded))
                .foregroundStyle(.primary)

            Text(String(format: "%.1fs", duration))
                .font(.system(size: 14, weight: .semibold, design: .monospaced))
                .foregroundStyle(.secondary)

            Spacer()

            Text("Pause to auto-stop")
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(.secondary)
        }
        .padding(14)
        .background(.red.opacity(0.07), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(.red.opacity(0.16), lineWidth: 1)
        )
    }
}

private struct TranscribingTestStateView: View {
    let message: String?

    var body: some View {
        HStack(spacing: 12) {
            ProgressView()
                .controlSize(.small)

            Text(message ?? "Transcribing...")
                .font(.system(size: 14, weight: .semibold, design: .rounded))
                .foregroundStyle(.secondary)

            Spacer()
        }
        .padding(14)
        .background(.white.opacity(0.045), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(.white.opacity(0.08), lineWidth: 1)
        )
    }
}

struct TestResultsView: View {
    let originalText: String
    let transcribedText: String
    let accuracy: Double
    let onTryAgain: () -> Void
    
    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack(alignment: .center, spacing: 12) {
                Image(systemName: accuracyIcon)
                    .font(.system(size: 24, weight: .bold))
                    .foregroundStyle(accuracyColor)
                    .frame(width: 42, height: 42)
                    .background(accuracyColor.opacity(0.14), in: Circle())

                VStack(alignment: .leading, spacing: 3) {
                    Text("\(Int(accuracy))% match")
                        .font(.system(size: 24, weight: .bold, design: .rounded))
                        .foregroundStyle(.primary)

                    Text(accuracyMessage)
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(.secondary)
                }

                Spacer()
            }

            VStack(alignment: .leading, spacing: 10) {
                ResultLine(label: "Original", text: originalText)
                ResultLine(label: "Heard", text: transcribedText)
            }

            HStack(spacing: 12) {
                Button(action: onTryAgain) {
                    Label("Try Again", systemImage: "arrow.clockwise")
                        .font(.system(size: 14, weight: .semibold, design: .rounded))
                        .frame(minWidth: 116)
                }
                .buttonStyle(.bordered)
                .controlSize(.large)
            }
        }
        .frame(maxWidth: 455, alignment: .leading)
    }
    
    private var accuracyIcon: String {
        if accuracy >= 90 {
            return "checkmark.circle.fill"
        } else if accuracy >= 70 {
            return "exclamationmark.circle.fill"
        } else {
            return "xmark.circle.fill"
        }
    }
    
    private var accuracyColor: Color {
        if accuracy >= 90 {
            return .green
        } else if accuracy >= 70 {
            return .orange
        } else {
            return .red
        }
    }
    
    private var accuracyMessage: String {
        if accuracy >= 90 {
            return "Setup is ready."
        } else if accuracy >= 70 {
            return "Usable result. Try once more if the room was noisy."
        } else {
            return "Try another recording or switch models."
        }
    }
}

private struct ResultLine: View {
    let label: String
    let text: String

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(label)
                .font(.system(size: 11, weight: .bold, design: .rounded))
                .tracking(1)
                .textCase(.uppercase)
                .foregroundStyle(.secondary)

            Text(text)
                .font(.system(size: 14, weight: .medium))
                .lineSpacing(2)
                .foregroundStyle(.primary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

// MARK: - Preview

#Preview {
    TestTranscriptionOnboardingView(onNext: {}, onBack: {})
        .environmentObject(AppState())
        .frame(width: 800, height: 750)
}
