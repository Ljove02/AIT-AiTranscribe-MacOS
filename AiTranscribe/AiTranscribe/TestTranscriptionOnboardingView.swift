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
    
    /// Go back TWO steps to reach the Models download page
    private func goToModelsPage() {
        onBack() // Go to Shortcuts page
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) {
            onBack() // Go to Models page
        }
    }
    
    var body: some View {
        VStack(spacing: 0) {
            // Header - FIXED HEIGHT
            VStack(spacing: 8) {
                Text("Test Your Setup")
                    .font(.system(size: 32, weight: .bold))
                
                Text("Read the sentence below to test your transcription")
                    .font(.title3)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
            }
            .frame(height: 100)
            .padding(.top, 20)
            
            // Content area - SCROLLABLE with remaining space
            ScrollView {
                VStack(spacing: 24) {
                    // Model selector (only downloaded models) - CENTERED
                    if !appState.availableModels.filter({ $0.downloaded }).isEmpty {
                        VStack(spacing: 8) {
                            Text("Select Model:")
                                .font(.subheadline)
                                .foregroundColor(.secondary)

                            HStack(spacing: 12) {
                                Picker("", selection: Binding(
                                    get: { appState.loadedModelId ?? "" },
                                    set: { modelId in
                                        if !modelId.isEmpty && !isLoadingModel {
                                            // Save as preferred model
                                            appState.setPreferredModel(modelId)

                                            Task {
                                                // Load the model if not already loaded
                                                if appState.loadedModelId != modelId {
                                                    isLoadingModel = true
                                                    await appState.loadModel(modelId: modelId)
                                                    isLoadingModel = false
                                                }
                                            }
                                        }
                                    }
                                )) {
                                    Text("Select a model").tag("")
                                    ForEach(appState.availableModels.filter { $0.downloaded }) { model in
                                        Text(model.displayName).tag(model.id)
                                    }
                                }
                                .pickerStyle(.menu)
                                .frame(width: 300)
                                .disabled(isLoadingModel)

                                // Loading indicator
                                if isLoadingModel {
                                    ProgressView()
                                        .controlSize(.small)
                                }
                            }

                            // Loading message
                            if isLoadingModel {
                                Text("Loading model...")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.horizontal, 40)
                    }
                    
                    // Test sentence display - BIGGER TEXT
                    VStack(spacing: 16) {
                        Text("Read this sentence:")
                            .font(.headline)
                            .foregroundColor(.secondary)
                        
                        Text(currentSentence)
                            .font(.system(size: 28, weight: .semibold))
                            .foregroundColor(.primary)
                            .multilineTextAlignment(.center)
                            .lineSpacing(6)
                            .padding(24)
                            .frame(maxWidth: .infinity)
                            .background(Color.accentColor.opacity(0.05))
                            .cornerRadius(12)
                            .overlay(
                                RoundedRectangle(cornerRadius: 12)
                                    .stroke(Color.accentColor.opacity(0.3), lineWidth: 2)
                            )
                    }
                    .padding(.horizontal, 40)
                    
                    // Recording indicator or results
                    if let result = transcriptionResult, let acc = accuracy {
                        // Show results
                        TestResultsView(
                            originalText: currentSentence,
                            transcribedText: result,
                            accuracy: acc,
                            onTryAgain: tryAgain,
                            onDownloadMore: goToModelsPage,
                            onContinue: onNext
                        )
                    } else if isRecording {
                        // Recording state
                        VStack(spacing: 12) {
                            HStack(spacing: 12) {
                                Circle()
                                    .fill(Color.red)
                                    .frame(width: 12, height: 12)
                                    .opacity(0.8)
                                    .animation(.easeInOut(duration: 0.8).repeatForever(autoreverses: true), value: isRecording)
                                
                                Text("Recording... Speak now!")
                                    .font(.headline)
                                    .foregroundColor(.red)
                            }
                            
                            Text(String(format: "%.1fs", recordingDuration))
                                .font(.system(.title2, design: .monospaced))
                                .foregroundColor(.secondary)
                            
                            Text("Listening for silence to auto-stop...")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        .padding(20)
                        .background(Color.red.opacity(0.05))
                        .cornerRadius(12)
                    } else if isTranscribing {
                        // Transcribing state
                        VStack(spacing: 12) {
                            ProgressView()
                                .controlSize(.large)
                            
                            Text("Transcribing...")
                                .font(.headline)
                                .foregroundColor(.secondary)
                        }
                        .padding(20)
                    }
                }
                .padding(.vertical, 20)
            }
            
            // Action Buttons - FIXED HEIGHT at bottom
            VStack(spacing: 0) {
                if transcriptionResult == nil {
                    VStack(spacing: 16) {
                        if !isRecording && !isTranscribing {
                            // Start button - full width
                            Button(action: startRecording) {
                                HStack {
                                    Image(systemName: "mic.circle.fill")
                                    Text("Start Recording")
                                        .font(.title3.weight(.semibold))
                                }
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 14)
                            }
                            .buttonStyle(.borderedProminent)
                            .controlSize(.large)
                            .disabled(appState.loadedModelId == nil || isLoadingModel)
                            .padding(.horizontal, 60)

                            if appState.loadedModelId == nil {
                                Text("Please select a model first")
                                    .font(.caption)
                                    .foregroundColor(.red)
                            } else if isLoadingModel {
                                Text("Please wait for model to load...")
                                    .font(.caption)
                                    .foregroundColor(.orange)
                            }
                            
                            // Back and Skip in one row
                            HStack(spacing: 16) {
                                // Back button - centered group
                                Button(action: onBack) {
                                    HStack {
                                        Image(systemName: "chevron.left")
                                        Text("Back")
                                    }
                                    .font(.body.weight(.medium))
                                }
                                .buttonStyle(.bordered)
                                .controlSize(.large)

                                // Skip button - next to back, centered as a pair
                                Button(action: onNext) {
                                    HStack {
                                        Text("Skip Test")
                                        Image(systemName: "chevron.right")
                                    }
                                    .font(.body.weight(.medium))
                                }
                                .buttonStyle(.bordered)
                                .controlSize(.large)
                            }
                            .frame(maxWidth: .infinity, alignment: .center)
                            .padding(.horizontal, 60)
                            
                        } else if isRecording {
                            // Manual stop button (backup)
                            Button(action: stopRecording) {
                                HStack {
                                    Image(systemName: "stop.circle.fill")
                                    Text("Stop Recording")
                                        .font(.title3.weight(.semibold))
                                }
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 14)
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.large)
                            .padding(.horizontal, 60)
                        }
                    }
                }
            }
            .frame(height: 140)
            .padding(.bottom, 20)
        }
        .padding()
    }
    
    // MARK: - Recording Control
    
    private func startRecording() {
        guard appState.loadedModelId != nil else { return }
        
        // Reset state
        transcriptionResult = nil
        accuracy = nil
        recordingDuration = 0
        isRecording = true
        
        // Start audio recorder
        if appState.audioRecorder.startRecording() {
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
        guard let data = appState.audioRecorder.stopRecording() else {
            isTranscribing = false
            return
        }
        
        audioData = data
        
        // Transcribe
        Task {
            await transcribeAudio(data)
        }
    }
    
    private func transcribeAudio(_ data: Data) async {
        do {
            let result = try await apiClient.transcribeAudioData(data)
            
            await MainActor.run {
                transcriptionResult = result.text
                accuracy = calculateAccuracy(original: currentSentence, transcribed: result.text)
                isTranscribing = false
            }
        } catch {
            print("Transcription error: \(error)")
            await MainActor.run {
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
        currentSentenceIndex = (currentSentenceIndex + 1) % testSentences.count
    }
}

// MARK: - Test Results View

struct TestResultsView: View {
    let originalText: String
    let transcribedText: String
    let accuracy: Double
    let onTryAgain: () -> Void
    let onDownloadMore: () -> Void
    let onContinue: () -> Void
    
    var body: some View {
        VStack(spacing: 20) {
            // Accuracy indicator
            VStack(spacing: 8) {
                HStack(spacing: 12) {
                    Image(systemName: accuracyIcon)
                        .font(.system(size: 40))
                        .foregroundColor(accuracyColor)
                    
                    Text("Accuracy: \(Int(accuracy))%")
                        .font(.system(size: 28, weight: .bold))
                        .foregroundColor(accuracyColor)
                }
                
                Text(accuracyMessage)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }
            .padding(.vertical, 16)
            
            Divider()
            
            // Comparison
            VStack(alignment: .leading, spacing: 16) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Original:")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Text(originalText)
                        .font(.body)
                        .padding(12)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color.gray.opacity(0.05))
                        .cornerRadius(8)
                }
                
                VStack(alignment: .leading, spacing: 4) {
                    Text("Transcribed:")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Text(transcribedText)
                        .font(.body)
                        .padding(12)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(accuracyColor.opacity(0.05))
                        .cornerRadius(8)
                }
            }
            .padding(.horizontal, 20)
            
            // Action Buttons
            HStack(spacing: 12) {
                Button(action: onTryAgain) {
                    HStack {
                        Image(systemName: "arrow.clockwise")
                        Text("Try Again")
                    }
                }
                .buttonStyle(.bordered)
                
                if accuracy < 70 {
                    Button(action: onDownloadMore) {
                        HStack {
                            Image(systemName: "arrow.left")
                            Text("Download More Models")
                        }
                    }
                    .buttonStyle(.bordered)
                }
                
                Button(action: onContinue) {
                    HStack {
                        Text("All Set!")
                        Image(systemName: "arrow.right")
                    }
                }
                .buttonStyle(.borderedProminent)
            }
            .padding(.top, 8)
        }
        .padding(24)
        .background(Color.gray.opacity(0.03))
        .cornerRadius(16)
        .padding(.horizontal, 40)
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
            return "Excellent! Your setup is working great."
        } else if accuracy >= 70 {
            return "Good, but you might want to try a different model or adjust your microphone."
        } else {
            return "Low accuracy. Consider downloading a different model or checking your microphone settings."
        }
    }
}

// MARK: - Preview

#Preview {
    TestTranscriptionOnboardingView(onNext: {}, onBack: {})
        .environmentObject(AppState())
        .frame(width: 800, height: 750)
}
