/*
 AppState.swift
 ==============

 This file manages the "state" of our app - all the data that can change.

 WHAT IS STATE?
 --------------
 State = data that can change over time
 Examples:
 - Is the model loaded? (changes when you click Load/Unload)
 - Is it recording? (changes when you start/stop)
 - What's the last transcription? (changes after each recording)

 WHY ObservableObject?
 ---------------------
 ObservableObject is a protocol that says "I have data that views care about".
 When @Published properties change, SwiftUI automatically updates the UI.

 It's like a newspaper subscription:
 - AppState is the newspaper
 - Views "subscribe" to it
 - When news changes (@Published), subscribers get notified
 */

import Foundation
import SwiftUI
import CoreAudio
import Combine  // Apple's framework for reactive programming

/*
 @MainActor means "run this on the main thread".

 Why? UI updates MUST happen on the main thread.
 If you update UI from a background thread, the app can crash.
 @MainActor guarantees safety.
 */
@MainActor
class AppState: ObservableObject {

    // =========================================================================
    // PUBLISHED PROPERTIES - Changes trigger UI updates
    // =========================================================================

    /// Is the ASR model loaded in memory?
    @Published var isModelLoaded: Bool = false

    /// Which model is currently loaded (e.g., "parakeet-v2", "parakeet-v3")
    @Published var loadedModelId: String? = nil

    /// Available models from backend
    @Published var availableModels: [ModelInfoResponse] = []

    /// Is the model currently loading?
    @Published var isModelLoading: Bool = false

    /// Which model is currently being loaded (for UI feedback)
    @Published var loadingModelId: String? = nil

    /// Which model is currently being downloaded
    @Published var downloadingModelId: String? = nil

    /// Download progress (0.0 to 1.0)
    @Published var downloadProgress: Double = 0.0

    /// Is currently recording audio?
    @Published var isRecording: Bool = false

    /// Current recording duration in seconds
    @Published var recordingDuration: Double = 0.0

    /// Current audio volume (0.0 to 1.0)
    @Published var currentVolume: Double = 0.0

    /// Last transcribed text
    @Published var lastTranscription: String = ""

    /// Is the backend server running?
    @Published var isServerConnected: Bool = false

    /// Status message to show user
    @Published var statusMessage: String = "Checking server..."

    /// Available microphones (from CoreAudio)
    @Published var audioDevices: [InputDevice] = []

    /// Selected microphone device ID (CoreAudio AudioDeviceID)
    @Published var selectedDeviceId: AudioDeviceID? = nil

    /// Cleanup closure for device change listener
    private var removeDeviceListener: (() -> Void)?

    /// Transcription history (stored locally in JSON)
    @Published var transcriptionHistory: [TranscriptionEntry] = []
    
    /// Pagination state for history
    @Published var historyPage: Int = 0
    @Published var hasMoreHistory: Bool = true
    @Published var isLoadingMoreHistory: Bool = false

    // =========================================================================
    // NEMO STATE
    // =========================================================================

    /// Whether NeMo is available in the current backend
    @Published var nemoAvailable: Bool = false

    /// NeMo version if available
    @Published var nemoVersion: String?

    /// Whether the NeMo venv exists on disk
    @Published var nemoVenvExists: Bool = false

    /// Current backend mode
    @Published var backendMode: String = "development"

    /// NeMo setup manager for installation UI
    let nemoSetupManager = NemoSetupManager()

    // =========================================================================
    // AUDIO DUCKING SETTINGS
    // =========================================================================
    
    /// Whether to mute/lower playback during recording
    /// Note: Using @AppStorage without @Published to prevent publish-during-update warnings
    @AppStorage("mutePlaybackDuringRecording") private var _mutePlaybackDuringRecording = false
    var mutePlaybackDuringRecording: Bool {
        get { _mutePlaybackDuringRecording }
        set { 
            _mutePlaybackDuringRecording = newValue
            objectWillChange.send()
        }
    }
    
    /// Mute mode: "mute" or "lower"
    /// Note: Using @AppStorage without @Published to prevent publish-during-update warnings
    @AppStorage("muteMode") private var _muteMode: String = "mute"
    var muteMode: String {
        get { _muteMode }
        set {
            _muteMode = newValue
            // Defer objectWillChange to next run loop to prevent publishing during view updates
            DispatchQueue.main.async { [weak self] in
                self?.objectWillChange.send()
            }
        }
    }
    
    /// Volume percentage when in lower mode (0-100)
    /// Note: Using @AppStorage without @Published to prevent publish-during-update warnings
    @AppStorage("duckVolume") private var _duckVolume: Double = 10.0
    var duckVolume: Double {
        get { _duckVolume }
        set {
            _duckVolume = newValue
            // Defer objectWillChange to next run loop to prevent publishing during view updates
            DispatchQueue.main.async { [weak self] in
                self?.objectWillChange.send()
            }
        }
    }

    // =========================================================================
    // API CLIENT & HISTORY MANAGER
    // =========================================================================

    /// The client that talks to our Python backend
    private let apiClient = APIClient()

    /// History manager for JSON storage
    private let historyManager = HistoryManager()

    /// Combine subscriptions for BackendManager state sync
    private var cancellables = Set<AnyCancellable>()

    /// Weak reference to BackendManager for state synchronization
    weak var backendManager: BackendManager?
    
    /// Audio manager for system volume control
    let audioManager = AudioManager.shared

    /// Timer for polling recording status
    private var statusTimer: Timer?

    /// Floating recording indicator
    let recordingIndicator = RecordingIndicatorController()

    /// Incremental text typer for real-time transcription output
    private let incrementalTyper = IncrementalTextTyper()

    /// Swift-native audio recorder (uses AVFoundation for proper permissions)
    let audioRecorder = AudioRecorder()


    // =========================================================================
    // INITIALIZATION
    // =========================================================================

    init() {
        // Setup history manager and load initial page
        setupHistory()
        
        // Setup recording indicator
        recordingIndicator.setup(appState: self)

        // Note: checkServerStatus() is now called from AiTranscribeApp
        // after the backend has started, to avoid connection errors

        // Check accessibility permissions if real-time output is enabled
        checkAccessibilityPermissionsIfNeeded()
    }

    // =========================================================================
    // BACKEND MANAGER BINDING
    // =========================================================================

    /// Bind to BackendManager to sync server state with AppState
    /// This ensures UI always reflects actual backend status
    func bindToBackendManager(_ manager: BackendManager) {
        print("AppState.bindToBackendManager() - START")
        self.backendManager = manager

        // Start listening for audio device changes (AirPods connect/disconnect)
        startDeviceChangeListener()

        // Persist device selection when user changes it via picker
        $selectedDeviceId
            .dropFirst() // Skip initial value
            .sink { [weak self] newId in
                guard let self = self, let newId = newId else { return }
                if let device = self.audioDevices.first(where: { $0.id == newId }) {
                    UserDefaults.standard.set(device.name, forKey: "preferredMicDeviceName")
                }
            }
            .store(in: &cancellables)

        // Sync isServerReady -> isServerConnected
        manager.$isServerReady
            .receive(on: DispatchQueue.main)
            .sink { [weak self] isReady in
                guard let self = self else { return }
                print("AppState: isServerReady changed to \(isReady)")
                self.isServerConnected = isReady
                if isReady && self.statusMessage == "Checking server..." {
                    self.statusMessage = self.isModelLoaded ? "Ready" : "Model not loaded"
                }
            }
            .store(in: &cancellables)

        // When server stops (after having been running), update status
        manager.$isRunning
            .dropFirst() // Skip initial value to avoid setting "Server stopped" at startup
            .receive(on: DispatchQueue.main)
            .sink { [weak self] isRunning in
                guard let self = self else { return }
                print("AppState: isRunning changed to \(isRunning)")
                if !isRunning {
                    self.isServerConnected = false
                    self.statusMessage = "Server stopped"
                }
            }
            .store(in: &cancellables)

        // Sync status message from BackendManager when it changes
        manager.$statusMessage
            .receive(on: DispatchQueue.main)
            .sink { [weak self] message in
                guard let self = self else { return }
                print("AppState: BackendManager.statusMessage changed to '\(message)'")
                // Only update if we're in a "waiting" state, not if model is already loaded
                if !self.isModelLoaded && !self.isServerConnected {
                    self.statusMessage = message
                }
            }
            .store(in: &cancellables)

        print("AppState.bindToBackendManager() - END, subscriptions count: \(cancellables.count)")
    }

    // =========================================================================
    // PREFERRED MODEL AUTO-LOAD
    // =========================================================================
    
    /// Load the user's preferred model on app start
    /// Priority: 1) Last selected model, 2) First downloaded model, 3) None
    @MainActor
    func loadPreferredModel() async {
        print("DEBUG: loadPreferredModel() called")
        
        // First, ensure we have the latest model list
        await fetchAvailableModels()
        
        print("DEBUG: availableModels count: \(availableModels.count)")
        for model in availableModels {
            print("DEBUG:   - \(model.id): downloaded=\(model.downloaded)")
        }
        
        // Migration: Check for old UserDefaults key and migrate to new one
        let oldKey = "lastLoadedModelId"
        let newKey = "preferredModelId"
        
        // If new key doesn't exist but old key does, migrate it
        if UserDefaults.standard.string(forKey: newKey) == nil,
           let oldValue = UserDefaults.standard.string(forKey: oldKey) {
            UserDefaults.standard.set(oldValue, forKey: newKey)
            print("DEBUG: Migrated preferred model from old key: \(oldValue)")
        }
        
        // Get preferred model ID from UserDefaults
        let preferredModelId = UserDefaults.standard.string(forKey: newKey)
        print("DEBUG: preferredModelId from UserDefaults: \(preferredModelId ?? "nil")")
        
        // Determine which model to load
        let modelIdToLoad: String?
        
        if let preferred = preferredModelId {
            print("DEBUG: Preferred model ID: \(preferred)")
            
            // Check if preferred model is downloaded
            if let model = availableModels.first(where: { $0.id == preferred }) {
                print("DEBUG: Found model in list: \(model.id), downloaded=\(model.downloaded)")
                
                if model.downloaded {
                    modelIdToLoad = preferred
                    print("DEBUG: Will load preferred model: \(preferred)")
                } else {
                    print("DEBUG: Preferred model NOT downloaded, looking for first downloaded model")
                    modelIdToLoad = availableModels.first(where: { $0.downloaded })?.id
                }
            } else {
                print("DEBUG: Preferred model NOT found in availableModels list")
                modelIdToLoad = availableModels.first(where: { $0.downloaded })?.id
            }
        } else {
            print("DEBUG: No preferred model set, using first downloaded model")
            modelIdToLoad = availableModels.first(where: { $0.downloaded })?.id
        }
        
        if let modelId = modelIdToLoad {
            print("DEBUG: Auto-loading preferred model: \(modelId)")
            await loadModel(modelId: modelId)
        } else {
            print("DEBUG: No model to load (modelIdToLoad is nil)")
        }
    }
    
    /// Save a model as the user's preferred model
    func setPreferredModel(_ modelId: String) {
        UserDefaults.standard.set(modelId, forKey: "preferredModelId")
        print("Saved preferred model: \(modelId)")
    }
    
    // =========================================================================
    // HISTORY MANAGEMENT (NEW - Uses HistoryManager)
    // =========================================================================
    
    /// Setup history - migrate from UserDefaults if needed and load first page
    private func setupHistory() {
        Task {
            // Migrate from UserDefaults if needed
            let migrated = await historyManager.migrateFromUserDefaults()
            
            if migrated {
                print("Migrated history from UserDefaults to JSON")
            }
            
            // Load first page
            await loadHistoryPage(page: 0)
        }
    }
    
    /// Load a specific page of history
    func loadHistoryPage(page: Int) async {
        let entries = await historyManager.loadPage(page: page)
        
        if page == 0 {
            // First page - replace all
            transcriptionHistory = entries
        } else {
            // Subsequent page - append
            transcriptionHistory.append(contentsOf: entries)
        }
        
        // Update pagination state
        historyPage = page
        hasMoreHistory = entries.count == 20  // If we got a full page, there might be more
        isLoadingMoreHistory = false
    }
    
    /// Load next page (for infinite scroll)
    func loadMoreHistory() async {
        guard !isLoadingMoreHistory, hasMoreHistory else { return }
        
        isLoadingMoreHistory = true
        await loadHistoryPage(page: historyPage + 1)
    }
    
    /// Refresh history (reload from beginning)
    func refreshHistory() async {
        await loadHistoryPage(page: 0)
    }
    
    /// Add entry to history (saves to JSON)
    func addToHistory(_ entry: TranscriptionEntry) {
        // Add to beginning (newest first)
        transcriptionHistory.insert(entry, at: 0)
        
        // Save to JSON via HistoryManager
        Task {
            await historyManager.save(entry)
        }
    }
    
    /// Delete entry from history
    func deleteFromHistory(_ entry: TranscriptionEntry) {
        transcriptionHistory.removeAll { $0.id == entry.id }
        
        // Delete from JSON via HistoryManager
        Task {
            await historyManager.delete(id: entry.id)
        }
    }
    
    /// Delete multiple entries
    func deleteFromHistory(ids: Set<UUID>) {
        transcriptionHistory.removeAll { ids.contains($0.id) }
        
        // Delete from JSON via HistoryManager
        Task {
            await historyManager.delete(ids: ids)
        }
    }
    
    /// Clear all history
    func clearHistory() {
        transcriptionHistory = []
        
        // Clear from JSON via HistoryManager
        Task {
            await historyManager.clearAll()
        }
    }

    /// Check if accessibility permissions are needed and prompt if not granted
    func checkAccessibilityPermissionsIfNeeded() {
        let realTimeOutput = UserDefaults.standard.bool(forKey: "realTimeOutput")
        let autoPaste = UserDefaults.standard.bool(forKey: "autoPasteAfterTranscription")

        if (realTimeOutput || autoPaste) && !KeyboardSimulator.shared.hasAccessibilityPermission {
            // Request permission (shows system dialog)
            KeyboardSimulator.shared.requestAccessibilityPermission()
        }
    }

    /// Check if accessibility permission is currently granted
    var hasAccessibilityPermission: Bool {
        return KeyboardSimulator.shared.hasAccessibilityPermission
    }


    // =========================================================================
    // SERVER COMMUNICATION
    // =========================================================================

    /// Check if the backend server is running and get status
    /// Retries with exponential backoff for up to 60 seconds
    func checkServerStatus() async {
        let maxAttempts = 30
        var delay: UInt64 = 1_000_000_000  // 1 second

        for attempt in 1...maxAttempts {
            do {
                let status = try await apiClient.getStatus()
                isServerConnected = true
                isModelLoaded = status.modelLoaded
                loadedModelId = status.modelId
                statusMessage = status.modelLoaded ? "Ready" : "Model not loaded"

                // Fetch available models
                await fetchAvailableModels()

                // Also fetch available devices
                await fetchAudioDevices()

                print("Server connected after \(attempt) attempt(s)")
                return
            } catch {
                isServerConnected = false

                if attempt < maxAttempts {
                    statusMessage = "Waiting for server... (\(attempt)/\(maxAttempts))"
                    try? await Task.sleep(nanoseconds: delay)
                    delay = min(delay * 2, 2_000_000_000)
                } else {
                    statusMessage = "Server not running"
                }
            }
        }
    }

    /// Fetch available models from backend
    func fetchAvailableModels() async {
        do {
            availableModels = try await apiClient.listModels()
        } catch {
            print("Failed to fetch models: \(error)")
        }
    }

    /// Fetch available audio input devices using CoreAudio
    func fetchAudioDevices() async {
        audioDevices = AudioRecorder.listInputDevices()

        // Restore persisted device by name, or fall back to default
        let preferredName = UserDefaults.standard.string(forKey: "preferredMicDeviceName")
        if let preferredName = preferredName,
           let match = audioDevices.first(where: { $0.name == preferredName }) {
            selectedDeviceId = match.id
        } else if selectedDeviceId == nil {
            selectedDeviceId = audioDevices.first(where: { $0.isDefault })?.id
        }
    }

    /// Start listening for audio device changes (AirPods connect/disconnect, etc.)
    func startDeviceChangeListener() {
        removeDeviceListener = AudioRecorder.onDeviceChange { [weak self] in
            guard let self = self else { return }
            self.handleDeviceChange()
        }
    }

    /// Handle audio device change (device added/removed or default changed)
    private func handleDeviceChange() {
        let oldDefault = audioDevices.first(where: { $0.isDefault })?.name
        audioDevices = AudioRecorder.listInputDevices()
        let newDefault = audioDevices.first(where: { $0.isDefault })?.name

        // If our selected device disappeared, fall back to new default
        if let selectedId = selectedDeviceId,
           !audioDevices.contains(where: { $0.id == selectedId }) {
            selectedDeviceId = audioDevices.first(where: { $0.isDefault })?.id
            print("Selected device disconnected, switched to default: \(newDefault ?? "unknown")")
        }

        // Notify user if default changed and they have a preferred device
        let preferredName = UserDefaults.standard.string(forKey: "preferredMicDeviceName")
        if let oldDefault = oldDefault, let newDefault = newDefault,
           oldDefault != newDefault, preferredName != nil {
            let selectedName = audioDevices.first(where: { $0.id == selectedDeviceId })?.name ?? "default"
            sendDeviceChangeNotification(newDefault: newDefault, usingDevice: selectedName)
        }
    }

    /// Send a macOS notification about device change
    private func sendDeviceChangeNotification(newDefault: String, usingDevice: String) {
        let notification = NSUserNotification()
        notification.title = "Microphone Changed"
        notification.informativeText = "System switched to \(newDefault). AiTranscribe is using \(usingDevice)."
        notification.soundName = nil
        NSUserNotificationCenter.default.deliver(notification)
    }

    /// Persist device selection when user picks a mic
    func selectMicrophone(_ device: InputDevice) {
        selectedDeviceId = device.id
        UserDefaults.standard.set(device.name, forKey: "preferredMicDeviceName")
    }

    /// Load a specific ASR model
    func loadModel(modelId: String = "parakeet-v2") async {
        isModelLoading = true
        loadingModelId = modelId  // Track which model is being loaded
        statusMessage = "Loading model..."
        do {
            let response = try await apiClient.loadModel(modelId: modelId)
            if response.success {
                isModelLoaded = true
                loadedModelId = modelId
                statusMessage = "Ready"
                // Save as preferred model for auto-load on restart
                setPreferredModel(modelId)
            } else {
                statusMessage = response.message
            }
        } catch {
            statusMessage = "Failed to load model"
            print("Load error: \(error)")
        }
        isModelLoading = false
        loadingModelId = nil  // Clear loading state
    }

    /// Unload the ASR model
    func unloadModel() async {
        isModelLoading = true
        statusMessage = "Unloading model..."
        do {
            let response = try await apiClient.unloadModel()
            if response.success {
                isModelLoaded = false
                loadedModelId = nil
                statusMessage = "Model unloaded"
            }
        } catch {
            statusMessage = "Failed to unload"
            print("Unload error: \(error)")
        }
        isModelLoading = false
    }

    /// Download a model (Whisper models only)
    /// Returns true if successful, false if failed
    func downloadModel(modelId: String) async -> Bool {
        downloadingModelId = modelId
        downloadProgress = 0.0
        statusMessage = "Downloading model..."
        
        var downloadError: String?
        
        do {
            try await apiClient.downloadModel(modelId: modelId) { event in
                switch event.status {
                case "downloading":
                    self.downloadProgress = event.progress ?? 0.0
                    if let downloaded = event.downloadedMB, let total = event.totalMB {
                        self.statusMessage = "Downloading... \(downloaded)/\(total) MB"
                    }
                case "verifying":
                    self.statusMessage = "Verifying download..."
                case "complete":
                    self.statusMessage = "Download complete"
                    self.downloadProgress = 1.0
                    Task {
                        await self.fetchAvailableModels()
                    }
                case "error":
                    self.statusMessage = event.message ?? "Download failed"
                    downloadError = event.message ?? "Download failed"
                default:
                    break
                }
            }
        } catch {
            statusMessage = "Download failed: \(error.localizedDescription)"
            downloadError = error.localizedDescription
        }

        downloadingModelId = nil
        downloadProgress = 0.0
        
        if let error = downloadError {
            statusMessage = "Download failed: \(error)"
            return false
        }
        
        return true
    }

    /// Delete a downloaded model
    func deleteModel(modelId: String) async {
        statusMessage = "Deleting model..."

        do {
            let response = try await apiClient.deleteModel(modelId: modelId)
            if response.success {
                statusMessage = "Model deleted"
                // Refresh model list to update download status
                await fetchAvailableModels()
            } else {
                statusMessage = response.message
            }
        } catch {
            statusMessage = "Failed to delete model"
            print("Delete error: \(error)")
        }
    }


    // =========================================================================
    // NEMO STATUS
    // =========================================================================

    /// Check NeMo availability from the backend
    func checkNemoStatus() async {
        do {
            let status = try await apiClient.getNemoStatus()
            nemoAvailable = status.nemoAvailable
            nemoVersion = status.nemoVersion
            backendMode = status.backendMode
        } catch {
            // If the endpoint doesn't exist (older backend), assume NeMo is available
            // This maintains backwards compatibility
            print("Failed to check NeMo status: \(error)")
        }

        // Also check if venv exists locally
        refreshNemoVenvState()
    }

    /// Refresh local NeMo venv state (without backend call)
    func refreshNemoVenvState() {
        let exists = nemoSetupManager.checkNemoVenvExists()
        // Defer publishing to next run loop to prevent publishing during view updates
        DispatchQueue.main.async { [weak self] in
            self?.nemoVenvExists = exists
        }
    }

    /// Check if a specific model can be loaded (has required dependencies)
    func canLoadModel(_ model: ModelInfoResponse) -> Bool {
        if model.nemoRequired {
            return nemoAvailable
        }
        return true
    }


    // =========================================================================
    // RECORDING
    // =========================================================================

    /// Track if we're in streaming mode
    private var isStreamingMode: Bool = false

    /// Task for streaming session
    private var streamingTask: Task<Void, Never>?

    /// Start recording audio
    func startRecording() async {
        guard isModelLoaded else {
            statusMessage = "Load model first"
            return
        }
        
        // Duck audio if setting is enabled - use session-based API
        if mutePlaybackDuringRecording {
            // Mute mode = 99.99% reduction (effectively mute)
            // Lower mode = duckVolume as the reduction percentage
            let reductionPercentage = (muteMode == "mute") ? 99.99 : duckVolume
            audioManager.startDuckingSession(reductionPercentage: reductionPercentage)
        }
        
        // Check if streaming is enabled
        let streamingEnabled = UserDefaults.standard.bool(forKey: "realTimeStreaming")

        if streamingEnabled && KeyboardSimulator.shared.hasAccessibilityPermission {
            await startStreamingRecording()
        } else {
            await startNormalRecording()
        }
    }

    /// Start normal (non-streaming) recording
    /// Uses Swift's AVFoundation for proper microphone permission handling
    private func startNormalRecording() async {
        // Check microphone permission
        if !AudioRecorder.hasMicrophonePermission {
            AudioRecorder.requestMicrophonePermission { [weak self] granted in
                guard let self = self else { return }
                if granted {
                    Task {
                        await self.startNormalRecording()
                    }
                } else {
                    self.statusMessage = "Microphone permission denied"
                }
            }
            return
        }

        // Start Swift-native recording with selected device
        if audioRecorder.startRecording(deviceId: selectedDeviceId) {
            isRecording = true
            isStreamingMode = false
            statusMessage = "Recording..."

            // Play start sound
            SoundManager.shared.playStartSound()

            // Show floating indicator
            recordingIndicator.show()

            // Start polling audio recorder for volume updates
            startVolumePolling()
        } else {
            statusMessage = "Failed to start recording"
        }
    }

    /// Poll audio recorder for volume updates
    private func startVolumePolling() {
        statusTimer?.invalidate()
        statusTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            guard let self = self, self.audioRecorder.isRecording else { return }
            DispatchQueue.main.async {
                self.currentVolume = Double(self.audioRecorder.currentVolume)
                self.recordingDuration = self.audioRecorder.duration
            }
        }
    }

    /// Start streaming recording with real-time transcription
    private func startStreamingRecording() async {
        isRecording = true
        isStreamingMode = true
        statusMessage = "Recording (streaming)..."

        // Reset incremental typer
        incrementalTyper.reset()

        // Play start sound
        SoundManager.shared.playStartSound()

        // Show floating indicator
        recordingIndicator.show()

        // Start streaming in background task
        streamingTask = Task {
            do {
                try await apiClient.startStreamingRecording(
                    onPartial: { [weak self] text in
                        guard let self = self else { return }
                        // Type incrementally as text comes in
                        self.incrementalTyper.typeIncremental(text)
                        self.lastTranscription = text
                    },
                    onFinal: { [weak self] text in
                        guard let self = self else { return }
                        self.lastTranscription = text

                        // ALWAYS copy to clipboard (default behavior)
                        self.copyToClipboard(text)

                        // Save to history
                        let entry = TranscriptionEntry(
                            text: text,
                            duration: 0,
                            timestamp: Date(),
                            modelId: self.loadedModelId ?? "streaming",
                            modelName: self.availableModels.first { $0.id == self.loadedModelId }?.displayName ?? "Streaming"
                        )
                        self.addToHistory(entry)
                    }
                )
            } catch {
                print("Streaming error: \(error)")
            }

            // Streaming ended
            await MainActor.run {
                self.isRecording = false
                self.isStreamingMode = false
                self.statusMessage = "Ready"
                self.recordingIndicator.hide()
                
                // End ducking session
                if self.mutePlaybackDuringRecording {
                    self.audioManager.endDuckingSession()
                }
            }
        }
    }

    /// Stop recording and get transcription
    func stopRecording() async {
        stopStatusPolling()

        guard isRecording else { return }

        // Play stop sound
        SoundManager.shared.playStopSound()

        if isStreamingMode {
            await stopStreamingRecording()
        } else {
            await stopNormalRecording()
        }
    }

    /// Stop normal recording and transcribe
    /// Uses Swift AudioRecorder to get audio data, then sends to backend for transcription
    private func stopNormalRecording() async {
        // Show transcribing state on indicator (dots rotate)
        recordingIndicator.setTranscribing(true)

        statusMessage = "Transcribing..."

        // Reset incremental typer for next session
        incrementalTyper.reset()

        // Stop Swift AudioRecorder and get WAV data
        guard let audioData = audioRecorder.stopRecording() else {
            isRecording = false
            statusMessage = "No audio recorded"
            recordingIndicator.hide()
            
            // End ducking session (error path)
            if mutePlaybackDuringRecording {
                audioManager.endDuckingSession()
            }
            return
        }

        let recordedDuration = audioRecorder.duration
        
        // Restore volume IMMEDIATELY when recording stops (not during transcription)
        if mutePlaybackDuringRecording {
            audioManager.endDuckingSession()
        }

        do {
            // Send audio data to backend for transcription with progress updates
            let result = try await apiClient.transcribeAudioDataWithProgress(audioData) { progress, elapsed in
                let percent = Int(progress * 100)
                if percent > 0 {
                    self.statusMessage = "Transcribing... \(percent)%"
                } else {
                    self.statusMessage = "Transcribing... (\(Int(elapsed))s)"
                }
            }
            isRecording = false
            lastTranscription = result.text
            statusMessage = "Ready"

            // Hide indicator after transcription completes
            recordingIndicator.hide()

            // Save to history
            let entry = TranscriptionEntry(
                text: result.text,
                duration: result.durationSeconds > 0 ? result.durationSeconds : recordedDuration,
                timestamp: Date(),
                modelId: self.loadedModelId ?? "unknown",
                modelName: self.availableModels.first { $0.id == self.loadedModelId }?.displayName ?? "Unknown"
            )
            addToHistory(entry)

            // ALWAYS copy to clipboard (default behavior)
            copyToClipboard(result.text)

            // Auto-paste if enabled (simulates Cmd+V)
            let autoPaste = UserDefaults.standard.bool(forKey: "autoPasteAfterTranscription")
            if autoPaste && hasAccessibilityPermission {
                // Small delay to ensure clipboard is ready
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                    KeyboardSimulator.shared.pasteAsync()
                }
            }

        } catch {
            isRecording = false
            if "\(error)".contains("timed out") || "\(error)".contains("timeout") {
                statusMessage = "Transcription timed out - try shorter audio"
            } else {
                statusMessage = "Transcription failed"
            }
            recordingIndicator.hide()
            print("Stop recording error: \(error)")
        }
    }

    /// Stop streaming recording
    private func stopStreamingRecording() async {
        statusMessage = "Finishing..."
        
        do {
            _ = try await apiClient.stopStreamingRecording()
            // The streaming task will handle cleanup when it receives the final event
        } catch {
            print("Stop streaming error: \(error)")
            // Force cleanup
            streamingTask?.cancel()
            isRecording = false
            isStreamingMode = false
            statusMessage = "Ready"
            recordingIndicator.hide()
            
            // End ducking session
            if mutePlaybackDuringRecording {
                audioManager.endDuckingSession()
            }
        }
    }

    /// Cancel recording without transcribing
    func cancelRecording() async {
        stopStatusPolling()

        // Hide floating indicator
        recordingIndicator.hide()
        
        // End ducking session
        if mutePlaybackDuringRecording {
            audioManager.endDuckingSession()
        }

        if isStreamingMode {
            // Cancel streaming recording via backend
            do {
                _ = try await apiClient.cancelRecording()
            } catch {
                print("Cancel streaming error: \(error)")
            }
            streamingTask?.cancel()
            isStreamingMode = false
        } else {
            // Cancel Swift AudioRecorder (no backend call needed)
            audioRecorder.cancelRecording()
        }

        isRecording = false
        statusMessage = "Cancelled"
    }

    /// Poll recording status for UI updates (volume, duration)
    /// NOTE: Currently disabled to prevent cascading re-renders in SettingsView.
    /// The RecordingIndicator uses its own local animation timer instead.
    private func startStatusPolling() {
        // DISABLED: The polling was updating @Published properties every 0.1s,
        // which caused all views with @EnvironmentObject appState to re-render,
        // including SettingsView tabs (causing the icon "shivering" bug).
        //
        // If we need to re-enable this for actual volume visualization:
        // 1. Create a separate ObservableObject for recording stats
        // 2. Only subscribe to it from views that need it
        // 3. Don't pass it through @EnvironmentObject to avoid broad re-renders
    }

    private func stopStatusPolling() {
        statusTimer?.invalidate()
        statusTimer = nil
        recordingDuration = 0.0
        currentVolume = 0.0
    }

    private func pollRecordingStatus() async {
        // Only poll if still recording
        guard isRecording else { return }

        // NOTE: We no longer update recordingDuration and currentVolume here
        // because they were causing cascading re-renders across all views
        // that use @EnvironmentObject appState (including SettingsView).
        //
        // The RecordingIndicator now uses its own local animation timer,
        // and the MenuBarView no longer displays duration or volume.
        //
        // If we need these values in the future, consider:
        // 1. Using a separate ObservableObject just for recording stats
        // 2. Using Combine publishers with specific subscribers
        // 3. Only updating when the Settings window is closed
    }


    // =========================================================================
    // CLIPBOARD
    // =========================================================================

    /// Copy text to the system clipboard
    func copyToClipboard(_ text: String) {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(text, forType: .string)
    }


    // =========================================================================
    // OLD HISTORY MANAGEMENT (REMOVED - Now uses HistoryManager above)
    // =========================================================================
    // The old UserDefaults-based history methods have been replaced by
    // the JSON-based HistoryManager methods defined earlier in this file.
}


// =============================================================================
// DATA MODELS
// =============================================================================

/// Represents an audio input device (microphone)
struct AudioDevice: Identifiable, Codable {
    let id: Int
    let name: String
    let channels: Int
    let sampleRate: Double
    let isDefault: Bool

    enum CodingKeys: String, CodingKey {
        case id, name, channels
        case sampleRate = "sample_rate"
        case isDefault = "is_default"
    }
}
