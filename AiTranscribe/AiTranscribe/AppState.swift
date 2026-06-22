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

private struct PolishApplicationResult {
    let text: String
    let estimatedCostUSD: Double?
}

/*
 @MainActor means "run this on the main thread".

 Why? UI updates MUST happen on the main thread.
 If you update UI from a background thread, the app can crash.
 @MainActor guarantees safety.
 */
@MainActor
class AppState: ObservableObject {

    /// Shared singleton so AppDelegate can access it at launch (for onboarding)
    static let shared = AppState()

    // =========================================================================
    // PUBLISHED PROPERTIES - Changes trigger UI updates
    // =========================================================================

    /// Is the ASR model loaded in memory?
    @Published var isModelLoaded: Bool = false

    /// Which model is currently loaded (e.g., "parakeet-v2", "parakeet-v3")
    @Published var loadedModelId: String? = nil

    /// Available models from backend
    @Published var availableModels: [ModelInfoResponse] = []

    /// Available summarization models from backend
    @Published var availableSummaryModels: [SummaryModelInfoResponse] = []

    /// User-editable summary presets (built-ins + custom)
    @Published var summaryPresetDefinitions: [SummaryPresetDefinition] = SummaryPresetLibrary.load()

    /// User-editable polish presets used after quick transcription
    @Published var polishPresets: [PolishPreset] = PolishPresetStore.load()

    /// Selected polish preset for quick transcription. Nil means raw transcript only.
    @Published var selectedPolishPresetId: String? = UserDefaults.standard.string(forKey: "selectedPolishPresetId")

    /// Is the model currently loading?
    @Published var isModelLoading: Bool = false

    /// Which model is currently being loaded (for UI feedback)
    @Published var loadingModelId: String? = nil

    /// Which model is currently being downloaded
    @Published var downloadingModelId: String? = nil

    /// Download progress (0.0 to 1.0)
    @Published var downloadProgress: Double = 0.0

    /// Which summary model is currently being downloaded
    @Published var downloadingSummaryModelId: String? = nil

    /// Summary model download progress (0.0 to 1.0)
    @Published var summaryDownloadProgress: Double = 0.0

    /// Is currently recording audio?
    @Published var isRecording: Bool = false

    /// True while quick recording is starting/stopping; prevents rapid double actions from wedging the recorder.
    @Published var isQuickRecordingTransitioning: Bool = false

    /// True after quick recording stops while STT/polish/history output is still being produced.
    @Published var isQuickTranscriptionInFlight: Bool = false

    /// Set by SessionManager to prevent concurrent recording conflicts
    var isSessionRecordingActive: Bool = false

    /// Current recording duration in seconds
    @Published var recordingDuration: Double = 0.0

    /// Current audio volume (0.0 to 1.0)
    @Published var currentVolume: Double = 0.0

    /// Last transcribed text
    @Published var lastTranscription: String = ""

    /// Metadata for the most recent transcription, used by upcoming dashboard/provider work
    @Published var lastTranscriptionUsage: TranscriptionUsageMetadata?

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
    @Published var totalHistoryCount: Int = 0

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

    /// Summary runtime installer
    let summarySetupManager = SummarySetupManager()

    /// Whether the summary runtime exists on disk
    @Published var summaryRuntimeInstalled: Bool = false

    /// Whether the summary runtime is ready for use
    @Published var summaryRuntimeReady: Bool = false

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
    let apiClient = APIClient()

    /// Providers currently available to the app.
    var transcriptionProviders: [TranscriptionProviderDescriptor] {
        [.localBackend, .openAI, .openRouter, .elevenLabs]
    }

    private lazy var localTranscriptionProvider = LocalTranscriptionProvider(apiClient: apiClient)
    private lazy var openAITranscriptionProvider = OpenAITranscriptionProvider()
    private lazy var openRouterTranscriptionProvider = OpenRouterTranscriptionProvider()
    private lazy var elevenLabsTranscriptionProvider = ElevenLabsTranscriptionProvider()

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

    /// Timer for auto-unloading model after idle period
    private var idleUnloadTimer: Timer?
    /// Idle timeout before model is auto-unloaded (2 minutes)
    private let idleTimeoutSeconds: TimeInterval = 120
    /// Tracks whether we triggered an auto-load for the current recording session
    private var isAutoLoadingModel: Bool = false

    /// Floating recording indicator
    let recordingIndicator = RecordingIndicatorController()

    /// Incremental text typer for real-time transcription output
    private let incrementalTyper = IncrementalTextTyper()

    /// Swift-native audio recorder (uses AVFoundation for proper permissions)
    let audioRecorder = AudioRecorder()

    /// Processes sealed basic-recording chunks while recording continues.
    private var rollingBasicBatchSession: RollingBasicBatchSession?
    private var rollingAPIBatchSession: RollingAPIBatchSession?


    // =========================================================================
    // INITIALIZATION
    // =========================================================================

    init() {
        // Setup history manager and load initial page
        setupHistory()
        
        // Setup recording indicator
        recordingIndicator.setup(appState: self)

        nemoVenvExists = nemoSetupManager.checkNemoVenvExists()
        summaryRuntimeInstalled = summarySetupManager.checkSummaryVenvExists()
        summaryPresetDefinitions = SummaryPresetLibrary.load()

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

        // Refresh total count
        totalHistoryCount = await historyManager.getTotalCount()
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
        totalHistoryCount += 1

        // Save to JSON + index the new entry
        historyManager.indexEntry(entry)
        Task {
            await historyManager.save(entry)
        }
    }
    
    /// Delete entry from history
    func deleteFromHistory(_ entry: TranscriptionEntry) {
        transcriptionHistory.removeAll { $0.id == entry.id }
        totalHistoryCount = max(0, totalHistoryCount - 1)

        historyManager.removeFromIndex(ids: [entry.id])
        Task {
            await historyManager.delete(id: entry.id)
        }
    }

    /// Delete multiple entries
    func deleteFromHistory(ids: Set<UUID>) {
        let removedCount = transcriptionHistory.filter { ids.contains($0.id) }.count
        transcriptionHistory.removeAll { ids.contains($0.id) }
        totalHistoryCount = max(0, totalHistoryCount - removedCount)

        historyManager.removeFromIndex(ids: ids)
        Task {
            await historyManager.delete(ids: ids)
        }
    }

    /// Whether the semantic search index is built
    var isHistoryIndexed: Bool { historyManager.isIndexed }

    /// Build the semantic search index (user-triggered)
    func buildHistoryIndex() async {
        await historyManager.buildSearchIndex()
    }

    /// Search history — uses semantic search if indexed, text search otherwise
    func searchHistory(query: String) async -> [TranscriptionEntry] {
        if historyManager.isIndexed {
            return historyManager.semanticSearch(query: query)
        } else {
            return await historyManager.textSearch(query: query)
        }
    }

    /// Load all history entries (for stats computation)
    func loadAllHistoryEntries() async -> [TranscriptionEntry] {
        await historyManager.loadAllEntries()
    }

    /// Clear all history
    func clearHistory() {
        transcriptionHistory = []
        totalHistoryCount = 0

        historyManager.clearIndex()
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

    private func polishTranscript(_ transcript: String, using preset: PolishPreset) async throws -> PolishApplicationResult {
        let resolvedModel = resolvedLanguageModelForPolish(preset)
        let startedAt = Date()
        print("AppState: polish start provider=\(resolvedModel.providerName) model=\(resolvedModel.modelId) preset=\(preset.displayName) inputWords=\(transcript.split(separator: " ").count)")
        let apiKey = try resolvedAPIKeyForPolish(providerId: resolvedModel.providerId)
        let response: OpenAILanguageModelResponse

        if resolvedModel.providerId == TranscriptionProviderDescriptor.openRouter.id {
            response = try await OpenRouterLanguageModelClient().polishText(
                transcript: transcript,
                preset: preset,
                modelId: resolvedModel.modelId,
                apiKey: apiKey,
                providerOnly: resolvedModel.openRouterProviderOnly
            )
        } else {
            response = try await OpenAILanguageModelClient().polishText(
                transcript: transcript,
                preset: preset,
                modelId: resolvedModel.modelId,
                apiKey: apiKey
            )
        }

        let estimatedCost = resolvedModel.estimatedCost(
            inputTokens: response.inputTokens,
            outputTokens: response.outputTokens
        )
        print(String(
            format: "AppState: polish complete provider=%@ model=%@ elapsed=%.2fs responseElapsed=%.2fs inputTokens=%d outputTokens=%d estimatedCost=$%.6f",
            resolvedModel.providerName,
            resolvedModel.modelId,
            Date().timeIntervalSince(startedAt),
            response.processingTimeSeconds,
            response.inputTokens ?? 0,
            response.outputTokens ?? 0,
            estimatedCost ?? 0
        ))

        return PolishApplicationResult(
            text: response.text,
            estimatedCostUSD: estimatedCost
        )
    }

    func testPolishPreset(_ preset: PolishPreset) async throws -> String {
        let sample = "Hey so can you turn this messy transcript into a clean useful result, keep the important constraints, and make it easy to copy into another AI tool."
        let result = try await polishTranscript(sample, using: preset)
        let trimmed = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { throw OpenAILanguageModelError.emptyOutput }
        return trimmed
    }

    private func resolvedLanguageModelForPolish(_ preset: PolishPreset) -> APILanguageModel {
        let models = APILanguageModelStore.load()
        let selectedModelId = UserDefaults.standard.string(forKey: "selectedOpenAILanguageModel")

        if preset.isBuiltIn,
           let selectedModelId,
           let selectedModel = models.first(where: { $0.modelId == selectedModelId }) {
            return routedLanguageModelForPolish(selectedModel)
        }

        if let presetModel = models.first(where: { $0.providerId == preset.modelProviderId && $0.modelId == preset.modelId }) {
            return routedLanguageModelForPolish(presetModel)
        }

        if let selectedModelId,
           let selectedModel = models.first(where: { $0.modelId == selectedModelId }) {
            return routedLanguageModelForPolish(selectedModel)
        }

        if let selectedModelId,
           selectedModelId.contains("/"),
           APIKeychainStore.hasAPIKey(providerId: TranscriptionProviderDescriptor.openRouter.id) {
            return APILanguageModel(
                providerId: TranscriptionProviderDescriptor.openRouter.id,
                providerName: TranscriptionProviderDescriptor.openRouter.displayName,
                modelId: selectedModelId,
                displayName: selectedModelId
            )
        }

        return routedLanguageModelForPolish(models.first ?? APILanguageModelStore.defaultModels[0])
    }

    private func routedLanguageModelForPolish(_ model: APILanguageModel) -> APILanguageModel {
        guard model.providerId == TranscriptionProviderDescriptor.openAI.id,
              model.modelId.contains("/"),
              APIKeychainStore.hasAPIKey(providerId: TranscriptionProviderDescriptor.openRouter.id) else {
            return model
        }

        return APILanguageModel(
            id: model.id,
            providerId: TranscriptionProviderDescriptor.openRouter.id,
            providerName: TranscriptionProviderDescriptor.openRouter.displayName,
            modelId: model.modelId,
            displayName: model.displayName,
            inputCostPerMillionTokensUSD: model.inputCostPerMillionTokensUSD,
            outputCostPerMillionTokensUSD: model.outputCostPerMillionTokensUSD,
            openRouterProviderOnly: model.openRouterProviderOnly,
            createdAt: model.createdAt
        )
    }

    private func resolvedAPIKeyForPolish(providerId: String) throws -> String {
        if let savedKey = try APIKeychainStore.apiKey(providerId: providerId),
           !savedKey.isEmpty {
            return savedKey
        }
        if providerId == TranscriptionProviderDescriptor.openRouter.id {
            throw OpenRouterLanguageModelError.missingAPIKey
        }
        throw OpenAILanguageModelError.missingAPIKey
    }

    private func combinedEstimatedCost(_ first: Double?, _ second: Double?) -> Double? {
        guard first != nil || second != nil else { return nil }
        return (first ?? 0) + (second ?? 0)
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
            await fetchSummaryModels()
            await fetchSummaryRuntimeStatus()
            polishPresets = PolishPresetStore.load()

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
            // Force all windows to redisplay — on macOS 26 Tahoe, @Published
            // changes can fail to trigger SwiftUI view updates due to the
            // display cycle exception (caught by DisplayCycleFix).
            for window in NSApp.windows {
                window.display()
            }
        } catch {
            print("Failed to fetch models: \(error)")
        }
    }

    func fetchSummaryRuntimeStatus() async {
        do {
            let status = try await apiClient.getSummaryRuntimeStatus()
            summaryRuntimeInstalled = status.installed
            summaryRuntimeReady = status.ready
        } catch {
            summaryRuntimeInstalled = summarySetupManager.checkSummaryVenvExists()
            summaryRuntimeReady = false
            print("Failed to fetch summary runtime status: \(error)")
        }
    }

    func fetchSummaryModels() async {
        do {
            availableSummaryModels = try await apiClient.listSummaryModels()
        } catch {
            print("Failed to fetch summary models: \(error)")
        }
    }

    func setPreferredSummaryModel(_ modelId: String) {
        UserDefaults.standard.set(modelId, forKey: "preferredSummaryModelId")
    }

    var preferredSummaryModelId: String? {
        UserDefaults.standard.string(forKey: "preferredSummaryModelId")
    }

    func reloadSummaryPresets() {
        summaryPresetDefinitions = SummaryPresetLibrary.load()
    }

    func saveSummaryPresets(_ presets: [SummaryPresetDefinition]) {
        SummaryPresetLibrary.save(presets)
        summaryPresetDefinitions = SummaryPresetLibrary.load()
    }

    func upsertSummaryPreset(_ preset: SummaryPresetDefinition) {
        var presets = summaryPresetDefinitions
        if let index = presets.firstIndex(where: { $0.id == preset.id }) {
            presets[index] = preset
        } else {
            presets.append(preset)
        }
        saveSummaryPresets(presets)
    }

    func deleteSummaryPreset(id: String) {
        let updated = summaryPresetDefinitions.filter { $0.id != id || $0.isBuiltIn }
        saveSummaryPresets(updated)
    }

    func resetBuiltInSummaryPreset(id: String) {
        guard let defaultPreset = SummaryPresetLibrary.defaultDefinition(for: id) else { return }
        var presets = summaryPresetDefinitions
        if let index = presets.firstIndex(where: { $0.id == id }) {
            presets[index] = defaultPreset
        } else {
            presets.append(defaultPreset)
        }
        saveSummaryPresets(presets)
    }

    func reloadPolishPresets() {
        polishPresets = PolishPresetStore.load()
    }

    func savePolishPresets(_ presets: [PolishPreset]) {
        PolishPresetStore.save(presets)
        polishPresets = PolishPresetStore.load()
    }

    func upsertPolishPreset(_ preset: PolishPreset) {
        polishPresets = PolishPresetStore.upsert(preset, in: polishPresets)
    }

    func deletePolishPreset(id: String) {
        polishPresets = PolishPresetStore.delete(id: id, from: polishPresets)
        if selectedPolishPresetId == id {
            setSelectedPolishPreset(nil)
        }
    }

    func setSelectedPolishPreset(_ id: String?) {
        selectedPolishPresetId = id
        if let id {
            UserDefaults.standard.set(id, forKey: "selectedPolishPresetId")
        } else {
            UserDefaults.standard.removeObject(forKey: "selectedPolishPresetId")
        }
        UserDefaults.standard.synchronize()
        print("AppState: selected polish preset = \(id ?? "raw")")
    }

    var selectedPolishPreset: PolishPreset? {
        let persistedId = UserDefaults.standard.string(forKey: "selectedPolishPresetId")
        guard let candidateId = selectedPolishPresetId ?? persistedId else { return nil }
        guard let preset = polishPresets.first(where: { $0.id == candidateId && $0.isEnabled }) else {
            return nil
        }
        if selectedPolishPresetId != candidateId {
            selectedPolishPresetId = candidateId
        }
        return preset
    }

    func downloadSummaryModel(modelId: String) async -> Bool {
        downloadingSummaryModelId = modelId
        summaryDownloadProgress = 0
        statusMessage = "Downloading summary model..."

        var failed = false
        do {
            try await apiClient.downloadSummaryModel(modelId: modelId) { event in
                switch event.status {
                case "downloading":
                    self.summaryDownloadProgress = event.progress ?? 0
                    if let downloaded = event.downloadedMB, let total = event.totalMB {
                        self.statusMessage = "Downloading summary model... \(downloaded)/\(total) MB"
                    }
                case "complete":
                    self.summaryDownloadProgress = 1
                    self.statusMessage = "Summary model downloaded"
                case "error":
                    failed = true
                    self.statusMessage = event.message ?? "Summary model download failed"
                default:
                    break
                }
            }
            await fetchSummaryModels()
        } catch {
            failed = true
            statusMessage = "Summary model download failed: \(error.localizedDescription)"
        }

        downloadingSummaryModelId = nil
        summaryDownloadProgress = 0
        return !failed
    }

    func deleteSummaryModel(modelId: String) async {
        statusMessage = "Deleting summary model..."
        do {
            let response = try await apiClient.deleteSummaryModel(modelId: modelId)
            statusMessage = response.message
            await fetchSummaryModels()
        } catch {
            statusMessage = "Failed to delete summary model"
            print("Delete summary model error: \(error)")
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
        defer {
            isModelLoading = false
            loadingModelId = nil  // Clear loading state
        }

        do {
            try await prepareBackendForModelLoad(modelId)
            statusMessage = "Loading model..."

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
            statusMessage = "Failed to load model: \(error.localizedDescription)"
            print("Load error: \(error)")
        }
    }

    private func prepareBackendForModelLoad(_ modelId: String) async throws {
        guard let backendManager else { return }

        if modelRequiresNemo(modelId) {
            if backendManager.currentBackendMode == .nemoVenv {
                try await ensureBackendReady(backendManager, modeName: "NeMo")
                return
            }

            guard backendManager.nemoVenvExists else {
                throw modelLoadError("NeMo runtime is not installed")
            }

            statusMessage = "Switching to NeMo backend..."
            clearLoadedModelStateForBackendSwitch()
            await backendManager.switchToNemoMode()
            try await ensureBackendReady(backendManager, modeName: "NeMo")
            await checkNemoStatus()
            await fetchAvailableModels()
            return
        }

        if backendManager.currentBackendMode == .nemoVenv {
            statusMessage = "Switching to Whisper backend..."
            clearLoadedModelStateForBackendSwitch()
            await backendManager.switchToWhisperMode()
            try await ensureBackendReady(backendManager, modeName: "Whisper")
            await checkNemoStatus()
            await fetchAvailableModels()
        } else {
            try await ensureBackendReady(backendManager, modeName: backendManager.currentBackendMode.displayName)
        }
    }

    private func ensureBackendReady(_ backendManager: BackendManager, modeName: String) async throws {
        if !backendManager.isRunning {
            backendManager.start()
        }

        guard backendManager.isServerReady else {
            await backendManager.waitForServerReady(timeout: 30.0)
            if backendManager.isServerReady { return }
            throw modelLoadError("\(modeName) backend is not ready")
        }
    }

    private func clearLoadedModelStateForBackendSwitch() {
        isModelLoaded = false
        loadedModelId = nil
    }

    private func modelRequiresNemo(_ modelId: String) -> Bool {
        if let model = availableModels.first(where: { $0.id == modelId }) {
            return model.nemoRequired
        }

        return modelId.lowercased().contains("parakeet") || modelId.lowercased().contains("nemotron")
    }

    private func modelLoadError(_ message: String) -> NSError {
        NSError(
            domain: "AiTranscribe.AppState.ModelLoad",
            code: 1,
            userInfo: [NSLocalizedDescriptionKey: message]
        )
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
        nemoVenvExists = exists
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

    /// Provider selected when the current quick recording started.
    /// UI can switch Local/Public while recording, but the active pipeline must stay stable.
    private var activeQuickRecordingProviderId: String?

    /// Task for streaming session
    private var streamingTask: Task<Void, Never>?

    /// Start recording audio
    func startRecording() async {
        guard !isQuickRecordingTransitioning, !isQuickTranscriptionInFlight, !isRecording else {
            AppLogger.warning("QuickPipeline start ignored reason=busy recording=\(isRecording) transitioning=\(isQuickRecordingTransitioning) transcribing=\(isQuickTranscriptionInFlight)")
            statusMessage = isQuickTranscriptionInFlight ? "Finishing previous recording..." : statusMessage
            return
        }

        // Mutual exclusion: don't start quick-transcribe while a session is recording
        // (The session recorder uses the same mic and audio engine)
        guard !isSessionRecordingActive else {
            statusMessage = "Stop session recording first"
            return
        }

        isQuickRecordingTransitioning = true

        // Cancel idle unload timer — user is active
        idleUnloadTimer?.invalidate()
        idleUnloadTimer = nil

        let selectedProviderId = selectedTranscriptionProviderId

        if selectedProviderId != TranscriptionProviderDescriptor.localBackend.id {
            guard APIKeychainStore.hasAPIKey(providerId: selectedProviderId) else {
                isQuickRecordingTransitioning = false
                statusMessage = "Add \(publicProviderName(for: selectedProviderId)) API key first"
                return
            }
        } else {
            // Smart auto-load: if model isn't loaded, try to load preferred model in background.
            if !isModelLoaded && !isModelLoading {
                if let preferredModelId = UserDefaults.standard.string(forKey: "preferredModelId") {
                    // Auto-load preferred model in background while recording starts
                    isAutoLoadingModel = true
                    Task {
                        await loadModel(modelId: preferredModelId)
                        isAutoLoadingModel = false
                    }
                } else {
                    // No preferred model — first-time user needs to pick one
                    isQuickRecordingTransitioning = false
                    statusMessage = "Load model first"
                    return
                }
            }
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

        activeQuickRecordingProviderId = selectedProviderId

        if selectedProviderId == TranscriptionProviderDescriptor.localBackend.id,
           streamingEnabled,
           KeyboardSimulator.shared.hasAccessibilityPermission {
            await startStreamingRecording()
        } else {
            await startNormalRecording(providerId: selectedProviderId)
        }
    }

    /// Start normal (non-streaming) recording
    /// Uses Swift's AVFoundation for proper microphone permission handling
    private func startNormalRecording(providerId: String) async {
        // Check microphone permission
        if !AudioRecorder.hasMicrophonePermission {
            AudioRecorder.requestMicrophonePermission { [weak self] granted in
                guard let self = self else { return }
                if granted {
                    Task {
                        await self.startNormalRecording(providerId: providerId)
                    }
                } else {
                    self.isQuickRecordingTransitioning = false
                    self.activeQuickRecordingProviderId = nil
                    self.statusMessage = "Microphone permission denied"
                    if self.mutePlaybackDuringRecording {
                        self.audioManager.endDuckingSession()
                    }
                }
            }
            return
        }

        configureRollingBasicBatchingIfNeeded(providerId: providerId)

        // Start Swift-native recording with selected device
        if await audioRecorder.startRecording(deviceId: selectedDeviceId) {
            isRecording = true
            isStreamingMode = false
            isQuickRecordingTransitioning = false
            statusMessage = "Recording..."

            // Play start sound
            SoundManager.shared.playStartSound()

            // Show floating indicator
            recordingIndicator.show()

            // Start polling audio recorder for volume updates
            startVolumePolling()
        } else {
            cleanupRollingBasicBatching()
            activeQuickRecordingProviderId = nil
            isQuickRecordingTransitioning = false
            statusMessage = "Failed to start recording"
            if mutePlaybackDuringRecording {
                audioManager.endDuckingSession()
            }
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
        isQuickRecordingTransitioning = false
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
                self.isQuickRecordingTransitioning = false
                self.isQuickTranscriptionInFlight = false
                self.activeQuickRecordingProviderId = nil
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
        guard isRecording else { return }
        guard !isQuickRecordingTransitioning else {
            AppLogger.warning("QuickPipeline stop ignored reason=transitioning")
            return
        }

        isQuickRecordingTransitioning = true
        isQuickTranscriptionInFlight = true
        stopStatusPolling()

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
        let pipelineStartedAt = Date()
        // Show transcribing state on indicator (dots rotate)
        recordingIndicator.setTranscribing(true)

        let selectedProviderId = activeQuickRecordingProviderId ?? selectedTranscriptionProviderId

        // If a local model was auto-loading when recording started, wait for it to finish.
        if selectedProviderId == TranscriptionProviderDescriptor.localBackend.id,
           (isAutoLoadingModel || isModelLoading) {
            statusMessage = "Loading model..."
            // Poll until model is loaded or timeout (60 seconds)
            let deadline = Date().addingTimeInterval(60)
            while (isAutoLoadingModel || isModelLoading) && Date() < deadline {
                try? await Task.sleep(nanoseconds: 500_000_000) // 500ms
            }
            if !isModelLoaded {
                isRecording = false
                isQuickRecordingTransitioning = false
                isQuickTranscriptionInFlight = false
                activeQuickRecordingProviderId = nil
                statusMessage = "Model failed to load"
                recordingIndicator.hide()
                if mutePlaybackDuringRecording { audioManager.endDuckingSession() }
                return
            }
        }

        statusMessage = "Transcribing..."

        // Reset incremental typer for next session
        incrementalTyper.reset()

        // Stop Swift AudioRecorder and get WAV data
        let audioStopStartedAt = Date()
        guard let audioData = await audioRecorder.stopRecording() else {
            isRecording = false
            isQuickRecordingTransitioning = false
            isQuickTranscriptionInFlight = false
            activeQuickRecordingProviderId = nil
            statusMessage = "No audio recorded"
            recordingIndicator.hide()
            cleanupRollingBasicBatching()
            
            // End ducking session (error path)
            if mutePlaybackDuringRecording {
                audioManager.endDuckingSession()
            }
            return
        }
        let audioStopElapsed = Date().timeIntervalSince(audioStopStartedAt)

        let recordedDuration = audioRecorder.duration
        isRecording = false
        isQuickRecordingTransitioning = false

        // Restore volume immediately when recording capture stops, before any early exits.
        if mutePlaybackDuringRecording {
            audioManager.endDuckingSession()
        }

        guard recordedDuration >= 0.75 else {
            cleanupRollingBasicBatching()
            isQuickTranscriptionInFlight = false
            activeQuickRecordingProviderId = nil
            statusMessage = "Recording too short"
            recordingIndicator.hide()
            AppLogger.warning(String(
                format: "QuickPipeline discarded too-short recording duration=%.2fs bytes=%d",
                recordedDuration,
                audioData.count
            ))
            return
        }

        let rollingSession = rollingBasicBatchSession
        let rollingAPISession = rollingAPIBatchSession
        let rollingWasUsed = rollingSession?.hasStarted == true || rollingAPISession?.hasStarted == true
        let rollingTailChunk = rollingWasUsed ? audioRecorder.finalizeRollingTailChunk() : nil
        AppLogger.info(String(
            format: "QuickPipeline audioFinalized duration=%.2fs bytes=%d finalizeElapsed=%.2fs provider=%@ rollingBatches=%@ tailChunk=%@",
            recordedDuration,
            audioData.count,
            audioStopElapsed,
            publicProviderName(for: selectedProviderId),
            rollingWasUsed ? "true" : "false",
            rollingTailChunk == nil ? "false" : "true"
        ))

        do {
            let result: TranscriptionProviderResult
            let transcriptionStartedAt = Date()

            if let rollingSession, rollingSession.hasStarted {
                statusMessage = "Finishing batches..."
                AppLogger.info("QuickPipeline using rolling local batches for final transcript")
                result = try await rollingSession.finish(
                    finalChunk: rollingTailChunk,
                    totalDurationSeconds: recordedDuration
                )
            } else if let rollingAPISession, rollingAPISession.hasStarted {
                statusMessage = "Finishing API batches..."
                AppLogger.info("QuickPipeline using rolling public API batches for final transcript")
                result = try await rollingAPISession.finish(
                    finalChunk: rollingTailChunk,
                    totalDurationSeconds: recordedDuration
                )
            } else {
                AppLogger.info(String(
                    format: "QuickPipeline using single-shot transcription provider=%@ duration=%.2fs localRollingSession=%@ apiRollingSession=%@ rollingStarted=%@",
                    publicProviderName(for: selectedProviderId),
                    recordedDuration,
                    rollingSession == nil ? "false" : "true",
                    rollingAPISession == nil ? "false" : "true",
                    rollingWasUsed ? "true" : "false"
                ))
                let provider = transcriptionProvider(for: selectedProviderId)
                let request = TranscriptionRequest(
                    audioData: audioData,
                    recordedDurationSeconds: recordedDuration,
                    workflow: .basic,
                    model: currentTranscriptionModel(for: provider.descriptor)
                )

                result = try await provider.transcribe(request) { update in
                    let percent = Int(update.progress * 100)
                    if let message = update.message, !message.isEmpty {
                        self.statusMessage = message
                    } else if percent > 0 {
                        self.statusMessage = "Transcribing... \(percent)%"
                    } else {
                        self.statusMessage = "Transcribing... (\(Int(update.elapsedSeconds))s)"
                    }
                }
            }
            let transcriptionElapsed = Date().timeIntervalSince(transcriptionStartedAt)
            AppLogger.info(String(
                format: "QuickPipeline sttComplete provider=%@ model=%@ workflow=%@ elapsed=%.2fs providerElapsed=%.2fs audioDuration=%.2fs rtf=%.3f estimatedCost=$%.6f outputWords=%d",
                result.provider.displayName,
                result.model.id,
                result.workflow.rawValue,
                transcriptionElapsed,
                result.processingTimeSeconds,
                result.durationSeconds,
                result.realtimeFactor,
                result.usage.estimatedCostUSD ?? 0,
                result.text.split(separator: " ").count
            ))

            cleanupRollingBasicBatching()
            activeQuickRecordingProviderId = nil
            let usage = result.usage
            lastTranscriptionUsage = usage
            let rawTranscriptionText = result.text
            var finalTranscriptionText = rawTranscriptionText
            var polishedTranscriptionText: String?
            var appliedPolishPresetId: String?
            var appliedPolishPresetName: String?
            var polishEstimatedCostUSD: Double?
            var polishFailureMessage: String?

            if let polishPreset = selectedPolishPreset {
                statusMessage = "Polishing with \(polishPreset.displayName)..."
                let polishStartedAt = Date()
                do {
                    let polishResult = try await polishTranscript(rawTranscriptionText, using: polishPreset)
                    let polished = polishResult.text.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !polished.isEmpty {
                        polishedTranscriptionText = polished
                        appliedPolishPresetId = polishPreset.id
                        appliedPolishPresetName = polishPreset.displayName
                        polishEstimatedCostUSD = polishResult.estimatedCostUSD
                        finalTranscriptionText = polishPreset.outputBehavior == .copyRaw ? rawTranscriptionText : polished
                    }
                    print(String(
                        format: "AppState: quick polish applied preset=%@ elapsed=%.2fs estimatedCost=$%.6f outputWords=%d",
                        polishPreset.displayName,
                        Date().timeIntervalSince(polishStartedAt),
                        polishEstimatedCostUSD ?? 0,
                        polishedTranscriptionText?.split(separator: " ").count ?? 0
                    ))
                } catch {
                    print("Polish failed: \(error)")
                    let description = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
                    polishFailureMessage = "Polish failed: \(description)"
                }
            }

            lastTranscription = finalTranscriptionText
            statusMessage = polishFailureMessage ?? "Ready"

            // Hide indicator after transcription completes
            recordingIndicator.hide()

            // Save to history
            let entry = TranscriptionEntry(
                text: finalTranscriptionText,
                rawText: polishedTranscriptionText == nil ? nil : rawTranscriptionText,
                polishedText: polishedTranscriptionText,
                polishPresetId: appliedPolishPresetId,
                polishPresetName: appliedPolishPresetName,
                duration: result.durationSeconds,
                timestamp: Date(),
                modelId: result.model.id,
                modelName: result.model.displayName,
                providerId: result.provider.id,
                providerName: result.provider.displayName,
                providerSource: result.provider.source,
                workflow: result.workflow,
                processingTimeSeconds: result.processingTimeSeconds,
                realtimeFactor: result.realtimeFactor,
                estimatedCostUSD: combinedEstimatedCost(usage.estimatedCostUSD, polishEstimatedCostUSD)
            )
            addToHistory(entry)
            AppLogger.info(String(
                format: "QuickPipeline complete totalElapsed=%.2fs sttCost=$%.6f polishCost=$%.6f totalCost=$%.6f finalWords=%d",
                Date().timeIntervalSince(pipelineStartedAt),
                usage.estimatedCostUSD ?? 0,
                polishEstimatedCostUSD ?? 0,
                combinedEstimatedCost(usage.estimatedCostUSD, polishEstimatedCostUSD) ?? 0,
                finalTranscriptionText.split(separator: " ").count
            ))

            // ALWAYS copy to clipboard (default behavior)
            copyToClipboard(finalTranscriptionText)

            // Auto-paste if enabled (simulates Cmd+V)
            let autoPaste = UserDefaults.standard.bool(forKey: "autoPasteAfterTranscription")
            if autoPaste && hasAccessibilityPermission {
                // Small delay to ensure clipboard is ready
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                    KeyboardSimulator.shared.pasteAsync()
                }
            }

            isQuickTranscriptionInFlight = false

            // Start idle unload timer — unloads model after 2 minutes of inactivity
            startIdleUnloadTimer()

        } catch {
            cleanupRollingBasicBatching()
            activeQuickRecordingProviderId = nil
            isRecording = false
            isQuickRecordingTransitioning = false
            isQuickTranscriptionInFlight = false
            if "\(error)".contains("timed out") || "\(error)".contains("timeout") {
                statusMessage = "Transcription timed out - try shorter audio"
            } else if selectedProviderId != TranscriptionProviderDescriptor.localBackend.id,
                      let localizedError = error as? LocalizedError,
                      let description = localizedError.errorDescription {
                statusMessage = description
            } else {
                statusMessage = "Transcription failed"
            }
            recordingIndicator.hide()
            print("Stop recording error: \(error)")
        }
    }

    private func configureRollingBasicBatchingIfNeeded(providerId: String) {
        cleanupRollingBasicBatching()

        guard isBasicBatchingEnabled else {
            AppLogger.debug("QuickPipeline rolling batching disabled reason=user-setting-off")
            return
        }
        guard providerId == TranscriptionProviderDescriptor.localBackend.id else {
            if !configureRollingAPIBatchingIfNeeded(providerId: providerId) {
                AppLogger.debug("QuickPipeline public rolling batching disabled provider=\(publicProviderName(for: providerId))")
            }
            return
        }
        guard !isStreamingMode else {
            AppLogger.debug("QuickPipeline rolling batching disabled reason=streaming-mode")
            return
        }
        guard isModelLoaded || isModelLoading || isAutoLoadingModel else {
            AppLogger.debug("QuickPipeline rolling batching disabled reason=local-model-not-ready")
            return
        }

        let session = RollingBasicBatchSession(
            apiClient: apiClient,
            provider: .localBackend,
            modelProvider: { [weak self] in
                self?.currentLocalTranscriptionModel() ?? TranscriptionModelDescriptor(
                    id: "unknown",
                    displayName: "Unknown",
                    providerId: TranscriptionProviderDescriptor.localBackend.id,
                    source: .local
                )
            },
            isReadyToProcess: { [weak self] in
                self?.isModelLoaded == true && self?.isModelLoading == false
            }
        )
        session.onStatusUpdate = { [weak self] message in
            self?.statusMessage = message
        }
        rollingBasicBatchSession = session

        AppLogger.info(String(
            format: "QuickPipeline rolling batching enabled threshold=%.1fs overlap=3.0s modelLoaded=%@ modelLoading=%@ autoLoading=%@",
            basicBatchingThresholdSeconds,
            isModelLoaded ? "true" : "false",
            isModelLoading ? "true" : "false",
            isAutoLoadingModel ? "true" : "false"
        ))
        audioRecorder.configureRollingChunks(
            chunkDurationSeconds: basicBatchingThresholdSeconds,
            overlapSeconds: 3
        ) { [weak self] chunk in
            self?.rollingBasicBatchSession?.enqueue(chunk)
        }
    }

    private func configureRollingAPIBatchingIfNeeded(providerId: String) -> Bool {
        guard isBasicBatchingEnabled else {
            return false
        }
        guard providerId != TranscriptionProviderDescriptor.localBackend.id else {
            return false
        }

        let provider = transcriptionProvider(for: providerId)
        guard provider.descriptor.capabilities.supportsParallelBatching else {
            AppLogger.debug("QuickPipeline public rolling batching disabled reason=provider-no-parallel provider=\(provider.descriptor.displayName)")
            return false
        }

        let model = currentTranscriptionModel(for: provider.descriptor)
        let session = RollingAPIBatchSession(
            provider: provider,
            model: model,
            maxConcurrentUploads: 4
        )
        session.onStatusUpdate = { [weak self] message in
            self?.statusMessage = message
        }
        rollingAPIBatchSession = session

        AppLogger.info(String(
            format: "QuickPipeline public rolling batching enabled provider=%@ model=%@ threshold=%.1fs overlap=3.0s maxConcurrent=4",
            provider.descriptor.displayName,
            model.id,
            basicBatchingThresholdSeconds
        ))
        audioRecorder.configureRollingChunks(
            chunkDurationSeconds: basicBatchingThresholdSeconds,
            overlapSeconds: 3
        ) { [weak self] chunk in
            self?.rollingAPIBatchSession?.enqueue(chunk)
        }
        return true
    }

    private func cleanupRollingBasicBatching() {
        rollingBasicBatchSession?.cancel()
        rollingBasicBatchSession = nil
        rollingAPIBatchSession?.cancel()
        rollingAPIBatchSession = nil
        audioRecorder.disableRollingChunks()
    }

    private var isBasicBatchingEnabled: Bool {
        if UserDefaults.standard.object(forKey: "basicTranscriptionBatchingEnabled") == nil {
            return true
        }
        return UserDefaults.standard.bool(forKey: "basicTranscriptionBatchingEnabled")
    }

    private var basicBatchingThresholdSeconds: Double {
        let configured = UserDefaults.standard.double(forKey: "basicTranscriptionBatchingThresholdSeconds")
        return configured > 0 ? configured : 120
    }

    private func currentLocalTranscriptionModel() -> TranscriptionModelDescriptor {
        let modelId = loadedModelId ?? "unknown"
        let modelName = availableModels.first { $0.id == modelId }?.displayName ?? "Unknown"
        return TranscriptionModelDescriptor(
            id: modelId,
            displayName: modelName,
            providerId: TranscriptionProviderDescriptor.localBackend.id,
            source: .local
        )
    }

    private var selectedTranscriptionProviderId: String {
        let storedProviderId = UserDefaults.standard.string(forKey: "selectedTranscriptionProviderId")
        if storedProviderId == TranscriptionProviderDescriptor.openAI.id {
            return TranscriptionProviderDescriptor.openAI.id
        }
        if storedProviderId == TranscriptionProviderDescriptor.openRouter.id {
            return TranscriptionProviderDescriptor.openRouter.id
        }
        if storedProviderId == TranscriptionProviderDescriptor.elevenLabs.id {
            return TranscriptionProviderDescriptor.elevenLabs.id
        }
        return TranscriptionProviderDescriptor.localBackend.id
    }

    private func selectedTranscriptionProvider() -> any TranscriptionProvider {
        transcriptionProvider(for: selectedTranscriptionProviderId)
    }

    private func transcriptionProvider(for providerId: String) -> any TranscriptionProvider {
        if providerId == TranscriptionProviderDescriptor.openAI.id {
            return openAITranscriptionProvider
        }
        if providerId == TranscriptionProviderDescriptor.openRouter.id {
            return openRouterTranscriptionProvider
        }
        if providerId == TranscriptionProviderDescriptor.elevenLabs.id {
            return elevenLabsTranscriptionProvider
        }
        return localTranscriptionProvider
    }

    private func publicProviderName(for providerId: String) -> String {
        switch providerId {
        case TranscriptionProviderDescriptor.openAI.id:
            return TranscriptionProviderDescriptor.openAI.displayName
        case TranscriptionProviderDescriptor.openRouter.id:
            return TranscriptionProviderDescriptor.openRouter.displayName
        case TranscriptionProviderDescriptor.elevenLabs.id:
            return TranscriptionProviderDescriptor.elevenLabs.displayName
        default:
            return "Public"
        }
    }

    private func currentTranscriptionModel(for provider: TranscriptionProviderDescriptor) -> TranscriptionModelDescriptor {
        if provider.id == TranscriptionProviderDescriptor.openAI.id {
            let modelId = UserDefaults.standard.string(forKey: "selectedOpenAITranscriptionModel")
            let supportedAPIModels = APITranscriptionModelStore.load().filter {
                $0.providerId == provider.id &&
                APITranscriptionModelStore.isSupportedModel($0.modelId, providerId: $0.providerId)
            }
            if let apiModel = supportedAPIModels.first(where: { $0.modelId == modelId }) {
                return TranscriptionModelDescriptor(
                    id: apiModel.modelId,
                    displayName: apiModel.displayName,
                    providerId: apiModel.providerId,
                    source: .api
                )
            }
            if let descriptor = TranscriptionModelDescriptor.openAIModels.first(where: { $0.id == modelId }) {
                return descriptor
            }
            if let fallbackModel = supportedAPIModels.first {
                UserDefaults.standard.set(fallbackModel.modelId, forKey: "selectedOpenAITranscriptionModel")
                return TranscriptionModelDescriptor(
                    id: fallbackModel.modelId,
                    displayName: fallbackModel.displayName,
                    providerId: fallbackModel.providerId,
                    source: .api
                )
            }
            let fallbackDescriptor = TranscriptionModelDescriptor.openAIModels[0]
            UserDefaults.standard.set(fallbackDescriptor.id, forKey: "selectedOpenAITranscriptionModel")
            return fallbackDescriptor
        }
        if provider.id == TranscriptionProviderDescriptor.openRouter.id {
            let modelId = UserDefaults.standard.string(forKey: "selectedOpenAITranscriptionModel")
            let supportedAPIModels = APITranscriptionModelStore.load().filter {
                $0.providerId == provider.id &&
                APITranscriptionModelStore.isSupportedModel($0.modelId, providerId: $0.providerId)
            }
            if let apiModel = supportedAPIModels.first(where: { $0.modelId == modelId }) ?? supportedAPIModels.first {
                UserDefaults.standard.set(apiModel.modelId, forKey: "selectedOpenAITranscriptionModel")
                return TranscriptionModelDescriptor(
                    id: apiModel.modelId,
                    displayName: apiModel.displayName,
                    providerId: apiModel.providerId,
                    source: .api
                )
            }
            return TranscriptionModelDescriptor(
                id: "openai/whisper-1",
                displayName: "OpenRouter Whisper",
                providerId: provider.id,
                source: .api
            )
        }
        if provider.id == TranscriptionProviderDescriptor.elevenLabs.id {
            let modelId = UserDefaults.standard.string(forKey: "selectedOpenAITranscriptionModel")
            let supportedAPIModels = APITranscriptionModelStore.load().filter {
                $0.providerId == provider.id &&
                APITranscriptionModelStore.isSupportedModel($0.modelId, providerId: $0.providerId)
            }
            if let apiModel = supportedAPIModels.first(where: { $0.modelId == modelId }) ?? supportedAPIModels.first {
                UserDefaults.standard.set(apiModel.modelId, forKey: "selectedOpenAITranscriptionModel")
                return TranscriptionModelDescriptor(
                    id: apiModel.modelId,
                    displayName: apiModel.displayName,
                    providerId: apiModel.providerId,
                    source: .api
                )
            }
            return TranscriptionModelDescriptor(
                id: "scribe_v2",
                displayName: "ElevenLabs Scribe v2",
                providerId: provider.id,
                source: .api
            )
        }
        return currentLocalTranscriptionModel()
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
            isQuickRecordingTransitioning = false
            isQuickTranscriptionInFlight = false
            activeQuickRecordingProviderId = nil
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
            await audioRecorder.cancelRecording()
            cleanupRollingBasicBatching()
        }

        isRecording = false
        isQuickRecordingTransitioning = false
        isQuickTranscriptionInFlight = false
        activeQuickRecordingProviderId = nil
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

    // =========================================================================
    // SMART IDLE UNLOADING
    // =========================================================================

    /// Start the idle unload timer — fires after 2 minutes of inactivity
    private func startIdleUnloadTimer() {
        idleUnloadTimer?.invalidate()
        idleUnloadTimer = Timer.scheduledTimer(withTimeInterval: idleTimeoutSeconds, repeats: false) { [weak self] _ in
            guard let self = self else { return }
            Task { @MainActor in
                await self.idleUnloadIfNeeded()
            }
        }
    }

    /// Unload model if it's not a streaming-native model
    private func idleUnloadIfNeeded() async {
        // Don't unload streaming models (nemotron) — they need to stay ready
        if let modelId = loadedModelId,
           let model = availableModels.first(where: { $0.id == modelId }),
           model.streamingNative {
            return
        }

        guard isModelLoaded else { return }
        print("Idle timeout reached — auto-unloading model to free RAM")
        await unloadModel()
    }

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
