import SwiftUI
import AppKit
import ScreenCaptureKit

// MARK: - Sessions Settings View

struct SessionsSettingsView: View {
    @EnvironmentObject var sessionManager: SessionManager
    @EnvironmentObject var appState: AppState
    var onNavigateToModels: ((ModelMode) -> Void)? = nil

    let hasAnimated: Bool
    let onAnimated: () -> Void
    let initialSelectedSessionId: UUID?

    @State private var selectedSessionId: UUID? = nil
    @State private var showDeleteAllAlert = false
    @State private var showDeleteTranscribedAlert = false
    @State private var showDeleteSelectedSessionsAlert = false
    @State private var showClearTranscriptionAlert = false
    @State private var screenRecordingPermission = false
    @State private var isSelectingSessions = false
    @State private var selectedSessionIds: Set<UUID> = []
    @State private var showAudioMixSheet = false
    @State private var showAutomationSheet = false
    @State private var showImportPage = false
    @State private var selectedDate: Date? = nil
    @State private var showDatePicker = false
    @State private var apiTranscriptionModels: [APITranscriptionModel] = APITranscriptionModelStore.load()
    @State private var apiLanguageModels: [APILanguageModel] = APILanguageModelStore.load()
    @State private var automationSelectedPresetIds: Set<String> = Set(
        UserDefaults.standard.stringArray(forKey: SessionAutomationPreferences.summaryPresetIdsKey)
            ?? [SessionSummaryPreset.meetingNotes.rawValue]
    )

    @AppStorage("hideSessionInfoBanner") private var hideBanner = false
    @AppStorage(SessionAudioMixPreferences.micTrimDBKey) private var sessionMicTrimDB = SessionAudioMixPreferences.defaultMicTrimDB
    @AppStorage(SessionAudioMixPreferences.systemTrimDBKey) private var sessionSystemTrimDB = SessionAudioMixPreferences.defaultSystemTrimDB
    @AppStorage(SessionAutomationPreferences.autoTranscribeKey) private var automationAutoTranscribe = false
    @AppStorage(SessionAutomationPreferences.transcriptionSourceKey) private var automationTranscriptionSource = SessionAutomationSource.local.rawValue
    @AppStorage(SessionAutomationPreferences.localTranscriptionModelKey) private var automationLocalTranscriptionModelId = "parakeet-v2"
    @AppStorage(SessionAutomationPreferences.transcriptionProviderKey) private var automationTranscriptionProviderId = TranscriptionProviderDescriptor.openAI.id
    @AppStorage(SessionAutomationPreferences.transcriptionModelKey) private var automationTranscriptionModelId = TranscriptionModelDescriptor.openAIModels[0].id
    @AppStorage(SessionAutomationPreferences.localRamBudgetGBKey) private var automationLocalRamBudgetGB = 4.0
    @AppStorage(SessionAutomationPreferences.publicBatchTargetKey) private var automationPublicBatchTarget = 0
    @AppStorage(SessionAutomationPreferences.autoSummarizeKey) private var automationAutoSummarize = false
    @AppStorage(SessionAutomationPreferences.summarySourceKey) private var automationSummarySource = SessionAutomationSource.local.rawValue
    @AppStorage(SessionAutomationPreferences.localSummaryModelKey) private var automationLocalSummaryModelId = "gemma-4-e2b-it-4bit"
    @AppStorage(SessionAutomationPreferences.summaryProviderKey) private var automationSummaryProviderId = TranscriptionProviderDescriptor.openAI.id
    @AppStorage(SessionAutomationPreferences.summaryModelKey) private var automationSummaryModelId = APILanguageModelStore.defaultModels[0].modelId
    @AppStorage(SessionAutomationPreferences.summaryLengthOptionKey) private var automationSummaryLengthOption = SummaryLengthOption.medium.rawValue
    @AppStorage(SessionAutomationPreferences.summaryCustomWordsKey) private var automationSummaryCustomWords = SummaryLengthOption.custom.defaultWordTarget

    /// Animation state
    @State private var appeared: Bool

    init(hasAnimated: Bool, onAnimated: @escaping () -> Void, onNavigateToModels: ((ModelMode) -> Void)? = nil, initialSelectedSessionId: UUID? = nil) {
        self.hasAnimated = hasAnimated
        self.onAnimated = onAnimated
        self.onNavigateToModels = onNavigateToModels
        self.initialSelectedSessionId = initialSelectedSessionId
        _appeared = State(initialValue: hasAnimated)
        _selectedSessionId = State(initialValue: initialSelectedSessionId)
    }

    // MARK: - Computed

    private var filteredSessions: [Session] {
        guard let date = selectedDate else { return sessionManager.sessions }
        return sessionManager.sessions.filter { Calendar.current.isDate($0.createdAt, inSameDayAs: date) }
    }

    /// Group sessions by date (newest first)
    private var groupedSessions: [(date: Date, sessions: [Session])] {
        let grouped = Dictionary(grouping: filteredSessions) { session in
            Calendar.current.startOfDay(for: session.createdAt)
        }
        return grouped
            .map { (date: $0.key, sessions: $0.value.sorted { $0.createdAt > $1.createdAt }) }
            .sorted { $0.date > $1.date }
    }

    private var totalStorageSize: Int64 {
        sessionManager.getTotalStorageSize()
    }

    private var uniqueDates: [Date] {
        let dates = Set(sessionManager.sessions.map { Calendar.current.startOfDay(for: $0.createdAt) })
        return dates.sorted(by: >)
    }

    private var localSessionModels: [ModelInfoResponse] {
        appState.availableModels.filter { $0.downloaded == true && $0.sessionCompatible }
    }

    private var selectedLocalSessionModel: ModelInfoResponse? {
        localSessionModels.first { $0.id == automationLocalTranscriptionModelId }
            ?? localSessionModels.first
    }

    private var publicSessionModels: [APITranscriptionModel] {
        apiTranscriptionModels.filter {
            APITranscriptionModelStore.isSupportedModel($0.modelId, providerId: $0.providerId)
        }
    }

    private var selectedPublicSessionModel: APITranscriptionModel? {
        publicSessionModels.first {
            $0.providerId == automationTranscriptionProviderId &&
                $0.modelId == automationTranscriptionModelId
        } ?? publicSessionModels.first
    }

    private var summaryModels: [SummaryModelInfoResponse] {
        appState.availableSummaryModels.filter(\.downloaded)
    }

    private var selectedLocalSummaryModel: SummaryModelInfoResponse? {
        summaryModels.first { $0.id == automationLocalSummaryModelId }
    }

    private var selectedPublicSummaryModel: APILanguageModel? {
        apiLanguageModels.first {
            $0.providerId == automationSummaryProviderId &&
                $0.modelId == automationSummaryModelId
        } ?? apiLanguageModels.first
    }

    private var automationSummaryPresets: [SummaryPresetDefinition] {
        appState.summaryPresetDefinitions.filter {
            !$0.instructions.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
    }

    private var automationSummaryLength: SummaryLengthOption {
        get { SummaryLengthOption(rawValue: automationSummaryLengthOption) ?? .medium }
        nonmutating set { automationSummaryLengthOption = newValue.rawValue }
    }

    private var automationTargetSummaryWords: Int {
        automationSummaryLength == .custom
            ? max(120, automationSummaryCustomWords)
            : automationSummaryLength.defaultWordTarget
    }

    var body: some View {
        VStack(spacing: 0) {
            if showImportPage {
                SessionImportView(
                    onBack: { showImportPage = false },
                    onImported: { sessionId in
                        showImportPage = false
                        withAnimation(.easeInOut(duration: 0.15)) {
                            selectedSessionId = sessionId
                        }
                    }
                )
            } else if let selectedId = selectedSessionId,
               sessionManager.sessions.contains(where: { $0.id == selectedId }) {
                SessionDetailView(sessionId: selectedId, onBack: {
                    selectedSessionId = nil
                }, onNavigateToModels: onNavigateToModels)
            } else {
                sessionsListView
            }
        }
        .task {
            sessionManager.loadSessions()
            screenRecordingPermission = await SystemAudioCapture.checkPermission()
        }
        .task(id: "stagger") {
            guard !hasAnimated else { return }
            try? await Task.sleep(for: .milliseconds(80))
            appeared = true
            onAnimated()
        }
        .onChange(of: sessionManager.sessions) { _, sessions in
            let validIds = Set(sessions.map(\.id))
            selectedSessionIds = selectedSessionIds.intersection(validIds)
            if sessions.isEmpty { isSelectingSessions = false }
        }
    }

    // MARK: - Sessions List

    private var sessionsListView: some View {
        ScrollView {
            VStack(spacing: 0) {
                // Header
                sessionsHeader
                    .staggerIn(index: 0, appeared: appeared)

                // Permission pills
                permissionPills
                    .padding(.top, 8)
                    .staggerIn(index: 1, appeared: appeared)

                SettingsDivider()
                    .padding(.top, 12)
                    .staggerIn(index: 2, appeared: appeared)

                // Info banner
                if !hideBanner {
                    infoBanner
                        .padding(.top, 12)
                        .staggerIn(index: 3, appeared: appeared)
                }

                // Date filter chips
                if !uniqueDates.isEmpty {
                    dateFilterRow
                        .padding(.top, 12)
                        .staggerIn(index: 4, appeared: appeared)
                }

                SettingsDivider()
                    .padding(.top, 12)
                    .staggerIn(index: 5, appeared: appeared)

                // Session list or empty state
                if sessionManager.sessions.isEmpty {
                    emptyStateView
                        .staggerIn(index: 6, appeared: appeared)
                } else if filteredSessions.isEmpty {
                    noMatchView
                        .staggerIn(index: 6, appeared: appeared)
                } else {
                    sessionGroups
                        .staggerIn(index: 6, appeared: appeared)
                }

                // Footer actions
                if !sessionManager.sessions.isEmpty {
                    SettingsDivider()
                        .padding(.top, 8)
                    footerActions
                        .padding(.top, 8)
                        .staggerIn(index: 7, appeared: appeared)
                }

                Spacer(minLength: 20)
            }
            .padding(.horizontal, 28)
            .padding(.top, 8)
        }
        .scrollIndicators(.automatic)
        .sheet(isPresented: $showAudioMixSheet) {
            AudioMixSheetView(
                micTrimDB: $sessionMicTrimDB,
                systemTrimDB: $sessionSystemTrimDB,
                latestSessionName: latestSessionWithAudio?.name,
                latestAudioURL: latestSessionAudioURL
            )
        }
        .sheet(isPresented: $showAutomationSheet) {
            sessionAutomationSettingsSheet
        }
        .task {
            apiTranscriptionModels = APITranscriptionModelStore.load()
            apiLanguageModels = APILanguageModelStore.load()
            normalizeAutomationSelections()
        }
    }

    // MARK: - Header

    private var sessionsHeader: some View {
        VStack(spacing: 12) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Sessions")
                        .font(.system(size: 22, weight: .bold, design: .rounded))

                    HStack(spacing: 6) {
                        Text("\(sessionManager.sessions.count) sessions")
                            .font(.system(size: 11))
                            .foregroundStyle(.secondary)
                        if totalStorageSize > 0 {
                            Text("·")
                                .foregroundStyle(.tertiary)
                            Text(SessionManager.formatFileSize(totalStorageSize))
                                .font(.system(size: 11))
                                .foregroundStyle(.secondary)
                        }
                    }
                }

                Spacer()

                // Import audio button
                Button {
                    withAnimation(.easeInOut(duration: 0.15)) {
                        showImportPage = true
                    }
                } label: {
                    HStack(spacing: 5) {
                        Image(systemName: "square.and.arrow.down")
                            .font(.system(size: 10, weight: .medium))
                        Text("Import")
                            .font(.system(size: 11, weight: .medium))
                    }
                    .foregroundStyle(.primary)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 7)
                    .background(.thinMaterial, in: Capsule())
                    .overlay(Capsule().strokeBorder(.white.opacity(0.06), lineWidth: 0.5))
                }
                .buttonStyle(.plain)

                // Audio mix button
                Button {
                    showAudioMixSheet = true
                } label: {
                    HStack(spacing: 5) {
                        Image(systemName: "slider.horizontal.3")
                            .font(.system(size: 10, weight: .medium))
                        Text("Audio Mix")
                            .font(.system(size: 11, weight: .medium))
                    }
                    .foregroundStyle(.primary)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 7)
                    .background(.thinMaterial, in: Capsule())
                    .overlay(Capsule().strokeBorder(.white.opacity(0.06), lineWidth: 0.5))
                }
                .buttonStyle(.plain)

                // Select button
                Button {
                    if isSelectingSessions {
                        exitSelectionMode()
                    } else {
                        isSelectingSessions = true
                    }
                } label: {
                    Text(isSelectingSessions ? "Done" : "Select")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(isSelectingSessions ? .white : .secondary)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 7)
                        .background(
                            isSelectingSessions
                                ? AnyShapeStyle(Color.accentColor)
                                : AnyShapeStyle(Color.primary.opacity(0.06))
                        )
                        .clipShape(Capsule())
                }
                .buttonStyle(.plain)
                .disabled(sessionManager.sessions.isEmpty)
            }
        }
    }

    private var automationDockControl: some View {
        let activeCount = sessionManager.sessions.filter { $0.automation?.isActive == true }.count
        let statusText = activeCount > 0 ? "\(activeCount) active" : (automationAutoTranscribe ? "On" : "Off")
        let tint: Color = activeCount > 0 ? .orange : (automationAutoTranscribe ? .green : .secondary)

        return HStack(spacing: 0) {
            Button {
                showAutomationSheet = true
            } label: {
                HStack(spacing: 7) {
                    Image(systemName: "arrow.triangle.2.circlepath")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(tint)

                    Text("Auto Process")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(.primary)
                    Text(statusText)
                        .font(.system(size: 9, weight: .medium))
                        .foregroundStyle(tint)
                }
                .padding(.leading, 10)
                .padding(.trailing, 8)
                .padding(.vertical, 5)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            Rectangle()
                .fill(Color.primary.opacity(0.08))
                .frame(width: 1, height: 20)

            Toggle("", isOn: $automationAutoTranscribe)
                .toggleStyle(.switch)
                .controlSize(.small)
                .labelsHidden()
                .padding(.horizontal, 7)
        }
        .background(.thinMaterial, in: Capsule())
        .overlay(
            Capsule()
                .strokeBorder(tint.opacity(automationAutoTranscribe || activeCount > 0 ? 0.24 : 0.08), lineWidth: 0.6)
        )
    }

    // MARK: - Permission Pills

    private var permissionPills: some View {
        HStack(spacing: 8) {
            permissionPill(
                label: "Microphone",
                granted: AudioRecorder.hasMicrophonePermission
            )
            permissionPill(
                label: "Screen Recording",
                granted: screenRecordingPermission,
                onRequest: !screenRecordingPermission ? {
                    SystemAudioCapture.requestPermission()
                    Task {
                        try? await Task.sleep(for: .seconds(2))
                        screenRecordingPermission = await SystemAudioCapture.checkPermission()
                    }
                } : nil
            )

            if isSelectingSessions {
                footerPill(label: "Delete Selected", tint: .red) {
                    showDeleteSelectedSessionsAlert = true
                }
                .disabled(selectedSessionIds.isEmpty)
                .alert("Delete Selected Sessions?", isPresented: $showDeleteSelectedSessionsAlert) {
                    Button("Cancel", role: .cancel) {}
                    Button("Delete", role: .destructive) {
                        sessionManager.bulkDeleteSessions(ids: selectedSessionIds)
                        exitSelectionMode()
                    }
                } message: {
                    Text("Selected sessions and their data will be permanently deleted.")
                }

                footerPill(label: "Clear Transcription", tint: .orange) {
                    showClearTranscriptionAlert = true
                }
                .disabled(selectedSessionIds.isEmpty)
                .alert("Clear Transcription?", isPresented: $showClearTranscriptionAlert) {
                    Button("Cancel", role: .cancel) {}
                    Button("Clear", role: .destructive) {
                        sessionManager.bulkResetTranscription(ids: selectedSessionIds)
                        exitSelectionMode()
                    }
                } message: {
                    Text("Transcriptions for selected sessions will be deleted. Audio files are kept.")
                }
            }

            Spacer()

            automationDockControl
        }
    }

    private func permissionPill(label: String, granted: Bool, onRequest: (() -> Void)? = nil) -> some View {
        Button {
            onRequest?()
        } label: {
            HStack(spacing: 5) {
                Circle()
                    .fill(granted ? .green : .red)
                    .frame(width: 6, height: 6)
                Text(label)
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(.primary)
                if !granted && onRequest != nil {
                    Text("Grant")
                        .font(.system(size: 9, weight: .semibold))
                        .foregroundStyle(.blue)
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .background(.thinMaterial, in: Capsule())
            .overlay(Capsule().strokeBorder(.white.opacity(0.06), lineWidth: 0.5))
        }
        .buttonStyle(.plain)
        .disabled(granted || onRequest == nil)
    }

    // MARK: - Info Banner

    private var infoBanner: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "info.circle.fill")
                .font(.system(size: 14))
                .foregroundStyle(.blue)

            VStack(alignment: .leading, spacing: 4) {
                Text("Record long-form audio sessions")
                    .font(.system(size: 12, weight: .semibold))

                Text("Capture meetings, lectures, or interviews from the menu bar, then transcribe here.")
                    .font(.system(size: 10))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Spacer()

            Button {
                withAnimation { hideBanner = true }
            } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 9, weight: .medium))
                    .foregroundStyle(.tertiary)
                    .frame(width: 20, height: 20)
                    .background(.quaternary, in: .circle)
            }
            .buttonStyle(.plain)
        }
        .padding(12)
        .background(.thinMaterial, in: .rect(cornerRadius: 12, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(.blue.opacity(0.15), lineWidth: 0.5)
        )
    }

    // MARK: - Session Automation

    private var sessionAutomationSettingsSheet: some View {
        VStack(spacing: 0) {
            HStack(spacing: 14) {
                ZStack {
                    Circle()
                        .fill(Color.accentColor.opacity(0.14))
                        .frame(width: 42, height: 42)
                    Image(systemName: "arrow.triangle.2.circlepath")
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundStyle(Color.accentColor)
                }

                VStack(alignment: .leading, spacing: 3) {
                    HStack(spacing: 8) {
                        Text("Session Auto Processing")
                            .font(.system(size: 17, weight: .semibold, design: .rounded))
                        automationStatusPill
                    }
                    Text("Choose what happens after each new session recording ends.")
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                }

                Spacer()

                Toggle("", isOn: $automationAutoTranscribe)
                    .toggleStyle(.switch)
                    .labelsHidden()
            }
            .padding(20)

            Divider()

            ScrollView {
                sessionAutomationPanel
                    .padding(20)
            }
            .scrollIndicators(.automatic)

            Divider()

            HStack {
                Text(automationAutoTranscribe ? "New sessions will be processed automatically." : "Automation is off for new sessions.")
                    .font(.system(size: 10))
                    .foregroundStyle(.tertiary)

                Spacer()

                Button("Done") {
                    showAutomationSheet = false
                }
                .buttonStyle(.plain)
                .modifier(SessionSheetActionPillStyle(tint: .accentColor, prominent: true))
                .keyboardShortcut(.defaultAction)
            }
            .padding(20)
        }
        .frame(width: 680, height: automationAutoSummarize && automationAutoTranscribe ? 620 : 430)
    }

    private var sessionAutomationPanel: some View {
        VStack(alignment: .leading, spacing: 12) {
            if automationAutoTranscribe {
                VStack(alignment: .leading, spacing: 12) {
                    automationTranscriptionControls

                    Divider().opacity(0.2)

                    HStack(alignment: .center, spacing: 12) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Summarize after transcription")
                                .font(.system(size: 11, weight: .semibold))
                            Text("Runs only after the automatic transcript succeeds.")
                                .font(.system(size: 10))
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                        Toggle("", isOn: $automationAutoSummarize)
                            .toggleStyle(.switch)
                            .labelsHidden()
                    }

                    if automationAutoSummarize {
                        automationSummaryControls
                    }
                }
            } else {
                VStack(alignment: .leading, spacing: 8) {
                    Text("After session ends")
                        .font(.system(size: 13, weight: .semibold, design: .rounded))
                    Text("Turn on automation to transcribe new sessions automatically, then optionally generate summaries.")
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(14)
                .background(Color.primary.opacity(0.045), in: .rect(cornerRadius: 12, style: .continuous))
            }
        }
        .padding(14)
        .background(.thinMaterial, in: .rect(cornerRadius: 14, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(Color.accentColor.opacity(automationAutoTranscribe ? 0.18 : 0.08), lineWidth: 0.5)
        )
    }

    private var automationStatusPill: some View {
        let activeCount = sessionManager.sessions.filter { $0.automation?.isActive == true }.count
        let label: String = {
            if activeCount > 0 { return "\(activeCount) active" }
            return automationAutoTranscribe ? "Enabled" : "Off"
        }()
        let tint: Color = activeCount > 0 ? .orange : (automationAutoTranscribe ? .green : .secondary)

        return HStack(spacing: 4) {
            Circle()
                .fill(tint)
                .frame(width: 5, height: 5)
            Text(label)
                .font(.system(size: 9, weight: .semibold))
        }
        .foregroundStyle(tint)
        .padding(.horizontal, 7)
        .padding(.vertical, 3)
        .background(tint.opacity(0.12), in: Capsule())
    }

    private func automationSourcePills(selection: Binding<String>) -> some View {
        HStack(spacing: 2) {
            ForEach(SessionAutomationSource.allCases) { source in
                let selected = selection.wrappedValue == source.rawValue
                Button {
                    withAnimation(.easeOut(duration: 0.12)) {
                        selection.wrappedValue = source.rawValue
                    }
                } label: {
                    Text(source.displayName)
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(selected ? .white : .secondary)
                        .frame(width: source == .local ? 78 : 96, height: 26)
                        .background(selected ? AnyShapeStyle(Color.accentColor) : AnyShapeStyle(.clear), in: Capsule())
                }
                .buttonStyle(.plain)
            }
        }
        .padding(3)
        .background(Color.primary.opacity(0.075), in: Capsule())
        .overlay(Capsule().strokeBorder(.white.opacity(0.06), lineWidth: 0.5))
    }

    private var publicTranscriptionModelMenu: some View {
        Menu {
            ForEach(publicSessionModels) { model in
                Button("\(model.displayName) · \(model.providerName)") {
                    automationTranscriptionProviderId = model.providerId
                    automationTranscriptionModelId = model.modelId
                }
            }
        } label: {
            automationSelectPill(
                title: selectedPublicSessionModel.map { "\($0.displayName) · \($0.providerName)" } ?? "Choose API transcription model",
                icon: "waveform",
                width: 330
            )
        }
        .menuStyle(.borderlessButton)
    }

    private var localTranscriptionModelMenu: some View {
        Menu {
            ForEach(localSessionModels) { model in
                Button(model.displayName) {
                    automationLocalTranscriptionModelId = model.id
                }
            }
        } label: {
            automationSelectPill(
                title: selectedLocalSessionModel?.displayName ?? "Choose local transcription model",
                icon: "externaldrive.fill",
                width: 330
            )
        }
        .menuStyle(.borderlessButton)
    }

    private var publicSummaryModelMenu: some View {
        Menu {
            ForEach(apiLanguageModels) { model in
                Button("\(model.displayName) · \(model.providerName)") {
                    automationSummaryProviderId = model.providerId
                    automationSummaryModelId = model.modelId
                }
            }
        } label: {
            automationSelectPill(
                title: selectedPublicSummaryModel.map { "\($0.displayName) · \($0.providerName)" } ?? "Choose API summary model",
                icon: "sparkles",
                width: 330
            )
        }
        .menuStyle(.borderlessButton)
    }

    private var localSummaryModelMenu: some View {
        Menu {
            ForEach(summaryModels) { model in
                Button(model.displayName) {
                    automationLocalSummaryModelId = model.id
                }
            }
        } label: {
            automationSelectPill(
                title: selectedLocalSummaryModel?.displayName ?? "Choose local summary model",
                icon: "sparkles",
                width: 330
            )
        }
        .menuStyle(.borderlessButton)
    }

    private func automationSelectPill(title: String, icon: String, width: CGFloat) -> some View {
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

    private func automationIntegerStepperPill(
        label: String,
        value: Binding<Int>,
        range: ClosedRange<Int>,
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
                .frame(width: 42)

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

    private func automationDoubleStepperPill(
        label: String,
        value: Binding<Double>,
        range: ClosedRange<Double>,
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

    private var automationLengthPills: some View {
        HStack(spacing: 2) {
            ForEach(SummaryLengthOption.allCases) { option in
                let selected = automationSummaryLength == option
                Button {
                    withAnimation(.easeOut(duration: 0.12)) {
                        automationSummaryLength = option
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

    private var automationTranscriptionControls: some View {
        VStack(alignment: .leading, spacing: 12) {
            automationFieldLabel("TRANSCRIBE")

            automationSourcePills(selection: Binding(
                get: { automationTranscriptionSource },
                set: { automationTranscriptionSource = $0 }
            ))

            if automationTranscriptionSource == SessionAutomationSource.public.rawValue {
                if publicSessionModels.isEmpty {
                    automationEmptyModelRow(
                        title: "No public transcription model",
                        actionTitle: "Open API Models",
                        mode: .speechToText
                    )
                } else {
                    HStack(spacing: 12) {
                        publicTranscriptionModelMenu
                        automationIntegerStepperPill(
                            label: "Batches",
                            value: $automationPublicBatchTarget,
                            range: 0...30,
                            step: 6,
                            zeroLabel: "Auto"
                        )
                        Spacer()
                    }
                }
            } else {
                if localSessionModels.isEmpty {
                    automationEmptyModelRow(
                        title: "No downloaded session model",
                        actionTitle: "Open Models",
                        mode: .speechToText
                    )
                } else {
                    HStack(spacing: 12) {
                        localTranscriptionModelMenu
                        automationDoubleStepperPill(
                            label: "RAM",
                            value: $automationLocalRamBudgetGB,
                            range: 2...max(2, maxAutomationRamBudget),
                            step: 0.5,
                            suffix: "GB"
                        )
                        Spacer()
                    }
                }
            }
        }
    }

    private var automationSummaryControls: some View {
        VStack(alignment: .leading, spacing: 12) {
            automationFieldLabel("SUMMARY")

            automationSourcePills(selection: Binding(
                get: { automationSummarySource },
                set: { automationSummarySource = $0 }
            ))

            HStack(spacing: 12) {
                if automationSummarySource == SessionAutomationSource.public.rawValue {
                    if apiLanguageModels.isEmpty {
                        automationEmptyModelRow(
                            title: "No public language model",
                            actionTitle: "Open API LLM",
                            mode: .summarization
                        )
                    } else {
                        publicSummaryModelMenu
                    }
                } else {
                    if summaryModels.isEmpty {
                        automationEmptyModelRow(
                            title: "No local summary model",
                            actionTitle: "Open Models",
                            mode: .summarization
                        )
                    } else {
                        localSummaryModelMenu
                        automationInfoPill(icon: "externaldrive.fill", text: "Downloaded", tint: .secondary)
                    }
                }
                Spacer()
            }

            HStack(spacing: 12) {
                automationLengthPills

                if automationSummaryLength == .custom {
                    automationIntegerStepperPill(
                        label: "Words",
                        value: $automationSummaryCustomWords,
                        range: 180...2200,
                        step: 80,
                        zeroLabel: nil
                    )
                } else {
                    automationInfoPill(icon: "text.word.spacing", text: "~\(automationTargetSummaryWords) words", tint: .secondary)
                }
                Spacer()
            }

            VStack(alignment: .leading, spacing: 6) {
                Text("Presets")
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(.secondary)

                FlowLayout(spacing: 6, rowSpacing: 6) {
                    ForEach(automationSummaryPresets) { preset in
                        automationPresetChip(preset)
                    }
                }
            }
        }
    }

    private var maxAutomationRamBudget: Double {
        let totalRAM = Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824
        return max(4.0, totalRAM - 4.0)
    }

    private func automationFieldLabel(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 9, weight: .semibold, design: .rounded))
            .foregroundStyle(.tertiary)
            .tracking(0.5)
    }

    private func automationEmptyModelRow(title: String, actionTitle: String, mode: ModelMode) -> some View {
        HStack(spacing: 8) {
            Image(systemName: "exclamationmark.triangle")
                .font(.system(size: 10))
                .foregroundStyle(.orange)
            Text(title)
                .font(.system(size: 10))
                .foregroundStyle(.secondary)
            Spacer()
            if let onNavigateToModels {
                Button(actionTitle) { onNavigateToModels(mode) }
                    .font(.system(size: 10, weight: .medium))
                    .buttonStyle(.plain)
                    .foregroundStyle(Color.accentColor)
            }
        }
    }

    private func automationInfoPill(icon: String, text: String, tint: Color) -> some View {
        HStack(spacing: 4) {
            Image(systemName: icon)
                .font(.system(size: 8))
            Text(text)
                .font(.system(size: 9, weight: .medium))
        }
        .foregroundStyle(tint)
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(tint.opacity(0.1), in: Capsule())
    }

    private func automationPresetChip(_ preset: SummaryPresetDefinition) -> some View {
        let selected = automationSelectedPresetIds.contains(preset.id)
        return Button {
            toggleAutomationPreset(preset.id)
        } label: {
            HStack(spacing: 5) {
                Image(systemName: preset.symbolName)
                    .font(.system(size: 9, weight: .medium))
                Text(preset.displayName)
                    .font(.system(size: 10, weight: selected ? .semibold : .medium))
            }
            .foregroundStyle(selected ? .white : .secondary)
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(selected ? AnyShapeStyle(Color.accentColor) : AnyShapeStyle(Color.primary.opacity(0.06)), in: Capsule())
        }
        .buttonStyle(.plain)
    }

    // MARK: - Date Filter

    private static let dateChipFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "MMM d"
        return f
    }()

    private var dateFilterRow: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 4) {
                // "All" chip
                dateChip(label: "All", isSelected: selectedDate == nil) {
                    selectedDate = nil
                }

                ForEach(Array(uniqueDates.prefix(10)), id: \.self) { date in
                    let label = Calendar.current.isDateInToday(date) ? "Today"
                        : Calendar.current.isDateInYesterday(date) ? "Yesterday"
                        : Self.dateChipFormatter.string(from: date)

                    let count = sessionManager.sessions.filter {
                        Calendar.current.isDate($0.createdAt, inSameDayAs: date)
                    }.count

                    dateChip(
                        label: "\(label) (\(count))",
                        isSelected: selectedDate.map { Calendar.current.isDate($0, inSameDayAs: date) } ?? false
                    ) {
                        selectedDate = (selectedDate.map { Calendar.current.isDate($0, inSameDayAs: date) } ?? false) ? nil : date
                    }
                }

                // Calendar picker
                Button {
                    showDatePicker = true
                } label: {
                    Image(systemName: "calendar")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 6)
                        .background(
                            showDatePicker
                                ? AnyShapeStyle(Color.accentColor.opacity(0.1))
                                : AnyShapeStyle(Color.primary.opacity(0.06))
                        )
                        .clipShape(Capsule())
                }
                .buttonStyle(.plain)
                .popover(isPresented: $showDatePicker, arrowEdge: .bottom) {
                    SessionCalendarPopover(
                        sessionDates: sessionDateCounts,
                        selectedDate: $selectedDate,
                        isPresented: $showDatePicker
                    )
                }
            }
        }
    }

    private var sessionDateCounts: [Date: Int] {
        var counts: [Date: Int] = [:]
        for session in sessionManager.sessions {
            let day = Calendar.current.startOfDay(for: session.createdAt)
            counts[day, default: 0] += 1
        }
        return counts
    }

    private func dateChip(label: String, isSelected: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(label)
                .font(.system(size: 10, weight: isSelected ? .semibold : .regular))
                .foregroundStyle(isSelected ? .primary : .secondary)
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(
                    isSelected
                        ? AnyShapeStyle(Color.primary.opacity(0.08))
                        : AnyShapeStyle(.clear)
                )
                .clipShape(Capsule())
                .animation(.easeOut(duration: 0.15), value: isSelected)
        }
        .buttonStyle(.plain)
    }

    // MARK: - Session Groups

    private static let groupHeaderFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "EEEE, MMMM d"
        return f
    }()

    private var sessionGroups: some View {
        VStack(spacing: 16) {
            ForEach(groupedSessions, id: \.date) { group in
                VStack(alignment: .leading, spacing: 8) {
                    // Date group header
                    HStack(spacing: 6) {
                        let label = Calendar.current.isDateInToday(group.date) ? "Today"
                            : Calendar.current.isDateInYesterday(group.date) ? "Yesterday"
                            : Self.groupHeaderFormatter.string(from: group.date)
                        Text(label.uppercased())
                            .font(.system(size: 10, weight: .semibold, design: .rounded))
                            .foregroundStyle(.tertiary)
                            .tracking(0.6)

                        Text("(\(group.sessions.count))")
                            .font(.system(size: 9))
                            .foregroundStyle(.quaternary)
                    }
                    .padding(.leading, 4)

                    // Session cards
                    VStack(spacing: 6) {
                        ForEach(group.sessions) { session in
                            SessionCardView(
                                session: session,
                                selectionMode: isSelectingSessions,
                                isSelected: selectedSessionIds.contains(session.id),
                                onRename: { newName in
                                    sessionManager.renameSession(id: session.id, newName: newName)
                                }
                            )
                            .onTapGesture {
                                if isSelectingSessions {
                                    toggleSessionSelection(session.id)
                                } else {
                                    withAnimation(.easeInOut(duration: 0.15)) {
                                        selectedSessionId = session.id
                                    }
                                }
                            }
                            .contextMenu {
                                if session.hasTranscription, let text = session.transcriptionText {
                                    Button("Copy Transcription") {
                                        NSPasteboard.general.clearContents()
                                        NSPasteboard.general.setString(text, forType: .string)
                                    }
                                    Divider()
                                }
                                if session.hasAudio {
                                    Button("Show in Finder") {
                                        let audioURL = sessionManager.getAudioURL(for: session)
                                        NSWorkspace.shared.activateFileViewerSelecting([audioURL])
                                    }
                                    Button("Delete Audio Only") {
                                        sessionManager.deleteSessionAudio(id: session.id)
                                    }
                                    Divider()
                                }
                                Button("Delete Session", role: .destructive) {
                                    sessionManager.deleteSession(id: session.id)
                                }
                            }
                        }
                    }
                }
            }
        }
        .padding(.top, 12)
    }

    // MARK: - Empty / No Match

    private var emptyStateView: some View {
        VStack(spacing: 12) {
            Image(systemName: "waveform.circle")
                .font(.system(size: 36))
                .foregroundStyle(.tertiary)
            Text("No sessions yet")
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(.secondary)
            Text("Record your first session from the menu bar, or import an audio file.")
                .font(.system(size: 11))
                .foregroundStyle(.tertiary)

            Button {
                withAnimation(.easeInOut(duration: 0.15)) {
                    showImportPage = true
                }
            } label: {
                HStack(spacing: 5) {
                    Image(systemName: "square.and.arrow.down")
                        .font(.system(size: 10, weight: .medium))
                    Text("Import Audio File")
                        .font(.system(size: 11, weight: .medium))
                }
                .foregroundStyle(.white)
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .background(Color.accentColor, in: Capsule())
            }
            .buttonStyle(.plain)
            .padding(.top, 4)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 60)
    }

    private var noMatchView: some View {
        VStack(spacing: 8) {
            Image(systemName: "calendar.badge.exclamationmark")
                .font(.system(size: 24))
                .foregroundStyle(.tertiary)
            Text("No sessions on this date")
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(.secondary)
            Button("Show all") { selectedDate = nil }
                .font(.system(size: 11))
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 40)
    }

    // MARK: - Footer Actions

    private var footerActions: some View {
        HStack(spacing: 8) {
            Text(isSelectingSessions ? "\(selectedSessionIds.count) of \(sessionManager.sessions.count) selected" : "")
                .font(.system(size: 10))
                .foregroundStyle(.tertiary)

            Spacer()

            if isSelectingSessions {
                footerPill(label: allSessionsSelected ? "Clear All" : "Select All", tint: .secondary) {
                    toggleSelectAllSessions()
                }
            }
        }
    }

    private func footerPill(label: String, tint: Color, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(label)
                .font(.system(size: 10, weight: .medium))
                .foregroundStyle(tint)
                .padding(.horizontal, 10)
                .padding(.vertical, 5)
                .background(.thinMaterial, in: Capsule())
        }
        .buttonStyle(.plain)
    }

    // MARK: - Audio Mix Helpers

    private var latestSessionWithAudio: Session? {
        sessionManager.sessions.first { $0.hasAudio }
    }

    private var latestSessionAudioURL: URL? {
        guard let session = latestSessionWithAudio else { return nil }
        let url = sessionManager.getAudioURL(for: session)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    // MARK: - Selection Helpers

    private var allSessionsSelected: Bool {
        !sessionManager.sessions.isEmpty && selectedSessionIds.count == sessionManager.sessions.count
    }

    private func toggleSessionSelection(_ sessionId: UUID) {
        if selectedSessionIds.contains(sessionId) {
            selectedSessionIds.remove(sessionId)
        } else {
            selectedSessionIds.insert(sessionId)
        }
    }

    private func toggleSelectAllSessions() {
        if allSessionsSelected {
            selectedSessionIds.removeAll()
        } else {
            selectedSessionIds = Set(sessionManager.sessions.map(\.id))
        }
    }

    private func exitSelectionMode() {
        isSelectingSessions = false
        selectedSessionIds.removeAll()
    }

    // MARK: - Automation Helpers

    private func normalizeAutomationSelections() {
        let firstLocalSessionModel = localSessionModels.first?.id
        if let firstLocalSessionModel,
           !localSessionModels.contains(where: { $0.id == automationLocalTranscriptionModelId }) {
            automationLocalTranscriptionModelId = firstLocalSessionModel
        }

        if let publicModel = selectedPublicSessionModel {
            automationTranscriptionProviderId = publicModel.providerId
            automationTranscriptionModelId = publicModel.modelId
        }

        if let firstSummaryModel = summaryModels.first,
           !summaryModels.contains(where: { $0.id == automationLocalSummaryModelId }) {
            automationLocalSummaryModelId = firstSummaryModel.id
        }

        if let publicSummaryModel = selectedPublicSummaryModel {
            automationSummaryProviderId = publicSummaryModel.providerId
            automationSummaryModelId = publicSummaryModel.modelId
        }

        let validPresetIds = Set(automationSummaryPresets.map(\.id))
        automationSelectedPresetIds = automationSelectedPresetIds.intersection(validPresetIds)
        if automationSelectedPresetIds.isEmpty,
           let fallbackPresetId = automationSummaryPresets.first?.id {
            automationSelectedPresetIds = [fallbackPresetId]
        }
        persistAutomationPresetSelection()
    }

    private func toggleAutomationPreset(_ presetId: String) {
        if automationSelectedPresetIds.contains(presetId) {
            automationSelectedPresetIds.remove(presetId)
        } else {
            automationSelectedPresetIds.insert(presetId)
        }

        if automationSelectedPresetIds.isEmpty {
            automationSelectedPresetIds.insert(presetId)
        }

        persistAutomationPresetSelection()
    }

    private func persistAutomationPresetSelection() {
        let orderedIds = automationSummaryPresets
            .map(\.id)
            .filter { automationSelectedPresetIds.contains($0) }
        UserDefaults.standard.set(orderedIds, forKey: SessionAutomationPreferences.summaryPresetIdsKey)
    }
}

private struct FlowLayout: Layout {
    var spacing: CGFloat = 8
    var rowSpacing: CGFloat? = nil

    func sizeThatFits(
        proposal: ProposedViewSize,
        subviews: Subviews,
        cache: inout ()
    ) -> CGSize {
        let maxWidth = proposal.width ?? 320
        let rows = layoutRows(for: subviews, maxWidth: maxWidth)
        let verticalSpacing = rowSpacing ?? spacing
        return CGSize(
            width: maxWidth,
            height: rows.reduce(CGFloat.zero) { $0 + $1.height } + CGFloat(max(rows.count - 1, 0)) * verticalSpacing
        )
    }

    func placeSubviews(
        in bounds: CGRect,
        proposal: ProposedViewSize,
        subviews: Subviews,
        cache: inout ()
    ) {
        let rows = layoutRows(for: subviews, maxWidth: bounds.width)
        let verticalSpacing = rowSpacing ?? spacing
        var y = bounds.minY

        for row in rows {
            var x = bounds.minX
            for item in row.items {
                item.subview.place(
                    at: CGPoint(x: x, y: y),
                    anchor: .topLeading,
                    proposal: ProposedViewSize(item.size)
                )
                x += item.size.width + spacing
            }
            y += row.height + verticalSpacing
        }
    }

    private func layoutRows(for subviews: Subviews, maxWidth: CGFloat) -> [FlowRow] {
        var rows: [FlowRow] = []
        var currentItems: [FlowItem] = []
        var currentWidth: CGFloat = 0
        var currentHeight: CGFloat = 0

        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            let proposedWidth = currentItems.isEmpty ? size.width : currentWidth + spacing + size.width

            if proposedWidth > maxWidth, !currentItems.isEmpty {
                rows.append(FlowRow(items: currentItems, height: currentHeight))
                currentItems = [FlowItem(subview: subview, size: size)]
                currentWidth = size.width
                currentHeight = size.height
            } else {
                currentItems.append(FlowItem(subview: subview, size: size))
                currentWidth = proposedWidth
                currentHeight = max(currentHeight, size.height)
            }
        }

        if !currentItems.isEmpty {
            rows.append(FlowRow(items: currentItems, height: currentHeight))
        }

        return rows
    }

    private struct FlowRow {
        let items: [FlowItem]
        let height: CGFloat
    }

    private struct FlowItem {
        let subview: LayoutSubview
        let size: CGSize
    }
}

// MARK: - Audio Mix Sheet

private struct SessionSheetActionPillStyle: ViewModifier {
    let tint: Color
    let prominent: Bool

    func body(content: Content) -> some View {
        content
            .font(.system(size: 10, weight: .semibold))
            .foregroundStyle(prominent ? .white : tint)
            .padding(.horizontal, 12)
            .frame(height: 28)
            .background(prominent ? AnyShapeStyle(tint) : AnyShapeStyle(Color.primary.opacity(0.075)), in: Capsule())
            .overlay(Capsule().strokeBorder(prominent ? tint.opacity(0.28) : .white.opacity(0.06), lineWidth: 0.6))
            .contentShape(Capsule())
    }
}

struct AudioMixSheetView: View {
    @Binding var micTrimDB: Double
    @Binding var systemTrimDB: Double
    let latestSessionName: String?
    let latestAudioURL: URL?
    @Environment(\.dismiss) var dismiss

    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack(spacing: 16) {
                Image(systemName: "slider.horizontal.3")
                    .font(.system(size: 36))
                    .foregroundStyle(.orange.gradient)

                VStack(alignment: .leading, spacing: 4) {
                    Text("Session Audio Mix")
                        .font(.title2)
                        .fontWeight(.semibold)
                    Text("Adjust mic and desktop levels for session recordings")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                Spacer()
            }
            .padding(20)

            Divider()

            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    // Mic slider
                    mixSlider(
                        title: "Microphone Boost",
                        icon: "mic.fill",
                        value: $micTrimDB,
                        range: SessionAudioMixPreferences.micTrimRange,
                        valueStr: SessionAudioMixPreferences.displayString(for: micTrimDB),
                        gainStr: SessionAudioMixPreferences.multiplierString(
                            for: SessionAudioMixPreferences.effectiveMicGain(forTrimDB: micTrimDB)
                        ),
                        minLabel: "Less",
                        maxLabel: "More",
                        tint: .blue
                    )

                    // System slider
                    mixSlider(
                        title: "Desktop Audio",
                        icon: "desktopcomputer",
                        value: $systemTrimDB,
                        range: SessionAudioMixPreferences.systemTrimRange,
                        valueStr: SessionAudioMixPreferences.displayString(for: systemTrimDB),
                        gainStr: SessionAudioMixPreferences.multiplierString(
                            for: SessionAudioMixPreferences.effectiveSystemGain(forTrimDB: systemTrimDB)
                        ),
                        minLabel: "Quieter",
                        maxLabel: "Louder",
                        tint: .purple
                    )

                    Divider()

                    // Quick check
                    VStack(alignment: .leading, spacing: 10) {
                        Text("QUICK CHECK")
                            .font(.system(size: 10, weight: .semibold, design: .rounded))
                            .foregroundStyle(.tertiary)
                            .tracking(0.6)

                        if let audioURL = latestAudioURL, let name = latestSessionName {
                            HStack(spacing: 8) {
                                Button {
                                    NSWorkspace.shared.open(audioURL)
                                } label: {
                                    HStack(spacing: 4) {
                                        Image(systemName: "play.fill")
                                            .font(.system(size: 9))
                                        Text("Play Latest")
                                            .font(.system(size: 11, weight: .medium))
                                    }
                                    .foregroundStyle(.white)
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 7)
                                    .background(.orange, in: Capsule())
                                }
                                .buttonStyle(.plain)

                                Button {
                                    NSWorkspace.shared.activateFileViewerSelecting([audioURL])
                                } label: {
                                    HStack(spacing: 4) {
                                        Image(systemName: "folder")
                                            .font(.system(size: 9))
                                        Text("Show in Finder")
                                            .font(.system(size: 11, weight: .medium))
                                    }
                                    .foregroundStyle(.primary)
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 7)
                                    .background(.quaternary, in: Capsule())
                                }
                                .buttonStyle(.plain)

                                Spacer()
                            }

                            Text(name)
                                .font(.system(size: 10))
                                .foregroundStyle(.tertiary)
                        } else {
                            Text("Record a session to audition your mix here.")
                                .font(.system(size: 11))
                                .foregroundStyle(.tertiary)
                        }
                    }

                    Text("These settings only affect session recordings. Basic recording is not affected.")
                        .font(.system(size: 10))
                        .foregroundStyle(.quaternary)
                }
                .padding(24)
            }

            Divider()

            // Footer
            HStack {
                Button("Reset") {
                    micTrimDB = SessionAudioMixPreferences.defaultMicTrimDB
                    systemTrimDB = SessionAudioMixPreferences.defaultSystemTrimDB
                }
                .buttonStyle(.plain)
                .modifier(SessionSheetActionPillStyle(tint: .secondary, prominent: false))
                .disabled(abs(micTrimDB) < 0.05 && abs(systemTrimDB) < 0.05)

                Spacer()

                Button("Done") { dismiss() }
                    .buttonStyle(.plain)
                    .modifier(SessionSheetActionPillStyle(tint: .accentColor, prominent: true))
                    .keyboardShortcut(.defaultAction)
            }
            .padding(20)
        }
        .frame(width: 500, height: 480)
    }

    private func mixSlider(
        title: String,
        icon: String,
        value: Binding<Double>,
        range: ClosedRange<Double>,
        valueStr: String,
        gainStr: String,
        minLabel: String,
        maxLabel: String,
        tint: Color
    ) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: icon)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(tint)
                Text(title)
                    .font(.system(size: 13, weight: .semibold))
                Spacer()
                Text(valueStr)
                    .font(.system(size: 12, weight: .medium, design: .monospaced))
                Text("· \(gainStr)")
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(.secondary)
            }

            Slider(
                value: value,
                in: range,
                step: 0.5,
                minimumValueLabel: Text(minLabel).font(.system(size: 9)).foregroundStyle(.tertiary),
                maximumValueLabel: Text(maxLabel).font(.system(size: 9)).foregroundStyle(.tertiary)
            ) { Text(title) }
            .tint(tint)
        }
    }
}

// MARK: - Session Calendar Popover

struct SessionCalendarPopover: View {
    let sessionDates: [Date: Int]
    @Binding var selectedDate: Date?
    @Binding var isPresented: Bool

    @State private var displayedMonth: Date

    private let calendar = Calendar.current

    private static let monthFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "MMMM yyyy"
        return f
    }()

    private static let infoDateFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "MMM d, yyyy"
        return f
    }()

    init(sessionDates: [Date: Int], selectedDate: Binding<Date?>, isPresented: Binding<Bool>) {
        self.sessionDates = sessionDates
        self._selectedDate = selectedDate
        self._isPresented = isPresented
        let month = selectedDate.wrappedValue ?? Date()
        self._displayedMonth = State(initialValue: Calendar.current.startOfDay(for: month))
    }

    private var monthStart: Date {
        let comps = calendar.dateComponents([.year, .month], from: displayedMonth)
        return calendar.date(from: comps)!
    }

    private var daysInMonth: Int {
        calendar.range(of: .day, in: .month, for: monthStart)!.count
    }

    private var firstWeekdayOffset: Int {
        (calendar.component(.weekday, from: monthStart) - calendar.firstWeekday + 7) % 7
    }

    private var adjustedWeekdaySymbols: [String] {
        let symbols = calendar.veryShortWeekdaySymbols
        let first = calendar.firstWeekday - 1
        return Array(symbols[first...]) + Array(symbols[..<first])
    }

    private var sessionCountForMonth: Int {
        (1...daysInMonth).reduce(0) { total, day in
            let date = calendar.date(byAdding: .day, value: day - 1, to: monthStart)!
            return total + (sessionDates[calendar.startOfDay(for: date)] ?? 0)
        }
    }

    private var isCurrentMonth: Bool {
        calendar.isDate(monthStart, equalTo: Date(), toGranularity: .month)
    }

    var body: some View {
        VStack(spacing: 12) {
            // Month navigation
            HStack {
                Button { goToPreviousMonth() } label: {
                    Image(systemName: "chevron.left")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(.secondary)
                        .frame(width: 28, height: 28)
                        .background(.quaternary, in: .circle)
                }
                .buttonStyle(.plain)

                Spacer()

                VStack(spacing: 2) {
                    Text(Self.monthFormatter.string(from: displayedMonth))
                        .font(.system(size: 13, weight: .semibold))
                    if sessionCountForMonth > 0 {
                        Text("\(sessionCountForMonth) session\(sessionCountForMonth == 1 ? "" : "s")")
                            .font(.system(size: 9))
                            .foregroundStyle(.secondary)
                    }
                }

                Spacer()

                Button { goToNextMonth() } label: {
                    Image(systemName: "chevron.right")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(.secondary)
                        .frame(width: 28, height: 28)
                        .background(.quaternary, in: .circle)
                }
                .buttonStyle(.plain)
                .disabled(isCurrentMonth || monthStart > Date())
            }

            // Weekday headers
            HStack(spacing: 0) {
                ForEach(adjustedWeekdaySymbols, id: \.self) { symbol in
                    Text(symbol)
                        .font(.system(size: 9, weight: .medium))
                        .foregroundStyle(.tertiary)
                        .frame(maxWidth: .infinity)
                }
            }

            // Day grid
            let columns = Array(repeating: GridItem(.flexible(), spacing: 2), count: 7)
            LazyVGrid(columns: columns, spacing: 4) {
                ForEach(0..<firstWeekdayOffset, id: \.self) { _ in
                    Color.clear.frame(height: 34)
                }

                ForEach(1...daysInMonth, id: \.self) { day in
                    let date = calendar.date(byAdding: .day, value: day - 1, to: monthStart)!
                    dayCell(day: day, date: date)
                }
            }

            // Selected date info
            if let sel = selectedDate {
                let count = sessionDates[calendar.startOfDay(for: sel)] ?? 0
                HStack(spacing: 6) {
                    Text(Self.infoDateFormatter.string(from: sel))
                        .font(.system(size: 10, weight: .semibold))
                    Text("·")
                        .foregroundStyle(.tertiary)
                    Text("\(count) session\(count == 1 ? "" : "s")")
                        .font(.system(size: 10))
                        .foregroundStyle(count > 0 ? .green : .secondary)
                }
            }
        }
        .padding(16)
        .frame(width: 270)
    }

    private func dayCell(day: Int, date: Date) -> some View {
        let isToday = calendar.isDateInToday(date)
        let isSelected = selectedDate.map { calendar.isDate($0, inSameDayAs: date) } ?? false
        let count = sessionDates[calendar.startOfDay(for: date)] ?? 0
        let hasSession = count > 0
        let isFuture = date > Date() && !isToday

        return Button {
            if isSelected {
                selectedDate = nil
            } else {
                selectedDate = date
            }
            isPresented = false
        } label: {
            VStack(spacing: 2) {
                Text("\(day)")
                    .font(.system(size: 12, weight: isToday ? .bold : (hasSession ? .semibold : .regular)))
                    .foregroundStyle(
                        isSelected ? .white :
                        isFuture ? .secondary.opacity(0.4) :
                        hasSession ? .primary : .secondary
                    )

                Circle()
                    .fill(hasSession ? .green : .clear)
                    .frame(width: 4, height: 4)
            }
            .frame(maxWidth: .infinity)
            .frame(height: 34)
            .background(
                isSelected
                    ? AnyShapeStyle(Color.accentColor)
                    : isToday
                        ? AnyShapeStyle(Color.accentColor.opacity(0.1))
                        : AnyShapeStyle(.clear)
            )
            .clipShape(.rect(cornerRadius: 6, style: .continuous))
        }
        .buttonStyle(.plain)
        .disabled(isFuture)
    }

    private func goToPreviousMonth() {
        displayedMonth = calendar.date(byAdding: .month, value: -1, to: displayedMonth) ?? displayedMonth
    }

    private func goToNextMonth() {
        displayedMonth = calendar.date(byAdding: .month, value: 1, to: displayedMonth) ?? displayedMonth
    }
}
