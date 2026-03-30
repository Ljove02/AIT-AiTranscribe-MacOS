import SwiftUI
import AppKit
import ScreenCaptureKit

// MARK: - Sessions Settings View

/// Main container for the Sessions tab — shows a list of recorded sessions
/// or navigates to a detail view when a session is selected.
struct SessionsSettingsView: View {
    @EnvironmentObject var sessionManager: SessionManager
    var onNavigateToModels: (() -> Void)? = nil
    @State private var selectedSessionId: UUID? = nil
    @State private var showDeleteAllAlert = false
    @State private var showDeleteTranscribedAlert = false
    @State private var showDeleteSelectedSessionsAlert = false
    @State private var screenRecordingPermission = false
    @State private var isSessionAudioMixExpanded = false
    @State private var isSelectingSessions = false
    @State private var selectedSessionIds: Set<UUID> = []
    @AppStorage(SessionAudioMixPreferences.micTrimDBKey) private var sessionMicTrimDB = SessionAudioMixPreferences.defaultMicTrimDB
    @AppStorage(SessionAudioMixPreferences.systemTrimDBKey) private var sessionSystemTrimDB = SessionAudioMixPreferences.defaultSystemTrimDB

    var body: some View {
        VStack(spacing: 0) {
            if let selectedId = selectedSessionId,
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
        .onChange(of: sessionManager.sessions) { _, sessions in
            let validIds = Set(sessions.map(\.id))
            selectedSessionIds = selectedSessionIds.intersection(validIds)
            if sessions.isEmpty {
                isSelectingSessions = false
            }
        }
    }

    // MARK: - Sessions List

    private var sessionsListView: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text("Sessions")
                    .font(.title2.bold())

                Spacer()

                // Total storage display
                let totalSize = sessionManager.getTotalStorageSize()
                if totalSize > 0 {
                    Text("Storage: \(SessionManager.formatFileSize(totalSize))")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3)
                        .background(
                            Capsule()
                                .fill(Color.secondary.opacity(0.1))
                        )
                }

                Button(isSelectingSessions ? "Done" : "Select") {
                    if isSelectingSessions {
                        exitSelectionMode()
                    } else {
                        isSelectingSessions = true
                    }
                }
                .font(.caption)
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(sessionManager.sessions.isEmpty)
            }
            .padding(.horizontal, 16)
            .padding(.top, 16)
            .padding(.bottom, 8)

            // Info banner
            sessionInfoBanner
                .padding(.horizontal, 16)
                .padding(.bottom, 8)

            // Permission status
            permissionStatusView
                .padding(.horizontal, 16)
                .padding(.bottom, 8)

            sessionAudioMixDisclosure
                .padding(.horizontal, 16)
                .padding(.bottom, 8)

            Divider()

            // Session list or empty state
            if sessionManager.sessions.isEmpty {
                emptyStateView
            } else {
                ScrollView {
                    LazyVStack(spacing: 0) {
                        ForEach(sessionManager.sessions) { session in
                            SessionCardView(
                                session: session,
                                selectionMode: isSelectingSessions,
                                isSelected: selectedSessionIds.contains(session.id)
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

                            Divider()
                        }
                    }
                }
            }

            Divider()

            // Footer with bulk actions
            footerView
                .padding(8)
        }
    }

    // MARK: - Info Banner

    @AppStorage("hideSessionInfoBanner") private var hideBanner = false

    private var sessionInfoBanner: some View {
        Group {
            if !hideBanner {
                VStack(alignment: .leading, spacing: 8) {
                    HStack(alignment: .top) {
                        Image(systemName: "info.circle.fill")
                            .foregroundColor(.blue)
                            .font(.title3)

                        VStack(alignment: .leading, spacing: 4) {
                            Text("Record long-form audio sessions")
                                .font(.subheadline.weight(.semibold))

                            Text("Sessions capture your full audio — meetings, lectures, interviews, or any longer recording. Start a session from the menu bar icon, then transcribe it here when you're done.")
                                .font(.caption)
                                .foregroundColor(.secondary)
                                .fixedSize(horizontal: false, vertical: true)

                            HStack(spacing: 4) {
                                Image(systemName: "menubar.arrow.up.rectangle")
                                    .font(.caption2)
                                Text("Menu bar")
                                Image(systemName: "arrow.right")
                                    .font(.caption2)
                                Text("Record Session")
                                    .fontWeight(.medium)
                            }
                            .font(.caption)
                            .foregroundColor(.blue)
                            .padding(.top, 2)
                        }

                        Spacer()

                        Button {
                            withAnimation { hideBanner = true }
                        } label: {
                            Image(systemName: "xmark")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        .buttonStyle(.plain)
                        .help("Dismiss")
                    }
                }
                .padding(12)
                .background(Color.blue.opacity(0.06))
                .cornerRadius(8)
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .strokeBorder(Color.blue.opacity(0.15), lineWidth: 1)
                )
            }
        }
    }

    // MARK: - Permission Status

    private var permissionStatusView: some View {
        HStack(spacing: 12) {
            // Microphone permission
            HStack(spacing: 4) {
                Circle()
                    .fill(AudioRecorder.hasMicrophonePermission ? Color.green : Color.red)
                    .frame(width: 6, height: 6)
                Text("Microphone")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            // Screen Recording permission
            HStack(spacing: 4) {
                Circle()
                    .fill(screenRecordingPermission ? Color.green : Color.red)
                    .frame(width: 6, height: 6)
                Text("Screen Recording")
                    .font(.caption)
                    .foregroundColor(.secondary)

                if !screenRecordingPermission {
                    Button("Grant") {
                        SystemAudioCapture.requestPermission()
                        // Recheck after a delay
                        Task {
                            try? await Task.sleep(for: .seconds(2))
                            screenRecordingPermission = await SystemAudioCapture.checkPermission()
                        }
                    }
                    .font(.caption)
                    .buttonStyle(.bordered)
                    .controlSize(.mini)
                }
            }

            Spacer()
        }
    }

    // MARK: - Session Audio Mix

    private var latestSessionWithAudio: Session? {
        sessionManager.sessions.first { $0.hasAudio }
    }

    private var latestSessionAudioURL: URL? {
        guard let latestSessionWithAudio else { return nil }
        let url = sessionManager.getAudioURL(for: latestSessionWithAudio)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    private var sessionAudioMixDisclosure: some View {
        DisclosureGroup(isExpanded: $isSessionAudioMixExpanded) {
            sessionAudioMixControls
                .padding(.top, 10)
        } label: {
            HStack(alignment: .center, spacing: 10) {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Session Audio Mix")
                        .font(.subheadline.weight(.semibold))
                    Text("Compact controls for session mic and desktop levels.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                Spacer()

                Text("Mic \(SessionAudioMixPreferences.displayString(for: sessionMicTrimDB))")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                    .monospacedDigit()

                Text("Desktop \(SessionAudioMixPreferences.displayString(for: sessionSystemTrimDB))")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                    .monospacedDigit()
            }
        }
        .padding(12)
        .background(Color.orange.opacity(0.06))
        .cornerRadius(8)
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .strokeBorder(Color.orange.opacity(0.15), lineWidth: 1)
        )
    }

    private var sessionAudioMixControls: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Session recordings only. Basic recording is not affected.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                Spacer()

                Button("Reset") {
                    sessionMicTrimDB = SessionAudioMixPreferences.defaultMicTrimDB
                    sessionSystemTrimDB = SessionAudioMixPreferences.defaultSystemTrimDB
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(abs(sessionMicTrimDB) < 0.05 && abs(sessionSystemTrimDB) < 0.05)
                .help("Reset both session mix trims back to the default tuned profile.")
            }

            sessionMixSlider(
                title: "Microphone Boost",
                value: $sessionMicTrimDB,
                range: SessionAudioMixPreferences.micTrimRange,
                valueDescription: SessionAudioMixPreferences.displayString(for: sessionMicTrimDB),
                effectiveGainDescription: SessionAudioMixPreferences.multiplierString(
                    for: SessionAudioMixPreferences.effectiveMicGain(forTrimDB: sessionMicTrimDB)
                ),
                minimumLabel: "Less",
                maximumLabel: "More",
                helpText: "Adds trim on top of the current session mic baseline. Increase this if Google Meet still makes your voice too quiet."
            )

            sessionMixSlider(
                title: "Desktop Audio Level",
                value: $sessionSystemTrimDB,
                range: SessionAudioMixPreferences.systemTrimRange,
                valueDescription: SessionAudioMixPreferences.displayString(for: sessionSystemTrimDB),
                effectiveGainDescription: SessionAudioMixPreferences.multiplierString(
                    for: SessionAudioMixPreferences.effectiveSystemGain(forTrimDB: sessionSystemTrimDB)
                ),
                minimumLabel: "Quieter",
                maximumLabel: "Louder",
                helpText: "Adjusts the captured call/app audio in session recordings. Lower this if Google Meet or desktop playback is overwhelming your microphone."
            )

            Divider()

            VStack(alignment: .leading, spacing: 8) {
                Text("Quick Check")
                    .font(.subheadline.weight(.semibold))

                if let session = latestSessionWithAudio, let audioURL = latestSessionAudioURL {
                    HStack(spacing: 8) {
                        Button("Play Latest Session Audio") {
                            NSWorkspace.shared.open(audioURL)
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.small)
                        .help("Open the newest recorded session audio in your default player.")

                        Button("Show in Finder") {
                            NSWorkspace.shared.activateFileViewerSelecting([audioURL])
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)

                        Spacer()
                    }

                    Text("Latest: \(session.name)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                } else {
                    Text("Record a session, then use the latest audio button here to audition your current session mix.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
        }
    }

    @ViewBuilder
    private func sessionMixSlider(
        title: String,
        value: Binding<Double>,
        range: ClosedRange<Double>,
        valueDescription: String,
        effectiveGainDescription: String,
        minimumLabel: String,
        maximumLabel: String,
        helpText: String
    ) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline) {
                Text(title)
                    .font(.subheadline.weight(.medium))

                Spacer()

                Text(valueDescription)
                    .font(.caption.weight(.medium))
                    .monospacedDigit()

                Text("• \(effectiveGainDescription)")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .monospacedDigit()
            }

            Slider(
                value: value,
                in: range,
                step: 0.5,
                minimumValueLabel: Text(minimumLabel).font(.caption2).foregroundColor(.secondary),
                maximumValueLabel: Text(maximumLabel).font(.caption2).foregroundColor(.secondary)
            ) {
                Text(title)
            }

            Text(helpText)
                .font(.caption2)
                .foregroundColor(.secondary)
        }
    }

    // MARK: - Empty State

    private var emptyStateView: some View {
        VStack(spacing: 12) {
            Image(systemName: "waveform.circle")
                .font(.system(size: 40))
                .foregroundColor(.secondary)
            Text("No sessions yet")
                .font(.headline)
                .foregroundColor(.secondary)
            Text("Record your first session from the menu bar.")
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Footer

    private var footerView: some View {
        HStack {
            Text(footerSummaryText)
                .font(.caption)
                .foregroundColor(.secondary)

            Spacer()

            if isSelectingSessions {
                Button(allSessionsSelected ? "Clear All" : "Select All") {
                    toggleSelectAllSessions()
                }
                .font(.caption)

                Button("Delete Selected") {
                    showDeleteSelectedSessionsAlert = true
                }
                .font(.caption)
                .foregroundColor(.red)
                .disabled(selectedSessionIds.isEmpty)
                .alert("Delete Selected Sessions?", isPresented: $showDeleteSelectedSessionsAlert) {
                    Button("Cancel", role: .cancel) {}
                    Button("Delete", role: .destructive) {
                        sessionManager.bulkDeleteSessions(ids: selectedSessionIds)
                        exitSelectionMode()
                    }
                } message: {
                    Text("Selected sessions will be deleted completely, including audio, transcription, and metadata.")
                }

                Button("Done") {
                    exitSelectionMode()
                }
                .font(.caption)
            } else {
            let hasAudio = sessionManager.sessions.contains(where: { $0.hasAudio })
            let hasTranscribedAudio = sessionManager.sessions.contains(where: { $0.hasAudio && $0.hasTranscription })

            if hasTranscribedAudio {
                Button("Clean Transcribed Audio") {
                    showDeleteTranscribedAlert = true
                }
                .font(.caption)
                .alert("Delete Transcribed Audio?", isPresented: $showDeleteTranscribedAlert) {
                    Button("Cancel", role: .cancel) {}
                    Button("Delete", role: .destructive) {
                        sessionManager.bulkDeleteAudio(transcribedOnly: true)
                    }
                } message: {
                    Text("Audio files for sessions that have been transcribed will be deleted. Transcriptions will be kept.")
                }
            }

            if hasAudio {
                Button("Delete All Audio") {
                    showDeleteAllAlert = true
                }
                .font(.caption)
                .foregroundColor(.red)
                .alert("Delete All Audio?", isPresented: $showDeleteAllAlert) {
                    Button("Cancel", role: .cancel) {}
                    Button("Delete All", role: .destructive) {
                        sessionManager.bulkDeleteAudio(transcribedOnly: false)
                    }
                } message: {
                    let totalSize = SessionManager.formatFileSize(sessionManager.getTotalStorageSize())
                    Text("All audio files (\(totalSize)) will be deleted. Transcriptions will be kept. You won't be able to re-transcribe.")
                }
            }
            }
        }
    }

    private var allSessionsSelected: Bool {
        !sessionManager.sessions.isEmpty && selectedSessionIds.count == sessionManager.sessions.count
    }

    private var footerSummaryText: String {
        if isSelectingSessions {
            return "\(selectedSessionIds.count) of \(sessionManager.sessions.count) selected"
        }
        return "\(sessionManager.sessions.count) sessions"
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
}
