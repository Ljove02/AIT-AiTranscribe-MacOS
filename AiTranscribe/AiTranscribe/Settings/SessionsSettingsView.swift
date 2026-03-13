import SwiftUI
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
    @State private var screenRecordingPermission = false

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

            Divider()

            // Session list or empty state
            if sessionManager.sessions.isEmpty {
                emptyStateView
            } else {
                ScrollView {
                    LazyVStack(spacing: 0) {
                        ForEach(sessionManager.sessions) { session in
                            SessionCardView(session: session)
                                .onTapGesture {
                                    withAnimation(.easeInOut(duration: 0.15)) {
                                        selectedSessionId = session.id
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
            Text("\(sessionManager.sessions.count) sessions")
                .font(.caption)
                .foregroundColor(.secondary)

            Spacer()

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

