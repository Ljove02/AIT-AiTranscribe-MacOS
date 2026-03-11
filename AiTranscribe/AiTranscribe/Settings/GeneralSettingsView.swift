import SwiftUI
import AVFoundation

// MARK: - General Settings View

struct GeneralSettingsView: View {
    @EnvironmentObject var appState: AppState

    /// Auto-paste after transcription completes (simulates Cmd+V)
    @AppStorage("autoPasteAfterTranscription") private var autoPasteAfterTranscription = false

    /// Auto-start recording on app launch
    @AppStorage("autoLoadModel") private var autoLoadModel = true

    /// Play sound when recording starts/stops
    @AppStorage("playSounds") private var playSounds = true

    /// Recording indicator position
    @AppStorage("indicatorPosition") private var indicatorPosition = "topCenter"

    var body: some View {
        Form {
            Section("Transcription") {
                Toggle("Auto-paste after transcription", isOn: $autoPasteAfterTranscription)
                    .help("Transcribed text is copied to clipboard. When enabled, it also pastes automatically at cursor position.")
                    .onChange(of: autoPasteAfterTranscription) { _, newValue in
                        if newValue {
                            appState.checkAccessibilityPermissionsIfNeeded()
                        }
                    }
            }

            Section("Recording Indicator") {
                Picker("Position", selection: $indicatorPosition) {
                    ForEach(RecordingIndicatorController.ScreenPosition.allCases, id: \.rawValue) { pos in
                        Text(pos.displayName).tag(pos.rawValue)
                    }
                }
                .onChange(of: indicatorPosition) { _, newValue in
                    if let pos = RecordingIndicatorController.ScreenPosition(rawValue: newValue) {
                        appState.recordingIndicator.setPosition(pos)
                    }
                }
            }

            Section("Startup") {
                Toggle("Load model on app launch", isOn: $autoLoadModel)
            }

            Section("Feedback") {
                Toggle("Play sounds", isOn: $playSounds)
            }

            Section("Audio") {
                AudioDuckingSettings()
            }

            Section("Permissions") {
                // Microphone permission status
                HStack {
                    Text("Microphone Access")
                    Spacer()
                    microphoneStatusView
                }

                // Request permission button (only show if not determined or denied)
                if microphoneStatus != .authorized {
                    Button("Request Microphone Permission") {
                        requestMicrophonePermission()
                    }
                    .buttonStyle(.borderedProminent)
                }

                // Open System Settings button
                Button("Open Microphone Settings...") {
                    openMicrophoneSettings()
                }

                Text("If the app doesn't appear in Microphone settings, click 'Request Microphone Permission' first, then check again.")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .formStyle(.grouped)
        .padding()
    }

    // MARK: - Microphone Permission

    /// Current microphone authorization status
    private var microphoneStatus: AVAuthorizationStatus {
        AVCaptureDevice.authorizationStatus(for: .audio)
    }

    /// View showing microphone permission status
    @ViewBuilder
    private var microphoneStatusView: some View {
        switch microphoneStatus {
        case .authorized:
            HStack(spacing: 4) {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundColor(.green)
                Text("Granted")
                    .foregroundColor(.green)
            }
        case .denied, .restricted:
            HStack(spacing: 4) {
                Image(systemName: "xmark.circle.fill")
                    .foregroundColor(.red)
                Text("Denied")
                    .foregroundColor(.red)
            }
        case .notDetermined:
            HStack(spacing: 4) {
                Image(systemName: "questionmark.circle.fill")
                    .foregroundColor(.orange)
                Text("Not Requested")
                    .foregroundColor(.orange)
            }
        @unknown default:
            Text("Unknown")
                .foregroundColor(.secondary)
        }
    }

    /// Request microphone permission - this triggers the system dialog
    private func requestMicrophonePermission() {
        AVCaptureDevice.requestAccess(for: .audio) { granted in
            DispatchQueue.main.async {
                if granted {
                    appState.statusMessage = "Microphone permission granted"
                } else {
                    appState.statusMessage = "Microphone permission denied - enable in System Settings"
                }
            }
        }
    }

    /// Open System Settings > Privacy & Security > Microphone
    private func openMicrophoneSettings() {
        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone") {
            NSWorkspace.shared.open(url)
        }
    }

    /// Open System Settings > Privacy & Security > Accessibility
    private func openAccessibilitySettings() {
        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility") {
            NSWorkspace.shared.open(url)
        }
    }
}
