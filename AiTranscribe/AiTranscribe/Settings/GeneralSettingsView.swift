import SwiftUI
import AVFoundation

// MARK: - General Settings View

struct GeneralSettingsView: View {
    @EnvironmentObject var appState: AppState
    let hasAnimated: Bool
    let onAnimated: () -> Void

    @AppStorage("autoPasteAfterTranscription") private var autoPasteAfterTranscription = false
    @AppStorage("playSounds") private var playSounds = true
    @AppStorage("indicatorPosition") private var indicatorPosition = "topCenter"

    // Init from hasAnimated so the very first frame is correct — no jump
    @State private var appeared: Bool

    init(hasAnimated: Bool, onAnimated: @escaping () -> Void) {
        self.hasAnimated = hasAnimated
        self.onAnimated = onAnimated
        _appeared = State(initialValue: hasAnimated)
    }

    var body: some View {
        ScrollView {
            VStack(spacing: 0) {
                // ── Title Header ─────────────────────────────────
                HStack(alignment: .firstTextBaseline) {
                    VStack(alignment: .leading, spacing: 3) {
                        Text("General")
                            .font(.system(size: 22, weight: .bold, design: .rounded))

                        Text("5 options")
                            .font(.system(size: 11))
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                }
                .staggerIn(index: 0, appeared: appeared)

                // Transcription section
                Group {
                    SettingsSectionHeader(title: "Transcription")

                    SettingsRow(
                        title: "Auto-paste after transcription",
                        description: "Transcribed text is copied to clipboard. When enabled, it also pastes automatically at cursor position."
                    ) {
                        Toggle("", isOn: $autoPasteAfterTranscription)
                            .toggleStyle(.switch)
                            .labelsHidden()
                            .onChange(of: autoPasteAfterTranscription) { _, newValue in
                                if newValue {
                                    appState.checkAccessibilityPermissionsIfNeeded()
                                }
                            }
                    }
                }
                .staggerIn(index: 1, appeared: appeared)

                SettingsDivider()
                    .staggerIn(index: 2, appeared: appeared)

                // Recording Indicator section
                Group {
                    SettingsSectionHeader(title: "Recording Indicator")

                    SettingsRow(
                        title: "Position",
                        description: "Where the recording indicator appears on screen."
                    ) {
                        Picker("", selection: $indicatorPosition) {
                            ForEach(RecordingIndicatorController.ScreenPosition.allCases, id: \.rawValue) { pos in
                                Text(pos.displayName).tag(pos.rawValue)
                            }
                        }
                        .labelsHidden()
                        .frame(width: 140)
                        .onChange(of: indicatorPosition) { _, newValue in
                            if let pos = RecordingIndicatorController.ScreenPosition(rawValue: newValue) {
                                appState.recordingIndicator.setPosition(pos)
                            }
                        }
                    }
                }
                .staggerIn(index: 3, appeared: appeared)

                SettingsDivider()
                    .staggerIn(index: 4, appeared: appeared)

                // Feedback section
                Group {
                    SettingsSectionHeader(title: "Feedback")

                    SettingsRow(
                        title: "Play sounds",
                        description: "Play audio feedback when recording starts and stops."
                    ) {
                        Toggle("", isOn: $playSounds)
                            .toggleStyle(.switch)
                            .labelsHidden()
                    }
                }
                .staggerIn(index: 5, appeared: appeared)

                SettingsDivider()
                    .staggerIn(index: 6, appeared: appeared)

                // Audio section
                Group {
                    SettingsSectionHeader(title: "Audio")

                    AudioDuckingSettings()
                }
                .staggerIn(index: 7, appeared: appeared)

                SettingsDivider()
                    .staggerIn(index: 8, appeared: appeared)

                // Permissions section
                Group {
                    SettingsSectionHeader(title: "Permissions")

                    SettingsRow(
                        title: "Microphone Access",
                        description: microphoneDescription
                    ) {
                        microphoneStatusBadge
                    }

                    if microphoneStatus != .authorized {
                        SettingsRow(
                            title: "Request Permission",
                            description: "If the app doesn't appear in Microphone settings, click this first."
                        ) {
                            Button("Request") {
                                requestMicrophonePermission()
                            }
                            .buttonStyle(.borderedProminent)
                            .controlSize(.small)
                        }
                    }

                    SettingsRow(
                        title: "System Settings",
                        description: "Open macOS Privacy & Security to manage microphone access."
                    ) {
                        Button("Open") {
                            openMicrophoneSettings()
                        }
                        .controlSize(.small)
                    }
                }
                .staggerIn(index: 9, appeared: appeared)

                // Bottom spacing
                Spacer(minLength: 20)
            }
            .padding(.horizontal, 28)
            .padding(.top, 8)
        }
        .scrollIndicators(.automatic)
        .task(id: "stagger") {
            guard !hasAnimated else { return }
            try? await Task.sleep(for: .milliseconds(80))
            appeared = true
            onAnimated()
        }
    }

    // MARK: - Microphone Permission

    private var microphoneStatus: AVAuthorizationStatus {
        AVCaptureDevice.authorizationStatus(for: .audio)
    }

    private var microphoneDescription: String {
        switch microphoneStatus {
        case .authorized:
            return "Microphone access is granted."
        case .denied, .restricted:
            return "Microphone access was denied. Enable it in System Settings."
        case .notDetermined:
            return "Microphone permission has not been requested yet."
        @unknown default:
            return "Unknown status."
        }
    }

    private var microphoneStatusBadge: some View {
        HStack(spacing: 5) {
            Circle()
                .fill(microphoneStatus == .authorized ? .green : (microphoneStatus == .notDetermined ? .orange : .red))
                .frame(width: 8, height: 8)

            Text(microphoneStatus == .authorized ? "Granted" : (microphoneStatus == .notDetermined ? "Not Requested" : "Denied"))
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(microphoneStatus == .authorized ? .green : (microphoneStatus == .notDetermined ? .orange : .red))
        }
    }

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

    private func openMicrophoneSettings() {
        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone") {
            NSWorkspace.shared.open(url)
        }
    }
}

// MARK: - Reusable Settings Components

/// Section header — small caps title above a group of rows
struct SettingsSectionHeader: View {
    let title: String

    var body: some View {
        HStack {
            Text(title.uppercased())
                .font(.system(size: 11, weight: .semibold, design: .rounded))
                .foregroundStyle(.secondary)
                .tracking(0.8)
            Spacer()
        }
        .padding(.top, 20)
        .padding(.bottom, 8)
    }
}

/// A single settings row: title + description on the left, control on the right
struct SettingsRow<Control: View>: View {
    let title: String
    let description: String
    @ViewBuilder let control: Control

    var body: some View {
        HStack(alignment: .center, spacing: 16) {
            VStack(alignment: .leading, spacing: 3) {
                Text(title)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(.primary)

                Text(description)
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
                    .lineLimit(3)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Spacer()

            control
        }
        .padding(.vertical, 10)
        .padding(.horizontal, 14)
        .background(.thinMaterial, in: .rect(cornerRadius: 10, style: .continuous))
        .padding(.vertical, 2)
    }
}

/// Transparent divider between sections
struct SettingsDivider: View {
    var body: some View {
        Rectangle()
            .fill(.white.opacity(0.06))
            .frame(height: 0.5)
            .padding(.horizontal, 4)
            .padding(.vertical, 4)
    }
}

// MARK: - Staggered Appear Modifier

/// Cascading slide-up animation for settings rows.
/// Uses offset + clip only (no opacity) — opacity breaks material compositing
/// and causes system controls (Toggle, Picker, Slider) to flash black.
struct StaggerInModifier: ViewModifier {
    let index: Int
    let appeared: Bool

    func body(content: Content) -> some View {
        content
            .offset(y: appeared ? 0 : 16)
            .animation(
                .spring(duration: 0.4, bounce: 0.12).delay(Double(index) * 0.05),
                value: appeared
            )
    }
}

extension View {
    func staggerIn(index: Int, appeared: Bool) -> some View {
        modifier(StaggerInModifier(index: index, appeared: appeared))
    }
}
