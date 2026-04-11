import SwiftUI
import Carbon.HIToolbox

// MARK: - Shortcuts Settings View

struct ShortcutsSettingsView: View {
    let hasAnimated: Bool
    let onAnimated: () -> Void

    // Recording shortcuts
    @AppStorage("toggleRecordingShortcut") private var toggleShortcut = "⌃P"
    @AppStorage("cancelRecordingShortcut") private var cancelShortcut = "⌃K"

    // Session recording shortcuts
    @AppStorage("toggleSessionShortcut") private var sessionToggleShortcut = "⌃⇧R"
    @AppStorage("stopSessionShortcut") private var sessionStopShortcut = "⌃⇧S"

    @State private var isRecordingToggleShortcut = false
    @State private var isRecordingCancelShortcut = false
    @State private var isRecordingSessionToggle = false
    @State private var isRecordingSessionStop = false

    /// Animation state
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
                        Text("Shortcuts")
                            .font(.system(size: 22, weight: .bold, design: .rounded))

                        Text("4 shortcuts")
                            .font(.system(size: 11))
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                }
                .staggerIn(index: 0, appeared: appeared)

                // ── Recording Shortcuts ──────────────────────────
                Group {
                    SettingsSectionHeader(title: "Recording")

                    SettingsRow(
                        title: "Toggle Recording",
                        description: "Start or stop a quick transcription recording."
                    ) {
                        ShortcutPill(
                            shortcut: $toggleShortcut,
                            isRecording: $isRecordingToggleShortcut,
                            onChanged: { HotkeyManager.shared.refreshHotkeys() }
                        )
                    }

                    SettingsRow(
                        title: "Cancel Recording",
                        description: "Discard the current recording without transcribing."
                    ) {
                        ShortcutPill(
                            shortcut: $cancelShortcut,
                            isRecording: $isRecordingCancelShortcut,
                            onChanged: { HotkeyManager.shared.refreshHotkeys() }
                        )
                    }
                }
                .staggerIn(index: 1, appeared: appeared)

                SettingsDivider()
                    .staggerIn(index: 2, appeared: appeared)

                // ── Session Shortcuts ────────────────────────────
                Group {
                    SettingsSectionHeader(title: "Session Recording")

                    SettingsRow(
                        title: "Start Session",
                        description: "Begin a long-form session recording with batch transcription."
                    ) {
                        ShortcutPill(
                            shortcut: $sessionToggleShortcut,
                            isRecording: $isRecordingSessionToggle,
                            onChanged: { HotkeyManager.shared.refreshHotkeys() }
                        )
                    }

                    SettingsRow(
                        title: "Stop Session",
                        description: "End the current session and save the recording."
                    ) {
                        ShortcutPill(
                            shortcut: $sessionStopShortcut,
                            isRecording: $isRecordingSessionStop,
                            onChanged: { HotkeyManager.shared.refreshHotkeys() }
                        )
                    }
                }
                .staggerIn(index: 3, appeared: appeared)

                SettingsDivider()
                    .staggerIn(index: 4, appeared: appeared)

                // ── Hint ─────────────────────────────────────────
                Group {
                    SettingsSectionHeader(title: "How It Works")

                    HStack(spacing: 12) {
                        Image(systemName: "keyboard")
                            .font(.system(size: 20, weight: .light))
                            .foregroundStyle(.tertiary)

                        VStack(alignment: .leading, spacing: 4) {
                            Text("Click any shortcut, then press your desired key combination.")
                                .font(.system(size: 12, weight: .medium))
                                .foregroundStyle(.secondary)
                            Text("Shortcuts work globally — even when the app is in the background.")
                                .font(.system(size: 11))
                                .foregroundStyle(.tertiary)
                        }

                        Spacer()
                    }
                    .padding(.vertical, 12)
                    .padding(.horizontal, 14)
                    .background(.thinMaterial, in: .rect(cornerRadius: 10, style: .continuous))
                }
                .staggerIn(index: 5, appeared: appeared)

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
}

// MARK: - Shortcut Pill

/// A capsule-style button that captures keyboard shortcuts.
/// Matches the design system's pill / capsule aesthetic.
struct ShortcutPill: View {
    @Binding var shortcut: String
    @Binding var isRecording: Bool
    var onChanged: (() -> Void)?

    @State private var isHovered = false

    var body: some View {
        Button {
            isRecording = true
        } label: {
            HStack(spacing: 6) {
                if isRecording {
                    Circle()
                        .fill(.red)
                        .frame(width: 6, height: 6)
                        .transition(.scale.combined(with: .opacity))
                    Text("Press keys...")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(.secondary)
                } else {
                    // Render each key symbol as a separate mini-key cap
                    let keys = parseShortcutDisplay(shortcut)
                    ForEach(Array(keys.enumerated()), id: \.offset) { _, key in
                        Text(key)
                            .font(.system(size: 11, weight: .semibold, design: .rounded))
                            .foregroundStyle(.primary.opacity(0.85))
                            .padding(.horizontal, 6)
                            .padding(.vertical, 3)
                            .background(
                                RoundedRectangle(cornerRadius: 5, style: .continuous)
                                    .fill(.white.opacity(0.06))
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 5, style: .continuous)
                                            .strokeBorder(.white.opacity(0.1), lineWidth: 0.5)
                                    )
                            )
                    }
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(
                Capsule(style: .continuous)
                    .fill(.ultraThinMaterial)
                    .overlay(
                        Capsule(style: .continuous)
                            .strokeBorder(
                                isRecording ? .red.opacity(0.4) : .white.opacity(isHovered ? 0.15 : 0.08),
                                lineWidth: 0.5
                            )
                    )
            )
            .animation(.easeInOut(duration: 0.2), value: isRecording)
            .animation(.easeInOut(duration: 0.15), value: isHovered)
        }
        .buttonStyle(.plain)
        .onHover { hovering in
            isHovered = hovering
        }
        .background(
            KeyEventHandler(isActive: $isRecording, onKeyPress: { key in
                shortcut = key
                isRecording = false
                onChanged?()
            })
        )
    }

    /// Split a shortcut string into displayable key parts
    private func parseShortcutDisplay(_ shortcut: String) -> [String] {
        var parts: [String] = []
        var remaining = shortcut

        // Extract modifier symbols
        for modifier in ["⌃", "⌥", "⇧", "⌘"] {
            if remaining.contains(modifier) {
                parts.append(modifier)
                remaining = remaining.replacingOccurrences(of: modifier, with: "")
            }
        }

        // Whatever remains is the key
        if !remaining.isEmpty {
            parts.append(remaining)
        }

        return parts
    }
}

// MARK: - Key Event Handler

/// Captures key events (simplified implementation)
struct KeyEventHandler: NSViewRepresentable {
    @Binding var isActive: Bool
    let onKeyPress: (String) -> Void

    func makeNSView(context: Context) -> NSView {
        let view = KeyCaptureView()
        view.onKeyPress = onKeyPress
        view.isActiveBinding = $isActive
        return view
    }

    func updateNSView(_ nsView: NSView, context: Context) {
        if let view = nsView as? KeyCaptureView {
            view.isActiveBinding = $isActive
            if isActive {
                nsView.window?.makeFirstResponder(nsView)
            }
        }
    }
}

class KeyCaptureView: NSView {
    var onKeyPress: ((String) -> Void)?
    var isActiveBinding: Binding<Bool>?

    override var acceptsFirstResponder: Bool { true }

    override func keyDown(with event: NSEvent) {
        guard isActiveBinding?.wrappedValue == true else {
            super.keyDown(with: event)
            return
        }

        var parts: [String] = []

        // Add modifier symbols
        if event.modifierFlags.contains(.control) { parts.append("⌃") }
        if event.modifierFlags.contains(.option) { parts.append("⌥") }
        if event.modifierFlags.contains(.shift) { parts.append("⇧") }
        if event.modifierFlags.contains(.command) { parts.append("⌘") }

        // Add the key character
        if let chars = event.charactersIgnoringModifiers?.uppercased() {
            // Map special keys
            let keyString: String
            switch event.keyCode {
            case 49: keyString = "Space"
            case 36: keyString = "Return"
            case 53: keyString = "Escape"
            case 51: keyString = "Delete"
            case 48: keyString = "Tab"
            default: keyString = chars
            }
            parts.append(keyString)
        }

        let shortcut = parts.joined()
        onKeyPress?(shortcut)
    }
}
