import SwiftUI
import Carbon.HIToolbox

// MARK: - Shortcuts Settings View

struct ShortcutsSettingsView: View {
    let hasAnimated: Bool
    let onAnimated: () -> Void

    @EnvironmentObject private var appState: AppState

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

    /// The Polish preset (if any) whose pill is currently capturing keys
    @State private var recordingPolishPresetId: String?

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

                        Text("\(shortcutCount) shortcuts")
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

                // ── Polish Shortcuts ─────────────────────────────
                if !polishShortcutPresets.isEmpty {
                    SettingsDivider()
                        .staggerIn(index: 4, appeared: appeared)

                    Group {
                        SettingsSectionHeader(title: "Polishes")

                        ForEach(polishShortcutPresets) { preset in
                            polishShortcutRow(preset)
                        }
                    }
                    .staggerIn(index: 5, appeared: appeared)
                }

                SettingsDivider()
                    .staggerIn(index: polishShortcutPresets.isEmpty ? 4 : 6, appeared: appeared)

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
                .staggerIn(index: polishShortcutPresets.isEmpty ? 5 : 7, appeared: appeared)

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

    // MARK: - Polish shortcuts

    /// Only Polishes that already carry a shortcut are listed — no empty
    /// "assign one" rows. Disabled presets stay listed (dimmed) so every
    /// stored key combination is accounted for.
    private var polishShortcutPresets: [PolishPreset] {
        appState.polishPresets.filter { normalizedPolishShortcut($0.shortcut ?? "") != nil }
    }

    private var shortcutCount: Int {
        4 + polishShortcutPresets.count
    }

    private func polishShortcutRow(_ preset: PolishPreset) -> some View {
        let losesConflict = shortcutLosesConflict(preset)
        return HStack(alignment: .center, spacing: 16) {
            Image(systemName: preset.symbolName)
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(.secondary)
                .frame(width: 28, height: 28)
                .background(.white.opacity(0.06), in: .rect(cornerRadius: 8, style: .continuous))

            VStack(alignment: .leading, spacing: 3) {
                Text(preset.displayName)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(.primary)

                if !preset.isEnabled {
                    Text("Disabled — this hotkey is inactive.")
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                } else if losesConflict {
                    Text("Key combination already in use — this hotkey won't trigger.")
                        .font(.system(size: 11))
                        .foregroundStyle(.orange)
                } else {
                    Text("Global hotkey for this Polish.")
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                }
            }

            Spacer()

            if preset.isEnabled && losesConflict {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(.orange)
                    .help("Another shortcut registered this key combination first.")
            }

            ShortcutPill(
                shortcut: polishShortcutBinding(for: preset),
                isRecording: Binding(
                    get: { recordingPolishPresetId == preset.id },
                    set: { recordingPolishPresetId = $0 ? preset.id : nil }
                )
            )
            .help("Shortcut for this Polish")

            Button {
                polishShortcutBinding(for: preset).wrappedValue = ""
            } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 9, weight: .bold))
                    .foregroundStyle(.secondary)
                    .frame(width: 24, height: 24)
                    .background(Color.primary.opacity(0.045), in: Circle())
            }
            .buttonStyle(.plain)
            .help("Clear shortcut")
        }
        .padding(.vertical, 10)
        .padding(.horizontal, 14)
        .background(.thinMaterial, in: .rect(cornerRadius: 10, style: .continuous))
        .padding(.vertical, 2)
        .opacity(preset.isEnabled ? 1 : 0.55)
    }

    /// No `onChanged` on the pill: `appState.upsertPolishPreset` already calls
    /// `HotkeyManager.shared.refreshHotkeys()`, so wiring one here would
    /// unregister and re-register every hotkey twice per change.
    private func polishShortcutBinding(for preset: PolishPreset) -> Binding<String> {
        Binding(
            get: {
                appState.polishPresets.first(where: { $0.id == preset.id })?.shortcut ?? ""
            },
            set: { newValue in
                guard var updated = appState.polishPresets.first(where: { $0.id == preset.id }) else { return }
                updated.shortcut = normalizedPolishShortcut(newValue)
                appState.upsertPolishPreset(updated)
            }
        )
    }

    /// Display-only duplicate flag; registration behavior is unchanged.
    /// HotkeyManager registers the four recording/session hotkeys first, then
    /// enabled Polishes in list order — the first registration of a key
    /// combination wins and later ones fail silently, so a Polish always loses
    /// to a system shortcut and a later Polish loses to an earlier enabled one.
    private func shortcutLosesConflict(_ preset: PolishPreset) -> Bool {
        guard let shortcut = normalizedPolishShortcut(preset.shortcut ?? "") else { return false }

        let systemShortcuts = [toggleShortcut, cancelShortcut, sessionToggleShortcut, sessionStopShortcut]
            .compactMap { normalizedPolishShortcut($0) }
        if systemShortcuts.contains(shortcut) { return true }

        for other in appState.polishPresets {
            if other.id == preset.id { return false }
            guard other.isEnabled,
                  let otherShortcut = normalizedPolishShortcut(other.shortcut ?? "") else { continue }
            if otherShortcut == shortcut { return true }
        }
        return false
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
                    if keys.isEmpty {
                        Label("Shortcut", systemImage: "keyboard")
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundStyle(.secondary)
                    } else {
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
