import SwiftUI
import Carbon.HIToolbox

// MARK: - Shortcuts Settings View

struct ShortcutsSettingsView: View {
    @AppStorage("toggleRecordingShortcut") private var toggleShortcut = "⌃P"
    @AppStorage("cancelRecordingShortcut") private var cancelShortcut = "⌃K"

    @State private var isRecordingToggleShortcut = false
    @State private var isRecordingCancelShortcut = false

    var body: some View {
        Form {
            Section("Recording Shortcuts") {
                HStack {
                    Text("Toggle Recording:")
                    Spacer()
                    ShortcutButton(
                        shortcut: $toggleShortcut,
                        isRecording: $isRecordingToggleShortcut,
                        onShortcutChanged: {
                            HotkeyManager.shared.refreshHotkeys()
                        }
                    )
                }

                HStack {
                    Text("Cancel Recording:")
                    Spacer()
                    ShortcutButton(
                        shortcut: $cancelShortcut,
                        isRecording: $isRecordingCancelShortcut,
                        onShortcutChanged: {
                            HotkeyManager.shared.refreshHotkeys()
                        }
                    )
                }
            }

            Section {
                Text("Click a shortcut button and press your desired key combination.")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Text("Default: Control+P to toggle, Control+K to cancel")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .formStyle(.grouped)
        .padding()
    }
}

// MARK: - Shortcut Button

/// A button that records keyboard shortcuts
struct ShortcutButton: View {
    @Binding var shortcut: String
    @Binding var isRecording: Bool
    var onShortcutChanged: (() -> Void)?

    var body: some View {
        Button {
            isRecording = true
        } label: {
            Text(isRecording ? "Press keys..." : shortcut)
                .frame(minWidth: 100)
        }
        .buttonStyle(.bordered)
        .background(
            KeyEventHandler(isActive: $isRecording, onKeyPress: { key in
                shortcut = key
                isRecording = false
                onShortcutChanged?()
            })
        )
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
