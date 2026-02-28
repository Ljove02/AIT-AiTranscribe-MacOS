/*
 MenuBarView.swift
 =================

 This is the dropdown menu that appears when you click the menu bar icon.

 SWIFTUI BASICS
 --------------

 SwiftUI uses a DECLARATIVE syntax. Instead of:
   "Create a button, set its title, add it to the view, position it"

 You write:
   "There is a button with this title"

 SwiftUI figures out the rest.

 KEY CONCEPTS:
 - View: Anything visual (text, button, image, etc.)
 - body: Required property that returns what to display
 - @EnvironmentObject: Access shared data from parent views
 - VStack/HStack: Vertical/Horizontal stacks (like flexbox)

 MODIFIERS:
 SwiftUI uses "modifiers" to customize views:

   Text("Hello")
       .font(.headline)        // Change font
       .foregroundColor(.red)  // Change color
       .padding()              // Add spacing

 Each modifier returns a new view, so you can chain them.
 */

import SwiftUI
import CoreAudio

struct MenuBarView: View {
    /*
     @EnvironmentObject gets the AppState we passed down from AiTranscribeApp.
     It's like a global variable, but SwiftUI manages it properly.

     If AppState changes, this view automatically updates.
     */
    @EnvironmentObject var appState: AppState
    
    /// Backend manager to control the server
    @EnvironmentObject var backendManager: BackendManager

    /// SwiftUI's environment action to open settings (macOS 14+)
    @Environment(\.openSettings) private var openSettings
    
    /// SwiftUI's environment action to open windows (macOS 13+)
    @Environment(\.openWindow) private var openWindow

    /*
     'body' is required by the View protocol.
     It describes what this view looks like.
     */
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // ─────────────────────────────────────────────────────────────────
            // STATUS SECTION
            // ─────────────────────────────────────────────────────────────────

            statusSection

            Divider()
                .padding(.vertical, 4)

            // ─────────────────────────────────────────────────────────────────
            // RECORDING SECTION (simplified - indicator handles the UI)
            // ─────────────────────────────────────────────────────────────────

            if appState.isModelLoaded {
                recordingControls
            }

            Divider()
                .padding(.vertical, 4)

            // ─────────────────────────────────────────────────────────────────
            // MODEL CONTROLS
            // ─────────────────────────────────────────────────────────────────

            modelControlsSection

            Divider()
                .padding(.vertical, 4)

            // ─────────────────────────────────────────────────────────────────
            // MICROPHONE SELECTION
            // ─────────────────────────────────────────────────────────────────

            microphoneSection

            Divider()
                .padding(.vertical, 4)

            // ─────────────────────────────────────────────────────────────────
            // FOOTER
            // ─────────────────────────────────────────────────────────────────

            footerSection
        }
        .padding(.vertical, 8)
        .frame(width: 250)
    }


    // =========================================================================
    // VIEW COMPONENTS
    // =========================================================================
    /*
     Breaking the UI into smaller pieces makes it:
     - Easier to read
     - Easier to modify
     - Reusable

     'some View' means "returns a View, but I won't specify which type"
     This is called an "opaque return type".
     */

    /// Shows connection status and current state
    private var statusSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Circle()
                    .frame(width: 12, height: 12)
                    .glassEffect(.regular.tint(statusColor))

                Text(appState.statusMessage)
                    .font(.headline)
            }

            if !appState.isServerConnected {
                Text("Start the backend server first")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 4)
    }

    private var statusColor: Color {
        if !appState.isServerConnected {
            return .red
        } else if appState.isRecording {
            return .orange
        } else if appState.statusMessage.hasPrefix("Transcribing") {
            return .blue
        } else if appState.isModelLoaded {
            return .green
        } else {
            return .yellow
        }
    }

    /// Simple recording controls - Start or Stop button only
    private var recordingControls: some View {
        VStack(alignment: .leading, spacing: 4) {
            if appState.isRecording {
                // Stop button when recording
                Button {
                    Task {
                        await appState.stopRecording()
                    }
                } label: {
                    Label("Stop Recording", systemImage: "stop.fill")
                }

                Button {
                    Task {
                        await appState.cancelRecording()
                    }
                } label: {
                    Label("Cancel", systemImage: "xmark")
                }
            } else {
                // Start button when not recording
                Button {
                    Task {
                        await appState.startRecording()
                    }
                } label: {
                    Label("Start Recording", systemImage: "mic.fill")
                }
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 4)
    }

    /// Model load/unload controls
    private var modelControlsSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            if appState.isModelLoaded {
                Button {
                    Task {
                        await appState.unloadModel()
                    }
                } label: {
                    Label("Unload Model", systemImage: "square.and.arrow.up")
                }
                .disabled(appState.isRecording)
            } else {
                Button {
                    Task {
                        await appState.loadModel()
                    }
                } label: {
                    Label("Load Model", systemImage: "square.and.arrow.down")
                }
                .disabled(!appState.isServerConnected)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 4)
    }

    /// Microphone selection - single dropdown
    private var microphoneSection: some View {
        // Picker with .menu style - click once to see dropdown options
        Picker(selection: $appState.selectedDeviceId) {
            ForEach(appState.audioDevices) { device in
                Text(device.displayName)
                    .tag(device.id as AudioDeviceID?)
            }
        } label: {
            Label("Microphone", systemImage: "mic")
        }
        .pickerStyle(.menu)
        .padding(.horizontal, 12)
        .padding(.vertical, 4)
    }

    /// App info and quit button
    private var footerSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            // Debug-only button: Show Introduction
            // Only visible in debug builds - users see onboarding on first launch automatically
            #if DEBUG
            Button {
                showIntroductionScreen()
            } label: {
                Label("Show Introduction", systemImage: "play.circle")
            }
            #endif

            // Factory Reset - available to all users for troubleshooting
            Button {
                showFactoryResetConfirmation()
            } label: {
                Label("Factory Reset", systemImage: "exclamationmark.triangle")
            }
            .foregroundColor(.red)

            Divider()
                .padding(.vertical, 4)

            // Settings button - opens and focuses settings window
            Button {
                openAndFocusSettings()
            } label: {
                Label("Settings...", systemImage: "gear")
            }
            .keyboardShortcut(",", modifiers: .command)

            Divider()
                .padding(.vertical, 4)

            // Version
            Text("AiTranscribe v0.1.0")
                .font(.caption)
                .foregroundColor(.secondary)

            // Quit button
            Button {
                NSApplication.shared.terminate(nil)
            } label: {
                Label("Quit", systemImage: "power")
            }
            .keyboardShortcut("q", modifiers: .command)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 4)
    }

    /// Opens settings window and brings it to focus if already open
    private func openAndFocusSettings() {
        // Use SwiftUI's openSettings environment action (macOS 14+)
        openSettings()

        // Bring the settings window to front
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            NSApp.activate(ignoringOtherApps: true)
            // Find and focus the settings window
            for window in NSApp.windows {
                if window.title.contains("Settings") || window.identifier?.rawValue.contains("settings") == true {
                    window.makeKeyAndOrderFront(nil)
                    break
                }
            }
        }
    }
    
    /// Shows the introduction/onboarding screen (for testing)
    private func showIntroductionScreen() {
        // Open the onboarding window using environment action
        openWindow(id: "onboarding")
        
        // Bring it to front
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
            NSApp.activate(ignoringOtherApps: true)
            if let window = NSApp.windows.first(where: { $0.identifier?.rawValue == "onboarding" }) {
                window.makeKeyAndOrderFront(nil)
            }
        }
    }
    
    /// Shows factory reset confirmation dialog
    private func showFactoryResetConfirmation() {
        let alert = NSAlert()
        alert.messageText = "Factory Reset AIT?"
        alert.informativeText = """
        This will permanently delete:

        • All downloaded models
        • All transcription history
        • All app settings and preferences
        • Onboarding completion status
        • NeMo support installation

        This action cannot be undone.
        """
        alert.alertStyle = .critical
        alert.icon = NSImage(systemSymbolName: "exclamationmark.triangle.fill", accessibilityDescription: "Warning")

        alert.addButton(withTitle: "Cancel")
        alert.addButton(withTitle: "Factory Reset")

        let response = alert.runModal()

        if response == .alertSecondButtonReturn {
            performFactoryReset()
        }
    }
    
    /// Performs the actual factory reset
    private func performFactoryReset() {
        Task {
            // 1. Stop the backend first (important: must happen before deleting NeMo venv)
            backendManager.stop()

            // Give the backend time to fully stop
            try? await Task.sleep(nanoseconds: 2_000_000_000) // 2 seconds

            // 2. Clear transcription history
            appState.clearHistory()

            // 3. Unload model if loaded (may fail since backend is stopped, that's OK)
            if appState.isModelLoaded {
                await appState.unloadModel()
            }

            // 4. Clear all UserDefaults (do this before model deletion since backend is stopped)
            if let bundleID = Bundle.main.bundleIdentifier {
                UserDefaults.standard.removePersistentDomain(forName: bundleID)
                UserDefaults.standard.synchronize()
            }

            // 5. Clear hotkeys
            HotkeyManager.shared.unregisterAll()

            // 6. Delete Application Support folder contents (includes NeMo venv)
            let fileManager = FileManager.default
            if let appSupport = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first {
                let appFolder = appSupport.appendingPathComponent("AiTranscribe", isDirectory: true)
                try? fileManager.removeItem(at: appFolder)
            }

            // 7. Delete HuggingFace model cache (downloaded models)
            let homeDir = fileManager.homeDirectoryForCurrentUser
            let hfCacheDir = homeDir.appendingPathComponent(".cache/huggingface/hub")

            // Delete specific model folders we use
            let modelPrefixes = [
                "models--nvidia--parakeet",
                "models--nvidia--nemotron",
                "models--Systran--faster-whisper",
                "models--deepdml--faster-whisper"
            ]

            if let contents = try? fileManager.contentsOfDirectory(at: hfCacheDir, includingPropertiesForKeys: nil) {
                for item in contents {
                    let name = item.lastPathComponent
                    if modelPrefixes.contains(where: { name.hasPrefix($0) }) {
                        try? fileManager.removeItem(at: item)
                    }
                }
            }

            // 8. Show completion message and restart
            DispatchQueue.main.async {
                let successAlert = NSAlert()
                successAlert.messageText = "Factory Reset Complete"
                successAlert.informativeText = "AIT will now quit. Launch it again to go through onboarding."
                successAlert.alertStyle = .informational
                successAlert.addButton(withTitle: "Quit")
                successAlert.runModal()

                // Quit the app
                NSApplication.shared.terminate(nil)
            }
        }
    }
}


// =============================================================================
// PREVIEW
// =============================================================================
/*
 #Preview lets you see the view in Xcode's canvas without running the app.
 Super useful for designing UI quickly.
 */

#Preview {
    MenuBarView()
        .environmentObject(AppState())
        .environmentObject(BackendManager())
}
