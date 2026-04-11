/*
 HotkeyManager.swift
 ===================

 Manages global keyboard shortcuts that work even when the app is not focused.

 CARBON HOTKEYS:
 ---------------
 macOS has several ways to register global hotkeys:
 1. Carbon RegisterEventHotKey - older but reliable API
 2. CGEvent tap - lower level, requires accessibility permissions
 3. NSEvent.addGlobalMonitorForEvents - doesn't work for all key combos

 We use Carbon's RegisterEventHotKey because:
 - Works reliably for modifier+key combinations
 - Doesn't require accessibility permissions
 - Simpler than CGEvent taps
 
 HOW IT WORKS:
 1. Register a hotkey with a unique ID
 2. Install an event handler that gets called when hotkey is pressed
 3. Match the ID in the handler to trigger the right action
 */

import Carbon.HIToolbox
import AppKit

/// Manages global keyboard shortcuts
class HotkeyManager {
    /// Singleton instance
    static let shared = HotkeyManager()

    /// Reference to app state for triggering actions
    private var appState: AppState?

    /// Reference to session manager for session hotkeys
    private var sessionManager: SessionManager?

    /// Registered hotkey references (needed to unregister)
    private var hotkeyRefs: [EventHotKeyRef] = []

    /// Hotkey IDs
    private enum HotkeyID: UInt32 {
        case toggleRecording = 1
        case cancelRecording = 2
        case toggleSession = 3
        case stopSession = 4
    }

    /// Event handler reference
    private var eventHandler: EventHandlerRef?

    private init() {}

    /// Setup the hotkey manager with app state and session manager
    func setup(appState: AppState, sessionManager: SessionManager? = nil) {
        self.appState = appState
        self.sessionManager = sessionManager
        registerHotkeys()
    }

    /// Register all global hotkeys
    private func registerHotkeys() {
        // Install the event handler first
        installEventHandler()

        // Read shortcuts from UserDefaults
        let toggleShortcut = UserDefaults.standard.string(forKey: "toggleRecordingShortcut") ?? "⌃P"
        let cancelShortcut = UserDefaults.standard.string(forKey: "cancelRecordingShortcut") ?? "⌃K"

        // Register Control+P for toggle recording
        if let (keyCode, modifiers) = parseShortcut(toggleShortcut) {
            registerHotkey(id: .toggleRecording, keyCode: keyCode, modifiers: modifiers)
        } else {
            // Default: Control+P
            registerHotkey(id: .toggleRecording, keyCode: UInt32(kVK_ANSI_P), modifiers: UInt32(controlKey))
        }

        // Register Control+K for cancel recording
        if let (keyCode, modifiers) = parseShortcut(cancelShortcut) {
            registerHotkey(id: .cancelRecording, keyCode: keyCode, modifiers: modifiers)
        } else {
            // Default: Control+K
            registerHotkey(id: .cancelRecording, keyCode: UInt32(kVK_ANSI_K), modifiers: UInt32(controlKey))
        }

        // Session recording shortcuts
        let sessionToggleShortcut = UserDefaults.standard.string(forKey: "toggleSessionShortcut") ?? "⌃⇧R"
        let sessionStopShortcut = UserDefaults.standard.string(forKey: "stopSessionShortcut") ?? "⌃⇧S"

        if let (keyCode, modifiers) = parseShortcut(sessionToggleShortcut) {
            registerHotkey(id: .toggleSession, keyCode: keyCode, modifiers: modifiers)
        } else {
            registerHotkey(id: .toggleSession, keyCode: UInt32(kVK_ANSI_R), modifiers: UInt32(controlKey) | UInt32(shiftKey))
        }

        if let (keyCode, modifiers) = parseShortcut(sessionStopShortcut) {
            registerHotkey(id: .stopSession, keyCode: keyCode, modifiers: modifiers)
        } else {
            registerHotkey(id: .stopSession, keyCode: UInt32(kVK_ANSI_S), modifiers: UInt32(controlKey) | UInt32(shiftKey))
        }
    }

    /// Parse a shortcut string like "⌥Space" into keyCode and modifiers
    private func parseShortcut(_ shortcut: String) -> (keyCode: UInt32, modifiers: UInt32)? {
        var modifiers: UInt32 = 0
        var keyPart = shortcut

        // Parse modifiers
        if keyPart.contains("⌃") {
            modifiers |= UInt32(controlKey)
            keyPart = keyPart.replacingOccurrences(of: "⌃", with: "")
        }
        if keyPart.contains("⌥") {
            modifiers |= UInt32(optionKey)
            keyPart = keyPart.replacingOccurrences(of: "⌥", with: "")
        }
        if keyPart.contains("⇧") {
            modifiers |= UInt32(shiftKey)
            keyPart = keyPart.replacingOccurrences(of: "⇧", with: "")
        }
        if keyPart.contains("⌘") {
            modifiers |= UInt32(cmdKey)
            keyPart = keyPart.replacingOccurrences(of: "⌘", with: "")
        }

        // Parse key
        let keyCode: UInt32
        switch keyPart.uppercased() {
        case "SPACE": keyCode = UInt32(kVK_Space)
        case "ESCAPE": keyCode = UInt32(kVK_Escape)
        case "RETURN": keyCode = UInt32(kVK_Return)
        case "TAB": keyCode = UInt32(kVK_Tab)
        case "DELETE": keyCode = UInt32(kVK_Delete)
        case "A": keyCode = UInt32(kVK_ANSI_A)
        case "B": keyCode = UInt32(kVK_ANSI_B)
        case "C": keyCode = UInt32(kVK_ANSI_C)
        case "D": keyCode = UInt32(kVK_ANSI_D)
        case "E": keyCode = UInt32(kVK_ANSI_E)
        case "F": keyCode = UInt32(kVK_ANSI_F)
        case "G": keyCode = UInt32(kVK_ANSI_G)
        case "H": keyCode = UInt32(kVK_ANSI_H)
        case "I": keyCode = UInt32(kVK_ANSI_I)
        case "J": keyCode = UInt32(kVK_ANSI_J)
        case "K": keyCode = UInt32(kVK_ANSI_K)
        case "L": keyCode = UInt32(kVK_ANSI_L)
        case "M": keyCode = UInt32(kVK_ANSI_M)
        case "N": keyCode = UInt32(kVK_ANSI_N)
        case "O": keyCode = UInt32(kVK_ANSI_O)
        case "P": keyCode = UInt32(kVK_ANSI_P)
        case "Q": keyCode = UInt32(kVK_ANSI_Q)
        case "R": keyCode = UInt32(kVK_ANSI_R)
        case "S": keyCode = UInt32(kVK_ANSI_S)
        case "T": keyCode = UInt32(kVK_ANSI_T)
        case "U": keyCode = UInt32(kVK_ANSI_U)
        case "V": keyCode = UInt32(kVK_ANSI_V)
        case "W": keyCode = UInt32(kVK_ANSI_W)
        case "X": keyCode = UInt32(kVK_ANSI_X)
        case "Y": keyCode = UInt32(kVK_ANSI_Y)
        case "Z": keyCode = UInt32(kVK_ANSI_Z)
        default: return nil
        }

        return (keyCode, modifiers)
    }

    /// Install the Carbon event handler
    private func installEventHandler() {
        // Event types we want to handle
        var eventType = EventTypeSpec(
            eventClass: OSType(kEventClassKeyboard),
            eventKind: UInt32(kEventHotKeyPressed)
        )

        // Install the handler
        let status = InstallEventHandler(
            GetApplicationEventTarget(),
            { (_, event, userData) -> OSStatus in
                // Get the HotkeyManager instance from userData
                guard let userData = userData else { return OSStatus(eventNotHandledErr) }
                let manager = Unmanaged<HotkeyManager>.fromOpaque(userData).takeUnretainedValue()
                return manager.handleHotkeyEvent(event)
            },
            1,
            &eventType,
            Unmanaged.passUnretained(self).toOpaque(),
            &eventHandler
        )

        if status != noErr {
            print("Failed to install hotkey event handler: \(status)")
        }
    }

    /// Register a single hotkey
    private func registerHotkey(id: HotkeyID, keyCode: UInt32, modifiers: UInt32) {
        var hotkeyRef: EventHotKeyRef?
        let hotkeyID = EventHotKeyID(signature: OSType(0x4149_5472), id: id.rawValue)  // "AITr"

        let status = RegisterEventHotKey(
            keyCode,
            modifiers,
            hotkeyID,
            GetApplicationEventTarget(),
            0,
            &hotkeyRef
        )

        if status == noErr, let ref = hotkeyRef {
            hotkeyRefs.append(ref)
            print("Registered hotkey \(id) with keyCode \(keyCode), modifiers \(modifiers)")
        } else {
            print("Failed to register hotkey \(id): \(status)")
        }
    }

    /// Handle a hotkey event
    private func handleHotkeyEvent(_ event: EventRef?) -> OSStatus {
        guard let event = event else { return OSStatus(eventNotHandledErr) }

        // Get the hotkey ID from the event
        var hotkeyID = EventHotKeyID()
        let status = GetEventParameter(
            event,
            EventParamName(kEventParamDirectObject),
            EventParamType(typeEventHotKeyID),
            nil,
            MemoryLayout<EventHotKeyID>.size,
            nil,
            &hotkeyID
        )

        guard status == noErr else { return status }

        // Handle based on hotkey ID
        switch hotkeyID.id {
        case HotkeyID.toggleRecording.rawValue:
            handleToggleRecording()
            return noErr

        case HotkeyID.cancelRecording.rawValue:
            handleCancelRecording()
            return noErr

        case HotkeyID.toggleSession.rawValue:
            handleToggleSession()
            return noErr

        case HotkeyID.stopSession.rawValue:
            handleStopSession()
            return noErr

        default:
            return OSStatus(eventNotHandledErr)
        }
    }

    /// Handle toggle recording hotkey
    private func handleToggleRecording() {
        guard let appState = appState else { return }

        Task { @MainActor in
            if appState.isRecording {
                await appState.stopRecording()
            } else {
                // startRecording() handles auto-loading the model if needed
                await appState.startRecording()
            }
        }
    }

    /// Handle cancel recording hotkey
    private func handleCancelRecording() {
        guard let appState = appState else { return }

        Task { @MainActor in
            if appState.isRecording {
                await appState.cancelRecording()
            }
        }
    }

    /// Handle start session recording hotkey
    private func handleToggleSession() {
        guard let appState = appState, let sessionManager = sessionManager else { return }

        Task { @MainActor in
            if sessionManager.isSessionRecording {
                await sessionManager.stopSessionRecording()
            } else {
                await sessionManager.startSessionRecording(
                    micDeviceId: appState.selectedDeviceId
                )
            }
        }
    }

    /// Handle stop session recording hotkey
    private func handleStopSession() {
        guard let sessionManager = sessionManager else { return }

        Task { @MainActor in
            if sessionManager.isSessionRecording {
                await sessionManager.stopSessionRecording()
            }
        }
    }

    /// Unregister all hotkeys
    func unregisterAll() {
        for ref in hotkeyRefs {
            UnregisterEventHotKey(ref)
        }
        hotkeyRefs.removeAll()

        if let handler = eventHandler {
            RemoveEventHandler(handler)
            eventHandler = nil
        }
    }

    /// Re-register hotkeys (call after settings change)
    func refreshHotkeys() {
        unregisterAll()
        registerHotkeys()
    }
}
