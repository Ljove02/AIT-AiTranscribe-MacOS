/*
 KeyboardSimulator.swift
 =======================

 Simulates keyboard input at the cursor position using CGEvent.
 Used for real-time transcription output and auto-paste functionality.

 REQUIREMENTS:
 - Accessibility permissions must be granted in System Settings
 - Without permissions, keyboard simulation will fail silently

 HOW IT WORKS:
 - Uses CGEvent to create keyboard events
 - Events are sent to whichever app/field currently has focus
 - If no text field is focused, the events are ignored by the system
 */

import Foundation
import AppKit
import Carbon.HIToolbox

/// Singleton class for simulating keyboard input
class KeyboardSimulator {
    static let shared = KeyboardSimulator()

    private init() {}

    // =========================================================================
    // ACCESSIBILITY CHECK
    // =========================================================================

    /// Check if Accessibility permissions are granted
    /// Returns true if the app can simulate keyboard input
    var hasAccessibilityPermission: Bool {
        return AXIsProcessTrusted()
    }

    /// Request Accessibility permissions by showing the system prompt
    /// This opens System Settings if permissions aren't granted
    func requestAccessibilityPermission() {
        let options = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true] as CFDictionary
        AXIsProcessTrustedWithOptions(options)
    }

    // =========================================================================
    // KEYBOARD SIMULATION
    // =========================================================================

    /// Type a string of text at the current cursor position
    /// Text appears wherever the user has focus (any app, any text field)
    /// If no text field is focused, the input is silently ignored
    ///
    /// - Parameter text: The text to type
    /// - Parameter delayBetweenChars: Delay between characters (seconds), default 0.01
    func typeText(_ text: String, delayBetweenChars: TimeInterval = 0.01) {
        guard hasAccessibilityPermission else {
            print("KeyboardSimulator: No accessibility permission")
            return
        }

        guard !text.isEmpty else { return }

        // Type each character
        for char in text {
            typeCharacter(char)
            if delayBetweenChars > 0 {
                Thread.sleep(forTimeInterval: delayBetweenChars)
            }
        }
    }

    /// Type text asynchronously without blocking the main thread
    func typeTextAsync(_ text: String, delayBetweenChars: TimeInterval = 0.01) {
        DispatchQueue.global(qos: .userInteractive).async { [weak self] in
            self?.typeText(text, delayBetweenChars: delayBetweenChars)
        }
    }

    /// Simulate pressing Cmd+V (paste from clipboard)
    func paste() {
        guard hasAccessibilityPermission else {
            print("KeyboardSimulator: No accessibility permission")
            return
        }

        // Create key down event for 'V' with Command modifier
        let source = CGEventSource(stateID: .hidSystemState)

        // Key code for 'V' is 9
        let keyDown = CGEvent(keyboardEventSource: source, virtualKey: CGKeyCode(kVK_ANSI_V), keyDown: true)
        let keyUp = CGEvent(keyboardEventSource: source, virtualKey: CGKeyCode(kVK_ANSI_V), keyDown: false)

        // Add Command modifier
        keyDown?.flags = .maskCommand
        keyUp?.flags = .maskCommand

        // Post the events
        keyDown?.post(tap: .cghidEventTap)
        keyUp?.post(tap: .cghidEventTap)
    }

    /// Simulate pressing Cmd+V asynchronously
    func pasteAsync() {
        DispatchQueue.global(qos: .userInteractive).async { [weak self] in
            self?.paste()
        }
    }

    // =========================================================================
    // PRIVATE HELPERS
    // =========================================================================

    /// Type a single character using CGEvent
    private func typeCharacter(_ char: Character) {
        let source = CGEventSource(stateID: .hidSystemState)

        // For simple characters, we can use the Unicode approach
        // This handles most characters including spaces, punctuation, etc.
        let utf16Chars = Array(String(char).utf16)

        // Create key event and set the Unicode string
        guard let keyDown = CGEvent(keyboardEventSource: source, virtualKey: 0, keyDown: true),
              let keyUp = CGEvent(keyboardEventSource: source, virtualKey: 0, keyDown: false) else {
            return
        }

        // Set the Unicode string for the event
        keyDown.keyboardSetUnicodeString(stringLength: utf16Chars.count, unicodeString: utf16Chars)
        keyUp.keyboardSetUnicodeString(stringLength: utf16Chars.count, unicodeString: utf16Chars)

        // Post the events
        keyDown.post(tap: .cghidEventTap)
        keyUp.post(tap: .cghidEventTap)
    }
}


// =============================================================================
// INCREMENTAL TEXT TYPER
// =============================================================================

/// Helper class for tracking and typing only new words in real-time transcription
/// This is used to type only the new portion of text when streaming partial results
class IncrementalTextTyper {
    private var lastTypedText: String = ""

    /// Reset the typer (call when starting a new transcription)
    func reset() {
        lastTypedText = ""
    }

    /// Type only the new portion of text that hasn't been typed yet
    /// For example, if lastTypedText is "Hello" and newText is "Hello world",
    /// this will type " world" (the new portion)
    ///
    /// - Parameter newText: The full transcription so far
    func typeIncremental(_ newText: String) {
        // Find what's new compared to what we've already typed
        if newText.hasPrefix(lastTypedText) {
            // The new text extends our previous text
            let startIndex = newText.index(newText.startIndex, offsetBy: lastTypedText.count)
            let newPortion = String(newText[startIndex...])

            if !newPortion.isEmpty {
                KeyboardSimulator.shared.typeTextAsync(newPortion, delayBetweenChars: 0.005)
            }
        } else {
            // Text changed completely (unlikely in real-time streaming, but handle it)
            // In this case, we'd need to clear and retype, which is complex
            // For now, just type the new text as-is
            KeyboardSimulator.shared.typeTextAsync(newText, delayBetweenChars: 0.005)
        }

        lastTypedText = newText
    }

    /// Get how much text has been typed
    var typedLength: Int {
        return lastTypedText.count
    }
}
