/*
 AiTranscribeApp.swift
 =====================

 This is the ENTRY POINT of the app - where everything starts.

 WHAT IS @main?
 --------------
 @main tells Swift "start here". Every app needs exactly one @main.
 When you double-click the app icon, macOS runs the code marked with @main.

 WHAT IS SwiftUI App PROTOCOL?
 -----------------------------
 A "protocol" in Swift is like a contract - it says "if you want to be
 an App, you must provide a 'body' property".

 struct vs class:
 - struct = value type (copied when passed around) - preferred in SwiftUI
 - class = reference type (shared when passed around)

 MENU BAR APPS:
 --------------
 Normal apps have a window. Menu bar apps live in the menu bar (top right).
 We use MenuBarExtra (macOS 13+) to create the menu bar icon and dropdown.
 */

import SwiftUI
import AVFoundation

@main
struct AiTranscribeApp: App {
    @StateObject private var appState = AppState()
    /// Use the shared singleton BackendManager
    @ObservedObject private var backendManager = BackendManager.shared
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    @State private var hotkeysRegistered = false
    @State private var isInitialized = false

    init() {
        // Force singleton creation early to ensure backend starts
        print("AiTranscribeApp.init() - BackendManager.shared accessed")
        _ = BackendManager.shared
    }

    var body: some Scene {
        MenuBarExtra("AIT", systemImage: menuBarIcon) {
            MenuBarView()
                .environmentObject(appState)
                .environmentObject(backendManager)
                .task {
                    // Initialize once when the view appears
                    if !isInitialized {
                        isInitialized = true
                        await initializeApp()
                    }

                    if !hotkeysRegistered {
                        HotkeyManager.shared.setup(appState: appState)
                        hotkeysRegistered = true
                    }
                    requestMicrophonePermissionAtLaunch()
                }
                .onChange(of: appState.isRecording) { _, _ in }
        }

        Settings {
            SettingsView()
                .environmentObject(appState)
                .environmentObject(backendManager)
        }
    }

    /// Initialize the app - bindings, onboarding, model loading
    @MainActor
    private func initializeApp() async {
        print("AiTranscribeApp: Initializing...")

        // Bind state synchronization
        appState.bindToBackendManager(backendManager)

        // Store references in AppDelegate
        appDelegate.configure(appState: appState, backendManager: backendManager)

        // Show onboarding if needed
        let hasCompletedOnboarding = UserDefaults.standard.bool(forKey: "hasCompletedOnboarding")
        if !hasCompletedOnboarding {
            appDelegate.showOnboarding(appState: appState, backendManager: backendManager)
        }

        // Wait for server ready and load model
        await backendManager.waitForServerReady()
        if backendManager.isServerReady {
            await appState.checkServerStatus()
            if appState.isServerConnected {
                await appState.loadPreferredModel()
            }
        }

        print("AiTranscribeApp: Initialization complete")
    }

    @MainActor private var menuBarIcon: String {
        if appState.isRecording {
            return "waveform.circle.fill"
        } else if appState.isModelLoaded {
            return "mic.circle.fill"
        } else {
            return "mic.circle"
        }
    }

    private func requestMicrophonePermissionAtLaunch() {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .audio) { granted in
                if granted {
                    print("Microphone permission granted")
                } else {
                    print("Microphone permission denied")
                    DispatchQueue.main.async {
                        self.appState.statusMessage = "Microphone access denied - enable in System Settings"
                    }
                }
            }
        case .denied, .restricted:
            print("Microphone permission was previously denied")
            DispatchQueue.main.async {
                self.appState.statusMessage = "Microphone access denied - enable in System Settings"
            }
        case .authorized:
            print("Microphone permission already granted")
        @unknown default:
            break
        }
    }
}
