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
    /// Use shared singletons so AppDelegate can access them at launch
    @ObservedObject private var appState = AppState.shared
    @ObservedObject private var sessionManager = SessionManager.shared
    @ObservedObject private var backendManager = BackendManager.shared
    @ObservedObject private var updateChecker = UpdateChecker.shared
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    @State private var hotkeysRegistered = false

    init() {
        // Force singleton creation early to ensure backend starts
        print("AiTranscribeApp.init() - singletons created")
        _ = BackendManager.shared
        _ = AppState.shared
        _ = SessionManager.shared
    }

    var body: some Scene {
        MenuBarExtra {
            MenuBarView()
                .environmentObject(appState)
                .environmentObject(backendManager)
                .environmentObject(sessionManager)
                .task {
                    if !hotkeysRegistered {
                        HotkeyManager.shared.setup(appState: appState, sessionManager: sessionManager)
                        hotkeysRegistered = true
                    }
                    requestMicrophonePermissionAtLaunch()
                }
                .onChange(of: appState.isRecording) { _, _ in }
        } label: {
            menuBarLabel
        }

        .menuBarExtraStyle(.window)

        Window("AiTranscribe Settings", id: "settings") {
            SettingsView()
                .environmentObject(appState)
                .environmentObject(backendManager)
                .environmentObject(sessionManager)
                .environmentObject(updateChecker)
                .task {
                    await updateChecker.checkForUpdates()
                }
        }
        .windowStyle(.hiddenTitleBar)
        .windowResizability(.contentSize)
        .defaultSize(width: 1080, height: 740)
    }

    @MainActor @ViewBuilder private var menuBarLabel: some View {
        if sessionManager.isSessionRecording {
            Label("AIT", systemImage: "bubble.left.and.bubble.right.fill")
        } else if appState.isRecording {
            Label("AIT", systemImage: "sensor.radiowaves.left.and.right.fill")
        } else {
            Label("AIT", image: "MenuBarIcon")
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
