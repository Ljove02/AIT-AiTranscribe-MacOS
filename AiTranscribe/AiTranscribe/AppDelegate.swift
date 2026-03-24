import SwiftUI
import AppKit

class AppDelegate: NSObject, NSApplicationDelegate {
    private var onboardingWindowController: OnboardingWindowController?

    func applicationDidFinishLaunching(_ notification: Notification) {
        print("AppDelegate: applicationDidFinishLaunching")

        // Fix macOS 26 Tahoe bug: NSHostingView triggers re-entrant _postWindowNeeds*
        // calls during the display cycle, crashing ANY window with SwiftUI content.
        DisplayCycleFix.install()

        // Use singletons — available immediately, no need to wait for SwiftUI
        let appState = AppState.shared
        let backendManager = BackendManager.shared
        let sessionManager = SessionManager.shared

        // Bind state synchronization
        appState.bindToBackendManager(backendManager)

        // Wire session manager to app state for mutual exclusion
        sessionManager.appState = appState

        // Load saved sessions
        sessionManager.loadSessions()

        // Show onboarding if needed — immediately on launch, no click required
        let hasCompletedOnboarding = UserDefaults.standard.bool(forKey: "hasCompletedOnboarding")
        if !hasCompletedOnboarding {
            showOnboarding(appState: appState, backendManager: backendManager)
        }

        // Monitor onboarding completion
        UserDefaults.standard.addObserver(
            self,
            forKeyPath: "hasCompletedOnboarding",
            options: [.new],
            context: nil
        )

        // Background: wait for server ready, then fetch status
        Task { @MainActor in
            // Health check will detect readiness (no timeout wall)
            while !backendManager.isServerReady {
                try? await Task.sleep(for: .milliseconds(500))
            }
            await appState.checkServerStatus()
            print("AppDelegate: Server ready, initial status fetched")
        }
    }

    /// Gracefully stop session recording before app quits
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        let sessionManager = SessionManager.shared
        guard sessionManager.isSessionRecording else {
            return .terminateNow
        }

        // Stop session recording before quitting
        Task { @MainActor in
            await sessionManager.stopSessionRecording()
            try? await Task.sleep(for: .milliseconds(500))
            NSApplication.shared.reply(toApplicationShouldTerminate: true)
        }

        return .terminateLater
    }

    /// Show onboarding window
    func showOnboarding(appState: AppState, backendManager: BackendManager) {
        print("AppDelegate.showOnboarding() - showing onboarding window")
        let controller = OnboardingWindowController()
        controller.show(appState: appState, backendManager: backendManager)
        self.onboardingWindowController = controller
    }

    override func observeValue(forKeyPath keyPath: String?, of object: Any?, change: [NSKeyValueChangeKey : Any]?, context: UnsafeMutableRawPointer?) {
        if keyPath == "hasCompletedOnboarding" {
            let hasCompleted = UserDefaults.standard.bool(forKey: "hasCompletedOnboarding")
            if hasCompleted {
                closeOnboarding()
            }
        }
    }

    private func closeOnboarding() {
        onboardingWindowController?.close()
        onboardingWindowController = nil
    }

    deinit {
        UserDefaults.standard.removeObserver(self, forKeyPath: "hasCompletedOnboarding")
    }
}
