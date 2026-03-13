import SwiftUI
import AppKit

class AppDelegate: NSObject, NSApplicationDelegate {
    private var appState: AppState?
    private var backendManager: BackendManager?
    private var sessionManager: SessionManager?
    private var onboardingWindowController: OnboardingWindowController?

    func applicationDidFinishLaunching(_ notification: Notification) {
        print("AppDelegate: applicationDidFinishLaunching")
        // Backend initialization is now handled in MenuBarView.task
    }

    func configure(appState: AppState, backendManager: BackendManager) {
        print("AppDelegate.configure() - storing references")
        self.appState = appState
        self.backendManager = backendManager

        UserDefaults.standard.addObserver(
            self,
            forKeyPath: "hasCompletedOnboarding",
            options: [.new],
            context: nil
        )
    }

    /// Store session manager reference for graceful shutdown
    func configureSessionManager(_ sessionManager: SessionManager) {
        self.sessionManager = sessionManager
    }

    /// Gracefully stop session recording before app quits
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard let sessionManager, sessionManager.isSessionRecording else {
            return .terminateNow
        }

        // Stop session recording before quitting
        Task { @MainActor in
            await sessionManager.stopSessionRecording()
            // Small delay to let file finalize, then quit
            try? await Task.sleep(for: .milliseconds(500))
            NSApplication.shared.reply(toApplicationShouldTerminate: true)
        }

        return .terminateLater
    }

    /// Show onboarding window (called from SwiftUI)
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
