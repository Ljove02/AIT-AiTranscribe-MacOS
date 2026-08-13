import SwiftUI
import AppKit
import Combine

final class OnboardingWindowController: NSObject, NSWindowDelegate {
    private var window: NSWindow?
    private var hostingController: NSHostingController<AnyView>?

    private var appState: AppState?
    private var backendManager: BackendManager?

    var isVisible: Bool = false

    func show(appState: AppState, backendManager: BackendManager, previewMode: Bool = false) {
        self.appState = appState
        self.backendManager = backendManager

        guard window == nil else {
            window?.makeKeyAndOrderFront(nil)
            isVisible = true
            return
        }

        let onboardingView = OnboardingContainerView(previewMode: previewMode)
            .environmentObject(appState)
            .environmentObject(backendManager)

        hostingController = NSHostingController(rootView: AnyView(onboardingView))

        guard let hostingController = hostingController else { return }

        hostingController.view.frame = NSRect(x: 0, y: 0, width: 900, height: 720)

        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 900, height: 720),
            styleMask: [.titled, .closable, .resizable, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )

        guard let window = window else { return }

        window.identifier = NSUserInterfaceItemIdentifier(previewMode ? "onboarding-preview" : "onboarding")
        window.title = previewMode ? "AiTranscribe Introduction Preview" : "Welcome to AIT"
        window.delegate = self
        window.titleVisibility = .hidden
        window.titlebarAppearsTransparent = true
        window.titlebarSeparatorStyle = .none
        window.toolbar = nil
        window.isMovableByWindowBackground = true
        window.center()
        window.level = .modalPanel
        window.isOpaque = false
        window.backgroundColor = .clear
        window.hasShadow = true
        window.isReleasedWhenClosed = false
        window.standardWindowButton(.closeButton)?.isHidden = true
        window.standardWindowButton(.miniaturizeButton)?.isHidden = true
        window.standardWindowButton(.zoomButton)?.isHidden = true

        window.contentView?.addSubview(hostingController.view)
        hostingController.view.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            hostingController.view.topAnchor.constraint(equalTo: window.contentView!.topAnchor),
            hostingController.view.leadingAnchor.constraint(equalTo: window.contentView!.leadingAnchor),
            hostingController.view.trailingAnchor.constraint(equalTo: window.contentView!.trailingAnchor),
            hostingController.view.bottomAnchor.constraint(equalTo: window.contentView!.bottomAnchor)
        ])

        NSApp.setActivationPolicy(.regular)
        window.orderFrontRegardless()
        window.makeKeyAndOrderFront(nil)

        NSApp.activate(ignoringOtherApps: true)

        isVisible = true
    }

    func windowWillClose(_ notification: Notification) {
        window = nil
        hostingController = nil
        isVisible = false
    }

    func close() {
        guard let window = window else { return }

        window.orderOut(nil)
        window.close()
        self.window = nil
        hostingController = nil

        isVisible = false
    }

    func checkIfCompleted() -> Bool {
        let defaults = UserDefaults.standard
        return defaults.bool(forKey: "hasCompletedOnboarding")
    }
}
