import AppKit
import ApplicationServices
import Combine
import CoreGraphics
import Foundation
import QuartzCore
import SwiftUI

enum MeetingProvider: String, Equatable {
    case googleMeet
    case zoom

    var displayName: String {
        switch self {
        case .googleMeet: return "Google Meet"
        case .zoom: return "Zoom"
        }
    }

    var symbolName: String {
        switch self {
        case .googleMeet: return "video.bubble.left.fill"
        case .zoom: return "video.fill"
        }
    }
}

enum MeetingPresenceConfidence: Equatable {
    case possible
    case active
}

struct MeetingReminder: Identifiable, Equatable {
    let provider: MeetingProvider
    let title: String?
    let meetingURL: String?
    let confidence: MeetingPresenceConfidence
    let detectedAt: Date

    var id: String { suppressionKey }

    static func == (lhs: MeetingReminder, rhs: MeetingReminder) -> Bool {
        lhs.provider == rhs.provider &&
            lhs.title == rhs.title &&
            lhs.meetingURL == rhs.meetingURL &&
            lhs.confidence == rhs.confidence
    }

    var suppressionKey: String {
        if let meetingURL {
            return "\(provider.rawValue):\(meetingURL.lowercased())"
        }

        let normalizedTitle = (title ?? provider.displayName)
            .lowercased()
            .filter { $0.isLetter || $0.isNumber || $0.isWhitespace }
            .split(separator: " ")
            .prefix(8)
            .joined(separator: "-")
        return "\(provider.rawValue):\(normalizedTitle)"
    }

    var headline: String {
        switch confidence {
        case .active:
            return "You're in \(provider.displayName)"
        case .possible:
            return "\(provider.displayName) may be active"
        }
    }

    var detail: String {
        if provider == .googleMeet, let googleMeetCode {
            return "meet.google.com/\(googleMeetCode)"
        }

        if let title, !title.isEmpty {
            return title
        }
        return "Start a session recording before the conversation gets going."
    }

    var contextLabel: String {
        if provider == .googleMeet, googleMeetCode != nil {
            return "Copyable link"
        }

        if let meetingURL,
           let components = URLComponents(string: meetingURL),
           let host = components.host {
            let compactPath = components.path
                .split(separator: "/")
                .first
                .map(String.init)
            if let compactPath, !compactPath.isEmpty {
                return "\(host)/\(compactPath)"
            }
            return host
        }

        switch provider {
        case .googleMeet:
            return confidence == .active ? "Browser call" : "Possible browser call"
        case .zoom:
            return confidence == .active ? "Desktop app" : "Possible desktop call"
        }
    }

    var googleMeetCode: String? {
        guard provider == .googleMeet,
              let meetingURL,
              let components = URLComponents(string: meetingURL) else {
            return nil
        }

        return components.path
            .split(separator: "/")
            .first
            .map(String.init)
    }
}

private struct MeetingSignal {
    let provider: MeetingProvider
    let title: String?
    let meetingURL: String?
    let confidence: MeetingPresenceConfidence
}

@MainActor
final class MeetingPresenceMonitor: ObservableObject {
    static let shared = MeetingPresenceMonitor()

    static let enabledKey = "meetingRecordingReminderEnabled"
    static let preciseBrowserDetectionKey = "meetingRecordingReminderPreciseBrowserDetection"
    private static let preciseBrowserDetectionMigrationKey = "meetingRecordingReminderPreciseBrowserDetectionDefaulted"

    @Published private(set) var currentPrompt: MeetingReminder?

    private var timer: Timer?
    private var activationObserver: NSObjectProtocol?
    private var hiddenPromptKeys: Set<String> = []
    private var recordingSuppressedUntilByKey: [String: Date] = [:]
    private var lastSignalKey: String?
    private var lastCandidateActivationDate: Date?
    private var isRefreshing = false
    private var isRunning = false

    private init() {
        UserDefaults.standard.register(defaults: [
            Self.enabledKey: true,
            Self.preciseBrowserDetectionKey: true
        ])

        if !UserDefaults.standard.bool(forKey: Self.preciseBrowserDetectionMigrationKey) {
            UserDefaults.standard.set(true, forKey: Self.preciseBrowserDetectionKey)
            UserDefaults.standard.set(true, forKey: Self.preciseBrowserDetectionMigrationKey)
        }
    }

    func start() {
        guard !isRunning else { return }
        isRunning = true

        activationObserver = NotificationCenter.default.addObserver(
            forName: NSWorkspace.didActivateApplicationNotification,
            object: NSWorkspace.shared,
            queue: .main
        ) { _ in
            Task { @MainActor in
                if MeetingDetector.frontmostCandidateKind() != nil {
                    MeetingPresenceMonitor.shared.lastCandidateActivationDate = Date()
                }
                MeetingPresenceMonitor.shared.refresh()
            }
        }

        if MeetingDetector.frontmostCandidateKind() != nil {
            lastCandidateActivationDate = Date()
        }
        refresh()
    }

    func stop() {
        isRunning = false
        timer?.invalidate()
        timer = nil

        if let activationObserver {
            NotificationCenter.default.removeObserver(activationObserver)
            self.activationObserver = nil
        }

        currentPrompt = nil
    }

    func refresh() {
        guard !isRefreshing else { return }
        guard UserDefaults.standard.bool(forKey: Self.enabledKey) else {
            currentPrompt = nil
            scheduleNextRefresh()
            return
        }

        isRefreshing = true
        let signal = MeetingDetector.detectFrontmostMeeting()
        apply(signal)
        isRefreshing = false
        scheduleNextRefresh()
    }

    func dismissCurrentPrompt() {
        guard let currentPrompt else { return }
        hiddenPromptKeys.insert(currentPrompt.suppressionKey)
        self.currentPrompt = nil
    }

    func markStartedFromPrompt() {
        guard let currentPrompt else { return }
        recordingSuppressedUntilByKey[currentPrompt.suppressionKey] = Date().addingTimeInterval(12 * 60 * 60)
        self.currentPrompt = nil
    }

    private func apply(_ signal: MeetingSignal?) {
        guard let signal else {
            if currentPrompt != nil {
                currentPrompt = nil
            }
            hiddenPromptKeys.removeAll()
            lastSignalKey = nil
            return
        }

        let reminder = MeetingReminder(
            provider: signal.provider,
            title: signal.title,
            meetingURL: signal.meetingURL,
            confidence: signal.confidence,
            detectedAt: Date()
        )

        if lastSignalKey != reminder.suppressionKey {
            hiddenPromptKeys.removeAll()
            lastSignalKey = reminder.suppressionKey
        }

        if hiddenPromptKeys.contains(reminder.suppressionKey) {
            currentPrompt = nil
            return
        }

        if let dismissedUntil = recordingSuppressedUntilByKey[reminder.suppressionKey],
           dismissedUntil > Date() {
            currentPrompt = nil
            return
        }

        if recordingSuppressedUntilByKey[reminder.suppressionKey] != nil {
            recordingSuppressedUntilByKey[reminder.suppressionKey] = nil
        }

        if currentPrompt != reminder {
            currentPrompt = reminder
            MeetingReminderCoordinator.shared.presentFloatingPromptIfNeeded()
        }
    }

    private func scheduleNextRefresh() {
        guard isRunning else { return }

        timer?.invalidate()
        let interval = nextRefreshInterval()
        timer = Timer.scheduledTimer(withTimeInterval: interval, repeats: false) { _ in
            Task { @MainActor in
                MeetingPresenceMonitor.shared.refresh()
            }
        }
    }

    private func nextRefreshInterval() -> TimeInterval {
        if currentPrompt != nil {
            return 5
        }

        switch MeetingDetector.frontmostCandidateKind() {
        case .zoom:
            if let lastCandidateActivationDate,
               Date().timeIntervalSince(lastCandidateActivationDate) < 15 {
                return 0.8
            }
            return 1.25
        case .browser:
            return 2.5
        case nil:
            return 20
        }
    }
}

private enum MeetingCandidateKind {
    case zoom
    case browser
}

private enum MeetingDetector {
    private static let chromiumBrowserBundleIds: Set<String> = [
        "com.google.Chrome",
        "com.google.Chrome.canary",
        "com.microsoft.edgemac",
        "com.brave.Browser",
        "com.thebrowser.Browser",
        "com.vivaldi.Vivaldi",
        "org.chromium.Chromium",
        "net.imput.helium"
    ]

    private static let safariBundleIds: Set<String> = [
        "com.apple.Safari",
        "com.apple.SafariTechnologyPreview"
    ]

    private static let browserBundleIdsWithMeetingFallback: Set<String> = [
        "app.zen-browser.zen",
        "org.mozilla.firefox"
    ]

    private static let browserNameFragmentsWithMeetingFallback: Set<String> = [
        "helium",
        "zen"
    ]

    private static let browserNameFragments: Set<String> = [
        "arc",
        "brave",
        "chrome",
        "chromium",
        "dia",
        "edge",
        "firefox",
        "helium",
        "opera",
        "orion",
        "safari",
        "vivaldi",
        "zen"
    ]

    static func detectFrontmostMeeting() -> MeetingSignal? {
        guard let app = NSWorkspace.shared.frontmostApplication else { return nil }
        let bundleId = app.bundleIdentifier ?? ""
        let appName = app.localizedName ?? ""
        let titles = visibleWindowTitles(for: app.processIdentifier)

        if isZoom(bundleId: bundleId, appName: appName) {
            return detectZoom(titles: titles)
        }

        if isSupportedBrowser(bundleId: bundleId, appName: appName) {
            return detectGoogleMeet(bundleId: bundleId, appName: appName, titles: titles)
        }

        return nil
    }

    static func isFrontmostMeetingCandidateApp() -> Bool {
        frontmostCandidateKind() != nil
    }

    static func frontmostCandidateKind() -> MeetingCandidateKind? {
        guard let app = NSWorkspace.shared.frontmostApplication else { return nil }
        let bundleId = app.bundleIdentifier ?? ""
        let appName = app.localizedName ?? ""
        if isZoom(bundleId: bundleId, appName: appName) {
            return .zoom
        }
        if isSupportedBrowser(bundleId: bundleId, appName: appName) {
            return .browser
        }
        return nil
    }

    private static func detectZoom(titles: [String]) -> MeetingSignal? {
        let meetingTitle = titles.first { title in
            isZoomMeetingTitle(title)
        }
        if let meetingTitle {
            return MeetingSignal(provider: .zoom, title: cleanedTitle(meetingTitle), meetingURL: nil, confidence: .active)
        }

        guard titles.isEmpty else { return nil }
        return MeetingSignal(provider: .zoom, title: nil, meetingURL: nil, confidence: .possible)
    }

    private static func detectGoogleMeet(bundleId: String, appName: String, titles: [String]) -> MeetingSignal? {
        let preciseBrowserDetectionEnabled = UserDefaults.standard.bool(forKey: MeetingPresenceMonitor.preciseBrowserDetectionKey)

        if let title = titles.first(where: { title in isActiveGoogleMeetTitle(title) }) {
            let meetingURL = googleMeetURL(fromTitle: title)
            return MeetingSignal(
                provider: .googleMeet,
                title: cleanedTitle(title),
                meetingURL: meetingURL,
                confidence: .active
            )
        }

        if preciseBrowserDetectionEnabled && supportsBrowserTabScripting(bundleId: bundleId) {
            if let tab = activeBrowserTab(bundleId: bundleId),
               isActiveGoogleMeetTitle(tab.title) {
                let meetingURL = normalizedGoogleMeetURL(tab.url) ?? googleMeetURL(fromTitle: tab.title)
                return MeetingSignal(
                    provider: .googleMeet,
                    title: cleanedTitle(tab.title),
                    meetingURL: meetingURL,
                    confidence: .active
                )
            }
        }

        return nil
    }

    private static func isZoom(bundleId: String, appName: String) -> Bool {
        let bundle = bundleId.lowercased()
        let name = appName.lowercased()
        return bundle.contains("zoom") || name.contains("zoom")
    }

    private static func isSupportedBrowser(bundleId: String, appName: String) -> Bool {
        supportsBrowserTabScripting(bundleId: bundleId) ||
            hasMeetingFallback(bundleId: bundleId, appName: appName) ||
            browserNameFragments.contains { appName.lowercased().contains($0) }
    }

    private static func supportsBrowserTabScripting(bundleId: String) -> Bool {
        chromiumBrowserBundleIds.contains(bundleId) || safariBundleIds.contains(bundleId)
    }

    private static func hasMeetingFallback(bundleId: String, appName: String) -> Bool {
        let appName = appName.lowercased()
        return browserBundleIdsWithMeetingFallback.contains(bundleId) ||
            browserNameFragmentsWithMeetingFallback.contains { appName.contains($0) }
    }

    private static func isZoomMeetingTitle(_ title: String) -> Bool {
        let lower = title.lowercased()
        return lower.contains("zoom meeting") ||
            lower.contains("webinar") ||
            lower.contains("share screen") ||
            lower.contains("participants") ||
            lower.contains("breakout rooms") ||
            (lower.contains("meeting") && lower.contains("zoom"))
    }

    private static func isGoogleMeetURL(_ url: String?) -> Bool {
        normalizedGoogleMeetURL(url) != nil
    }

    private static func normalizedGoogleMeetURL(_ url: String?) -> String? {
        guard let url,
              let components = URLComponents(string: url),
              components.host?.lowercased() == "meet.google.com" else {
            return nil
        }

        guard let code = components.path.split(separator: "/").first.map(String.init),
              isGoogleMeetCode(code) else {
            return nil
        }

        return "https://meet.google.com/\(code.lowercased())"
    }

    private static func isActiveGoogleMeetTitle(_ title: String?) -> Bool {
        guard let title,
              googleMeetCode(from: title) != nil else {
            return false
        }

        let lower = title.lowercased()
        return lower.hasPrefix("meet - ") ||
            lower.hasSuffix(" - meet")
    }

    private static func googleMeetURL(fromTitle title: String?) -> String? {
        guard let code = googleMeetCode(from: title) else { return nil }
        return "https://meet.google.com/\(code)"
    }

    private static func googleMeetCode(from title: String?) -> String? {
        guard let title else { return nil }
        let pattern = #"\b[a-z]{3}-[a-z]{4}-[a-z]{3}\b"#
        guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) else {
            return nil
        }

        let range = NSRange(title.startIndex..<title.endIndex, in: title)
        guard let match = regex.firstMatch(in: title, options: [], range: range),
              let codeRange = Range(match.range, in: title) else {
            return nil
        }

        return String(title[codeRange]).lowercased()
    }

    private static func isGoogleMeetCode(_ value: String) -> Bool {
        googleMeetCode(from: value) == value.lowercased()
    }

    private static func visibleWindowTitles(for pid: pid_t) -> [String] {
        guard let rawWindows = CGWindowListCopyWindowInfo(
            [.optionOnScreenOnly, .excludeDesktopElements],
            kCGNullWindowID
        ) as? [[String: Any]] else {
            return []
        }

        let cgTitles: [String] = rawWindows.compactMap { window in
            guard let ownerPID = window[kCGWindowOwnerPID as String] as? pid_t,
                  ownerPID == pid,
                  let layer = window[kCGWindowLayer as String] as? Int,
                  layer == 0,
                  let title = window[kCGWindowName as String] as? String else {
                return nil
            }
            let trimmed = title.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty ? nil : trimmed
        }

        return deduplicated(cgTitles + accessibilityWindowTitles(for: pid))
    }

    private static func accessibilityWindowTitles(for pid: pid_t) -> [String] {
        guard AXIsProcessTrusted() else { return [] }

        let appElement = AXUIElementCreateApplication(pid)
        var windowsValue: CFTypeRef?
        let windowsResult = AXUIElementCopyAttributeValue(
            appElement,
            kAXWindowsAttribute as CFString,
            &windowsValue
        )
        guard windowsResult == .success,
              let windows = windowsValue as? [AXUIElement] else {
            return []
        }

        return windows.compactMap { window in
            var titleValue: CFTypeRef?
            let titleResult = AXUIElementCopyAttributeValue(
                window,
                kAXTitleAttribute as CFString,
                &titleValue
            )
            guard titleResult == .success,
                  let title = titleValue as? String else {
                return nil
            }
            let trimmed = title.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty ? nil : trimmed
        }
    }

    private static func deduplicated(_ values: [String]) -> [String] {
        var seen = Set<String>()
        return values.filter { value in
            let key = value.lowercased()
            guard !seen.contains(key) else { return false }
            seen.insert(key)
            return true
        }
    }

    private static func activeBrowserTab(bundleId: String) -> (url: String?, title: String?)? {
        if safariBundleIds.contains(bundleId) {
            return runBrowserScript(bundleId: bundleId, scriptBody: """
            if (count of documents) is 0 then return ""
            return (URL of front document) & linefeed & (name of front document)
            """)
        }

        return runBrowserScript(bundleId: bundleId, scriptBody: """
        if (count of windows) is 0 then return ""
        set activeTab to active tab of front window
        return (URL of activeTab) & linefeed & (title of activeTab)
        """)
    }

    private static func runBrowserScript(bundleId: String, scriptBody: String) -> (url: String?, title: String?)? {
        let scriptSource = """
        tell application id "\(bundleId)"
        \(scriptBody)
        end tell
        """

        guard let script = NSAppleScript(source: scriptSource) else { return nil }
        var error: NSDictionary?
        let result = script.executeAndReturnError(&error)
        guard error == nil,
              let output = result.stringValue,
              !output.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return nil
        }

        let parts = output.components(separatedBy: .newlines)
        let url = parts.first?.trimmingCharacters(in: .whitespacesAndNewlines)
        let title = parts.dropFirst().joined(separator: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return (
            url?.isEmpty == false ? url : nil,
            title.isEmpty ? nil : title
        )
    }

    private static func cleanedTitle(_ title: String?) -> String? {
        guard let title else { return nil }
        let cleaned = title
            .replacingOccurrences(of: " - Google Meet", with: "")
            .replacingOccurrences(of: " - Meet", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return cleaned.isEmpty ? nil : cleaned
    }
}

@MainActor
final class MeetingReminderCoordinator {
    static let shared = MeetingReminderCoordinator()

    private let windowController = MeetingReminderWindowController()
    private var cancellables: Set<AnyCancellable> = []
    private weak var appState: AppState?
    private weak var sessionManager: SessionManager?
    private weak var monitor: MeetingPresenceMonitor?
    private var floatingDismissedUntilByKey: [String: Date] = [:]
    private var lastFloatingPromptKey: String?
    private let googleMeetFloatingDismissalInterval: TimeInterval = 20 * 60
    private let zoomFloatingDismissalInterval: TimeInterval = 2 * 60

    private init() {}

    func start(
        appState: AppState,
        sessionManager: SessionManager,
        monitor: MeetingPresenceMonitor
    ) {
        guard cancellables.isEmpty else { return }

        self.appState = appState
        self.sessionManager = sessionManager
        self.monitor = monitor

        monitor.start()

        monitor.$currentPrompt
            .sink { [weak self] _ in self?.render() }
            .store(in: &cancellables)

        appState.$isRecording
            .sink { [weak self] _ in self?.render() }
            .store(in: &cancellables)

        appState.$isQuickRecordingTransitioning
            .sink { [weak self] _ in self?.render() }
            .store(in: &cancellables)

        appState.$isQuickTranscriptionInFlight
            .sink { [weak self] _ in self?.render() }
            .store(in: &cancellables)

        appState.$isServerConnected
            .sink { [weak self] _ in self?.render() }
            .store(in: &cancellables)

        sessionManager.$isSessionRecording
            .sink { [weak self] _ in self?.render() }
            .store(in: &cancellables)

        render()
    }

    func presentFloatingPromptIfNeeded() {
        render()
    }

    private func render() {
        guard let appState, let sessionManager, let monitor else {
            windowController.hide()
            return
        }

        syncFloatingDismissalState(promptKey: monitor.currentPrompt?.suppressionKey)

        guard let prompt = monitor.currentPrompt, canStartSession(appState: appState, sessionManager: sessionManager) else {
            windowController.hide()
            return
        }

        guard !isFloatingPromptDismissed(prompt) else {
            windowController.hide()
            return
        }

        windowController.show(
            prompt: prompt,
            onStart: { [weak appState, weak sessionManager, weak monitor, weak controller = windowController] in
                guard let appState, let sessionManager, let monitor else { return }
                controller?.hide()
                Task {
                    let started = await sessionManager.startSessionRecording(micDeviceId: appState.selectedDeviceId)
                    await MainActor.run {
                        if started {
                            monitor.markStartedFromPrompt()
                        }
                    }
                }
            },
            onDismiss: { [weak self, weak controller = windowController] in
                self?.dismissFloatingPrompt(prompt)
                controller?.hide()
            }
        )
    }

    private func canStartSession(appState: AppState, sessionManager: SessionManager) -> Bool {
        appState.isServerConnected &&
            !appState.isRecording &&
            !appState.isQuickRecordingTransitioning &&
            !appState.isQuickTranscriptionInFlight &&
            !sessionManager.isSessionRecording
    }

    private func isFloatingPromptDismissed(_ prompt: MeetingReminder) -> Bool {
        pruneExpiredFloatingDismissals()
        guard let dismissedUntil = floatingDismissedUntilByKey[prompt.suppressionKey] else {
            return false
        }

        if dismissedUntil > Date() {
            return true
        }

        floatingDismissedUntilByKey[prompt.suppressionKey] = nil
        return false
    }

    private func dismissFloatingPrompt(_ prompt: MeetingReminder) {
        let dismissalInterval = prompt.provider == .googleMeet
            ? googleMeetFloatingDismissalInterval
            : zoomFloatingDismissalInterval
        floatingDismissedUntilByKey[prompt.suppressionKey] = Date().addingTimeInterval(dismissalInterval)
    }

    private func syncFloatingDismissalState(promptKey: String?) {
        pruneExpiredFloatingDismissals()

        guard let promptKey else {
            let now = Date()
            floatingDismissedUntilByKey = floatingDismissedUntilByKey.filter { key, dismissedUntil in
                dismissedUntil > now && key.hasPrefix("\(MeetingProvider.googleMeet.rawValue):")
            }
            lastFloatingPromptKey = nil
            return
        }

        if lastFloatingPromptKey != promptKey {
            lastFloatingPromptKey = promptKey
        }
    }

    private func pruneExpiredFloatingDismissals(now: Date = Date()) {
        floatingDismissedUntilByKey = floatingDismissedUntilByKey.filter { _, dismissedUntil in
            dismissedUntil > now
        }
    }
}

@MainActor
private final class MeetingReminderWindowController {
    private var window: MeetingReminderPanel?
    private var hostingController: NSHostingController<MeetingReminderFloatingView>?
    private var lastPromptID: String?
    private let panelSize = NSSize(width: 560, height: 104)

    func show(
        prompt: MeetingReminder,
        onStart: @escaping () -> Void,
        onDismiss: @escaping () -> Void
    ) {
        let rootView = MeetingReminderFloatingView(
            prompt: prompt,
            onStart: onStart,
            onDismiss: onDismiss
        )

        if let window, let hostingController {
            let shouldAnimate = lastPromptID != prompt.id || !window.isVisible
            lastPromptID = prompt.id
            hostingController.rootView = rootView
            window.orderFront(nil)
            position(window, animated: shouldAnimate)
            return
        }

        let panel = MeetingReminderPanel(
            contentRect: NSRect(origin: .zero, size: panelSize),
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )
        panel.isReleasedWhenClosed = false
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = false
        panel.level = .popUpMenu
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .transient]
        panel.hidesOnDeactivate = false
        panel.alphaValue = 0

        let hostingController = NSHostingController(rootView: rootView)
        hostingController.view.frame = panel.contentView?.bounds ?? NSRect(origin: .zero, size: panelSize)
        hostingController.view.wantsLayer = true
        hostingController.view.layer?.backgroundColor = NSColor.clear.cgColor
        panel.contentViewController = hostingController
        panel.contentView?.wantsLayer = true
        panel.contentView?.layer?.backgroundColor = NSColor.clear.cgColor

        self.window = panel
        self.hostingController = hostingController
        lastPromptID = prompt.id

        panel.orderFront(nil)
        position(panel, animated: true)
    }

    func hide(animated: Bool = false) {
        guard let window, window.isVisible else { return }

        guard animated else {
            window.orderOut(nil)
            window.alphaValue = 0
            return
        }

        let finalFrame = targetFrame(for: window)
        let endFrame = finalFrame.offsetBy(dx: 0, dy: 16)

        NSAnimationContext.runAnimationGroup { context in
            context.duration = 0.16
            context.timingFunction = CAMediaTimingFunction(controlPoints: 0.36, 0, 0.66, -0.56)
            window.animator().setFrame(endFrame, display: true)
            window.animator().alphaValue = 0
        } completionHandler: {
            window.orderOut(nil)
            window.setFrame(finalFrame, display: false)
        }
    }

    private func position(_ window: NSWindow, animated: Bool) {
        let finalFrame = targetFrame(for: window)
        guard animated else {
            window.setFrame(finalFrame, display: true)
            window.alphaValue = 1
            return
        }

        let startFrame = finalFrame.offsetBy(dx: 0, dy: 20)
        window.setFrame(startFrame, display: true)
        window.alphaValue = 0

        NSAnimationContext.runAnimationGroup { context in
            context.duration = 0.22
            context.timingFunction = CAMediaTimingFunction(controlPoints: 0.23, 1, 0.32, 1)
            window.animator().setFrame(finalFrame, display: true)
            window.animator().alphaValue = 1
        }
    }

    private func targetFrame(for window: NSWindow) -> NSRect {
        guard let screen = NSScreen.main else {
            return window.frame
        }
        let frame = screen.frame
        let size = window.frame.size
        let x = frame.midX - (size.width / 2)
        let y = frame.maxY - size.height
        return NSRect(origin: NSPoint(x: x, y: y), size: size)
    }
}

private final class MeetingReminderPanel: NSPanel {
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { false }
}

private struct MeetingReminderFloatingView: View {
    let prompt: MeetingReminder
    let onStart: () -> Void
    let onDismiss: () -> Void

    @State private var didCopyLink = false
    @State private var isExitNeutralized = false
    @State private var isClosing = false
    @State private var isPresented = false

    var body: some View {
        ZStack(alignment: .top) {
            notchSurface
                .scaleEffect(
                    x: surfaceScaleX,
                    y: surfaceScaleY,
                    anchor: .top
                )
                .opacity(surfaceOpacity)
                .offset(y: isClosing ? -7 : 0)

            content
                .opacity(contentOpacity)
                .scaleEffect(isClosing ? 0.96 : 1, anchor: .top)
                .offset(y: contentOffsetY)
        }
        .frame(width: 560, height: 104, alignment: .top)
        .compositingGroup()
        .onAppear {
            resetPresentationState()
        }
        .onChange(of: prompt.id) {
            resetPresentationState()
        }
    }

    private var content: some View {
        HStack(spacing: 11) {
            Image(systemName: prompt.provider.symbolName)
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(tint)
                .frame(width: 30, height: 30)
                .background(tint.opacity(0.18), in: .circle)

            VStack(alignment: .leading, spacing: 3) {
                Text(prompt.headline)
                    .font(.system(size: 12.5, weight: .semibold))
                    .foregroundStyle(.primary)
                    .lineLimit(1)

                HStack(spacing: 7) {
                    Text(prompt.detail)
                        .lineLimit(1)
                        .truncationMode(.middle)

                    Text(prompt.contextLabel)
                        .font(.system(size: 9, weight: .semibold))
                        .foregroundStyle(tint)
                        .lineLimit(1)
                        .padding(.horizontal, 7)
                        .frame(height: 18)
                        .background(tint.opacity(0.14), in: Capsule())
                }
                .font(.system(size: 10, weight: .medium))
                .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            HStack(spacing: 7) {
                MeetingReminderPillButton(
                    title: "Record session",
                    systemImage: "waveform.badge.plus",
                    tint: tint,
                    isProminent: true,
                    action: {
                        performExit(completion: onStart)
                    }
                )

                if let meetingURL = prompt.meetingURL {
                    MeetingReminderIconButton(
                        title: didCopyLink ? "Copied" : "Copy link",
                        systemImage: didCopyLink ? "checkmark" : "link",
                        tint: tint
                    ) {
                        copyMeetingURL(meetingURL)
                    }
                }

                Button {
                    performExit(completion: onDismiss)
                } label: {
                    Image(systemName: "xmark")
                        .font(.system(size: 8.5, weight: .bold))
                        .foregroundStyle(.red.opacity(0.92))
                        .frame(width: 24, height: 24)
                        .background(.red.opacity(0.13), in: .circle)
                }
                .buttonStyle(.plain)
                .help("Dismiss")
            }
        }
        .padding(.leading, 22)
        .padding(.trailing, 16)
        .frame(width: 560, height: 76, alignment: .center)
    }

    private var notchSurface: some View {
        MeetingReminderGlassNotchShape(
            cornerRadius: 23,
            isClosing: isClosing,
            isExitNeutralized: isExitNeutralized
        )
            .frame(width: 560, height: 76)
    }

    private var contentOpacity: Double {
        guard isPresented else { return 0 }
        return isClosing || isExitNeutralized ? 0 : 1
    }

    private var contentOffsetY: CGFloat {
        if isClosing { return -12 }
        return isPresented ? 0 : -8
    }

    private var surfaceOpacity: Double {
        guard isPresented else { return 0.88 }
        return isClosing ? 0.64 : 1
    }

    private var surfaceScaleX: CGFloat {
        guard isPresented else { return 0.34 }
        return isClosing ? 0.22 : 1
    }

    private var surfaceScaleY: CGFloat {
        guard isPresented else { return 0.36 }
        return isClosing ? 0.18 : 1
    }

    private var tint: Color {
        prompt.provider == .googleMeet ? .green : .blue
    }

    private func performExit(completion: @escaping () -> Void) {
        guard !isClosing else { return }

        var transaction = Transaction()
        transaction.disablesAnimations = true
        withTransaction(transaction) {
            didCopyLink = false
            isExitNeutralized = true
        }

        DispatchQueue.main.async {
            withAnimation(.spring(response: 0.26, dampingFraction: 0.78, blendDuration: 0.02)) {
                isClosing = true
            }
        }

        DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
            completion()
        }
    }

    private func resetPresentationState() {
        didCopyLink = false
        isExitNeutralized = false
        isClosing = false
        isPresented = false

        DispatchQueue.main.async {
            withAnimation(.spring(response: 0.34, dampingFraction: 0.82, blendDuration: 0.04)) {
                isPresented = true
            }
        }
    }

    private func copyMeetingURL(_ meetingURL: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(meetingURL, forType: .string)
        didCopyLink = true

        DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) {
            didCopyLink = false
        }
    }
}

private struct MeetingReminderNotchShape: Shape {
    let cornerRadius: CGFloat

    func path(in rect: CGRect) -> Path {
        let radius = min(cornerRadius, rect.width / 2, rect.height)
        var path = Path()
        path.move(to: CGPoint(x: rect.minX, y: rect.minY))
        path.addLine(to: CGPoint(x: rect.maxX, y: rect.minY))
        path.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY - radius))
        path.addQuadCurve(
            to: CGPoint(x: rect.maxX - radius, y: rect.maxY),
            control: CGPoint(x: rect.maxX, y: rect.maxY)
        )
        path.addLine(to: CGPoint(x: rect.minX + radius, y: rect.maxY))
        path.addQuadCurve(
            to: CGPoint(x: rect.minX, y: rect.maxY - radius),
            control: CGPoint(x: rect.minX, y: rect.maxY)
        )
        path.closeSubpath()
        return path
    }
}

private struct MeetingReminderGlassNotchShape: View {
    let cornerRadius: CGFloat
    let isClosing: Bool
    let isExitNeutralized: Bool

    var body: some View {
        let shape = MeetingReminderNotchShape(cornerRadius: cornerRadius)
        let neutralTintOpacity = isClosing || isExitNeutralized ? 0 : 0.045
        let fallbackFillOpacity = isClosing || isExitNeutralized ? 0 : 0.055
        let strokeOpacity = isClosing || isExitNeutralized ? 0 : 0.16
        let strokeWidth: CGFloat = isClosing || isExitNeutralized ? 0 : 0.6

        if #available(macOS 26.0, *) {
            shape
                .fill(.clear)
                .glassEffect(.regular.tint(.white.opacity(neutralTintOpacity)), in: shape)
                .overlay(
                    shape
                        .stroke(.white.opacity(strokeOpacity), lineWidth: strokeWidth)
                )
        } else {
            shape
                .fill(.ultraThinMaterial)
                .overlay(
                    shape
                        .fill(.white.opacity(fallbackFillOpacity))
                )
                .overlay(
                    shape
                        .stroke(.white.opacity(strokeOpacity), lineWidth: strokeWidth)
                )
        }
    }
}

private struct MeetingReminderIconButton: View {
    let title: String
    let systemImage: String
    let tint: Color
    let action: () -> Void

    @State private var isHovered = false

    var body: some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(isHovered ? tint : .white.opacity(0.58))
                .frame(width: 28, height: 28)
                .background(.white.opacity(isHovered ? 0.12 : 0.075), in: .circle)
        }
        .buttonStyle(.plain)
        .help(title)
        .onHover { hovered in
            withAnimation(.easeOut(duration: 0.12)) {
                isHovered = hovered
            }
        }
    }
}

private struct MeetingReminderPillButton: View {
    let title: String
    let systemImage: String
    let tint: Color
    let isProminent: Bool
    let action: () -> Void

    @State private var isHovered = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 5) {
                Image(systemName: systemImage)
                    .font(.system(size: 10, weight: .semibold))

                Text(title)
                    .font(.system(size: 10, weight: .semibold))
                    .lineLimit(1)
            }
            .foregroundStyle(foregroundStyle)
            .padding(.horizontal, isProminent ? 11 : 10)
            .frame(height: 28)
            .background(backgroundStyle, in: Capsule())
            .overlay(
                Capsule()
                    .strokeBorder(borderColor, lineWidth: 0.5)
            )
        }
        .buttonStyle(.plain)
        .onHover { hovered in
            withAnimation(.easeOut(duration: 0.12)) {
                isHovered = hovered
            }
        }
    }

    private var foregroundStyle: Color {
        isProminent ? .white : (isHovered ? .primary : .secondary)
    }

    private var backgroundStyle: some ShapeStyle {
        if isProminent {
            return AnyShapeStyle(tint.opacity(isHovered ? 0.86 : 0.72))
        }
        return AnyShapeStyle(.white.opacity(isHovered ? 0.09 : 0.055))
    }

    private var borderColor: Color {
        if isProminent {
            return tint.opacity(isHovered ? 0.5 : 0.32)
        }
        return .white.opacity(isHovered ? 0.14 : 0.08)
    }
}
