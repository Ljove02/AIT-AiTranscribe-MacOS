/*
 RecordingIndicator.swift
 ========================

 A small floating window that shows when recording is active.
 Features:
 - Glassy capsule with layered glass depth (top highlight, bottom shadow)
 - RECORDING: Side-wave arcs pulse outward from the core when speaking
 - TRANSCRIBING: Three dots expand/contract with slow rotation
 - All animations driven by TimelineView (GPU-synced, no Timer jank)
 - Drag-and-snap to 8 positions (corners, centers, middle sides)
 - Glassy placeholders appear during drag showing snap targets
 - Magnetic snap when close to a position
 - Always on top

 ARCHITECTURE NOTE:
 ------------------
 This view does NOT use @EnvironmentObject appState because that would cause
 it to re-render whenever ANY @Published property in AppState changes.
 Instead, RecordingIndicatorController has its own isolated volume polling
 that only this view observes
 */

import SwiftUI
import AppKit
import Combine
import QuartzCore

// =============================================================================
// RECORDING INDICATOR VIEW
// =============================================================================

struct RecordingIndicatorView: View {
    /// Observe the controller for volume updates
    @ObservedObject var controller: RecordingIndicatorController

    /// Smoothed volume level (for gradual transitions)
    @State private var smoothedVolume: CGFloat = 0

    /// Appear/disappear animation state
    @State private var appearScale: CGFloat = 1.0
    @State private var appearOpacity: Double = 1.0
    @State private var hasAppeared: Bool = false

    /// Whether to show the transcription dots animation
    @State private var showTranscriptionDots: Bool = false

    /// Start time for GPU-synced animations
    @State private var animationStartTime: Date = Date()

    /// Wave spawn timestamps — each entry is when a wave was born
    @State private var waveSpawnTimes: [Date] = []

    /// Detect light/dark mode
    @Environment(\.colorScheme) private var colorScheme

    /// Core circle size
    private let coreSize: CGFloat = 22

    /// Wave arc parameters
    private let waveLifetime: Double = 1.3     // faster expansion
    private let maxWaveOffset: CGFloat = 11    // fade well before capsule edge
    private let volumeThreshold: CGFloat = 0.10

    var body: some View {
        TimelineView(.animation(minimumInterval: nil, paused: false)) { timeline in
            let now = timeline.date

            ZStack {
                // Side-wave arcs (recording mode)
                if !showTranscriptionDots {
                    waveArcs(now: now)
                }

                // Core circle — layered glass effect
                coreCircle

                // Transcription dots
                if showTranscriptionDots {
                    transcriptionDots(now: now)
                }
            }
            .frame(width: 64, height: 36)
            .background(glassBackground)
            .overlay(glassStroke)
            .scaleEffect(appearScale)
            .opacity(appearOpacity)
        }
        .onAppear {
            animationStartTime = Date()
            if !hasAppeared {
                appearScale = 0.3
                appearOpacity = 0
                hasAppeared = true
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) {
                    withAnimation(.spring(response: 0.4, dampingFraction: 0.7)) {
                        appearScale = 1.0
                        appearOpacity = 1.0
                    }
                }
            }
        }
        .onDisappear {
            hasAppeared = false
            waveSpawnTimes.removeAll()
        }
        .onChange(of: controller.currentVolume) { _, newVolume in
            if !controller.isTranscribing {
                updateSmoothedVolume(newVolume)
                spawnWaveIfNeeded()
            }
        }
        .onChange(of: controller.isTranscribing) { _, isTranscribing in
            if isTranscribing {
                waveSpawnTimes.removeAll()
                smoothedVolume = 0
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) {
                    animationStartTime = Date()
                    showTranscriptionDots = true
                }
            } else {
                showTranscriptionDots = false
            }
        }
        .onChange(of: controller.isDisappearing) { _, isDisappearing in
            if isDisappearing {
                withAnimation(.easeInOut(duration: 0.3)) {
                    appearScale = 0.3
                    appearOpacity = 0
                }
            }
        }
    }

    // MARK: - Glass Background

    private var glassBackground: some View {
        Capsule()
            .fill(
                colorScheme == .dark
                    ? Color(white: 0.12, opacity: 0.72)
                    : Color(white: 0.92, opacity: 0.85)
            )
            .overlay(
                // Inner highlight — top edge glow for glass depth
                Capsule()
                    .fill(
                        LinearGradient(
                            colors: colorScheme == .dark
                                ? [Color.white.opacity(0.12), Color.clear]
                                : [Color.white.opacity(0.5), Color.clear],
                            startPoint: .top,
                            endPoint: .center
                        )
                    )
            )
            .overlay(
                // Subtle inner shadow at bottom
                Capsule()
                    .fill(
                        LinearGradient(
                            colors: [Color.clear, Color.black.opacity(colorScheme == .dark ? 0.2 : 0.06)],
                            startPoint: .center,
                            endPoint: .bottom
                        )
                    )
            )
    }

    private var glassStroke: some View {
        Capsule()
            .stroke(
                LinearGradient(
                    colors: colorScheme == .dark
                        ? [Color.white.opacity(0.25), Color.white.opacity(0.06)]
                        : [Color.white.opacity(0.6), Color.black.opacity(0.08)],
                    startPoint: .top,
                    endPoint: .bottom
                ),
                lineWidth: 0.5
            )
    }

    // MARK: - Core Circle

    private var coreCircle: some View {
        Circle()
            .fill(
                RadialGradient(
                    colors: colorScheme == .dark
                        ? [Color(white: 0.45, opacity: 0.9), Color(white: 0.2, opacity: 0.95)]
                        : [Color(white: 0.80, opacity: 0.95), Color(white: 0.58, opacity: 0.9)],
                    center: .topLeading,
                    startRadius: 0,
                    endRadius: coreSize
                )
            )
            .frame(width: coreSize, height: coreSize)
            .overlay(
                // Glass highlight on core
                Circle()
                    .fill(
                        LinearGradient(
                            colors: [Color.white.opacity(colorScheme == .dark ? 0.25 : 0.4), Color.clear],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .frame(width: coreSize * 0.6, height: coreSize * 0.4)
                    .offset(x: -2, y: -3)
                    .blur(radius: 1.5)
            )
            .overlay(
                Circle()
                    .stroke(
                        LinearGradient(
                            colors: colorScheme == .dark
                                ? [Color.white.opacity(0.35), Color.white.opacity(0.08)]
                                : [Color.white.opacity(0.6), Color.black.opacity(0.1)],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        ),
                        lineWidth: 0.5
                    )
            )
    }

    // MARK: - Side Wave Arcs

    @ViewBuilder
    private func waveArcs(now: Date) -> some View {
        let liveWaves = waveSpawnTimes.filter { now.timeIntervalSince($0) < waveLifetime }

        ForEach(Array(liveWaves.enumerated()), id: \.offset) { _, spawnTime in
            let age = now.timeIntervalSince(spawnTime)
            let progress = CGFloat(age / waveLifetime)

            let opacity = progress < 0.25
                ? Double(progress / 0.25) * 0.35
                : Double(max(0, 0.35 * (1 - (progress - 0.25) / 0.75)))

            // All waves use the same radius (= core circle radius), just offset outward
            let waveOffset = 5 + progress * maxWaveOffset
            let arcColor = colorScheme == .dark
                ? Color.white.opacity(opacity)
                : Color.black.opacity(opacity * 0.6)

            // Arc grows as it travels: starts at 80% of coreSize, grows to 140%
            let growFactor = 0.8 + progress * 0.6
            let arcSize = coreSize * growFactor

            // Right  )
            SemiCircleArc()
                .stroke(arcColor, style: StrokeStyle(lineWidth: 0.8, lineCap: .round))
                .frame(width: arcSize, height: arcSize)
                .offset(x: waveOffset)

            // Left  (
            SemiCircleArc()
                .stroke(arcColor, style: StrokeStyle(lineWidth: 0.8, lineCap: .round))
                .frame(width: arcSize, height: arcSize)
                .scaleEffect(x: -1)
                .offset(x: -waveOffset)
        }
    }

    // MARK: - Transcription Dots (original — TimelineView driven)

    @ViewBuilder
    private func transcriptionDots(now: Date) -> some View {
        let elapsed = now.timeIntervalSince(animationStartTime)
        let cycleDuration: Double = 1.2
        let phase = CGFloat(elapsed.truncatingRemainder(dividingBy: cycleDuration) / cycleDuration)

        let expandFactor = CoreGraphics.sin(phase * .pi)
        let maxRadius: CGFloat = 3.5
        let currentRadius = maxRadius * expandFactor

        ZStack {
            ForEach(0..<3, id: \.self) { index in
                let angle = CGFloat(Double(index) * 120 - 90) * .pi / 180
                Circle()
                    .fill(Color.white)
                    .frame(width: 3, height: 3)
                    .offset(
                        x: CoreGraphics.cos(angle) * currentRadius,
                        y: CoreGraphics.sin(angle) * currentRadius
                    )
            }
        }
        .frame(width: coreSize, height: coreSize)
        .rotationEffect(.degrees(Double(phase) * 180), anchor: .center)
    }

    // MARK: - Volume & Wave Spawning

    private func updateSmoothedVolume(_ volume: Double) {
        let noiseGate: Double = 0.008
        let gatedVolume = volume > noiseGate ? volume - noiseGate : 0
        let amplified = min(1.0, gatedVolume * 100.0)

        let newSmoothed: CGFloat
        if CGFloat(amplified) > smoothedVolume {
            newSmoothed = smoothedVolume * 0.3 + CGFloat(amplified) * 0.7
        } else {
            newSmoothed = smoothedVolume * 0.70 + CGFloat(amplified) * 0.30
        }
        smoothedVolume = newSmoothed
    }

    private func spawnWaveIfNeeded() {
        guard smoothedVolume > volumeThreshold else { return }

        let now = Date()

        // Prune dead waves
        waveSpawnTimes.removeAll { now.timeIntervalSince($0) >= waveLifetime }

        // Minimum 0.5s between waves
        if let lastSpawn = waveSpawnTimes.last, now.timeIntervalSince(lastSpawn) < 0.5 {
            return
        }

        // Max 2 concurrent waves
        if waveSpawnTimes.count < 2 {
            waveSpawnTimes.append(now)
        }
    }
}

// MARK: - Semi-Circle Arc Shape

/// Draws the right half of a circle `)` — a 180-degree arc from top to bottom.
/// The arc's size matches the frame, so frame(width: 22, height: 22) gives
/// an arc with the same curvature as a 22pt diameter circle.
struct SemiCircleArc: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        let center = CGPoint(x: rect.midX, y: rect.midY)
        let radius = rect.height / 2
        // Trimmed semicircle: 40% cut from top and bottom
        // Full semi = -90° to +90° (180°), trimmed = -54° to +54° (108°)
        path.addArc(
            center: center,
            radius: radius,
            startAngle: .degrees(-54),
            endAngle: .degrees(54),
            clockwise: false
        )
        return path
    }
}


// =============================================================================
// SNAP PLACEHOLDER VIEW
// =============================================================================

/// A glassy placeholder that appears at snap positions during drag
struct SnapPlaceholderView: View {
    let isHighlighted: Bool

    var body: some View {
        Capsule()
            .fill(.ultraThinMaterial)
            .opacity(isHighlighted ? 0.9 : 0.4)
            .overlay(
                Capsule()
                    .stroke(Color.white.opacity(isHighlighted ? 0.5 : 0.2), lineWidth: 1)
            )
            .shadow(color: .black.opacity(0.1), radius: 8, x: 0, y: 2)
            .frame(width: 64, height: 36)
            .scaleEffect(isHighlighted ? 1.05 : 1.0)
            .animation(.spring(response: 0.3, dampingFraction: 0.7), value: isHighlighted)
    }
}

// =============================================================================
// SNAP PLACEHOLDER CONTROLLER
// =============================================================================

/// Manages the 8 placeholder windows shown during drag
class SnapPlaceholderController {
    private var windows: [RecordingIndicatorController.ScreenPosition: NSWindow] = [:]
    private var highlightedPosition: RecordingIndicatorController.ScreenPosition?

    /// Show all placeholder windows
    func show() {
        guard let screen = NSScreen.main else { return }

        for position in RecordingIndicatorController.ScreenPosition.allCases {
            let origin = calculateOrigin(for: position, screen: screen)
            let window = createPlaceholderWindow(at: origin, highlighted: false)
            window.orderFront(nil)
            windows[position] = window
        }
    }

    /// Hide all placeholder windows
    func hide() {
        for window in windows.values {
            window.orderOut(nil)
        }
        windows.removeAll()
        highlightedPosition = nil
    }

    /// Update highlight based on indicator position
    func updateHighlight(indicatorFrame: CGRect) -> RecordingIndicatorController.ScreenPosition? {
        guard let screen = NSScreen.main else { return nil }

        let indicatorCenter = CGPoint(
            x: indicatorFrame.midX,
            y: indicatorFrame.midY
        )

        var closestPosition: RecordingIndicatorController.ScreenPosition?
        var closestDistance: CGFloat = .infinity
        let snapThreshold: CGFloat = 50  // Magnetic snap distance

        for position in RecordingIndicatorController.ScreenPosition.allCases {
            let origin = calculateOrigin(for: position, screen: screen)
            let center = CGPoint(
                x: origin.x + 35,  // Half of window width (70)
                y: origin.y + 21   // Half of window height (42)
            )

            let distance = hypot(indicatorCenter.x - center.x, indicatorCenter.y - center.y)

            if distance < snapThreshold && distance < closestDistance {
                closestDistance = distance
                closestPosition = position
            }
        }

        // Update highlighting
        if closestPosition != highlightedPosition {
            // Unhighlight previous
            if let prev = highlightedPosition, let window = windows[prev] {
                updateWindowHighlight(window, highlighted: false)
            }
            // Highlight new
            if let new = closestPosition, let window = windows[new] {
                updateWindowHighlight(window, highlighted: true)
            }
            highlightedPosition = closestPosition
        }

        return closestPosition
    }

    /// Get the origin point for a snap position
    func getSnapOrigin(for position: RecordingIndicatorController.ScreenPosition) -> CGPoint? {
        guard let screen = NSScreen.main else { return nil }
        return calculateOrigin(for: position, screen: screen)
    }

    private func calculateOrigin(for position: RecordingIndicatorController.ScreenPosition, screen: NSScreen) -> CGPoint {
        let screenFrame = screen.visibleFrame
        let windowWidth: CGFloat = 70
        let windowHeight: CGFloat = 42
        let sideMargin: CGFloat = 16
        let topMargin: CGFloat = 8   // Closer to top edge
        let bottomMargin: CGFloat = 16

        switch position {
        case .topLeft:
            return CGPoint(
                x: screenFrame.minX + sideMargin,
                y: screenFrame.maxY - windowHeight - topMargin
            )
        case .topCenter:
            return CGPoint(
                x: screenFrame.midX - windowWidth / 2,
                y: screenFrame.maxY - windowHeight - topMargin
            )
        case .topRight:
            return CGPoint(
                x: screenFrame.maxX - windowWidth - sideMargin,
                y: screenFrame.maxY - windowHeight - topMargin
            )
        case .middleLeft:
            return CGPoint(
                x: screenFrame.minX + sideMargin,
                y: screenFrame.midY - windowHeight / 2
            )
        case .middleRight:
            return CGPoint(
                x: screenFrame.maxX - windowWidth - sideMargin,
                y: screenFrame.midY - windowHeight / 2
            )
        case .bottomLeft:
            return CGPoint(
                x: screenFrame.minX + sideMargin,
                y: screenFrame.minY + bottomMargin
            )
        case .bottomCenter:
            return CGPoint(
                x: screenFrame.midX - windowWidth / 2,
                y: screenFrame.minY + bottomMargin
            )
        case .bottomRight:
            return CGPoint(
                x: screenFrame.maxX - windowWidth - sideMargin,
                y: screenFrame.minY + bottomMargin
            )
        }
    }

    private func createPlaceholderWindow(at origin: CGPoint, highlighted: Bool) -> NSWindow {
        let contentView = SnapPlaceholderView(isHighlighted: highlighted)
        let hostingView = NSHostingView(rootView: contentView)
        hostingView.frame = CGRect(x: 0, y: 0, width: 70, height: 42)
        if #available(macOS 13.0, *) { hostingView.sizingOptions = [] }

        let window = FloatingIndicatorWindow(
            contentRect: hostingView.frame,
            styleMask: [.borderless],
            backing: .buffered,
            defer: false
        )
        window.contentView = hostingView
        window.isOpaque = false
        window.backgroundColor = .clear
        window.level = .floating
        window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        window.setFrameOrigin(origin)

        return window
    }

    private func updateWindowHighlight(_ window: NSWindow, highlighted: Bool) {
        let contentView = SnapPlaceholderView(isHighlighted: highlighted)
        let hostingView = NSHostingView(rootView: contentView)
        hostingView.frame = window.contentView?.frame ?? CGRect(x: 0, y: 0, width: 70, height: 42)
        if #available(macOS 13.0, *) { hostingView.sizingOptions = [] }
        window.contentView = hostingView
    }
}

// =============================================================================
// FLOATING WINDOW CONTROLLER
// =============================================================================

/// Manages the floating recording indicator window
/// Has its own volume polling isolated from AppState to prevent cascading re-renders
class RecordingIndicatorController: NSObject, ObservableObject, NSWindowDelegate {
    private var window: NSWindow?
    private var appState: AppState?
    private var volumeTimer: Timer?

    /// Snap placeholder controller for drag-and-snap
    private let snapPlaceholders = SnapPlaceholderController()

    /// Track if we're currently dragging
    private var isDragging: Bool = false

    /// Current volume level from microphone (0.0 to 1.0)
    /// Only this view observes this, so updates don't affect Settings or other views
    @Published var currentVolume: Double = 0.0

    /// Whether transcription is in progress (for dots rotation animation)
    @Published var isTranscribing: Bool = false

    /// Whether the indicator is in the process of disappearing (for fade-out animation)
    @Published var isDisappearing: Bool = false

    /// Screen position for the indicator
    enum ScreenPosition: String, CaseIterable {
        case topLeft, topCenter, topRight
        case middleLeft, middleRight
        case bottomLeft, bottomCenter, bottomRight

        var displayName: String {
            switch self {
            case .topLeft: return "Top Left"
            case .topCenter: return "Top Center"
            case .topRight: return "Top Right"
            case .middleLeft: return "Middle Left"
            case .middleRight: return "Middle Right"
            case .bottomLeft: return "Bottom Left"
            case .bottomCenter: return "Bottom Center"
            case .bottomRight: return "Bottom Right"
            }
        }
    }

    @Published var position: ScreenPosition = .topCenter
    @Published var isVisible: Bool = false

    override init() {
        super.init()
        // Load saved position
        if let savedPosition = UserDefaults.standard.string(forKey: "indicatorPosition"),
           let pos = ScreenPosition(rawValue: savedPosition) {
            position = pos
        }
    }

    func setup(appState: AppState) {
        self.appState = appState
    }

    func show() {
        // Always recreate window for fresh animation state
        if window != nil {
            window?.orderOut(nil)
            window = nil
        }
        createWindow()

        updatePosition()
        window?.orderFront(nil)
        isVisible = true

        // Reset states
        currentVolume = 0.0
        isTranscribing = false
        isDisappearing = false

        // Start volume polling when indicator is shown
        startVolumePolling()

        // Start drag monitoring
        startDragMonitoring()
    }

    func hide() {
        // Stop volume polling when hidden
        stopVolumePolling()

        // Stop drag monitoring
        stopDragMonitoring()

        // Hide any placeholders
        snapPlaceholders.hide()

        // Trigger disappear animation
        isDisappearing = true

        // Wait for animation to complete, then hide window
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) { [weak self] in
            self?.window?.orderOut(nil)
            self?.window = nil  // Destroy window so it's recreated fresh next time
            self?.isVisible = false
            self?.currentVolume = 0.0
            self?.isTranscribing = false
            self?.isDisappearing = false
        }
    }

    /// Set transcribing state (triggers dots rotation animation)
    func setTranscribing(_ transcribing: Bool) {
        isTranscribing = transcribing
    }

    func setPosition(_ newPosition: ScreenPosition) {
        position = newPosition
        UserDefaults.standard.set(newPosition.rawValue, forKey: "indicatorPosition")
        if isVisible {
            updatePosition()
        }
    }

    // =========================================================================
    // DRAG AND SNAP
    // =========================================================================

    /// Timer to detect when dragging has stopped
    private var dragEndTimer: Timer?

    /// Flag to ignore windowDidMove events caused by our snap animation
    private var isSnapping: Bool = false

    private func startDragMonitoring() {
        // Set ourselves as the window delegate to get windowDidMove notifications
        window?.delegate = self
    }

    private func stopDragMonitoring() {
        dragEndTimer?.invalidate()
        dragEndTimer = nil
        if isDragging {
            snapPlaceholders.hide()
            isDragging = false
        }
        isSnapping = false
    }

    // NSWindowDelegate method - called when window position changes
    func windowDidMove(_ notification: Notification) {
        // Ignore moves caused by our snap animation
        guard !isSnapping else { return }

        guard let window = window, isVisible, !isDisappearing else { return }

        // Show placeholders on first move
        if !isDragging {
            isDragging = true
            snapPlaceholders.show()
        }

        // Update highlights based on current position
        _ = snapPlaceholders.updateHighlight(indicatorFrame: window.frame)

        // Reset the end-drag timer - it fires when movement stops
        dragEndTimer?.invalidate()
        dragEndTimer = Timer.scheduledTimer(withTimeInterval: 0.2, repeats: false) { [weak self] _ in
            self?.handleDragEnd()
        }
    }

    /// Called when dragging appears to have ended (no movement for 200ms)
    private func handleDragEnd() {
        guard let window = window, isDragging else { return }

        isDragging = false

        // Hide placeholders first
        snapPlaceholders.hide()

        // Check if we're near a snap position
        if let snapPosition = snapPlaceholders.updateHighlight(indicatorFrame: window.frame),
           let snapOrigin = snapPlaceholders.getSnapOrigin(for: snapPosition) {

            // Set snapping flag to ignore windowDidMove during animation
            isSnapping = true

            // Calculate target frame (same size, new origin)
            let targetFrame = CGRect(origin: snapOrigin, size: window.frame.size)

            // Use NSWindow's built-in animation
            NSAnimationContext.runAnimationGroup({ context in
                context.duration = 0.3
                context.allowsImplicitAnimation = true
                window.setFrame(targetFrame, display: true, animate: true)
            }, completionHandler: { [weak self] in
                // Clear snapping flag after a delay
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                    self?.isSnapping = false
                }
            })

            // Save the new position
            position = snapPosition
            UserDefaults.standard.set(snapPosition.rawValue, forKey: "indicatorPosition")
        }
    }

    // =========================================================================
    // VOLUME POLLING (Isolated from AppState)
    // =========================================================================

    private func startVolumePolling() {
        // Poll for volume at 10Hz (every 100ms) for responsive visualization
        volumeTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            self?.fetchVolume()
        }
        // Add to common run loop so it works during UI interactions
        if let timer = volumeTimer {
            RunLoop.main.add(timer, forMode: .common)
        }
    }

    private func stopVolumePolling() {
        volumeTimer?.invalidate()
        volumeTimer = nil
    }

    private func fetchVolume() {
        // Get volume from Swift's AudioRecorder (not Python backend)
        // The AudioRecorder is recording directly in the app, so we read its currentVolume
        guard isVisible, let appState = appState else { return }

        // Get volume from the Swift audio recorder
        let volume = Double(appState.audioRecorder.currentVolume)

        // Update on main thread
        DispatchQueue.main.async { [weak self] in
            self?.currentVolume = volume
            // Force window to redraw
            self?.window?.display()
        }
    }

    private func createWindow() {
        // Pass self to the view so it can observe our currentVolume
        let contentView = RecordingIndicatorView(controller: self)

        let hostingView = NSHostingView(rootView: contentView)
        hostingView.frame = CGRect(x: 0, y: 0, width: 70, height: 42)

        // Prevent NSHostingView from trying to auto-update the window's
        // content size min/max. This crashes on borderless floating windows
        // because the constraint system throws an NSException.
        if #available(macOS 13.0, *) {
            hostingView.sizingOptions = []
        }

        // Ensure the hosting view layer is fully transparent — prevents rectangular box artifact
        hostingView.wantsLayer = true
        hostingView.layer?.backgroundColor = .clear

        // Use FloatingIndicatorWindow to prevent the NSHostingView constraint crash
        // on macOS 26 Tahoe. It overrides _postWindowNeedsUpdateConstraints to be a
        // no-op, preventing the NSException that crashes borderless floating windows.
        let window = FloatingIndicatorWindow(
            contentRect: hostingView.frame,
            styleMask: [.borderless],
            backing: .buffered,
            defer: false
        )

        window.contentView = hostingView
        window.isOpaque = false
        window.backgroundColor = .clear
        window.hasShadow = false
        window.level = .floating
        window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        window.isMovableByWindowBackground = true
        window.ignoresMouseEvents = false

        // Force the window's content view to be fully transparent at the AppKit level
        if let contentView = window.contentView {
            contentView.wantsLayer = true
            contentView.layer?.backgroundColor = .clear
            contentView.layer?.isOpaque = false
        }

        self.window = window
    }

    private func updatePosition() {
        guard let window = window,
              let screen = NSScreen.main else { return }

        let screenFrame = screen.visibleFrame
        let windowSize = window.frame.size
        let sideMargin: CGFloat = 16
        let topMargin: CGFloat = 8   // Closer to top edge
        let bottomMargin: CGFloat = 16

        var newOrigin: CGPoint

        switch position {
        case .topLeft:
            newOrigin = CGPoint(
                x: screenFrame.minX + sideMargin,
                y: screenFrame.maxY - windowSize.height - topMargin
            )
        case .topCenter:
            newOrigin = CGPoint(
                x: screenFrame.midX - windowSize.width / 2,
                y: screenFrame.maxY - windowSize.height - topMargin
            )
        case .topRight:
            newOrigin = CGPoint(
                x: screenFrame.maxX - windowSize.width - sideMargin,
                y: screenFrame.maxY - windowSize.height - topMargin
            )
        case .middleLeft:
            newOrigin = CGPoint(
                x: screenFrame.minX + sideMargin,
                y: screenFrame.midY - windowSize.height / 2
            )
        case .middleRight:
            newOrigin = CGPoint(
                x: screenFrame.maxX - windowSize.width - sideMargin,
                y: screenFrame.midY - windowSize.height / 2
            )
        case .bottomLeft:
            newOrigin = CGPoint(
                x: screenFrame.minX + sideMargin,
                y: screenFrame.minY + bottomMargin
            )
        case .bottomCenter:
            newOrigin = CGPoint(
                x: screenFrame.midX - windowSize.width / 2,
                y: screenFrame.minY + bottomMargin
            )
        case .bottomRight:
            newOrigin = CGPoint(
                x: screenFrame.maxX - windowSize.width - sideMargin,
                y: screenFrame.minY + bottomMargin
            )
        }

        window.setFrameOrigin(newOrigin)
    }
}
