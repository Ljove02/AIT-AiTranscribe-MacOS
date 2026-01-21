/*
 RecordingIndicator.swift
 ========================

 A small floating window that shows when recording is active.
 Features:
 - Circular design with concentric rings
 - RECORDING: Rings pulse OUTWARD from center when speaking
 - TRANSCRIBING: Rings pulse INWARD toward center (implode effect)
 - Dark center (#0C0D10)
 - Glass UI background (Apple material effect)
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

    /// Individual ring phases for OUTWARD pulse (0 to 1, controls expansion + fade) - max 2 rings
    @State private var ringPhases: [CGFloat] = [0, 0]

    /// Timer for continuous ring animation
    @State private var animationTimer: Timer?

    /// Appear/disappear animation state
    @State private var appearScale: CGFloat = 1.0
    @State private var appearOpacity: Double = 1.0
    @State private var hasAppeared: Bool = false

    /// Transcription dots animation phase (0 to 1, controls expand/contract cycle)
    @State private var dotsPhase: CGFloat = 0

    /// Whether to show the transcription dots animation
    @State private var showTranscriptionDots: Bool = false

    /// Start time for smooth dots animation
    @State private var dotsAnimationStartTime: Date = Date()

    /// Core circle color (#0C0D10)
    private let coreColor = Color(red: 12/255, green: 13/255, blue: 16/255)

    /// Ring color
    private let ringColor = Color(white: 0.55)

    /// Core circle size (smaller, more compact)
    private let coreSize: CGFloat = 22

    /// Maximum ring expansion from core
    private let maxRingExpansion: CGFloat = 14

    var body: some View {
        ZStack {
            // OUTWARD pulsing rings (during recording/speaking) - max 2 rings
            if !controller.isTranscribing {
                ForEach(0..<2, id: \.self) { index in
                    let phase = ringPhases[index]
                    // Ring expands from core size outward
                    let ringDiameter = coreSize + 4 + phase * maxRingExpansion

                    Circle()
                        .stroke(ringColor.opacity(ringOpacity(for: phase)), lineWidth: 1.2)
                        .frame(width: ringDiameter, height: ringDiameter)
                }
            }

            // Core circle with clear glass effect and white border
            Circle()
                .frame(width: coreSize, height: coreSize)
                .glassEffect()
                .overlay(
                    Circle()
                        .stroke(Color.white.opacity(0.3), lineWidth: 0.5)
                )

            // Transcription dots animation
            // 3 dots that expand from center to triangle, then contract back
            // Using TimelineView for GPU-synchronized smooth animation
            if showTranscriptionDots {
                TimelineView(.animation) { timeline in
                    let phase = computeDotsPhase(from: timeline.date)
                    ZStack {
                        ForEach(0..<3, id: \.self) { index in
                            Circle()
                                .fill(Color.white)
                                .frame(width: 3, height: 3)
                                .offset(transcriptionDotOffsetSmooth(for: index, phase: phase))
                        }
                    }
                    .frame(width: coreSize, height: coreSize)
                    .rotationEffect(.degrees(Double(phase) * 180), anchor: .center)
                }
            }
        }
        .frame(width: 64, height: 36)
        .background(
            ZStack {
                // Glass background - more transparent for better see-through effect
                Capsule()
                    .fill(.ultraThinMaterial.opacity(0.6))

                // Subtle inner glow/highlight at top
                Capsule()
                    .fill(
                        LinearGradient(
                            colors: [Color.white.opacity(0.12), Color.clear],
                            startPoint: .top,
                            endPoint: .center
                        )
                    )
            }
            .shadow(color: .black.opacity(0.15), radius: 10, x: 0, y: 3)
        )
        .overlay(
            Capsule()
                .stroke(
                    LinearGradient(
                        colors: [Color.white.opacity(0.25), Color.white.opacity(0.08)],
                        startPoint: .top,
                        endPoint: .bottom
                    ),
                    lineWidth: 0.5
                )
        )
        // Appear/disappear animation
        .scaleEffect(appearScale)
        .opacity(appearOpacity)
        .onAppear {
            startAnimationLoop()
            // Animate in: small dot -> full size (only on first appear)
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
            stopAnimationLoop()
            hasAppeared = false
        }
        .onChange(of: controller.currentVolume) { _, newVolume in
            if !controller.isTranscribing {
                updateSmoothedVolume(newVolume)
            }
        }
        .onChange(of: controller.isTranscribing) { _, isTranscribing in
            if isTranscribing {
                // Start transcription dots animation
                startTranscriptionDots()
            } else {
                // Stop transcription dots
                stopTranscriptionDots()
            }
        }
        .onChange(of: controller.isDisappearing) { _, isDisappearing in
            if isDisappearing {
                // Animate out: full size -> small dot and fade
                withAnimation(.easeInOut(duration: 0.3)) {
                    appearScale = 0.3
                    appearOpacity = 0
                }
            }
        }
    }

    /// Opacity for OUTWARD rings - fades out as ring expands
    private func ringOpacity(for phase: CGFloat) -> Double {
        // Fade in quickly, then fade out as it expands
        if phase < 0.15 {
            return Double(phase) * 4  // Quick fade in
        } else {
            return Double(max(0, 0.6 - (phase - 0.15) * 0.7))  // Gradual fade out
        }
    }

    /// Calculate dot offset for transcription animation (legacy timer-based)
    /// Dots expand from center (0) to triangle position, then contract back
    private func transcriptionDotOffset(for index: Int) -> CGSize {
        let maxRadius: CGFloat = 3.5  // Max distance from center - smaller, tighter triangle

        // Use sin to create smooth expand/contract: 0 → max → 0
        // dotsPhase goes 0 to 1, sin(phase * π) gives 0 → 1 → 0
        let expandFactor = CoreGraphics.sin(dotsPhase * .pi)
        let currentRadius = maxRadius * expandFactor

        // Each dot is 120 degrees apart, starting at top (-90 degrees)
        let angle = CGFloat(Double(index) * 120 - 90) * .pi / 180

        return CGSize(
            width: CoreGraphics.cos(angle) * currentRadius,
            height: CoreGraphics.sin(angle) * currentRadius
        )
    }

    /// Compute continuous phase (0 to 1) from current time for smooth animation
    private func computeDotsPhase(from date: Date) -> CGFloat {
        let elapsed = date.timeIntervalSince(dotsAnimationStartTime)
        let cycleDuration: Double = 1.2  // Seconds per full expand/contract cycle
        let phase = elapsed.truncatingRemainder(dividingBy: cycleDuration) / cycleDuration
        return CGFloat(phase)
    }

    /// Calculate dot offset with explicit phase parameter (for TimelineView)
    private func transcriptionDotOffsetSmooth(for index: Int, phase: CGFloat) -> CGSize {
        let maxRadius: CGFloat = 3.5

        // Use sin for smooth expand/contract: 0 → max → 0
        let expandFactor = CoreGraphics.sin(phase * .pi)
        let currentRadius = maxRadius * expandFactor

        // Each dot is 120 degrees apart, starting at top (-90 degrees)
        let angle = CGFloat(Double(index) * 120 - 90) * .pi / 180

        return CGSize(
            width: CoreGraphics.cos(angle) * currentRadius,
            height: CoreGraphics.sin(angle) * currentRadius
        )
    }

    /// Smooth the incoming volume for gradual transitions
    private func updateSmoothedVolume(_ volume: Double) {
        // Noise gate: ignore very low ambient noise
        // Raised from 0.004 to 0.008 to match AudioRecorder's sensitivity reduction
        let noiseGate: Double = 0.008
        let gatedVolume = volume > noiseGate ? volume - noiseGate : 0

        // Reduced amplification to match AudioRecorder (3x instead of 200x)
        // AudioRecorder now sends pre-scaled values, so we just need gentle boosting
        let amplified = min(1.0, gatedVolume * 100.0)

        // Smooth: fast attack, FASTER decay (was 0.85, now 0.70 for quicker stop)
        let newSmoothed: CGFloat
        if CGFloat(amplified) > smoothedVolume {
            newSmoothed = smoothedVolume * 0.3 + CGFloat(amplified) * 0.7  // Fast attack
        } else {
            newSmoothed = smoothedVolume * 0.70 + CGFloat(amplified) * 0.30  // Faster decay - stops quicker
        }
        smoothedVolume = newSmoothed
    }

    /// Start the continuous animation loop for ring pulsing
    private func startAnimationLoop() {
        // 60fps for smooth animation
        animationTimer = Timer.scheduledTimer(withTimeInterval: 0.016, repeats: true) { _ in
            if showTranscriptionDots {
                updateTranscriptionDots()
            } else {
                updateOutwardRingPhases()
            }
        }
        if let timer = animationTimer {
            RunLoop.main.add(timer, forMode: .common)
        }
    }

    private func stopAnimationLoop() {
        animationTimer?.invalidate()
        animationTimer = nil
    }

    /// Update outward ring phases - one constant pulsing ring when speaking
    private func updateOutwardRingPhases() {
        let speed: CGFloat = 0.02  // How fast rings expand (halved for 60fps)
        let threshold: CGFloat = 0.10  // Reduced from 0.15 to match new sensitivity (less reactive)

        // Update existing rings
        for i in 0..<2 {
            if ringPhases[i] > 0 {
                // Ring is active - continue expanding until it fades out
                ringPhases[i] = ringPhases[i] + speed
                if ringPhases[i] >= 1.0 {
                    ringPhases[i] = 0  // Reset when fully expanded
                }
            }
        }

        // Trigger ONE constant ring when speaking (regardless of volume intensity)
        if smoothedVolume > threshold {
            // Always keep exactly one ring active when speaking
            let activeRings = ringPhases.filter { $0 > 0 }.count

            // Start new ring only if none are active
            if activeRings == 0 {
                ringPhases[0] = 0.01  // Start first ring slot
            }
        }
    }

    /// Start the transcription dots animation
    private func startTranscriptionDots() {
        // Reset outward rings
        for i in 0..<2 {
            ringPhases[i] = 0
        }
        smoothedVolume = 0

        // Show dots after a brief delay
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) {
            dotsAnimationStartTime = Date()  // Record start time for smooth animation
            showTranscriptionDots = true
            dotsPhase = 0
        }
    }

    /// Stop the transcription dots animation
    private func stopTranscriptionDots() {
        showTranscriptionDots = false
        dotsPhase = 0
    }

    /// Update transcription dots - phase controls expand/contract cycle with rotation
    private func updateTranscriptionDots() {
        guard showTranscriptionDots else { return }

        let speed: CGFloat = 0.012  // Smooth, slower animation

        dotsPhase += speed
        if dotsPhase >= 1.0 {
            dotsPhase = 0  // Reset for next cycle
        }
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

        let window = NSWindow(
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
        }
    }

    private func createWindow() {
        // Pass self to the view so it can observe our currentVolume
        let contentView = RecordingIndicatorView(controller: self)

        let hostingView = NSHostingView(rootView: contentView)
        hostingView.frame = CGRect(x: 0, y: 0, width: 70, height: 42)

        let window = NSWindow(
            contentRect: hostingView.frame,
            styleMask: [.borderless],
            backing: .buffered,
            defer: false
        )

        window.contentView = hostingView
        window.isOpaque = false
        window.backgroundColor = .clear
        window.level = .floating  // Always on top
        window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        window.isMovableByWindowBackground = true  // Can drag to reposition

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
