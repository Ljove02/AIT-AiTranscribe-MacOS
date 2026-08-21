/*
 SessionIndicatorView.swift
 ==========================

 A floating capsule indicator shown during session recording.
 Wider than the quick-transcribe indicator, with a pulsing red dot
 and HH:MM:SS timer display.

 ARCHITECTURE NOTE:
 ------------------
 On macOS 26 Tahoe, NSHostingView's SwiftUI rendering pipeline fails on
 borderless floating windows because _postWindowNeedsDisplay throws during
 layout. This prevents SwiftUI from ever re-rendering — the initial render
 works but no updates are possible.

 To work around this, the session indicator is built entirely with Core
 Animation (CALayer + CATextLayer + CAShapeLayer). Core Animation renders
 through the GPU compositor, completely bypassing AppKit's view display
 system and the broken _postWindowNeedsDisplay path. Timer updates go
 directly to CATextLayer.string, which Core Animation composites immediately.
 */

import AppKit
import Combine
import QuartzCore

// =============================================================================
// SESSION INDICATOR CONTROLLER
// =============================================================================

/// Manages the floating session indicator window.
/// Uses pure Core Animation layers (no SwiftUI) for macOS 26 Tahoe compatibility.
class SessionIndicatorController: NSObject, ObservableObject, NSWindowDelegate {
    private var window: NSWindow?
    private var durationTimer: Timer?

    // Core Animation layers
    private var containerLayer: CALayer?
    private var dotLayer: CALayer?
    private var timerLayer: CATextLayer?

    /// Meter arcs. Mic runs along the top of the capsule, desktop along the
    /// bottom. Each is drawn as an unlit track plus three lit segments, one
    /// per zone, so the bar changes colour along its length the way OBS does
    /// rather than recolouring all at once.
    private var micMeterLayers: MeterArcLayers?
    private var systemMeterLayers: MeterArcLayers?
    private var meterTimer: Timer?

    /// Pulled from, not pushed to: the meter applies its ballistics from the
    /// time since the last read, so the indicator can redraw at its own rate.
    weak var levelSource: SessionRecorder?

    /// Snap placeholder controller for drag-and-snap (reuses existing snap system)
    private let snapPlaceholders = SessionSnapPlaceholderController()

    private var isDragging = false
    private var dragEndTimer: Timer?
    private var isSnapping = false

    @Published var duration: TimeInterval = 0
    @Published var isVisible = false
    @Published var isDisappearing = false
    @Published var isProcessing = false
    @Published var position: ScreenPosition = .topLeft

    /// Reuse the same 8-position enum as RecordingIndicatorController
    enum ScreenPosition: String, CaseIterable {
        case topLeft, topCenter, topRight
        case middleLeft, middleRight
        case bottomLeft, bottomCenter, bottomRight
    }

    override init() {
        super.init()
        // Load saved position (separate key from recording indicator)
        if let saved = UserDefaults.standard.string(forKey: "sessionIndicatorPosition"),
           let pos = ScreenPosition(rawValue: saved) {
            position = pos
        }
    }

    func show() {
        if window != nil {
            window?.orderOut(nil)
            window = nil
        }
        dotLayer = nil
        timerLayer = nil
        containerLayer = nil

        duration = 0
        isDisappearing = false
        isProcessing = false
        createWindow()
        updatePosition()
        window?.orderFront(nil)
        isVisible = true

        // Appear animation (spring scale + fade)
        if let layer = window?.contentView?.layer {
            layer.opacity = 0
            layer.transform = CATransform3DMakeScale(0.3, 0.3, 1.0)

            DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) {
                CATransaction.begin()
                CATransaction.setAnimationDuration(0.4)
                CATransaction.setAnimationTimingFunction(CAMediaTimingFunction(name: .easeOut))

                layer.opacity = 1.0
                layer.transform = CATransform3DIdentity

                CATransaction.commit()
            }
        }

        // Start timer to update duration every second
        durationTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            guard let self else { return }
            self.duration += 1
            self.updateTimerText()
        }
        if let timer = durationTimer {
            RunLoop.main.add(timer, forMode: .common)
        }

        startMeterUpdates()
        window?.delegate = self
    }

    func showProcessing() {
        durationTimer?.invalidate()
        durationTimer = nil
        stopMeterUpdates()
        isProcessing = true

        CATransaction.begin()
        CATransaction.setDisableActions(true)
        // Hide the red dot and shift text left to use the freed space
        dotLayer?.isHidden = true
        timerLayer?.frame = CGRect(x: 14, y: 9, width: 84, height: 18)
        CATransaction.commit()

        updateTimerText()
    }

    func hide() {
        durationTimer?.invalidate()
        durationTimer = nil
        stopMeterUpdates()
        snapPlaceholders.hide()

        isProcessing = false
        isDisappearing = true

        // Disappear animation (scale down + fade out)
        if let layer = window?.contentView?.layer {
            CATransaction.begin()
            CATransaction.setAnimationDuration(0.3)
            CATransaction.setAnimationTimingFunction(CAMediaTimingFunction(name: .easeInEaseOut))

            layer.opacity = 0
            layer.transform = CATransform3DMakeScale(0.3, 0.3, 1.0)

            CATransaction.commit()
        }

        DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) { [weak self] in
            self?.dotLayer = nil
            self?.timerLayer = nil
            self?.micMeterLayers = nil
            self?.systemMeterLayers = nil
            self?.containerLayer = nil
            self?.window?.orderOut(nil)
            self?.window = nil
            self?.isVisible = false
            self?.isDisappearing = false
            self?.duration = 0
        }
    }

    /// Sync duration from SessionManager (call from outside when duration updates)
    func updateDuration(_ newDuration: TimeInterval) {
        duration = newDuration
    }

    // MARK: - Meters

    private func startMeterUpdates() {
        meterTimer?.invalidate()
        let timer = Timer.scheduledTimer(withTimeInterval: 1.0 / 30.0, repeats: true) { [weak self] _ in
            self?.updateMeters()
        }
        RunLoop.main.add(timer, forMode: .common)
        meterTimer = timer
    }

    private func stopMeterUpdates() {
        meterTimer?.invalidate()
        meterTimer = nil
        guard let mic = micMeterLayers, let system = systemMeterLayers else { return }
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        MeterArcStyle.apply(.silent, to: mic)
        MeterArcStyle.apply(.silent, to: system)
        CATransaction.commit()
    }

    private func updateMeters() {
        guard let source = levelSource,
              let mic = micMeterLayers,
              let system = systemMeterLayers else { return }

        // Core Animation would otherwise animate every strokeEnd change over
        // its default quarter second, which lags the audio badly at 30 fps.
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        MeterArcStyle.apply(source.micMeter.reading(), to: mic)
        MeterArcStyle.apply(source.systemMeter.reading(), to: system)
        CATransaction.commit()
    }

    // MARK: - Timer Text

    private func updateTimerText() {
        let text = isProcessing
            ? "Processing..."
            : SessionManager.formatDurationHHMMSS(duration)

        CATransaction.begin()
        CATransaction.setDisableActions(true)  // No implicit animation on text change
        timerLayer?.string = makeAttributedString(text)
        CATransaction.commit()
    }

    private func makeAttributedString(_ text: String) -> NSAttributedString {
        let isDark = NSApp.effectiveAppearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
        let font = isProcessing
            ? NSFont.systemFont(ofSize: 13, weight: .medium)
            : NSFont.monospacedSystemFont(ofSize: 13, weight: .medium)
        let color = isDark ? NSColor.white : NSColor.labelColor

        return NSAttributedString(string: text, attributes: [
            .font: font,
            .foregroundColor: color
        ])
    }

    // MARK: - Drag and Snap

    nonisolated func windowDidMove(_ notification: Notification) {
        guard !isSnapping else { return }
        guard let window, isVisible, !isDisappearing else { return }

        if !isDragging {
            isDragging = true
            snapPlaceholders.show()
        }

        _ = snapPlaceholders.updateHighlight(indicatorFrame: window.frame)

        dragEndTimer?.invalidate()
        dragEndTimer = Timer.scheduledTimer(withTimeInterval: 0.2, repeats: false) { [weak self] _ in
            self?.handleDragEnd()
        }
    }

    private func handleDragEnd() {
        guard let window, isDragging else { return }
        isDragging = false
        snapPlaceholders.hide()

        if let snapPos = snapPlaceholders.updateHighlight(indicatorFrame: window.frame),
           let snapOrigin = snapPlaceholders.getSnapOrigin(for: snapPos) {

            isSnapping = true
            let targetFrame = CGRect(origin: snapOrigin, size: window.frame.size)

            NSAnimationContext.runAnimationGroup({ context in
                context.duration = 0.3
                context.allowsImplicitAnimation = true
                window.setFrame(targetFrame, display: true, animate: true)
            }, completionHandler: { [weak self] in
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                    self?.isSnapping = false
                }
            })

            position = snapPos
            UserDefaults.standard.set(snapPos.rawValue, forKey: "sessionIndicatorPosition")
        }
    }

    // MARK: - Window Management

    private func createWindow() {
        let isDark = NSApp.effectiveAppearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
        let scale = NSScreen.main?.backingScaleFactor ?? 2.0
        let windowWidth: CGFloat = 110
        let windowHeight: CGFloat = 36

        // Container NSView — pure layer-backed, no SwiftUI
        let container = NSView(frame: CGRect(x: 0, y: 0, width: windowWidth, height: windowHeight))
        container.wantsLayer = true
        container.layer?.backgroundColor = .clear

        // --- Capsule background ---
        let bgLayer = CALayer()
        bgLayer.frame = container.bounds
        bgLayer.cornerRadius = windowHeight / 2  // Makes it a capsule
        bgLayer.backgroundColor = isDark
            ? NSColor(white: 0.15, alpha: 0.65).cgColor
            : NSColor(white: 0.92, alpha: 0.85).cgColor
        container.layer?.addSublayer(bgLayer)

        // --- Capsule border ---
        let borderLayer = CAShapeLayer()
        let borderRect = container.bounds.insetBy(dx: 0.25, dy: 0.25)
        borderLayer.path = CGPath(roundedRect: borderRect,
                                  cornerWidth: borderRect.height / 2,
                                  cornerHeight: borderRect.height / 2,
                                  transform: nil)
        borderLayer.strokeColor = isDark
            ? NSColor.white.withAlphaComponent(0.15).cgColor
            : NSColor.black.withAlphaComponent(0.1).cgColor
        borderLayer.fillColor = nil
        borderLayer.lineWidth = 0.5
        container.layer?.addSublayer(borderLayer)

        // --- Meter arcs (behind the dot and timer) ---
        micMeterLayers = MeterArcStyle.makeLayers(in: container.bounds, top: true,
                                                  parent: container.layer!)
        systemMeterLayers = MeterArcStyle.makeLayers(in: container.bounds, top: false,
                                                     parent: container.layer!)

        // --- Red dot (pulsing) ---
        let dot = CALayer()
        dot.frame = CGRect(x: 14, y: (windowHeight - 8) / 2, width: 8, height: 8)
        dot.cornerRadius = 4
        dot.backgroundColor = NSColor.red.cgColor
        container.layer?.addSublayer(dot)
        self.dotLayer = dot

        // Pulsing animation
        let pulse = CABasicAnimation(keyPath: "opacity")
        pulse.fromValue = 1.0
        pulse.toValue = 0.3
        pulse.duration = 1.5
        pulse.autoreverses = true
        pulse.repeatCount = .infinity
        pulse.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
        dot.add(pulse, forKey: "pulse")

        // --- Timer text ---
        let textLayer = CATextLayer()
        textLayer.string = makeAttributedString("00:00:00")
        // Position: after dot (8) + spacing (8), starting at x=30
        // Vertically centered: (36 - 18) / 2 = 9
        textLayer.frame = CGRect(x: 30, y: 9, width: 68, height: 18)
        textLayer.alignmentMode = .left
        textLayer.contentsScale = scale
        textLayer.truncationMode = .end
        container.layer?.addSublayer(textLayer)
        self.timerLayer = textLayer

        self.containerLayer = container.layer

        // --- Window ---
        let window = FloatingIndicatorWindow(
            contentRect: container.frame,
            styleMask: [.borderless],
            backing: .buffered,
            defer: false
        )

        window.contentView = container
        window.isOpaque = false
        window.backgroundColor = .clear
        window.hasShadow = false
        window.level = .floating
        window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        window.isMovableByWindowBackground = true
        window.ignoresMouseEvents = false

        self.window = window
    }

    private func updatePosition() {
        guard let window, let screen = NSScreen.main else { return }

        let screenFrame = screen.visibleFrame
        let windowSize = window.frame.size
        let sideMargin: CGFloat = 16
        let topMargin: CGFloat = 8
        let bottomMargin: CGFloat = 16

        var origin: CGPoint
        switch position {
        case .topLeft:
            origin = CGPoint(x: screenFrame.minX + sideMargin, y: screenFrame.maxY - windowSize.height - topMargin)
        case .topCenter:
            origin = CGPoint(x: screenFrame.midX - windowSize.width / 2, y: screenFrame.maxY - windowSize.height - topMargin)
        case .topRight:
            origin = CGPoint(x: screenFrame.maxX - windowSize.width - sideMargin, y: screenFrame.maxY - windowSize.height - topMargin)
        case .middleLeft:
            origin = CGPoint(x: screenFrame.minX + sideMargin, y: screenFrame.midY - windowSize.height / 2)
        case .middleRight:
            origin = CGPoint(x: screenFrame.maxX - windowSize.width - sideMargin, y: screenFrame.midY - windowSize.height / 2)
        case .bottomLeft:
            origin = CGPoint(x: screenFrame.minX + sideMargin, y: screenFrame.minY + bottomMargin)
        case .bottomCenter:
            origin = CGPoint(x: screenFrame.midX - windowSize.width / 2, y: screenFrame.minY + bottomMargin)
        case .bottomRight:
            origin = CGPoint(x: screenFrame.maxX - windowSize.width - sideMargin, y: screenFrame.minY + bottomMargin)
        }

        window.setFrameOrigin(origin)
    }
}

// =============================================================================
// SESSION SNAP PLACEHOLDER CONTROLLER
// =============================================================================

/// Manages the 8 placeholder windows shown during session indicator drag.
/// Same logic as SnapPlaceholderController but with session indicator dimensions.
class SessionSnapPlaceholderController {
    private var windows: [SessionIndicatorController.ScreenPosition: NSWindow] = [:]
    private var highlightedPosition: SessionIndicatorController.ScreenPosition?

    private let windowWidth: CGFloat = 110
    private let windowHeight: CGFloat = 36

    func show() {
        guard let screen = NSScreen.main else { return }
        for position in SessionIndicatorController.ScreenPosition.allCases {
            let origin = calculateOrigin(for: position, screen: screen)
            let window = createPlaceholderWindow(at: origin)
            window.orderFront(nil)
            windows[position] = window
        }
    }

    func hide() {
        for window in windows.values {
            window.orderOut(nil)
        }
        windows.removeAll()
        highlightedPosition = nil
    }

    func updateHighlight(indicatorFrame: CGRect) -> SessionIndicatorController.ScreenPosition? {
        guard let screen = NSScreen.main else { return nil }

        let center = CGPoint(x: indicatorFrame.midX, y: indicatorFrame.midY)
        let snapThreshold: CGFloat = 50
        var closest: SessionIndicatorController.ScreenPosition?
        var closestDist: CGFloat = .infinity

        for position in SessionIndicatorController.ScreenPosition.allCases {
            let origin = calculateOrigin(for: position, screen: screen)
            let posCenter = CGPoint(x: origin.x + windowWidth / 2, y: origin.y + windowHeight / 2)
            let dist = hypot(center.x - posCenter.x, center.y - posCenter.y)
            if dist < snapThreshold && dist < closestDist {
                closestDist = dist
                closest = position
            }
        }

        if closest != highlightedPosition {
            if let prev = highlightedPosition, let w = windows[prev] {
                updateWindowHighlight(w, highlighted: false)
            }
            if let new = closest, let w = windows[new] {
                updateWindowHighlight(w, highlighted: true)
            }
            highlightedPosition = closest
        }

        return closest
    }

    func getSnapOrigin(for position: SessionIndicatorController.ScreenPosition) -> CGPoint? {
        guard let screen = NSScreen.main else { return nil }
        return calculateOrigin(for: position, screen: screen)
    }

    private func calculateOrigin(for position: SessionIndicatorController.ScreenPosition, screen: NSScreen) -> CGPoint {
        let sf = screen.visibleFrame
        let sm: CGFloat = 16
        let tm: CGFloat = 8
        let bm: CGFloat = 16

        switch position {
        case .topLeft:     return CGPoint(x: sf.minX + sm, y: sf.maxY - windowHeight - tm)
        case .topCenter:   return CGPoint(x: sf.midX - windowWidth / 2, y: sf.maxY - windowHeight - tm)
        case .topRight:    return CGPoint(x: sf.maxX - windowWidth - sm, y: sf.maxY - windowHeight - tm)
        case .middleLeft:  return CGPoint(x: sf.minX + sm, y: sf.midY - windowHeight / 2)
        case .middleRight: return CGPoint(x: sf.maxX - windowWidth - sm, y: sf.midY - windowHeight / 2)
        case .bottomLeft:  return CGPoint(x: sf.minX + sm, y: sf.minY + bm)
        case .bottomCenter:return CGPoint(x: sf.midX - windowWidth / 2, y: sf.minY + bm)
        case .bottomRight: return CGPoint(x: sf.maxX - windowWidth - sm, y: sf.minY + bm)
        }
    }

    private func createPlaceholderWindow(at origin: CGPoint) -> NSWindow {
        let view = createPlaceholderView(highlighted: false)
        let window = FloatingIndicatorWindow(
            contentRect: CGRect(x: 0, y: 0, width: windowWidth, height: windowHeight),
            styleMask: [.borderless],
            backing: .buffered,
            defer: false
        )
        window.contentView = view
        window.isOpaque = false
        window.backgroundColor = .clear
        window.level = .floating
        window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        window.setFrameOrigin(origin)
        return window
    }

    private func updateWindowHighlight(_ window: NSWindow, highlighted: Bool) {
        let view = createPlaceholderView(highlighted: highlighted)
        window.contentView = view
    }

    /// Create a pure-AppKit placeholder view (no SwiftUI — avoids macOS 26 display issues)
    private func createPlaceholderView(highlighted: Bool) -> NSView {
        let view = NSView(frame: CGRect(x: 0, y: 0, width: windowWidth, height: windowHeight))
        view.wantsLayer = true
        view.layer?.backgroundColor = .clear

        let bg = CALayer()
        bg.frame = view.bounds
        bg.cornerRadius = windowHeight / 2
        bg.backgroundColor = NSColor(white: 0.5, alpha: highlighted ? 0.35 : 0.15).cgColor
        bg.borderColor = NSColor.white.withAlphaComponent(highlighted ? 0.5 : 0.2).cgColor
        bg.borderWidth = 1
        if highlighted {
            bg.transform = CATransform3DMakeScale(1.05, 1.05, 1.0)
        }
        view.layer?.addSublayer(bg)

        return view
    }
}

// =============================================================================
// METER ARCS
// =============================================================================

/// The layers making up one meter arc.
///
/// A lit segment per zone rather than one recoloured bar: the green part stays
/// green when the tip goes yellow, which is what tells you at a glance how much
/// of the signal is hot rather than just that some of it is.
struct MeterArcLayers {
    let track: CAShapeLayer
    let nominal: CAShapeLayer
    let warning: CAShapeLayer
    let error: CAShapeLayer
    let peakHold: CAShapeLayer
}

enum MeterArcStyle {
    /// OBS's Yami theme colours, lit and unlit.
    static let litNominal = NSColor(srgbRed: 0.216, green: 0.824, blue: 0.278, alpha: 1)   // #37D247
    static let litWarning = NSColor(srgbRed: 0.898, green: 0.686, blue: 0.141, alpha: 1)   // #E5AF24
    static let litError   = NSColor(srgbRed: 0.890, green: 0.231, blue: 0.341, alpha: 1)   // #E33B57
    static let dimNominal = NSColor(srgbRed: 0.090, green: 0.392, blue: 0.118, alpha: 1)   // #17641E
    static let dimWarning = NSColor(srgbRed: 0.431, green: 0.322, blue: 0.051, alpha: 1)   // #6E520D
    static let dimError   = NSColor(srgbRed: 0.490, green: 0.071, blue: 0.141, alpha: 1)   // #7D1224

    static let lineWidth: CGFloat = 1.6
    /// Gap between the capsule border and the meter.
    static let inset: CGFloat = 4.0

    /// Where each zone boundary falls along the bar, on the same -60..0 dB
    /// scale the meter reports.
    static var warningStart: CGFloat {
        CGFloat(1 - (AudioLevelMeter.warningDB / AudioLevelMeter.minimumDB))
    }
    static var errorStart: CGFloat {
        CGFloat(1 - (AudioLevelMeter.errorDB / AudioLevelMeter.minimumDB))
    }

    /// The arc a meter is drawn along: an inset copy of the capsule outline,
    /// cut in half. The top meter is the upper half, the bottom meter the
    /// lower half, and both run left to right so they fill in the same
    /// direction.
    ///
    /// It has to be built from the capsule geometry rather than drawn as a
    /// curve through roughly the right places. A hand-fitted bezier does not
    /// stay parallel to the border, and the mismatch is obvious at the ends
    /// where it should be hugging the rounded caps.
    static func path(in bounds: CGRect, top: Bool) -> CGPath {
        let rect = bounds.insetBy(dx: inset, dy: inset)
        let radius = rect.height / 2
        let leftCenter = CGPoint(x: rect.minX + radius, y: rect.midY)
        let rightCenter = CGPoint(x: rect.maxX - radius, y: rect.midY)

        let path = CGMutablePath()
        if top {
            // Left cap from the middle up to the top, across, then down the
            // right cap to the middle again.
            path.addArc(center: leftCenter, radius: radius,
                        startAngle: .pi, endAngle: .pi / 2, clockwise: true)
            path.addLine(to: CGPoint(x: rect.maxX - radius, y: rect.maxY))
            path.addArc(center: rightCenter, radius: radius,
                        startAngle: .pi / 2, endAngle: 0, clockwise: true)
        } else {
            path.addArc(center: leftCenter, radius: radius,
                        startAngle: .pi, endAngle: 3 * .pi / 2, clockwise: false)
            path.addLine(to: CGPoint(x: rect.maxX - radius, y: rect.minY))
            path.addArc(center: rightCenter, radius: radius,
                        startAngle: 3 * .pi / 2, endAngle: 2 * .pi, clockwise: false)
        }
        return path
    }

    static func makeLayers(in bounds: CGRect, top: Bool, parent: CALayer) -> MeterArcLayers {
        let arc = path(in: bounds, top: top)

        func stroke(_ color: NSColor, width: CGFloat = lineWidth) -> CAShapeLayer {
            let layer = CAShapeLayer()
            layer.path = arc
            layer.strokeColor = color.cgColor
            layer.fillColor = nil
            layer.lineWidth = width
            layer.lineCap = .round
            layer.strokeStart = 0
            layer.strokeEnd = 0
            parent.addSublayer(layer)
            return layer
        }

        // Unlit track, drawn once in three pieces so the zones are visible
        // before any sound arrives.
        let track = CAShapeLayer()
        track.path = arc
        track.strokeColor = dimNominal.withAlphaComponent(0.38).cgColor
        track.fillColor = nil
        track.lineWidth = lineWidth
        track.lineCap = .round
        track.strokeStart = 0
        track.strokeEnd = 1
        parent.addSublayer(track)

        let dimWarn = stroke(dimWarning.withAlphaComponent(0.38))
        dimWarn.strokeStart = warningStart
        dimWarn.strokeEnd = errorStart
        let dimErr = stroke(dimError.withAlphaComponent(0.38))
        dimErr.strokeStart = errorStart
        dimErr.strokeEnd = 1

        let nominal = stroke(litNominal)
        let warning = stroke(litWarning)
        let error = stroke(litError)
        let peakHold = stroke(.white, width: lineWidth)
        peakHold.opacity = 0.9

        return MeterArcLayers(track: track, nominal: nominal, warning: warning,
                              error: error, peakHold: peakHold)
    }

    /// Maps a reading onto the arc. Each zone layer is clipped to its own slice
    /// of the bar, so they light up in sequence instead of overlapping.
    static func apply(_ reading: AudioLevelReading, to layers: MeterArcLayers) {
        let level = CGFloat(reading.fraction)

        layers.nominal.strokeStart = 0
        layers.nominal.strokeEnd = min(level, warningStart)

        layers.warning.strokeStart = warningStart
        layers.warning.strokeEnd = max(warningStart, min(level, errorStart))

        layers.error.strokeStart = errorStart
        layers.error.strokeEnd = max(errorStart, min(level, 1))

        // A short tick sitting at the highest recent peak.
        let hold = CGFloat(reading.peakHoldFraction)
        if hold > 0.02 {
            layers.peakHold.strokeStart = max(0, hold - 0.015)
            layers.peakHold.strokeEnd = min(1, hold)
            layers.peakHold.isHidden = false
        } else {
            layers.peakHold.isHidden = true
        }
    }
}
