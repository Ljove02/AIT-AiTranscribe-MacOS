/*
 SessionIndicatorView.swift
 ==========================

 A floating capsule indicator shown during session recording.
 Wider than the quick-transcribe indicator, with a pulsing red dot
 and HH:MM:SS timer display.

 Follows the same visual language and window management pattern
 as RecordingIndicator.swift but with a calmer, session-focused design.
 */

import SwiftUI
import AppKit
import Combine

// =============================================================================
// SESSION INDICATOR VIEW
// =============================================================================

struct SessionIndicatorView: View {
    @ObservedObject var controller: SessionIndicatorController

    @State private var dotOpacity: Double = 1.0
    @State private var appearScale: CGFloat = 0.3
    @State private var appearOpacity: Double = 0

    @Environment(\.colorScheme) private var colorScheme

    var body: some View {
        HStack(spacing: 8) {
            if controller.isProcessing {
                // Processing state: spinner + text
                ProgressView()
                    .scaleEffect(0.6)
                    .frame(width: 12, height: 12)
                Text("Processing...")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundColor(colorScheme == .dark ? .white : .primary)
            } else {
                // Recording state: pulsing red dot + timer
                Circle()
                    .fill(.red)
                    .frame(width: 8, height: 8)
                    .opacity(dotOpacity)

                Text(SessionManager.formatDurationHHMMSS(controller.duration))
                    .font(.system(size: 13, weight: .medium, design: .monospaced))
                    .foregroundColor(colorScheme == .dark ? .white : .primary)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .background(
            Capsule()
                .fill(
                    colorScheme == .dark
                        ? Color(white: 0.15, opacity: 0.65)
                        : Color(white: 0.92, opacity: 0.85)
                )
        )
        .overlay(
            Capsule()
                .stroke(
                    colorScheme == .dark
                        ? Color.white.opacity(0.15)
                        : Color.black.opacity(0.1),
                    lineWidth: 0.5
                )
        )
        .scaleEffect(appearScale)
        .opacity(appearOpacity)
        .onAppear {
            // Appear animation
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) {
                withAnimation(.spring(response: 0.4, dampingFraction: 0.7)) {
                    appearScale = 1.0
                    appearOpacity = 1.0
                }
            }

            // Subtle pulsing red dot (slower than recording indicator)
            withAnimation(.easeInOut(duration: 1.5).repeatForever(autoreverses: true)) {
                dotOpacity = 0.3
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
}

// =============================================================================
// SESSION INDICATOR CONTROLLER
// =============================================================================

/// Manages the floating session indicator window.
/// Follows the same pattern as RecordingIndicatorController but simplified.
class SessionIndicatorController: NSObject, ObservableObject, NSWindowDelegate {
    private var window: NSWindow?
    private var durationTimer: Timer?

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

        duration = 0
        isDisappearing = false
        createWindow()
        updatePosition()
        window?.orderFront(nil)
        isVisible = true

        // Start timer to update duration every second
        durationTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            guard let self else { return }
            DispatchQueue.main.async {
                self.duration += 1
            }
        }
        if let timer = durationTimer {
            RunLoop.main.add(timer, forMode: .common)
        }

        window?.delegate = self
    }

    func showProcessing() {
        durationTimer?.invalidate()
        durationTimer = nil
        isProcessing = true
    }

    func hide() {
        durationTimer?.invalidate()
        durationTimer = nil
        snapPlaceholders.hide()

        isProcessing = false
        isDisappearing = true

        DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) { [weak self] in
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
        let contentView = SessionIndicatorView(controller: self)
        let hostingView = NSHostingView(rootView: contentView)
        hostingView.frame = CGRect(x: 0, y: 0, width: 140, height: 36)

        hostingView.wantsLayer = true
        hostingView.layer?.backgroundColor = .clear

        let window = NSWindow(
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

        if let cv = window.contentView {
            cv.wantsLayer = true
            cv.layer?.backgroundColor = .clear
            cv.layer?.isOpaque = false
        }

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

    private let windowWidth: CGFloat = 140
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
        let view = Capsule()
            .fill(.ultraThinMaterial)
            .opacity(0.4)
            .overlay(Capsule().stroke(Color.white.opacity(0.2), lineWidth: 1))
            .shadow(color: .black.opacity(0.1), radius: 8, x: 0, y: 2)
            .frame(width: windowWidth, height: windowHeight)

        let hostingView = NSHostingView(rootView: view)
        hostingView.frame = CGRect(x: 0, y: 0, width: windowWidth, height: windowHeight)

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
        let view = Capsule()
            .fill(.ultraThinMaterial)
            .opacity(highlighted ? 0.9 : 0.4)
            .overlay(Capsule().stroke(Color.white.opacity(highlighted ? 0.5 : 0.2), lineWidth: 1))
            .shadow(color: .black.opacity(0.1), radius: 8, x: 0, y: 2)
            .frame(width: windowWidth, height: windowHeight)
            .scaleEffect(highlighted ? 1.05 : 1.0)

        let hostingView = NSHostingView(rootView: view)
        hostingView.frame = window.contentView?.frame ?? CGRect(x: 0, y: 0, width: windowWidth, height: windowHeight)
        window.contentView = hostingView
    }
}
