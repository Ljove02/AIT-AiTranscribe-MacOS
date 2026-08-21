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
import AVFoundation
import Combine
import QuartzCore

enum RecordingIndicatorStyle: String, CaseIterable, Identifiable {
    case classic
    case glow
    case aurora
    case prism
    case graphite
    case ember

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .classic:
            return "Classic"
        case .glow:
            return "Glow"
        case .aurora:
            return "Aurora"
        case .prism:
            return "Prism"
        case .graphite:
            return "Graphite"
        case .ember:
            return "Ember"
        }
    }

    var description: String {
        switch self {
        case .classic:
            return "Side pulse"
        case .glow:
            return "Blue liquid"
        case .aurora:
            return "Green violet"
        case .prism:
            return "Rose gold"
        case .graphite:
            return "Silver smoke"
        case .ember:
            return "Warm violet"
        }
    }

    var isLiquid: Bool {
        self != .classic
    }

    var glowPalette: FluidGlowPalette {
        switch self {
        case .classic, .glow:
            return .glow
        case .aurora:
            return .aurora
        case .prism:
            return .prism
        case .graphite:
            return .graphite
        case .ember:
            return .ember
        }
    }
}

struct FluidGlowPalette {
    let baseStart: Color
    let baseMid: Color
    let baseEnd: Color
    let lowerStart: Color
    let lowerMid: Color
    let lowerEnd: Color
    let upperStart: Color
    let upperMid: Color
    let upperEnd: Color
    let ribbonA: Color
    let ribbonB: Color
    let separator: Color
    let grain: Color

    static let glow = FluidGlowPalette(
        baseStart: Color(red: 0.01, green: 0.04, blue: 0.14),
        baseMid: Color(red: 0.01, green: 0.32, blue: 0.72),
        baseEnd: Color(red: 0.60, green: 0.94, blue: 1.00),
        lowerStart: Color(red: 0.00, green: 0.10, blue: 0.58),
        lowerMid: Color(red: 0.00, green: 0.42, blue: 1.00),
        lowerEnd: Color(red: 0.07, green: 0.82, blue: 0.98),
        upperStart: Color(red: 0.84, green: 0.99, blue: 1.00),
        upperMid: Color(red: 0.96, green: 1.00, blue: 0.93),
        upperEnd: Color(red: 0.10, green: 0.60, blue: 0.70),
        ribbonA: Color(red: 0.70, green: 0.95, blue: 1.00),
        ribbonB: Color(red: 0.95, green: 1.00, blue: 0.88),
        separator: Color(red: 0.96, green: 1.00, blue: 0.98),
        grain: Color(red: 0.78, green: 0.97, blue: 1.00)
    )

    static let aurora = FluidGlowPalette(
        baseStart: Color(red: 0.00, green: 0.03, blue: 0.10),
        baseMid: Color(red: 0.00, green: 0.66, blue: 0.46),
        baseEnd: Color(red: 0.70, green: 0.30, blue: 1.00),
        lowerStart: Color(red: 0.00, green: 0.08, blue: 0.28),
        lowerMid: Color(red: 0.00, green: 0.88, blue: 0.56),
        lowerEnd: Color(red: 0.34, green: 0.18, blue: 0.95),
        upperStart: Color(red: 0.72, green: 1.00, blue: 0.78),
        upperMid: Color(red: 0.42, green: 0.94, blue: 1.00),
        upperEnd: Color(red: 0.30, green: 0.04, blue: 0.78),
        ribbonA: Color(red: 0.42, green: 1.00, blue: 0.66),
        ribbonB: Color(red: 0.74, green: 0.42, blue: 1.00),
        separator: Color(red: 0.78, green: 1.00, blue: 0.82),
        grain: Color(red: 0.62, green: 1.00, blue: 0.80)
    )

    static let prism = FluidGlowPalette(
        baseStart: Color(red: 0.13, green: 0.00, blue: 0.08),
        baseMid: Color(red: 0.94, green: 0.05, blue: 0.34),
        baseEnd: Color(red: 1.00, green: 0.78, blue: 0.28),
        lowerStart: Color(red: 0.34, green: 0.00, blue: 0.18),
        lowerMid: Color(red: 1.00, green: 0.12, blue: 0.40),
        lowerEnd: Color(red: 1.00, green: 0.67, blue: 0.18),
        upperStart: Color(red: 1.00, green: 0.94, blue: 0.70),
        upperMid: Color(red: 1.00, green: 0.50, blue: 0.78),
        upperEnd: Color(red: 0.46, green: 0.03, blue: 0.54),
        ribbonA: Color(red: 1.00, green: 0.54, blue: 0.78),
        ribbonB: Color(red: 1.00, green: 0.83, blue: 0.34),
        separator: Color(red: 1.00, green: 0.88, blue: 0.58),
        grain: Color(red: 1.00, green: 0.74, blue: 0.62)
    )

    static let graphite = FluidGlowPalette(
        baseStart: Color(red: 0.00, green: 0.00, blue: 0.01),
        baseMid: Color(red: 0.24, green: 0.28, blue: 0.32),
        baseEnd: Color(red: 0.88, green: 0.94, blue: 0.98),
        lowerStart: Color(red: 0.02, green: 0.03, blue: 0.045),
        lowerMid: Color(red: 0.30, green: 0.38, blue: 0.46),
        lowerEnd: Color(red: 0.86, green: 0.94, blue: 1.00),
        upperStart: Color(red: 0.98, green: 1.00, blue: 1.00),
        upperMid: Color(red: 0.62, green: 0.72, blue: 0.80),
        upperEnd: Color(red: 0.08, green: 0.10, blue: 0.13),
        ribbonA: Color(red: 0.78, green: 0.90, blue: 1.00),
        ribbonB: Color(red: 0.34, green: 0.42, blue: 0.50),
        separator: Color(red: 0.74, green: 0.90, blue: 1.00),
        grain: Color(red: 0.86, green: 0.95, blue: 1.00)
    )

    static let ember = FluidGlowPalette(
        baseStart: Color(red: 0.11, green: 0.00, blue: 0.12),
        baseMid: Color(red: 0.48, green: 0.08, blue: 0.92),
        baseEnd: Color(red: 1.00, green: 0.38, blue: 0.08),
        lowerStart: Color(red: 0.20, green: 0.00, blue: 0.34),
        lowerMid: Color(red: 0.66, green: 0.08, blue: 1.00),
        lowerEnd: Color(red: 1.00, green: 0.24, blue: 0.05),
        upperStart: Color(red: 1.00, green: 0.76, blue: 0.52),
        upperMid: Color(red: 0.95, green: 0.30, blue: 1.00),
        upperEnd: Color(red: 0.24, green: 0.00, blue: 0.42),
        ribbonA: Color(red: 1.00, green: 0.42, blue: 0.92),
        ribbonB: Color(red: 1.00, green: 0.62, blue: 0.24),
        separator: Color(red: 1.00, green: 0.68, blue: 0.42),
        grain: Color(red: 1.00, green: 0.58, blue: 0.82)
    )
}

struct RecordingIndicatorMiniVisual: View {
    let style: RecordingIndicatorStyle
    let level: CGFloat
    let now: Date
    let animationStartTime: Date
    let fluidPhase: Double
    let waveProgresses: [CGFloat]
    let showsTranscriptionDots: Bool
    let showsOnlyTranscriptionDots: Bool

    @Environment(\.colorScheme) private var colorScheme

    private let coreSize: CGFloat = 22

    var body: some View {
        ZStack {
            if showsOnlyTranscriptionDots {
                transcriptionDots
            } else {
                switch style {
                case .classic:
                    if !showsTranscriptionDots {
                        classicWaveArcs
                    }

                    coreCircle

                    if showsTranscriptionDots {
                        transcriptionDots
                    }
                case .glow, .aurora, .prism, .graphite, .ember:
                    liquidGradientField

                    if showsTranscriptionDots {
                        transcriptionDots
                    }
                }
            }
        }
        .frame(width: 64, height: 36)
        .accessibilityHidden(true)
    }

    private var normalizedLevel: CGFloat {
        min(max(level, 0), 1)
    }

    private var elapsed: TimeInterval {
        now.timeIntervalSince(animationStartTime)
    }

    @ViewBuilder
    private var classicWaveArcs: some View {
        ForEach(Array(waveProgresses.enumerated()), id: \.offset) { _, progress in
            let clampedProgress = min(max(progress, 0), 1)
            let opacity = classicWaveOpacity(for: clampedProgress)
            let waveOffset = 5 + clampedProgress * 11
            let arcColor = colorScheme == .dark
                ? Color.white.opacity(opacity)
                : Color.black.opacity(opacity * 0.6)
            let arcSize = classicWaveSize(for: clampedProgress)

            SemiCircleArc()
                .stroke(arcColor, style: StrokeStyle(lineWidth: 0.8, lineCap: .round))
                .frame(width: arcSize, height: arcSize)
                .offset(x: waveOffset)

            SemiCircleArc()
                .stroke(arcColor, style: StrokeStyle(lineWidth: 0.8, lineCap: .round))
                .frame(width: arcSize, height: arcSize)
                .scaleEffect(x: -1)
                .offset(x: -waveOffset)
        }
    }

    private func classicWaveOpacity(for progress: CGFloat) -> Double {
        if progress < 0.25 {
            return Double(progress / 0.25) * 0.35
        }

        let fadeProgress = (progress - 0.25) / 0.75
        return Double(max(0, 0.35 * (1 - fadeProgress)))
    }

    private func classicWaveSize(for progress: CGFloat) -> CGFloat {
        coreSize * (0.8 + progress * 0.6)
    }

    private var liquidGradientField: some View {
        let activeLevel = reactiveLevel
        let voiceEase = Double(activeLevel * activeLevel * (3 - 2 * activeLevel))
        let phaseA = fluidPhase * 1.42 + 0.12 * sin(elapsed * 0.18) + voiceEase * 0.04 * sin(elapsed * 0.72)
        let phaseB = -fluidPhase * 1.08 + 1.20 + 0.10 * cos(elapsed * 0.15) + voiceEase * 0.03 * sin(elapsed * 0.58)
        let phaseC = fluidPhase * 1.74 + 2.10 + 0.14 * sin(elapsed * 0.14 + 0.90) + voiceEase * 0.04 * sin(elapsed * 0.66)

        return FluidGlowTexture(
            palette: style.glowPalette,
            level: activeLevel,
            phaseA: phaseA,
            phaseB: phaseB,
            phaseC: phaseC
        )
        .frame(width: 64, height: 36)
        .scaleEffect(1 + activeLevel * 0.022)
        .animation(.easeOut(duration: 0.42), value: activeLevel)
    }

    private var reactiveLevel: CGFloat {
        CGFloat(pow(Double(normalizedLevel), 0.72))
    }

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

    @ViewBuilder
    private var transcriptionDots: some View {
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
                    .shadow(color: .blue.opacity(style == .glow ? 0.35 : 0), radius: 3)
                    .offset(
                        x: CoreGraphics.cos(angle) * currentRadius,
                        y: CoreGraphics.sin(angle) * currentRadius
                    )
            }
        }
        .frame(width: coreSize, height: coreSize)
        .rotationEffect(.degrees(Double(phase) * 180), anchor: .center)
    }
}

private struct FluidGlowTexture: View {
    let palette: FluidGlowPalette
    let level: CGFloat
    let phaseA: Double
    let phaseB: Double
    let phaseC: Double

    var body: some View {
        Canvas { context, size in
            let rect = CGRect(origin: .zero, size: size)
            let capsulePath = Path(roundedRect: rect, cornerRadius: size.height / 2)
            let voice = Double(min(max(level, 0), 1))

            context.clip(to: capsulePath)

            var rectPath = Path()
            rectPath.addRect(rect)
            context.fill(
                rectPath,
                with: .linearGradient(
                    Gradient(stops: [
                        .init(color: palette.baseStart.opacity(0.94), location: 0.00),
                        .init(color: palette.baseMid.opacity(0.70 + voice * 0.22), location: 0.46),
                        .init(color: palette.baseEnd.opacity(0.50 + voice * 0.28), location: 1.00)
                    ]),
                    startPoint: CGPoint(x: rect.minX, y: rect.maxY),
                    endPoint: CGPoint(x: rect.maxX, y: rect.minY)
                )
            )

            context.drawLayer { layer in
                layer.addFilter(.blur(radius: 4.4))
                layer.fill(
                    lowerLiquidPath(size: size),
                    with: .linearGradient(
                        Gradient(stops: [
                            .init(color: palette.lowerStart.opacity(0.82 + voice * 0.14), location: 0.00),
                            .init(color: palette.lowerMid.opacity(0.72 + voice * 0.26), location: 0.52),
                            .init(color: palette.lowerEnd.opacity(0.30 + voice * 0.28), location: 1.00)
                        ]),
                        startPoint: CGPoint(x: 0, y: size.height),
                        endPoint: CGPoint(x: size.width, y: size.height * 0.18)
                    )
                )
            }

            context.drawLayer { layer in
                layer.addFilter(.blur(radius: 4.4))
                layer.fill(
                    upperLiquidPath(size: size),
                    with: .linearGradient(
                        Gradient(stops: [
                            .init(color: palette.upperStart.opacity(0.56 + voice * 0.26), location: 0.00),
                            .init(color: palette.upperMid.opacity(0.28 + voice * 0.24), location: 0.42),
                            .init(color: palette.upperEnd.opacity(0.28 + voice * 0.24), location: 1.00)
                        ]),
                        startPoint: CGPoint(x: size.width * 0.28, y: 0),
                        endPoint: CGPoint(x: size.width, y: size.height)
                    )
                )
            }

            context.drawLayer { layer in
                layer.addFilter(.blur(radius: 7.4))
                layer.fill(
                    boundaryRibbonPath(size: size, verticalOffset: 0, amplitudeScale: 0.84 + level * 0.28, thickness: 8.4 + level * 5.2),
                    with: .linearGradient(
                        Gradient(stops: [
                            .init(color: Color.clear, location: 0.00),
                            .init(color: palette.ribbonA.opacity(0.12 + voice * 0.18), location: 0.18),
                            .init(color: palette.separator.opacity(0.18 + voice * 0.26), location: 0.48),
                            .init(color: palette.ribbonB.opacity(0.10 + voice * 0.18), location: 0.72),
                            .init(color: Color.clear, location: 1.00)
                        ]),
                        startPoint: CGPoint(x: 0, y: size.height * 0.46),
                        endPoint: CGPoint(x: size.width, y: size.height * 0.62)
                    )
                )
            }

            for index in 0..<3 {
                let offset = CGFloat(index - 1) * 5.8
                let amplitude = 0.58 + CGFloat(index) * 0.18

                context.drawLayer { layer in
                    layer.addFilter(.blur(radius: 8.0 + CGFloat(index) * 0.8))
                    layer.fill(
                        boundaryRibbonPath(
                            size: size,
                            verticalOffset: offset,
                            amplitudeScale: amplitude + level * 0.18,
                            thickness: 4.2 + level * 3.4
                        ),
                        with: .linearGradient(
                            Gradient(stops: [
                                .init(color: Color.clear, location: 0.00),
                                .init(color: palette.ribbonA.opacity(0.07 + voice * 0.12), location: 0.30),
                                .init(color: palette.separator.opacity(0.045 + voice * 0.12), location: 0.56),
                                .init(color: Color.clear, location: 1.00)
                            ]),
                            startPoint: CGPoint(x: size.width * 0.08, y: size.height * 0.22),
                            endPoint: CGPoint(x: size.width * 0.92, y: size.height * 0.80)
                        )
                    )
                }
            }

            context.drawLayer { layer in
                layer.addFilter(.blur(radius: 3.4))

                for index in 0..<9 {
                    let seed = Double(index)
                    let x = size.width * (0.08 + 0.84 * CGFloat((sin(seed * 2.31 + phaseC * 0.72) + 1) / 2))
                    let y = size.height * (0.18 + 0.62 * CGFloat((cos(seed * 1.73 + phaseB * 0.58) + 1) / 2))
                    let grainRect = CGRect(
                        x: x - 8 - CGFloat(index % 3) * 1.5,
                        y: y - 4,
                        width: 12 + CGFloat(index % 4) * 4,
                        height: 7 + CGFloat(index % 3) * 3
                    )

                    layer.fill(
                        Path(ellipseIn: grainRect),
                        with: .radialGradient(
                            Gradient(colors: [
                                Color.white.opacity(0.016 + voice * 0.038),
                                palette.grain.opacity(0.010 + voice * 0.026),
                                Color.clear
                            ]),
                            center: CGPoint(x: grainRect.midX, y: grainRect.midY),
                            startRadius: 0,
                            endRadius: max(grainRect.width, grainRect.height) * 0.58
                        )
                    )
                }
            }

            context.stroke(
                capsulePath,
                with: .linearGradient(
                    Gradient(colors: [
                        Color.white.opacity(0.20),
                        Color.white.opacity(0.035),
                        Color.black.opacity(0.20)
                    ]),
                    startPoint: CGPoint(x: rect.minX, y: rect.minY),
                    endPoint: CGPoint(x: rect.maxX, y: rect.maxY)
                ),
                lineWidth: 0.55
            )
        }
        .compositingGroup()
    }

    private func lowerLiquidPath(size: CGSize) -> Path {
        let padding: CGFloat = 18
        let points = wavePoints(
            size: size,
            baseY: size.height * (0.48 + 0.07 * CGFloat(sin(phaseA * 0.62))),
            amplitude: size.height * (0.10 + level * 0.045),
            phase: phaseA,
            tilt: size.height * (0.08 + level * 0.035)
        )

        var path = Path()
        path.move(to: CGPoint(x: -padding, y: size.height + padding))
        path.addLine(to: points[0])
        appendSmoothWave(to: &path, points: points)
        path.addLine(to: CGPoint(x: size.width + padding, y: size.height + padding))
        path.closeSubpath()
        return path
    }

    private func upperLiquidPath(size: CGSize) -> Path {
        let padding: CGFloat = 18
        let points = wavePoints(
            size: size,
            baseY: size.height * (0.50 + 0.06 * CGFloat(cos(phaseB * 0.58))),
            amplitude: size.height * (0.09 + level * 0.04),
            phase: phaseB + 1.4,
            tilt: -size.height * (0.08 + level * 0.035)
        )

        var path = Path()
        path.move(to: CGPoint(x: -padding, y: -padding))
        path.addLine(to: points[0])
        appendSmoothWave(to: &path, points: points)
        path.addLine(to: CGPoint(x: size.width + padding, y: -padding))
        path.closeSubpath()
        return path
    }

    private func boundaryRibbonPath(
        size: CGSize,
        verticalOffset: CGFloat,
        amplitudeScale: CGFloat,
        thickness: CGFloat
    ) -> Path {
        let centerPoints = wavePoints(
            size: size,
            baseY: size.height * (0.50 + 0.05 * CGFloat(sin(phaseC * 0.56))) + verticalOffset,
            amplitude: size.height * 0.12 * amplitudeScale,
            phase: phaseC + Double(verticalOffset) * 0.08,
            tilt: size.height * 0.02 * CGFloat(cos(phaseA * 0.44))
        )

        let topPoints = centerPoints.enumerated().map { index, point in
            CGPoint(
                x: point.x,
                y: point.y - thickness - CGFloat(sin(phaseB + Double(index) * 0.72)) * 1.8
            )
        }

        let bottomPoints = Array(centerPoints.enumerated().map { index, point in
            CGPoint(
                x: point.x,
                y: point.y + thickness + CGFloat(cos(phaseA + Double(index) * 0.61)) * 1.8
            )
        }.reversed())

        var path = Path()
        path.move(to: topPoints[0])
        appendSmoothWave(to: &path, points: topPoints)
        path.addLine(to: bottomPoints[0])
        appendSmoothWave(to: &path, points: bottomPoints)
        path.closeSubpath()
        return path
    }

    private func wavePoints(
        size: CGSize,
        baseY: CGFloat,
        amplitude: CGFloat,
        phase: Double,
        tilt: CGFloat
    ) -> [CGPoint] {
        let padding: CGFloat = 18
        let count = 8

        return (0...count).map { index in
            let fraction = CGFloat(index) / CGFloat(count)
            let x = -padding + (size.width + padding * 2) * fraction
            let primary = CGFloat(sin(Double(fraction) * .pi * 2.0 + phase)) * amplitude
            let secondary = CGFloat(sin(Double(fraction) * .pi * 3.4 - phase * 0.72)) * amplitude * 0.24
            let drift = (fraction - 0.5) * tilt

            return CGPoint(x: x, y: baseY + primary + secondary + drift)
        }
    }

    private func appendSmoothWave(to path: inout Path, points: [CGPoint]) {
        guard points.count > 1 else { return }

        for index in 1..<points.count {
            let previous = points[index - 1]
            let current = points[index]
            let midpoint = CGPoint(
                x: (previous.x + current.x) / 2,
                y: (previous.y + current.y) / 2
            )

            path.addQuadCurve(to: midpoint, control: previous)
        }

        path.addLine(to: points[points.count - 1])
    }
}

// =============================================================================
// RECORDING INDICATOR VIEW
// =============================================================================

struct RecordingIndicatorHostView: View {
    @ObservedObject var controller: RecordingIndicatorController

    var body: some View {
        RecordingIndicatorView(controller: controller)
    }
}

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

    /// Continuous liquid phase for Glow. Speed changes with voice, position does not jump.
    @State private var fluidPhase: Double = 0
    @State private var fluidSpeed: Double = 0.72
    @State private var lastFluidPhaseDate: Date?

    /// Detect light/dark mode
    @Environment(\.colorScheme) private var colorScheme

    private var isPolishMorphing: Bool {
        controller.polishResponseModel != nil
    }

    private var isPolishExpandingToReader: Bool {
        controller.polishResponsePresentationPhase == .expanding
    }

    private var isPolishReader: Bool {
        controller.polishResponsePresentationPhase == .reader
    }

    private var polishModel: PolishFloatingResponseModel? {
        controller.polishResponseModel
    }

    private var isPolishWaitingForOutput: Bool {
        guard let polishModel else { return false }
        return polishModel.fullText.isEmpty
    }

    private var isPolishLoaderOnly: Bool {
        isPolishMorphing && (!isPolishReader || isPolishWaitingForOutput)
    }

    private var indicatorStyle: RecordingIndicatorStyle {
        controller.indicatorStyle
    }

    private var polishScreenshotImage: NSImage? {
        guard let screenshotData = polishModel?.screenshotData else { return nil }
        return NSImage(data: screenshotData)
    }

    private var glassCornerRadius: CGFloat {
        guard isPolishMorphing else { return 999 }
        return (isPolishExpandingToReader || isPolishReader) ? 28 : 999
    }

    /// Wave arc parameters
    private let waveLifetime: Double = 1.3     // faster expansion
    private let volumeThreshold: CGFloat = 0.10

    var body: some View {
        TimelineView(.animation(minimumInterval: nil, paused: false)) { timeline in
            let now = timeline.date

            ZStack(alignment: isPolishMorphing ? .top : .center) {
                if isPolishReader {
                    polishReaderContent
                        .transition(.opacity)
                }

                if !isPolishReader || isPolishWaitingForOutput {
                    indicatorCore(now: now)
                        .frame(width: 64, height: 36)
                        .padding(.top, isPolishMorphing ? 5 : 0)
                }
            }
            .frame(width: isPolishMorphing ? nil : 64, height: isPolishMorphing ? nil : 36)
            .frame(
                maxWidth: isPolishMorphing ? .infinity : nil,
                maxHeight: isPolishMorphing ? .infinity : nil,
                alignment: isPolishMorphing ? .top : .center
            )
            .background(glassBackground)
            .overlay(glassStroke)
            .clipShape(RoundedRectangle(cornerRadius: glassCornerRadius, style: .continuous))
            .scaleEffect(appearScale)
            .opacity(appearOpacity)
            .onChange(of: now) { _, updatedNow in
                updateFluidPhase(now: updatedNow)
            }
        }
        .onAppear {
            let now = Date()
            animationStartTime = now
            lastFluidPhaseDate = now
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
            lastFluidPhaseDate = nil
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
                lastFluidPhaseDate = nil
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

    @ViewBuilder
    private func indicatorCore(now: Date) -> some View {
        RecordingIndicatorMiniVisual(
            style: indicatorStyle,
            level: isPolishMorphing ? 0 : smoothedVolume,
            now: now,
            animationStartTime: animationStartTime,
            fluidPhase: fluidPhase,
            waveProgresses: isPolishMorphing ? [] : waveProgresses(now: now),
            showsTranscriptionDots: showTranscriptionDots || isPolishLoaderOnly,
            showsOnlyTranscriptionDots: isPolishLoaderOnly
        )
    }

    private var polishReaderContent: some View {
        VStack(alignment: .leading, spacing: 16) {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    if let polishScreenshotImage {
                        Image(nsImage: polishScreenshotImage)
                            .resizable()
                            .scaledToFit()
                            .frame(maxWidth: 560, maxHeight: 300)
                            .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                            .shadow(color: .black.opacity(0.18), radius: 18, y: 8)
                            .transition(.opacity.combined(with: .scale(scale: 0.985, anchor: .top)))
                    }

                    if let polishModel, !polishModel.visibleText.isEmpty {
                        PolishMarkdownView(text: polishModel.visibleText)
                            .textSelection(.enabled)
                            .frame(maxWidth: 560, alignment: .topLeading)
                            .transition(.opacity)
                    } else {
                        VStack(alignment: .leading, spacing: 10) {
                            RoundedRectangle(cornerRadius: 3, style: .continuous)
                                .fill(.white.opacity(0.12))
                                .frame(width: 430, height: 8)
                            RoundedRectangle(cornerRadius: 3, style: .continuous)
                                .fill(.white.opacity(0.08))
                                .frame(width: 360, height: 8)
                        }
                        .frame(maxWidth: 560, alignment: .topLeading)
                        .padding(.top, polishScreenshotImage == nil ? 90 : 0)
                    }
                }
                .frame(maxWidth: .infinity)
                .padding(.horizontal, 28)
                .padding(.top, 58)
                .padding(.bottom, 24)
            }
            .scrollIndicators(.hidden)
            .clipShape(RoundedRectangle(cornerRadius: 22, style: .continuous))

            HStack {
                Text(polishModel?.isComplete == true ? "\(polishModel?.fullText.split(separator: " ").count ?? 0) words" : "Streaming response...")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(.tertiary)

                Spacer()

                Button(action: {
                    controller.closePolishResponse()
                }) {
                    Text("Close")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 7)
                        .background(Color.primary.opacity(0.055), in: Capsule())
                }
                .buttonStyle(.plain)

                Button(action: {
                    controller.copyPolishResponseAndClose()
                }) {
                    Label("Copy", systemImage: "doc.on.doc")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 13)
                        .padding(.vertical, 7)
                        .background(Color.blue, in: Capsule())
                }
                .buttonStyle(.plain)
                .disabled(polishModel?.fullText.isEmpty ?? true)
                .opacity((polishModel?.fullText.isEmpty ?? true) ? 0.45 : 1)
            }
            .padding(.horizontal, 20)
            .padding(.bottom, 18)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Glass Background

    private var glassBackground: some View {
        let shape = RoundedRectangle(cornerRadius: glassCornerRadius, style: .continuous)

        return shape
            .fill(.ultraThinMaterial)
            .overlay(
                shape
                    .fill(colorScheme == .dark ? Color.white.opacity(0.05) : Color.white.opacity(0.18))
                    .blendMode(.screen)
            )
            .overlay(
                shape
                    .fill(colorScheme == .dark ? Color.black.opacity(0.08) : Color.black.opacity(0.015))
                    .blendMode(.multiply)
            )
    }

    private var glassStroke: some View {
        RoundedRectangle(cornerRadius: glassCornerRadius, style: .continuous)
            .strokeBorder(colorScheme == .dark ? Color.white.opacity(0.14) : Color.white.opacity(0.52), lineWidth: 0.55)
    }

    // MARK: - Wave Progress

    private func waveProgresses(now: Date) -> [CGFloat] {
        waveSpawnTimes
            .map { CGFloat(now.timeIntervalSince($0) / waveLifetime) }
            .filter { $0 >= 0 && $0 < 1 }
    }

    // MARK: - Volume & Wave Spawning

    private func updateSmoothedVolume(_ volume: Double) {
        let target = min(max(volume, 0), 1)

        let newSmoothed: CGFloat
        if CGFloat(target) > smoothedVolume {
            newSmoothed = smoothedVolume * 0.35 + CGFloat(target) * 0.65
        } else {
            newSmoothed = smoothedVolume * 0.78 + CGFloat(target) * 0.22
        }
        smoothedVolume = newSmoothed
    }

    private func updateFluidPhase(now: Date) {
        guard indicatorStyle.isLiquid, !showTranscriptionDots, !isPolishLoaderOnly else {
            lastFluidPhaseDate = now
            return
        }

        guard let lastFluidPhaseDate else {
            self.lastFluidPhaseDate = now
            return
        }

        let delta = min(max(now.timeIntervalSince(lastFluidPhaseDate), 0), 0.066)
        self.lastFluidPhaseDate = now
        guard delta > 0 else { return }

        let voice = Double(min(max(smoothedVolume, 0), 1))
        let voiceEase = pow(voice, 0.62)
        let targetSpeed = 0.72 + voiceEase * 2.35
        let responseRate = targetSpeed > fluidSpeed ? 5.6 : 2.4
        let response = 1 - exp(-delta * responseRate)

        fluidSpeed += (targetSpeed - fluidSpeed) * response
        fluidPhase += fluidSpeed * delta

        if fluidPhase > 10_000 {
            fluidPhase.formTruncatingRemainder(dividingBy: 10_000)
        }
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

private final class RecordingIndicatorLevelMonitor: NSObject, ObservableObject, AVCaptureAudioDataOutputSampleBufferDelegate {
    private enum MonitorError: Error {
        case noDevice
        case cannotAddInput
        case cannotAddOutput
    }

    private static let calibratedNoiseFloorKey = "recordingIndicatorNoiseFloorRMS"

    @Published private(set) var level: Double = 0

    private let session = AVCaptureSession()
    private let queue = DispatchQueue(label: "com.aitranscribe.recording-indicator.microphone-level")
    private var isConfigured = false
    /// The device the capture session was built against. If the default input
    /// changes, the session has to be rebuilt rather than restarted.
    private var configuredDeviceUID: String?
    /// When audio last actually arrived, used to notice a dead session.
    private var lastBufferAt: TimeInterval = 0
    private var recoveryAttempts = 0
    private var lastRawRMS: Double = 0
    private var lastAdjustedRMS: Double = 0
    private var smoothedLevel: Double = 0
    private var lastPublishedAt: TimeInterval = 0
    private var noiseFloor: Double = 0.002
    private var hasNoiseFloor = false
    private var usesCalibratedNoiseFloor = false
    private var calibrationSamples: [Double]?

    override init() {
        super.init()
        applySavedNoiseFloor()
    }

    // MARK: - Diagnostics
    //
    // print output is lost for a normally launched app, and this only
    // misbehaves after a specific sequence involving another app holding the
    // microphone, so the numbers have to survive to disk to be readable.

    private static let logURL: URL? = {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first?
            .appendingPathComponent("AiTranscribe/indicator_level.log")
    }()
    private var lastLoggedAt: TimeInterval = 0

    private func log(_ line: String) {
        guard let url = Self.logURL else { return }
        let stamped = ISO8601DateFormatter().string(from: Date()) + "  " + line + "\n"
        guard let data = stamped.data(using: .utf8) else { return }
        if let handle = try? FileHandle(forWritingTo: url) {
            defer { try? handle.close() }
            // Keep it small: this is a rolling diagnostic, not a record.
            if (try? handle.seekToEnd()) ?? 0 > 200_000 {
                try? handle.truncate(atOffset: 0)
            }
            try? handle.write(contentsOf: data)
        } else {
            try? FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try? stamped.write(to: url, atomically: true, encoding: .utf8)
        }
    }

    func start() {
        queue.async { [weak self] in
            guard let self else { return }
            self.recoveryAttempts = 0
            self.startOnQueue()
        }
    }

    private func startOnQueue() {
        // The default input can change while the app is running, and the
        // handle can go stale after another process has had the device.
        let currentUID = AVCaptureDevice.default(for: .audio)?.uniqueID
        if isConfigured, currentUID != configuredDeviceUID {
            teardownOnQueue()
        }

        do {
            try configureIfNeeded()
        } catch {
            publish(level: 0)
            return
        }

        if !session.isRunning {
            session.startRunning()
        }

        log("start: running=\(session.isRunning) device=\(AVCaptureDevice.default(for: .audio)?.localizedName ?? "none") floor=\(noiseFloor) calibrated=\(usesCalibratedNoiseFloor)")
        publish(level: smoothedLevel)
        scheduleStalenessCheck()
    }

    /// A running session that delivers nothing is indistinguishable from
    /// silence, and no amount of restarting it helps because the input itself
    /// is stale. Rebuild it instead, which is the only thing that recovers a
    /// device another process has disturbed.
    private func scheduleStalenessCheck() {
        let startedAt = Date().timeIntervalSince1970
        queue.asyncAfter(deadline: .now() + 1.5) { [weak self] in
            guard let self, self.session.isRunning else { return }
            guard self.lastBufferAt < startedAt else { return }
            guard self.recoveryAttempts < 2 else {
                print("RecordingIndicatorLevelMonitor: no audio after rebuild; giving up")
                return
            }
            self.recoveryAttempts += 1
            self.log("no audio 1.5s after start — rebuilding capture session")
            self.teardownOnQueue()
            self.startOnQueue()
        }
    }

    private func teardownOnQueue() {
        if session.isRunning {
            session.stopRunning()
        }
        session.beginConfiguration()
        for input in session.inputs { session.removeInput(input) }
        for output in session.outputs { session.removeOutput(output) }
        session.commitConfiguration()
        isConfigured = false
        configuredDeviceUID = nil
    }

    func stop() {
        queue.async { [weak self] in
            guard let self else { return }

            if self.session.isRunning {
                self.session.stopRunning()
            }

            self.smoothedLevel = 0
            self.applySavedNoiseFloor()
            self.publish(level: 0)
        }
    }

    func calibrateNoiseFloor(duration: TimeInterval = 2.4, completion: @escaping (Double?) -> Void) {
        queue.async { [weak self] in
            guard let self else { return }

            do {
                try self.configureIfNeeded()
            } catch {
                DispatchQueue.main.async {
                    completion(nil)
                }
                return
            }

            let wasRunning = self.session.isRunning
            if !wasRunning {
                self.session.startRunning()
            }

            self.calibrationSamples = []

            self.queue.asyncAfter(deadline: .now() + duration) { [weak self] in
                guard let self else { return }

                let samples = self.calibrationSamples ?? []
                self.calibrationSamples = nil

                guard !samples.isEmpty else {
                    if !wasRunning {
                        self.session.stopRunning()
                    }
                    DispatchQueue.main.async {
                        completion(nil)
                    }
                    return
                }

                let sortedSamples = samples.sorted()
                let quietSampleCount = max(1, Int(Double(sortedSamples.count) * 0.80))
                let quietSamples = sortedSamples.prefix(quietSampleCount)
                let averageQuietRMS = quietSamples.reduce(0, +) / Double(quietSamples.count)
                let calibratedFloor = min(max(averageQuietRMS * 1.35, 0.0008), 0.08)

                self.noiseFloor = calibratedFloor
                self.hasNoiseFloor = true
                self.usesCalibratedNoiseFloor = true
                self.smoothedLevel = 0
                UserDefaults.standard.set(calibratedFloor, forKey: Self.calibratedNoiseFloorKey)

                if !wasRunning {
                    self.session.stopRunning()
                }

                DispatchQueue.main.async {
                    completion(calibratedFloor)
                }
            }
        }
    }

    func resetNoiseCalibration() {
        queue.async { [weak self] in
            guard let self else { return }
            UserDefaults.standard.removeObject(forKey: Self.calibratedNoiseFloorKey)
            self.applySavedNoiseFloor()
        }
    }

    private func configureIfNeeded() throws {
        guard !isConfigured else { return }

        session.beginConfiguration()
        session.sessionPreset = .low
        defer { session.commitConfiguration() }

        guard let device = AVCaptureDevice.default(for: .audio) else {
            throw MonitorError.noDevice
        }

        let input = try AVCaptureDeviceInput(device: device)
        guard session.canAddInput(input) else {
            throw MonitorError.cannotAddInput
        }
        session.addInput(input)

        let output = AVCaptureAudioDataOutput()
        output.setSampleBufferDelegate(self, queue: queue)
        guard session.canAddOutput(output) else {
            throw MonitorError.cannotAddOutput
        }
        session.addOutput(output)

        configuredDeviceUID = device.uniqueID
        isConfigured = true
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        lastBufferAt = Date().timeIntervalSince1970
        let measuredLevel = normalizedLevel(from: sampleBuffer)

        if lastBufferAt - lastLoggedAt > 0.5 {
            lastLoggedAt = lastBufferAt
            var formatText = "?"
            if let fd = CMSampleBufferGetFormatDescription(sampleBuffer),
               let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(fd)?.pointee {
                let isFloat = (asbd.mFormatFlags & kAudioFormatFlagIsFloat) != 0
                let interleaved = (asbd.mFormatFlags & kAudioFormatFlagIsNonInterleaved) == 0
                formatText = "\(Int(asbd.mSampleRate))Hz/\(asbd.mChannelsPerFrame)ch/"
                    + "\(isFloat ? "f" : "i")\(asbd.mBitsPerChannel)"
                    + "/\(interleaved ? "inter" : "planar")"
            }
            log(String(format: "fmt=%@ rawRMS=%.6f floor=%.6f adj=%.6f level=%.4f smoothed=%.4f",
                       formatText, lastRawRMS, noiseFloor, lastAdjustedRMS, measuredLevel, smoothedLevel))
        }
        let responseWeight = measuredLevel > smoothedLevel ? 0.36 : 0.20
        smoothedLevel = smoothedLevel * (1 - responseWeight) + measuredLevel * responseWeight

        let now = Date().timeIntervalSince1970
        guard now - lastPublishedAt > 0.05 else { return }
        lastPublishedAt = now
        publish(level: smoothedLevel)
    }

    private func normalizedLevel(from sampleBuffer: CMSampleBuffer) -> Double {
        guard let formatDescription = CMSampleBufferGetFormatDescription(sampleBuffer),
              let streamDescription = CMAudioFormatDescriptionGetStreamBasicDescription(formatDescription),
              let format = AVAudioFormat(streamDescription: streamDescription) else {
            return 0
        }

        let frameCount = AVAudioFrameCount(CMSampleBufferGetNumSamples(sampleBuffer))
        guard frameCount > 0,
              let pcmBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            return 0
        }

        pcmBuffer.frameLength = frameCount
        let status = CMSampleBufferCopyPCMDataIntoAudioBufferList(
            sampleBuffer,
            at: 0,
            frameCount: Int32(frameCount),
            into: pcmBuffer.mutableAudioBufferList
        )

        guard status == noErr else {
            return 0
        }

        let rms = rmsLevel(from: pcmBuffer)
        lastRawRMS = rms
        if calibrationSamples != nil {
            calibrationSamples?.append(rms)
        }

        let adjustedRMS = noiseReducedRMS(from: rms)
        lastAdjustedRMS = adjustedRMS
        guard adjustedRMS > 0.00012 else { return 0 }

        return min(1, pow(adjustedRMS * 23, 0.64))
    }

    private func applySavedNoiseFloor() {
        let savedFloor = UserDefaults.standard.double(forKey: Self.calibratedNoiseFloorKey)
        if savedFloor > 0 {
            noiseFloor = savedFloor
            hasNoiseFloor = true
            usesCalibratedNoiseFloor = true
        } else {
            noiseFloor = 0.002
            hasNoiseFloor = false
            usesCalibratedNoiseFloor = false
        }
    }

    private func noiseReducedRMS(from rms: Double) -> Double {
        if !hasNoiseFloor {
            noiseFloor = max(rms, 0.0008)
            hasNoiseFloor = true
        }

        if !usesCalibratedNoiseFloor {
            let ambientCeiling = max(noiseFloor * 1.75, noiseFloor + 0.0025)
            if rms < ambientCeiling {
                noiseFloor = noiseFloor * 0.94 + rms * 0.06
            } else if rms < noiseFloor {
                noiseFloor = noiseFloor * 0.98 + rms * 0.02
            }
        }

        let subtractionFloor = max(noiseFloor * 1.12, noiseFloor + 0.00055)
        return max(0, rms - subtractionFloor)
    }

    private func rmsLevel(from buffer: AVAudioPCMBuffer) -> Double {
        let frames = Int(buffer.frameLength)
        let channels = Int(buffer.format.channelCount)
        guard frames > 0, channels > 0 else { return 0 }

        if let channelData = buffer.floatChannelData {
            return rmsLevel(channelData: channelData, frames: frames, channels: channels)
        }

        if let channelData = buffer.int16ChannelData {
            var sum: Double = 0
            var count = 0

            for channel in 0..<channels {
                for frame in 0..<frames {
                    let sample = Double(channelData[channel][frame]) / Double(Int16.max)
                    sum += sample * sample
                    count += 1
                }
            }

            return sqrt(sum / Double(max(count, 1)))
        }

        if let channelData = buffer.int32ChannelData {
            var sum: Double = 0
            var count = 0

            for channel in 0..<channels {
                for frame in 0..<frames {
                    let sample = Double(channelData[channel][frame]) / Double(Int32.max)
                    sum += sample * sample
                    count += 1
                }
            }

            return sqrt(sum / Double(max(count, 1)))
        }

        return 0
    }

    private func rmsLevel(channelData: UnsafePointer<UnsafeMutablePointer<Float>>, frames: Int, channels: Int) -> Double {
        var sum: Double = 0
        var count = 0

        for channel in 0..<channels {
            for frame in 0..<frames {
                let sample = Double(channelData[channel][frame])
                sum += sample * sample
                count += 1
            }
        }

        return sqrt(sum / Double(max(count, 1)))
    }

    private func publish(level: Double) {
        DispatchQueue.main.async {
            self.level = level
        }
    }
}

/// Manages the floating recording indicator window
/// Has its own volume polling isolated from AppState to prevent cascading re-renders
class RecordingIndicatorController: NSObject, ObservableObject, NSWindowDelegate {
    enum PolishResponsePresentationPhase {
        case indicator
        case expanding
        case reader
    }

    private var window: NSWindow?
    private var appState: AppState?
    private var volumeTimer: Timer?
    private let levelMonitor = RecordingIndicatorLevelMonitor()
    private var levelCancellable: AnyCancellable?

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

    /// When non-nil, the existing indicator window morphs into this Polish response.
    @Published var polishResponseModel: PolishFloatingResponseModel?
    @Published var polishResponsePresentationPhase: PolishResponsePresentationPhase = .indicator

    /// Visual style for the small floating recording indicator.
    @Published var indicatorStyle: RecordingIndicatorStyle = .classic

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
        if let savedStyle = UserDefaults.standard.string(forKey: "recordingIndicatorStyle"),
           let style = RecordingIndicatorStyle(rawValue: savedStyle) {
            indicatorStyle = style
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
            self?.polishResponseModel = nil
            self?.polishResponsePresentationPhase = .indicator
        }
    }

    func presentPolishResponse(presetName: String, screenshotData: Data?) {
        stopVolumePolling()
        stopDragMonitoring()
        snapPlaceholders.hide()

        if window == nil {
            createWindow()
            updatePosition()
            window?.orderFront(nil)
            isVisible = true
        }

        isDisappearing = false
        isTranscribing = false

        polishResponsePresentationPhase = .indicator

        let responseModel: PolishFloatingResponseModel
        if let existing = polishResponseModel {
            existing.reset(presetName: presetName, screenshotData: screenshotData)
            responseModel = existing
        } else {
            responseModel = PolishFloatingResponseModel(presetName: presetName, screenshotData: screenshotData)
            polishResponseModel = responseModel
        }

        window?.hasShadow = true
        window?.isMovableByWindowBackground = false
        animatePolishResponseShell(screenshotData: screenshotData, model: responseModel)
    }

    func completePolishResponse(text: String, screenshotData: Data?) {
        polishResponseModel?.complete(text: text, screenshotData: screenshotData)
    }

    func appendPolishResponse(delta: String) {
        polishResponseModel?.append(delta: delta)
    }

    func copyPolishResponseAndClose() {
        let text = polishResponseModel?.fullText ?? ""
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
        closePolishResponse()
    }

    func closePolishResponse() {
        polishResponseModel?.cancelReveal()
        stopVolumePolling()
        stopDragMonitoring()
        snapPlaceholders.hide()
        window?.orderOut(nil)
        window = nil
        isVisible = false
        currentVolume = 0.0
        isTranscribing = false
        isDisappearing = false
        polishResponseModel = nil
        polishResponsePresentationPhase = .indicator
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

    func setStyle(_ newStyle: RecordingIndicatorStyle) {
        indicatorStyle = newStyle
        UserDefaults.standard.set(newStyle.rawValue, forKey: "recordingIndicatorStyle")
    }

    func calibrateNoiseFloor(completion: @escaping (Double?) -> Void) {
        levelMonitor.calibrateNoiseFloor(completion: completion)
    }

    func resetNoiseCalibration() {
        levelMonitor.resetNoiseCalibration()
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
        levelCancellable?.cancel()
        levelCancellable = levelMonitor.$level
            .receive(on: RunLoop.main)
            .sink { [weak self] level in
                guard let self, self.isVisible else { return }
                self.currentVolume = level
                self.window?.display()
            }
        levelMonitor.start()
    }

    private func stopVolumePolling() {
        volumeTimer?.invalidate()
        volumeTimer = nil
        levelCancellable?.cancel()
        levelCancellable = nil
        levelMonitor.stop()
        currentVolume = 0
    }

    private func createWindow() {
        // Pass self to the view so it can observe our currentVolume
        let contentView = RecordingIndicatorHostView(controller: self)

        let hostingView = NSHostingView(rootView: contentView)
        hostingView.frame = CGRect(x: 0, y: 0, width: 70, height: 42)
        hostingView.autoresizingMask = [.width, .height]

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

    private func animatePolishResponseShell(screenshotData: Data?, model: PolishFloatingResponseModel) {
        guard let window else { return }

        let currentFrame = window.frame
        let centerX = currentFrame.midX
        let topY = currentFrame.maxY
        let wideSize = NSSize(width: 420, height: 46)
        let screenFrame = (window.screen ?? NSScreen.main)?.visibleFrame ?? .zero
        let finalWidth: CGFloat = min(720, max(420, screenFrame.width - 48))
        let finalHeight: CGFloat = min(screenshotData == nil ? 560 : 720, max(420, screenFrame.height - 72))
        let finalSize = NSSize(width: finalWidth, height: finalHeight)

        let wideFrame = NSRect(
            x: centerX - wideSize.width / 2,
            y: topY - wideSize.height,
            width: wideSize.width,
            height: wideSize.height
        )

        let unclampedFinalX = centerX - finalSize.width / 2
        let finalX = min(
            max(unclampedFinalX, screenFrame.minX + 24),
            screenFrame.maxX - finalSize.width - 24
        )
        let finalY = max(screenFrame.minY + 24, topY - finalSize.height)
        let finalFrame = NSRect(
            x: finalX,
            y: finalY,
            width: finalSize.width,
            height: finalSize.height
        )

        NSAnimationContext.runAnimationGroup { context in
            context.duration = 0.24
            context.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
            window.animator().setFrame(wideFrame, display: true)
        } completionHandler: {
            DispatchQueue.main.async {
                model.showWide()
            }

            DispatchQueue.main.async { [weak self] in
                self?.polishResponsePresentationPhase = .expanding
            }
            NSAnimationContext.runAnimationGroup { context in
                context.duration = 0.34
                context.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
                window.animator().setFrame(finalFrame, display: true)
            } completionHandler: { [weak self] in
                DispatchQueue.main.async { [weak self] in
                    model.showShell()
                    self?.polishResponsePresentationPhase = .reader
                }
            }
        }
    }
}
