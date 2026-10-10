/*
 MicrophoneOnboardingView.swift
 ==============================

 Microphone permission screen (MANDATORY).
 
 Features:
 - Check microphone permission status
 - Request permission if not determined
 - Guide user to System Settings if denied
 - Cannot proceed without authorization
 */

import SwiftUI
import AVFoundation
import Combine

struct MicrophoneOnboardingView: View {
    let onNext: () -> Void
    let onBack: () -> Void
    
    @State private var permissionState: PermissionState = .notDetermined
    @State private var isCheckingPermission = false
    @StateObject private var micPreview = MicrophonePreviewLevelMonitor()
    
    var body: some View {
        ZStack {
            MicrophoneReactiveBackground(
                state: permissionState,
                level: micPreview.level
            )

            VStack(alignment: .leading, spacing: 0) {
                HStack(alignment: .top, spacing: 54) {
                    VStack(alignment: .leading, spacing: 28) {
                        microphoneGlyph

                        VStack(alignment: .leading, spacing: 14) {
                            Text("PERMISSION")
                                .font(.system(size: 11, weight: .bold, design: .rounded))
                                .tracking(1.7)
                                .foregroundStyle(.secondary)

                            Text("Set up microphone access")
                                .font(.system(size: 43, weight: .bold, design: .rounded))
                                .tracking(-1.3)
                                .lineSpacing(-2)
                                .foregroundStyle(.primary)
                                .fixedSize(horizontal: false, vertical: true)
                        }

                        PermissionStatusView(state: permissionState)

                        VStack(alignment: .leading, spacing: 12) {
                            ForEach(instructionItems) { item in
                                MicrophoneInstructionRow(icon: item.icon, text: item.text)
                            }
                        }
                        .frame(maxWidth: 560, alignment: .leading)

                        MicrophoneInfoNote(
                            text: "Input preview only. No audio is saved or transcribed here."
                        )
                        .frame(maxWidth: 520, alignment: .leading)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)

                    MicrophoneSignalField(
                        state: permissionState,
                        level: micPreview.level,
                        isRunning: micPreview.isRunning
                    )
                    .frame(width: 245)
                    .padding(.top, 54)
                }

                Spacer()

                HStack {
                    OnboardingSecondaryActionButton(
                        title: "Back",
                        systemImage: "arrow.left",
                        action: onBack
                    )

                    Spacer()

                    primaryAction
                }
            }
            .padding(.horizontal, 42)
            .padding(.top, 44)
            .padding(.bottom, 34)
        }
        .onAppear {
            checkPermission()
        }
        .onDisappear {
            micPreview.stop()
        }
    }

    private var microphoneGlyph: some View {
        ZStack {
            Circle()
                .fill(iconColor.opacity(0.13))
                .frame(width: 94, height: 94)

            Circle()
                .strokeBorder(iconColor.opacity(0.22), lineWidth: 1)
                .frame(width: 94, height: 94)

            Image(systemName: "mic.fill")
                .font(.system(size: 39, weight: .semibold))
                .foregroundStyle(iconColor)
        }
        .shadow(color: iconColor.opacity(0.18), radius: 18, x: 0, y: 10)
    }

    private var instructionItems: [MicrophoneInstructionItem] {
        switch permissionState {
        case .authorized:
            return [
                MicrophoneInstructionItem(icon: "waveform", text: "Speak; the signal should move."),
                MicrophoneInstructionItem(icon: "checkmark", text: "Continue when the signal responds.")
            ]
        case .denied:
            return [
                MicrophoneInstructionItem(icon: "gear", text: "Enable AiTranscribe in Privacy & Security > Microphone."),
                MicrophoneInstructionItem(icon: "arrow.clockwise", text: "Return here; access re-checks automatically.")
            ]
        case .notDetermined:
            return [
                MicrophoneInstructionItem(icon: "cursorarrow.click", text: "Request microphone access."),
                MicrophoneInstructionItem(icon: "checkmark", text: "Approve the macOS prompt.")
            ]
        }
    }

    @ViewBuilder
    private var primaryAction: some View {
        switch permissionState {
        case .authorized:
            OnboardingPrimaryActionButton(
                title: "Continue",
                systemImage: "arrow.right",
                action: onNext
            )
        case .denied:
            OnboardingPrimaryActionButton(
                title: "Open System Settings",
                systemImage: "gear",
                action: openSystemSettings
            )
        case .notDetermined:
            OnboardingPrimaryActionButton(
                title: isCheckingPermission ? "Checking..." : "Request Permission",
                systemImage: "mic.badge.plus",
                isLoading: isCheckingPermission,
                action: requestPermission
            )
        }
    }
    
    // MARK: - Permission Handling
    
    private func checkPermission() {
        let status = AVCaptureDevice.authorizationStatus(for: .audio)
        
        switch status {
        case .authorized:
            updatePermissionState(.authorized)
        case .denied, .restricted:
            updatePermissionState(.denied)
        case .notDetermined:
            updatePermissionState(.notDetermined)
        @unknown default:
            updatePermissionState(.notDetermined)
        }
    }
    
    private func requestPermission() {
        isCheckingPermission = true
        
        AVCaptureDevice.requestAccess(for: .audio) { granted in
            DispatchQueue.main.async {
                isCheckingPermission = false
                updatePermissionState(granted ? .authorized : .denied)
            }
        }
    }

    private func updatePermissionState(_ state: PermissionState) {
        permissionState = state

        if state == .authorized {
            micPreview.start()
        } else {
            micPreview.stop()
        }
    }
    
    private func openSystemSettings() {
        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone") {
            NSWorkspace.shared.open(url)
        }
    }
    
    private var iconColor: Color {
        switch permissionState {
        case .authorized:
            return .green
        case .denied:
            return .red
        case .notDetermined:
            return .accentColor
        }
    }
    
}

private struct MicrophoneInstructionItem: Identifiable {
    let icon: String
    let text: String

    var id: String { "\(icon)-\(text)" }
}

private struct MicrophoneInstructionRow: View {
    let icon: String
    let text: String

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: icon)
                .font(.system(size: 10, weight: .bold))
                .foregroundStyle(.white)
                .frame(width: 20, height: 20)
                .background(.white.opacity(0.10), in: Circle())
                .overlay(
                    Circle()
                        .strokeBorder(.white.opacity(0.12), lineWidth: 1)
                )
            Text(text)
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

private struct MicrophoneInfoNote: View {
    let text: String

    var body: some View {
        HStack(alignment: .top, spacing: 9) {
            Image(systemName: "info.circle.fill")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(.blue)
                .padding(.top, 1)

            Text(text)
                .font(.caption.weight(.medium))
                .lineSpacing(2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

private struct MicrophoneReactiveBackground: View {
    let state: PermissionState
    let level: Double

    private var activeLevel: Double {
        state == .authorized ? min(max(level, 0), 1) : 0
    }

    var body: some View {
        ZStack {
            LinearGradient(
                colors: [
                    Color.clear,
                    statusColor.opacity(0.03 + activeLevel * 0.09),
                    Color.teal.opacity(state == .authorized ? 0.04 + activeLevel * 0.08 : 0.02)
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )

            Circle()
                .fill(statusColor.opacity(0.13 + activeLevel * 0.20))
                .frame(width: 420 + activeLevel * 120, height: 420 + activeLevel * 120)
                .blur(radius: 88)
                .offset(x: 315, y: -76)
                .animation(.spring(duration: 0.28, bounce: 0.10), value: activeLevel)

            Circle()
                .fill(Color.teal.opacity(state == .authorized ? 0.08 + activeLevel * 0.16 : 0.03))
                .frame(width: 360 + activeLevel * 100, height: 360 + activeLevel * 100)
                .blur(radius: 104)
                .offset(x: 190, y: 260)
                .animation(.spring(duration: 0.32, bounce: 0.08), value: activeLevel)
        }
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }

    private var statusColor: Color {
        switch state {
        case .authorized:
            return .green
        case .denied:
            return .red
        case .notDetermined:
            return .blue
        }
    }
}

private struct MicrophoneSignalField: View {
    let state: PermissionState
    let level: Double
    let isRunning: Bool

    private var activeLevel: Double {
        state == .authorized ? min(max(level, 0), 1) : 0
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text("Live input")
                .font(.system(size: 12, weight: .bold, design: .rounded))
                .tracking(1.2)
                .foregroundStyle(.secondary)

            ZStack {
                ForEach(0..<3, id: \.self) { index in
                    Circle()
                        .strokeBorder(
                            statusColor.opacity(ringOpacity(for: index)),
                            lineWidth: index == 0 ? 1.5 : 1
                        )
                        .frame(
                            width: ringSize(for: index),
                            height: ringSize(for: index)
                        )
                        .scaleEffect(1 + activeLevel * (0.06 + Double(index) * 0.025))
                        .animation(.spring(duration: 0.28, bounce: 0.10), value: activeLevel)
                }

                Circle()
                    .fill(
                        RadialGradient(
                            colors: [
                                statusColor.opacity(0.23 + activeLevel * 0.25),
                                Color.teal.opacity(state == .authorized ? 0.10 + activeLevel * 0.22 : 0.04),
                                Color.clear
                            ],
                            center: .center,
                            startRadius: 0,
                            endRadius: 106
                        )
                    )
                    .frame(width: 210, height: 210)
                    .blur(radius: 2)
                    .scaleEffect(1 + activeLevel * 0.10)
                    .animation(.spring(duration: 0.24, bounce: 0.12), value: activeLevel)

                Circle()
                    .fill(.white.opacity(0.08 + activeLevel * 0.12))
                    .frame(width: 9 + activeLevel * 9, height: 9 + activeLevel * 9)
            }
            .frame(width: 230, height: 230)
            .accessibilityHidden(true)

            VStack(alignment: .leading, spacing: 7) {
                HStack(spacing: 8) {
                    Circle()
                        .fill(statusColor)
                        .frame(width: 8, height: 8)
                    Text(statusText)
                        .font(.system(size: 14, weight: .bold, design: .rounded))
                        .foregroundStyle(.primary)
                }

                Text(detailText)
                    .font(.system(size: 12, weight: .medium))
                    .lineSpacing(2)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            HStack(alignment: .top, spacing: 9) {
                Image(systemName: "display")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(.secondary)
                    .padding(.top, 1)

                Text("Desktop/session audio uses Screen Recording permission later.")
                    .font(.system(size: 12, weight: .medium))
                    .lineSpacing(2)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(.leading, 10)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Microphone input preview")
        .accessibilityValue(accessibilityValue)
    }

    private func ringSize(for index: Int) -> CGFloat {
        CGFloat(106 + index * 48) + CGFloat(activeLevel) * CGFloat(16 + index * 9)
    }

    private func ringOpacity(for index: Int) -> Double {
        let base = [0.28, 0.20, 0.14][index]
        let activeBoost = activeLevel * (0.20 - Double(index) * 0.04)
        let fadeForOuterRing = index == 2 ? (1 - activeLevel * 0.78) : 1
        return max(0.035, (base + activeBoost) * fadeForOuterRing)
    }

    private var statusColor: Color {
        switch state {
        case .authorized:
            return .green
        case .denied:
            return .red
        case .notDetermined:
            return .orange
        }
    }

    private var statusText: String {
        switch state {
        case .authorized:
            return isRunning ? "Listening" : "Starting preview"
        case .denied:
            return "Access blocked"
        case .notDetermined:
            return "Waiting for access"
        }
    }

    private var detailText: String {
        switch state {
        case .authorized:
            return "Signal responds to your voice."
        case .denied:
            return "Enable access to activate the signal."
        case .notDetermined:
            return "Request access to start the signal."
        }
    }

    private var accessibilityValue: String {
        switch state {
        case .authorized:
            return "Input level \(Int(activeLevel * 100)) percent"
        case .denied:
            return "Microphone access denied"
        case .notDetermined:
            return "Microphone access not requested"
        }
    }
}

// MARK: - Permission State (Microphone)

fileprivate enum PermissionState: Equatable {
    case notDetermined
    case authorized
    case denied
}

// MARK: - Permission Status View (Microphone)

fileprivate struct PermissionStatusView: View {
    let state: PermissionState
    
    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: iconName)
                .font(.title2)
                .foregroundStyle(iconColor)
            
            Text(message)
                .font(.headline)
                .foregroundStyle(textColor)
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 12)
        .background(backgroundColor.opacity(0.1))
        .clipShape(.rect(cornerRadius: 12))
    }
    
    private var iconName: String {
        switch state {
        case .authorized:
            return "checkmark.circle.fill"
        case .denied:
            return "xmark.circle.fill"
        case .notDetermined:
            return "questionmark.circle.fill"
        }
    }
    
    private var iconColor: Color {
        switch state {
        case .authorized:
            return .green
        case .denied:
            return .red
        case .notDetermined:
            return .orange
        }
    }
    
    private var message: String {
        switch state {
        case .authorized:
            return "Microphone access granted"
        case .denied:
            return "Microphone access denied"
        case .notDetermined:
            return "Permission not yet requested"
        }
    }
    
    private var textColor: Color {
        switch state {
        case .authorized:
            return .green
        case .denied:
            return .red
        case .notDetermined:
            return .orange
        }
    }
    
    private var backgroundColor: Color {
        switch state {
        case .authorized:
            return .green
        case .denied:
            return .red
        case .notDetermined:
            return .orange
        }
    }
}

private final class MicrophonePreviewLevelMonitor: NSObject, ObservableObject, AVCaptureAudioDataOutputSampleBufferDelegate {
    @Published private(set) var level: Double = 0
    @Published private(set) var isRunning = false

    private let session = AVCaptureSession()
    private let queue = DispatchQueue(label: "com.aitranscribe.onboarding.microphone-preview")
    private var audioOutput: AVCaptureAudioDataOutput?
    private var isConfigured = false
    private var smoothedLevel: Double = 0
    private var lastPublishedAt: TimeInterval = 0

    func start() {
        guard !isRunning else { return }

        queue.async { [weak self] in
            guard let self else { return }

            do {
                try self.configureIfNeeded()
            } catch {
                self.publish(level: 0, running: false)
                return
            }

            guard !self.session.isRunning else {
                self.publish(level: self.smoothedLevel, running: true)
                return
            }

            self.session.startRunning()
            self.publish(level: self.smoothedLevel, running: self.session.isRunning)
        }
    }

    func stop() {
        queue.async { [weak self] in
            guard let self else { return }

            if self.session.isRunning {
                self.session.stopRunning()
            }

            self.smoothedLevel = 0
            self.publish(level: 0, running: false)
        }
    }

    private func configureIfNeeded() throws {
        guard !isConfigured else { return }

        session.beginConfiguration()
        session.sessionPreset = .low
        defer { session.commitConfiguration() }

        guard let device = AudioRecorder.currentRecordingCaptureDevice() else {
            throw MicrophonePreviewError.noDevice
        }

        let input = try AVCaptureDeviceInput(device: device)
        guard session.canAddInput(input) else {
            throw MicrophonePreviewError.cannotAddInput
        }
        session.addInput(input)

        let output = AVCaptureAudioDataOutput()
        output.setSampleBufferDelegate(self, queue: queue)
        guard session.canAddOutput(output) else {
            throw MicrophonePreviewError.cannotAddOutput
        }
        session.addOutput(output)
        audioOutput = output
        isConfigured = true
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        let measuredLevel = normalizedLevel(from: sampleBuffer)
        smoothedLevel = smoothedLevel * 0.70 + measuredLevel * 0.30

        let now = Date().timeIntervalSince1970
        guard now - lastPublishedAt > 0.05 else { return }
        lastPublishedAt = now
        publish(level: smoothedLevel, running: session.isRunning)
    }

    private func normalizedLevel(from sampleBuffer: CMSampleBuffer) -> Double {
        guard let formatDescription = CMSampleBufferGetFormatDescription(sampleBuffer),
              let streamDescription = CMAudioFormatDescriptionGetStreamBasicDescription(formatDescription) else {
            return 0
        }

        guard let format = AVAudioFormat(streamDescription: streamDescription) else {
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
        return min(1, pow(rms * 14, 0.72))
    }

    private func rmsLevel(from buffer: AVAudioPCMBuffer) -> Double {
        let frames = Int(buffer.frameLength)
        let channels = Int(buffer.format.channelCount)
        guard frames > 0, channels > 0 else { return 0 }

        if let channelData = buffer.floatChannelData {
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

    private func publish(level: Double, running: Bool) {
        DispatchQueue.main.async {
            self.level = level
            self.isRunning = running
        }
    }
}

private enum MicrophonePreviewError: Error {
    case noDevice
    case cannotAddInput
    case cannotAddOutput
}

// MARK: - Preview

#Preview {
    MicrophoneOnboardingView(onNext: {}, onBack: {})
        .frame(width: 900, height: 720)
}
