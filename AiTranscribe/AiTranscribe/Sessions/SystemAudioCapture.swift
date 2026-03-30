/*
 SystemAudioCapture.swift
 ========================

 Captures system audio (what you hear) using ScreenCaptureKit.

 WHY ScreenCaptureKit?
 ---------------------
 macOS doesn't expose system audio output through AVAudioEngine.
 ScreenCaptureKit (macOS 13+) can capture system audio without
 needing third-party virtual audio devices like BlackHole.

 HOW IT WORKS:
 -------------
 1. Request Screen Recording permission (required for system audio)
 2. Create an SCStream configured for audio-only (no video)
 3. SCStreamOutput delegate receives CMSampleBuffers with audio
 4. Convert CMSampleBuffers to AVAudioPCMBuffers for mixing

 PERMISSIONS:
 ------------
 Screen Recording permission is required even though we only capture audio.
 This is a macOS limitation — ScreenCaptureKit bundles screen + audio capture
 under the same permission. The user must grant it in:
 System Settings > Privacy & Security > Screen Recording
 */

import Foundation
import ScreenCaptureKit
import AVFoundation
import CoreMedia
import Combine

/// Captures system audio output using ScreenCaptureKit
class SystemAudioCapture: NSObject, ObservableObject {

    // MARK: - State

    @Published var isCapturing = false

    /// Callback invoked with each audio buffer received from system audio
    var onAudioBuffer: ((CMSampleBuffer) -> Void)?

    /// Callback invoked with each microphone buffer received from ScreenCaptureKit.
    /// Available on macOS 15+ when captureMicrophone is enabled.
    var onMicrophoneBuffer: ((CMSampleBuffer) -> Void)?

    // MARK: - Private

    private var stream: SCStream?
    private var streamOutput: AudioStreamOutput?
    private var microphoneOutput: MicrophoneStreamOutput?
    private var videoOutput: VideoDiscardOutput?

    // MARK: - Permission Checking

    /// Synchronous, crash-safe permission pre-check.
    /// Only returns true on macOS 15+ where CGPreflightScreenCaptureAccess is available.
    /// On older macOS, returns false (system audio skipped) to avoid SCShareableContent crashes.
    static func preflightPermission() -> Bool {
        if #available(macOS 15.0, *) {
            return CGPreflightScreenCaptureAccess()
        } else {
            // On older macOS, we can't safely check without risking a crash
            // on unsigned/quarantined apps. Skip system audio.
            return false
        }
    }

    /// Async permission check — used internally by startCapture().
    /// Uses CGPreflightScreenCaptureAccess on macOS 15+, falls back to SCShareableContent check.
    static func checkPermission() async -> Bool {
        if #available(macOS 15.0, *) {
            return CGPreflightScreenCaptureAccess()
        } else {
            do {
                _ = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false)
                return true
            } catch {
                print("SystemAudioCapture: Permission check failed: \(error)")
                return false
            }
        }
    }

    /// Request Screen Recording permission.
    /// On macOS 15+, opens the system prompt. On older versions, triggers via content enumeration.
    static func requestPermission() {
        if #available(macOS 15.0, *) {
            CGRequestScreenCaptureAccess()
        } else {
            // Trigger the permission dialog by attempting to enumerate content
            Task {
                _ = try? await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false)
            }
        }
    }

    // MARK: - Capture Control

    /// Start capturing system audio and/or microphone audio.
    /// - Returns: true if capture started, false if permission denied or error
    func startCapture(
        captureSystemAudio: Bool = true,
        captureMicrophone: Bool = false,
        microphoneCaptureDeviceID: String? = nil
    ) async -> Bool {
        guard !isCapturing else { return false }
        guard captureSystemAudio || captureMicrophone else {
            print("SystemAudioCapture: No audio sources requested")
            return false
        }

        if captureMicrophone, #unavailable(macOS 15.0) {
            print("SystemAudioCapture: Microphone capture via ScreenCaptureKit requires macOS 15+")
            return false
        }

        // Check permission
        let hasPermission = await SystemAudioCapture.checkPermission()
        guard hasPermission else {
            print("SystemAudioCapture: Screen Recording permission not granted")
            return false
        }

        do {
            // Get shareable content to build a content filter
            let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false)

            guard let display = content.displays.first else {
                print("SystemAudioCapture: No displays found")
                return false
            }

            // Create a filter that captures the entire display (we only want audio, but we need a display filter)
            let filter = SCContentFilter(display: display, excludingWindows: [])

            // Configure for audio-only capture
            let config = SCStreamConfiguration()
            config.capturesAudio = captureSystemAudio
            config.excludesCurrentProcessAudio = true  // Don't capture our own app's sounds
            config.sampleRate = 48000                   // ScreenCaptureKit native rate
            config.channelCount = 2                     // Stereo system audio
            if #available(macOS 15.0, *), captureMicrophone {
                config.captureMicrophone = true
                config.microphoneCaptureDeviceID = microphoneCaptureDeviceID
            }

            // Minimize video overhead since we don't need it
            config.width = 2
            config.height = 2
            config.minimumFrameInterval = CMTime(value: 10, timescale: 1) // 0.1 FPS — one frame every 10s
            config.showsCursor = false

            // Create stream
            let stream = SCStream(filter: filter, configuration: config, delegate: nil)

            // Audio output handlers
            let output = AudioStreamOutput()
            output.onAudioBuffer = { [weak self] sampleBuffer in
                self?.onAudioBuffer?(sampleBuffer)
            }

            let microphoneOutput: MicrophoneStreamOutput?
            if #available(macOS 15.0, *), captureMicrophone {
                let micOutput = MicrophoneStreamOutput()
                micOutput.onMicrophoneBuffer = { [weak self] sampleBuffer in
                    self?.onMicrophoneBuffer?(sampleBuffer)
                }
                microphoneOutput = micOutput
            } else {
                microphoneOutput = nil
            }

            // Separate discard handler for video — using the same object for both
            // .audio and .screen causes SCStream to serialize delivery, which
            // blocks audio when a video frame arrives and causes stuttering.
            let videoDiscard = VideoDiscardOutput()

            if captureSystemAudio {
                try stream.addStreamOutput(output, type: .audio, sampleHandlerQueue: .global(qos: .userInitiated))
            }
            if let microphoneOutput, #available(macOS 15.0, *) {
                try stream.addStreamOutput(microphoneOutput, type: .microphone, sampleHandlerQueue: .global(qos: .userInitiated))
            }
            try stream.addStreamOutput(videoDiscard, type: .screen, sampleHandlerQueue: .global(qos: .background))
            try await stream.startCapture()

            self.stream = stream
            self.streamOutput = output
            self.microphoneOutput = microphoneOutput
            self.videoOutput = videoDiscard
            self.isCapturing = true

            var sources: [String] = []
            if captureSystemAudio {
                sources.append("system audio")
            }
            if captureMicrophone {
                sources.append("microphone")
            }
            print("SystemAudioCapture: Started capturing \(sources.joined(separator: " + "))")
            return true

        } catch {
            print("SystemAudioCapture: Failed to start capture: \(error)")
            return false
        }
    }

    /// Stop capturing system audio.
    func stopCapture() async {
        guard isCapturing, let stream = stream else { return }

        do {
            try await stream.stopCapture()
        } catch {
            print("SystemAudioCapture: Error stopping capture: \(error)")
        }

        self.stream = nil
        self.streamOutput = nil
        self.microphoneOutput = nil
        self.videoOutput = nil
        self.isCapturing = false
        print("SystemAudioCapture: Stopped")
    }
}

// MARK: - Stream Output Handler

/// Receives audio sample buffers from ScreenCaptureKit
private class AudioStreamOutput: NSObject, SCStreamOutput {

    var onAudioBuffer: ((CMSampleBuffer) -> Void)?

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        // We only care about audio buffers
        guard type == .audio else { return }
        guard sampleBuffer.isValid else { return }

        onAudioBuffer?(sampleBuffer)
    }
}

/// Receives microphone sample buffers from ScreenCaptureKit
@available(macOS 15.0, *)
private class MicrophoneStreamOutput: NSObject, SCStreamOutput {

    var onMicrophoneBuffer: ((CMSampleBuffer) -> Void)?

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .microphone else { return }
        guard sampleBuffer.isValid else { return }

        onMicrophoneBuffer?(sampleBuffer)
    }
}

/// Receives and discards video frames so ScreenCaptureKit doesn't error
private class VideoDiscardOutput: NSObject, SCStreamOutput {
    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        // Intentionally empty — we just need to consume the frames
    }
}
