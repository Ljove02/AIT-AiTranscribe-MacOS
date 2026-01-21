/*
 AudioRecorder.swift
 ===================

 Records audio using AVFoundation (Apple's audio framework).

 WHY RECORD IN SWIFT?
 --------------------
 macOS permissions are per-executable. When Python runs as a subprocess,
 it doesn't inherit the Swift app's microphone permission.

 By recording in Swift:
 1. The app gets the proper permission prompt
 2. Audio is captured with the app's own permission
 3. We send the audio to Python backend for transcription only

 HOW IT WORKS:
 -------------
 1. AVAudioEngine captures audio from the microphone
 2. Audio is stored in a buffer as we record
 3. When stopped, we convert to WAV format
 4. WAV data is sent to backend's /transcribe-bytes endpoint
 */

import Foundation
import AVFoundation
import Combine

/// Records audio from the microphone using AVFoundation
class AudioRecorder: NSObject, ObservableObject {

    // MARK: - Published State

    /// Is currently recording
    @Published var isRecording = false

    /// Current audio volume (0.0 to 1.0)
    @Published var currentVolume: Float = 0.0

    /// Recording duration in seconds
    @Published var duration: TimeInterval = 0.0

    // MARK: - Audio Engine

    /// The audio engine that manages audio processing
    private var audioEngine: AVAudioEngine?

    /// Buffer to store recorded audio
    private var audioBuffer: AVAudioPCMBuffer?

    /// All recorded buffers
    private var recordedBuffers: [AVAudioPCMBuffer] = []

    /// Timer for updating duration
    private var durationTimer: Timer?

    /// Start time of recording
    private var startTime: Date?

    /// Sample rate (16kHz for speech recognition)
    private let sampleRate: Double = 16000

    /// Audio format for recording
    private var recordingFormat: AVAudioFormat?

    // MARK: - Initialization

    override init() {
        super.init()
        setupAudioSession()
    }

    /// Setup the audio session for recording
    private func setupAudioSession() {
        // On macOS, we don't need to configure AVAudioSession like on iOS
        // The microphone permission is handled by the entitlements
    }

    // MARK: - Recording Control

    /// Start recording audio
    /// - Returns: True if recording started successfully
    func startRecording() -> Bool {
        guard !isRecording else { return false }

        // Clear previous recording
        recordedBuffers.removeAll()
        currentVolume = 0.0
        duration = 0.0

        // Create audio engine
        audioEngine = AVAudioEngine()
        guard let audioEngine = audioEngine else { return false }

        let inputNode = audioEngine.inputNode

        // Get the native format of the input
        let inputFormat = inputNode.outputFormat(forBus: 0)

        // Create our target format (16kHz mono for speech recognition)
        guard let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: 1,
            interleaved: false
        ) else {
            print("Failed to create target audio format")
            return false
        }

        recordingFormat = targetFormat

        // Create a converter if sample rates differ
        let converter = AVAudioConverter(from: inputFormat, to: targetFormat)

        // Install tap on input node to capture audio
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: inputFormat) { [weak self] buffer, time in
            guard let self = self else { return }

            // Convert to target format if needed
            if let converter = converter {
                let frameCount = AVAudioFrameCount(Double(buffer.frameLength) * self.sampleRate / inputFormat.sampleRate)
                guard let convertedBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: frameCount) else { return }

                var error: NSError?
                let inputBlock: AVAudioConverterInputBlock = { inNumPackets, outStatus in
                    outStatus.pointee = .haveData
                    return buffer
                }

                converter.convert(to: convertedBuffer, error: &error, withInputFrom: inputBlock)

                if error == nil {
                    self.processAudioBuffer(convertedBuffer)
                }
            } else {
                self.processAudioBuffer(buffer)
            }
        }

        // Start the engine
        do {
            try audioEngine.start()
            isRecording = true
            startTime = Date()

            // Start duration timer
            durationTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
                guard let self = self, let start = self.startTime else { return }
                DispatchQueue.main.async {
                    self.duration = Date().timeIntervalSince(start)
                }
            }

            return true
        } catch {
            print("Failed to start audio engine: \(error)")
            return false
        }
    }

    /// Process incoming audio buffer
    private func processAudioBuffer(_ buffer: AVAudioPCMBuffer) {
        // Make a copy of the buffer
        guard let bufferCopy = buffer.copy() as? AVAudioPCMBuffer else { return }
        recordedBuffers.append(bufferCopy)

        // Calculate volume (RMS)
        if let channelData = buffer.floatChannelData?[0] {
            let frameCount = Int(buffer.frameLength)
            var sum: Float = 0
            for i in 0..<frameCount {
                let sample = channelData[i]
                sum += sample * sample
            }
            let rms = sqrt(sum / Float(frameCount))
            
            // Apply noise gate - ignore very quiet sounds (hand movements, keyboard taps, etc.)
            // Typical speech is around 0.01-0.1 RMS, background noise is < 0.005
            let noiseGateThreshold: Float = 0.008  // Adjust this to change sensitivity
            let gatedRMS = rms > noiseGateThreshold ? rms : 0
            
            // Reduce overall sensitivity - only amplify by 3x instead of 5x
            // This makes it less reactive to small sounds
            let scaledVolume = min(gatedRMS * 3.0, 1.0)

            DispatchQueue.main.async {
                self.currentVolume = scaledVolume
            }
        }
    }

    /// Stop recording and return audio data
    /// - Returns: Audio data as WAV format, or nil if no audio
    func stopRecording() -> Data? {
        guard isRecording else { return nil }

        // Stop timer
        durationTimer?.invalidate()
        durationTimer = nil

        // Stop engine
        audioEngine?.inputNode.removeTap(onBus: 0)
        audioEngine?.stop()
        audioEngine = nil

        isRecording = false
        currentVolume = 0.0

        // Combine all buffers
        guard !recordedBuffers.isEmpty, let format = recordingFormat else { return nil }

        // Calculate total frame count
        let totalFrames = recordedBuffers.reduce(0) { $0 + Int($1.frameLength) }
        guard totalFrames > 0 else { return nil }

        // Create combined buffer
        guard let combinedBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(totalFrames)) else {
            return nil
        }

        // Copy all buffers into combined buffer
        var offset: AVAudioFrameCount = 0
        for buffer in recordedBuffers {
            let frameCount = buffer.frameLength
            if let srcData = buffer.floatChannelData?[0],
               let dstData = combinedBuffer.floatChannelData?[0] {
                memcpy(dstData.advanced(by: Int(offset)), srcData, Int(frameCount) * MemoryLayout<Float>.size)
            }
            offset += frameCount
        }
        combinedBuffer.frameLength = AVAudioFrameCount(totalFrames)

        // Convert to WAV data
        return createWAVData(from: combinedBuffer)
    }

    /// Cancel recording without returning data
    func cancelRecording() {
        durationTimer?.invalidate()
        durationTimer = nil

        audioEngine?.inputNode.removeTap(onBus: 0)
        audioEngine?.stop()
        audioEngine = nil

        isRecording = false
        currentVolume = 0.0
        recordedBuffers.removeAll()
    }

    // MARK: - WAV Conversion

    /// Convert audio buffer to WAV data
    private func createWAVData(from buffer: AVAudioPCMBuffer) -> Data? {
        guard let channelData = buffer.floatChannelData?[0] else { return nil }

        let frameCount = Int(buffer.frameLength)
        let sampleRate = Int(buffer.format.sampleRate)

        // Convert float samples to Int16
        var int16Samples = [Int16](repeating: 0, count: frameCount)
        for i in 0..<frameCount {
            let sample = channelData[i]
            let clipped = max(-1.0, min(1.0, sample))
            int16Samples[i] = Int16(clipped * Float(Int16.max))
        }

        // Create WAV header
        var wavData = Data()

        // RIFF header
        wavData.append(contentsOf: "RIFF".utf8)
        let fileSize = UInt32(36 + frameCount * 2)
        wavData.append(contentsOf: withUnsafeBytes(of: fileSize.littleEndian) { Array($0) })
        wavData.append(contentsOf: "WAVE".utf8)

        // fmt chunk
        wavData.append(contentsOf: "fmt ".utf8)
        wavData.append(contentsOf: withUnsafeBytes(of: UInt32(16).littleEndian) { Array($0) })  // Chunk size
        wavData.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian) { Array($0) })   // PCM format
        wavData.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian) { Array($0) })   // Mono
        wavData.append(contentsOf: withUnsafeBytes(of: UInt32(sampleRate).littleEndian) { Array($0) })  // Sample rate
        wavData.append(contentsOf: withUnsafeBytes(of: UInt32(sampleRate * 2).littleEndian) { Array($0) })  // Byte rate
        wavData.append(contentsOf: withUnsafeBytes(of: UInt16(2).littleEndian) { Array($0) })   // Block align
        wavData.append(contentsOf: withUnsafeBytes(of: UInt16(16).littleEndian) { Array($0) })  // Bits per sample

        // data chunk
        wavData.append(contentsOf: "data".utf8)
        wavData.append(contentsOf: withUnsafeBytes(of: UInt32(frameCount * 2).littleEndian) { Array($0) })

        // Audio data
        int16Samples.withUnsafeBufferPointer { ptr in
            wavData.append(UnsafeBufferPointer(start: UnsafeRawPointer(ptr.baseAddress)?.assumingMemoryBound(to: UInt8.self),
                                                count: frameCount * 2))
        }

        return wavData
    }

    // MARK: - Permissions

    /// Check if microphone permission is granted
    static var hasMicrophonePermission: Bool {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:
            return true
        default:
            return false
        }
    }

    /// Request microphone permission
    static func requestMicrophonePermission(completion: @escaping (Bool) -> Void) {
        AVCaptureDevice.requestAccess(for: .audio) { granted in
            DispatchQueue.main.async {
                completion(granted)
            }
        }
    }
    
    /// Get current authorization status
    static var authorizationStatus: AVAuthorizationStatus {
        AVCaptureDevice.authorizationStatus(for: .audio)
    }
}
