/*
 AudioRecorder.swift
 ===================

 Records audio using AVCaptureSession (Apple's capture framework).

 WHY RECORD IN SWIFT?
 --------------------
 macOS permissions are per-executable. When Python runs as a subprocess,
 it doesn't inherit the Swift app's microphone permission.

 By recording in Swift:
 1. The app gets the proper permission prompt
 2. Audio is captured with the app's own permission
 3. We send the audio to Python backend for transcription only

 WHY AVCaptureSession (NOT AVAudioEngine)?
 ------------------------------------------
 AVAudioEngine's inputNode tap silently dies when another app activates
 a Voice Processing AudioUnit (Google Meet, Zoom, etc.). AVCaptureSession
 captures at the HAL level and coexists with video call apps.

 HOW IT WORKS:
 -------------
 1. AVCaptureSession captures audio from the microphone as CMSampleBuffers
 2. Buffers are converted to 16kHz mono and stored in memory
 3. When stopped, we combine buffers into WAV format
 4. WAV data is sent to backend's /transcribe-bytes endpoint
 */

import Foundation
import AVFoundation
import CoreAudio
import AudioToolbox
import Combine

/// Records audio from the microphone using AVFoundation
class AudioRecorder: NSObject, ObservableObject {
    private static func fourCCString(_ value: OSType) -> String {
        let bytes: [CChar] = [
            CChar((value >> 24) & 0xFF),
            CChar((value >> 16) & 0xFF),
            CChar((value >> 8) & 0xFF),
            CChar(value & 0xFF),
            0
        ]
        return String(cString: bytes)
    }

    private static func canonicalPCMFormat(from formatDesc: CMAudioFormatDescription) -> AVAudioFormat? {
        guard let asbdPointer = CMAudioFormatDescriptionGetStreamBasicDescription(formatDesc) else {
            return nil
        }

        let asbd = asbdPointer.pointee
        guard asbd.mFormatID == kAudioFormatLinearPCM,
              asbd.mSampleRate > 0,
              asbd.mChannelsPerFrame > 0 else {
            return nil
        }

        let isFloat = (asbd.mFormatFlags & kLinearPCMFormatFlagIsFloat) != 0
        let isNonInterleaved = (asbd.mFormatFlags & kAudioFormatFlagIsNonInterleaved) != 0

        let commonFormat: AVAudioCommonFormat
        switch (isFloat, asbd.mBitsPerChannel) {
        case (true, 32):
            commonFormat = .pcmFormatFloat32
        case (true, 64):
            commonFormat = .pcmFormatFloat64
        case (false, 16):
            commonFormat = .pcmFormatInt16
        case (false, 32):
            commonFormat = .pcmFormatInt32
        default:
            return nil
        }

        return AVAudioFormat(
            commonFormat: commonFormat,
            sampleRate: asbd.mSampleRate,
            channels: AVAudioChannelCount(asbd.mChannelsPerFrame),
            interleaved: !isNonInterleaved
        )
    }

    private static func stringProperty(
        for deviceId: AudioDeviceID,
        selector: AudioObjectPropertySelector,
        scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal
    ) -> String? {
        var address = AudioObjectPropertyAddress(
            mSelector: selector,
            mScope: scope,
            mElement: kAudioObjectPropertyElementMain
        )
        var value: CFString = "" as CFString
        var size = UInt32(MemoryLayout<CFString>.size)
        let status = AudioObjectGetPropertyData(deviceId, &address, 0, nil, &size, &value)
        guard status == noErr else { return nil }
        return value as String
    }

    static func getInputDeviceUID(_ deviceId: AudioDeviceID) -> String? {
        stringProperty(for: deviceId, selector: kAudioDevicePropertyDeviceUID)
    }

    static func getInputDeviceName(_ deviceId: AudioDeviceID) -> String? {
        stringProperty(for: deviceId, selector: kAudioDevicePropertyDeviceNameCFString)
    }

    static func captureDevice(for deviceId: AudioDeviceID?) -> AVCaptureDevice? {
        let captureDevices = AVCaptureDevice.devices(for: .audio)
        guard let deviceId else {
            return AVCaptureDevice.default(for: .audio) ?? captureDevices.first
        }

        let targetUID = getInputDeviceUID(deviceId)
        let targetName = getInputDeviceName(deviceId)

        if let targetUID,
           let matchingDevice = captureDevices.first(where: { $0.uniqueID == targetUID }) {
            return matchingDevice
        }

        if let targetName,
           let matchingDevice = captureDevices.first(where: { $0.localizedName == targetName }) {
            return matchingDevice
        }

        return AVCaptureDevice.default(for: .audio) ?? captureDevices.first
    }

    private func copyPCMData(from sampleBuffer: CMSampleBuffer,
                             frameCount: AVAudioFrameCount,
                             into pcmBuffer: AVAudioPCMBuffer) -> Bool {
        let status = CMSampleBufferCopyPCMDataIntoAudioBufferList(
            sampleBuffer,
            at: 0,
            frameCount: Int32(frameCount),
            into: pcmBuffer.mutableAudioBufferList
        )
        guard status == noErr else {
            print("AudioRecorder: Failed to copy PCM data: \(status)")
            return false
        }
        return true
    }

    private func normalizeMicBuffer(from sampleBuffer: CMSampleBuffer) -> AVAudioPCMBuffer? {
        guard let formatDesc = sampleBuffer.formatDescription else { return nil }
        let mediaSubType = CMFormatDescriptionGetMediaSubType(formatDesc)
        guard mediaSubType == kAudioFormatLinearPCM else {
            let fourCC = Self.fourCCString(mediaSubType)
            print("AudioRecorder: Unsupported mic buffer format: \(fourCC)")
            return nil
        }
        guard let sourceFormat = Self.canonicalPCMFormat(from: formatDesc) else {
            print("AudioRecorder: Failed to canonicalize mic PCM format")
            return nil
        }

        let frameCount = AVAudioFrameCount(sampleBuffer.numSamples)
        guard frameCount > 0 else { return nil }
        guard let sourcePCM = AVAudioPCMBuffer(pcmFormat: sourceFormat, frameCapacity: frameCount) else { return nil }
        sourcePCM.frameLength = frameCount

        guard copyPCMData(from: sampleBuffer, frameCount: frameCount, into: sourcePCM) else { return nil }

        if sourceFormat.commonFormat == .pcmFormatFloat32, !sourceFormat.isInterleaved {
            return sourcePCM
        }

        guard let normalizedFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sourceFormat.sampleRate,
            channels: sourceFormat.channelCount,
            interleaved: false
        ) else {
            return nil
        }
        guard let normalizedPCM = AVAudioPCMBuffer(pcmFormat: normalizedFormat, frameCapacity: frameCount),
              let normalizedChannelData = normalizedPCM.floatChannelData else {
            return nil
        }
        normalizedPCM.frameLength = frameCount

        let frames = Int(frameCount)
        let channels = Int(sourceFormat.channelCount)
        let sourceBuffers = UnsafeMutableAudioBufferListPointer(sourcePCM.mutableAudioBufferList)

        func clamp(_ value: Float) -> Float {
            min(max(value, -1.0), 1.0)
        }

        switch sourceFormat.commonFormat {
        case .pcmFormatFloat32:
            if sourceFormat.isInterleaved {
                guard let rawData = sourceBuffers[0].mData?.assumingMemoryBound(to: Float.self) else { return nil }
                for frame in 0..<frames {
                    let baseIndex = frame * channels
                    for channel in 0..<channels {
                        normalizedChannelData[channel][frame] = clamp(rawData[baseIndex + channel])
                    }
                }
            } else {
                for channel in 0..<channels {
                    guard let rawData = sourceBuffers[channel].mData?.assumingMemoryBound(to: Float.self) else { return nil }
                    for frame in 0..<frames {
                        normalizedChannelData[channel][frame] = clamp(rawData[frame])
                    }
                }
            }
        case .pcmFormatFloat64:
            if sourceFormat.isInterleaved {
                guard let rawData = sourceBuffers[0].mData?.assumingMemoryBound(to: Double.self) else { return nil }
                for frame in 0..<frames {
                    let baseIndex = frame * channels
                    for channel in 0..<channels {
                        normalizedChannelData[channel][frame] = clamp(Float(rawData[baseIndex + channel]))
                    }
                }
            } else {
                for channel in 0..<channels {
                    guard let rawData = sourceBuffers[channel].mData?.assumingMemoryBound(to: Double.self) else { return nil }
                    for frame in 0..<frames {
                        normalizedChannelData[channel][frame] = clamp(Float(rawData[frame]))
                    }
                }
            }
        case .pcmFormatInt16:
            let scale = Float(Int16.max)
            if sourceFormat.isInterleaved {
                guard let rawData = sourceBuffers[0].mData?.assumingMemoryBound(to: Int16.self) else { return nil }
                for frame in 0..<frames {
                    let baseIndex = frame * channels
                    for channel in 0..<channels {
                        normalizedChannelData[channel][frame] = clamp(Float(rawData[baseIndex + channel]) / scale)
                    }
                }
            } else {
                for channel in 0..<channels {
                    guard let rawData = sourceBuffers[channel].mData?.assumingMemoryBound(to: Int16.self) else { return nil }
                    for frame in 0..<frames {
                        normalizedChannelData[channel][frame] = clamp(Float(rawData[frame]) / scale)
                    }
                }
            }
        case .pcmFormatInt32:
            let scale = Float(Int32.max)
            if sourceFormat.isInterleaved {
                guard let rawData = sourceBuffers[0].mData?.assumingMemoryBound(to: Int32.self) else { return nil }
                for frame in 0..<frames {
                    let baseIndex = frame * channels
                    for channel in 0..<channels {
                        normalizedChannelData[channel][frame] = clamp(Float(rawData[baseIndex + channel]) / scale)
                    }
                }
            } else {
                for channel in 0..<channels {
                    guard let rawData = sourceBuffers[channel].mData?.assumingMemoryBound(to: Int32.self) else { return nil }
                    for frame in 0..<frames {
                        normalizedChannelData[channel][frame] = clamp(Float(rawData[frame]) / scale)
                    }
                }
            }
        default:
            return nil
        }

        return normalizedPCM
    }


    // MARK: - Published State

    /// Is currently recording
    @Published var isRecording = false

    /// Current audio volume (0.0 to 1.0)
    @Published var currentVolume: Float = 0.0

    /// Recording duration in seconds
    @Published var duration: TimeInterval = 0.0

    // MARK: - Capture Session

    /// AVCaptureSession captures at the HAL level — coexists with
    /// video call apps that use Voice Processing AudioUnits.
    private var captureSession: AVCaptureSession?
    private let captureQueue = DispatchQueue(label: "com.aitranscribe.audio-capture")
    private let compatibilityMicCapture = CoreAudioMicrophoneCapture(queueLabel: "com.aitranscribe.audioqueue-mic")
    private var usingCompatibilityMicrophone = false
    private var fallbackMonitorTask: Task<Void, Never>?
    private var recordingToken = UUID()
    private var micCallbackCount: Int64 = 0

    /// Converts native mic format to 16kHz mono for speech recognition
    private var audioConverter: AVAudioConverter?

    /// Adaptive gain — compensates for hardware gain changes
    private var micGain: Float = 2.0

    /// All recorded buffers (16kHz mono)
    private var recordedBuffers: [AVAudioPCMBuffer] = []

    /// Timer for updating duration
    private var durationTimer: Timer?

    /// Start time of recording
    private var startTime: Date?

    /// Sample rate (16kHz for speech recognition)
    private let sampleRate: Double = 16000

    /// Audio format for recording (16kHz mono float32)
    private var recordingFormat: AVAudioFormat?

    /// Request native Linear PCM from AVCapture. We deliberately avoid
    /// forcing float/non-interleaved output because video-call apps can
    /// expose voice-processed telephony formats that normalize more
    /// reliably when we preserve their native PCM layout first.
    private let captureAudioSettings: [String: Any] = [
        AVFormatIDKey: kAudioFormatLinearPCM,
    ]

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

    /// Start recording audio with an optional specific device
    /// - Parameter deviceId: CoreAudio AudioDeviceID to record from. If nil, uses system default.
    /// - Returns: True if recording started successfully
    func startRecording(deviceId: AudioDeviceID? = nil) async -> Bool {
        guard !isRecording else { return false }

        // Clear previous recording
        recordedBuffers.removeAll()
        audioConverter = nil
        micGain = 2.0
        currentVolume = 0.0
        duration = 0.0
        micCallbackCount = 0
        fallbackMonitorTask?.cancel()
        recordingToken = UUID()

        // Create our target format (16kHz mono for speech recognition)
        guard let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: 1,
            interleaved: false
        ) else {
            print("AudioRecorder: Failed to create target audio format")
            return false
        }
        recordingFormat = targetFormat

        // Setup AVCaptureSession
        let session = AVCaptureSession()

        guard let micDevice = AudioRecorder.captureDevice(for: deviceId) else {
            print("AudioRecorder: No mic device available")
            return false
        }

        do {
            let input = try AVCaptureDeviceInput(device: micDevice)
            guard session.canAddInput(input) else {
                print("AudioRecorder: Cannot add mic input")
                return false
            }
            session.addInput(input)
        } catch {
            print("AudioRecorder: Failed to create mic input: \(error)")
            return false
        }

        let audioOutput = AVCaptureAudioDataOutput()
        audioOutput.audioSettings = captureAudioSettings
        audioOutput.setSampleBufferDelegate(self, queue: captureQueue)
        guard session.canAddOutput(audioOutput) else {
            print("AudioRecorder: Cannot add audio output")
            return false
        }
        session.addOutput(audioOutput)

        session.startRunning()
        captureSession = session
        usingCompatibilityMicrophone = false
        isRecording = true
        startTime = Date()
        startDurationTimer()
        scheduleCompatibilityFallback(recordingToken: recordingToken, deviceId: deviceId)

        if let deviceId {
            let coreAudioName = AudioRecorder.getInputDeviceName(deviceId) ?? "unknown"
            let coreAudioUID = AudioRecorder.getInputDeviceUID(deviceId) ?? "unknown"
            print("AudioRecorder: Started via AVCaptureSession (CoreAudio: \(coreAudioName) [\(deviceId)] / UID: \(coreAudioUID), capture: \(micDevice.localizedName) / \(micDevice.uniqueID))")
        } else {
            print("AudioRecorder: Started via AVCaptureSession (device: \(micDevice.localizedName) / \(micDevice.uniqueID))")
        }
        return true
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
    func stopRecording() async -> Data? {
        guard isRecording else { return nil }

        // Stop timer
        durationTimer?.invalidate()
        durationTimer = nil
        fallbackMonitorTask?.cancel()
        fallbackMonitorTask = nil

        if usingCompatibilityMicrophone {
            compatibilityMicCapture.stop()
            usingCompatibilityMicrophone = false
        } else {
            captureSession?.stopRunning()
            captureSession = nil
        }
        audioConverter = nil

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

        print("AudioRecorder: Stopped (frames: \(totalFrames), duration: \(String(format: "%.1f", duration))s)")

        // Convert to WAV data
        return createWAVData(from: combinedBuffer)
    }

    /// Cancel recording without returning data
    func cancelRecording() async {
        durationTimer?.invalidate()
        durationTimer = nil
        fallbackMonitorTask?.cancel()
        fallbackMonitorTask = nil

        if usingCompatibilityMicrophone {
            compatibilityMicCapture.stop()
            usingCompatibilityMicrophone = false
        } else {
            captureSession?.stopRunning()
            captureSession = nil
        }
        audioConverter = nil

        isRecording = false
        currentVolume = 0.0
        recordedBuffers.removeAll()
    }

    private func startDurationTimer() {
        durationTimer?.invalidate()
        durationTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            guard let self, let start = self.startTime else { return }
            DispatchQueue.main.async {
                self.duration = Date().timeIntervalSince(start)
            }
        }
    }

    private func scheduleCompatibilityFallback(recordingToken: UUID, deviceId: AudioDeviceID?) {
        fallbackMonitorTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 1_000_000_000)
            guard let self else { return }
            guard !Task.isCancelled else { return }
            guard self.recordingToken == recordingToken else { return }
            guard self.isRecording, !self.usingCompatibilityMicrophone else { return }
            guard self.captureSession != nil else { return }

            if self.micCallbackCount == 0 {
                print("AudioRecorder: No AVCapture microphone callbacks detected, switching to CoreAudio microphone compatibility path")
                let started = await self.switchToCompatibilityMicrophone(deviceId: deviceId)
                if started {
                    print("AudioRecorder: Compatibility microphone fallback enabled")
                } else {
                    print("AudioRecorder: Compatibility microphone fallback failed")
                }
            }
        }
    }

    private func switchToCompatibilityMicrophone(deviceId: AudioDeviceID?) async -> Bool {
        guard isRecording, !usingCompatibilityMicrophone else { return false }

        let started = startCompatibilityMicrophone(deviceId: deviceId)
        guard started else { return false }

        captureSession?.stopRunning()
        captureSession = nil
        audioConverter = nil
        micCallbackCount = 0

        usingCompatibilityMicrophone = true
        return true
    }

    private func startCompatibilityMicrophone(deviceId: AudioDeviceID?) -> Bool {
        compatibilityMicCapture.onBuffer = { [weak self] buffer in
            self?.handleNormalizedMicBuffer(buffer)
        }

        let started = compatibilityMicCapture.start(deviceId: deviceId)
        guard started else { return false }

        if let deviceId,
           let micDevice = AudioRecorder.captureDevice(for: deviceId),
           let coreAudioUID = AudioRecorder.getInputDeviceUID(deviceId) {
            let coreAudioName = AudioRecorder.getInputDeviceName(deviceId) ?? "unknown"
            print("AudioRecorder: Started via CoreAudio AudioQueue microphone (CoreAudio: \(coreAudioName) [\(deviceId)] / UID: \(coreAudioUID), capture: \(micDevice.localizedName) / \(micDevice.uniqueID))")
        } else if let micDevice = AudioRecorder.captureDevice(for: deviceId) {
            print("AudioRecorder: Started via CoreAudio AudioQueue microphone (device: \(micDevice.localizedName) / \(micDevice.uniqueID))")
        } else {
            print("AudioRecorder: Started via CoreAudio AudioQueue microphone (system default device)")
        }

        return true
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

    // MARK: - CoreAudio Device Management

    /// List all available audio input devices using CoreAudio
    static func listInputDevices() -> [InputDevice] {
        var propertyAddress = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDevices,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )

        // Get size of device list
        var dataSize: UInt32 = 0
        var status = AudioObjectGetPropertyDataSize(
            AudioObjectID(kAudioObjectSystemObject),
            &propertyAddress, 0, nil, &dataSize
        )
        guard status == noErr else { return [] }

        // Get device IDs
        let deviceCount = Int(dataSize) / MemoryLayout<AudioDeviceID>.size
        var deviceIDs = [AudioDeviceID](repeating: 0, count: deviceCount)
        status = AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject),
            &propertyAddress, 0, nil, &dataSize, &deviceIDs
        )
        guard status == noErr else { return [] }

        // Get default input device
        let defaultDeviceId = getDefaultInputDeviceId()

        // Filter to input devices and get their names
        var inputDevices: [InputDevice] = []
        for deviceID in deviceIDs {
            // Check if device has input channels
            var streamAddress = AudioObjectPropertyAddress(
                mSelector: kAudioDevicePropertyStreamConfiguration,
                mScope: kAudioDevicePropertyScopeInput,
                mElement: kAudioObjectPropertyElementMain
            )
            var streamSize: UInt32 = 0
            status = AudioObjectGetPropertyDataSize(deviceID, &streamAddress, 0, nil, &streamSize)
            guard status == noErr, streamSize > 0 else { continue }

            let bufferListPointer = UnsafeMutablePointer<AudioBufferList>.allocate(capacity: 1)
            defer { bufferListPointer.deallocate() }
            status = AudioObjectGetPropertyData(deviceID, &streamAddress, 0, nil, &streamSize, bufferListPointer)
            guard status == noErr else { continue }

            let bufferList = UnsafeMutableAudioBufferListPointer(bufferListPointer)
            let inputChannels = bufferList.reduce(0) { $0 + Int($1.mNumberChannels) }
            guard inputChannels > 0 else { continue }

            // Get device name
            var nameAddress = AudioObjectPropertyAddress(
                mSelector: kAudioDevicePropertyDeviceNameCFString,
                mScope: kAudioObjectPropertyScopeGlobal,
                mElement: kAudioObjectPropertyElementMain
            )
            var name: CFString = "" as CFString
            var nameSize = UInt32(MemoryLayout<CFString>.size)
            status = AudioObjectGetPropertyData(deviceID, &nameAddress, 0, nil, &nameSize, &name)
            let deviceName = status == noErr ? name as String : "Unknown Device"

            // Skip virtual aggregate devices created by CoreAudio (not real microphones)
            if deviceName.localizedCaseInsensitiveContains("aggregate")
                || deviceName.contains("CADefaultDevice") {
                continue
            }

            inputDevices.append(InputDevice(
                id: deviceID,
                name: deviceName,
                isDefault: deviceID == defaultDeviceId
            ))
        }

        return inputDevices
    }

    /// Set the system default input device
    @discardableResult
    static func setDefaultInputDevice(_ deviceId: AudioDeviceID) -> Bool {
        var propertyAddress = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultInputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var devId = deviceId
        let status = AudioObjectSetPropertyData(
            AudioObjectID(kAudioObjectSystemObject),
            &propertyAddress, 0, nil,
            UInt32(MemoryLayout<AudioDeviceID>.size),
            &devId
        )
        if status != noErr {
            print("Failed to set default input device \(deviceId): OSStatus \(status)")
        }
        return status == noErr
    }

    /// Get the default input device ID
    static func getDefaultInputDeviceId() -> AudioDeviceID {
        var propertyAddress = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultInputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var deviceId: AudioDeviceID = 0
        var size = UInt32(MemoryLayout<AudioDeviceID>.size)
        AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject),
            &propertyAddress, 0, nil, &size, &deviceId
        )
        return deviceId
    }

    /// Register a listener for audio device changes (device added/removed, default changed)
    /// Returns a closure that removes the listener when called
    static func onDeviceChange(callback: @escaping () -> Void) -> (() -> Void) {
        let systemObject = AudioObjectID(kAudioObjectSystemObject)

        // Listen for device list changes
        var devicesAddress = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDevices,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )

        // Listen for default input device changes
        var defaultAddress = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultInputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )

        let queue = DispatchQueue.main

        let devicesBlock: AudioObjectPropertyListenerBlock = { _, _ in
            callback()
        }
        let defaultBlock: AudioObjectPropertyListenerBlock = { _, _ in
            callback()
        }

        AudioObjectAddPropertyListenerBlock(systemObject, &devicesAddress, queue, devicesBlock)
        AudioObjectAddPropertyListenerBlock(systemObject, &defaultAddress, queue, defaultBlock)

        // Return cleanup closure
        return {
            var da = devicesAddress
            var dfa = defaultAddress
            AudioObjectRemovePropertyListenerBlock(systemObject, &da, queue, devicesBlock)
            AudioObjectRemovePropertyListenerBlock(systemObject, &dfa, queue, defaultBlock)
        }
    }
}

// MARK: - Input Device Model

// MARK: - AVCaptureSession Mic Delegate

extension AudioRecorder: AVCaptureAudioDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        handleMicSampleBuffer(sampleBuffer)
    }

    private func handleMicSampleBuffer(_ sampleBuffer: CMSampleBuffer) {
        guard isRecording else { return }
        guard let sourcePCM = normalizeMicBuffer(from: sampleBuffer) else { return }
        handleNormalizedMicBuffer(sourcePCM)
    }

    private func handleNormalizedMicBuffer(_ sourcePCM: AVAudioPCMBuffer) {
        guard isRecording else { return }
        micCallbackCount += 1
        let sourceFormat = sourcePCM.format
        let frameCount = sourcePCM.frameLength

        // Create converter lazily from first buffer's format
        if audioConverter == nil, let targetFormat = recordingFormat {
            audioConverter = AVAudioConverter(from: sourceFormat, to: targetFormat)
        }

        // Adaptive mic gain — targets consistent RMS regardless of
        // hardware gain changes (e.g. Google Meet lowering mic volume)
        if let channelData = sourcePCM.floatChannelData {
            let frames = Int(sourcePCM.frameLength)
            let channels = Int(sourceFormat.channelCount)

            var sum: Float = 0
            for i in 0..<frames { let s = channelData[0][i]; sum += s * s }
            let rms = sqrt(sum / Float(max(frames, 1)))

            let targetRMS: Float = 0.05
            if rms > 0.002 {
                let desiredGain = min(targetRMS / rms, 10.0)
                let alpha: Float = desiredGain > micGain ? 0.3 : 0.05
                micGain += alpha * (desiredGain - micGain)
            }
            micGain = max(micGain, 1.0)

            for ch in 0..<channels {
                for i in 0..<frames {
                    channelData[ch][i] = min(max(channelData[ch][i] * micGain, -1.0), 1.0)
                }
            }
        }

        // Convert to 16kHz mono target format
        if let converter = audioConverter, let targetFormat = recordingFormat {
            let outputFrameCount = AVAudioFrameCount(
                Double(frameCount) * targetFormat.sampleRate / sourceFormat.sampleRate
            )
            guard outputFrameCount > 0 else { return }
            guard let convertedBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: outputFrameCount) else { return }

            var error: NSError?
            var providedInput = false
            let inputBlock: AVAudioConverterInputBlock = { _, outStatus in
                if providedInput {
                    outStatus.pointee = .noDataNow
                    return nil
                }
                providedInput = true
                outStatus.pointee = .haveData
                return sourcePCM
            }
            let status = converter.convert(to: convertedBuffer, error: &error, withInputFrom: inputBlock)

            if let error {
                print("AudioRecorder: Target conversion failed: \(error.localizedDescription)")
            } else if convertedBuffer.frameLength > 0 {
                processAudioBuffer(convertedBuffer)
            } else {
                print("AudioRecorder: Target conversion produced no frames (status: \(status))")
            }
        } else {
            processAudioBuffer(sourcePCM)
        }
    }
}

final class CoreAudioMicrophoneCapture {
    var onBuffer: ((AVAudioPCMBuffer) -> Void)?

    private let callbackQueue: DispatchQueue
    private var audioQueue: AudioQueueRef?
    private var audioQueueBuffers: [AudioQueueBufferRef] = []
    private var streamDescription: AudioStreamBasicDescription?

    init(queueLabel: String) {
        self.callbackQueue = DispatchQueue(label: queueLabel)
    }

    deinit {
        stop()
    }

    var outputFormatSummary: String? {
        guard let streamDescription else { return nil }
        return "sampleRate=\(streamDescription.mSampleRate), channels=\(streamDescription.mChannelsPerFrame), commonFormat=float32, interleaved=false (AudioQueue int16 input)"
    }

    func start(deviceId: AudioDeviceID?) -> Bool {
        stop()

        let candidateFormats = [
            Self.makeStreamDescription(sampleRate: 48_000, channels: 1),
            Self.makeStreamDescription(sampleRate: 48_000, channels: 2),
        ]

        var queue: AudioQueueRef?
        var activeDescription: AudioStreamBasicDescription?

        for var candidate in candidateFormats {
            let status = AudioQueueNewInputWithDispatchQueue(&queue, &candidate, 0, callbackQueue) { [weak self] inAQ, inBuffer, _, _, _ in
                self?.handleInputBuffer(queue: inAQ, buffer: inBuffer)
            }
            if status == noErr, let queue {
                activeDescription = candidate
                self.audioQueue = queue
                break
            }
        }

        guard let audioQueue, let activeDescription else {
            print("CoreAudioMicrophoneCapture: Failed to create input queue")
            return false
        }

        if let deviceId, let deviceUID = AudioRecorder.getInputDeviceUID(deviceId) {
            var deviceUIDRef: CFString = deviceUID as CFString
            let status = withUnsafePointer(to: &deviceUIDRef) { pointer in
                AudioQueueSetProperty(
                    audioQueue,
                    kAudioQueueProperty_CurrentDevice,
                    pointer,
                    UInt32(MemoryLayout<CFString>.size)
                )
            }
            if status != noErr {
                print("CoreAudioMicrophoneCapture: Failed to bind input queue to device UID \(deviceUID): \(status)")
            }
        }

        streamDescription = activeDescription

        let framesPerBuffer = max(UInt32(activeDescription.mSampleRate / 10.0), 2_048)
        let bufferByteSize = max(framesPerBuffer * activeDescription.mBytesPerFrame, 4_096)

        for _ in 0..<4 {
            var bufferRef: AudioQueueBufferRef?
            let allocateStatus = AudioQueueAllocateBuffer(audioQueue, bufferByteSize, &bufferRef)
            guard allocateStatus == noErr, let bufferRef else {
                print("CoreAudioMicrophoneCapture: Failed to allocate input buffer: \(allocateStatus)")
                stop()
                return false
            }

            let enqueueStatus = AudioQueueEnqueueBuffer(audioQueue, bufferRef, 0, nil)
            guard enqueueStatus == noErr else {
                print("CoreAudioMicrophoneCapture: Failed to enqueue input buffer: \(enqueueStatus)")
                stop()
                return false
            }

            audioQueueBuffers.append(bufferRef)
        }

        let startStatus = AudioQueueStart(audioQueue, nil)
        guard startStatus == noErr else {
            print("CoreAudioMicrophoneCapture: Failed to start input queue: \(startStatus)")
            stop()
            return false
        }

        return true
    }

    func stop() {
        guard let audioQueue else { return }
        AudioQueueStop(audioQueue, true)
        AudioQueueDispose(audioQueue, true)
        self.audioQueue = nil
        self.audioQueueBuffers.removeAll()
        self.streamDescription = nil
    }

    private func handleInputBuffer(queue: AudioQueueRef, buffer: AudioQueueBufferRef) {
        defer {
            let status = AudioQueueEnqueueBuffer(queue, buffer, 0, nil)
            if status != noErr {
                print("CoreAudioMicrophoneCapture: Failed to re-enqueue input buffer: \(status)")
            }
        }

        let byteCount = Int(buffer.pointee.mAudioDataByteSize)
        guard byteCount > 0, let streamDescription else { return }
        let inputData = buffer.pointee.mAudioData.assumingMemoryBound(to: Int16.self)

        let channels = Int(max(streamDescription.mChannelsPerFrame, 1))
        let samplesPerBuffer = byteCount / MemoryLayout<Int16>.size
        let frameCount = samplesPerBuffer / channels
        guard frameCount > 0 else { return }

        guard let outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: streamDescription.mSampleRate,
            channels: AVAudioChannelCount(channels),
            interleaved: false
        ),
        let pcmBuffer = AVAudioPCMBuffer(
            pcmFormat: outputFormat,
            frameCapacity: AVAudioFrameCount(frameCount)
        ),
        let floatChannelData = pcmBuffer.floatChannelData else {
            return
        }

        pcmBuffer.frameLength = AVAudioFrameCount(frameCount)
        let scale = Float(Int16.max)

        for frame in 0..<frameCount {
            let baseIndex = frame * channels
            for channel in 0..<channels {
                let sample = Float(inputData[baseIndex + channel]) / scale
                floatChannelData[channel][frame] = min(max(sample, -1.0), 1.0)
            }
        }

        onBuffer?(pcmBuffer)
    }

    private static func makeStreamDescription(sampleRate: Double, channels: UInt32) -> AudioStreamBasicDescription {
        AudioStreamBasicDescription(
            mSampleRate: sampleRate,
            mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: kLinearPCMFormatFlagIsSignedInteger | kAudioFormatFlagIsPacked,
            mBytesPerPacket: channels * UInt32(MemoryLayout<Int16>.size),
            mFramesPerPacket: 1,
            mBytesPerFrame: channels * UInt32(MemoryLayout<Int16>.size),
            mChannelsPerFrame: channels,
            mBitsPerChannel: 16,
            mReserved: 0
        )
    }
}

/// Represents an audio input device from CoreAudio
struct InputDevice: Identifiable, Equatable, Hashable {
    let id: AudioDeviceID       // CoreAudio UInt32 device ID
    let name: String
    let isDefault: Bool

    var displayName: String {
        name + (isDefault ? " (Default)" : "")
    }
}
