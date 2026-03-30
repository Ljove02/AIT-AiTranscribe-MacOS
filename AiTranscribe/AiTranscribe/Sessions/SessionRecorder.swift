/*
 SessionRecorder.swift
 =====================

 Records both system audio and microphone input into separate temp
 files, then mixes and converts to M4A after recording stops.

 ZERO CONVERSION DURING RECORDING:
 ----------------------------------
 Previous versions used AVAudioConverter in real-time callbacks to
 convert formats during recording. This causes state accumulation
 in the converter over long sessions (30+ min), leading to audio
 distortion and corruption.

 Now:
 - Mic: captured via AVCaptureSession (not AVAudioEngine) so it
   coexists with video call apps (Google Meet, Zoom) that activate
   Voice Processing AudioUnits. We request stable Linear PCM buffers
   from AVCapture and write them directly to disk.
 - System audio: writes CMSampleBuffer data directly at its capture
   format (typically 48kHz stereo from ScreenCaptureKit).

 Final conversion (mixing, channel downmix, sample rate, AAC encoding)
 happens OFFLINE after recording stops, using AVMutableComposition
 which is designed for this and handles long files reliably.

 WHY TWO SEPARATE FILES?
 -----------------------
 Both mic and system audio arrive on different threads. If written to
 the same file, AVAudioFile.write() appends sequentially — scrambling
 the timeline. Separate files + composition mixing solves this.

 FALLBACK:
 ---------
 If Screen Recording permission is denied, we record mic-only.
 */

import Foundation
import AVFoundation
import CoreMedia
import CoreAudio
import Combine

#if canImport(AVFAudio)
import AVFAudio
#endif

/// Records both system audio and microphone into a single M4A file
class SessionRecorder: NSObject, ObservableObject {
    private struct AudioLevelAnalysis {
        let activeRMS: Float
        let peak: Float
        let activeSamples: Int64
        let totalSamples: Int64
    }

    private struct MixVolumePlan {
        let micGain: Float
        let systemGain: Float
        let micAnalysis: AudioLevelAnalysis?
        let systemAnalysis: AudioLevelAnalysis?
    }

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

    private static func clampUnit(_ value: Float) -> Float {
        min(max(value, -1.0), 1.0)
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

    private static func commonFormatDescription(_ format: AVAudioCommonFormat) -> String {
        switch format {
        case .pcmFormatFloat32: return "float32"
        case .pcmFormatFloat64: return "float64"
        case .pcmFormatInt16: return "int16"
        case .pcmFormatInt32: return "int32"
        case .otherFormat: return "other"
        @unknown default: return "unknown"
        }
    }

    private static func formatSummary(from sampleBuffer: CMSampleBuffer, format: AVAudioFormat) -> String {
        let subtype = fourCCString(CMFormatDescriptionGetMediaSubType(sampleBuffer.formatDescription!))
        return [
            "subtype=\(subtype)",
            "sampleRate=\(format.sampleRate)",
            "channels=\(format.channelCount)",
            "commonFormat=\(commonFormatDescription(format.commonFormat))",
            "interleaved=\(format.isInterleaved)"
        ].joined(separator: ", ")
    }

    private static func formatDuration(_ seconds: TimeInterval) -> String {
        let totalSeconds = Int(seconds.rounded())
        let hours = totalSeconds / 3600
        let minutes = (totalSeconds % 3600) / 60
        let secs = totalSeconds % 60

        if hours > 0 {
            return String(format: "%d:%02d:%02d", hours, minutes, secs)
        }
        return String(format: "%d:%02d", minutes, secs)
    }

    private func copyPCMData(from sampleBuffer: CMSampleBuffer,
                             frameCount: AVAudioFrameCount,
                             into pcmBuffer: AVAudioPCMBuffer,
                             sourceName: String) -> Bool {
        let status = CMSampleBufferCopyPCMDataIntoAudioBufferList(
            sampleBuffer,
            at: 0,
            frameCount: Int32(frameCount),
            into: pcmBuffer.mutableAudioBufferList
        )
        guard status == noErr else {
            print("SessionRecorder: Failed to copy \(sourceName) PCM data: \(status)")
            return false
        }
        return true
    }

    private func applyGain(_ gain: Float, to buffer: AVAudioPCMBuffer) {
        guard gain != 1.0 else { return }

        let format = buffer.format
        let frames = Int(buffer.frameLength)
        guard frames > 0 else { return }

        let channels = max(Int(format.channelCount), 1)
        let audioBuffers = UnsafeMutableAudioBufferListPointer(buffer.mutableAudioBufferList)

        switch format.commonFormat {
        case .pcmFormatFloat32:
            if format.isInterleaved {
                guard let rawData = audioBuffers[0].mData?.assumingMemoryBound(to: Float.self) else { return }
                for index in 0..<(frames * channels) {
                    rawData[index] = Self.clampUnit(rawData[index] * gain)
                }
            } else {
                for channel in 0..<channels {
                    guard let rawData = audioBuffers[channel].mData?.assumingMemoryBound(to: Float.self) else { return }
                    for frame in 0..<frames {
                        rawData[frame] = Self.clampUnit(rawData[frame] * gain)
                    }
                }
            }
        case .pcmFormatFloat64:
            if format.isInterleaved {
                guard let rawData = audioBuffers[0].mData?.assumingMemoryBound(to: Double.self) else { return }
                for index in 0..<(frames * channels) {
                    rawData[index] = Double(Self.clampUnit(Float(rawData[index]) * gain))
                }
            } else {
                for channel in 0..<channels {
                    guard let rawData = audioBuffers[channel].mData?.assumingMemoryBound(to: Double.self) else { return }
                    for frame in 0..<frames {
                        rawData[frame] = Double(Self.clampUnit(Float(rawData[frame]) * gain))
                    }
                }
            }
        case .pcmFormatInt16:
            let maxValue = Float(Int16.max)
            let minValue = Float(Int16.min)
            if format.isInterleaved {
                guard let rawData = audioBuffers[0].mData?.assumingMemoryBound(to: Int16.self) else { return }
                for index in 0..<(frames * channels) {
                    let scaled = Float(rawData[index]) * gain
                    rawData[index] = Int16(min(max(scaled, minValue), maxValue))
                }
            } else {
                for channel in 0..<channels {
                    guard let rawData = audioBuffers[channel].mData?.assumingMemoryBound(to: Int16.self) else { return }
                    for frame in 0..<frames {
                        let scaled = Float(rawData[frame]) * gain
                        rawData[frame] = Int16(min(max(scaled, minValue), maxValue))
                    }
                }
            }
        case .pcmFormatInt32:
            let maxValue = Float(Int32.max)
            let minValue = Float(Int32.min)
            if format.isInterleaved {
                guard let rawData = audioBuffers[0].mData?.assumingMemoryBound(to: Int32.self) else { return }
                for index in 0..<(frames * channels) {
                    let scaled = Float(rawData[index]) * gain
                    rawData[index] = Int32(min(max(scaled, minValue), maxValue))
                }
            } else {
                for channel in 0..<channels {
                    guard let rawData = audioBuffers[channel].mData?.assumingMemoryBound(to: Int32.self) else { return }
                    for frame in 0..<frames {
                        let scaled = Float(rawData[frame]) * gain
                        rawData[frame] = Int32(min(max(scaled, minValue), maxValue))
                    }
                }
            }
        case .otherFormat:
            return
        @unknown default:
            return
        }
    }

    private func normalizeMicBuffer(from sampleBuffer: CMSampleBuffer) -> AVAudioPCMBuffer? {
        guard let formatDesc = sampleBuffer.formatDescription else { return nil }
        let mediaSubType = CMFormatDescriptionGetMediaSubType(formatDesc)
        guard mediaSubType == kAudioFormatLinearPCM else {
            let fourCC = Self.fourCCString(mediaSubType)
            print("SessionRecorder: Unsupported mic buffer format: \(fourCC)")
            return nil
        }
        guard let sourceFormat = Self.canonicalPCMFormat(from: formatDesc) else {
            print("SessionRecorder: Failed to canonicalize mic PCM format")
            return nil
        }

        let frameCount = AVAudioFrameCount(sampleBuffer.numSamples)
        guard frameCount > 0 else { return nil }
        guard let sourcePCM = AVAudioPCMBuffer(pcmFormat: sourceFormat, frameCapacity: frameCount) else { return nil }
        sourcePCM.frameLength = frameCount

        guard copyPCMData(from: sampleBuffer, frameCount: frameCount, into: sourcePCM, sourceName: "mic") else { return nil }

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

        switch sourceFormat.commonFormat {
        case .pcmFormatFloat32:
            if sourceFormat.isInterleaved {
                guard let rawData = sourceBuffers[0].mData?.assumingMemoryBound(to: Float.self) else { return nil }
                for frame in 0..<frames {
                    let baseIndex = frame * channels
                    for channel in 0..<channels {
                        normalizedChannelData[channel][frame] = Self.clampUnit(rawData[baseIndex + channel])
                    }
                }
            } else {
                for channel in 0..<channels {
                    guard let rawData = sourceBuffers[channel].mData?.assumingMemoryBound(to: Float.self) else { return nil }
                    for frame in 0..<frames {
                        normalizedChannelData[channel][frame] = Self.clampUnit(rawData[frame])
                    }
                }
            }
        case .pcmFormatFloat64:
            if sourceFormat.isInterleaved {
                guard let rawData = sourceBuffers[0].mData?.assumingMemoryBound(to: Double.self) else { return nil }
                for frame in 0..<frames {
                    let baseIndex = frame * channels
                    for channel in 0..<channels {
                        normalizedChannelData[channel][frame] = Self.clampUnit(Float(rawData[baseIndex + channel]))
                    }
                }
            } else {
                for channel in 0..<channels {
                    guard let rawData = sourceBuffers[channel].mData?.assumingMemoryBound(to: Double.self) else { return nil }
                    for frame in 0..<frames {
                        normalizedChannelData[channel][frame] = Self.clampUnit(Float(rawData[frame]))
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
                        normalizedChannelData[channel][frame] = Self.clampUnit(Float(rawData[baseIndex + channel]) / scale)
                    }
                }
            } else {
                for channel in 0..<channels {
                    guard let rawData = sourceBuffers[channel].mData?.assumingMemoryBound(to: Int16.self) else { return nil }
                    for frame in 0..<frames {
                        normalizedChannelData[channel][frame] = Self.clampUnit(Float(rawData[frame]) / scale)
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
                        normalizedChannelData[channel][frame] = Self.clampUnit(Float(rawData[baseIndex + channel]) / scale)
                    }
                }
            } else {
                for channel in 0..<channels {
                    guard let rawData = sourceBuffers[channel].mData?.assumingMemoryBound(to: Int32.self) else { return nil }
                    for frame in 0..<frames {
                        normalizedChannelData[channel][frame] = Self.clampUnit(Float(rawData[frame]) / scale)
                    }
                }
            }
        default:
            return nil
        }

        return normalizedPCM
    }


    // MARK: - Published State

    @Published var isRecording = false
    @Published var duration: TimeInterval = 0
    @Published var micVolume: Float = 0
    @Published var hasSystemAudio = false
    @Published var isConverting = false

    // MARK: - Private Properties

    private let systemCapture = SystemAudioCapture()
    private let micCapture = CoreAudioMicrophoneCapture(queueLabel: "com.aitranscribe.session-coreaudio-mic")

    /// Mic captured via AVCaptureSession — works alongside video call
    /// apps (Google Meet, Zoom) that use Voice Processing AudioUnits.
    /// AVAudioEngine's inputNode tap silently dies when another app
    /// activates voice processing; AVCaptureSession captures at the
    /// HAL level and is unaffected.
    private var micCaptureSession: AVCaptureSession?
    private let micCaptureQueue = DispatchQueue(label: "com.aitranscribe.mic-capture")

    /// Separate files for each audio source — no lock contention
    private var micFile: AVAudioFile?
    private var sysFile: AVAudioFile?

    /// Format of the system audio (set from first received buffer)
    private var sysFileFormat: AVAudioFormat?

    private var outputURL: URL?       // Final audio.m4a
    private var tempMicURL: URL?      // Temporary mic_temp.caf
    private var tempSysURL: URL?      // Temporary sys_temp.caf
    private var durationTimer: Timer?
    private var startTime: Date?

    /// Mic diagnostics
    private var micFramesWritten: Int64 = 0
    private var micFileCreated = false
    private var sysFramesWritten: Int64 = 0
    private var micCallbackCount: Int64 = 0
    private var micActiveBuffers: Int64 = 0
    private var micMaxRMS: Float = 0
    private var micMaxPeak: Float = 0
    private var micFormatSummary: String?
    private var micCaptureSummary: String?

    /// Adaptive mic gain — compensates for hardware gain changes
    /// (e.g. Google Meet lowering mic volume when joining a call)
    private var micGain: Float = 2.0
    private var appliedMixMicGain: Float = 1.0
    private var appliedMixSystemGain: Float = 1.0
    private var analyzedMixMicRMS: Float = 0
    private var analyzedMixMicPeak: Float = 0
    private var analyzedMixSystemRMS: Float = 0
    private var analyzedMixSystemPeak: Float = 0
    private var sessionRecordedMicGain: Float = SessionAudioMixPreferences.baselineMicGain
    private var sessionRecordedSystemGain: Float = SessionAudioMixPreferences.baselineSystemGain
    private var sessionMicTrimDB: Double = SessionAudioMixPreferences.defaultMicTrimDB
    private var sessionSystemTrimDB: Double = SessionAudioMixPreferences.defaultSystemTrimDB
    private let sessionMixPeakCeiling: Float = 0.94
    private let sessionMixFixedMicGain: Float = 1.0
    private let sessionMixFixedSystemGain: Float = 1.0

    /// Final output format: 16kHz mono AAC
    private let outputSampleRate: Double = 16000

    /// Request native Linear PCM from AVCapture. We avoid forcing float32
    /// or non-interleaved layouts because voice-processed call apps can
    /// expose telephony-style PCM that normalizes more reliably when the
    /// capture pipeline preserves the source layout first.
    private let micCaptureAudioSettings: [String: Any] = [
        AVFormatIDKey: kAudioFormatLinearPCM,
    ]

    // MARK: - Recording Control

    func startRecording(sessionDir: URL, micDeviceId: AudioDeviceID? = nil) async -> Bool {
        guard !isRecording else { return false }

        // Check microphone permission BEFORE creating capture session.
        let micPermission = AVCaptureDevice.authorizationStatus(for: .audio)
        switch micPermission {
        case .notDetermined:
            let granted = await AVCaptureDevice.requestAccess(for: .audio)
            if !granted {
                print("SessionRecorder: Microphone permission denied by user")
                return false
            }
        case .denied, .restricted:
            print("SessionRecorder: Microphone permission not granted (status: \(micPermission.rawValue))")
            return false
        case .authorized:
            break
        @unknown default:
            break
        }

        let m4aURL = sessionDir.appendingPathComponent("audio.m4a")
        let micURL = sessionDir.appendingPathComponent("mic_temp.caf")
        let sysURL = sessionDir.appendingPathComponent("sys_temp.caf")
        outputURL = m4aURL
        tempMicURL = micURL
        tempSysURL = sysURL

        // System audio file is created lazily from the first buffer's format
        sysFile = nil
        sysFileFormat = nil
        sysFramesWritten = 0
        micCallbackCount = 0
        micActiveBuffers = 0
        micMaxRMS = 0
        micMaxPeak = 0
        micFormatSummary = nil
        micCaptureSummary = nil
        appliedMixMicGain = 1.0
        appliedMixSystemGain = 1.0
        analyzedMixMicRMS = 0
        analyzedMixMicPeak = 0
        analyzedMixSystemRMS = 0
        analyzedMixSystemPeak = 0
        sessionMicTrimDB = SessionAudioMixPreferences.micTrimDB()
        sessionSystemTrimDB = SessionAudioMixPreferences.systemTrimDB()
        sessionRecordedMicGain = SessionAudioMixPreferences.effectiveMicGain()
        sessionRecordedSystemGain = SessionAudioMixPreferences.effectiveSystemGain()

        // System audio capture via ScreenCaptureKit.
        // Only attempt if Screen Recording permission is already granted.
        // ScreenCaptureKit can hard-crash on unsigned/quarantined apps,
        // so we check permission first and skip entirely if not granted.
        var systemStarted = false
        let hasScreenPermission = SystemAudioCapture.preflightPermission()
        if hasScreenPermission {
            systemStarted = await startSystemAudioCapture()
        } else {
            print("SessionRecorder: Screen Recording permission not granted — skipping system audio")
        }
        hasSystemAudio = systemStarted

        if !systemStarted {
            print("SessionRecorder: Recording mic-only")
            tempSysURL = nil
        }

        // Start microphone capture (creates micFile from the first PCM buffer)
        micFramesWritten = 0
        let micStarted = startMicrophoneCapture(sessionDir: sessionDir, deviceId: micDeviceId)
        if !micStarted {
            print("SessionRecorder: Failed to start microphone — aborting")
            await systemCapture.stopCapture()
            micFile = nil
            sysFile = nil
            return false
        }

        // Duration timer — use .common mode so it fires even when menu bar is open
        startTime = Date()
        duration = 0

        let timer = Timer(timeInterval: 1.0, repeats: true) { [weak self] _ in
            guard let self, let start = self.startTime else { return }
            Task { @MainActor in
                self.duration = Date().timeIntervalSince(start)
            }
        }
        RunLoop.main.add(timer, forMode: .common)
        durationTimer = timer

        isRecording = true
        print("SessionRecorder: Recording started")
        return true
    }

    func stopRecording() async -> URL? {
        guard isRecording else { return nil }

        durationTimer?.invalidate()
        durationTimer = nil

        stopMicrophoneCapture()
        await systemCapture.stopCapture()

        // Close both files
        micFile = nil
        sysFile = nil

        isRecording = false
        hasSystemAudio = false
        micVolume = 0

        if let start = startTime {
            duration = Date().timeIntervalSince(start)
        }
        startTime = nil

        // Mix and convert to M4A
        guard let micURL = tempMicURL, let m4aURL = outputURL else { return nil }

        isConverting = true
        print("SessionRecorder: Finalizing audio (mic frames: \(micFramesWritten), system frames: \(sysFramesWritten))...")

        let success = await exportRecordedAudio(micURL: micURL, sysURL: tempSysURL, outputURL: m4aURL)
        isConverting = false

        if success {
            let outputSize = fileSize(at: m4aURL)
            if micActiveBuffers == 0 || micMaxPeak < 0.001 || outputSize < 10_000 {
                writeLowAudioDiagnostics(outputURL: m4aURL)
            }
            try? FileManager.default.removeItem(at: micURL)
            if let sysURL = tempSysURL {
                try? FileManager.default.removeItem(at: sysURL)
            }
            print("SessionRecorder: Recording stopped. Duration: \(Self.formatDuration(duration))")
            return m4aURL
        } else {
            removeExistingFile(at: m4aURL)
            writeFailureDiagnostics(micURL: micURL, sysURL: tempSysURL)
            print("SessionRecorder: Conversion failed; preserved temp files for diagnostics")
            return nil
        }
    }

    func cancelRecording() async {
        guard isRecording else { return }

        durationTimer?.invalidate()
        durationTimer = nil
        stopMicrophoneCapture()
        await systemCapture.stopCapture()

        micFile = nil
        sysFile = nil

        if let url = tempMicURL { try? FileManager.default.removeItem(at: url) }
        if let url = tempSysURL { try? FileManager.default.removeItem(at: url) }
        if let url = outputURL { try? FileManager.default.removeItem(at: url) }

        isRecording = false
        hasSystemAudio = false
        micVolume = 0
        duration = 0
        startTime = nil

        print("SessionRecorder: Recording cancelled")
    }

    // MARK: - System Audio (zero conversion — write at native stereo format)

    private func startSystemAudioCapture() async -> Bool {
        systemCapture.onAudioBuffer = { [weak self] sampleBuffer in
            self?.handleSystemAudioBuffer(sampleBuffer)
        }
        systemCapture.onMicrophoneBuffer = nil
        return await Task {
            await systemCapture.startCapture(
                captureSystemAudio: true,
                captureMicrophone: false,
                microphoneCaptureDeviceID: nil
            )
        }.value
    }

    /// Write system audio directly at native format — NO AVAudioConverter.
    /// The CMSampleBuffer data is copied to an AVAudioPCMBuffer (just memcpy)
    /// and written to file. All format conversion happens in post-processing.
    private func handleSystemAudioBuffer(_ sampleBuffer: CMSampleBuffer) {
        guard let formatDesc = sampleBuffer.formatDescription else { return }
        guard sampleBuffer.dataBuffer != nil else { return }

        guard let sourceFormat = Self.canonicalPCMFormat(from: formatDesc) else {
            print("SessionRecorder: Failed to canonicalize system audio format")
            return
        }
        let frameCount = AVAudioFrameCount(sampleBuffer.numSamples)
        guard frameCount > 0 else { return }

        // Create sysFile lazily from the first buffer's actual format
        if sysFile == nil, let sysURL = tempSysURL {
            do {
                sysFile = try AVAudioFile(
                    forWriting: sysURL,
                    settings: sourceFormat.settings,
                    commonFormat: sourceFormat.commonFormat,
                    interleaved: sourceFormat.isInterleaved
                )
                sysFileFormat = sourceFormat
                print("SessionRecorder: System audio file created (\(sourceFormat))")
            } catch {
                print("SessionRecorder: Failed to create sys audio file: \(error)")
                return
            }
        }

        // Copy PCM samples into an AVAudioPCMBuffer without format conversion.
        guard let pcmBuffer = AVAudioPCMBuffer(pcmFormat: sourceFormat, frameCapacity: frameCount) else {
            return
        }
        pcmBuffer.frameLength = frameCount

        guard copyPCMData(from: sampleBuffer, frameCount: frameCount, into: pcmBuffer, sourceName: "system") else {
            return
        }
        applyGain(sessionRecordedSystemGain, to: pcmBuffer)

        // Write directly — no lock needed, only this callback writes to sysFile
        guard let sysFile else { return }
        do {
            try sysFile.write(from: pcmBuffer)
            sysFramesWritten += Int64(frameCount)
        } catch {
            if isRecording {
                print("SessionRecorder: Sys write error: \(error)")
            }
        }
    }

    // MARK: - Microphone (AVCaptureSession — coexists with video calls)

    private func startMicrophoneCapture(sessionDir: URL, deviceId: AudioDeviceID? = nil) -> Bool {
        _ = sessionDir

        micFramesWritten = 0
        micFileCreated = false
        micCapture.onBuffer = { [weak self] buffer in
            self?.handleMicPCMBuffer(buffer)
        }

        guard micCapture.start(deviceId: deviceId) else {
            print("SessionRecorder: Failed to start CoreAudio microphone capture")
            return false
        }
        micCaptureSession = nil
        micFormatSummary = micCapture.outputFormatSummary

        if let deviceId,
           let micDevice = AudioRecorder.captureDevice(for: deviceId),
           let coreAudioUID = AudioRecorder.getInputDeviceUID(deviceId) {
            let coreAudioName = AudioRecorder.getInputDeviceName(deviceId) ?? "unknown"
            let summary = "CoreAudio AudioQueue: \(coreAudioName) [\(deviceId)] / UID: \(coreAudioUID), capture: \(micDevice.localizedName) / \(micDevice.uniqueID)"
            micCaptureSummary = summary
            print("SessionRecorder: Mic capture started via CoreAudio AudioQueue (\(summary))")
        } else if let micDevice = AudioRecorder.captureDevice(for: deviceId) {
            let summary = "CoreAudio AudioQueue capture: \(micDevice.localizedName) / \(micDevice.uniqueID)"
            micCaptureSummary = summary
            print("SessionRecorder: Mic capture started via CoreAudio AudioQueue (\(summary))")
        } else {
            micCaptureSummary = "CoreAudio AudioQueue system default microphone"
            print("SessionRecorder: Mic capture started via CoreAudio AudioQueue (system default microphone)")
        }
        return true
    }

    /// Write mic audio from AVCaptureSession. The capture output is
    /// configured to deliver Linear PCM so this path never depends on
    /// device-specific compressed/native mic formats.
    private func handleMicBuffer(_ sampleBuffer: CMSampleBuffer) {
        guard isRecording else { return }
        guard let pcmBuffer = normalizeMicBuffer(from: sampleBuffer) else { return }
        if micFormatSummary == nil {
            micFormatSummary = Self.formatSummary(from: sampleBuffer, format: pcmBuffer.format)
        }
        handleMicPCMBuffer(pcmBuffer)
    }

    private func handleMicPCMBuffer(_ pcmBuffer: AVAudioPCMBuffer) {
        guard isRecording else { return }
        let sourceFormat = pcmBuffer.format
        let frameCount = pcmBuffer.frameLength
        micCallbackCount += 1

        if micFormatSummary == nil {
            micFormatSummary = [
                "sampleRate=\(sourceFormat.sampleRate)",
                "channels=\(sourceFormat.channelCount)",
                "commonFormat=float32",
                "interleaved=\(sourceFormat.isInterleaved)"
            ].joined(separator: ", ")
        }

        if !micFileCreated, let micURL = tempMicURL {
            do {
                micFile = try AVAudioFile(
                    forWriting: micURL,
                    settings: sourceFormat.settings,
                    commonFormat: sourceFormat.commonFormat,
                    interleaved: sourceFormat.isInterleaved
                )
                micFileCreated = true
                print("SessionRecorder: Mic file created from first buffer (\(sourceFormat))")
            } catch {
                print("SessionRecorder: Failed to create mic file: \(error)")
                return
            }
        }

        // Adaptive mic gain — compensates for hardware gain changes
        // (e.g. Google Meet lowering mic volume when joining a call).
        // Targets a comfortable speech RMS (~0.05). Gain adjusts smoothly
        // to avoid pumping artifacts: fast attack, slow release.
        if let channelData = pcmBuffer.floatChannelData {
            let frames = Int(pcmBuffer.frameLength)
            let channels = Int(sourceFormat.channelCount)

            // Calculate RMS from first channel
            var sum: Float = 0
            var peak: Float = 0
            for i in 0..<frames {
                let s = channelData[0][i]
                sum += s * s
                peak = max(peak, abs(s))
            }
            let rms = sqrt(sum / Float(max(frames, 1)))
            micMaxRMS = max(micMaxRMS, rms)
            micMaxPeak = max(micMaxPeak, peak)
            if rms > 0.0005 || peak > 0.001 {
                micActiveBuffers += 1
            }

            // Adjust gain toward target RMS (only when speech detected)
            let targetRMS: Float = 0.05
            if rms > 0.002 {
                let desiredGain = min(targetRMS / rms, 10.0)
                let alpha: Float = desiredGain > micGain ? 0.3 : 0.05
                micGain += alpha * (desiredGain - micGain)
            }
            micGain = max(micGain, 1.0) // Never go below unity

            for ch in 0..<channels {
                for i in 0..<frames {
                    channelData[ch][i] = min(max(channelData[ch][i] * micGain, -1.0), 1.0)
                }
            }
        }

        applyGain(sessionRecordedMicGain, to: pcmBuffer)

        // Update volume meter
        updateMicVolume(from: pcmBuffer)

        // Write directly — only this callback writes to micFile
        guard let micFile else { return }
        do {
            try micFile.write(from: pcmBuffer)
            micFramesWritten += Int64(frameCount)
        } catch {
            if isRecording {
                print("SessionRecorder: Mic write error: \(error)")
            }
        }
    }

    private func stopMicrophoneCapture() {
        micCaptureSession?.stopRunning()
        micCaptureSession = nil
        micCapture.stop()
        micFileCreated = false
        print("SessionRecorder: Mic stopped (total frames written: \(micFramesWritten))")
    }

    private func updateMicVolume(from buffer: AVAudioPCMBuffer) {
        guard let channelData = buffer.floatChannelData?[0] else { return }
        let frameCount = Int(buffer.frameLength)
        guard frameCount > 0 else { return }

        var sum: Float = 0
        for i in 0..<frameCount {
            let sample = channelData[i]
            sum += sample * sample
        }
        let rms = sqrt(sum / Float(frameCount))

        let noiseGateThreshold: Float = 0.008
        let gatedRMS = rms > noiseGateThreshold ? rms : 0
        let scaledVolume = min(gatedRMS * 3.0, 1.0)

        Task { @MainActor in
            self.micVolume = scaledVolume
        }
    }

    private func exportRecordedAudio(micURL: URL, sysURL: URL?, outputURL: URL) async -> Bool {
        let fileManager = FileManager.default
        let hasMicAudio = micFramesWritten > 0 && fileManager.fileExists(atPath: micURL.path)
        let hasSystemAudio = {
            guard let sysURL else { return false }
            return sysFramesWritten > 0 && fileManager.fileExists(atPath: sysURL.path)
        }()

        if hasMicAudio, let sysURL, hasSystemAudio {
            removeExistingFile(at: outputURL)
            if await mixTracksToM4A(micURL: micURL, sysURL: sysURL, outputURL: outputURL) {
                return true
            }
            print("SessionRecorder: Mixed export failed; falling back to single-track export")
        }

        if hasMicAudio {
            removeExistingFile(at: outputURL)
            if await convertSingleTrackToM4A(inputURL: micURL, outputURL: outputURL) {
                return true
            }
            print("SessionRecorder: Mic-only export failed")
        }

        if hasSystemAudio, let sysURL {
            removeExistingFile(at: outputURL)
            if await convertSingleTrackToM4A(inputURL: sysURL, outputURL: outputURL) {
                return true
            }
            print("SessionRecorder: System-only export failed")
        }

        return false
    }

    private func removeExistingFile(at url: URL) {
        let fileManager = FileManager.default
        guard fileManager.fileExists(atPath: url.path) else { return }
        try? fileManager.removeItem(at: url)
    }

    private func fileSize(at url: URL) -> Int64 {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: url.path) else {
            return 0
        }
        return (attrs[.size] as? Int64) ?? 0
    }

    private func writeFailureDiagnostics(micURL: URL, sysURL: URL?) {
        let sessionDir = micURL.deletingLastPathComponent()
        let diagnosticsURL = sessionDir.appendingPathComponent("recording_failure.txt")

        var lines = [
            "micCapture=\(micCaptureSummary ?? "unknown")",
            "micFormat=\(micFormatSummary ?? "unknown")",
            "micCallbackCount=\(micCallbackCount)",
            "micActiveBuffers=\(micActiveBuffers)",
            "micMaxRMS=\(micMaxRMS)",
            "micMaxPeak=\(micMaxPeak)",
            "sessionMicTrimDB=\(sessionMicTrimDB)",
            "sessionSystemTrimDB=\(sessionSystemTrimDB)",
            "sessionRecordedMicGain=\(sessionRecordedMicGain)",
            "sessionRecordedSystemGain=\(sessionRecordedSystemGain)",
            "mixMicRMS=\(analyzedMixMicRMS)",
            "mixMicPeak=\(analyzedMixMicPeak)",
            "mixSystemRMS=\(analyzedMixSystemRMS)",
            "mixSystemPeak=\(analyzedMixSystemPeak)",
            "mixMicGain=\(appliedMixMicGain)",
            "mixSystemGain=\(appliedMixSystemGain)",
            "micFramesWritten=\(micFramesWritten)",
            "sysFramesWritten=\(sysFramesWritten)",
            "micExists=\(FileManager.default.fileExists(atPath: micURL.path))",
            "micSize=\(fileSize(at: micURL))",
        ]

        if let sysURL {
            lines.append("sysExists=\(FileManager.default.fileExists(atPath: sysURL.path))")
            lines.append("sysSize=\(fileSize(at: sysURL))")
        } else {
            lines.append("sysExists=false")
            lines.append("sysSize=0")
        }

        let contents = lines.joined(separator: "\n") + "\n"
        try? contents.write(to: diagnosticsURL, atomically: true, encoding: .utf8)
    }

    private func writeLowAudioDiagnostics(outputURL: URL) {
        let sessionDir = outputURL.deletingLastPathComponent()
        let diagnosticsURL = sessionDir.appendingPathComponent("capture_debug.txt")
        let lines = [
            "micCapture=\(micCaptureSummary ?? "unknown")",
            "micFormat=\(micFormatSummary ?? "unknown")",
            "micCallbackCount=\(micCallbackCount)",
            "micActiveBuffers=\(micActiveBuffers)",
            "micMaxRMS=\(micMaxRMS)",
            "micMaxPeak=\(micMaxPeak)",
            "sessionMicTrimDB=\(sessionMicTrimDB)",
            "sessionSystemTrimDB=\(sessionSystemTrimDB)",
            "sessionRecordedMicGain=\(sessionRecordedMicGain)",
            "sessionRecordedSystemGain=\(sessionRecordedSystemGain)",
            "mixMicRMS=\(analyzedMixMicRMS)",
            "mixMicPeak=\(analyzedMixMicPeak)",
            "mixSystemRMS=\(analyzedMixSystemRMS)",
            "mixSystemPeak=\(analyzedMixSystemPeak)",
            "mixMicGain=\(appliedMixMicGain)",
            "mixSystemGain=\(appliedMixSystemGain)",
            "micFramesWritten=\(micFramesWritten)",
            "sysFramesWritten=\(sysFramesWritten)",
            "outputExists=\(FileManager.default.fileExists(atPath: outputURL.path))",
            "outputSize=\(fileSize(at: outputURL))",
        ]
        let contents = lines.joined(separator: "\n") + "\n"
        try? contents.write(to: diagnosticsURL, atomically: true, encoding: .utf8)
    }

    // MARK: - Post-Recording: Mix and Convert

    private func enumerateMonoSamples(in buffer: AVAudioPCMBuffer, _ body: (Float) -> Void) {
        let format = buffer.format
        let frames = Int(buffer.frameLength)
        guard frames > 0 else { return }

        let channels = max(Int(format.channelCount), 1)
        let sourceBuffers = UnsafeMutableAudioBufferListPointer(buffer.mutableAudioBufferList)

        switch format.commonFormat {
        case .pcmFormatFloat32:
            if format.isInterleaved {
                guard let rawData = sourceBuffers[0].mData?.assumingMemoryBound(to: Float.self) else { return }
                for frame in 0..<frames {
                    let baseIndex = frame * channels
                    var mono: Float = 0
                    for channel in 0..<channels {
                        mono += rawData[baseIndex + channel]
                    }
                    body(Self.clampUnit(mono / Float(channels)))
                }
            } else {
                let channelPointers = (0..<channels).compactMap {
                    sourceBuffers[$0].mData?.assumingMemoryBound(to: Float.self)
                }
                guard channelPointers.count == channels else { return }
                for frame in 0..<frames {
                    var mono: Float = 0
                    for pointer in channelPointers {
                        mono += pointer[frame]
                    }
                    body(Self.clampUnit(mono / Float(channels)))
                }
            }
        case .pcmFormatFloat64:
            if format.isInterleaved {
                guard let rawData = sourceBuffers[0].mData?.assumingMemoryBound(to: Double.self) else { return }
                for frame in 0..<frames {
                    let baseIndex = frame * channels
                    var mono: Float = 0
                    for channel in 0..<channels {
                        mono += Float(rawData[baseIndex + channel])
                    }
                    body(Self.clampUnit(mono / Float(channels)))
                }
            } else {
                let channelPointers = (0..<channels).compactMap {
                    sourceBuffers[$0].mData?.assumingMemoryBound(to: Double.self)
                }
                guard channelPointers.count == channels else { return }
                for frame in 0..<frames {
                    var mono: Float = 0
                    for pointer in channelPointers {
                        mono += Float(pointer[frame])
                    }
                    body(Self.clampUnit(mono / Float(channels)))
                }
            }
        case .pcmFormatInt16:
            let scale = Float(Int16.max)
            if format.isInterleaved {
                guard let rawData = sourceBuffers[0].mData?.assumingMemoryBound(to: Int16.self) else { return }
                for frame in 0..<frames {
                    let baseIndex = frame * channels
                    var mono: Float = 0
                    for channel in 0..<channels {
                        mono += Float(rawData[baseIndex + channel]) / scale
                    }
                    body(Self.clampUnit(mono / Float(channels)))
                }
            } else {
                let channelPointers = (0..<channels).compactMap {
                    sourceBuffers[$0].mData?.assumingMemoryBound(to: Int16.self)
                }
                guard channelPointers.count == channels else { return }
                for frame in 0..<frames {
                    var mono: Float = 0
                    for pointer in channelPointers {
                        mono += Float(pointer[frame]) / scale
                    }
                    body(Self.clampUnit(mono / Float(channels)))
                }
            }
        case .pcmFormatInt32:
            let scale = Float(Int32.max)
            if format.isInterleaved {
                guard let rawData = sourceBuffers[0].mData?.assumingMemoryBound(to: Int32.self) else { return }
                for frame in 0..<frames {
                    let baseIndex = frame * channels
                    var mono: Float = 0
                    for channel in 0..<channels {
                        mono += Float(rawData[baseIndex + channel]) / scale
                    }
                    body(Self.clampUnit(mono / Float(channels)))
                }
            } else {
                let channelPointers = (0..<channels).compactMap {
                    sourceBuffers[$0].mData?.assumingMemoryBound(to: Int32.self)
                }
                guard channelPointers.count == channels else { return }
                for frame in 0..<frames {
                    var mono: Float = 0
                    for pointer in channelPointers {
                        mono += Float(pointer[frame]) / scale
                    }
                    body(Self.clampUnit(mono / Float(channels)))
                }
            }
        case .otherFormat:
            return
        @unknown default:
            return
        }
    }

    private func analyzeAudioLevels(at url: URL) -> AudioLevelAnalysis? {
        do {
            let audioFile = try AVAudioFile(forReading: url)
            let chunkSize: AVAudioFrameCount = 8192
            guard let buffer = AVAudioPCMBuffer(pcmFormat: audioFile.processingFormat, frameCapacity: chunkSize) else {
                return nil
            }

            var totalSamples: Int64 = 0
            var activeSamples: Int64 = 0
            var totalSumSquares: Double = 0
            var activeSumSquares: Double = 0
            var peak: Float = 0
            let activityThreshold: Float = 0.004

            while true {
                try audioFile.read(into: buffer, frameCount: chunkSize)
                let frames = Int(buffer.frameLength)
                if frames == 0 {
                    break
                }

                enumerateMonoSamples(in: buffer) { sample in
                    let magnitude = abs(sample)
                    let square = Double(sample * sample)
                    totalSamples += 1
                    totalSumSquares += square
                    peak = max(peak, magnitude)

                    if magnitude >= activityThreshold {
                        activeSamples += 1
                        activeSumSquares += square
                    }
                }
            }

            guard totalSamples > 0 else { return nil }
            let rmsSamples = max(activeSamples, 1)
            let rmsSumSquares = activeSamples > 0 ? activeSumSquares : totalSumSquares
            let activeRMS = sqrt(rmsSumSquares / Double(rmsSamples))

            return AudioLevelAnalysis(
                activeRMS: Float(activeRMS),
                peak: peak,
                activeSamples: activeSamples,
                totalSamples: totalSamples
            )
        } catch {
            print("SessionRecorder: Failed to analyze levels for \(url.lastPathComponent): \(error)")
            return nil
        }
    }

    private func determineMixVolumePlan(micURL: URL, sysURL: URL) -> MixVolumePlan {
        let micAnalysis = analyzeAudioLevels(at: micURL)
        let systemAnalysis = analyzeAudioLevels(at: sysURL)

        analyzedMixMicRMS = micAnalysis?.activeRMS ?? 0
        analyzedMixMicPeak = micAnalysis?.peak ?? 0
        analyzedMixSystemRMS = systemAnalysis?.activeRMS ?? 0
        analyzedMixSystemPeak = systemAnalysis?.peak ?? 0

        var micGain: Float = sessionMixFixedMicGain
        var systemGain: Float = systemAnalysis == nil ? 1.0 : sessionMixFixedSystemGain

        if let micAnalysis, let systemAnalysis {
            let combinedPeak = (micAnalysis.peak * micGain) + (systemAnalysis.peak * systemGain)
            if combinedPeak > sessionMixPeakCeiling, systemAnalysis.peak > 0.001 {
                let allowableSystemPeak = max(sessionMixPeakCeiling - (micAnalysis.peak * micGain), 0)
                let duckedSystemGain = allowableSystemPeak / systemAnalysis.peak
                systemGain = max(0.25, min(systemGain, duckedSystemGain))
            }

            let adjustedCombinedPeak = (micAnalysis.peak * micGain) + (systemAnalysis.peak * systemGain)
            if adjustedCombinedPeak > sessionMixPeakCeiling {
                let safeMicGain = max(
                    1.0,
                    (sessionMixPeakCeiling - (systemAnalysis.peak * systemGain)) / max(micAnalysis.peak, 0.001)
                )
                micGain = min(micGain, safeMicGain)
            }
        }

        appliedMixMicGain = micGain
        appliedMixSystemGain = systemGain

        let micRMSDescription = String(format: "%.4f", analyzedMixMicRMS)
        let micPeakDescription = String(format: "%.4f", analyzedMixMicPeak)
        let systemRMSDescription = String(format: "%.4f", analyzedMixSystemRMS)
        let systemPeakDescription = String(format: "%.4f", analyzedMixSystemPeak)
        let micGainDescription = String(format: "%.2f", micGain)
        let systemGainDescription = String(format: "%.2f", systemGain)
        print(
            "SessionRecorder: Mix loudness analysis micRMS=\(micRMSDescription) " +
            "micPeak=\(micPeakDescription) sysRMS=\(systemRMSDescription) " +
            "sysPeak=\(systemPeakDescription) -> micGain=\(micGainDescription) " +
            "sysGain=\(systemGainDescription)"
        )

        return MixVolumePlan(
            micGain: micGain,
            systemGain: systemGain,
            micAnalysis: micAnalysis,
            systemAnalysis: systemAnalysis
        )
    }

    /// Mix mic + system audio tracks and convert to M4A (AAC 16kHz mono).
    /// AVMutableComposition layers both tracks at the same start time.
    /// AVAssetReaderAudioMixOutput handles mixing, channel conversion,
    /// and sample rate conversion — all offline, no real-time pressure.
    private func mixTracksToM4A(micURL: URL, sysURL: URL, outputURL: URL) async -> Bool {
        do {
            let micAsset = AVURLAsset(url: micURL)
            let sysAsset = AVURLAsset(url: sysURL)

            let micTracks = try await micAsset.loadTracks(withMediaType: .audio)
            let sysTracks = try await sysAsset.loadTracks(withMediaType: .audio)

            guard let micTrack = micTracks.first else {
                print("SessionRecorder: No mic track found")
                return false
            }

            let mixVolumePlan = determineMixVolumePlan(micURL: micURL, sysURL: sysURL)
            let composition = AVMutableComposition()
            var mixParameters: [AVMutableAudioMixInputParameters] = []

            let micDuration = try await micAsset.load(.duration)
            if let compTrack = composition.addMutableTrack(
                withMediaType: .audio,
                preferredTrackID: kCMPersistentTrackID_Invalid
            ) {
                try compTrack.insertTimeRange(
                    CMTimeRange(start: .zero, duration: micDuration),
                    of: micTrack,
                    at: .zero
                )
                let parameters = AVMutableAudioMixInputParameters(track: compTrack)
                parameters.setVolume(mixVolumePlan.micGain, at: .zero)
                mixParameters.append(parameters)
            }

            if let sysTrack = sysTracks.first {
                let sysDuration = try await sysAsset.load(.duration)
                if let compTrack = composition.addMutableTrack(
                    withMediaType: .audio,
                    preferredTrackID: kCMPersistentTrackID_Invalid
                ) {
                    try compTrack.insertTimeRange(
                        CMTimeRange(start: .zero, duration: sysDuration),
                        of: sysTrack,
                        at: .zero
                    )
                    let parameters = AVMutableAudioMixInputParameters(track: compTrack)
                    parameters.setVolume(mixVolumePlan.systemGain, at: .zero)
                    mixParameters.append(parameters)
                }
            }

            // Read the composition — AudioMixOutput handles mixing + format conversion
            let reader = try AVAssetReader(asset: composition)
            let compositionTracks = try await composition.loadTracks(withMediaType: .audio)
            let audioMix = AVMutableAudioMix()
            audioMix.inputParameters = mixParameters

            let readerOutput = AVAssetReaderAudioMixOutput(
                audioTracks: compositionTracks,
                audioSettings: [
                    AVFormatIDKey: kAudioFormatLinearPCM,
                    AVSampleRateKey: outputSampleRate,
                    AVNumberOfChannelsKey: 1,
                    AVLinearPCMBitDepthKey: 32,
                    AVLinearPCMIsFloatKey: true,
                    AVLinearPCMIsNonInterleaved: false,
                ]
            )
            readerOutput.audioMix = audioMix
            reader.add(readerOutput)

            // Write as M4A AAC
            let writer = try AVAssetWriter(outputURL: outputURL, fileType: .m4a)
            let writerInput = AVAssetWriterInput(mediaType: .audio, outputSettings: [
                AVFormatIDKey: kAudioFormatMPEG4AAC,
                AVSampleRateKey: outputSampleRate,
                AVNumberOfChannelsKey: 1,
                AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue,
            ])
            writerInput.expectsMediaDataInRealTime = false
            writer.add(writerInput)

            reader.startReading()
            writer.startWriting()
            writer.startSession(atSourceTime: .zero)

            return await withCheckedContinuation { continuation in
                let queue = DispatchQueue(label: "com.aitranscribe.audiomix")
                writerInput.requestMediaDataWhenReady(on: queue) {
                    while writerInput.isReadyForMoreMediaData {
                        if let sampleBuffer = readerOutput.copyNextSampleBuffer() {
                            writerInput.append(sampleBuffer)
                        } else {
                            writerInput.markAsFinished()
                            writer.finishWriting {
                                let success = writer.status == .completed
                                if !success {
                                    print("SessionRecorder: Writer error: \(writer.error?.localizedDescription ?? "unknown")")
                                }
                                continuation.resume(returning: success)
                            }
                            return
                        }
                    }
                }
            }

        } catch {
            print("SessionRecorder: Mix error: \(error)")
            return false
        }
    }

    /// Convert a single track (mic-only fallback) to M4A.
    private func convertSingleTrackToM4A(inputURL: URL, outputURL: URL) async -> Bool {
        let asset = AVURLAsset(url: inputURL)

        let audioTrack: AVAssetTrack
        do {
            guard let track = try await asset.loadTracks(withMediaType: .audio).first else {
                print("SessionRecorder: No audio track in file")
                return false
            }
            audioTrack = track
        } catch {
            print("SessionRecorder: Failed to load tracks: \(error)")
            return false
        }

        guard let reader = try? AVAssetReader(asset: asset) else {
            print("SessionRecorder: Failed to create asset reader")
            return false
        }

        let readerOutput = AVAssetReaderTrackOutput(track: audioTrack, outputSettings: [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: outputSampleRate,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 32,
            AVLinearPCMIsFloatKey: true,
            AVLinearPCMIsNonInterleaved: false,
        ])
        reader.add(readerOutput)

        guard let writer = try? AVAssetWriter(outputURL: outputURL, fileType: .m4a) else {
            print("SessionRecorder: Failed to create asset writer")
            return false
        }

        let writerInput = AVAssetWriterInput(mediaType: .audio, outputSettings: [
            AVFormatIDKey: kAudioFormatMPEG4AAC,
            AVSampleRateKey: outputSampleRate,
            AVNumberOfChannelsKey: 1,
            AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue,
        ])
        writerInput.expectsMediaDataInRealTime = false
        writer.add(writerInput)

        reader.startReading()
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)

        return await withCheckedContinuation { continuation in
            let queue = DispatchQueue(label: "com.aitranscribe.audioconvert")
            writerInput.requestMediaDataWhenReady(on: queue) {
                while writerInput.isReadyForMoreMediaData {
                    if let sampleBuffer = readerOutput.copyNextSampleBuffer() {
                        writerInput.append(sampleBuffer)
                    } else {
                        writerInput.markAsFinished()
                        writer.finishWriting {
                            let success = writer.status == .completed
                            if !success {
                                print("SessionRecorder: Writer error: \(writer.error?.localizedDescription ?? "unknown")")
                            }
                            continuation.resume(returning: success)
                        }
                        return
                    }
                }
            }
        }
    }
}

// MARK: - AVCaptureSession Mic Delegate

extension SessionRecorder: AVCaptureAudioDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        handleMicBuffer(sampleBuffer)
    }
}
