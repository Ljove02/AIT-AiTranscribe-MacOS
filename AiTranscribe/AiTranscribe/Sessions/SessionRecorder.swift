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

 Now: both sources write at their NATIVE format with zero conversion.
 - Mic: writes at whatever format AVAudioEngine provides (e.g. 48kHz)
 - System audio: writes CMSampleBuffer data directly at native format
   (48kHz stereo from ScreenCaptureKit)

 All conversion (channel downmix, sample rate, AAC encoding) happens
 OFFLINE after recording stops, using AVMutableComposition which is
 designed for this and handles long files reliably.

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

/// Records both system audio and microphone into a single M4A file
class SessionRecorder: NSObject, ObservableObject {

    // MARK: - Published State

    @Published var isRecording = false
    @Published var duration: TimeInterval = 0
    @Published var micVolume: Float = 0
    @Published var hasSystemAudio = false
    @Published var isConverting = false

    // MARK: - Private Properties

    private let systemCapture = SystemAudioCapture()
    private var micEngine: AVAudioEngine?

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

    /// Final output format: 16kHz mono AAC
    private let outputSampleRate: Double = 16000

    // MARK: - Recording Control

    func startRecording(sessionDir: URL, micDeviceId: AudioDeviceID? = nil) async -> Bool {
        guard !isRecording else { return false }

        let m4aURL = sessionDir.appendingPathComponent("audio.m4a")
        let micURL = sessionDir.appendingPathComponent("mic_temp.caf")
        let sysURL = sessionDir.appendingPathComponent("sys_temp.caf")
        outputURL = m4aURL
        tempMicURL = micURL
        tempSysURL = sysURL

        // System audio file is created lazily from the first buffer's format
        sysFile = nil
        sysFileFormat = nil

        // Start system audio capture
        let systemStarted = await startSystemAudioCapture()
        hasSystemAudio = systemStarted

        if !systemStarted {
            print("SessionRecorder: System audio unavailable, recording mic-only")
            tempSysURL = nil
        }

        // Start microphone capture (creates micFile at native format)
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
        print("SessionRecorder: Recording started (native formats, zero conversion)")
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
        print("SessionRecorder: Mixing and converting to M4A...")

        let success: Bool
        if let sysURL = tempSysURL,
           FileManager.default.fileExists(atPath: sysURL.path) {
            success = await mixTracksToM4A(micURL: micURL, sysURL: sysURL, outputURL: m4aURL)
        } else {
            success = await convertSingleTrackToM4A(inputURL: micURL, outputURL: m4aURL)
        }
        isConverting = false

        // Clean up temp files
        try? FileManager.default.removeItem(at: micURL)
        if let sysURL = tempSysURL {
            try? FileManager.default.removeItem(at: sysURL)
        }

        if success {
            print("SessionRecorder: Recording stopped. Duration: \(SessionManager.formatDuration(duration))")
            return m4aURL
        } else {
            print("SessionRecorder: Conversion failed")
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
        return await systemCapture.startCapture()
    }

    /// Write system audio directly at native format — NO AVAudioConverter.
    /// The CMSampleBuffer data is copied to an AVAudioPCMBuffer (just memcpy)
    /// and written to file. All format conversion happens in post-processing.
    private func handleSystemAudioBuffer(_ sampleBuffer: CMSampleBuffer) {
        guard let formatDesc = sampleBuffer.formatDescription else { return }
        guard sampleBuffer.dataBuffer != nil else { return }

        let sourceFormat = AVAudioFormat(cmAudioFormatDescription: formatDesc)
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

        // Copy CMSampleBuffer data into AVAudioPCMBuffer (just a memcpy, no conversion)
        guard let pcmBuffer = AVAudioPCMBuffer(pcmFormat: sourceFormat, frameCapacity: frameCount) else {
            return
        }
        pcmBuffer.frameLength = frameCount

        guard let blockBuffer = sampleBuffer.dataBuffer else { return }
        var totalLength: Int = 0
        var dataPointer: UnsafeMutablePointer<CChar>?

        let status = CMBlockBufferGetDataPointer(
            blockBuffer, atOffset: 0, lengthAtOffsetOut: nil,
            totalLengthOut: &totalLength, dataPointerOut: &dataPointer
        )
        guard status == noErr, let dataPointer else { return }

        // Copy raw audio bytes directly — no format conversion
        let bytesToCopy = min(totalLength, Int(frameCount) * Int(sourceFormat.streamDescription.pointee.mBytesPerFrame))
        if sourceFormat.isInterleaved {
            // Interleaved: single data plane
            if let channelData = pcmBuffer.floatChannelData {
                memcpy(channelData[0], dataPointer, bytesToCopy)
            }
        } else {
            // Non-interleaved: copy each channel plane
            let channels = Int(sourceFormat.channelCount)
            let bytesPerChannel = Int(frameCount) * MemoryLayout<Float>.size
            if let channelData = pcmBuffer.floatChannelData {
                for ch in 0..<channels {
                    let offset = ch * bytesPerChannel
                    if offset + bytesPerChannel <= totalLength {
                        memcpy(channelData[ch], dataPointer.advanced(by: offset), bytesPerChannel)
                    }
                }
            }
        }

        // Write directly — no lock needed, only this callback writes to sysFile
        guard let sysFile else { return }
        do {
            try sysFile.write(from: pcmBuffer)
        } catch {
            if isRecording {
                print("SessionRecorder: Sys write error: \(error)")
            }
        }
    }

    // MARK: - Microphone (zero conversion — write at native format)

    private func startMicrophoneCapture(sessionDir: URL, deviceId: AudioDeviceID? = nil) -> Bool {
        var previousDefaultDevice: AudioDeviceID?
        if let deviceId {
            let currentDefault = AudioRecorder.getDefaultInputDeviceId()
            if currentDefault != deviceId {
                previousDefaultDevice = currentDefault
                AudioRecorder.setDefaultInputDevice(deviceId)
                usleep(50_000)
            }
        }

        micEngine = AVAudioEngine()
        guard let engine = micEngine else {
            if let prev = previousDefaultDevice { AudioRecorder.setDefaultInputDevice(prev) }
            return false
        }

        let inputNode = engine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)

        // Create mic file at the NATIVE format — no conversion needed
        guard let micURL = tempMicURL else { return false }
        do {
            micFile = try AVAudioFile(
                forWriting: micURL,
                settings: inputFormat.settings,
                commonFormat: inputFormat.commonFormat,
                interleaved: inputFormat.isInterleaved
            )
            print("SessionRecorder: Mic file created (\(inputFormat))")
        } catch {
            print("SessionRecorder: Failed to create mic audio file: \(error)")
            return false
        }

        // Install tap — write directly at native format, ZERO conversion
        inputNode.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { [weak self] buffer, _ in
            guard let self else { return }

            self.updateMicVolume(from: buffer)

            // Write directly — no converter, no format change
            guard let micFile = self.micFile else { return }
            do {
                try micFile.write(from: buffer)
            } catch {
                if self.isRecording {
                    print("SessionRecorder: Mic write error: \(error)")
                }
            }
        }

        do {
            try engine.start()
            print("SessionRecorder: Microphone capture started (native format)")
            return true
        } catch {
            print("SessionRecorder: Failed to start mic engine: \(error)")
            return false
        }
    }

    private func stopMicrophoneCapture() {
        micEngine?.inputNode.removeTap(onBus: 0)
        micEngine?.stop()
        micEngine = nil
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

    // MARK: - Post-Recording: Mix and Convert

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

            let composition = AVMutableComposition()

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
                }
            }

            // Read the composition — AudioMixOutput handles mixing + format conversion
            let reader = try AVAssetReader(asset: composition)
            let compositionTracks = try await composition.loadTracks(withMediaType: .audio)

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
