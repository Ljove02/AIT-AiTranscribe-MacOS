import AVFoundation
import CoreMedia
import Foundation

final class QuickSystemAudioRecorder {
    private(set) var isRecording = false
    private(set) var duration: TimeInterval = 0

    private let capture = SystemAudioCapture()
    private let lock = NSLock()
    private var samples: [Float] = []
    private var sourceSampleRate: Double?
    private var durationTimer: Timer?
    private var startTime: Date?

    func startRecording() async -> Bool {
        guard !isRecording else { return false }
        guard SystemAudioCapture.preflightPermission() else {
            AppLogger.warning("QuickSystemAudioRecorder start skipped reason=screen-recording-permission-missing")
            return false
        }

        lock.withLock {
            samples.removeAll(keepingCapacity: true)
            sourceSampleRate = nil
        }

        capture.onAudioBuffer = { [weak self] sampleBuffer in
            self?.append(sampleBuffer)
        }
        capture.onMicrophoneBuffer = nil

        let started = await capture.startCapture(
            captureSystemAudio: true,
            captureMicrophone: false,
            microphoneCaptureDeviceID: nil
        )
        guard started else { return false }

        await MainActor.run {
            self.startTime = Date()
            self.duration = 0
            self.isRecording = true
            let timer = Timer(timeInterval: 0.25, repeats: true) { [weak self] _ in
                guard let self, let start = self.startTime else { return }
                Task { @MainActor in
                    self.duration = Date().timeIntervalSince(start)
                }
            }
            RunLoop.main.add(timer, forMode: .common)
            self.durationTimer = timer
        }

        AppLogger.info("QuickSystemAudioRecorder started")
        return true
    }

    func stopRecording() async -> QuickSystemAudioRecording? {
        guard isRecording else { return nil }
        await capture.stopCapture()

        let finalDuration: TimeInterval = await MainActor.run {
            durationTimer?.invalidate()
            durationTimer = nil
            let elapsed = startTime.map { Date().timeIntervalSince($0) } ?? duration
            duration = elapsed
            startTime = nil
            isRecording = false
            return elapsed
        }

        let captured = lock.withLock {
            let capturedSamples = samples
            let sampleRate = sourceSampleRate ?? 48_000
            samples.removeAll(keepingCapacity: false)
            sourceSampleRate = nil
            return (capturedSamples, sampleRate)
        }
        let capturedSamples = captured.0
        let sampleRate = captured.1

        guard capturedSamples.count > Int(sampleRate * 0.35),
              let wavData = Self.makeWAVData(
                monoSamples: capturedSamples,
                sourceSampleRate: sampleRate,
                targetSampleRate: 16_000
              ) else {
            AppLogger.warning(String(
                format: "QuickSystemAudioRecorder stopped with no usable audio duration=%.2fs samples=%d",
                finalDuration,
                capturedSamples.count
            ))
            return nil
        }

        AppLogger.info(String(
            format: "QuickSystemAudioRecorder stopped duration=%.2fs samples=%d wavBytes=%d",
            finalDuration,
            capturedSamples.count,
            wavData.count
        ))
        return QuickSystemAudioRecording(audioData: wavData, durationSeconds: finalDuration)
    }

    func cancelRecording() async {
        guard isRecording || capture.isCapturing else { return }
        await capture.stopCapture()

        await MainActor.run {
            durationTimer?.invalidate()
            durationTimer = nil
            duration = 0
            startTime = nil
            isRecording = false
        }

        lock.withLock {
            samples.removeAll(keepingCapacity: false)
            sourceSampleRate = nil
        }

        AppLogger.info("QuickSystemAudioRecorder cancelled")
    }

    private func append(_ sampleBuffer: CMSampleBuffer) {
        guard sampleBuffer.isValid,
              let formatDescription = sampleBuffer.formatDescription,
              sampleBuffer.numSamples > 0,
              let sourceFormat = Self.canonicalPCMFormat(from: formatDescription),
              let pcmBuffer = Self.copyPCMBuffer(from: sampleBuffer, format: sourceFormat) else {
            return
        }

        let monoSamples = Self.monoSamples(from: pcmBuffer)
        guard !monoSamples.isEmpty else { return }

        let didAppend = lock.withLock {
            if let sourceSampleRate, abs(sourceSampleRate - sourceFormat.sampleRate) > 1 {
                return false
            }
            sourceSampleRate = sourceFormat.sampleRate
            samples.append(contentsOf: monoSamples)
            return true
        }
        if !didAppend {
            AppLogger.warning(String(
                format: "QuickSystemAudioRecorder dropped buffer reason=sample-rate-changed old=%.1f new=%.1f",
                sourceSampleRate ?? 0,
                sourceFormat.sampleRate
            ))
            return
        }
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

    private static func copyPCMBuffer(from sampleBuffer: CMSampleBuffer, format: AVAudioFormat) -> AVAudioPCMBuffer? {
        let frameCount = AVAudioFrameCount(sampleBuffer.numSamples)
        guard frameCount > 0,
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            return nil
        }
        buffer.frameLength = frameCount
        let status = CMSampleBufferCopyPCMDataIntoAudioBufferList(
            sampleBuffer,
            at: 0,
            frameCount: Int32(frameCount),
            into: buffer.mutableAudioBufferList
        )
        guard status == noErr else {
            AppLogger.warning("QuickSystemAudioRecorder failed to copy PCM data status=\(status)")
            return nil
        }
        return buffer
    }

    private static func monoSamples(from buffer: AVAudioPCMBuffer) -> [Float] {
        let frames = Int(buffer.frameLength)
        let channels = max(Int(buffer.format.channelCount), 1)
        guard frames > 0 else { return [] }

        var output = [Float](repeating: 0, count: frames)
        let audioBuffers = UnsafeMutableAudioBufferListPointer(buffer.mutableAudioBufferList)

        func clamp(_ value: Float) -> Float {
            min(max(value, -1), 1)
        }

        switch buffer.format.commonFormat {
        case .pcmFormatFloat32:
            if buffer.format.isInterleaved {
                guard let raw = audioBuffers[0].mData?.assumingMemoryBound(to: Float.self) else { return [] }
                for frame in 0..<frames {
                    var sum: Float = 0
                    for channel in 0..<channels {
                        sum += clamp(raw[frame * channels + channel])
                    }
                    output[frame] = sum / Float(channels)
                }
            } else {
                for frame in 0..<frames {
                    var sum: Float = 0
                    for channel in 0..<channels {
                        guard channel < audioBuffers.count,
                              let raw = audioBuffers[channel].mData?.assumingMemoryBound(to: Float.self) else {
                            continue
                        }
                        sum += clamp(raw[frame])
                    }
                    output[frame] = sum / Float(channels)
                }
            }

        case .pcmFormatFloat64:
            if buffer.format.isInterleaved {
                guard let raw = audioBuffers[0].mData?.assumingMemoryBound(to: Double.self) else { return [] }
                for frame in 0..<frames {
                    var sum: Float = 0
                    for channel in 0..<channels {
                        sum += clamp(Float(raw[frame * channels + channel]))
                    }
                    output[frame] = sum / Float(channels)
                }
            } else {
                for frame in 0..<frames {
                    var sum: Float = 0
                    for channel in 0..<channels {
                        guard channel < audioBuffers.count,
                              let raw = audioBuffers[channel].mData?.assumingMemoryBound(to: Double.self) else {
                            continue
                        }
                        sum += clamp(Float(raw[frame]))
                    }
                    output[frame] = sum / Float(channels)
                }
            }

        case .pcmFormatInt16:
            let scale = Float(Int16.max)
            if buffer.format.isInterleaved {
                guard let raw = audioBuffers[0].mData?.assumingMemoryBound(to: Int16.self) else { return [] }
                for frame in 0..<frames {
                    var sum: Float = 0
                    for channel in 0..<channels {
                        sum += clamp(Float(raw[frame * channels + channel]) / scale)
                    }
                    output[frame] = sum / Float(channels)
                }
            } else {
                for frame in 0..<frames {
                    var sum: Float = 0
                    for channel in 0..<channels {
                        guard channel < audioBuffers.count,
                              let raw = audioBuffers[channel].mData?.assumingMemoryBound(to: Int16.self) else {
                            continue
                        }
                        sum += clamp(Float(raw[frame]) / scale)
                    }
                    output[frame] = sum / Float(channels)
                }
            }

        case .pcmFormatInt32:
            let scale = Float(Int32.max)
            if buffer.format.isInterleaved {
                guard let raw = audioBuffers[0].mData?.assumingMemoryBound(to: Int32.self) else { return [] }
                for frame in 0..<frames {
                    var sum: Float = 0
                    for channel in 0..<channels {
                        sum += clamp(Float(raw[frame * channels + channel]) / scale)
                    }
                    output[frame] = sum / Float(channels)
                }
            } else {
                for frame in 0..<frames {
                    var sum: Float = 0
                    for channel in 0..<channels {
                        guard channel < audioBuffers.count,
                              let raw = audioBuffers[channel].mData?.assumingMemoryBound(to: Int32.self) else {
                            continue
                        }
                        sum += clamp(Float(raw[frame]) / scale)
                    }
                    output[frame] = sum / Float(channels)
                }
            }

        case .otherFormat:
            return []
        @unknown default:
            return []
        }

        return output
    }

    private static func makeWAVData(
        monoSamples: [Float],
        sourceSampleRate: Double,
        targetSampleRate: Int
    ) -> Data? {
        guard !monoSamples.isEmpty, sourceSampleRate > 0 else { return nil }

        let outputSamples: [Float]
        if abs(sourceSampleRate - Double(targetSampleRate)) < 1 {
            outputSamples = monoSamples
        } else {
            let ratio = sourceSampleRate / Double(targetSampleRate)
            let outputCount = max(Int(Double(monoSamples.count) / ratio), 1)
            var resampled = [Float](repeating: 0, count: outputCount)
            for index in 0..<outputCount {
                let sourcePosition = Double(index) * ratio
                let lower = min(Int(sourcePosition), monoSamples.count - 1)
                let upper = min(lower + 1, monoSamples.count - 1)
                let fraction = Float(sourcePosition - Double(lower))
                resampled[index] = monoSamples[lower] + (monoSamples[upper] - monoSamples[lower]) * fraction
            }
            outputSamples = resampled
        }

        let frameCount = outputSamples.count
        var int16Samples = [Int16](repeating: 0, count: frameCount)
        for index in 0..<frameCount {
            let clipped = min(max(outputSamples[index], -1), 1)
            int16Samples[index] = Int16(clipped * Float(Int16.max))
        }

        var wavData = Data()
        wavData.append(contentsOf: "RIFF".utf8)
        wavData.append(contentsOf: withUnsafeBytes(of: UInt32(36 + frameCount * 2).littleEndian) { Array($0) })
        wavData.append(contentsOf: "WAVE".utf8)
        wavData.append(contentsOf: "fmt ".utf8)
        wavData.append(contentsOf: withUnsafeBytes(of: UInt32(16).littleEndian) { Array($0) })
        wavData.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian) { Array($0) })
        wavData.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian) { Array($0) })
        wavData.append(contentsOf: withUnsafeBytes(of: UInt32(targetSampleRate).littleEndian) { Array($0) })
        wavData.append(contentsOf: withUnsafeBytes(of: UInt32(targetSampleRate * 2).littleEndian) { Array($0) })
        wavData.append(contentsOf: withUnsafeBytes(of: UInt16(2).littleEndian) { Array($0) })
        wavData.append(contentsOf: withUnsafeBytes(of: UInt16(16).littleEndian) { Array($0) })
        wavData.append(contentsOf: "data".utf8)
        wavData.append(contentsOf: withUnsafeBytes(of: UInt32(frameCount * 2).littleEndian) { Array($0) })
        int16Samples.withUnsafeBufferPointer { pointer in
            if let baseAddress = pointer.baseAddress {
                wavData.append(
                    UnsafeBufferPointer(
                        start: UnsafeRawPointer(baseAddress).assumingMemoryBound(to: UInt8.self),
                        count: frameCount * 2
                    )
                )
            }
        }
        return wavData
    }
}

struct QuickSystemAudioRecording {
    let audioData: Data
    let durationSeconds: TimeInterval
}
