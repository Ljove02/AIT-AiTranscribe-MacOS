import Foundation
import AVFoundation

/// The fader for one channel, applied to the audio as it is captured.
///
/// The gain goes into the file. That is deliberate: storing fader moves and
/// applying them at export would need a timeline of changes whose timestamps
/// line up with the audio, which is more machinery and more to get wrong, and
/// it is how OBS behaves.
///
/// Two details keep it from being audible in the wrong way. The gain slides
/// across each buffer instead of jumping between them, because a step change
/// at a buffer boundary is a click, and you would hear one every time you
/// touched the fader. And a soft limiter runs immediately after, so pushing a
/// fader up cannot clip the file: samples above the threshold are compressed
/// toward the ceiling and never reach it.
final class GainStage: @unchecked Sendable {
    /// Above this, samples are compressed rather than passed through.
    private let limiterThreshold: Float = 0.80
    /// Compressed samples approach this and never exceed it.
    private let limiterCeiling: Float = 0.99

    private let lock = NSLock()
    private var targetGain: Float = 1.0

    /// Only touched on the audio thread.
    private var currentGain: Float = 1.0

    var decibels: Float {
        lock.lock(); defer { lock.unlock() }
        return targetGain > 0 ? 20 * log10f(targetGain) : -60
    }

    func setDecibels(_ db: Float) {
        lock.lock()
        targetGain = powf(10, db / 20)
        lock.unlock()
    }

    func reset() {
        lock.lock()
        targetGain = 1.0
        lock.unlock()
        currentGain = 1.0
    }

    private func softLimit(_ value: Float) -> Float {
        let magnitude = abs(value)
        guard magnitude > limiterThreshold else { return value }
        let range = limiterCeiling - limiterThreshold
        let compressed = limiterThreshold + range * tanhf((magnitude - limiterThreshold) / range)
        return value < 0 ? -compressed : compressed
    }

    /// Applies the fader to a buffer in place, sliding from the gain the last
    /// buffer ended on to the current target across this one.
    func apply(to buffer: AVAudioPCMBuffer) {
        lock.lock()
        let target = targetGain
        lock.unlock()

        let frames = Int(buffer.frameLength)
        guard frames > 0 else { return }
        let format = buffer.format
        let channels = max(Int(format.channelCount), 1)
        let buffers = UnsafeMutableAudioBufferListPointer(buffer.mutableAudioBufferList)

        let startGain = currentGain
        let step = (target - startGain) / Float(frames)
        // Nothing to do when the fader is at unity and has not moved.
        if startGain == 1.0 && target == 1.0 { return }
        currentGain = target

        func gain(_ frame: Int) -> Float { startGain + step * Float(frame) }

        switch format.commonFormat {
        case .pcmFormatFloat32:
            if format.isInterleaved {
                guard let raw = buffers[0].mData?.assumingMemoryBound(to: Float.self) else { return }
                for frame in 0..<frames {
                    let g = gain(frame)
                    for channel in 0..<channels {
                        let index = frame * channels + channel
                        raw[index] = softLimit(raw[index] * g)
                    }
                }
            } else {
                for channel in 0..<channels {
                    guard let raw = buffers[channel].mData?.assumingMemoryBound(to: Float.self) else { continue }
                    for frame in 0..<frames {
                        raw[frame] = softLimit(raw[frame] * gain(frame))
                    }
                }
            }
        case .pcmFormatInt16:
            let scale = Float(Int16.max)
            if format.isInterleaved {
                guard let raw = buffers[0].mData?.assumingMemoryBound(to: Int16.self) else { return }
                for frame in 0..<frames {
                    let g = gain(frame)
                    for channel in 0..<channels {
                        let index = frame * channels + channel
                        raw[index] = Int16(softLimit(Float(raw[index]) / scale * g) * scale)
                    }
                }
            } else {
                for channel in 0..<channels {
                    guard let raw = buffers[channel].mData?.assumingMemoryBound(to: Int16.self) else { continue }
                    for frame in 0..<frames {
                        raw[frame] = Int16(softLimit(Float(raw[frame]) / scale * gain(frame)) * scale)
                    }
                }
            }
        case .pcmFormatInt32:
            let scale = Float(Int32.max)
            if format.isInterleaved {
                guard let raw = buffers[0].mData?.assumingMemoryBound(to: Int32.self) else { return }
                for frame in 0..<frames {
                    let g = gain(frame)
                    for channel in 0..<channels {
                        let index = frame * channels + channel
                        raw[index] = Int32(softLimit(Float(raw[index]) / scale * g) * scale)
                    }
                }
            } else {
                for channel in 0..<channels {
                    guard let raw = buffers[channel].mData?.assumingMemoryBound(to: Int32.self) else { continue }
                    for frame in 0..<frames {
                        raw[frame] = Int32(softLimit(Float(raw[frame]) / scale * gain(frame)) * scale)
                    }
                }
            }
        default:
            return
        }
    }
}
