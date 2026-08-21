import Foundation
import AVFoundation

/// Where a level sits on the meter. The boundaries come from the broadcast
/// standard OBS follows: -20 dBFS alignment level, -9 dBFS permitted maximum.
enum AudioLevelZone {
    case nominal   // green
    case warning   // yellow
    case error     // red
}

/// One meter reading, in dBFS.
struct AudioLevelReading: Equatable {
    /// RMS, smoothed over 300 ms. This is the solid part of the bar.
    var magnitudeDB: Float
    /// Peak, falling at a fixed rate once the sound stops.
    var peakDB: Float
    /// Highest peak seen recently, held so a spike you missed is still there
    /// when you look over.
    var peakHoldDB: Float

    static let silent = AudioLevelReading(
        magnitudeDB: AudioLevelMeter.minimumDB,
        peakDB: AudioLevelMeter.minimumDB,
        peakHoldDB: AudioLevelMeter.minimumDB
    )

    var zone: AudioLevelZone {
        if peakHoldDB >= AudioLevelMeter.errorDB { return .error }
        if peakHoldDB >= AudioLevelMeter.warningDB { return .warning }
        return .nominal
    }

    /// Position on the meter, 0 at the bottom of the scale and 1 at full scale.
    var fraction: Float {
        let clamped = min(max(magnitudeDB, AudioLevelMeter.minimumDB), 0)
        return 1.0 - (clamped / AudioLevelMeter.minimumDB)
    }

    var peakFraction: Float {
        let clamped = min(max(peakDB, AudioLevelMeter.minimumDB), 0)
        return 1.0 - (clamped / AudioLevelMeter.minimumDB)
    }

    var peakHoldFraction: Float {
        let clamped = min(max(peakHoldDB, AudioLevelMeter.minimumDB), 0)
        return 1.0 - (clamped / AudioLevelMeter.minimumDB)
    }
}

/// Level meter with the ballistics OBS uses, taken from its
/// frontend/components/VolumeMeter.cpp defaults.
///
/// Audio threads push measurements in; the UI pulls a reading out whenever it
/// draws. The two are decoupled, so a slow redraw cannot stall capture and a
/// fast redraw cannot outrun the audio.
final class AudioLevelMeter: @unchecked Sendable {
    /// Bottom of the scale.
    static let minimumDB: Float = -60
    /// Green turns yellow here.
    static let warningDB: Float = -20
    /// Yellow turns red here.
    static let errorDB: Float = -9

    /// 20 dB in 1.7 seconds.
    private let peakDecayPerSecond: Float = 11.76
    /// The bar reaches 99% of a new level in 300 ms.
    ///
    /// OBS states this in a comment but implements it as a fraction of the
    /// redraw interval, which makes the speed depend on frame rate and lands
    /// well short of 99%. Since the meter here is decoupled from drawing, the
    /// exponential is used directly: the same 300 ms whatever the frame rate.
    private let magnitudeIntegrationTime: Float = 0.3
    private var magnitudeRate: Float { logf(100) / magnitudeIntegrationTime }
    /// How long the peak marker stays put.
    private let peakHoldDuration: TimeInterval = 20.0

    private let lock = NSLock()
    private var pendingSumSquares: Double = 0
    private var pendingSampleCount: Int = 0
    private var pendingPeak: Float = 0

    private var magnitudeDB: Float = AudioLevelMeter.minimumDB
    private var peakDB: Float = AudioLevelMeter.minimumDB
    private var peakHoldDB: Float = AudioLevelMeter.minimumDB
    private var lastReadingAt: Date?
    private var peakHeldSince: Date?

    static func decibels(_ amplitude: Float) -> Float {
        guard amplitude > 0 else { return minimumDB }
        return max(20 * log10f(amplitude), minimumDB)
    }

    /// Called from an audio thread, once per buffer.
    func add(sumSquares: Double, peak: Float, sampleCount: Int) {
        guard sampleCount > 0 else { return }
        lock.lock()
        pendingSumSquares += sumSquares
        pendingSampleCount += sampleCount
        pendingPeak = max(pendingPeak, peak)
        lock.unlock()
    }

    func reset() {
        lock.lock()
        pendingSumSquares = 0
        pendingSampleCount = 0
        pendingPeak = 0
        lock.unlock()
        magnitudeDB = Self.minimumDB
        peakDB = Self.minimumDB
        peakHoldDB = Self.minimumDB
        lastReadingAt = nil
        peakHeldSince = nil
    }

    /// Called from the UI when it draws. Applies the ballistics for however
    /// much time has passed since the last draw, so the meter behaves the same
    /// whatever frame rate it is running at.
    func reading(now: Date = Date()) -> AudioLevelReading {
        lock.lock()
        let sumSquares = pendingSumSquares
        let sampleCount = pendingSampleCount
        let peak = pendingPeak
        pendingSumSquares = 0
        pendingSampleCount = 0
        pendingPeak = 0
        lock.unlock()

        let elapsed = Float(now.timeIntervalSince(lastReadingAt ?? now))
        lastReadingAt = now

        // The bar slides toward the new level rather than jumping to it.
        // Skipped when no audio arrived, so a redraw faster than the capture
        // callback does not drag the bar down between buffers.
        if sampleCount > 0 {
            let instant = Self.decibels(Float((sumSquares / Double(sampleCount)).squareRoot()))
            if elapsed > 0 {
                let progress = 1 - expf(-elapsed * magnitudeRate)
                magnitudeDB = min(max(magnitudeDB + (instant - magnitudeDB) * progress, Self.minimumDB), 0)
            } else {
                magnitudeDB = instant
            }
        }

        // Peak jumps up instantly and falls at a fixed rate.
        let instantPeak = Self.decibels(peak)
        if instantPeak > peakDB {
            peakDB = instantPeak
        } else {
            peakDB = max(peakDB - peakDecayPerSecond * elapsed, Self.minimumDB)
        }

        // The marker follows a new high immediately, then sits until it expires.
        if peakDB > peakHoldDB || peakHeldSince == nil {
            peakHoldDB = peakDB
            peakHeldSince = now
        } else if let since = peakHeldSince, now.timeIntervalSince(since) > peakHoldDuration {
            peakHoldDB = peakDB
            peakHeldSince = now
        }

        return AudioLevelReading(magnitudeDB: magnitudeDB, peakDB: peakDB, peakHoldDB: peakHoldDB)
    }
}
