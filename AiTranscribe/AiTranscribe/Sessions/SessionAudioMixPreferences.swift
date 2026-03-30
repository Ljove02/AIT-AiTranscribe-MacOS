import Foundation

enum SessionAudioMixPreferences {
    static let micTrimDBKey = "sessionMixMicTrimDB"
    static let systemTrimDBKey = "sessionMixSystemTrimDB"

    static let defaultMicTrimDB: Double = 0
    static let defaultSystemTrimDB: Double = 0

    // These preserve the current working session balance. User sliders
    // apply trim around this baseline instead of replacing it.
    static let baselineMicGain: Float = 1.9
    static let baselineSystemGain: Float = 0.3

    static let micTrimRange: ClosedRange<Double> = -6...18
    static let systemTrimRange: ClosedRange<Double> = -18...6

    static func micTrimDB(defaults: UserDefaults = .standard) -> Double {
        trimDB(forKey: micTrimDBKey, defaults: defaults, fallback: defaultMicTrimDB)
    }

    static func systemTrimDB(defaults: UserDefaults = .standard) -> Double {
        trimDB(forKey: systemTrimDBKey, defaults: defaults, fallback: defaultSystemTrimDB)
    }

    static func effectiveMicGain(defaults: UserDefaults = .standard) -> Float {
        effectiveMicGain(forTrimDB: micTrimDB(defaults: defaults))
    }

    static func effectiveSystemGain(defaults: UserDefaults = .standard) -> Float {
        effectiveSystemGain(forTrimDB: systemTrimDB(defaults: defaults))
    }

    static func effectiveMicGain(forTrimDB trimDB: Double) -> Float {
        baselineMicGain * multiplier(forTrimDB: trimDB)
    }

    static func effectiveSystemGain(forTrimDB trimDB: Double) -> Float {
        baselineSystemGain * multiplier(forTrimDB: trimDB)
    }

    static func displayString(for value: Double) -> String {
        if abs(value) < 0.05 {
            return "0 dB"
        }
        return String(format: "%@%.1f dB", value > 0 ? "+" : "", value)
    }

    static func multiplierString(for gain: Float) -> String {
        String(format: "%.2fx", gain)
    }

    private static func trimDB(forKey key: String, defaults: UserDefaults, fallback: Double) -> Double {
        guard defaults.object(forKey: key) != nil else { return fallback }
        return defaults.double(forKey: key)
    }

    private static func multiplier(forTrimDB db: Double) -> Float {
        Float(pow(10.0, db / 20.0))
    }
}
