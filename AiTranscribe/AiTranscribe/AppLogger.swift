import Foundation

enum AppLogger {
    static func debug(_ message: String) {
        log(message, level: .debug)
    }

    static func info(_ message: String) {
        log(message, level: .info)
    }

    static func warning(_ message: String) {
        log(message, level: .warning)
    }

    static func error(_ message: String) {
        log(message, level: .error)
    }

    private static func log(_ message: String, level: LogEntry.LogLevel) {
        let line = "AiTranscribe \(level.rawValue) \(message)"
        print(line)
        Task { @MainActor in
            BackendManager.shared.addAppLog(message, level: level)
        }
    }
}
