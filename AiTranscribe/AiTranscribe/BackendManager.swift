/*
 BackendManager.swift
 ====================

 Manages the Python backend server lifecycle.

 This class handles:
 - Starting the backend when the app launches
 - Capturing stdout/stderr for the debug console
 - Restarting if the backend crashes
 - Clean shutdown when the app quits 

 DEVELOPMENT vs PRODUCTION:
 --------------------------
 - Development: Runs `python server.py` from the backend directory
 - Production: Runs bundled `aitranscribe-server` executable from app bundle

 PROCESS MANAGEMENT:
 -------------------
 Swift's Process class (formerly NSTask) lets us run external commands.
 We use Pipe to capture the output for the debug console.
 */

import Foundation
import AppKit
import Combine

/// Backend execution mode
enum BackendMode: String, CaseIterable {
    case pyinstaller = "pyinstaller"   // Bundled PyInstaller executable (Whisper-only)
    case nemoVenv = "nemo_venv"        // NeMo virtual environment (all models)
    case development = "development"    // Development mode (python server.py)

    var displayName: String {
        switch self {
        case .pyinstaller: return "Whisper Only"
        case .nemoVenv: return "All Models (NeMo)"
        case .development: return "Development"
        }
    }
}

/// A single log entry from the backend
struct LogEntry: Identifiable, Equatable {
    let id = UUID()
    let timestamp: Date
    let level: LogLevel
    let message: String

    enum LogLevel: String {
        case info = "INFO"
        case warning = "WARN"
        case error = "ERROR"
        case debug = "DEBUG"

        var color: String {
            switch self {
            case .info: return "secondary"
            case .warning: return "orange"
            case .error: return "red"
            case .debug: return "gray"
            }
        }
    }

    /// Parse log level from a log line
    static func parseLevel(from line: String) -> LogLevel {
        let upper = line.uppercased()
        if upper.contains("ERROR") || upper.contains("EXCEPTION") || upper.contains("TRACEBACK") {
            return .error
        } else if upper.contains("WARNING") || upper.contains("WARN") {
            return .warning
        } else if upper.contains("DEBUG") {
            return .debug
        }
        return .info
    }
}


/// Manages the backend Python server process
@MainActor
class BackendManager: ObservableObject {

    // =========================================================================
    // SINGLETON
    // =========================================================================

    /// Shared instance - ensures only ONE BackendManager exists
    static let shared = BackendManager()

    // =========================================================================
    // PUBLISHED STATE
    // =========================================================================

    /// All captured log entries
    @Published var logs: [LogEntry] = []

    /// Whether the backend process is currently running
    @Published var isRunning: Bool = false

    /// Whether the server is ready to accept requests
    @Published var isServerReady: Bool = false

    /// Status message for display
    @Published var statusMessage: String = "Not started"

    /// Number of restart attempts
    @Published var restartCount: Int = 0

    /// Current backend execution mode
    @Published var currentBackendMode: BackendMode = .development

    // =========================================================================
    // NEMO VENV PATHS
    // =========================================================================

    /// Application Support directory
    static var appSupportURL: URL {
        let paths = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)
        return paths[0].appendingPathComponent("AiTranscribe")
    }

    /// Path to the NeMo virtual environment
    var nemoVenvPath: URL {
        Self.appSupportURL.appendingPathComponent("nemo-venv")
    }

    /// Path to Python in the NeMo venv
    var nemoPythonPath: URL {
        nemoVenvPath.appendingPathComponent("bin").appendingPathComponent("python3")
    }

    /// Check if NeMo venv exists and has Python
    var nemoVenvExists: Bool {
        FileManager.default.fileExists(atPath: nemoPythonPath.path)
    }

    // =========================================================================
    // PRIVATE STATE
    // =========================================================================

    /// The backend process
    private var process: Process?

    /// Pipe for capturing stdout
    private var stdoutPipe: Pipe?

    /// Pipe for capturing stderr
    private var stderrPipe: Pipe?

    /// Timer for checking process health
    private var healthCheckTimer: Timer?

    /// Maximum log entries to keep (prevent memory issues)
    private let maxLogEntries = 5000

    /// Restart backoff delay (increases with each failure)
    private var restartDelay: TimeInterval = 2.0

    /// Maximum restart delay
    private let maxRestartDelay: TimeInterval = 30.0

    /// Maximum restart attempts before giving up
    private let maxRestartAttempts = 5

    /// Whether we're intentionally stopping (don't restart)
    private var isIntentionallyStopping = false

    // =========================================================================
    // INITIALIZATION
    // =========================================================================

    init() {
        print("BackendManager.init() called - singleton being created")

        // Add initial log entry to show init happened
        let entry = LogEntry(timestamp: Date(), level: .info, message: "BackendManager initialized")
        logs.append(entry)

        // Register for app termination notification
        NotificationCenter.default.addObserver(
            forName: NSApplication.willTerminateNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            guard let self = self else { return }
            // Since the queue is .main and BackendManager is @MainActor,
            // we can safely call stop() directly without a Task
            MainActor.assumeIsolated {
                self.stop()
            }
        }

        // Auto-start the backend when BackendManager is created
        // Using DispatchQueue to run after init completes
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            print("BackendManager: Auto-starting backend...")
            self.start()
        }
    }

    deinit {
        // Clean up synchronously without calling main actor-isolated stop()
        // This is safe because deinit only runs when there are no more references
        healthCheckTimer?.invalidate()
        
        if let process = process, process.isRunning {
            process.terminate()
        }
        
        // Clean up pipe handlers
        stdoutPipe?.fileHandleForReading.readabilityHandler = nil
        stderrPipe?.fileHandleForReading.readabilityHandler = nil
    }

    // =========================================================================
    // PATH DETECTION
    // =========================================================================

    /// Check if this is a production build (bundled executable exists)
    var isProductionBuild: Bool {
        Bundle.main.url(forResource: "aitranscribe-server", withExtension: nil) != nil
    }

    /// Get the path to the backend executable or script
    var backendExecutablePath: String? {
        if isProductionBuild {
            // Production: use bundled executable
            return Bundle.main.path(forResource: "aitranscribe-server", ofType: nil)
        } else {
            // Development: find the backend directory
            // The backend is typically at ../backend relative to the .app or project
            return findDevelopmentBackendPath()
        }
    }

    /// Find the backend path in development mode
    private func findDevelopmentBackendPath() -> String? {
        // Try multiple possible locations

        // 1. Check if AITRANSCRIBE_BACKEND_PATH environment variable is set
        if let envPath = ProcessInfo.processInfo.environment["AITRANSCRIBE_BACKEND_PATH"] {
            let serverPath = (envPath as NSString).appendingPathComponent("server.py")
            if FileManager.default.fileExists(atPath: serverPath) {
                return serverPath
            }
        }

        // 2. Common development paths
        let possiblePaths = [
            // Relative to the app bundle (when running from Xcode)
            Bundle.main.bundlePath + "/../../../../backend/server.py",
            // Home directory based
            NSHomeDirectory() + "/Projects/AiTranscribe/backend/server.py",
            NSHomeDirectory() + "/Developer/AiTranscribe/backend/server.py",
        ]

        for path in possiblePaths {
            let expandedPath = (path as NSString).standardizingPath
            if FileManager.default.fileExists(atPath: expandedPath) {
                return expandedPath
            }
        }

        return nil
    }

    /// Find Python executable (prioritize Python with ML dependencies installed)
    private func findPythonPath() -> String {
        // Try Python paths in order of preference
        // Python.framework versions are more likely to have ML packages installed
        let possiblePaths = [
            "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3",
            "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3",
            "/Library/Frameworks/Python.framework/Versions/3.10/bin/python3",
            "/opt/homebrew/bin/python3",
            "/usr/local/bin/python3",
            "/usr/bin/python3",
        ]

        for path in possiblePaths {
            if FileManager.default.fileExists(atPath: path) {
                return path
            }
        }

        // Fallback to PATH-based python3
        return "/usr/bin/env"
    }

    // =========================================================================
    // PROCESS MANAGEMENT
    // =========================================================================

    /// Wait for the backend server to be ready (with timeout)
    func waitForServerReady(timeout: TimeInterval = 30.0) async {
        let startTime = Date()

        while Date().timeIntervalSince(startTime) < timeout {
            let elapsed = Int(Date().timeIntervalSince(startTime))

            if isRunning {
                // Update status during polling
                statusMessage = "Starting... (\(elapsed)s)"

                // Try to actually connect to the server
                do {
                    let url = URL(string: "http://localhost:8765/health")!
                    let (_, response) = try await URLSession.shared.data(from: url)
                    if let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 {
                        isServerReady = true
                        statusMessage = "Ready - \(currentBackendMode.displayName)"
                        addLog("Server is ready and accepting requests", level: .info)
                        return
                    }
                } catch {
                    // Server not ready yet, keep trying
                }
                try? await Task.sleep(nanoseconds: 500_000_000) // 0.5 seconds
            } else {
                try? await Task.sleep(nanoseconds: 100_000_000) // 0.1 seconds
            }
        }

        addLog("Timeout waiting for server to be ready", level: .warning)
        isServerReady = false
        statusMessage = "Timeout - server not responding"
    }

    /// Kill any existing process using port 8765
    /// This prevents "address already in use" errors when restarting
    private func killExistingBackendProcess() {
        // First try graceful SIGTERM
        let termTask = Process()
        termTask.executableURL = URL(fileURLWithPath: "/bin/sh")
        termTask.arguments = ["-c", "lsof -ti:8765 | xargs kill -15 2>/dev/null"]

        do {
            try termTask.run()
            termTask.waitUntilExit()
            if termTask.terminationStatus == 0 {
                // Give the process time to clean up gracefully
                Thread.sleep(forTimeInterval: 1.0)
            }
        } catch {
            // Ignore errors
        }

        // Now forcefully kill any remaining processes
        let killTask = Process()
        killTask.executableURL = URL(fileURLWithPath: "/bin/sh")
        killTask.arguments = ["-c", "lsof -ti:8765 | xargs kill -9 2>/dev/null"]

        do {
            try killTask.run()
            killTask.waitUntilExit()
            if killTask.terminationStatus == 0 {
                addLog("Killed existing process on port 8765", level: .debug)
            }
        } catch {
            // Ignore errors - no process to kill is fine
        }

        // Wait for port to be fully released (important for NeMo's heavy resources)
        Thread.sleep(forTimeInterval: 1.0)
    }

    /// Start the backend server
    func start() {
        guard !isRunning else {
            addLog("Backend already running", level: .warning)
            return
        }

        isIntentionallyStopping = false

        // Kill any existing process on port 8765 to prevent "address already in use" errors
        killExistingBackendProcess()

        // Determine which mode to use if not explicitly set
        if currentBackendMode == .development && isProductionBuild {
            currentBackendMode = determineBestMode()
        }

        addLog("Starting backend in \(currentBackendMode.displayName) mode...", level: .info)
        statusMessage = "Starting..."

        // Create the process
        let process = Process()
        self.process = process

        // Configure based on backend mode
        switch currentBackendMode {
        case .pyinstaller:
            // Production: run the bundled executable directly
            guard let executablePath = Bundle.main.path(forResource: "aitranscribe-server", ofType: nil) else {
                addLog("Bundled executable not found", level: .error)
                statusMessage = "Backend not found"
                return
            }
            process.executableURL = URL(fileURLWithPath: executablePath)
            process.arguments = []
            // Inherit system environment (including PATH needed by dependencies)
            var env = ProcessInfo.processInfo.environment
            env["AITRANSCRIBE_BACKEND_MODE"] = "pyinstaller"
            process.environment = env
            addLog("Running bundled executable", level: .debug)

        case .nemoVenv:
            // NeMo venv: run server.py with venv Python
            guard nemoVenvExists else {
                addLog("NeMo venv not found, falling back to development mode", level: .warning)
                currentBackendMode = .development
                start()
                return
            }

            guard let backendPath = findDevelopmentBackendPath() else {
                addLog("Could not find server.py for NeMo mode", level: .error)
                statusMessage = "Backend not found"
                return
            }

            process.executableURL = nemoPythonPath
            process.arguments = [backendPath]

            // Build environment with system PATH included (needed by lhotse/NeMo)
            var env = ProcessInfo.processInfo.environment
            let venvBinPath = "\(nemoVenvPath.path)/bin"
            if let existingPath = env["PATH"] {
                env["PATH"] = "\(venvBinPath):\(existingPath)"
            } else {
                env["PATH"] = "\(venvBinPath):/usr/local/bin:/usr/bin:/bin"
            }
            env["AITRANSCRIBE_BACKEND_MODE"] = "nemo_venv"
            process.environment = env

            let backendDir = (backendPath as NSString).deletingLastPathComponent
            process.currentDirectoryURL = URL(fileURLWithPath: backendDir)
            addLog("Running with NeMo venv: \(nemoPythonPath.path)", level: .debug)

        case .development:
            // Development: run Python with the script
            guard let backendPath = backendExecutablePath else {
                addLog("Could not find backend. Set AITRANSCRIBE_BACKEND_PATH environment variable.", level: .error)
                statusMessage = "Backend not found"
                return
            }

            let pythonPath = findPythonPath()
            addLog("Using Python: \(pythonPath)", level: .debug)

            if pythonPath == "/usr/bin/env" {
                process.executableURL = URL(fileURLWithPath: pythonPath)
                process.arguments = ["python3", backendPath]
            } else {
                process.executableURL = URL(fileURLWithPath: pythonPath)
                process.arguments = [backendPath]
            }

            // Inherit system environment (including PATH needed by lhotse/NeMo)
            var env = ProcessInfo.processInfo.environment
            env["AITRANSCRIBE_BACKEND_MODE"] = "development"
            process.environment = env

            let backendDir = (backendPath as NSString).deletingLastPathComponent
            process.currentDirectoryURL = URL(fileURLWithPath: backendDir)
        }

        addLog("Path: \(process.executableURL?.path ?? "unknown")", level: .debug)

        // Set up pipes to capture output
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        self.stdoutPipe = stdoutPipe
        self.stderrPipe = stderrPipe

        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        // Handle stdout
        stdoutPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            if !data.isEmpty, let output = String(data: data, encoding: .utf8) {
                Task { @MainActor [weak self] in
                    self?.processOutput(output)
                }
            }
        }

        // Handle stderr
        stderrPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            if !data.isEmpty, let output = String(data: data, encoding: .utf8) {
                Task { @MainActor [weak self] in
                    self?.processOutput(output, isError: true)
                }
            }
        }

        // Handle process termination
        // Capture the process reference to avoid race conditions when switching backends
        let thisProcess = process
        process.terminationHandler = { [weak self] proc in
            Task { @MainActor [weak self] in
                guard let self = self else { return }
                // Only handle termination if this is still the current process
                // (avoids race condition when switching backends)
                if self.process === thisProcess {
                    self.handleTermination(exitCode: proc.terminationStatus)
                }
            }
        }

        // Start the process
        do {
            try process.run()
            isRunning = true
            statusMessage = "Running"
            restartDelay = 2.0
            addLog("Backend process started (PID: \(process.processIdentifier))", level: .info)
            addLog("Waiting for server to be ready...", level: .debug)

            // Start health check timer
            startHealthCheck()

        } catch {
            addLog("Failed to start backend: \(error.localizedDescription)", level: .error)
            statusMessage = "Failed to start"
            isRunning = false
        }
    }

    /// Stop the backend server
    func stop() {
        isIntentionallyStopping = true
        isServerReady = false
        healthCheckTimer?.invalidate()
        healthCheckTimer = nil

        guard let process = process, process.isRunning else {
            isRunning = false
            return
        }

        addLog("Stopping backend...", level: .info)
        statusMessage = "Stopping..."

        // Send SIGTERM for graceful shutdown
        process.terminate()

        // Capture the process we're stopping to avoid race conditions
        // (a new process might be started before the delayed block runs)
        let stoppingProcess = process

        // Give it a moment to shut down gracefully
        DispatchQueue.global().asyncAfter(deadline: .now() + 2.0) { [weak self] in
            // Only force-kill if this same process is still running
            if stoppingProcess.isRunning {
                // Force kill if still running
                stoppingProcess.interrupt()
            }

            Task { @MainActor [weak self] in
                guard let self = self else { return }
                // Only update status if no new process has been started
                // (self.process would be different or isRunning would be true for new process)
                if self.process === stoppingProcess || self.process == nil {
                    self.isRunning = false
                    self.statusMessage = "Stopped"
                    self.addLog("Backend stopped", level: .info)
                }
            }
        }

        // Clean up pipes
        stdoutPipe?.fileHandleForReading.readabilityHandler = nil
        stderrPipe?.fileHandleForReading.readabilityHandler = nil
    }

    /// Restart the backend
    func restart() {
        addLog("Restarting backend...", level: .info)
        stop()

        // Wait a moment before restarting
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) { [weak self] in
            self?.start()
        }
    }

    // =========================================================================
    // BACKEND MODE SWITCHING
    // =========================================================================

    /// Switch to NeMo mode (uses NeMo venv Python)
    func switchToNemoMode() async {
        guard nemoVenvExists else {
            addLog("Cannot switch to NeMo mode: venv does not exist", level: .error)
            return
        }

        addLog("Switching to NeMo mode...", level: .info)

        // Stop current backend and wait for it to fully terminate
        await stopAndWait()

        // Switch mode and restart
        currentBackendMode = .nemoVenv
        start()

        // Wait for server to be ready
        await waitForServerReady(timeout: 30.0)
    }

    /// Switch to Whisper mode (uses PyInstaller or development Python)
    func switchToWhisperMode() async {
        addLog("Switching to Whisper mode...", level: .info)

        // Stop current backend and wait for it to fully terminate
        await stopAndWait()

        // Switch mode and restart
        if isProductionBuild {
            currentBackendMode = .pyinstaller
        } else {
            currentBackendMode = .development
        }
        start()

        // Wait for server to be ready
        await waitForServerReady(timeout: 30.0)
    }

    /// Stop the backend and wait for it to fully terminate
    private func stopAndWait() async {
        guard let proc = process, proc.isRunning else {
            isRunning = false
            return
        }

        addLog("Stopping backend for mode switch...", level: .info)
        isIntentionallyStopping = true
        isServerReady = false
        healthCheckTimer?.invalidate()
        healthCheckTimer = nil

        // Send SIGTERM
        proc.terminate()

        // Wait for process to actually terminate (up to 5 seconds)
        let startTime = Date()
        while proc.isRunning && Date().timeIntervalSince(startTime) < 5.0 {
            try? await Task.sleep(nanoseconds: 100_000_000) // 0.1 seconds
        }

        // Force kill if still running
        if proc.isRunning {
            addLog("Force killing backend...", level: .warning)
            proc.interrupt()
            try? await Task.sleep(nanoseconds: 500_000_000) // 0.5 seconds
        }

        // Clean up
        isRunning = false
        statusMessage = "Stopped"
        stdoutPipe?.fileHandleForReading.readabilityHandler = nil
        stderrPipe?.fileHandleForReading.readabilityHandler = nil
        process = nil

        // Kill any remaining process on the port
        killExistingBackendProcess()

        addLog("Backend stopped, ready for restart", level: .info)
    }

    /// Determine the best backend mode to use
    func determineBestMode() -> BackendMode {
        if isProductionBuild {
            // In production, prefer NeMo venv if it exists
            if nemoVenvExists {
                return .nemoVenv
            }
            return .pyinstaller
        } else {
            // In development, always use development mode
            return .development
        }
    }

    // =========================================================================
    // OUTPUT PROCESSING
    // =========================================================================

    /// Process output from the backend
    private func processOutput(_ output: String, isError: Bool = false) {
        let lines = output.components(separatedBy: .newlines)

        for line in lines where !line.isEmpty {
            let level: LogEntry.LogLevel = isError ? .error : LogEntry.parseLevel(from: line)
            addLog(line, level: level)
        }
    }

    /// Add a log entry
    private func addLog(_ message: String, level: LogEntry.LogLevel) {
        let entry = LogEntry(timestamp: Date(), level: level, message: message)
        logs.append(entry)

        // Trim old logs if needed
        if logs.count > maxLogEntries {
            logs.removeFirst(logs.count - maxLogEntries)
        }
    }

    /// Clear all logs
    func clearLogs() {
        logs.removeAll()
        addLog("Logs cleared", level: .info)
    }

    /// Reset restart counter and try again
    func resetAndStart() {
        restartCount = 0
        restartDelay = 2.0
        addLog("Restart counter reset", level: .info)
        start()
    }

    /// Get all logs as a single string (for copying)
    func logsAsString() -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss"

        return logs.map { entry in
            let time = formatter.string(from: entry.timestamp)
            return "[\(time)] \(entry.level.rawValue): \(entry.message)"
        }.joined(separator: "\n")
    }

    // =========================================================================
    // HEALTH MONITORING
    // =========================================================================

    /// Start periodic health checks
    private func startHealthCheck() {
        healthCheckTimer?.invalidate()
        healthCheckTimer = Timer.scheduledTimer(withTimeInterval: 5.0, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.checkHealth()
            }
        }
    }

    /// Pause health checks (e.g., when Settings window is open to prevent UI re-renders)
    func pauseHealthCheck() {
        healthCheckTimer?.invalidate()
        healthCheckTimer = nil
    }

    /// Resume health checks (e.g., when Settings window closes)
    func resumeHealthCheck() {
        guard healthCheckTimer == nil else { return }
        startHealthCheck()
    }

    /// Check if the process is still healthy
    private func checkHealth() {
        guard let process = process else {
            isRunning = false
            isServerReady = false
            return
        }

        if !process.isRunning {
            isRunning = false
            isServerReady = false
            if !isIntentionallyStopping {
                handleUnexpectedTermination()
            }
        }
    }

    /// Handle process termination
    private func handleTermination(exitCode: Int32) {
        isRunning = false
        isServerReady = false
        stdoutPipe?.fileHandleForReading.readabilityHandler = nil
        stderrPipe?.fileHandleForReading.readabilityHandler = nil

        if exitCode == 0 {
            statusMessage = "Stopped"
            addLog("Backend exited normally", level: .info)
        } else {
            statusMessage = "Crashed (exit \(exitCode))"
            addLog("Backend crashed with exit code \(exitCode)", level: .error)

            if !isIntentionallyStopping {
                handleUnexpectedTermination()
            }
        }
    }

    /// Handle unexpected termination (auto-restart)
    private func handleUnexpectedTermination() {
        // Check if we've already exceeded max restart attempts
        guard restartCount < maxRestartAttempts else {
            // Already at max, don't log again
            return
        }

        restartCount += 1
        addLog("Backend terminated unexpectedly (restart #\(restartCount))", level: .warning)

        // Check if we've now exceeded max restart attempts
        if restartCount >= maxRestartAttempts {
            addLog("Max restart attempts (\(maxRestartAttempts)) reached. Giving up.", level: .error)
            addLog("Tip: Kill any existing server with 'lsof -ti:8765 | xargs kill' then click Reset & Retry", level: .error)
            statusMessage = "Failed - click Reset & Retry"
            return
        }

        // Auto-restart with exponential backoff
        statusMessage = "Restarting in \(Int(restartDelay))s..."

        DispatchQueue.main.asyncAfter(deadline: .now() + restartDelay) { [weak self] in
            guard let self = self else { return }
            guard !self.isIntentionallyStopping else { return }
            guard self.restartCount < self.maxRestartAttempts else { return }
            self.start()
        }

        // Increase backoff for next restart
        restartDelay = min(restartDelay * 2, maxRestartDelay)
    }
}
