/*
 NemoSetupManager.swift
 ======================

 Manages the installation and removal of NeMo support.

 NeMo is NVIDIA's toolkit for Parakeet/Nemotron models. It requires ~3GB
 of dependencies (PyTorch, NeMo toolkit). Since this is large, we install
 it on-demand into a dedicated virtual environment.

 Location: ~/Library/Application Support/AiTranscribe/nemo-venv/

 The manager:
 - Checks for existing Python installation
 - Creates a virtual environment
 - Installs NeMo dependencies
 - Verifies the installation
 - Tracks progress for UI updates
 */

import Foundation
import Combine

/// Errors that can occur during NeMo setup
enum NemoSetupError: Error, LocalizedError {
    case pythonNotFound
    case pythonVersionTooOld(found: String, required: String)
    case venvCreationFailed(reason: String)
    case installationFailed(reason: String)
    case verificationFailed(reason: String)
    case setupScriptNotFound
    case cancelled
    case unknown(Error)

    var errorDescription: String? {
        switch self {
        case .pythonNotFound:
            return "Python 3 is not installed. Please install Python 3.9+ from python.org"
        case .pythonVersionTooOld(let found, let required):
            return "Python version \(found) is too old. Please install Python \(required) or higher"
        case .venvCreationFailed(let reason):
            return "Failed to create virtual environment: \(reason)"
        case .installationFailed(let reason):
            return "Failed to install NeMo: \(reason)"
        case .verificationFailed(let reason):
            return "NeMo installation verification failed: \(reason)"
        case .setupScriptNotFound:
            return "Setup script not found. The app may be corrupted."
        case .cancelled:
            return "Installation was cancelled"
        case .unknown(let error):
            return "An unexpected error occurred: \(error.localizedDescription)"
        }
    }
}

/// Steps in the NeMo setup process
enum NemoSetupStep: String, CaseIterable {
    case idle = "Idle"
    case checkingPython = "Checking Python"
    case creatingVenv = "Creating Environment"
    case upgradingPip = "Upgrading pip"
    case installingPackages = "Installing Packages"
    case verifying = "Verifying"
    case complete = "Complete"
    case error = "Error"

    var displayName: String {
        return rawValue
    }
}

/// Progress event from the setup script
struct NemoSetupProgressEvent: Decodable {
    let step: String
    let progress: Double
    let message: String
    let package: String?
    let details: String?
    let nemoVersion: String?
    let torchVersion: String?
    let venvPath: String?

    enum CodingKeys: String, CodingKey {
        case step, progress, message, package, details
        case nemoVersion = "nemo_version"
        case torchVersion = "torch_version"
        case venvPath = "venv_path"
    }
}

/// Manages NeMo virtual environment setup
@MainActor
class NemoSetupManager: ObservableObject {

    // MARK: - Published State

    /// Whether installation is in progress
    @Published var isInstalling = false

    /// Current step in the setup process
    @Published var currentStep: NemoSetupStep = .idle

    /// Progress value (0.0 to 1.0)
    @Published var progress: Double = 0.0

    /// Current status message
    @Published var statusMessage: String = ""

    /// Currently installing package name (if any)
    @Published var currentPackage: String?

    /// Last error that occurred
    @Published var error: NemoSetupError?

    /// NeMo version after successful installation
    @Published var installedNemoVersion: String?

    /// Whether NeMo venv exists
    @Published var nemoVenvExists: Bool = false

    // MARK: - Private State

    private var installTask: Task<Void, Never>?
    private var process: Process?

    // MARK: - Paths

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
    func checkNemoVenvExists() -> Bool {
        let exists = FileManager.default.fileExists(atPath: nemoPythonPath.path)
        nemoVenvExists = exists
        return exists
    }

    // MARK: - Python Detection

    /// Find Python 3 installation on the system
    func findPython() -> (path: String, version: String)? {
        // Check common Python paths in order of preference
        let pythonPaths = [
            "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3",
            "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3",
            "/Library/Frameworks/Python.framework/Versions/3.10/bin/python3",
            "/Library/Frameworks/Python.framework/Versions/3.9/bin/python3",
            "/opt/homebrew/bin/python3",
            "/usr/local/bin/python3",
            "/usr/bin/python3"
        ]

        for path in pythonPaths {
            if FileManager.default.fileExists(atPath: path) {
                // Get version
                if let version = getPythonVersion(path) {
                    return (path, version)
                }
            }
        }

        return nil
    }

    /// Get Python version from a Python executable
    private func getPythonVersion(_ pythonPath: String) -> String? {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: pythonPath)
        process.arguments = ["--version"]

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe

        do {
            try process.run()
            process.waitUntilExit()

            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            if let output = String(data: data, encoding: .utf8) {
                // Output is "Python X.Y.Z"
                let version = output.trimmingCharacters(in: .whitespacesAndNewlines)
                    .replacingOccurrences(of: "Python ", with: "")
                return version
            }
        } catch {
            return nil
        }

        return nil
    }

    /// Check if Python version meets minimum requirement
    private func isPythonVersionOk(_ version: String) -> Bool {
        let components = version.split(separator: ".").compactMap { Int($0) }
        guard components.count >= 2 else { return false }

        let major = components[0]
        let minor = components[1]

        // Require Python 3.9+
        return major == 3 && minor >= 9
    }

    // MARK: - Installation

    /// Install NeMo in a virtual environment
    func installNemo() async throws {
        guard !isInstalling else { return }

        isInstalling = true
        error = nil
        progress = 0.0
        currentStep = .checkingPython

        defer {
            isInstalling = false
        }

        // Step 1: Check Python
        statusMessage = "Checking Python installation..."

        guard let python = findPython() else {
            throw NemoSetupError.pythonNotFound
        }

        guard isPythonVersionOk(python.version) else {
            throw NemoSetupError.pythonVersionTooOld(found: python.version, required: "3.9")
        }

        statusMessage = "Found Python \(python.version)"
        progress = 0.05

        // Step 2: Find setup script
        // In development, it's in the backend directory
        // In production, it would be bundled with the app
        let setupScriptPath = findSetupScript()
        guard let scriptPath = setupScriptPath else {
            throw NemoSetupError.setupScriptNotFound
        }

        // Ensure Application Support directory exists
        try FileManager.default.createDirectory(at: Self.appSupportURL, withIntermediateDirectories: true)

        // Step 3: Run setup script
        currentStep = .creatingVenv
        statusMessage = "Running NeMo setup..."

        try await runSetupScript(pythonPath: python.path, scriptPath: scriptPath)

        // Update venv exists status
        _ = checkNemoVenvExists()

        currentStep = .complete
        statusMessage = "NeMo installed successfully!"
        progress = 1.0
    }

    /// Find the setup_nemo_venv.py script
    private func findSetupScript() -> String? {
        // Try bundled location first (production)
        if let bundledPath = Bundle.main.path(forResource: "setup_nemo_venv", ofType: "py") {
            return bundledPath
        }

        // Try development locations
        let devPaths = [
            Bundle.main.bundlePath + "/../../../../backend/setup_nemo_venv.py",
            NSHomeDirectory() + "/Projects/AiTranscribe/backend/setup_nemo_venv.py"
        ]

        for path in devPaths {
            let expandedPath = (path as NSString).standardizingPath
            if FileManager.default.fileExists(atPath: expandedPath) {
                return expandedPath
            }
        }

        return nil
    }

    /// Run the setup script and parse its JSON output
    private func runSetupScript(pythonPath: String, scriptPath: String) async throws {
        let process = Process()
        self.process = process
        process.executableURL = URL(fileURLWithPath: pythonPath)
        process.arguments = [scriptPath, nemoVenvPath.path]

        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        // Read output line by line
        let outputHandle = stdoutPipe.fileHandleForReading

        // Set up async reading
        var lastError: NemoSetupError?

        outputHandle.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }

            if let line = String(data: data, encoding: .utf8) {
                for jsonLine in line.components(separatedBy: .newlines) where !jsonLine.isEmpty {
                    Task { @MainActor [weak self] in
                        self?.processProgressLine(jsonLine, lastError: &lastError)
                    }
                }
            }
        }

        do {
            try process.run()
        } catch {
            throw NemoSetupError.unknown(error)
        }

        // Wait for process to complete
        await withCheckedContinuation { continuation in
            process.terminationHandler = { _ in
                continuation.resume()
            }
        }

        // Clean up
        outputHandle.readabilityHandler = nil
        self.process = nil

        // Check exit status
        if process.terminationStatus != 0 {
            if let lastError = lastError {
                throw lastError
            }
            throw NemoSetupError.installationFailed(reason: "Setup script exited with code \(process.terminationStatus)")
        }
    }

    /// Process a line of JSON output from the setup script
    private func processProgressLine(_ line: String, lastError: inout NemoSetupError?) {
        guard let data = line.data(using: .utf8) else { return }

        do {
            let event = try JSONDecoder().decode(NemoSetupProgressEvent.self, from: data)

            // Update state based on event
            switch event.step {
            case "checking_python":
                currentStep = .checkingPython
            case "creating_venv":
                currentStep = .creatingVenv
            case "upgrading_pip":
                currentStep = .upgradingPip
            case "installing_packages":
                currentStep = .installingPackages
                currentPackage = event.package
            case "verifying":
                currentStep = .verifying
            case "complete":
                currentStep = .complete
                installedNemoVersion = event.nemoVersion
            case "error":
                currentStep = .error
                lastError = .installationFailed(reason: event.message)
            default:
                break
            }

            if event.progress >= 0 {
                progress = event.progress
            }
            statusMessage = event.message
        } catch {
            // Ignore non-JSON lines (debug output, etc.)
            print("NemoSetupManager: Non-JSON line: \(line)")
        }
    }

    // MARK: - Removal

    /// Remove the NeMo virtual environment
    func removeNemoVenv() async throws {
        guard nemoVenvExists else { return }

        statusMessage = "Removing NeMo environment..."

        do {
            try FileManager.default.removeItem(at: nemoVenvPath)
            nemoVenvExists = false
            installedNemoVersion = nil
            statusMessage = "NeMo environment removed"
        } catch {
            throw NemoSetupError.unknown(error)
        }
    }

    // MARK: - Cancellation

    /// Cancel the current installation
    func cancel() {
        process?.terminate()
        isInstalling = false
        currentStep = .idle
        statusMessage = "Installation cancelled"
        error = .cancelled
    }
}
