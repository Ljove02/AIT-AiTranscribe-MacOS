import Foundation
import Combine

enum SummarySetupError: Error, LocalizedError {
    case pythonNotFound
    case pythonVersionTooOld(found: String, required: String)
    case pythonArchitectureUnsupported(found: String)
    case setupScriptNotFound
    case installationFailed(reason: String)
    case cancelled
    case unknown(Error)

    var errorDescription: String? {
        switch self {
        case .pythonNotFound:
            return "Python 3 is not installed. Please install Python 3.10+ from python.org"
        case .pythonVersionTooOld(let found, let required):
            return "Python version \(found) is too old. Please install Python \(required)+"
        case .pythonArchitectureUnsupported(let found):
            return "Summary runtime requires Apple Silicon and a native arm64 Python. Found \(found)."
        case .setupScriptNotFound:
            return "Summary setup script not found. The app may be corrupted."
        case .installationFailed(let reason):
            return "Failed to install summary runtime: \(reason)"
        case .cancelled:
            return "Installation was cancelled"
        case .unknown(let error):
            return error.localizedDescription
        }
    }
}

enum SummarySetupStep: String, CaseIterable {
    case idle = "Idle"
    case checkingPython = "Checking Python"
    case creatingVenv = "Creating Runtime"
    case upgradingPip = "Upgrading pip"
    case installingPackages = "Installing Packages"
    case verifying = "Verifying"
    case complete = "Complete"
    case error = "Error"
}

struct SummarySetupProgressEvent: Decodable {
    let step: String
    let progress: Double
    let message: String
    let package: String?
    let details: String?
    let mlxVersion: String?
    let mlxVlmVersion: String?
    let venvPath: String?

    enum CodingKeys: String, CodingKey {
        case step, progress, message, package, details
        case mlxVersion = "mlx_version"
        case mlxVlmVersion = "mlx_vlm_version"
        case venvPath = "venv_path"
    }
}

struct PythonInstallationInfo {
    let path: String
    let version: String
    let machine: String
}

@MainActor
class SummarySetupManager: ObservableObject {
    @Published var isInstalling = false
    @Published var currentStep: SummarySetupStep = .idle
    @Published var progress: Double = 0.0
    @Published var statusMessage: String = ""
    @Published var currentPackage: String?
    @Published var error: SummarySetupError?
    @Published var summaryVenvExists: Bool = false
    @Published var installedMLXVersion: String?
    @Published var installedMLXVLMVersion: String?

    private var process: Process?

    static var appSupportURL: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("AiTranscribe")
    }

    var summaryVenvPath: URL {
        Self.appSupportURL.appendingPathComponent("summary-venv")
    }

    var summaryPythonPath: URL {
        summaryVenvPath.appendingPathComponent("bin").appendingPathComponent("python3")
    }

    private var installationLogURL: URL {
        Self.appSupportURL.appendingPathComponent("summary-runtime-install.log")
    }

    func checkSummaryVenvExists() -> Bool {
        let exists = FileManager.default.fileExists(atPath: summaryPythonPath.path)
        summaryVenvExists = exists
        return exists
    }

    func findPython() -> PythonInstallationInfo? {
        let pythonPaths = [
            "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3",
            "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3",
            "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3",
            "/Library/Frameworks/Python.framework/Versions/3.10/bin/python3",
            "/opt/homebrew/bin/python3",
            "/usr/local/bin/python3",
            "/usr/bin/python3",
        ]

        var fallback: PythonInstallationInfo?
        for path in pythonPaths {
            guard FileManager.default.fileExists(atPath: path),
                  let info = getPythonInfo(path),
                  isPythonVersionOk(info.version) else {
                continue
            }
            if info.machine == "arm64" {
                return info
            }
            if fallback == nil {
                fallback = info
            }
        }
        return fallback
    }

    private func getPythonInfo(_ pythonPath: String) -> PythonInstallationInfo? {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: pythonPath)
        process.arguments = [
            "-c",
            """
            import json, platform
            print(json.dumps({
                "version": platform.python_version(),
                "machine": platform.machine()
            }))
            """,
        ]

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe

        do {
            try process.run()
            process.waitUntilExit()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            if process.terminationStatus != 0 {
                return nil
            }
            if let output = String(data: data, encoding: .utf8),
               let jsonData = output.data(using: .utf8),
               let json = try JSONSerialization.jsonObject(with: jsonData) as? [String: Any],
               let version = json["version"] as? String,
               let machine = json["machine"] as? String {
                return PythonInstallationInfo(path: pythonPath, version: version, machine: machine)
            }
        } catch {
            return nil
        }
        return nil
    }

    private func isPythonVersionOk(_ version: String) -> Bool {
        let components = version.split(separator: ".").compactMap { Int($0) }
        guard components.count >= 2 else { return false }
        return components[0] == 3 && components[1] >= 10
    }

    private func resetInstallationLog() {
        let header = "Summary runtime install log\n"
        try? header.write(to: installationLogURL, atomically: true, encoding: .utf8)
    }

    private func appendInstallationLog(_ line: String) {
        guard let data = line.data(using: .utf8) else { return }
        if FileManager.default.fileExists(atPath: installationLogURL.path),
           let handle = try? FileHandle(forWritingTo: installationLogURL) {
            defer { try? handle.close() }
            _ = try? handle.seekToEnd()
            try? handle.write(contentsOf: data)
        } else {
            try? data.write(to: installationLogURL)
        }
    }

    private func findSetupScript() -> String? {
        if let bundled = Bundle.main.path(forResource: "setup_summary_venv", ofType: "py") {
            return bundled
        }

        if let envPath = ProcessInfo.processInfo.environment["AITRANSCRIBE_BACKEND_PATH"] {
            let path = (envPath as NSString).appendingPathComponent("setup_summary_venv.py")
            if FileManager.default.fileExists(atPath: path) {
                return path
            }
        }

        let possiblePaths = [
            Bundle.main.bundlePath + "/../../../../backend/setup_summary_venv.py",
            Bundle.main.bundlePath + "/../../../../../backend/setup_summary_venv.py",
            Bundle.main.bundlePath + "/../../../../../../backend/setup_summary_venv.py",
        ]

        for path in possiblePaths {
            let standardized = (path as NSString).standardizingPath
            if FileManager.default.fileExists(atPath: standardized) {
                return standardized
            }
        }

        let backendPath = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("backend/setup_summary_venv.py").path
        if FileManager.default.fileExists(atPath: backendPath) {
            return backendPath
        }

        return nil
    }

    func installRuntime() async throws {
        guard !isInstalling else { return }

        isInstalling = true
        error = nil
        progress = 0
        currentStep = .checkingPython
        currentPackage = nil

        defer {
            isInstalling = false
        }

        guard let python = findPython() else {
            throw SummarySetupError.pythonNotFound
        }
        guard isPythonVersionOk(python.version) else {
            throw SummarySetupError.pythonVersionTooOld(found: python.version, required: "3.10")
        }
        guard python.machine == "arm64" else {
            throw SummarySetupError.pythonArchitectureUnsupported(found: python.machine)
        }
        guard let scriptPath = findSetupScript() else {
            throw SummarySetupError.setupScriptNotFound
        }

        try FileManager.default.createDirectory(at: Self.appSupportURL, withIntermediateDirectories: true)
        resetInstallationLog()
        appendInstallationLog("python_path=\(python.path)\n")
        appendInstallationLog("python_version=\(python.version)\n")
        appendInstallationLog("python_machine=\(python.machine)\n")
        appendInstallationLog("setup_script=\(scriptPath)\n")
        appendInstallationLog("venv_path=\(summaryVenvPath.path)\n")

        statusMessage = "Installing summary runtime..."

        let process = Process()
        self.process = process
        process.executableURL = URL(fileURLWithPath: python.path)
        process.arguments = [scriptPath, summaryVenvPath.path]

        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr

        try process.run()

        var lastProgressEvent: SummarySetupProgressEvent?
        let stream = AsyncThrowingStream<String, Error> { continuation in
            stdout.fileHandleForReading.readabilityHandler = { handle in
                let data = handle.availableData
                if data.isEmpty {
                    continuation.finish()
                } else if let text = String(data: data, encoding: .utf8) {
                    continuation.yield(text)
                }
            }
        }

        do {
            for try await chunk in stream {
                for line in chunk.split(separator: "\n") {
                    appendInstallationLog(String(line) + "\n")
                    guard let data = line.data(using: .utf8),
                          let event = try? JSONDecoder().decode(SummarySetupProgressEvent.self, from: data) else {
                        continue
                    }
                    lastProgressEvent = event
                    update(with: event)
                }
            }
            process.waitUntilExit()
            if process.terminationStatus != 0 {
                let data = stderr.fileHandleForReading.readDataToEndOfFile()
                let stderrText = String(data: data, encoding: .utf8)?
                    .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                if !stderrText.isEmpty {
                    appendInstallationLog(stderrText + "\n")
                }
                let message: String
                if !stderrText.isEmpty {
                    message = stderrText
                } else if let event = lastProgressEvent {
                    message = event.details ?? event.message
                } else {
                    message = "Unknown error. See \(installationLogURL.path)"
                }
                throw SummarySetupError.installationFailed(reason: message)
            }
        } catch let error as SummarySetupError {
            self.error = error
            currentStep = .error
            throw error
        } catch {
            let wrapped = SummarySetupError.unknown(error)
            self.error = wrapped
            currentStep = .error
            throw wrapped
        }

        _ = checkSummaryVenvExists()
    }

    private func update(with event: SummarySetupProgressEvent) {
        progress = event.progress
        statusMessage = event.message
        currentPackage = event.package
        installedMLXVersion = event.mlxVersion ?? installedMLXVersion
        installedMLXVLMVersion = event.mlxVlmVersion ?? installedMLXVLMVersion

        switch event.step {
        case "checking_python":
            currentStep = .checkingPython
        case "creating_venv":
            currentStep = .creatingVenv
        case "upgrading_pip":
            currentStep = .upgradingPip
        case "installing_packages":
            currentStep = .installingPackages
        case "verifying":
            currentStep = .verifying
        case "complete":
            currentStep = .complete
        case "error":
            currentStep = .error
            error = .installationFailed(reason: event.details ?? event.message)
        default:
            break
        }
    }

    func cancelInstall() {
        process?.terminate()
        process = nil
        error = .cancelled
        currentStep = .error
        isInstalling = false
    }

    func removeRuntime() {
        guard summaryVenvExists else { return }
        do {
            try FileManager.default.removeItem(at: summaryVenvPath)
            summaryVenvExists = false
            installedMLXVersion = nil
            installedMLXVLMVersion = nil
            currentStep = .idle
            progress = 0
            statusMessage = ""
        } catch {
            self.error = .unknown(error)
        }
    }
}
