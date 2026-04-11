import Foundation
import Combine

enum SummarySetupError: Error, LocalizedError {
    case pythonNotFound
    case pythonVersionTooOld(found: String, required: String)
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

    func checkSummaryVenvExists() -> Bool {
        let exists = FileManager.default.fileExists(atPath: summaryPythonPath.path)
        summaryVenvExists = exists
        return exists
    }

    func findPython() -> (path: String, version: String)? {
        let pythonPaths = [
            "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3",
            "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3",
            "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3",
            "/Library/Frameworks/Python.framework/Versions/3.10/bin/python3",
            "/opt/homebrew/bin/python3",
            "/usr/local/bin/python3",
            "/usr/bin/python3",
        ]

        for path in pythonPaths {
            guard FileManager.default.fileExists(atPath: path), let version = getPythonVersion(path) else {
                continue
            }
            return (path, version)
        }
        return nil
    }

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
                return output
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                    .replacingOccurrences(of: "Python ", with: "")
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
        guard let scriptPath = findSetupScript() else {
            throw SummarySetupError.setupScriptNotFound
        }

        try FileManager.default.createDirectory(at: Self.appSupportURL, withIntermediateDirectories: true)

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
                    guard let data = line.data(using: .utf8),
                          let event = try? JSONDecoder().decode(SummarySetupProgressEvent.self, from: data) else {
                        continue
                    }
                    update(with: event)
                }
            }
            process.waitUntilExit()
            if process.terminationStatus != 0 {
                let data = stderr.fileHandleForReading.readDataToEndOfFile()
                let message = String(data: data, encoding: .utf8) ?? "Unknown error"
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
