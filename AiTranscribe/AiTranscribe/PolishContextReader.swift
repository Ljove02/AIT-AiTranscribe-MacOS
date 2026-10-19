import AppKit
import Foundation

struct PolishRuntimeContext {
    var clipboardText: String? = nil
    var clipboardBaselineChangeCount: Int? = nil
    var screenshotURL: URL? = nil
    var screenshotData: Data? = nil
    var activeAppName: String? = nil
    var desktopAudioTranscript: String? = nil

    var hasContext: Bool {
        clipboardText?.isEmpty == false ||
        screenshotURL != nil ||
        activeAppName?.isEmpty == false ||
        desktopAudioTranscript?.isEmpty == false
    }

    var screenshotDataURL: String? {
        guard let data = screenshotData else { return nil }
        return "data:image/png;base64,\(data.base64EncodedString())"
    }

    var screenshotByteCount: Int {
        screenshotData?.count ?? 0
    }

}

struct PolishRoutableApplication: Identifiable, Hashable, Sendable, Codable {
    var displayName: String
    var bundleIdentifier: String?
    var isRunning: Bool
    var sourcePath: String?

    nonisolated var id: String {
        if let bundleIdentifier, !bundleIdentifier.isEmpty {
            return bundleIdentifier
        }
        return displayName.lowercased()
    }
}

struct PolishRouteApplicationFilterResult: Decodable {
    let hiddenApplicationIds: [String]
}

enum PolishContextReader {
    static func collect(for preset: PolishPreset) -> PolishRuntimeContext {
        PolishRuntimeContext(
            clipboardText: preset.usesClipboardText ? readClipboardText() : nil,
            activeAppName: preset.usesActiveApp ? readActiveAppName() : nil
        )
    }

    static func collectForQuickRun(for preset: PolishPreset) -> PolishRuntimeContext {
        PolishRuntimeContext(
            clipboardBaselineChangeCount: preset.usesClipboardText ? NSPasteboard.general.changeCount : nil,
            activeAppName: preset.usesActiveApp ? readActiveAppName() : nil
        )
    }

    static func captureScreenshotRegion() async -> URL? {
        let destinationURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("ait-polish-screenshot-\(UUID().uuidString).png")

        return await Task.detached(priority: .userInitiated) {
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/usr/sbin/screencapture")
            process.arguments = ["-i", "-x", destinationURL.path]

            do {
                try process.run()
                process.waitUntilExit()
                guard process.terminationStatus == 0,
                      FileManager.default.fileExists(atPath: destinationURL.path),
                      ((try? destinationURL.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0) > 0 else {
                    try? FileManager.default.removeItem(at: destinationURL)
                    return nil
                }
                return destinationURL
            } catch {
                try? FileManager.default.removeItem(at: destinationURL)
                return nil
            }
        }.value
    }

    static func readClipboardText() -> String? {
        let text = NSPasteboard.general.string(forType: .string)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return text?.isEmpty == false ? text : nil
    }

    static func readClipboardTextIfChanged(after baselineChangeCount: Int?) -> String? {
        guard let baselineChangeCount else {
            return readClipboardText()
        }

        guard NSPasteboard.general.changeCount != baselineChangeCount else {
            return nil
        }

        return readClipboardText()
    }

    static func readActiveAppName() -> String? {
        let name = NSWorkspace.shared.frontmostApplication?.localizedName?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return name?.isEmpty == false ? name : nil
    }

    static func readActiveBundleIdentifier() -> String? {
        let bundleIdentifier = NSWorkspace.shared.frontmostApplication?.bundleIdentifier?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return bundleIdentifier?.isEmpty == false ? bundleIdentifier : nil
    }

    @MainActor
    static func scanRoutableApplications(includeInstalled: Bool = true) async -> [PolishRoutableApplication] {
        var appsByKey: [String: PolishRoutableApplication] = [:]

        for app in NSWorkspace.shared.runningApplications {
            guard app.activationPolicy == .regular || app.activationPolicy == .accessory else { continue }
            let name = app.localizedName?.trimmingCharacters(in: .whitespacesAndNewlines)
            guard let name, !name.isEmpty else { continue }
            let bundleIdentifier = app.bundleIdentifier?.trimmingCharacters(in: .whitespacesAndNewlines)
            let entry = PolishRoutableApplication(
                displayName: name,
                bundleIdentifier: bundleIdentifier?.isEmpty == false ? bundleIdentifier : nil,
                isRunning: true,
                sourcePath: app.bundleURL?.path
            )
            appsByKey[entry.id] = entry
        }

        if includeInstalled {
            let installedApps = await scanInstalledApplications()
            for entry in installedApps {
                if var existing = appsByKey[entry.id] {
                    existing.sourcePath = existing.sourcePath ?? entry.sourcePath
                    appsByKey[entry.id] = existing
                } else {
                    appsByKey[entry.id] = entry
                }
            }
        }

        return appsByKey.values.sorted { lhs, rhs in
            if lhs.isRunning != rhs.isRunning { return lhs.isRunning && !rhs.isRunning }
            return lhs.displayName.localizedCaseInsensitiveCompare(rhs.displayName) == .orderedAscending
        }
    }

    nonisolated private static func scanInstalledApplications() async -> [PolishRoutableApplication] {
        await Task.detached(priority: .utility) {
            scanInstalledApplicationsSync()
        }.value
    }

    nonisolated private static func scanInstalledApplicationsSync() -> [PolishRoutableApplication] {
        var appsByKey: [String: PolishRoutableApplication] = [:]
        let appDirectories = [
            URL(fileURLWithPath: "/Applications", isDirectory: true),
            FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Applications", isDirectory: true)
        ]

        let resourceKeys: Set<URLResourceKey> = [.isApplicationKey, .localizedNameKey, .isDirectoryKey]
        for directory in appDirectories where FileManager.default.fileExists(atPath: directory.path) {
            guard let enumerator = FileManager.default.enumerator(
                at: directory,
                includingPropertiesForKeys: Array(resourceKeys),
                options: [.skipsHiddenFiles, .skipsPackageDescendants]
            ) else {
                continue
            }

            for case let url as URL in enumerator where url.pathExtension == "app" {
                let values = try? url.resourceValues(forKeys: resourceKeys)
                guard values?.isApplication == true || url.pathExtension == "app" else { continue }
                let bundle = Bundle(url: url)
                let bundleName = bundle?.object(forInfoDictionaryKey: "CFBundleDisplayName") as? String
                    ?? bundle?.object(forInfoDictionaryKey: "CFBundleName") as? String
                    ?? values?.localizedName
                    ?? url.deletingPathExtension().lastPathComponent
                let name = bundleName.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !name.isEmpty else { continue }
                let bundleIdentifier = bundle?.bundleIdentifier?.trimmingCharacters(in: .whitespacesAndNewlines)
                let entry = PolishRoutableApplication(
                    displayName: name,
                    bundleIdentifier: bundleIdentifier?.isEmpty == false ? bundleIdentifier : nil,
                    isRunning: false,
                    sourcePath: url.path
                )
                appsByKey[entry.id] = entry
            }
        }

        return appsByKey.values.sorted { lhs, rhs in
            return lhs.displayName.localizedCaseInsensitiveCompare(rhs.displayName) == .orderedAscending
        }
    }
}
