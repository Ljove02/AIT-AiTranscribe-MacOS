import Foundation
import SwiftUI
import Combine

// MARK: - GitHub Release Model

struct GitHubRelease: Codable {
    let tagName: String
    let name: String?
    let body: String?
    let htmlUrl: String
    let assets: [GitHubAsset]
    let publishedAt: String?

    enum CodingKeys: String, CodingKey {
        case tagName = "tag_name"
        case name
        case body
        case htmlUrl = "html_url"
        case assets
        case publishedAt = "published_at"
    }
}

struct GitHubAsset: Codable {
    let name: String
    let browserDownloadUrl: String
    let size: Int
    let contentType: String

    enum CodingKeys: String, CodingKey {
        case name
        case browserDownloadUrl = "browser_download_url"
        case size
        case contentType = "content_type"
    }
}

// MARK: - Update State

enum UpdateState: Equatable {
    case idle
    case checking
    case upToDate
    case available(version: String)
    case downloading(progress: Double)
    case readyToInstall
    case error(String)

    static func == (lhs: UpdateState, rhs: UpdateState) -> Bool {
        switch (lhs, rhs) {
        case (.idle, .idle), (.checking, .checking), (.upToDate, .upToDate), (.readyToInstall, .readyToInstall):
            return true
        case (.available(let a), .available(let b)):
            return a == b
        case (.downloading(let a), .downloading(let b)):
            return a == b
        case (.error(let a), .error(let b)):
            return a == b
        default:
            return false
        }
    }
}

// MARK: - Update Checker

@MainActor
class UpdateChecker: ObservableObject {
    static let shared = UpdateChecker()

    private let repoOwner = "Ljove02"
    private let repoName = "AIT-AiTranscribe-MacOS"
    private let checkIntervalHours: Double = 24

    @Published var state: UpdateState = .idle
    @Published var latestRelease: GitHubRelease?
    @Published var showUpdateWindow = false

    private var downloadTask: URLSessionDownloadTask?
    private var downloadedFileURL: URL?

    var currentVersion: String {
        Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "0.0.0"
    }

    var isUpdateAvailable: Bool {
        if case .available = state { return true }
        if case .readyToInstall = state { return true }
        return false
    }

    var latestVersion: String? {
        if case .available(let version) = state { return version }
        return nil
    }

    // MARK: - Check for Updates

    func checkForUpdates() async {
        state = .checking

        let urlString = "https://api.github.com/repos/\(repoOwner)/\(repoName)/releases/latest"
        guard let url = URL(string: urlString) else {
            state = .error("Invalid URL")
            return
        }

        var request = URLRequest(url: url)
        request.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
        request.cachePolicy = .reloadIgnoringLocalCacheData

        do {
            let (data, response) = try await URLSession.shared.data(for: request)

            guard let httpResponse = response as? HTTPURLResponse else {
                state = .error("Invalid response")
                return
            }

            guard httpResponse.statusCode == 200 else {
                if httpResponse.statusCode == 404 {
                    // No releases yet
                    state = .upToDate
                    return
                }
                state = .error("GitHub API error (\(httpResponse.statusCode))")
                return
            }

            let decoder = JSONDecoder()
            let release = try decoder.decode(GitHubRelease.self, from: data)
            latestRelease = release

            let remoteVersion = release.tagName.trimmingCharacters(in: CharacterSet(charactersIn: "vV"))

            if isNewerVersion(remoteVersion, than: currentVersion) {
                state = .available(version: remoteVersion)
            } else {
                state = .upToDate
            }
        } catch {
            state = .error(error.localizedDescription)
        }
    }

    // MARK: - Download & Install

    func downloadAndInstall() async {
        guard let release = latestRelease else { return }

        // Look for a .zip or .dmg asset
        let appAsset = release.assets.first { asset in
            let name = asset.name.lowercased()
            return name.hasSuffix(".zip") || name.hasSuffix(".dmg")
        }

        guard let asset = appAsset else {
            // No downloadable asset — open the release page instead
            openReleasePage()
            return
        }

        guard let downloadURL = URL(string: asset.browserDownloadUrl) else {
            state = .error("Invalid download URL")
            return
        }

        state = .downloading(progress: 0)

        do {
            let (tempURL, _) = try await URLSession.shared.download(from: downloadURL, delegate: nil)

            // Move to a known location
            let destDir = FileManager.default.temporaryDirectory.appendingPathComponent("AiTranscribeUpdate")
            try? FileManager.default.removeItem(at: destDir)
            try FileManager.default.createDirectory(at: destDir, withIntermediateDirectories: true)

            let destFile = destDir.appendingPathComponent(asset.name)
            try FileManager.default.moveItem(at: tempURL, to: destFile)
            downloadedFileURL = destFile

            if asset.name.lowercased().hasSuffix(".zip") {
                let success = await unzipAndReplace(zipURL: destFile)
                if success {
                    state = .readyToInstall
                } else {
                    state = .error("Failed to extract update")
                }
            } else {
                // For .dmg, just open it and let the user drag
                NSWorkspace.shared.open(destFile)
                state = .readyToInstall
            }
        } catch {
            state = .error("Download failed: \(error.localizedDescription)")
        }
    }

    func openReleasePage() {
        guard let release = latestRelease,
              let url = URL(string: release.htmlUrl) else { return }
        NSWorkspace.shared.open(url)
    }

    // MARK: - Auto-update: unzip and replace

    private func unzipAndReplace(zipURL: URL) async -> Bool {
        let extractDir = zipURL.deletingLastPathComponent().appendingPathComponent("extracted")
        try? FileManager.default.removeItem(at: extractDir)

        // Unzip using ditto (preserves code signing)
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/ditto")
        process.arguments = ["-xk", zipURL.path, extractDir.path]

        do {
            try process.run()
            process.waitUntilExit()

            guard process.terminationStatus == 0 else { return false }

            // Find the .app in extracted directory
            let contents = try FileManager.default.contentsOfDirectory(at: extractDir, includingPropertiesForKeys: nil)
            guard let newApp = contents.first(where: { $0.pathExtension == "app" }) else {
                return false
            }

            // Get the current app location
            let currentAppURL = Bundle.main.bundleURL

            // Move current app to trash
            var trashedURL: NSURL?
            try FileManager.default.trashItem(at: currentAppURL, resultingItemURL: &trashedURL)

            // Copy new app to the same location
            try FileManager.default.copyItem(at: newApp, to: currentAppURL)

            return true
        } catch {
            print("Update extraction failed: \(error)")
            return false
        }
    }

    func relaunchApp() {
        let appURL = Bundle.main.bundleURL
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/usr/bin/open")
        task.arguments = ["-n", appURL.path]

        do {
            try task.run()
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                NSApp.terminate(nil)
            }
        } catch {
            print("Relaunch failed: \(error)")
        }
    }

    // MARK: - Semver Comparison

    private func isNewerVersion(_ remote: String, than local: String) -> Bool {
        let remoteParts = remote.split(separator: ".").compactMap { Int($0) }
        let localParts = local.split(separator: ".").compactMap { Int($0) }

        let maxLen = max(remoteParts.count, localParts.count)
        for i in 0..<maxLen {
            let r = i < remoteParts.count ? remoteParts[i] : 0
            let l = i < localParts.count ? localParts[i] : 0
            if r > l { return true }
            if r < l { return false }
        }
        return false
    }
}
