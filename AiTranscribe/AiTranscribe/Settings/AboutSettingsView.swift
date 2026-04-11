import SwiftUI

// MARK: - About View

struct AboutView: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var backendManager: BackendManager
    @EnvironmentObject var updateChecker: UpdateChecker

    let hasAnimated: Bool
    let onAnimated: () -> Void

    @State private var appeared: Bool
    @State private var iconHovered = false

    private var appVersion: String {
        Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "0.2.0"
    }

    init(hasAnimated: Bool, onAnimated: @escaping () -> Void) {
        self.hasAnimated = hasAnimated
        self.onAnimated = onAnimated
        _appeared = State(initialValue: hasAnimated)
    }

    var body: some View {
        ScrollView {
            VStack(spacing: 0) {
                // ── App Identity ─────────────────────────────────
                appIdentitySection
                    .staggerIn(index: 0, appeared: appeared)

                SettingsDivider()
                    .padding(.top, 4)
                    .staggerIn(index: 1, appeared: appeared)

                // ── Links ───────────────────────────────────���────
                Group {
                    SettingsSectionHeader(title: "Links")
                    linksSection
                }
                .staggerIn(index: 2, appeared: appeared)

                SettingsDivider()
                    .staggerIn(index: 3, appeared: appeared)

                // ── Built With ───────────────────────────────────
                Group {
                    SettingsSectionHeader(title: "Built With")
                    builtWithSection
                }
                .staggerIn(index: 4, appeared: appeared)

                SettingsDivider()
                    .staggerIn(index: 5, appeared: appeared)

                // ── System ─────────────────────────────��─────────
                Group {
                    SettingsSectionHeader(title: "System")
                    systemSection
                }
                .staggerIn(index: 6, appeared: appeared)

                Spacer(minLength: 20)
            }
            .padding(.horizontal, 28)
            .padding(.top, 8)
        }
        .scrollIndicators(.automatic)
        .task(id: "stagger") {
            guard !hasAnimated else { return }
            try? await Task.sleep(for: .milliseconds(80))
            appeared = true
            onAnimated()
        }
    }

    // MARK: - App Identity

    private var appIdentitySection: some View {
        HStack(spacing: 16) {
            // App icon
            Group {
                if let nsImage = loadAppIconFromBundle() {
                    Image(nsImage: nsImage)
                        .resizable()
                        .scaledToFit()
                } else {
                    Image(systemName: "mic.circle.fill")
                        .resizable()
                        .scaledToFit()
                        .foregroundStyle(.primary.opacity(0.6))
                }
            }
            .frame(width: 64, height: 64)
            .clipShape(.rect(cornerRadius: 14, style: .continuous))
            .shadow(color: .black.opacity(0.2), radius: 8, y: 4)
            .scaleEffect(iconHovered ? 1.05 : 1.0)
            .animation(.spring(duration: 0.3, bounce: 0.3), value: iconHovered)
            .onHover { iconHovered = $0 }

            VStack(alignment: .leading, spacing: 4) {
                Text("AiTranscribe")
                    .font(.system(size: 20, weight: .bold, design: .rounded))

                Text("v\(appVersion)")
                    .font(.system(size: 12, weight: .medium, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 2)
                    .background(.quaternary, in: Capsule())

                Text("Open-source speech-to-text for macOS")
                    .font(.system(size: 11))
                    .foregroundStyle(.tertiary)
                    .padding(.top, 1)
            }

            Spacer()
        }
        .padding(.top, 12)
    }

    // MARK: - Links

    private var linksSection: some View {
        VStack(spacing: 4) {
            // GitHub
            AboutLinkRow(
                icon: "chevron.left.forwardslash.chevron.right",
                title: "GitHub Repository",
                subtitle: "View source, report issues, contribute",
                url: "https://github.com/Ljove02/AIT-AiTranscribe-MacOS"
            )

            // Support / Buy me a coffee
            AboutLinkRow(
                icon: "heart.fill",
                title: "Support the Project",
                subtitle: "Buy me a coffee if you find this useful",
                url: "https://ko-fi.com/veljkospasic"
            )
        }
    }

    // MARK: - Built With

    private var builtWithSection: some View {
        HStack(spacing: 8) {
            TechBadge(name: "NVIDIA", detail: "Parakeet & Nemotron", icon: "cpu")
            TechBadge(name: "OpenAI", detail: "Whisper", icon: "waveform")
            TechBadge(name: "Google", detail: "Gemma 4", icon: "text.bubble")
            TechBadge(name: "Apple", detail: "SwiftUI & CoreAudio", icon: "swift")
        }
    }

    // MARK: - System

    private var systemSection: some View {
        VStack(spacing: 4) {
            // Check for Updates row
            HStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Software Update")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(.primary)
                    Text(updateStatusText)
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                }

                Spacer()

                if updateChecker.isUpdateAvailable {
                    Button {
                        updateChecker.showUpdateWindow = true
                    } label: {
                        HStack(spacing: 5) {
                            Circle()
                                .fill(.green)
                                .frame(width: 5, height: 5)
                            Text("View Update")
                                .font(.system(size: 11, weight: .medium))
                        }
                        .foregroundStyle(.green)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 7)
                        .background(.green.opacity(0.12), in: Capsule())
                        .overlay(Capsule().strokeBorder(.green.opacity(0.2), lineWidth: 0.5))
                    }
                    .buttonStyle(.plain)
                } else {
                    Button {
                        Task { await updateChecker.checkForUpdates() }
                    } label: {
                        HStack(spacing: 5) {
                            if case .checking = updateChecker.state {
                                ProgressView()
                                    .controlSize(.small)
                            } else {
                                Image(systemName: "arrow.triangle.2.circlepath")
                                    .font(.system(size: 10, weight: .medium))
                            }
                            Text("Check")
                                .font(.system(size: 11, weight: .medium))
                        }
                        .foregroundStyle(.primary)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 7)
                        .background(.thinMaterial, in: Capsule())
                        .overlay(Capsule().strokeBorder(.white.opacity(0.06), lineWidth: 0.5))
                    }
                    .buttonStyle(.plain)
                    .disabled(updateChecker.state == .checking)
                }
            }
            .padding(.vertical, 10)
            .padding(.horizontal, 14)
            .background(.thinMaterial, in: .rect(cornerRadius: 10, style: .continuous))
            .padding(.vertical, 2)

            SettingsRow(
                title: "Backend Status",
                description: backendManager.statusMessage
            ) {
                Circle()
                    .fill(backendStatusColor)
                    .frame(width: 8, height: 8)
                    .overlay(
                        Circle()
                            .fill(backendStatusColor.opacity(0.3))
                            .frame(width: 16, height: 16)
                    )
            }

            // Debug console row
            HStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Debug Console")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(.primary)
                    Text("View backend logs and diagnostic information.")
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                }

                Spacer()

                Button {
                    DebugConsoleWindowController.shared.showWindow(backendManager: backendManager)
                } label: {
                    HStack(spacing: 5) {
                        Image(systemName: "terminal")
                            .font(.system(size: 10, weight: .medium))
                        Text("Open")
                            .font(.system(size: 11, weight: .medium))
                    }
                    .foregroundStyle(.primary)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 7)
                    .background(.thinMaterial, in: Capsule())
                    .overlay(Capsule().strokeBorder(.white.opacity(0.06), lineWidth: 0.5))
                }
                .buttonStyle(.plain)
            }
            .padding(.vertical, 10)
            .padding(.horizontal, 14)
            .background(.thinMaterial, in: .rect(cornerRadius: 10, style: .continuous))
            .padding(.vertical, 2)

            // Factory Reset
            HStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Factory Reset")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(.primary)
                    Text("Delete all models, history, and settings.")
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                }

                Spacer()

                Button {
                    showFactoryResetConfirmation()
                } label: {
                    HStack(spacing: 5) {
                        Image(systemName: "exclamationmark.triangle")
                            .font(.system(size: 10, weight: .medium))
                        Text("Reset")
                            .font(.system(size: 11, weight: .medium))
                    }
                    .foregroundStyle(.red)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 7)
                    .background(.red.opacity(0.08), in: Capsule())
                    .overlay(Capsule().strokeBorder(.red.opacity(0.15), lineWidth: 0.5))
                }
                .buttonStyle(.plain)
            }
            .padding(.vertical, 10)
            .padding(.horizontal, 14)
            .background(.thinMaterial, in: .rect(cornerRadius: 10, style: .continuous))
            .padding(.vertical, 2)
        }
    }

    // MARK: - Factory Reset

    private func showFactoryResetConfirmation() {
        let alert = NSAlert()
        alert.messageText = "Factory Reset AIT?"
        alert.informativeText = """
        This will permanently delete:

        • All downloaded models
        • All transcription history
        • All app settings and preferences
        • Onboarding completion status
        • NeMo support installation

        This action cannot be undone.
        """
        alert.alertStyle = .critical
        alert.icon = NSImage(systemSymbolName: "exclamationmark.triangle.fill", accessibilityDescription: "Warning")
        alert.addButton(withTitle: "Cancel")
        alert.addButton(withTitle: "Factory Reset")

        let response = alert.runModal()
        if response == .alertSecondButtonReturn {
            performFactoryReset()
        }
    }

    private func performFactoryReset() {
        Task {
            backendManager.stop()
            try? await Task.sleep(nanoseconds: 2_000_000_000)
            appState.clearHistory()

            if appState.isModelLoaded {
                await appState.unloadModel()
            }

            if let bundleID = Bundle.main.bundleIdentifier {
                UserDefaults.standard.removePersistentDomain(forName: bundleID)
                UserDefaults.standard.synchronize()
            }

            HotkeyManager.shared.unregisterAll()

            let fileManager = FileManager.default
            if let appSupport = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first {
                let appFolder = appSupport.appendingPathComponent("AiTranscribe", isDirectory: true)
                try? fileManager.removeItem(at: appFolder)
            }

            let homeDir = fileManager.homeDirectoryForCurrentUser
            let hfCacheDir = homeDir.appendingPathComponent(".cache/huggingface/hub")
            let modelPrefixes = [
                "models--nvidia--parakeet",
                "models--nvidia--nemotron",
                "models--Systran--faster-whisper",
                "models--deepdml--faster-whisper"
            ]

            if let contents = try? fileManager.contentsOfDirectory(at: hfCacheDir, includingPropertiesForKeys: nil) {
                for item in contents {
                    let name = item.lastPathComponent
                    if modelPrefixes.contains(where: { name.hasPrefix($0) }) {
                        try? fileManager.removeItem(at: item)
                    }
                }
            }

            DispatchQueue.main.async {
                let successAlert = NSAlert()
                successAlert.messageText = "Factory Reset Complete"
                successAlert.informativeText = "AIT will now quit. Launch it again to go through onboarding."
                successAlert.alertStyle = .informational
                successAlert.addButton(withTitle: "Quit")
                successAlert.runModal()
                NSApplication.shared.terminate(nil)
            }
        }
    }

    private var updateStatusText: String {
        switch updateChecker.state {
        case .idle: return "Check for the latest version"
        case .checking: return "Checking for updates..."
        case .upToDate: return "You're running the latest version"
        case .available(let version): return "Version \(version) is available"
        case .downloading: return "Downloading update..."
        case .readyToInstall: return "Update ready — relaunch to apply"
        case .error(let msg): return "Error: \(msg)"
        }
    }

    private var backendStatusColor: Color {
        backendManager.isServerReady ? .green : (backendManager.isRunning ? .orange : .red)
    }
}

// MARK: - About Link Row

/// A tappable link row with icon, title, subtitle, and external link indicator.
struct AboutLinkRow: View {
    let icon: String
    let title: String
    let subtitle: String
    let url: String

    @State private var isHovered = false

    var body: some View {
        Link(destination: URL(string: url)!) {
            HStack(spacing: 12) {
                Image(systemName: icon)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(.primary.opacity(0.7))
                    .frame(width: 28, height: 28)
                    .background(.quaternary, in: .rect(cornerRadius: 7, style: .continuous))

                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(.primary)
                    Text(subtitle)
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                }

                Spacer()

                Image(systemName: "arrow.up.right")
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(.tertiary)
                    .offset(x: isHovered ? 2 : 0, y: isHovered ? -2 : 0)
                    .animation(.spring(duration: 0.25, bounce: 0.3), value: isHovered)
            }
            .padding(.vertical, 10)
            .padding(.horizontal, 14)
            .background(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(isHovered ? .white.opacity(0.04) : .clear)
            )
            .background(.thinMaterial, in: .rect(cornerRadius: 10, style: .continuous))
            .padding(.vertical, 2)
        }
        .buttonStyle(.plain)
        .onHover { isHovered = $0 }
    }
}

// MARK: - Tech Badge

/// A compact badge showing a technology/framework used in the app.
struct TechBadge: View {
    let name: String
    let detail: String
    let icon: String

    var body: some View {
        VStack(spacing: 6) {
            Image(systemName: icon)
                .font(.system(size: 16, weight: .light))
                .foregroundStyle(.secondary)

            VStack(spacing: 2) {
                Text(name)
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(.primary)
                Text(detail)
                    .font(.system(size: 9, weight: .medium))
                    .foregroundStyle(.tertiary)
                    .multilineTextAlignment(.center)
                    .lineLimit(2)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 14)
        .padding(.horizontal, 8)
        .background(.thinMaterial, in: .rect(cornerRadius: 10, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(.white.opacity(0.04), lineWidth: 0.5)
        )
    }
}
