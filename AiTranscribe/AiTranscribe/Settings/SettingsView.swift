/*
 SettingsView.swift
 ==================

 The Settings window — fully custom chrome.
 One cohesive rounded bubble with integrated top bar,
 solid-glass sidebar, and transparent-blur content area.

 Uses a Window scene (not Settings) for full control
 over the title bar and window chrome.
 */

import SwiftUI
import AppKit

// MARK: - Transparent Window Helper

/// Strips all native chrome from the hosting NSWindow so we can
/// draw a single cohesive rounded-bubble container in SwiftUI.
struct TransparentWindow: NSViewRepresentable {
    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        DispatchQueue.main.async {
            guard let window = view.window else { return }
            window.isOpaque = false
            window.backgroundColor = .clear
            window.titlebarAppearsTransparent = true
            window.titleVisibility = .hidden
            window.isMovableByWindowBackground = true
            window.toolbar = nil
            window.titlebarSeparatorStyle = .none
            window.styleMask.insert(.fullSizeContentView)

            // Hide the standard window buttons — we draw our own
            window.standardWindowButton(.closeButton)?.isHidden = true
            window.standardWindowButton(.miniaturizeButton)?.isHidden = true
            window.standardWindowButton(.zoomButton)?.isHidden = true
        }
        return view
    }

    func updateNSView(_ nsView: NSView, context: Context) {}
}

// MARK: - Settings Window Focus

/// Brings the Settings window to the front and makes it *key*.
///
/// This exists because a fixed `asyncAfter` hop is not enough. When the
/// MenuBarExtra panel dismisses, macOS hands activation back to whatever app
/// was frontmost before the menu opened — and that handoff can land after our
/// delay, stealing key status right back. A non-key window renders every
/// accent-tinted control (NSSwitch, NSPopUpButton) in gray, which is what the
/// Settings toggles looked like.
///
/// `activate(ignoringOtherApps:)` is also deprecated on macOS 14+ and the
/// system frequently declines the "ignoring other apps" part, so a single shot
/// is not reliable either. Instead we poll: wait for the window to exist, ask
/// for activation, then verify `isKeyWindow` and retry until it sticks.
enum SettingsWindowFocus {
    private static let retryInterval: TimeInterval = 0.05
    private static let maxAttempts = 24 // ~1.2s of retries

    static func focus() {
        attempt(remaining: maxAttempts)
    }

    private static func attempt(remaining: Int) {
        guard remaining > 0 else { return }

        guard let window = settingsWindow() else {
            // Window scene hasn't been instantiated yet — wait for it.
            retry(remaining: remaining)
            return
        }

        // Start with the modern cooperative API. If it hasn't taken by the
        // halfway mark, fall back to the deprecated form, which still wins in
        // some accessory-app cases.
        activateApp(force: remaining <= maxAttempts / 2)
        window.makeKeyAndOrderFront(nil)

        // Verify on the next pass: activation is asynchronous, and the
        // MenuBarExtra dismissal may still be in flight.
        if !window.isKeyWindow || !NSApp.isActive {
            retry(remaining: remaining)
        }
    }

    private static func retry(remaining: Int) {
        DispatchQueue.main.asyncAfter(deadline: .now() + retryInterval) {
            attempt(remaining: remaining - 1)
        }
    }

    private static func activateApp(force: Bool) {
        if #available(macOS 14.0, *), !force {
            NSApp.activate()
        } else {
            NSApp.activate(ignoringOtherApps: true)
        }
    }

    private static func settingsWindow() -> NSWindow? {
        NSApp.windows.first { window in
            guard window.isVisible || window.isMiniaturized || window.canBecomeKey else { return false }
            return window.identifier?.rawValue.contains("settings") == true
                || window.title.contains("Settings")
        }
    }
}

// MARK: - Settings View

struct SettingsView: View {
    @State private var selectedSection: SettingsSection = .dashboard
    @State private var isSidebarExpanded: Bool = false
    @State private var animatedTabs: Set<SettingsSection> = []
    @State private var pendingHistoryEntryId: UUID? = nil
    @State private var pendingSessionId: UUID? = nil
    @State private var pendingModelMode: ModelMode = .speechToText
    @StateObject private var dashboardStatsManager = DashboardStatsManager()
    @EnvironmentObject var backendManager: BackendManager
    @EnvironmentObject var updateChecker: UpdateChecker

    var body: some View {
        ZStack {
            // Hook into NSWindow
            TransparentWindow()
                .frame(width: 0, height: 0)

            // One cohesive bubble container
            VStack(spacing: 0) {
                // Custom top bar (replaces native title bar)
                SettingsTopBar(
                    selectedSection: selectedSection,
                    isSidebarExpanded: $isSidebarExpanded
                )

                // Main split: sidebar + content
                HStack(spacing: 0) {
                    SettingsSidebar(
                        selectedSection: $selectedSection,
                        isExpanded: isSidebarExpanded
                    )

                    // Content area
                    settingsContent
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
            .background(.ultraThinMaterial)
            .clipShape(.rect(cornerRadius: 18, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .strokeBorder(.white.opacity(0.08), lineWidth: 0.5)
            )
            .shadow(color: .black.opacity(0.3), radius: 20, x: 0, y: 8)
        }
        .ignoresSafeArea()
        .frame(width: 1080, height: 740)
        .animation(.spring(duration: 0.35, bounce: 0.12), value: isSidebarExpanded)
        .onAppear {
            if backendManager.isServerReady {
                backendManager.pauseHealthCheck()
            }
        }
        .onDisappear {
            backendManager.resumeHealthCheck()
        }
        .sheet(isPresented: $updateChecker.showUpdateWindow) {
            UpdateWindowView()
                .environmentObject(updateChecker)
        }
    }

    @ViewBuilder
    private var settingsContent: some View {
        switch selectedSection {
        case .dashboard:
            DashboardView(
                statsManager: dashboardStatsManager,
                hasAnimated: animatedTabs.contains(.dashboard),
                onAnimated: { animatedTabs.insert(.dashboard) },
                onNavigateToHistory: { entryId in
                    pendingHistoryEntryId = entryId
                    // Clear animation cache so the view reinits with the new ID
                    animatedTabs.remove(.history)
                    withAnimation(.easeInOut(duration: 0.2)) {
                        selectedSection = .history
                    }
                },
                onNavigateToSessions: { sessionId in
                    pendingSessionId = sessionId
                    animatedTabs.remove(.sessions)
                    withAnimation(.easeInOut(duration: 0.2)) {
                        selectedSection = .sessions
                    }
                }
            )
        case .general:
            GeneralSettingsView(hasAnimated: animatedTabs.contains(.general), onAnimated: {
                animatedTabs.insert(.general)
            })
        case .polish:
            PolishSettingsView(hasAnimated: animatedTabs.contains(.polish), onAnimated: {
                animatedTabs.insert(.polish)
            })
        case .models:
            ModelsSettingsView(
                hasAnimated: animatedTabs.contains(.models),
                onAnimated: { animatedTabs.insert(.models) },
                initialMode: pendingModelMode
            )
            .onAppear { pendingModelMode = .speechToText }
        case .sessions:
            SessionsSettingsView(
                hasAnimated: animatedTabs.contains(.sessions),
                onAnimated: { animatedTabs.insert(.sessions) },
                onNavigateToModels: { mode in
                    pendingModelMode = mode
                    animatedTabs.remove(.models)
                    withAnimation(.easeInOut(duration: 0.2)) {
                        selectedSection = .models
                    }
                },
                initialSelectedSessionId: pendingSessionId
            )
            .onAppear { pendingSessionId = nil }
        case .history:
            HistorySettingsView(
                hasAnimated: animatedTabs.contains(.history),
                onAnimated: { animatedTabs.insert(.history) },
                initialSelectedEntryId: pendingHistoryEntryId
            )
            .onAppear { pendingHistoryEntryId = nil }
        case .shortcuts:
            ShortcutsSettingsView(hasAnimated: animatedTabs.contains(.shortcuts), onAnimated: {
                animatedTabs.insert(.shortcuts)
            })
        case .about:
            AboutView(hasAnimated: animatedTabs.contains(.about), onAnimated: {
                animatedTabs.insert(.about)
            })
        }
    }
}

// MARK: - Custom Top Bar

/// Custom window chrome: close/minimize buttons + sidebar toggle + section title pinned right.
struct SettingsTopBar: View {
    let selectedSection: SettingsSection
    @Binding var isSidebarExpanded: Bool
    @EnvironmentObject var updateChecker: UpdateChecker
    @State private var isCloseHovered = false
    @State private var isMinimizeHovered = false
    @State private var updateButtonHovered = false

    var body: some View {
        HStack(spacing: 0) {
            // Left zone: traffic lights + sidebar toggle
            HStack(spacing: 10) {
                // Custom traffic light buttons (no zoom — fixed size window)
                HStack(spacing: 8) {
                    WindowButton(color: .red, systemImage: "xmark", isHovered: $isCloseHovered) {
                        NSApp.keyWindow?.close()
                    }
                    WindowButton(color: .yellow, systemImage: "minus", isHovered: $isMinimizeHovered) {
                        NSApp.keyWindow?.miniaturize(nil)
                    }
                }
                .padding(.leading, 16)

                // Sidebar toggle — circular bubble
                Button {
                    withAnimation(.spring(duration: 0.35, bounce: 0.12)) {
                        isSidebarExpanded.toggle()
                    }
                } label: {
                    Image(systemName: "sidebar.left")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(.secondary)
                        .frame(width: 28, height: 28)
                        .background(.quaternary, in: .circle)
                }
                .buttonStyle(.plain)

                TranscriptionModeSwitch()
            }

            Spacer()

            // Update available pill
            if updateChecker.isUpdateAvailable {
                Button {
                    updateChecker.showUpdateWindow = true
                } label: {
                    HStack(spacing: 5) {
                        Circle()
                            .fill(.green)
                            .frame(width: 6, height: 6)
                        Text("Update Available")
                            .font(.system(size: 11, weight: .semibold))
                    }
                    .foregroundStyle(updateButtonHovered ? .white : .green)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 5)
                    .background(
                        Capsule()
                            .fill(updateButtonHovered ? .green : .green.opacity(0.15))
                    )
                    .overlay(Capsule().strokeBorder(.green.opacity(0.3), lineWidth: 0.5))
                }
                .buttonStyle(.plain)
                .onHover { updateButtonHovered = $0 }
                .transition(.opacity.combined(with: .scale(scale: 0.8)))
                .animation(.spring(duration: 0.3), value: updateChecker.isUpdateAvailable)
                .padding(.trailing, 10)
            }

            // Right: section title pinned to the right so it doesn't shift
            Text(selectedSection.label)
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(.secondary)
                .padding(.trailing, 20)
        }
        .frame(height: 48)
    }
}

struct TranscriptionModeSwitch: View {
    @AppStorage("selectedTranscriptionProviderId") private var selectedTranscriptionProviderId = TranscriptionProviderDescriptor.localBackend.id
    @AppStorage("hasSeenTranscriptionModeInfo") private var hasSeenTranscriptionModeInfo = false
    @State private var showingModeInfo = false

    private var isLocalMode: Bool {
        selectedTranscriptionProviderId == TranscriptionProviderDescriptor.localBackend.id
    }

    var body: some View {
        HStack(spacing: 6) {
            HStack(spacing: 2) {
                modeButton(title: "Local", isSelected: isLocalMode, tint: .green, selectedForeground: .green, selectedOpacity: 0.14) {
                    selectedTranscriptionProviderId = TranscriptionProviderDescriptor.localBackend.id
                }

                modeButton(title: "Public", isSelected: !isLocalMode, tint: .blue, selectedForeground: .white, selectedOpacity: 1.0) {
                    if selectedTranscriptionProviderId == TranscriptionProviderDescriptor.localBackend.id {
                        selectedTranscriptionProviderId = APITranscriptionModelStore.load().first?.providerId
                            ?? TranscriptionProviderDescriptor.openAI.id
                    }
                }
            }
            .padding(3)
            .background(Color.primary.opacity(0.06), in: Capsule())
            .overlay(Capsule().strokeBorder(.white.opacity(0.08), lineWidth: 0.5))

            Button {
                showingModeInfo = true
            } label: {
                ZStack {
                    Circle()
                        .fill(.quaternary)
                        .frame(width: 24, height: 24)

                    Image(systemName: "info.circle")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(.secondary)
                }
                .frame(width: 30, height: 30, alignment: .center)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help("How Local/Public mode works")

            if isLocalMode {
                ZStack {
                    Circle()
                        .fill(.green.opacity(0.14))
                        .frame(width: 24, height: 24)

                    Image(systemName: "lock.shield.fill")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(.green)
                }
                .frame(width: 30, height: 30, alignment: .center)
                .help("Local mode: audio and transcripts stay on this Mac.")
                .transition(.opacity.combined(with: .scale(scale: 0.8)))
            }
        }
        .onAppear {
            guard !hasSeenTranscriptionModeInfo else { return }
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.45) {
                showingModeInfo = true
            }
        }
        .sheet(isPresented: $showingModeInfo, onDismiss: {
            hasSeenTranscriptionModeInfo = true
        }) {
            TranscriptionModeInfoSheet {
                hasSeenTranscriptionModeInfo = true
                showingModeInfo = false
            }
        }
    }

    private func modeButton(
        title: String,
        isSelected: Bool,
        tint: Color,
        selectedForeground: Color,
        selectedOpacity: Double,
        action: @escaping () -> Void
    ) -> some View {
        Button {
            withAnimation(.spring(duration: 0.28, bounce: 0.16)) {
                action()
            }
        } label: {
            Text(title)
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(isSelected ? selectedForeground : Color.secondary)
                .padding(.horizontal, 11)
                .frame(height: 24)
                .background(
                    Capsule()
                        .fill(isSelected ? tint.opacity(selectedOpacity) : Color.clear)
                )
                .overlay(
                    Capsule()
                        .strokeBorder(isSelected ? tint.opacity(0.35) : Color.clear, lineWidth: 0.5)
                )
        }
        .buttonStyle(.plain)
    }
}

struct TranscriptionModeInfoSheet: View {
    let onClose: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack(alignment: .top, spacing: 12) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.system(size: 28, weight: .semibold))
                    .foregroundStyle(.orange)

                VStack(alignment: .leading, spacing: 5) {
                    Text("Choose The Active Engine")
                        .font(.system(size: 20, weight: .bold, design: .rounded))
                    Text("The Local/Public switch decides which transcription engine runs.")
                        .font(.system(size: 12))
                        .foregroundStyle(.secondary)
                }
            }

            VStack(alignment: .leading, spacing: 10) {
                explanationRow(title: "Local", body: "Uses the loaded model on this Mac. Audio stays local.")
                explanationRow(title: "Public", body: "Uses the selected API model. Audio is sent to the provider.")
                explanationRow(title: "Remembered models", body: "Each side remembers its selected model, but only the switch-selected side is active.")
            }

            HStack {
                Spacer()
                Button("I Understand") {
                    onClose()
                }
                .buttonStyle(.plain)
                .modifier(SettingsInfoActionPillStyle())
            }
        }
        .padding(24)
        .frame(width: 460)
    }

    private func explanationRow(title: String, body: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title)
                .font(.system(size: 12, weight: .semibold))
            Text(body)
                .font(.system(size: 11))
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }
}

private struct SettingsInfoActionPillStyle: ViewModifier {
    func body(content: Content) -> some View {
        content
            .font(.system(size: 11, weight: .semibold))
            .foregroundStyle(.white)
            .padding(.horizontal, 14)
            .frame(height: 30)
            .background(Color.accentColor, in: Capsule())
            .overlay(Capsule().strokeBorder(Color.accentColor.opacity(0.28), lineWidth: 0.6))
            .contentShape(Capsule())
    }
}

// MARK: - Custom Window Button (Traffic Light)

struct WindowButton: View {
    let color: Color
    let systemImage: String
    @Binding var isHovered: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            ZStack {
                Circle()
                    .fill(color)
                    .frame(width: 12, height: 12)

                if isHovered {
                    Image(systemName: systemImage)
                        .font(.system(size: 7, weight: .bold))
                        .foregroundStyle(.black.opacity(0.5))
                }
            }
        }
        .buttonStyle(.plain)
        .onHover { hovering in
            isHovered = hovering
        }
    }
}

// MARK: - Preview

#Preview {
    SettingsView()
        .environmentObject(AppState())
        .environmentObject(BackendManager())
        .environmentObject(UpdateChecker())
}
