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

// MARK: - Settings View

struct SettingsView: View {
    @State private var selectedSection: SettingsSection = .dashboard
    @State private var isSidebarExpanded: Bool = false
    @State private var animatedTabs: Set<SettingsSection> = []
    @State private var pendingHistoryEntryId: UUID? = nil
    @State private var pendingSessionId: UUID? = nil
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
        case .models:
            ModelsSettingsView(hasAnimated: animatedTabs.contains(.models), onAnimated: {
                animatedTabs.insert(.models)
            })
        case .sessions:
            SessionsSettingsView(
                hasAnimated: animatedTabs.contains(.sessions),
                onAnimated: { animatedTabs.insert(.sessions) },
                onNavigateToModels: {
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
