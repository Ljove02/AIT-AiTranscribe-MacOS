/*
 MenuBarView.swift
 =================

 Custom popover panel for the menu bar icon.
 Uses MenuBarExtra with .window style for a fully custom SwiftUI layout.
 Inspired by Osados-style compact utility panels.
 */

import SwiftUI
import CoreAudio

struct MenuBarView: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var backendManager: BackendManager
    @EnvironmentObject var sessionManager: SessionManager

    @Environment(\.openSettings) private var openSettings
    @Environment(\.openWindow) private var openWindow

    @State private var isHoveredSettings = false
    @State private var isHoveredQuit = false

    var body: some View {
        VStack(spacing: 0) {
            // ── Header: Identity + Status ──
            headerSection

            // ── Status Strip ──
            statusStrip
                .padding(.top, 8)

            // ── Main Controls ──
            controlsSection
                .padding(.top, 10)

            // ── Microphone ──
            microphoneRow
                .padding(.top, 8)

            // ── Footer ──
            footerBar
                .padding(.top, 10)
        }
        .padding(14)
        .frame(width: 280)
    }

    // MARK: - Header

    private var headerSection: some View {
        HStack(spacing: 10) {
            // App icon
            Image("MenuBarIcon")
                .resizable()
                .scaledToFit()
                .frame(width: 28, height: 28)

            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text("AiTranscribe")
                        .font(.system(size: 14, weight: .semibold, design: .rounded))

                    Text("v\(Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "0.1.5")")
                        .font(.system(size: 9, weight: .medium, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 5)
                        .padding(.vertical, 1)
                        .background(.white.opacity(0.06), in: Capsule())
                }

                HStack(spacing: 4) {
                    Circle()
                        .fill(statusColor)
                        .frame(width: 6, height: 6)

                    Text(statusLabel)
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(.secondary)
                }
            }

            Spacer()
        }
    }

    // MARK: - Status Strip

    private var statusStrip: some View {
        HStack(spacing: 8) {
            StatusChip(
                icon: "server.rack",
                label: appState.isServerConnected ? "Backend" : "Offline",
                isActive: appState.isServerConnected
            )

            Spacer()
        }
    }

    // MARK: - Controls Section

    private var isTranscribing: Bool {
        appState.statusMessage.hasPrefix("Transcribing")
    }

    private var controlsSection: some View {
        VStack(spacing: 6) {
            if appState.isRecording {
                recordingActiveControls
            } else if isTranscribing {
                transcribingIndicator
            } else if sessionManager.isSessionRecording {
                sessionActiveControls
            } else {
                idleControls
            }
        }
    }

    private var recordingActiveControls: some View {
        VStack(spacing: 6) {
            // Recording indicator bar
            HStack(spacing: 6) {
                Circle()
                    .fill(.red)
                    .frame(width: 6, height: 6)
                    .shadow(color: .red.opacity(0.5), radius: 4)

                Text("Recording…")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.primary)

                Spacer()

                if appState.recordingDuration > 0 {
                    Text(formatDuration(appState.recordingDuration))
                        .font(.system(size: 10, weight: .medium, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(.red.opacity(0.08), in: .rect(cornerRadius: 8, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .strokeBorder(.red.opacity(0.15), lineWidth: 0.5)
            )

            HStack(spacing: 6) {
                MenuBarButton(
                    label: "Stop",
                    icon: "stop.fill",
                    style: .prominent,
                    tint: .red
                ) {
                    Task { await appState.stopRecording() }
                }

                MenuBarButton(
                    label: "Cancel",
                    icon: "xmark",
                    style: .regular
                ) {
                    Task { await appState.cancelRecording() }
                }
            }
        }
    }

    private var sessionActiveControls: some View {
        VStack(spacing: 6) {
            // Session indicator bar
            HStack(spacing: 6) {
                Circle()
                    .fill(.orange)
                    .frame(width: 6, height: 6)
                    .shadow(color: .orange.opacity(0.5), radius: 4)

                Text("Session Recording")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.primary)

                Spacer()
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(.orange.opacity(0.08), in: .rect(cornerRadius: 8, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .strokeBorder(.orange.opacity(0.15), lineWidth: 0.5)
            )

            MenuBarButton(
                label: "Stop Session",
                icon: "stop.circle.fill",
                style: .prominent,
                tint: .orange
            ) {
                Task { await sessionManager.stopSessionRecording() }
            }
        }
    }

    private var transcribingIndicator: some View {
        HStack(spacing: 6) {
            ProgressView()
                .controlSize(.small)

            Text(appState.statusMessage)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(.primary)

            Spacer()
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(.blue.opacity(0.08), in: .rect(cornerRadius: 8, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .strokeBorder(.blue.opacity(0.15), lineWidth: 0.5)
        )
    }

    private var idleControls: some View {
        VStack(spacing: 6) {
            MenuBarButton(
                label: "Start Recording",
                icon: "mic.fill",
                style: .prominent
            ) {
                Task { await appState.startRecording() }
            }
            .disabled(!appState.isServerConnected)

            MenuBarButton(
                label: "Record Session",
                icon: "waveform.badge.plus",
                style: .regular
            ) {
                Task {
                    await sessionManager.startSessionRecording(
                        micDeviceId: appState.selectedDeviceId
                    )
                }
            }
            .disabled(!appState.isServerConnected)
        }
    }

    // MARK: - Microphone Row

    private var microphoneRow: some View {
        HStack(spacing: 6) {
            Image(systemName: "mic")
                .font(.system(size: 9, weight: .medium))
                .foregroundStyle(.secondary)

            Picker(selection: $appState.selectedDeviceId) {
                ForEach(appState.audioDevices) { device in
                    Text(device.displayName)
                        .tag(device.id as AudioDeviceID?)
                }
            } label: {
                EmptyView()
            }
            .pickerStyle(.menu)
            .labelsHidden()
            .controlSize(.small)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(.white.opacity(0.04), in: .rect(cornerRadius: 8, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .strokeBorder(.white.opacity(0.06), lineWidth: 0.5)
        )
    }

    // MARK: - Footer Bar

    private var footerBar: some View {
        HStack(spacing: 0) {
            // Settings
            Button {
                openAndFocusSettings()
            } label: {
                Image(systemName: "gear")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(isHoveredSettings ? .primary : .secondary)
                    .frame(width: 28, height: 28)
                    .background(
                        isHoveredSettings
                            ? AnyShapeStyle(.white.opacity(0.08))
                            : AnyShapeStyle(.clear)
                        , in: .rect(cornerRadius: 6, style: .continuous)
                    )
            }
            .buttonStyle(.plain)
            .onHover { h in
                withAnimation(.easeOut(duration: 0.12)) { isHoveredSettings = h }
            }
            .help("Settings")

            #if DEBUG
            Button {
                showIntroductionScreen()
            } label: {
                Image(systemName: "play.circle")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(.secondary)
                    .frame(width: 28, height: 28)
            }
            .buttonStyle(.plain)
            .help("Show Introduction")
            #endif

            Spacer()

            // Quit
            Button {
                NSApplication.shared.terminate(nil)
            } label: {
                Image(systemName: "power")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(isHoveredQuit ? .red : .secondary)
                    .frame(width: 28, height: 28)
                    .background(
                        isHoveredQuit
                            ? AnyShapeStyle(.red.opacity(0.1))
                            : AnyShapeStyle(.clear)
                        , in: .rect(cornerRadius: 6, style: .continuous)
                    )
            }
            .buttonStyle(.plain)
            .onHover { h in
                withAnimation(.easeOut(duration: 0.12)) { isHoveredQuit = h }
            }
            .help("Quit AiTranscribe")
        }
    }

    // MARK: - Helpers

    private var statusColor: Color {
        if !appState.isServerConnected {
            return .red
        } else if appState.isRecording {
            return .orange
        } else if isTranscribing {
            return .blue
        } else {
            return .green
        }
    }

    private var statusLabel: String {
        if !appState.isServerConnected {
            return "Server offline"
        } else if appState.isRecording {
            return "Recording"
        } else if isTranscribing {
            return "Transcribing…"
        } else {
            return "Ready"
        }
    }

    private func formatDuration(_ seconds: Double) -> String {
        let mins = Int(seconds) / 60
        let secs = Int(seconds) % 60
        return String(format: "%d:%02d", mins, secs)
    }

    private func openAndFocusSettings() {
        openWindow(id: "settings")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            NSApp.activate(ignoringOtherApps: true)
            for window in NSApp.windows {
                if window.title.contains("Settings") || window.identifier?.rawValue.contains("settings") == true {
                    window.makeKeyAndOrderFront(nil)
                    break
                }
            }
        }
    }

    private func showIntroductionScreen() {
        openWindow(id: "onboarding")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
            NSApp.activate(ignoringOtherApps: true)
            if let window = NSApp.windows.first(where: { $0.identifier?.rawValue == "onboarding" }) {
                window.makeKeyAndOrderFront(nil)
            }
        }
    }

}

// MARK: - Status Chip

private struct StatusChip: View {
    let icon: String
    let label: String
    let isActive: Bool

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: icon)
                .font(.system(size: 8, weight: .medium))
                .foregroundStyle(isActive ? .green : .secondary)

            Text(label)
                .font(.system(size: 9, weight: .medium))
                .foregroundStyle(isActive ? .primary : .secondary)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(
            isActive
                ? AnyShapeStyle(.green.opacity(0.08))
                : AnyShapeStyle(.white.opacity(0.04))
            , in: Capsule()
        )
        .overlay(
            Capsule()
                .strokeBorder(
                    isActive ? .green.opacity(0.15) : .white.opacity(0.06),
                    lineWidth: 0.5
                )
        )
    }
}

// MARK: - MenuBarButton

private struct MenuBarButton: View {
    let label: String
    let icon: String
    var style: Style = .regular
    var tint: Color = .accentColor
    var action: () -> Void

    enum Style {
        case prominent, regular
    }

    @State private var isHovered = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 5) {
                Image(systemName: icon)
                    .font(.system(size: 9, weight: .medium))

                Text(label)
                    .font(.system(size: 11, weight: .medium))
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 7)
            .foregroundStyle(foregroundColor)
            .background(backgroundColor, in: .rect(cornerRadius: 8, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .strokeBorder(borderColor, lineWidth: 0.5)
            )
        }
        .buttonStyle(.plain)
        .onHover { h in
            withAnimation(.easeOut(duration: 0.12)) { isHovered = h }
        }
    }

    private var foregroundColor: Color {
        switch style {
        case .prominent:
            return .white
        case .regular:
            return isHovered ? .primary : .secondary
        }
    }

    private var backgroundColor: some ShapeStyle {
        switch style {
        case .prominent:
            return AnyShapeStyle(tint.opacity(isHovered ? 0.85 : 0.7))
        case .regular:
            return AnyShapeStyle(.white.opacity(isHovered ? 0.08 : 0.04))
        }
    }

    private var borderColor: Color {
        switch style {
        case .prominent:
            return tint.opacity(isHovered ? 0.5 : 0.3)
        case .regular:
            return .white.opacity(isHovered ? 0.12 : 0.06)
        }
    }
}

// MARK: - Preview

#Preview {
    MenuBarView()
        .environmentObject(AppState())
        .environmentObject(BackendManager())
        .environmentObject(SessionManager())
}
