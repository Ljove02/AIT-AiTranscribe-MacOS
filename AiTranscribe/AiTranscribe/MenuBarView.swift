/*
 MenuBarView.swift
 =================

 Custom popover panel for the menu bar icon.
 Uses MenuBarExtra with .window style for a fully custom SwiftUI layout.
 Inspired by Osados-style compact utility panels.
 */

import AppKit
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
    @State private var isHistoryExpanded = false
    @State private var copiedHistoryEntryId: UUID?
    @State private var copiedMeetingURLKey: String?
    @AppStorage("selectedTranscriptionProviderId") private var selectedTranscriptionProviderId = TranscriptionProviderDescriptor.localBackend.id
    @ObservedObject private var meetingMonitor = MeetingPresenceMonitor.shared

    var body: some View {
        VStack(spacing: 0) {
            // ── Header: Identity + Status ──
            headerSection

            if let prompt = meetingMonitor.currentPrompt, canStartSessionFromMeetingReminder {
                meetingReminderCard(prompt)
                    .padding(.top, 8)
            }

            // ── Status Strip ──
            statusStrip
                .padding(.top, 8)

            // ── Active Model ──
            modelRow
                .padding(.top, 6)

            // ── Main Controls ──
            controlsSection
                .padding(.top, 10)

            // ── Microphone ──
            microphoneRow
                .padding(.top, 8)

            if !visiblePolishPresets.isEmpty {
                polishQuickSection
                    .padding(.top, 8)
            }

            // ── Recent History ──
            historySection
                .padding(.top, 8)

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
                        .lineLimit(1)
                        .truncationMode(.tail)
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

            headerActions
        }
    }

    private var headerActions: some View {
        HStack(spacing: 6) {
            Button {
                openAndFocusSettings()
            } label: {
                Image(systemName: "gear")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(isHoveredSettings ? .primary : .secondary)
                    .frame(width: 28, height: 28)
                    .background(
                        isHoveredSettings
                            ? AnyShapeStyle(.white.opacity(0.08))
                            : AnyShapeStyle(.clear),
                        in: .rect(cornerRadius: 8, style: .continuous)
                    )
            }
            .buttonStyle(.plain)
            .onHover { h in
                withAnimation(.easeOut(duration: 0.12)) { isHoveredSettings = h }
            }
            .help("Settings")

            Button {
                NSApplication.shared.terminate(nil)
            } label: {
                Image(systemName: "power")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(isHoveredQuit ? .white : .red)
                    .frame(width: 28, height: 28)
                    .background(
                        isHoveredQuit
                            ? AnyShapeStyle(.red.opacity(0.82))
                            : AnyShapeStyle(.red.opacity(0.10)),
                        in: .rect(cornerRadius: 8, style: .continuous)
                    )
            }
            .buttonStyle(.plain)
            .onHover { h in
                withAnimation(.easeOut(duration: 0.12)) { isHoveredQuit = h }
            }
            .help("Quit AiTranscribe")
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

            menuModeSwitch
        }
    }

    /// The active transcription model on its own full-width row, so real model
    /// names are readable inside the 280pt panel.
    private var modelRow: some View {
        HStack(spacing: 6) {
            Image(systemName: "cpu")
                .font(.system(size: 9, weight: .semibold))
                .foregroundStyle(.secondary)

            Text("Model")
                .font(.system(size: 9, weight: .bold))
                .foregroundStyle(.tertiary)
                .tracking(0.6)

            Spacer(minLength: 8)

            Text(appState.activeTranscriptionModelName)
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.middle)
        }
        .padding(.horizontal, 9)
        .padding(.vertical, 6)
        .frame(maxWidth: .infinity)
        .background(.white.opacity(0.06), in: RoundedRectangle(cornerRadius: 9, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 9, style: .continuous)
                .strokeBorder(.white.opacity(0.08), lineWidth: 0.5)
        )
        .help(appState.activeTranscriptionModelName)
    }

    private var menuModeSwitch: some View {
        HStack(spacing: 2) {
            providerModeButton(
                title: "Local",
                isSelected: isLocalProviderMode,
                tint: .green
            ) {
                selectedTranscriptionProviderId = TranscriptionProviderDescriptor.localBackend.id
            }

            providerModeButton(
                title: "Public",
                isSelected: !isLocalProviderMode,
                tint: .blue
            ) {
                if selectedTranscriptionProviderId == TranscriptionProviderDescriptor.localBackend.id {
                    selectedTranscriptionProviderId = APITranscriptionModelStore.load().first?.providerId
                        ?? TranscriptionProviderDescriptor.openAI.id
                }
            }
        }
        .padding(2)
        .background(.white.opacity(0.06), in: Capsule())
        .overlay(Capsule().strokeBorder(.white.opacity(0.08), lineWidth: 0.5))
        .opacity(providerModeSwitchDisabled ? 0.48 : 1)
        .disabled(providerModeSwitchDisabled)
        .help(providerModeSwitchDisabled ? "Finish the current recording before switching mode" : "Switch transcription mode")
    }

    private func providerModeButton(
        title: String,
        isSelected: Bool,
        tint: Color,
        action: @escaping () -> Void
    ) -> some View {
        Button {
            withAnimation(.spring(duration: 0.24, bounce: 0.14)) {
                action()
            }
        } label: {
            Text(title)
                .font(.system(size: 9, weight: .semibold))
                .foregroundStyle(isSelected ? .white : .secondary)
                .padding(.horizontal, 7)
                .frame(height: 20)
                .background(isSelected ? tint : .clear, in: Capsule())
        }
        .buttonStyle(.plain)
    }

    private var isLocalProviderMode: Bool {
        selectedTranscriptionProviderId == TranscriptionProviderDescriptor.localBackend.id
    }

    private var providerModeSwitchDisabled: Bool {
        appState.isRecording ||
        appState.isQuickRecordingTransitioning ||
        appState.isQuickTranscriptionInFlight ||
        sessionManager.isSessionRecording
    }

    // MARK: - Controls Section

    private var isTranscribing: Bool {
        appState.isQuickTranscriptionInFlight ||
        appState.statusMessage.hasPrefix("Transcribing") ||
        appState.statusMessage.hasPrefix("Finishing") ||
        appState.statusMessage.hasPrefix("Polishing")
    }

    private var controlsSection: some View {
        VStack(spacing: 6) {
            if isTranscribing {
                transcribingIndicator
            } else if appState.isRecording {
                recordingActiveControls
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

                RecordingDurationLabel(activity: appState.recordingActivity)
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
                .disabled(appState.isQuickRecordingTransitioning)

                MenuBarButton(
                    label: "Cancel",
                    icon: "xmark",
                    style: .regular
                ) {
                    Task { await appState.cancelRecording() }
                }
                .disabled(appState.isQuickRecordingTransitioning)
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

            // Levels live here rather than in Settings: this is the only
            // surface reachable without leaving the call.
            SessionMixerView(recorder: sessionManager.recorder)

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
            .disabled(!appState.isServerConnected || appState.isQuickRecordingTransitioning || appState.isQuickTranscriptionInFlight)

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
            .disabled(!appState.isServerConnected || appState.isQuickRecordingTransitioning || appState.isQuickTranscriptionInFlight)
        }
    }

    private var canStartSessionFromMeetingReminder: Bool {
        appState.isServerConnected &&
            !appState.isRecording &&
            !appState.isQuickRecordingTransitioning &&
            !appState.isQuickTranscriptionInFlight &&
            !sessionManager.isSessionRecording
    }

    private func meetingReminderCard(_ prompt: MeetingReminder) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .center, spacing: 8) {
                Image(systemName: prompt.provider.symbolName)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(promptTint(prompt))
                    .frame(width: 24, height: 24)
                    .background(promptTint(prompt).opacity(0.14), in: .circle)
                    .overlay(
                        Circle()
                            .strokeBorder(promptTint(prompt).opacity(0.24), lineWidth: 0.5)
                    )

                VStack(alignment: .leading, spacing: 2) {
                    Text(prompt.headline)
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(.primary)
                        .lineLimit(1)

                    HStack(spacing: 6) {
                        Text(prompt.detail)
                            .lineLimit(1)
                            .truncationMode(.middle)

                        Text(prompt.contextLabel)
                            .font(.system(size: 8, weight: .semibold))
                            .foregroundStyle(promptTint(prompt))
                            .lineLimit(1)
                            .padding(.horizontal, 5)
                            .frame(height: 15)
                            .background(promptTint(prompt).opacity(0.12), in: Capsule())
                            .overlay(
                                Capsule()
                                    .strokeBorder(promptTint(prompt).opacity(0.16), lineWidth: 0.5)
                            )
                    }
                    .font(.system(size: 9, weight: .medium))
                    .foregroundStyle(.secondary)
                }

                Spacer(minLength: 4)

                Button {
                    meetingMonitor.dismissCurrentPrompt()
                } label: {
                    Image(systemName: "xmark")
                        .font(.system(size: 8, weight: .bold))
                        .foregroundStyle(.red.opacity(0.85))
                        .frame(width: 22, height: 22)
                        .background(.red.opacity(0.10), in: .circle)
                }
                .buttonStyle(.plain)
                .help("Dismiss meeting reminder")
            }

            HStack(spacing: 7) {
                meetingReminderPillButton(
                    title: "Record session",
                    icon: "waveform.badge.plus",
                    tint: promptTint(prompt),
                    isProminent: true
                ) {
                    Task {
                        let started = await sessionManager.startSessionRecording(
                            micDeviceId: appState.selectedDeviceId
                        )
                        if started {
                            meetingMonitor.markStartedFromPrompt()
                        }
                    }
                }

                if let meetingURL = prompt.meetingURL {
                    meetingReminderPillButton(
                        title: copiedMeetingURLKey == prompt.suppressionKey ? "Copied" : "Copy link",
                        icon: copiedMeetingURLKey == prompt.suppressionKey ? "checkmark" : "link",
                        tint: promptTint(prompt),
                        isProminent: false
                    ) {
                        copyMeetingURL(meetingURL, key: prompt.suppressionKey)
                    }
                }

                Spacer(minLength: 0)
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 9)
        .background(.thinMaterial, in: .rect(cornerRadius: 14, style: .continuous))
        .background(promptTint(prompt).opacity(0.045), in: .rect(cornerRadius: 14, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(.white.opacity(0.12), lineWidth: 0.5)
        )
        .task(id: prompt.id) {
            MeetingReminderCoordinator.shared.presentFloatingPromptIfNeeded()
        }
    }

    // MARK: - Microphone Row

    private var microphoneRow: some View {
        Menu {
            if appState.isServerConnected && !appState.audioDevices.isEmpty {
                ForEach(appState.audioDevices) { device in
                    Button(device.displayName) {
                        appState.selectedDeviceId = device.id
                    }
                }
            } else {
                Text(appState.isServerConnected ? "Detecting microphone..." : "Starting backend...")
            }
        } label: {
            HStack(spacing: 7) {
                Image(systemName: "mic")
                    .font(.system(size: 9, weight: .medium))
                    .foregroundStyle(.secondary)

                Text(selectedMicrophoneLabel)
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(appState.isServerConnected && !appState.audioDevices.isEmpty ? .primary : .secondary)
                    .lineLimit(1)
                    .frame(maxWidth: .infinity, alignment: .leading)

                Image(systemName: "chevron.up.chevron.down")
                    .font(.system(size: 8, weight: .semibold))
                    .foregroundStyle(.tertiary)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(!appState.isServerConnected || appState.audioDevices.isEmpty)
        .padding(.horizontal, 10)
        .padding(.vertical, 7)
        .frame(maxWidth: .infinity, minHeight: 32)
        .background(.white.opacity(0.04), in: .rect(cornerRadius: 8, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .strokeBorder(.white.opacity(0.06), lineWidth: 0.5)
        )
    }

    private var selectedMicrophoneLabel: String {
        if !appState.isServerConnected {
            return "Starting backend..."
        }
        if appState.audioDevices.isEmpty {
            return "Detecting microphone..."
        }
        if let selected = appState.audioDevices.first(where: { $0.id == appState.selectedDeviceId }) {
            return selected.displayName
        }
        return appState.audioDevices.first?.displayName ?? "Default microphone"
    }

    // MARK: - Recent History

    private var visiblePolishPresets: [PolishPreset] {
        guard hasUsablePolishModel else {
            return []
        }
        return appState.polishPresets.filter { $0.isEnabled && $0.showInMenu }
    }

    private var hasUsablePolishModel: Bool {
        selectedTranscriptionProviderId == TranscriptionProviderDescriptor.localBackend.id
            ? hasDownloadedLocalLanguageModel
            : hasAnyConfiguredLanguageModelKey
    }

    private var hasDownloadedLocalLanguageModel: Bool {
        appState.availableSummaryModels.contains(where: \.downloaded)
    }

    private var hasAnyConfiguredLanguageModelKey: Bool {
        APILanguageModelStore.load().contains { model in
            if APIKeychainStore.hasAPIKey(providerId: model.providerId) {
                return true
            }
            return model.providerId == TranscriptionProviderDescriptor.openAI.id &&
                model.modelId.contains("/") &&
                APIKeychainStore.hasAPIKey(providerId: TranscriptionProviderDescriptor.openRouter.id)
        }
    }

    private var polishQuickSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 7) {
                Image(systemName: "sparkles")
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(.secondary)

                Text("Polish")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(.primary)

                if let selected = appState.selectedPolishPreset {
                    Text(selected.displayName)
                        .font(.system(size: 9, weight: .medium))
                        .foregroundStyle(.blue)
                        .lineLimit(1)
                        .padding(.horizontal, 5)
                        .padding(.vertical, 1)
                        .background(Color.blue.opacity(0.10), in: Capsule())
                } else {
                    Text("Raw")
                        .font(.system(size: 9, weight: .medium))
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 5)
                        .padding(.vertical, 1)
                        .background(.white.opacity(0.06), in: Capsule())
                }

                Spacer()
            }

            LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 6), count: 4), spacing: 6) {
                ForEach(visiblePolishPresets) { preset in
                    Button {
                        if appState.selectedPolishPresetId == preset.id {
                            appState.setSelectedPolishPreset(nil)
                        } else {
                            appState.setSelectedPolishPreset(preset.id)
                        }
                    } label: {
                        Image(systemName: preset.symbolName)
                            .font(.system(size: 12, weight: .semibold))
                            .foregroundStyle(appState.selectedPolishPresetId == preset.id ? .white : .secondary)
                            .frame(maxWidth: .infinity)
                            .frame(height: 30)
                            .background(
                                appState.selectedPolishPresetId == preset.id
                                    ? AnyShapeStyle(Color.blue)
                                    : AnyShapeStyle(.white.opacity(0.04)),
                                in: RoundedRectangle(cornerRadius: 8, style: .continuous)
                            )
                            .overlay(
                                RoundedRectangle(cornerRadius: 8, style: .continuous)
                                    .strokeBorder(
                                        appState.selectedPolishPresetId == preset.id ? Color.blue.opacity(0.6) : Color.white.opacity(0.06),
                                        lineWidth: 0.5
                                    )
                            )
                    }
                    .buttonStyle(.plain)
                    .help(preset.displayName)
                    .accessibilityLabel("Use \(preset.displayName) polish")
                }
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background(.white.opacity(0.035), in: .rect(cornerRadius: 8, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .strokeBorder(.white.opacity(0.06), lineWidth: 0.5)
        )
    }

    private var recentHistoryEntries: [TranscriptionEntry] {
        Array(appState.transcriptionHistory.prefix(5))
    }

    private var historySection: some View {
        VStack(spacing: 6) {
            Button {
                let animation: Animation = isHistoryExpanded
                    ? .easeOut(duration: 0.10)
                    : .easeOut(duration: 0.34)
                withAnimation(animation) {
                    isHistoryExpanded.toggle()
                }
            } label: {
                HStack(spacing: 7) {
                    Image(systemName: "clock.arrow.circlepath")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(.secondary)

                    Text("History")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(.primary)

                    if !recentHistoryEntries.isEmpty {
                        Text("\(recentHistoryEntries.count)")
                            .font(.system(size: 9, weight: .bold, design: .monospaced))
                            .foregroundStyle(.secondary)
                            .padding(.horizontal, 5)
                            .padding(.vertical, 1)
                            .background(.white.opacity(0.06), in: Capsule())
                    }

                    Spacer()

                    Image(systemName: "chevron.down")
                        .font(.system(size: 9, weight: .semibold))
                        .foregroundStyle(.secondary)
                        .rotationEffect(.degrees(isHistoryExpanded ? 180 : 0))
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 7)
                .background(.white.opacity(0.04), in: .rect(cornerRadius: 8, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .strokeBorder(.white.opacity(0.06), lineWidth: 0.5)
                )
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Recent transcription history")
            .accessibilityHint(isHistoryExpanded ? "Collapse recent history" : "Show recent history")

            if isHistoryExpanded {
                if recentHistoryEntries.isEmpty {
                    emptyHistoryView
                        .transition(historyContentTransition)
                } else {
                    VStack(spacing: 5) {
                        ForEach(recentHistoryEntries) { entry in
                            HistoryRow(
                                entry: entry,
                                isCopied: copiedHistoryEntryId == entry.id
                            ) {
                                copyHistoryEntry(entry)
                            }
                        }
                    }
                    .transition(historyContentTransition)
                }
            }
        }
    }

    private var historyContentTransition: AnyTransition {
        .asymmetric(
            insertion: .opacity
                .combined(with: .scale(scale: 0.985, anchor: .top)),
            removal: .opacity
        )
    }

    private var emptyHistoryView: some View {
        HStack(spacing: 6) {
            Image(systemName: "tray")
                .font(.system(size: 10, weight: .medium))
                .foregroundStyle(.secondary)

            Text("No recent transcriptions yet")
                .font(.system(size: 10, weight: .medium))
                .foregroundStyle(.secondary)

            Spacer()
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background(.white.opacity(0.025), in: .rect(cornerRadius: 8, style: .continuous))
    }

    // MARK: - Helpers

    private var statusColor: Color {
        if !appState.isServerConnected {
            return .red
        } else if isTranscribing {
            return .blue
        } else if appState.isRecording {
            return .orange
        } else {
            return .green
        }
    }

    private var statusLabel: String {
        if !appState.isServerConnected {
            return "Server offline"
        } else if isTranscribing {
            return "Transcribing…"
        } else if appState.isRecording {
            return "Recording"
        } else {
            return "Ready"
        }
    }

    private func openAndFocusSettings() {
        openWindow(id: "settings")
        SettingsWindowFocus.focus()
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

    private func copyHistoryEntry(_ entry: TranscriptionEntry) {
        appState.copyToClipboard(entry.text)
        copiedHistoryEntryId = entry.id

        DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) {
            if copiedHistoryEntryId == entry.id {
                copiedHistoryEntryId = nil
            }
        }
    }

    private func promptTint(_ prompt: MeetingReminder) -> Color {
        prompt.provider == .googleMeet ? .green : .blue
    }

    private func meetingReminderPillButton(
        title: String,
        icon: String,
        tint: Color,
        isProminent: Bool,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            HStack(spacing: 5) {
                Image(systemName: icon)
                    .font(.system(size: 9, weight: .semibold))

                Text(title)
                    .font(.system(size: 10, weight: .semibold))
                    .lineLimit(1)
            }
            .foregroundStyle(isProminent ? .white : .secondary)
            .padding(.horizontal, isProminent ? 10 : 9)
            .frame(height: 27)
            .background(
                isProminent
                    ? AnyShapeStyle(tint.opacity(0.72))
                    : AnyShapeStyle(.white.opacity(0.055)),
                in: Capsule()
            )
            .overlay(
                Capsule()
                    .strokeBorder(
                        isProminent ? tint.opacity(0.32) : .white.opacity(0.08),
                        lineWidth: 0.5
                    )
            )
        }
        .buttonStyle(.plain)
        .help(title)
    }

    private func copyMeetingURL(_ meetingURL: String, key: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(meetingURL, forType: .string)
        copiedMeetingURLKey = key

        DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) {
            if copiedMeetingURLKey == key {
                copiedMeetingURLKey = nil
            }
        }
    }

}

// MARK: - Recording Duration Label

/// The only view observing RecordingActivityModel's 10 Hz updates, so the
/// rest of the menu (and every other AppState observer) stays untouched
/// while recording.
private struct RecordingDurationLabel: View {
    @ObservedObject var activity: RecordingActivityModel

    var body: some View {
        if activity.duration > 0 {
            Text(Self.format(activity.duration))
                .font(.system(size: 10, weight: .medium, design: .monospaced))
                .foregroundStyle(.secondary)
        }
    }

    private static func format(_ seconds: Double) -> String {
        let mins = Int(seconds) / 60
        let secs = Int(seconds) % 60
        return String(format: "%d:%02d", mins, secs)
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

// MARK: - History Row

private struct HistoryRow: View {
    let entry: TranscriptionEntry
    let isCopied: Bool
    let action: () -> Void

    @State private var isHovered = false

    var body: some View {
        Button(action: action) {
            HStack(alignment: .top, spacing: 8) {
                Image(systemName: isCopied ? "checkmark.circle.fill" : "doc.on.doc")
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(isCopied ? .green : .secondary)
                    .frame(width: 14, height: 18)

                VStack(alignment: .leading, spacing: 3) {
                    Text(entry.text)
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(.primary)
                        .lineLimit(2)
                        .multilineTextAlignment(.leading)

                    HStack(spacing: 5) {
                        Text(entry.timestamp, formatter: Self.timestampFormatter)

                        Text("•")

                        Text("\(entry.wordCount) words")
                    }
                    .font(.system(size: 8, weight: .medium))
                    .foregroundStyle(.secondary)
                }

                Spacer(minLength: 0)
            }
            .padding(.horizontal, 9)
            .padding(.vertical, 7)
            .background(
                isHovered
                    ? AnyShapeStyle(.white.opacity(0.07))
                    : AnyShapeStyle(.white.opacity(0.03)),
                in: .rect(cornerRadius: 8, style: .continuous)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .strokeBorder(
                        isCopied ? .green.opacity(0.28) : .white.opacity(isHovered ? 0.1 : 0.05),
                        lineWidth: 0.5
                    )
            )
        }
        .buttonStyle(.plain)
        .onHover { h in
            withAnimation(.easeOut(duration: 0.12)) { isHovered = h }
        }
        .accessibilityLabel("Copy transcription from \(entry.timestamp, formatter: Self.timestampFormatter)")
        .accessibilityHint("Copies this transcription to the clipboard")
    }

    private static let timestampFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateStyle = .none
        formatter.timeStyle = .short
        formatter.doesRelativeDateFormatting = true
        return formatter
    }()
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
