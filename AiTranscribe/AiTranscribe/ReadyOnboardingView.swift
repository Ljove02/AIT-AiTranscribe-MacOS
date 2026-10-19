/*
 ReadyOnboardingView.swift
 =========================

 Final onboarding screen - confirms setup and launches the app.
 */

import SwiftUI

struct ReadyOnboardingView: View {
    let onFinish: () -> Void

    @EnvironmentObject var appState: AppState
    @AppStorage("toggleRecordingShortcut") private var toggleShortcut = "⌃P"
    @State private var isLaunching = false
    @State private var checkmarkPulse = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Spacer(minLength: 24)

            HStack(alignment: .center, spacing: 30) {
                readyMark

                VStack(alignment: .leading, spacing: 12) {
                    Text("Here we go.")
                        .font(.system(size: 54, weight: .bold, design: .rounded))
                        .tracking(-1)
                        .foregroundStyle(.primary)

                    Text("AIT is ready to capture, transcribe, and turn audio into useful output.")
                        .font(.system(size: 17, weight: .medium))
                        .lineSpacing(3)
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: 560, alignment: .leading)
                }
            }

            Spacer(minLength: 46)

            VStack(alignment: .leading, spacing: 18) {
                Text("Configured")
                    .font(.system(size: 11, weight: .bold, design: .rounded))
                    .tracking(1.4)
                    .textCase(.uppercase)
                    .foregroundStyle(.secondary)

                HStack(alignment: .top, spacing: 26) {
                    ReadyMetric(
                        icon: "mic.fill",
                        value: "Mic",
                        label: "Permission ready",
                        tint: .green
                    )

                    ReadyMetric(
                        icon: "cube.fill",
                        value: "\(downloadedModelsCount)",
                        label: "Local model\(downloadedModelsCount == 1 ? "" : "s")",
                        tint: downloadedModelsCount > 0 ? .green : .secondary
                    )

                    ReadyMetric(
                        icon: "globe",
                        value: "\(configuredAPIProviderCount)",
                        label: "API provider\(configuredAPIProviderCount == 1 ? "" : "s")",
                        tint: configuredAPIProviderCount > 0 ? .blue : .secondary
                    )

                    ReadyMetric(
                        icon: "keyboard",
                        value: toggleShortcut,
                        label: "Record shortcut",
                        tint: .blue
                    )
                }
            }

            Spacer()

            HStack(alignment: .firstTextBaseline) {
                Text(launchHint)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(.secondary)

                Spacer()

                Button(action: launch) {
                    Label("Open AIT", systemImage: "arrow.right.circle.fill")
                        .font(.system(size: 15, weight: .semibold, design: .rounded))
                        .frame(minWidth: 190)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .disabled(isLaunching)
            }
        }
        .padding(34)
    }

    private var readyMark: some View {
        ZStack {
            Circle()
                .fill(.green.opacity(0.12))
                .frame(width: 118, height: 118)

            Circle()
                .strokeBorder(.green.opacity(0.34), lineWidth: 1)
                .frame(width: 118, height: 118)

            Circle()
                .strokeBorder(.green.opacity(0.18), lineWidth: 1)
                .frame(width: 152, height: 152)

            Image(systemName: "checkmark")
                .font(.system(size: 42, weight: .bold))
                .foregroundStyle(.green)
                .scaleEffect(checkmarkPulse ? 1.22 : 1)
                .animation(.spring(duration: 0.18, bounce: 0.28), value: checkmarkPulse)
        }
    }

    private func launch() {
        guard !isLaunching else { return }
        isLaunching = true

        withAnimation(.spring(duration: 0.18, bounce: 0.28)) {
            checkmarkPulse = true
        }

        DispatchQueue.main.asyncAfter(deadline: .now() + 0.16) {
            withAnimation(.spring(duration: 0.18, bounce: 0.2)) {
                checkmarkPulse = false
            }
        }

        DispatchQueue.main.asyncAfter(deadline: .now() + 0.36) {
            onFinish()
        }
    }

    private var downloadedModelsCount: Int {
        appState.availableModels.filter(\.downloaded).count
    }

    private var configuredAPIProviderCount: Int {
        Set(
            APITranscriptionModelStore.load()
                .filter { APIKeychainStore.hasAPIKey(providerId: $0.providerId) }
                .map(\.providerId)
        ).count
    }

    private var launchHint: String {
        if downloadedModelsCount > 0 && configuredAPIProviderCount > 0 {
            return "Use Local for privacy or Public for speed from the menu bar."
        }
        if downloadedModelsCount > 0 {
            return "Your local setup is ready; public providers can be added later."
        }
        if configuredAPIProviderCount > 0 {
            return "Your public transcription setup is ready; local models can be added later."
        }
        return "You can add models and providers later from Settings."
    }
}

private struct ReadyMetric: View {
    let icon: String
    let value: String
    let label: String
    let tint: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Image(systemName: icon)
                .font(.system(size: 15, weight: .bold))
                .foregroundStyle(tint)
                .frame(width: 26, height: 26)
                .background(tint.opacity(0.14), in: Circle())

            Text(value)
                .font(.system(size: 25, weight: .bold, design: .rounded))
                .foregroundStyle(.primary)

            Text(label)
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(.secondary)
        }
        .frame(width: 142, alignment: .leading)
    }
}

#Preview {
    ReadyOnboardingView(onFinish: {})
        .environmentObject(AppState())
        .frame(width: 900, height: 720)
}
