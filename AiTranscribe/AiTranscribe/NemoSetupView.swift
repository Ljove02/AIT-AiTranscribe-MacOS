/*
 NemoSetupView.swift
 ===================

 UI for installing NeMo support.

 This is presented as a sheet when the user wants to enable NeMo models
 (Parakeet, Nemotron). It shows:
 - Information about NeMo (~3GB download)
 - Python requirement notice
 - Progress steps with checkmarks
 - Cancel button
 - Error display with retry
 */

import SwiftUI

struct NemoSetupView: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var backendManager: BackendManager
    @ObservedObject var setupManager: NemoSetupManager
    @Environment(\.dismiss) var dismiss

    /// Whether to automatically start installation on appear
    var autoStart: Bool = false

    /// Callback when installation completes
    var onComplete: (() -> Void)?

    var body: some View {
        VStack(spacing: 0) {
            // Header
            headerSection

            Divider()

            // Content
            ScrollView {
                VStack(spacing: 20) {
                    if setupManager.isInstalling || setupManager.currentStep == .complete {
                        // Installation progress
                        installationProgressView
                    } else if let error = setupManager.error {
                        // Error state
                        errorView(error)
                    } else {
                        // Pre-installation info
                        preInstallationView
                    }
                }
                .padding(24)
            }

            Divider()

            // Footer with action buttons
            footerSection
        }
        .frame(width: 500, height: 450)
        .onAppear {
            if autoStart && !setupManager.isInstalling {
                startInstallation()
            }
        }
    }

    // MARK: - Header

    private var headerSection: some View {
        HStack(spacing: 16) {
            // Icon
            Image(systemName: "cpu")
                .font(.system(size: 36))
                .foregroundStyle(.blue.gradient)

            VStack(alignment: .leading, spacing: 4) {
                Text("Install NeMo Support")
                    .font(.title2)
                    .fontWeight(.semibold)

                Text("Enable NVIDIA Parakeet & Nemotron models")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }

            Spacer()
        }
        .padding(20)
    }

    // MARK: - Pre-Installation View

    private var preInstallationView: some View {
        VStack(alignment: .leading, spacing: 20) {
            // What will be installed
            VStack(alignment: .leading, spacing: 12) {
                Label("What will be installed", systemImage: "shippingbox")
                    .font(.headline)

                VStack(alignment: .leading, spacing: 8) {
                    bulletPoint("PyTorch - Deep learning framework")
                    bulletPoint("NeMo Toolkit - NVIDIA's ASR framework")
                    bulletPoint("Dependencies - Audio processing libraries")
                }
                .padding(.leading, 8)

                HStack(spacing: 4) {
                    Image(systemName: "arrow.down.circle")
                    Text("Total download: ~3 GB")
                }
                .font(.caption)
                .foregroundColor(.secondary)
                .padding(.leading, 8)
            }

            Divider()

            // Requirements
            VStack(alignment: .leading, spacing: 12) {
                Label("Requirements", systemImage: "checklist")
                    .font(.headline)

                // Python status
                pythonStatusView
            }

            Divider()

            // Where it will be installed
            VStack(alignment: .leading, spacing: 8) {
                Label("Installation location", systemImage: "folder")
                    .font(.headline)

                Text(setupManager.nemoVenvPath.path)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .textSelection(.enabled)
            }
        }
    }

    private func bulletPoint(_ text: String) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Text("•")
                .foregroundColor(.secondary)
            Text(text)
                .foregroundColor(.secondary)
        }
        .font(.subheadline)
    }

    private var pythonStatusView: some View {
        HStack(spacing: 8) {
            if let python = setupManager.findPython() {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundColor(.green)
                Text("Python \(python.version) found")
                    .foregroundColor(.secondary)
            } else {
                Image(systemName: "xmark.circle.fill")
                    .foregroundColor(.red)
                VStack(alignment: .leading, spacing: 4) {
                    Text("Python 3.9+ not found")
                        .foregroundColor(.red)
                    Link("Download from python.org", destination: URL(string: "https://www.python.org/downloads/")!)
                        .font(.caption)
                }
            }
        }
        .font(.subheadline)
        .padding(.leading, 8)
    }

    // MARK: - Installation Progress View

    private var installationProgressView: some View {
        VStack(spacing: 24) {
            // Step indicators
            VStack(alignment: .leading, spacing: 16) {
                ForEach(NemoSetupStep.allCases.filter { $0 != .idle && $0 != .error }, id: \.self) { step in
                    stepRow(step)
                }
            }

            // Progress bar
            if setupManager.isInstalling {
                VStack(spacing: 8) {
                    ProgressView(value: setupManager.progress)
                        .progressViewStyle(.linear)

                    Text(setupManager.statusMessage)
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .lineLimit(2)
                        .multilineTextAlignment(.center)

                    if let package = setupManager.currentPackage {
                        Text("Installing: \(package)")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                }
            }

            // Completion message
            if setupManager.currentStep == .complete {
                VStack(spacing: 12) {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.system(size: 48))
                        .foregroundColor(.green)

                    Text("NeMo installed successfully!")
                        .font(.headline)

                    if let version = setupManager.installedNemoVersion {
                        Text("Version: \(version)")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }

                    Text("NVIDIA Parakeet and Nemotron models are now available.")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                }
                .padding(.top, 8)
            }
        }
    }

    private func stepRow(_ step: NemoSetupStep) -> some View {
        HStack(spacing: 12) {
            // Status icon
            stepIcon(for: step)
                .frame(width: 24)

            // Step name
            Text(step.displayName)
                .foregroundColor(stepTextColor(for: step))

            Spacer()

            // Spinner for current step
            if setupManager.currentStep == step && setupManager.isInstalling {
                ProgressView()
                    .scaleEffect(0.7)
            }
        }
    }

    @ViewBuilder
    private func stepIcon(for step: NemoSetupStep) -> some View {
        let stepOrder = NemoSetupStep.allCases.filter { $0 != .idle && $0 != .error }
        let currentIndex = stepOrder.firstIndex(of: setupManager.currentStep) ?? 0
        let stepIndex = stepOrder.firstIndex(of: step) ?? 0

        if setupManager.currentStep == .complete || stepIndex < currentIndex {
            // Completed
            Image(systemName: "checkmark.circle.fill")
                .foregroundColor(.green)
        } else if stepIndex == currentIndex && setupManager.isInstalling {
            // Current
            Image(systemName: "circle.fill")
                .foregroundColor(.blue)
        } else {
            // Pending
            Image(systemName: "circle")
                .foregroundColor(.secondary)
        }
    }

    private func stepTextColor(for step: NemoSetupStep) -> Color {
        let stepOrder = NemoSetupStep.allCases.filter { $0 != .idle && $0 != .error }
        let currentIndex = stepOrder.firstIndex(of: setupManager.currentStep) ?? 0
        let stepIndex = stepOrder.firstIndex(of: step) ?? 0

        if setupManager.currentStep == .complete || stepIndex <= currentIndex {
            return .primary
        }
        return .secondary
    }

    // MARK: - Error View

    private func errorView(_ error: NemoSetupError) -> some View {
        VStack(spacing: 16) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 48))
                .foregroundColor(.red)

            Text("Installation Failed")
                .font(.headline)

            Text(error.localizedDescription)
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal)

            Button("Try Again") {
                setupManager.error = nil
                startInstallation()
            }
            .buttonStyle(.borderedProminent)
        }
        .padding()
    }

    // MARK: - Footer

    private var footerSection: some View {
        HStack {
            // Cancel button
            Button("Cancel") {
                if setupManager.isInstalling {
                    setupManager.cancel()
                }
                dismiss()
            }
            .keyboardShortcut(.cancelAction)

            Spacer()

            // Action button
            if setupManager.currentStep == .complete {
                Button("Done") {
                    onComplete?()
                    dismiss()
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
            } else if !setupManager.isInstalling && setupManager.error == nil {
                Button("Install NeMo") {
                    startInstallation()
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .disabled(setupManager.findPython() == nil)
            }
        }
        .padding(20)
    }

    // MARK: - Actions

    private func startInstallation() {
        Task {
            do {
                try await setupManager.installNemo()

                // Switch backend to NeMo mode
                await backendManager.switchToNemoMode()

                // Refresh app state
                await appState.checkNemoStatus()
                await appState.fetchAvailableModels()

            } catch {
                if let nemoError = error as? NemoSetupError {
                    setupManager.error = nemoError
                } else {
                    setupManager.error = .unknown(error)
                }
            }
        }
    }
}


// MARK: - Preview

#Preview {
    NemoSetupView(setupManager: NemoSetupManager())
        .environmentObject(AppState())
        .environmentObject(BackendManager())
}
