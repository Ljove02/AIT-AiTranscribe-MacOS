/*
 SummarySetupView.swift
 ======================

 UI for installing the Summary runtime (MLX + Gemma).

 Presented as a sheet when the user clicks the Summarization pill.
 Shows runtime status, install/remove actions, and progress.
 Matches the app's glass-material aesthetic.
 */

import SwiftUI

struct SummarySetupView: View {
    @EnvironmentObject var appState: AppState
    @ObservedObject var setupManager: SummarySetupManager
    @Environment(\.dismiss) var dismiss

    /// Python info fetched off the main thread
    @State private var pythonInfo: (path: String, version: String)?
    @State private var pythonChecked = false
    @State private var showRemoveConfirmation = false

    var body: some View {
        VStack(spacing: 0) {
            // Header
            headerSection

            Divider()

            // Content
            ScrollView {
                VStack(spacing: 20) {
                    if setupManager.isInstalling || setupManager.currentStep == .complete {
                        installationProgressView
                    } else if let error = setupManager.error {
                        errorView(error)
                    } else if appState.summaryRuntimeInstalled {
                        installedView
                    } else {
                        preInstallationView
                    }
                }
                .padding(24)
            }

            Divider()

            // Footer
            footerSection
        }
        .frame(width: 500, height: 450)
        .task {
            let info = await Task.detached { [setupManager] in
                setupManager.findPython()
            }.value
            pythonInfo = info
            pythonChecked = true
        }
    }

    // MARK: - Header

    private var headerSection: some View {
        HStack(spacing: 16) {
            Image(systemName: "sparkles")
                .font(.system(size: 36))
                .foregroundStyle(.orange.gradient)

            VStack(alignment: .leading, spacing: 4) {
                Text("Summary Runtime")
                    .font(.title2)
                    .fontWeight(.semibold)

                Text("MLX-powered local summarization")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }

            Spacer()
        }
        .padding(20)
    }

    // MARK: - Installed View

    private var installedView: some View {
        VStack(alignment: .leading, spacing: 20) {
            // Status
            VStack(alignment: .leading, spacing: 12) {
                Label("Status", systemImage: "checkmark.shield")
                    .font(.headline)

                HStack(spacing: 8) {
                    Circle()
                        .fill(appState.summaryRuntimeReady ? .green : .orange)
                        .frame(width: 8, height: 8)
                    Text(appState.summaryRuntimeReady ? "Runtime is ready" : "Runtime installed but not active")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                .padding(.leading, 8)
            }

            Divider()

            // Info
            VStack(alignment: .leading, spacing: 12) {
                Label("About", systemImage: "info.circle")
                    .font(.headline)

                VStack(alignment: .leading, spacing: 8) {
                    bulletPoint("Runs Gemma models locally via MLX")
                    bulletPoint("Generates session summaries on-device")
                    bulletPoint("No data leaves your Mac")
                }
                .padding(.leading, 8)
            }

            Divider()

            // Location
            VStack(alignment: .leading, spacing: 8) {
                Label("Installation location", systemImage: "folder")
                    .font(.headline)

                Text(setupManager.summaryVenvPath.path)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .textSelection(.enabled)
            }

            Divider()

            // Remove option
            VStack(alignment: .leading, spacing: 8) {
                Label("Remove Runtime", systemImage: "trash")
                    .font(.headline)
                    .foregroundStyle(.red)

                Text("This will delete the virtual environment and all installed packages. Downloaded models are stored separately.")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Button("Remove Runtime") {
                    showRemoveConfirmation = true
                }
                .foregroundStyle(.red)
                .font(.system(size: 12, weight: .medium))
                .padding(.leading, 8)
                .confirmationDialog("Remove Summary Runtime?", isPresented: $showRemoveConfirmation) {
                    Button("Remove", role: .destructive) {
                        setupManager.removeRuntime()
                        Task { await appState.fetchSummaryRuntimeStatus() }
                    }
                    Button("Cancel", role: .cancel) {}
                } message: {
                    Text("This will remove the summary runtime environment. You can reinstall it later.")
                }
            }
        }
    }

    // MARK: - Pre-Installation View

    private var preInstallationView: some View {
        VStack(alignment: .leading, spacing: 20) {
            // What will be installed
            VStack(alignment: .leading, spacing: 12) {
                Label("What will be installed", systemImage: "shippingbox")
                    .font(.headline)

                VStack(alignment: .leading, spacing: 8) {
                    bulletPoint("MLX - Apple Silicon ML framework")
                    bulletPoint("MLX-LM - Language model inference")
                    bulletPoint("Dependencies - Tokenizers & utilities")
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

                pythonStatusView
            }

            Divider()

            // Installation location
            VStack(alignment: .leading, spacing: 8) {
                Label("Installation location", systemImage: "folder")
                    .font(.headline)

                Text(setupManager.summaryVenvPath.path)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .textSelection(.enabled)
            }
        }
    }

    private func bulletPoint(_ text: String) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Text("\u{2022}")
                .foregroundColor(.secondary)
            Text(text)
                .foregroundColor(.secondary)
        }
        .font(.subheadline)
    }

    private var pythonStatusView: some View {
        HStack(spacing: 8) {
            if !pythonChecked {
                ProgressView()
                    .scaleEffect(0.7)
                Text("Checking Python...")
                    .foregroundStyle(.secondary)
            } else if let python = pythonInfo {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundStyle(.green)
                Text("Python \(python.version) found")
                    .foregroundStyle(.secondary)
            } else {
                Image(systemName: "xmark.circle.fill")
                    .foregroundStyle(.red)
                VStack(alignment: .leading, spacing: 4) {
                    Text("Python 3.10+ not found")
                        .foregroundStyle(.red)
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
                ForEach(SummarySetupStep.allCases.filter { $0 != .idle && $0 != .error }, id: \.self) { step in
                    stepRow(step)
                }
            }

            // Progress bar
            if setupManager.isInstalling {
                VStack(spacing: 8) {
                    ProgressView(value: setupManager.progress)
                        .progressViewStyle(.linear)
                        .tint(.orange)

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

                    Text("Summary runtime installed!")
                        .font(.headline)

                    if let mlxVersion = setupManager.installedMLXVersion {
                        Text("MLX: \(mlxVersion)")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }

                    Text("You can now download and use summarization models.")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                }
                .padding(.top, 8)
            }
        }
    }

    private func stepRow(_ step: SummarySetupStep) -> some View {
        HStack(spacing: 12) {
            stepIcon(for: step)
                .frame(width: 24)

            Text(step.rawValue)
                .foregroundColor(stepTextColor(for: step))

            Spacer()

            if setupManager.currentStep == step && setupManager.isInstalling {
                ProgressView()
                    .scaleEffect(0.7)
            }
        }
    }

    @ViewBuilder
    private func stepIcon(for step: SummarySetupStep) -> some View {
        let stepOrder = SummarySetupStep.allCases.filter { $0 != .idle && $0 != .error }
        let currentIndex = stepOrder.firstIndex(of: setupManager.currentStep) ?? 0
        let stepIndex = stepOrder.firstIndex(of: step) ?? 0

        if setupManager.currentStep == .complete || stepIndex < currentIndex {
            Image(systemName: "checkmark.circle.fill")
                .foregroundColor(.green)
        } else if stepIndex == currentIndex && setupManager.isInstalling {
            Image(systemName: "circle.fill")
                .foregroundColor(.orange)
        } else {
            Image(systemName: "circle")
                .foregroundColor(.secondary)
        }
    }

    private func stepTextColor(for step: SummarySetupStep) -> Color {
        let stepOrder = SummarySetupStep.allCases.filter { $0 != .idle && $0 != .error }
        let currentIndex = stepOrder.firstIndex(of: setupManager.currentStep) ?? 0
        let stepIndex = stepOrder.firstIndex(of: step) ?? 0

        if setupManager.currentStep == .complete || stepIndex <= currentIndex {
            return .primary
        }
        return .secondary
    }

    // MARK: - Error View

    private func errorView(_ error: SummarySetupError) -> some View {
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
            .tint(.orange)
        }
        .padding()
    }

    // MARK: - Footer

    private var footerSection: some View {
        HStack {
            Button("Cancel") {
                if setupManager.isInstalling {
                    setupManager.cancelInstall()
                }
                dismiss()
            }
            .keyboardShortcut(.cancelAction)

            Spacer()

            if setupManager.currentStep == .complete {
                Button("Done") {
                    Task { await appState.fetchSummaryRuntimeStatus() }
                    dismiss()
                }
                .buttonStyle(.borderedProminent)
                .tint(.orange)
                .keyboardShortcut(.defaultAction)
            } else if appState.summaryRuntimeInstalled {
                Button("Done") {
                    dismiss()
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
            } else if !setupManager.isInstalling && setupManager.error == nil {
                Button("Install Runtime") {
                    startInstallation()
                }
                .buttonStyle(.borderedProminent)
                .tint(.orange)
                .keyboardShortcut(.defaultAction)
                .disabled(pythonInfo == nil && pythonChecked)
            }
        }
        .padding(20)
    }

    // MARK: - Actions

    private func startInstallation() {
        Task {
            do {
                try await setupManager.installRuntime()
                await appState.fetchSummaryRuntimeStatus()
                await appState.fetchSummaryModels()
            } catch {
                if let summaryError = error as? SummarySetupError {
                    setupManager.error = summaryError
                } else {
                    setupManager.error = .unknown(error)
                }
            }
        }
    }
}

// MARK: - Preview

#Preview {
    SummarySetupView(setupManager: SummarySetupManager())
        .environmentObject(AppState())
}
