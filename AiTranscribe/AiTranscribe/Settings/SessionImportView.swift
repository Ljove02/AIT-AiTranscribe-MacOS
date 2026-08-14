import SwiftUI
import AppKit
import UniformTypeIdentifiers

// MARK: - Session Import View

/// Full page (swapped in place of the sessions list, like SessionDetailView)
/// for importing any audio or video file as a new session.
struct SessionImportView: View {
    @EnvironmentObject var sessionManager: SessionManager
    let onBack: () -> Void
    let onImported: (UUID) -> Void

    @State private var pickedFile: ImportableAudioFile? = nil
    @State private var sessionName = ""
    @State private var isDropTargeted = false
    @State private var isProbing = false
    @State private var isImporting = false
    @State private var importProgress = 0.0
    @State private var errorMessage: String? = nil
    @State private var appeared = false

    var body: some View {
        VStack(spacing: 0) {
            // Back button header — same pattern as SessionDetailView
            HStack {
                Button {
                    withAnimation(.easeInOut(duration: 0.15)) { onBack() }
                } label: {
                    HStack(spacing: 5) {
                        Image(systemName: "chevron.left")
                            .font(.system(size: 10, weight: .semibold))
                        Text("Sessions")
                            .font(.system(size: 12, weight: .medium))
                    }
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(.quaternary, in: Capsule())
                }
                .buttonStyle(.plain)
                .disabled(isImporting)

                Spacer()
            }
            .padding(.horizontal, 28)
            .padding(.top, 12)
            .padding(.bottom, 8)

            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    header
                        .staggerIn(index: 0, appeared: appeared)

                    if let file = pickedFile {
                        pickedFileCard(file)
                            .staggerIn(index: 1, appeared: appeared)

                        nameCard
                            .staggerIn(index: 2, appeared: appeared)

                        importFooter(file)
                            .staggerIn(index: 3, appeared: appeared)
                    } else {
                        dropZone
                            .staggerIn(index: 1, appeared: appeared)

                        howItWorksCard
                            .staggerIn(index: 2, appeared: appeared)
                    }

                    if let errorMessage {
                        errorBanner(errorMessage)
                    }
                }
                .padding(.horizontal, 28)
                .padding(.bottom, 20)
            }
            .scrollIndicators(.automatic)
        }
        .task {
            try? await Task.sleep(for: .milliseconds(60))
            appeared = true
        }
    }

    // MARK: - Header

    private var header: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text("Import Audio")
                .font(.system(size: 22, weight: .bold, design: .rounded))

            Text("Turn any audio or video file into a session you can transcribe and summarize.")
                .font(.system(size: 11))
                .foregroundStyle(.secondary)
        }
    }

    // MARK: - Drop Zone

    private var dropZone: some View {
        VStack(spacing: 14) {
            ZStack {
                Circle()
                    .fill(Color.accentColor.opacity(isDropTargeted ? 0.2 : 0.1))
                    .frame(width: 64, height: 64)
                Image(systemName: isDropTargeted ? "arrow.down.circle.fill" : "square.and.arrow.down")
                    .font(.system(size: 26, weight: .medium))
                    .foregroundStyle(Color.accentColor)
            }
            .scaleEffect(isDropTargeted ? 1.06 : 1.0)

            VStack(spacing: 4) {
                Text(isDropTargeted ? "Release to add file" : "Drop an audio or video file here")
                    .font(.system(size: 14, weight: .semibold))

                Text("MP3 · WAV · M4A · AAC · AIFF · MP4 · MOV — converted automatically")
                    .font(.system(size: 10))
                    .foregroundStyle(.tertiary)
            }

            if isProbing {
                ProgressView()
                    .controlSize(.small)
            } else {
                Button {
                    browseForFile()
                } label: {
                    HStack(spacing: 5) {
                        Image(systemName: "folder")
                            .font(.system(size: 10, weight: .medium))
                        Text("Browse Files")
                            .font(.system(size: 11, weight: .medium))
                    }
                    .foregroundStyle(.white)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 8)
                    .background(Color.accentColor, in: Capsule())
                }
                .buttonStyle(.plain)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 44)
        .background(.thinMaterial, in: .rect(cornerRadius: 14, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(
                    isDropTargeted ? Color.accentColor.opacity(0.6) : .white.opacity(0.08),
                    style: StrokeStyle(lineWidth: isDropTargeted ? 1.5 : 1, dash: [6, 5])
                )
        )
        .animation(.easeOut(duration: 0.15), value: isDropTargeted)
        .dropDestination(for: URL.self) { urls, _ in
            guard let url = urls.first else { return false }
            handlePicked(url: url)
            return true
        } isTargeted: { targeted in
            isDropTargeted = targeted
        }
    }

    private var howItWorksCard: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "info.circle.fill")
                .font(.system(size: 14))
                .foregroundStyle(.blue)

            VStack(alignment: .leading, spacing: 4) {
                Text("Works like any recorded session")
                    .font(.system(size: 12, weight: .semibold))

                Text("The file is converted to the session audio format and saved as a new session. You can then transcribe it locally or with an API model, and generate summaries — exactly like a live recording.")
                    .font(.system(size: 10))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Spacer()
        }
        .padding(12)
        .background(.thinMaterial, in: .rect(cornerRadius: 12, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(.blue.opacity(0.15), lineWidth: 0.5)
        )
    }

    // MARK: - Picked File

    private func pickedFileCard(_ file: ImportableAudioFile) -> some View {
        HStack(spacing: 14) {
            ZStack {
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(Color.accentColor.opacity(0.1))
                    .frame(width: 44, height: 44)
                Image(systemName: "waveform")
                    .font(.system(size: 18))
                    .foregroundStyle(Color.accentColor)
            }

            VStack(alignment: .leading, spacing: 5) {
                Text(file.fileName)
                    .font(.system(size: 13, weight: .semibold))
                    .lineLimit(1)
                    .truncationMode(.middle)

                HStack(spacing: 5) {
                    metaPill(icon: "doc", text: SessionManager.formatFileSize(file.fileSize))
                    if file.duration > 0 {
                        metaPill(icon: "clock", text: SessionManager.formatDuration(file.duration))
                    }
                    metaPill(icon: "waveform.badge.magnifyingglass", text: file.fileExtension)
                }
            }

            Spacer()

            Button {
                withAnimation(.easeOut(duration: 0.15)) {
                    pickedFile = nil
                    errorMessage = nil
                }
            } label: {
                Text("Change")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 5)
                    .background(.quaternary, in: Capsule())
            }
            .buttonStyle(.plain)
            .disabled(isImporting)
        }
        .padding(14)
        .background(.thinMaterial, in: .rect(cornerRadius: 12, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(.white.opacity(0.06), lineWidth: 0.5)
        )
    }

    private var nameCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("SESSION NAME")
                .font(.system(size: 9, weight: .semibold, design: .rounded))
                .foregroundStyle(.tertiary)
                .tracking(0.5)

            TextField("Session name", text: $sessionName)
                .textFieldStyle(.plain)
                .font(.system(size: 13, weight: .medium))
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Color.primary.opacity(0.06), in: .rect(cornerRadius: 8, style: .continuous))
                .disabled(isImporting)
        }
        .padding(14)
        .background(.thinMaterial, in: .rect(cornerRadius: 12, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(.white.opacity(0.06), lineWidth: 0.5)
        )
    }

    private func importFooter(_ file: ImportableAudioFile) -> some View {
        HStack(spacing: 12) {
            if isImporting {
                VStack(alignment: .leading, spacing: 6) {
                    HStack(spacing: 6) {
                        Text("Converting to session audio…")
                            .font(.system(size: 11, weight: .medium))
                            .foregroundStyle(.secondary)
                        Spacer()
                        Text("\(Int(importProgress * 100))%")
                            .font(.system(size: 11, weight: .semibold, design: .monospaced))
                            .foregroundStyle(.secondary)
                    }
                    ProgressView(value: importProgress)
                        .progressViewStyle(.linear)
                        .tint(Color.accentColor)
                }
            } else {
                Text("The original file is not modified.")
                    .font(.system(size: 10))
                    .foregroundStyle(.tertiary)

                Spacer()

                Button {
                    startImport(file)
                } label: {
                    HStack(spacing: 5) {
                        Image(systemName: "plus.circle.fill")
                            .font(.system(size: 11, weight: .medium))
                        Text("Create Session")
                            .font(.system(size: 12, weight: .semibold))
                    }
                    .foregroundStyle(.white)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 8)
                    .background(Color.accentColor, in: Capsule())
                    .overlay(Capsule().strokeBorder(Color.accentColor.opacity(0.28), lineWidth: 0.6))
                }
                .buttonStyle(.plain)
                .keyboardShortcut(.defaultAction)
            }
        }
        .padding(14)
        .background(.thinMaterial, in: .rect(cornerRadius: 12, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(.white.opacity(0.06), lineWidth: 0.5)
        )
    }

    // MARK: - Error Banner

    private func errorBanner(_ message: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 11))
                .foregroundStyle(.red)
            Text(message)
                .font(.system(size: 11))
                .foregroundStyle(.primary)
                .fixedSize(horizontal: false, vertical: true)
            Spacer()
        }
        .padding(12)
        .background(.red.opacity(0.08), in: .rect(cornerRadius: 10, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(.red.opacity(0.2), lineWidth: 0.5)
        )
    }

    private func metaPill(icon: String, text: String) -> some View {
        HStack(spacing: 3) {
            Image(systemName: icon)
                .font(.system(size: 8))
            Text(text)
                .font(.system(size: 9))
        }
        .foregroundStyle(.secondary)
        .padding(.horizontal, 6)
        .padding(.vertical, 3)
        .background(.quaternary, in: Capsule())
    }

    // MARK: - Actions

    private func browseForFile() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.allowedContentTypes = [.audio, .audiovisualContent, .movie, .mpeg4Movie, .quickTimeMovie]
        panel.message = "Choose an audio or video file to import as a session"

        if panel.runModal() == .OK, let url = panel.url {
            handlePicked(url: url)
        }
    }

    private func handlePicked(url: URL) {
        errorMessage = nil
        isProbing = true

        Task {
            do {
                let file = try await SessionManager.probeImportableAudio(at: url)
                withAnimation(.easeOut(duration: 0.15)) {
                    pickedFile = file
                    sessionName = file.suggestedSessionName
                    isProbing = false
                }
            } catch {
                isProbing = false
                errorMessage = error.localizedDescription
            }
        }
    }

    private func startImport(_ file: ImportableAudioFile) {
        errorMessage = nil
        isImporting = true
        importProgress = 0

        Task {
            do {
                let session = try await sessionManager.importSession(
                    from: file.url,
                    name: sessionName,
                    progress: { value in
                        Task { @MainActor in
                            importProgress = value
                        }
                    }
                )
                isImporting = false
                onImported(session.id)
            } catch {
                isImporting = false
                errorMessage = error.localizedDescription
            }
        }
    }
}
