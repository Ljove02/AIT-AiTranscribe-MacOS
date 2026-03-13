import SwiftUI

// MARK: - About View

struct AboutView: View {
    @EnvironmentObject var backendManager: BackendManager

    var body: some View {
        VStack(spacing: 16) {
            // App Icon
            Group {
                if let nsImage = loadAppIconFromBundle() {
                    Image(nsImage: nsImage)
                        .resizable()
                        .scaledToFit()
                        .frame(width: 96, height: 96)
                        .padding(.top, 20)
                } else {
                    // Fallback to SF Symbol
                    Image(systemName: "mic.circle.fill")
                        .resizable()
                        .scaledToFit()
                        .frame(width: 96, height: 96)
                        .foregroundStyle(.blue.gradient)
                        .padding(.top, 20)
                }
            }

            // App name
            Text("AI-Transcribe")
                .font(.largeTitle)
                .fontWeight(.bold)

            // Version
            Text("Version 0.1.2")
                .font(.subheadline)
                .foregroundColor(.secondary)

            // Description
            Text("Open source picture text transcription")
                .font(.body)
                .multilineTextAlignment(.center)
                .foregroundColor(.secondary)
                .padding(.bottom, 8)

            // Links
            HStack(spacing: 24) {
                // GitHub link
                Link(destination: URL(string: "https://github.com/Ljove02")!) {
                    HStack(spacing: 6) {
                        Image(systemName: "chevron.left.forwardslash.chevron.right")
                            .font(.system(size: 14))
                        Text("GitHub")
                            .font(.caption)
                    }
                }
                .buttonStyle(.plain)
                .foregroundColor(.accentColor)

                // Buy me a coffee / Support
                Link(destination: URL(string: "https://ko-fi.com/yourusername")!) {
                    HStack(spacing: 6) {
                        Image(systemName: "cup.and.saucer.fill")
                            .font(.system(size: 14))
                        Text("Support")
                            .font(.caption)
                    }
                }
                .buttonStyle(.plain)
                .foregroundColor(.accentColor)
            }
            .padding(.vertical, 4)

            Divider()
                .padding(.vertical, 8)

            // Backend status with inline debug console button
            VStack(spacing: 8) {
                HStack(spacing: 8) {
                    Circle()
                        .fill(backendManager.isServerReady ? Color.green : (backendManager.isRunning ? Color.orange : Color.red))
                        .frame(width: 10, height: 10)
                    Text("Backend: \(backendManager.statusMessage)")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    Spacer()

                    // Debug console button (compact, inline)
                    Button {
                        DebugConsoleWindowController.shared.showWindow(backendManager: backendManager)
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: "terminal")
                                .font(.system(size: 11))
                            Text("Debug Console")
                                .font(.caption2)
                        }
                    }
                    .buttonStyle(.link)
                }
            }
            .padding(.horizontal, 12)

            Divider()
                .padding(.vertical, 8)

            // Credits - more prominent with better spacing
            VStack(spacing: 8) {
                Text("Powered by")
                    .font(.caption)
                    .foregroundColor(.secondary)

                HStack(spacing: 16) {
                    VStack(spacing: 2) {
                        Text("NVIDIA")
                            .font(.caption)
                            .fontWeight(.semibold)
                        Text("Parakeet & Nemotron")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }

                    Text("&")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    VStack(spacing: 2) {
                        Text("OpenAI")
                            .font(.caption)
                            .fontWeight(.semibold)
                        Text("Whisper")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                }
            }
            .padding(.bottom, 12)
        }
        .padding()
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
