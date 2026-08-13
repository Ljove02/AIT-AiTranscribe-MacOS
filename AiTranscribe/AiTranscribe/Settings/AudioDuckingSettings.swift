import SwiftUI

// MARK: - Audio Ducking Settings (Isolated Component)

/// Single expandable container for audio ducking — expands when toggled on.
struct AudioDuckingSettings: View {
    @EnvironmentObject var appState: AppState

    @State private var localMuteEnabled: Bool = false
    @State private var localMuteMode: String = "mute"
    @State private var localDuckVolume: Double = 10.0

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Top row: title + description + toggle
            HStack(alignment: .center, spacing: 16) {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Mute playback during recording")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(.primary)

                    Text("Silence or lower system audio while you're recording.")
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                }

                Spacer()

                Toggle("", isOn: $localMuteEnabled)
                    .toggleStyle(.switch)
                    .labelsHidden()
                    .onChange(of: localMuteEnabled) { _, newValue in
                        withAnimation(.easeInOut(duration: 0.2)) {
                            appState.mutePlaybackDuringRecording = newValue
                        }
                    }
            }

            // Expanded options (inset inside the same container)
            if localMuteEnabled {
                Rectangle()
                    .fill(.white.opacity(0.08))
                    .frame(height: 0.5)
                    .padding(.vertical, 10)

                // Mode picker
                HStack(alignment: .center, spacing: 16) {
                    VStack(alignment: .leading, spacing: 3) {
                        Text("Mode")
                            .font(.system(size: 13, weight: .medium))
                            .foregroundStyle(.primary)

                        Text(localMuteMode == "mute"
                             ? "All system audio will be muted."
                             : "Volume reduced by \(Int(localDuckVolume))% while recording.")
                            .font(.system(size: 11))
                            .foregroundStyle(.secondary)
                    }

                    Spacer()

                    muteModePills
                }
                .padding(.horizontal, 10)

                // Volume slider (only when "lower" mode)
                if localMuteMode == "lower" {
                    Rectangle()
                        .fill(.white.opacity(0.08))
                        .frame(height: 0.5)
                        .padding(.vertical, 10)

                    HStack(alignment: .center, spacing: 16) {
                        Text("Lower by")
                            .font(.system(size: 13, weight: .medium))
                            .foregroundStyle(.primary)

                        Slider(value: $localDuckVolume, in: 10...99, step: 1)
                            .onChange(of: localDuckVolume) { _, newValue in
                                DispatchQueue.main.async {
                                    appState.duckVolume = newValue
                                }
                            }

                        Text("\(Int(localDuckVolume))%")
                            .font(.system(size: 13, weight: .semibold, design: .rounded))
                            .monospacedDigit()
                            .foregroundStyle(.primary)
                            .frame(width: 40, alignment: .trailing)
                    }
                    .padding(.horizontal, 10)
                }
            }
        }
        .padding(.vertical, 10)
        .padding(.horizontal, 14)
        .background(.thinMaterial, in: .rect(cornerRadius: 10, style: .continuous))
        .padding(.vertical, 2)
        .animation(.easeInOut(duration: 0.2), value: localMuteEnabled)
        .animation(.easeInOut(duration: 0.2), value: localMuteMode)
        .onAppear {
            // Set initial state without animation to prevent expand flash
            var transaction = Transaction()
            transaction.disablesAnimations = true
            withTransaction(transaction) {
                localMuteEnabled = appState.mutePlaybackDuringRecording
                localMuteMode = appState.muteMode
                localDuckVolume = appState.duckVolume
            }
        }
    }

    private var muteModePills: some View {
        HStack(spacing: 2) {
            muteModeButton(title: "Mute completely", mode: "mute", width: 116)
            muteModeButton(title: "Lower volume", mode: "lower", width: 108)
        }
        .padding(3)
        .background(Color.primary.opacity(0.075), in: Capsule())
        .overlay(Capsule().strokeBorder(.white.opacity(0.06), lineWidth: 0.5))
    }

    private func muteModeButton(title: String, mode: String, width: CGFloat) -> some View {
        let selected = localMuteMode == mode
        return Button {
            withAnimation(.easeOut(duration: 0.12)) {
                localMuteMode = mode
                appState.muteMode = mode
            }
        } label: {
            Text(title)
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(selected ? .white : .secondary)
                .frame(width: width, height: 26)
                .background(selected ? AnyShapeStyle(Color.accentColor) : AnyShapeStyle(.clear), in: Capsule())
        }
        .buttonStyle(.plain)
    }
}
