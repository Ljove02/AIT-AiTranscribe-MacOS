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

                    Picker("", selection: $localMuteMode) {
                        Text("Mute completely").tag("mute")
                        Text("Lower volume").tag("lower")
                    }
                    .labelsHidden()
                    .pickerStyle(.segmented)
                    .frame(width: 200)
                    .onChange(of: localMuteMode) { _, newValue in
                        DispatchQueue.main.async {
                            appState.muteMode = newValue
                        }
                    }
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
}
