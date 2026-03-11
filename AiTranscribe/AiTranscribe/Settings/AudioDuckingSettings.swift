import SwiftUI

// MARK: - Audio Ducking Settings (Isolated Component)

/// Separate component to isolate audio ducking settings and prevent publishing warnings
struct AudioDuckingSettings: View {
    @EnvironmentObject var appState: AppState

    // Local state to buffer changes and prevent direct binding issues
    @State private var localMuteEnabled: Bool = false
    @State private var localMuteMode: String = "mute"
    @State private var localDuckVolume: Double = 10.0

    var body: some View {
        Group {
            Toggle("Mute playback during recording", isOn: $localMuteEnabled)
                .onChange(of: localMuteEnabled) { _, newValue in
                    // Use animation transaction to absorb the change
                    withAnimation(.easeInOut(duration: 0.2)) {
                        appState.mutePlaybackDuringRecording = newValue
                    }
                }

            if localMuteEnabled {
                VStack(alignment: .leading, spacing: 8) {
                    Picker("Mode", selection: $localMuteMode) {
                        Text("Mute completely").tag("mute")
                        Text("Lower volume").tag("lower")
                    }
                    .pickerStyle(.segmented)
                    .onChange(of: localMuteMode) { _, newValue in
                        // Defer the change to break the publish cycle
                        DispatchQueue.main.async {
                            appState.muteMode = newValue
                        }
                    }

                    if localMuteMode == "lower" {
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Text("Lower by:")
                                    .font(.caption)
                                Text("\(Int(localDuckVolume))%")
                                    .font(.caption)
                                    .fontWeight(.medium)
                                    .monospacedDigit()
                            }

                            Slider(value: $localDuckVolume, in: 10...99, step: 1)
                                .help("How much to reduce volume during recording")
                                .onChange(of: localDuckVolume) { _, newValue in
                                    // Debounce slider changes
                                    DispatchQueue.main.async {
                                        appState.duckVolume = newValue
                                    }
                                }

                            Text("Volume will be reduced by \(Int(localDuckVolume))% while recording. For example, if your volume is at 50%, it will drop to \(Int(Float(50) * (1.0 - Float(localDuckVolume)/100)))%.")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                        }
                        .transition(.opacity)
                    } else {
                        Text("All system audio will be muted during recording")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                            .transition(.opacity)
                    }
                }
                .transition(.opacity)
            }
        }
        .onAppear {
            // Initialize local state from appState
            localMuteEnabled = appState.mutePlaybackDuringRecording
            localMuteMode = appState.muteMode
            localDuckVolume = appState.duckVolume
        }
    }
}
