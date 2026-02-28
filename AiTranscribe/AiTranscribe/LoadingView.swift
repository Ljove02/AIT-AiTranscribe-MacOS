/*
 LoadingView.swift
 =================

 A splash/loading screen shown while the backend is starting up.

 This provides visual feedback to the user during the 1-2 minute
 backend startup time, especially on first launch.
 */

import SwiftUI
import Combine

struct LoadingView: View {
    @EnvironmentObject var backendManager: BackendManager
    var onComplete: (() -> Void)?

    @State private var dotCount = 0
    @State private var showingDetails = false
    @State private var hasError: Bool = false

    private let timer = Timer.publish(every: 0.5, on: .main, in: .common).autoconnect()

    var body: some View {
        VStack(spacing: 24) {
            Spacer()

            // App Icon
            Image(systemName: "waveform.circle.fill")
                .font(.system(size: 80))
                .foregroundStyle(.blue.gradient)
                .symbolEffect(.pulse, options: .repeating)

            // App Name
            Text("AiTranscribe")
                .font(.largeTitle)
                .fontWeight(.bold)

            // Loading indicator
            VStack(spacing: 12) {
                ProgressView()
                    .scaleEffect(1.2)
                    .tint(.blue)

                Text("Starting backend\(String(repeating: ".", count: dotCount))")
                    .font(.headline)
                    .foregroundStyle(.secondary)
                    .frame(width: 180, alignment: .leading)

                if showingDetails {
                    Text("This may take a minute on first launch")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                        .transition(.opacity)
                }
            }
            .padding(.top, 8)

            Spacer()

            // Version info at bottom
            Text("Version 0.1.2")
                .font(.caption)
                .foregroundStyle(.tertiary)
                .padding(.bottom, 20)
        }
        .frame(width: 350, height: 400)
        .background(.ultraThinMaterial)
        .onReceive(timer) { _ in
            dotCount = (dotCount + 1) % 4
        }
        .task {
            // Show details after 3 seconds
            try? await Task.sleep(for: .seconds(3))
            withAnimation {
                showingDetails = true
            }
        }
        .task {
            await backendManager.waitForServerReady(timeout: 180.0)
            onComplete?()
        }
    }
}

#Preview {
    LoadingView()
        .environmentObject(BackendManager())
}
