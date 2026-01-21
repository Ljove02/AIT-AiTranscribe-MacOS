/*
 OnboardingContainerView.swift
 ==============================

 Container view that manages the onboarding flow.
 
 Features:
 - Progress bar at top (full width)
 - Screen navigation with Next/Back buttons
 - Tracks completion state in UserDefaults
 */

import SwiftUI

struct OnboardingContainerView: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var backendManager: BackendManager

    /// Environment action to open settings
    @Environment(\.openSettings) private var openSettings
    
    /// Current screen index (0-5)
    @State private var currentScreen: Int = 0
    
    /// Total number of screens
    private let totalScreens: Int = 6
    
    /// Whether onboarding is complete
    @AppStorage("hasCompletedOnboarding") private var hasCompletedOnboarding = false
    
    var body: some View {
        VStack(spacing: 0) {
            // Progress bar at top (full width)
            ProgressBar(progress: progress)
            
            // Current screen content
            ZStack {
                switch currentScreen {
                case 0:
                    WelcomeOnboardingView(onNext: nextScreen)
                        .transition(.opacity)
                case 1:
                    MicrophoneOnboardingView(onNext: nextScreen, onBack: previousScreen)
                        .transition(.opacity)
                case 2:
                    ModelsOnboardingView(onNext: nextScreen, onBack: previousScreen)
                        .transition(.opacity)
                case 3:
                    ShortcutsOnboardingView(onNext: nextScreen, onBack: previousScreen)
                        .transition(.opacity)
                case 4:
                    TestTranscriptionOnboardingView(onNext: nextScreen, onBack: previousScreen)
                        .transition(.opacity)
                case 5:
                    ReadyOnboardingView(onFinish: finishOnboarding)
                        .transition(.opacity)
                default:
                    EmptyView()
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .frame(width: 800, height: 750)
    }
    
    /// Calculate progress (0.0 to 1.0)
    private var progress: Double {
        Double(currentScreen + 1) / Double(totalScreens)
    }
    
    /// Navigate to next screen
    private func nextScreen() {
        withAnimation(.easeInOut(duration: 0.3)) {
            if currentScreen < totalScreens - 1 {
                currentScreen += 1
            }
        }
    }
    
    /// Navigate to previous screen
    private func previousScreen() {
        withAnimation(.easeInOut(duration: 0.3)) {
            if currentScreen > 0 {
                currentScreen -= 1
            }
        }
    }
    
    /// Complete onboarding and close
    private func finishOnboarding() {
        hasCompletedOnboarding = true

        // Auto-load preferred model
        Task {
            await appState.loadPreferredModel()
        }

        // Close onboarding window
        if let window = NSApp.windows.first(where: { $0.identifier?.rawValue == "onboarding" }) {
            window.close()
        }

        // Open Settings window
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
            openSettings()
        }
    }
}

// MARK: - Progress Bar

struct ProgressBar: View {
    let progress: Double
    
    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                // Background track
                Rectangle()
                    .fill(Color.gray.opacity(0.2))
                
                // Progress fill
                Rectangle()
                    .fill(Color.accentColor)
                    .frame(width: geometry.size.width * progress)
            }
        }
        .frame(height: 4)
        .animation(.easeInOut(duration: 0.3), value: progress)
    }
}

// MARK: - Preview

#Preview {
    OnboardingContainerView()
        .environmentObject(AppState())
        .environmentObject(BackendManager())
}
