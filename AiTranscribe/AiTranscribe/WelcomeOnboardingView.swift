/*
 WelcomeOnboardingView.swift
 ===========================

 First screen of onboarding - introduces AIT to the user.
 
 Features:
 - App icon and welcome message
 - Brief description
 - Get Started button
 */

import SwiftUI
import Combine

struct WelcomeOnboardingView: View {
    let onNext: () -> Void

    @EnvironmentObject var backendManager: BackendManager

    /// Track backend readiness state
    @State private var isWaitingForBackend = true
    @State private var dotCount = 0

    private let timer = Timer.publish(every: 0.5, on: .main, in: .common).autoconnect()

    var body: some View {
        VStack(spacing: 30) {
            Spacer()
            
            // App Icon
            // Load from bundle resources if "Logo" asset is not available
            Group {
                if let nsImage = NSImage(named: "Logo") ?? loadIconFromBundle() {
                    Image(nsImage: nsImage)
                        .resizable()
                        .scaledToFit()
                        .frame(width: 128, height: 128)
                        .shadow(color: .black.opacity(0.2), radius: 20, x: 0, y: 10)
                } else {
                    // Fallback if icon not found
                    Image(systemName: "mic.circle.fill")
                        .resizable()
                        .scaledToFit()
                        .frame(width: 128, height: 128)
                        .foregroundStyle(.blue.gradient)
                        .shadow(color: .black.opacity(0.2), radius: 20, x: 0, y: 10)
                }
            }
            
            // Title
            Text("Welcome to AIT")
                .font(.system(size: 36, weight: .bold))
            
            // Subtitle
            Text("Your AI-powered transcription assistant for macOS")
                .font(.title3)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
            
            // Description - centralized features
            VStack(spacing: 12) {
                CentralizedFeatureRow(icon: "mic.fill", text: "Transcribe audio with AI models")
                CentralizedFeatureRow(icon: "bolt.fill", text: "Automate workflows with keyboard shortcuts")
                CentralizedFeatureRow(icon: "chart.line.uptrend.xyaxis", text: "Boost productivity with real-time transcription")
            }
            .padding(.horizontal, 40)
            
            Spacer()
            Spacer() // Extra spacer to push button down consistently
            
            // Get Started Button (or Loading state)
            VStack(spacing: 8) {
                if isWaitingForBackend {
                    // Show loading state while waiting for backend
                    VStack(spacing: 12) {
                        HStack(spacing: 8) {
                            ProgressView()
                                .scaleEffect(0.8)
                            Text("Starting backend\(String(repeating: ".", count: dotCount))")
                                .font(.title3.weight(.medium))
                                .foregroundColor(.secondary)
                                .frame(width: 180, alignment: .leading)
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 14)

                        Text("This may take a minute on first launch")
                            .font(.caption)
                            .foregroundStyle(.tertiary)
                    }
                    .padding(.horizontal, 60)
                } else {
                    // Backend ready - show Get Started button
                    Button(action: onNext) {
                        HStack {
                            Text("Get Started")
                                .font(.title3.weight(.semibold))
                            Image(systemName: "arrow.right")
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 14)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .padding(.horizontal, 60)

                    // Footer - moved inside button section
                    Text("First time? Let's set up your app.")
                        .font(.footnote)
                        .foregroundColor(.secondary)
                }

                Spacer().frame(height: 20)
            }
        }
        .padding()
        .onReceive(timer) { _ in
            dotCount = (dotCount + 1) % 4
        }
        .task {
            // Wait for backend to be ready
            while !backendManager.isServerReady {
                try? await Task.sleep(for: .milliseconds(500))
            }
            // Backend is ready
            isWaitingForBackend = false
        }
    }
}

// MARK: - Feature Row (Left-aligned - Original)

struct FeatureRow: View {
    let icon: String
    let text: String
    
    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .frame(width: 24)
                .foregroundColor(.accentColor)
            
            Text(text)
                .font(.body)
                .foregroundColor(.primary)
            
            Spacer()
        }
    }
}

// MARK: - Centralized Feature Row

struct CentralizedFeatureRow: View {
    let icon: String
    let text: String
    
    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .frame(width: 24)
                .foregroundColor(.accentColor)
            
            Text(text)
                .font(.body)
                .foregroundColor(.primary)
        }
        .frame(maxWidth: .infinity) // Center the content
    }
}

// MARK: - Icon Loading Helper

/// Attempts to load the app icon from the bundle resources
/// Looks for icon files like "Icon-iOS-Default-256x256@1x.png" in the bundle
private func loadIconFromBundle() -> NSImage? {
    // List of possible icon filenames to try
    let iconNames = [
        "Icon-iOS-Default-256x256@1x",
        "Icon-iOS-Default-128x128@1x",
        "Icon-iOS-Default-512x512@1x",
        "AppIcon",
        "Icon"
    ]
    
    for iconName in iconNames {
        // Try loading as PNG
        if let image = NSImage(named: iconName) {
            return image
        }
        
        // Try loading from bundle resources
        if let imagePath = Bundle.main.path(forResource: iconName, ofType: "png"),
           let image = NSImage(contentsOfFile: imagePath) {
            return image
        }
    }
    
    return nil
}

// MARK: - Preview

#Preview {
    WelcomeOnboardingView(onNext: {})
        .environmentObject(BackendManager())
        .frame(width: 800, height: 750)
}
