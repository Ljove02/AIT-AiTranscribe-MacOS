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
    @State private var waitTime: TimeInterval = 0

    private let timer = Timer.publish(every: 0.5, on: .main, in: .common).autoconnect()

    var body: some View {
        VStack(alignment: .center, spacing: 0) {
            Spacer(minLength: 26)

            VStack(alignment: .center, spacing: 28) {
                appIcon

                VStack(alignment: .center, spacing: 14) {
                    Text("AITRANSCRIBE")
                        .font(.system(size: 11, weight: .bold, design: .rounded))
                        .tracking(1.7)
                        .foregroundStyle(.secondary)

                    Text("Set up voice capture.")
                        .font(.system(size: 43, weight: .bold, design: .rounded))
                        .tracking(-1.3)
                        .lineSpacing(-2)
                        .foregroundStyle(.primary)
                        .multilineTextAlignment(.center)
                        .fixedSize(horizontal: false, vertical: true)

                    Text("Choose local privacy, public speed, global shortcuts, and the output tools that turn voice into something useful.")
                        .font(.system(size: 16, weight: .medium))
                        .lineSpacing(3)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                        .frame(maxWidth: 600, alignment: .center)
                }

                capabilityChips

                welcomeHighlights
            }
            .frame(maxWidth: .infinity, alignment: .center)

            Spacer()

            HStack(alignment: .center) {
                backendStatus

                Spacer()

                OnboardingPrimaryActionButton(
                    title: "Get Started",
                    systemImage: "arrow.right",
                    action: onNext
                )
            }
        }
        .padding(.horizontal, 42)
        .padding(.top, 44)
        .padding(.bottom, 34)
        .onReceive(timer) { _ in
            dotCount = (dotCount + 1) % 4
            // Update backend readiness (non-blocking)
            if backendManager.isServerReady {
                isWaitingForBackend = false
            }
        }
    }

    private var capabilityChips: some View {
        HStack(spacing: 8) {
            WelcomeChip(icon: "waveform", text: "Transcribe")
            WelcomeChip(icon: "sparkles", text: "Polish")
            WelcomeChip(icon: "rectangle.on.rectangle", text: "Sessions")
            WelcomeChip(icon: "chart.xyaxis.line", text: "Usage")
        }
    }

    private var welcomeHighlights: some View {
        HStack(alignment: .top, spacing: 28) {
            WelcomeHighlight(icon: "lock.shield", text: "Run private models on this Mac")
            WelcomeHighlight(icon: "globe", text: "Use APIs when speed matters")
            WelcomeHighlight(icon: "keyboard", text: "Start capture from anywhere")
        }
        .frame(maxWidth: 650)
    }

    private var appIcon: some View {
        Group {
            if let nsImage = NSImage(named: "Logo") ?? loadIconFromBundle() {
                Image(nsImage: nsImage)
                    .resizable()
                    .scaledToFit()
            } else {
                Image(systemName: "mic.circle.fill")
                    .resizable()
                    .scaledToFit()
                    .foregroundStyle(.blue.gradient)
            }
        }
        .frame(width: 94, height: 94)
        .shadow(color: .black.opacity(0.2), radius: 20, x: 0, y: 10)
    }

    private var backendStatus: some View {
        HStack(spacing: 8) {
            if isWaitingForBackend {
                ProgressView()
                    .scaleEffect(0.62)
                Text("Backend starting in background\(String(repeating: ".", count: dotCount))")
            } else {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundStyle(.green)
                Text("Ready for setup")
            }
        }
        .font(.caption.weight(.medium))
        .foregroundColor(.secondary)
    }
}

private struct WelcomeHighlight: View {
    let icon: String
    let text: String

    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: icon)
                .font(.system(size: 16, weight: .semibold))
                .foregroundStyle(.blue)

            Text(text)
                .font(.system(size: 12, weight: .semibold))
                .lineSpacing(2)
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity)
    }
}

private struct WelcomeChip: View {
    let icon: String
    let text: String

    var body: some View {
        HStack(spacing: 7) {
            Image(systemName: icon)
                .font(.system(size: 11, weight: .bold))
            Text(text)
                .font(.system(size: 12, weight: .bold, design: .rounded))
        }
        .foregroundStyle(.primary)
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(.white.opacity(0.07), in: Capsule())
        .overlay(
            Capsule()
                .strokeBorder(.white.opacity(0.10), lineWidth: 1)
        )
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

private struct WelcomeFeatureCard: View {
    let icon: String
    let title: String
    let text: String

    var body: some View {
        OnboardingGlassCard {
            VStack(alignment: .leading, spacing: 12) {
                Image(systemName: icon)
                    .font(.system(size: 22, weight: .semibold))
                    .foregroundStyle(.blue)
                    .frame(width: 44, height: 44)
                    .background(.blue.opacity(0.16), in: RoundedRectangle(cornerRadius: 14, style: .continuous))

                Text(title)
                    .font(.system(size: 16, weight: .bold, design: .rounded))

                Text(text)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .frame(maxWidth: .infinity, minHeight: 160, alignment: .topLeading)
        }
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
