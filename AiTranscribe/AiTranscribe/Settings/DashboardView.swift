import SwiftUI

// MARK: - Dashboard View

struct DashboardView: View {
    var body: some View {
        ScrollView {
            VStack(spacing: 24) {
            // Welcome Header - centered layout
            VStack(spacing: 12) {
                // App Icon centered at top
                Group {
                    if let nsImage = loadAppIconFromBundle() {
                        Image(nsImage: nsImage)
                            .resizable()
                            .scaledToFit()
                            .frame(width: 72, height: 72)
                    } else {
                        // Fallback to SF Symbol
                        Image(systemName: "mic.circle.fill")
                            .resizable()
                            .scaledToFit()
                            .frame(width: 72, height: 72)
                            .foregroundStyle(.blue.gradient)
                    }
                }

                VStack(spacing: 4) {
                    Text("Welcome to AI-Transcribe")
                        .font(.system(size: 28, weight: .bold))

                    Text("Your AI-powered transcription assistant")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
            }
            .frame(maxWidth: .infinity)
            .padding(.top, 24)

            Divider()
                .padding(.horizontal, 24)

            // Mock content area
            VStack(spacing: 16) {
                Text("Dashboard Coming Soon")
                    .font(.title2.weight(.semibold))
                    .foregroundColor(.secondary)

                Text("This is where you'll see your transcription stats, quick actions, and recent activity.")
                    .font(.body)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 40)

                // Placeholder cards
                HStack(spacing: 16) {
                    PlaceholderCard(
                        icon: "waveform",
                        title: "Quick Stats",
                        subtitle: "View your activity"
                    )

                    PlaceholderCard(
                        icon: "clock.arrow.circlepath",
                        title: "Recent",
                        subtitle: "Latest transcriptions"
                    )

                    PlaceholderCard(
                        icon: "bolt.fill",
                        title: "Quick Actions",
                        subtitle: "Common tasks"
                    )
                }
                .padding(.horizontal, 24)
            }

            // Spacer replaced with fixed padding (Spacer doesn't work in ScrollView)
            Spacer()
                .frame(minHeight: 20)

            // Hint to explore sidebar sections
            VStack(spacing: 8) {
                Text("Explore the sidebar to configure AIT")
                    .font(.footnote)
                    .foregroundColor(.secondary)

                HStack(spacing: 12) {
                    Label("General", systemImage: "gear")
                    Label("Models", systemImage: "cube.box")
                    Label("History", systemImage: "clock.arrow.circlepath")
                }
                .font(.caption)
                .foregroundColor(.secondary)
            }
            .padding(.bottom, 24)
            }
        }
    }
}

// MARK: - Placeholder Card

struct PlaceholderCard: View {
    let icon: String
    let title: String
    let subtitle: String

    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: icon)
                .font(.system(size: 32))
                .foregroundColor(.accentColor.opacity(0.5))

            VStack(spacing: 4) {
                Text(title)
                    .font(.headline)
                    .foregroundColor(.primary.opacity(0.5))

                Text(subtitle)
                    .font(.caption)
                    .foregroundColor(.secondary.opacity(0.5))
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 24)
        .background(Color.gray.opacity(0.05))
        .cornerRadius(12)
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(Color.gray.opacity(0.2), style: StrokeStyle(lineWidth: 1, dash: [5, 5]))
        )
    }
}
