import SwiftUI

// MARK: - Animated Downloading Bar

/// Animated indeterminate progress bar that moves left to right (shimmer effect)
struct AnimatedDownloadingBar: View {
    @State private var animationOffset: CGFloat = 0

    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                // Background track
                RoundedRectangle(cornerRadius: 3)
                    .fill(Color.secondary.opacity(0.2))

                // Animated shimmer/wave that moves left to right
                RoundedRectangle(cornerRadius: 3)
                    .fill(
                        LinearGradient(
                            colors: [
                                Color.accentColor.opacity(0.3),
                                Color.accentColor,
                                Color.accentColor.opacity(0.3)
                            ],
                            startPoint: .leading,
                            endPoint: .trailing
                        )
                    )
                    .frame(width: geometry.size.width * 0.4) // Wave is 40% of bar width
                    .offset(x: -geometry.size.width * 0.4 + (geometry.size.width * 1.4 * animationOffset))
            }
            .clipShape(RoundedRectangle(cornerRadius: 3))
            .onAppear {
                // Reset to start position
                animationOffset = 0
                // Start continuous animation
                withAnimation(
                    .linear(duration: 1.5)
                    .repeatForever(autoreverses: false)
                ) {
                    animationOffset = 1.0
                }
            }
        }
    }
}
