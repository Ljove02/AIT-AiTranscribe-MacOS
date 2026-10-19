import SwiftUI

struct PolishPreviewOnboardingView: View {
    let onNext: () -> Void
    let onBack: () -> Void

    private let steps: [NextStepItem] = [
        NextStepItem(
            number: "01",
            icon: "mic.fill",
            title: "Capture quick ideas",
            text: "Use the menu bar or shortcut to turn short voice notes into text."
        ),
        NextStepItem(
            number: "02",
            icon: "waveform",
            title: "Record longer sessions",
            text: "Capture meetings, desktop audio, or mixed audio, then transcribe when the session is done."
        ),
        NextStepItem(
            number: "03",
            icon: "sparkles",
            title: "Create your Polishes",
            text: "Clean transcripts, extract tasks, draft replies, or combine voice, clipboard, and screenshots."
        ),
        NextStepItem(
            number: "04",
            icon: "brain.head.profile",
            title: "Add local LLMs",
            text: "Download local language models later if you want private summaries and transformations."
        )
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            OnboardingHeader(
                eyebrow: "Next",
                title: "Next steps",
                subtitle: "AIT is ready for the basics. These are the first places worth exploring after setup."
            )
            .padding(.bottom, 34)

            VStack(alignment: .leading, spacing: 0) {
                ForEach(Array(steps.enumerated()), id: \.element.id) { index, item in
                    NextStepLine(
                        item: item,
                        isLast: index == steps.count - 1
                    )
                    .padding(.leading, CGFloat(index % 2) * 34)
                }
            }
            .frame(maxWidth: 760, alignment: .leading)

            Spacer()

            HStack(alignment: .firstTextBaseline) {
                Label("Everything here can be changed later in Settings.", systemImage: "slider.horizontal.3")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(.secondary)

                Spacer()

                OnboardingNavBar(onBack: onBack, onNext: onNext)
                    .frame(width: 520)
            }
        }
        .padding(34)
    }
}

private struct NextStepItem: Identifiable {
    let id = UUID()
    let number: String
    let icon: String
    let title: String
    let text: String
}

private struct NextStepLine: View {
    let item: NextStepItem
    let isLast: Bool

    var body: some View {
        HStack(alignment: .top, spacing: 18) {
            VStack(spacing: 9) {
                ZStack {
                    Circle()
                        .fill(.white.opacity(0.08))
                        .frame(width: 42, height: 42)
                        .overlay(
                            Circle()
                                .strokeBorder(.white.opacity(0.12), lineWidth: 1)
                        )

                    Image(systemName: item.icon)
                        .font(.system(size: 15, weight: .bold))
                        .foregroundStyle(.blue)
                }

                if !isLast {
                    Rectangle()
                        .fill(.white.opacity(0.11))
                        .frame(width: 2, height: 38)
                }
            }

            VStack(alignment: .leading, spacing: 6) {
                Text(item.number)
                    .font(.system(size: 11, weight: .bold, design: .rounded))
                    .tracking(1.2)
                    .foregroundStyle(.blue)

                Text(item.title)
                    .font(.system(size: 20, weight: .bold, design: .rounded))
                    .foregroundStyle(.primary)

                Text(item.text)
                    .font(.system(size: 14, weight: .medium))
                    .lineSpacing(2)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(.top, 2)

            Spacer()
        }
        .padding(.bottom, isLast ? 0 : 8)
    }
}

#Preview {
    PolishPreviewOnboardingView(onNext: {}, onBack: {})
        .frame(width: 900, height: 720)
}
