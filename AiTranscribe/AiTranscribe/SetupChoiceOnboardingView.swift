import SwiftUI

struct SetupChoiceOnboardingView: View {
    let selection: OnboardingSetupPreference?
    let onSelect: (OnboardingSetupPreference) -> Void
    let onBack: () -> Void

    @State private var hoveredPreference: OnboardingSetupPreference?

    private var previewPreference: OnboardingSetupPreference {
        hoveredPreference ?? selection ?? .both
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            OnboardingHeader(
                eyebrow: "Model setup",
                title: "Choose how AIT should transcribe",
                subtitle: "AIT can run privately on this Mac, use public API providers for speed, or configure both so you can switch later."
            )
            .padding(.bottom, 42)

            HStack(alignment: .top, spacing: 46) {
                VStack(alignment: .leading, spacing: 16) {
                    Text("Pick a setup path")
                        .font(.system(size: 13, weight: .bold, design: .rounded))
                        .tracking(1.1)
                        .textCase(.uppercase)
                        .foregroundStyle(.secondary)

                    VStack(alignment: .leading, spacing: 4) {
                        ForEach(OnboardingSetupPreference.allCases) { preference in
                            SetupChoiceRow(
                                preference: preference,
                                selected: previewPreference == preference,
                                action: { onSelect(preference) }
                            )
                            .onHover { isHovering in
                                hoveredPreference = isHovering ? preference : nil
                            }
                        }
                    }

                    HStack(alignment: .top, spacing: 10) {
                        Image(systemName: "info.circle.fill")
                            .font(.system(size: 13, weight: .semibold))
                            .foregroundStyle(.blue)
                            .padding(.top, 1)

                        Text("You can change this later in Settings > Models. This step only decides which setup pages to show now.")
                            .font(.system(size: 13, weight: .medium))
                            .lineSpacing(2)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .padding(.top, 18)
                }
                .frame(width: 308, alignment: .leading)

                SetupChoiceDetail(preference: previewPreference)
                    .frame(maxWidth: .infinity, alignment: .topLeading)
                    .id(previewPreference.id)
                    .transition(.opacity.combined(with: .scale(scale: 0.985)))
            }

            Spacer()

            HStack {
                OnboardingSecondaryActionButton(
                    title: "Back",
                    systemImage: "arrow.left",
                    action: onBack
                )

                Spacer()
            }
        }
        .padding(.horizontal, 42)
        .padding(.top, 34)
        .padding(.bottom, 34)
        .animation(.spring(duration: 0.28, bounce: 0.12), value: previewPreference)
    }
}

private struct SetupChoiceRow: View {
    let preference: OnboardingSetupPreference
    let selected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 14) {
                ZStack {
                    Circle()
                        .fill(selected ? Color.blue : .white.opacity(0.08))
                        .frame(width: 38, height: 38)

                    Image(systemName: preference.icon)
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundStyle(selected ? .white : .secondary)
                }

                VStack(alignment: .leading, spacing: 3) {
                    Text(preference.title)
                        .font(.system(size: 18, weight: .bold, design: .rounded))
                        .foregroundStyle(.primary)
                }

                Spacer()

                Image(systemName: selected ? "checkmark.circle.fill" : "arrow.right.circle")
                    .font(.system(size: 18, weight: .semibold))
                    .foregroundStyle(selected ? .blue : .secondary.opacity(0.68))
            }
            .padding(.vertical, 13)
            .padding(.horizontal, 12)
            .contentShape(.rect)
            .background(alignment: .leading) {
                Capsule()
                    .fill(Color.blue)
                    .frame(width: selected ? 4 : 0)
                    .padding(.vertical, 9)
            }
            .background {
                if selected {
                    RoundedRectangle(cornerRadius: 18, style: .continuous)
                        .fill(.white.opacity(0.07))
                }
            }
        }
        .buttonStyle(.plain)
        .accessibilityLabel("\(preference.title), \(preference.subtitle)")
        .accessibilityAddTraits(selected ? [.isSelected] : [])
    }
}

private struct SetupChoiceDetail: View {
    let preference: OnboardingSetupPreference

    var body: some View {
        VStack(alignment: .leading, spacing: 28) {
            HStack(alignment: .center, spacing: 18) {
                ZStack {
                    Circle()
                        .fill(
                            RadialGradient(
                                colors: [
                                    Color.blue.opacity(0.30),
                                    Color.teal.opacity(0.13),
                                    Color.clear
                                ],
                                center: .center,
                                startRadius: 4,
                                endRadius: 74
                            )
                        )
                        .frame(width: 132, height: 132)

                    Circle()
                        .strokeBorder(.white.opacity(0.12), lineWidth: 1)
                        .frame(width: 104, height: 104)

                    Image(systemName: preference.icon)
                        .font(.system(size: 38, weight: .semibold))
                        .foregroundStyle(.white)
                        .frame(width: 74, height: 74)
                        .background(.blue, in: Circle())
                        .shadow(color: .blue.opacity(0.24), radius: 16, x: 0, y: 10)
                }

                VStack(alignment: .leading, spacing: 8) {
                    Text(preference.title)
                        .font(.system(size: 34, weight: .bold, design: .rounded))
                        .foregroundStyle(.primary)

                    Text(preference.subtitle)
                        .font(.system(size: 12, weight: .bold, design: .rounded))
                        .tracking(1.1)
                        .textCase(.uppercase)
                        .foregroundStyle(.secondary)
                }
            }

            VStack(alignment: .leading, spacing: 16) {
                ForEach(points, id: \.self) { point in
                    HStack(alignment: .top, spacing: 12) {
                        Image(systemName: "checkmark")
                            .font(.system(size: 11, weight: .bold))
                            .foregroundStyle(.blue)
                            .frame(width: 22, height: 22)
                            .background(.blue.opacity(0.14), in: Circle())

                        Text(point)
                            .font(.system(size: 15, weight: .medium))
                            .lineSpacing(3)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
            .frame(maxWidth: 440, alignment: .leading)

            modePath
                .padding(.top, 6)
        }
    }

    private var points: [String] {
        switch preference {
        case .local:
            return [
                "Audio and transcripts stay on this Mac.",
                "Best default for private notes, meetings, and sensitive data.",
                "Requires model downloads and uses local memory/GPU."
            ]
        case .api:
            return [
                "Audio is sent to the selected provider for processing.",
                "Usually fastest and does not require large downloads.",
                "Provider retention/training policies are controlled by that provider."
            ]
        case .both:
            return [
                "Set up local privacy plus API speed in one pass.",
                "Switch from the menu bar depending on the task.",
                "Recommended if you want to compare quality and speed."
            ]
        }
    }

    private var modePath: some View {
        HStack(spacing: 10) {
            ForEach(pathItems) { item in
                HStack(spacing: 10) {
                    if item.index > 0 {
                        Rectangle()
                            .fill(.white.opacity(0.16))
                            .frame(width: 28, height: 1)
                    }

                    VStack(alignment: .leading, spacing: 4) {
                        Image(systemName: item.icon)
                            .font(.system(size: 12, weight: .bold))
                            .foregroundStyle(.blue)
                            .frame(width: 26, height: 26)
                            .background(.white.opacity(0.08), in: Circle())

                        Text(item.title)
                            .font(.system(size: 10, weight: .bold, design: .rounded))
                            .tracking(0.7)
                            .textCase(.uppercase)
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Setup route preview")
    }

    private var pathItems: [SetupPathItem] {
        switch preference {
        case .local:
            return [
                SetupPathItem(index: 0, icon: "lock.shield", title: "Local"),
                SetupPathItem(index: 1, icon: "keyboard", title: "Keys"),
                SetupPathItem(index: 2, icon: "checkmark", title: "Ready")
            ]
        case .api:
            return [
                SetupPathItem(index: 0, icon: "globe", title: "API"),
                SetupPathItem(index: 1, icon: "keyboard", title: "Keys"),
                SetupPathItem(index: 2, icon: "checkmark", title: "Ready")
            ]
        case .both:
            return [
                SetupPathItem(index: 0, icon: "lock.shield", title: "Local"),
                SetupPathItem(index: 1, icon: "globe", title: "API"),
                SetupPathItem(index: 2, icon: "keyboard", title: "Keys"),
                SetupPathItem(index: 3, icon: "checkmark", title: "Ready")
            ]
        }
    }
}

private struct SetupPathItem: Identifiable {
    let index: Int
    let icon: String
    let title: String

    var id: Int { index }
}

#Preview {
    SetupChoiceOnboardingView(selection: nil, onSelect: { _ in }, onBack: {})
        .frame(width: 900, height: 720)
}
