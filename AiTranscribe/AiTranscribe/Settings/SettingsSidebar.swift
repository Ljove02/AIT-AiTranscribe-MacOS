import SwiftUI

// MARK: - Settings Section Enum

enum SettingsSection: String, CaseIterable, Identifiable {
    case dashboard
    case general
    case models
    case sessions
    case history
    case shortcuts
    case about

    var id: String { rawValue }

    var label: String {
        switch self {
        case .dashboard: return "Dashboard"
        case .general:   return "General"
        case .models:    return "Models"
        case .sessions:  return "Sessions"
        case .history:   return "History"
        case .shortcuts: return "Shortcuts"
        case .about:     return "About"
        }
    }

    var icon: String {
        switch self {
        case .dashboard: return "square.grid.2x2"
        case .general:   return "gear"
        case .models:    return "cube.box"
        case .sessions:  return "waveform.circle"
        case .history:   return "clock.arrow.circlepath"
        case .shortcuts: return "keyboard"
        case .about:     return "info.circle"
        }
    }
}

// MARK: - Settings Sidebar

struct SettingsSidebar: View {
    @Binding var selectedSection: SettingsSection
    let isExpanded: Bool
    @State private var hoveredSection: SettingsSection?

    private var sidebarWidth: CGFloat { isExpanded ? 200 : 52 }

    var body: some View {
        VStack(spacing: 0) {
            // App branding
            sidebarHeader
                .padding(.top, 10)
                .padding(.bottom, 14)

            // Navigation items
            VStack(spacing: 3) {
                ForEach(SettingsSection.allCases) { section in
                    SidebarItem(
                        section: section,
                        isSelected: selectedSection == section,
                        isHovered: hoveredSection == section,
                        isExpanded: isExpanded
                    )
                    .onTapGesture {
                        withAnimation(.easeInOut(duration: 0.2)) {
                            selectedSection = section
                        }
                    }
                    .onHover { hovering in
                        hoveredSection = hovering ? section : nil
                    }
                }
            }
            .padding(.horizontal, isExpanded ? 10 : 6)

            Spacer()

            // Version badge
            if isExpanded {
                Text("v\(Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "0.2")")
                    .font(.system(size: 10, weight: .medium, design: .rounded))
                    .foregroundColor(.white.opacity(0.45))
                    .padding(.vertical, 3)
                    .padding(.horizontal, 9)
                    .background(.white.opacity(0.08), in: .capsule)
                    .padding(.bottom, 14)
                    .transition(.opacity.combined(with: .scale(scale: 0.8)))
            } else {
                Text("v\(Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "0.2")")
                    .font(.system(size: 8, weight: .medium, design: .rounded))
                    .foregroundStyle(.white.opacity(0.25))
                    .padding(.bottom, 14)
                    .transition(.opacity)
            }
        }
        .frame(width: sidebarWidth)
        .background(
            Color(nsColor: .windowBackgroundColor).opacity(0.45)
        )
        .clipShape(
            UnevenRoundedRectangle(
                topLeadingRadius: 0,
                bottomLeadingRadius: 0,
                bottomTrailingRadius: 0,
                topTrailingRadius: 14
            )
        )
    }

    private var sidebarHeader: some View {
        HStack(spacing: 8) {
            Group {
                if let nsImage = loadAppIconFromBundle() {
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
            .frame(width: 24, height: 24)
            .clipShape(.rect(cornerRadius: 6, style: .continuous))

            if isExpanded {
                Text("AI-Transcribe")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(.primary)
                    .lineLimit(1)
                    .transition(.opacity.combined(with: .move(edge: .leading)))

                Spacer()
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.horizontal, isExpanded ? 14 : 0)
    }
}

// MARK: - Sidebar Item

struct SidebarItem: View {
    let section: SettingsSection
    let isSelected: Bool
    let isHovered: Bool
    let isExpanded: Bool

    var body: some View {
        HStack(spacing: isExpanded ? 10 : 0) {
            Image(systemName: section.icon)
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(isSelected ? .white : .secondary)
                .frame(width: 22, height: 22)

            if isExpanded {
                Text(section.label)
                    .font(.system(size: 13, weight: isSelected ? .semibold : .regular))
                    .foregroundStyle(isSelected ? .white : .primary)
                    .transition(.opacity.combined(with: .move(edge: .leading)))

                Spacer()
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.horizontal, isExpanded ? 10 : 0)
        .padding(.vertical, 7)
        .background(itemBackground)
        .clipShape(.rect(cornerRadius: 10, style: .continuous))
        .contentShape(.rect(cornerRadius: 10))
        .help(isExpanded ? "" : section.label)
    }

    @ViewBuilder
    private var itemBackground: some View {
        if isSelected {
            Color.accentColor
        } else if isHovered {
            Color.primary.opacity(0.06)
        } else {
            Color.clear
        }
    }
}

// MARK: - Shared Icon Loading Helper

/// Attempts to load the app icon from the bundle resources
func loadAppIconFromBundle() -> NSImage? {
    let iconNames = [
        "Icon-iOS-Default-256x256@1x",
        "Icon-iOS-Default-128x128@1x",
        "Icon-iOS-Default-512x512@1x",
        "AppIcon",
        "Icon"
    ]

    for iconName in iconNames {
        if let image = NSImage(named: iconName) {
            return image
        }

        if let imagePath = Bundle.main.path(forResource: iconName, ofType: "png"),
           let image = NSImage(contentsOfFile: imagePath) {
            return image
        }
    }

    return nil
}
