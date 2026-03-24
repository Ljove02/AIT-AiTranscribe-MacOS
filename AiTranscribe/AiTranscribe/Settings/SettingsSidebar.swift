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

    var body: some View {
        VStack(spacing: 0) {
            // App icon at top
            Group {
                if let nsImage = loadAppIconFromBundle() {
                    Image(nsImage: nsImage)
                        .resizable()
                        .scaledToFit()
                        .frame(width: 48, height: 48)
                } else {
                    Image(systemName: "mic.circle.fill")
                        .resizable()
                        .scaledToFit()
                        .frame(width: 48, height: 48)
                        .foregroundStyle(.blue.gradient)
                }
            }
            .padding(.top, 20)
            .padding(.bottom, 4)

            Text("AI-Transcribe")
                .font(.system(size: 13, weight: .semibold))
                .foregroundColor(.primary)
                .padding(.bottom, 16)

            // Navigation items
            VStack(spacing: 2) {
                ForEach(SettingsSection.allCases) { section in
                    SidebarItem(
                        section: section,
                        isSelected: selectedSection == section
                    )
                    .onTapGesture {
                        withAnimation(.easeInOut(duration: 0.15)) {
                            selectedSection = section
                        }
                    }
                }
            }
            .padding(.horizontal, 12)

            Spacer()

            // Version at bottom
            Text("Version \(Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "0.1.5")")
                .font(.caption2)
                .foregroundColor(.secondary.opacity(0.6))
                .padding(.bottom, 16)
        }
        .frame(width: 200)
        .background(Color(NSColor.controlBackgroundColor))
    }
}

// MARK: - Sidebar Item

struct SidebarItem: View {
    let section: SettingsSection
    let isSelected: Bool

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: section.icon)
                .font(.system(size: 14))
                .foregroundColor(isSelected ? .accentColor : .secondary)
                .frame(width: 20)

            Text(section.label)
                .font(.system(size: 13, weight: isSelected ? .semibold : .regular))
                .foregroundColor(isSelected ? .accentColor : .primary)

            Spacer()
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background(
            RoundedRectangle(cornerRadius: 6)
                .fill(isSelected ? Color.accentColor.opacity(0.12) : Color.clear)
        )
        .contentShape(Rectangle())
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
