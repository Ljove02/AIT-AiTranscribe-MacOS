/*
 ShortcutsOnboardingView.swift
 =============================

 Keyboard shortcuts configuration screen (OPTIONAL).
 
 Features:
 - Shows pre-configured global shortcuts
 - Simple, error-free display
 - Users can customize later in Settings
 */

import SwiftUI

struct ShortcutsOnboardingView: View {
    let onNext: () -> Void
    let onBack: () -> Void
    
    @AppStorage("toggleRecordingShortcut") private var toggleShortcut = "⌃P"
    @AppStorage("cancelRecordingShortcut") private var cancelShortcut = "⌃K"
    @State private var isRecordingToggleShortcut = false
    @State private var isRecordingCancelShortcut = false
    
    var body: some View {
        VStack(spacing: 22) {
            OnboardingHeader(
                eyebrow: "Shortcuts",
                title: "Pick fast global controls",
                subtitle: "Click a shortcut pill and press new keys. These work globally, even in the background."
            )
            
            VStack(spacing: 14) {
                EditableShortcutOnboardingRow(
                    icon: "play.circle.fill",
                    title: "Toggle Recording",
                    description: "Start or stop recording",
                    shortcut: $toggleShortcut,
                    isRecording: $isRecordingToggleShortcut,
                    color: .green
                )

                EditableShortcutOnboardingRow(
                    icon: "xmark.circle.fill",
                    title: "Cancel Recording",
                    description: "Discard current recording",
                    shortcut: $cancelShortcut,
                    isRecording: $isRecordingCancelShortcut,
                    color: .red
                )
            }
            
            HStack(alignment: .top, spacing: 9) {
                Image(systemName: "info.circle.fill")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(.blue)
                    .padding(.top, 1)

                Text("More recording, session, and Polish shortcuts are available later in Settings.")
                    .font(.system(size: 13, weight: .medium))
                    .lineSpacing(2)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)

                Spacer()
            }
            .padding(.top, 2)
            
            Spacer()
            
            OnboardingNavBar(
                onBack: onBack,
                onNext: {
                    HotkeyManager.shared.refreshHotkeys()
                    onNext()
                }
            )
        }
        .padding(34)
    }
}

// MARK: - Shortcut Display Row

fileprivate struct ShortcutDisplayRow: View {
    let icon: String
    let title: String
    let description: String
    let shortcut: String
    let color: Color
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Title with icon
            HStack(spacing: 12) {
                Image(systemName: icon)
                    .font(.title2)
                    .foregroundColor(color)
                
                VStack(alignment: .leading, spacing: 4) {
                    Text(title)
                        .font(.headline)
                    
                    Text(description)
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
            }
            
            // Shortcut display
            HStack {
                Text("Shortcut:")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                
                Text(shortcut)
                    .font(.system(.title3, design: .monospaced).weight(.semibold))
                    .padding(.horizontal, 16)
                    .padding(.vertical, 8)
                    .background(color.opacity(0.1))
                    .foregroundColor(color)
                    .cornerRadius(8)
                
                Spacer()
            }
        }
        .padding()
        .background(Color.white.opacity(0.001)) // Invisible background for better tap area
    }
}

private struct EditableShortcutOnboardingRow: View {
    let icon: String
    let title: String
    let description: String
    @Binding var shortcut: String
    @Binding var isRecording: Bool
    let color: Color

    var body: some View {
        OnboardingGlassCard {
            HStack(spacing: 16) {
                Image(systemName: icon)
                    .font(.system(size: 24, weight: .semibold))
                    .foregroundColor(color)
                    .frame(width: 48, height: 48)
                    .background(color.opacity(0.14), in: RoundedRectangle(cornerRadius: 15, style: .continuous))

                VStack(alignment: .leading, spacing: 4) {
                    Text(title)
                        .font(.system(size: 16, weight: .bold, design: .rounded))

                    Text(description)
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(.secondary)
                }

                Spacer()

                ShortcutPill(
                    shortcut: $shortcut,
                    isRecording: $isRecording,
                    onChanged: {
                        HotkeyManager.shared.refreshHotkeys()
                    }
                )
            }
        }
    }
}

// MARK: - Preview

#Preview {
    ShortcutsOnboardingView(onNext: {}, onBack: {})
        .frame(width: 800, height: 750)
}
