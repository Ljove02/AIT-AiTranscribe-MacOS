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
    
    @AppStorage("toggleRecordingShortcut") private var toggleShortcut = "⌥Space"
    @AppStorage("cancelRecordingShortcut") private var cancelShortcut = "⌥Escape"
    
    var body: some View {
        VStack(spacing: 24) {
            // Header
            VStack(spacing: 8) {
                Text("Keyboard Shortcuts")
                    .font(.system(size: 32, weight: .bold))
                
                Text("Global shortcuts are pre-configured and ready to use")
                    .font(.title3)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
            }
            .padding(.top, 30)
            
            // Shortcuts Display - CENTERED
            VStack(spacing: 20) {
                // Toggle Recording
                ShortcutDisplayRow(
                    icon: "play.circle.fill",
                    title: "Toggle Recording",
                    description: "Start or stop recording",
                    shortcut: toggleShortcut,
                    color: .green
                )
                
                Divider()
                    .padding(.horizontal, 20)
                
                // Cancel Recording
                ShortcutDisplayRow(
                    icon: "xmark.circle.fill",
                    title: "Cancel Recording",
                    description: "Discard current recording",
                    shortcut: cancelShortcut,
                    color: .red
                )
            }
            .padding(.horizontal, 24)
            .padding(.vertical, 20)
            .background(Color.gray.opacity(0.05))
            .cornerRadius(12)
            .frame(maxWidth: 600) // Constrain width for better centering
            .padding(.horizontal, 80) // Center it in the window
            
            // Info box
            HStack(spacing: 12) {
                Image(systemName: "info.circle.fill")
                    .foregroundColor(.blue)
                
                Text("You can customize these shortcuts later in Settings")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }
            .padding()
            .background(Color.blue.opacity(0.05))
            .cornerRadius(10)
            .padding(.horizontal, 40)
            
            Spacer()
            
            // Action Buttons
            VStack(spacing: 12) {
                Button(action: {
                    // Refresh hotkeys and continue
                    HotkeyManager.shared.refreshHotkeys()
                    onNext()
                }) {
                    HStack {
                        Text("Continue")
                            .font(.title3.weight(.semibold))
                        Image(systemName: "arrow.right")
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .padding(.horizontal, 60)
                
                // Back button - more visible
                Button(action: onBack) {
                    HStack {
                        Image(systemName: "chevron.left")
                        Text("Back")
                    }
                    .font(.body.weight(.medium))
                }
                .buttonStyle(.bordered)
                .controlSize(.large)
                .padding(.bottom, 20)
            }
        }
        .padding()
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

// MARK: - Preview

#Preview {
    ShortcutsOnboardingView(onNext: {}, onBack: {})
        .frame(width: 800, height: 750)
}
