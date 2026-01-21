/*
 ReadyOnboardingView.swift
 =========================

 Final onboarding screen - celebrate completion and explain next steps.
 
 Features:
 - Celebration message
 - Checklist of completed setup
 - Next steps guidance
 - Open AIT button (goes to Dashboard)
 */

import SwiftUI

struct ReadyOnboardingView: View {
    let onFinish: () -> Void
    
    @EnvironmentObject var appState: AppState
    @AppStorage("toggleRecordingShortcut") private var toggleShortcut = "⌥Space"
    
    var body: some View {
        VStack(spacing: 0) {
            // Add top spacing to match other screens
            Spacer()
                .frame(height: 40)
            
            // Content area
            VStack(spacing: 30) {
                // Celebration Icon
                ZStack {
                    Circle()
                        .fill(
                            LinearGradient(
                                colors: [.green.opacity(0.2), .blue.opacity(0.2)],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                        .frame(width: 140, height: 140)
                        .blur(radius: 20)
                    
                    Text("🎉")
                        .font(.system(size: 80))
                }
                
                // Title
                Text("You're Ready!")
                    .font(.system(size: 36, weight: .bold))
                
                // Subtitle
                Text("AIT is configured and ready to use")
                    .font(.title3)
                    .foregroundColor(.secondary)
                
                // Checklist
                VStack(alignment: .leading, spacing: 12) {
                    ChecklistItem(
                        icon: "checkmark.circle.fill",
                        text: "Microphone configured",
                        isComplete: true
                    )
                    
                    ChecklistItem(
                        icon: downloadedModelsCount > 0 ? "checkmark.circle.fill" : "circle",
                        text: "\(downloadedModelsCount) model\(downloadedModelsCount == 1 ? "" : "s") downloaded",
                        isComplete: downloadedModelsCount > 0
                    )
                    
                    ChecklistItem(
                        icon: "checkmark.circle.fill",
                        text: "Shortcuts configured",
                        isComplete: true
                    )
                }
                .padding(.horizontal, 80)
                .padding(.vertical, 20)
                .background(Color.gray.opacity(0.05))
                .cornerRadius(12)
                
                // What's Next
                VStack(alignment: .leading, spacing: 12) {
                    Text("What's next:")
                        .font(.headline)
                    
                    VStack(alignment: .leading, spacing: 8) {
                        NextStepRow(icon: "keyboard", text: "Press \(toggleShortcut) to start recording")
                        NextStepRow(icon: "menubar.rectangle", text: "Access AIT from the menu bar")
                        NextStepRow(icon: "gear", text: "Configure more settings in Preferences")
                    }
                }
                .padding(.horizontal, 60)
            }
            
            Spacer()
            
            // Open AIT Button - FIXED POSITION
            VStack {
                Button(action: onFinish) {
                    HStack {
                        Text("🚀 Open AIT Now")
                            .font(.title2.weight(.semibold))
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 16)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .padding(.horizontal, 60)
            }
            .frame(height: 80)
            .padding(.bottom, 30)
        }
        .padding()
    }
    
    // MARK: - Helpers
    
    private var downloadedModelsCount: Int {
        appState.availableModels.filter { $0.downloaded }.count
    }
}

// MARK: - Checklist Item

struct ChecklistItem: View {
    let icon: String
    let text: String
    let isComplete: Bool
    
    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .font(.title3)
                .foregroundColor(isComplete ? .green : .gray)
                .frame(width: 24)
            
            Text(text)
                .font(.body)
                .foregroundColor(isComplete ? .primary : .secondary)
        }
    }
}

// MARK: - Next Step Row

struct NextStepRow: View {
    let icon: String
    let text: String
    
    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .font(.body)
                .foregroundColor(.accentColor)
                .frame(width: 20)
            
            Text(text)
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
    }
}

// MARK: - Preview

#Preview {
    ReadyOnboardingView(onFinish: {})
        .environmentObject(AppState())
        .frame(width: 800, height: 750)
}
