/*
 SettingsView.swift
 ==================

 The Settings window for configuring the app.
 Redesigned with a left sidebar navigation and larger window.

 This opens when:
 - User clicks "Settings..." in the menu
 - User presses Cmd+, (standard macOS shortcut)
 */

import SwiftUI

struct SettingsView: View {
    @State private var selectedSection: SettingsSection = .dashboard
    @EnvironmentObject var backendManager: BackendManager
    var body: some View {
        HStack(spacing: 0) {
            SettingsSidebar(selectedSection: $selectedSection)

            Divider()

            // Content area
            Group {
                switch selectedSection {
                case .dashboard:  DashboardView()
                case .general:    GeneralSettingsView()
                case .models:     ModelsSettingsView()
                case .history:    HistorySettingsView()
                case .shortcuts:  ShortcutsSettingsView()
                case .about:      AboutView()
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .transition(.opacity)
            .id(selectedSection)
            .animation(.easeInOut(duration: 0.15), value: selectedSection)
        }
        .frame(width: 900, height: 620)
        .onAppear {
            // Pause health check timer to prevent UI re-renders
            backendManager.pauseHealthCheck()
        }
        .onDisappear {
            // Resume health check timer when Settings window closes
            backendManager.resumeHealthCheck()
        }
    }
}

// MARK: - Preview

#Preview {
    SettingsView()
        .environmentObject(AppState())
        .environmentObject(BackendManager())
}
