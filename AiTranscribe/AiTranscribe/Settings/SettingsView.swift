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
                case .sessions:   SessionsSettingsView(onNavigateToModels: {
                    withAnimation(.easeInOut(duration: 0.15)) {
                        selectedSection = .models
                    }
                })
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
        .frame(width: 1050, height: 720)
        .onAppear {
            // Only pause health checks if server is already ready
            // (avoid pausing during startup — we need to detect readiness)
            if backendManager.isServerReady {
                backendManager.pauseHealthCheck()
            }
        }
        .onDisappear {
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
