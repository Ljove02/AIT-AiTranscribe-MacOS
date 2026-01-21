/*
 MicrophoneOnboardingView.swift
 ==============================

 Microphone permission screen (MANDATORY).
 
 Features:
 - Check microphone permission status
 - Request permission if not determined
 - Guide user to System Settings if denied
 - Cannot proceed without authorization
 */

import SwiftUI
import AVFoundation

struct MicrophoneOnboardingView: View {
    let onNext: () -> Void
    let onBack: () -> Void
    
    @State private var permissionState: PermissionState = .notDetermined
    @State private var isCheckingPermission = false
    
    var body: some View {
        VStack(spacing: 30) {
            Spacer()
            
            // Icon
            ZStack {
                Circle()
                    .fill(iconBackgroundColor.opacity(0.2))
                    .frame(width: 120, height: 120)
                
                Image(systemName: "mic.fill")
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(width: 60, height: 60)
                    .foregroundColor(iconColor)
            }
            
            // Title
            Text("Set Up Microphone")
                .font(.system(size: 32, weight: .bold))
            
            // Description
            Text("AIT needs microphone access to record audio for transcription.")
                .font(.title3)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)
            
            // Permission Status
            PermissionStatusView(state: permissionState)
                .padding(.top, 20)
            
            Spacer()
            
            // Action Buttons
            VStack(spacing: 12) {
                if permissionState == .authorized {
                    // Can proceed
                    Button(action: onNext) {
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
                    
                } else if permissionState == .denied {
                    // Guide to settings
                    Button(action: openSystemSettings) {
                        HStack {
                            Image(systemName: "gear")
                            Text("Open System Settings")
                                .font(.title3.weight(.semibold))
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 14)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .padding(.horizontal, 60)
                    
                } else {
                    // Request permission
                    Button(action: requestPermission) {
                        HStack {
                            if isCheckingPermission {
                                ProgressView()
                                    .controlSize(.small)
                                    .padding(.trailing, 4)
                            }
                            Text(isCheckingPermission ? "Checking..." : "Request Permission")
                                .font(.title3.weight(.semibold))
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 14)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .disabled(isCheckingPermission)
                    .padding(.horizontal, 60)
                }
                
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
        .onAppear {
            checkPermission()
        }
    }
    
    // MARK: - Permission Handling
    
    private func checkPermission() {
        let status = AVCaptureDevice.authorizationStatus(for: .audio)
        
        switch status {
        case .authorized:
            permissionState = .authorized
        case .denied, .restricted:
            permissionState = .denied
        case .notDetermined:
            permissionState = .notDetermined
        @unknown default:
            permissionState = .notDetermined
        }
    }
    
    private func requestPermission() {
        isCheckingPermission = true
        
        AVCaptureDevice.requestAccess(for: .audio) { granted in
            DispatchQueue.main.async {
                isCheckingPermission = false
                permissionState = granted ? .authorized : .denied
            }
        }
    }
    
    private func openSystemSettings() {
        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone") {
            NSWorkspace.shared.open(url)
        }
    }
    
    private var iconColor: Color {
        switch permissionState {
        case .authorized:
            return .green
        case .denied:
            return .red
        case .notDetermined:
            return .accentColor
        }
    }
    
    private var iconBackgroundColor: Color {
        switch permissionState {
        case .authorized:
            return .green
        case .denied:
            return .red
        case .notDetermined:
            return .accentColor
        }
    }
}

// MARK: - Permission State (Microphone)

fileprivate enum PermissionState {
    case notDetermined
    case authorized
    case denied
}

// MARK: - Permission Status View (Microphone)

fileprivate struct PermissionStatusView: View {
    let state: PermissionState
    
    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: iconName)
                .font(.title2)
                .foregroundColor(iconColor)
            
            Text(message)
                .font(.headline)
                .foregroundColor(textColor)
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 12)
        .background(backgroundColor.opacity(0.1))
        .cornerRadius(12)
    }
    
    private var iconName: String {
        switch state {
        case .authorized:
            return "checkmark.circle.fill"
        case .denied:
            return "xmark.circle.fill"
        case .notDetermined:
            return "questionmark.circle.fill"
        }
    }
    
    private var iconColor: Color {
        switch state {
        case .authorized:
            return .green
        case .denied:
            return .red
        case .notDetermined:
            return .orange
        }
    }
    
    private var message: String {
        switch state {
        case .authorized:
            return "Microphone access granted"
        case .denied:
            return "Microphone access denied"
        case .notDetermined:
            return "Permission not yet requested"
        }
    }
    
    private var textColor: Color {
        switch state {
        case .authorized:
            return .green
        case .denied:
            return .red
        case .notDetermined:
            return .orange
        }
    }
    
    private var backgroundColor: Color {
        switch state {
        case .authorized:
            return .green
        case .denied:
            return .red
        case .notDetermined:
            return .orange
        }
    }
}

// MARK: - Preview

#Preview {
    MicrophoneOnboardingView(onNext: {}, onBack: {})
        .frame(width: 800, height: 750)
}
