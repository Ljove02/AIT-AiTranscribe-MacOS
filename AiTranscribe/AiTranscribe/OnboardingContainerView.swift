/*
 OnboardingContainerView.swift
 ==============================

 Container view that manages the onboarding flow.
 
 Features:
 - Progress bar at top (full width)
 - Screen navigation with Next/Back buttons
 - Tracks completion state in UserDefaults
 */

import SwiftUI
import AppKit

struct OnboardingContainerView: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var backendManager: BackendManager

    /// Environment action to open windows
    @Environment(\.openWindow) private var openWindow

    let previewMode: Bool
    
    /// Current screen index in the current route list.
    @State private var currentScreen: Int = 0
    @State private var setupPreference: OnboardingSetupPreference?
    
    /// Whether onboarding is complete
    @AppStorage("hasCompletedOnboarding") private var hasCompletedOnboarding = false

    init(previewMode: Bool = false) {
        self.previewMode = previewMode
    }
    
    var body: some View {
        ZStack {
            OnboardingTransparentWindow()
                .frame(width: 0, height: 0)

            VStack(spacing: 0) {
                topRail
                    .padding(.horizontal, 28)
                    .padding(.top, 22)
                    .padding(.bottom, 14)

                Rectangle()
                    .fill(.white.opacity(0.07))
                    .frame(height: 1)

                ZStack {
                    switch currentRoute {
                    case .welcome:
                        WelcomeOnboardingView(onNext: nextScreen)
                            .transition(.opacity.combined(with: .scale(scale: 0.985)))
                    case .microphone:
                        MicrophoneOnboardingView(onNext: nextScreen, onBack: previousScreen)
                            .transition(.opacity.combined(with: .scale(scale: 0.985)))
                    case .setupChoice:
                        SetupChoiceOnboardingView(
                            selection: setupPreference,
                            onSelect: { preference in
                                setupPreference = preference
                                nextScreen()
                            },
                            onBack: previousScreen
                        )
                        .transition(.opacity.combined(with: .scale(scale: 0.985)))
                    case .localModels:
                        ModelsOnboardingView(onNext: nextScreen, onBack: previousScreen)
                            .transition(.opacity.combined(with: .scale(scale: 0.985)))
                    case .apiModels:
                        APISetupOnboardingView(onNext: nextScreen, onBack: previousScreen)
                            .transition(.opacity.combined(with: .scale(scale: 0.985)))
                    case .shortcuts:
                        ShortcutsOnboardingView(onNext: nextScreen, onBack: previousScreen)
                            .transition(.opacity.combined(with: .scale(scale: 0.985)))
                    case .test:
                        TestTranscriptionOnboardingView(onNext: nextScreen, onBack: previousScreen)
                            .transition(.opacity.combined(with: .scale(scale: 0.985)))
                    case .polishPreview:
                        PolishPreviewOnboardingView(onNext: nextScreen, onBack: previousScreen)
                            .transition(.opacity.combined(with: .scale(scale: 0.985)))
                    case .ready:
                        ReadyOnboardingView(onFinish: finishOnboarding)
                            .transition(.opacity.combined(with: .scale(scale: 0.985)))
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            .background {
                ZStack {
                    OnboardingBackground()
                    Rectangle()
                        .fill(.ultraThinMaterial)
                }
            }
            .clipShape(.rect(cornerRadius: 18, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .strokeBorder(.white.opacity(0.10), lineWidth: 0.7)
            )
            .shadow(color: .black.opacity(0.32), radius: 24, x: 0, y: 12)
        }
        .ignoresSafeArea()
        .frame(width: 900, height: 720)
    }

    private var topRail: some View {
        OnboardingStepRail(
            routes: routes,
            currentIndex: currentScreen,
            closeAction: closeOnboardingWindow
        )
    }
    
    /// Calculate progress (0.0 to 1.0)
    private var progress: Double {
        Double(currentScreen + 1) / Double(routes.count)
    }

    private var currentRoute: OnboardingRoute {
        routes[min(currentScreen, routes.count - 1)]
    }

    private var routes: [OnboardingRoute] {
        var result: [OnboardingRoute] = [.welcome, .microphone, .setupChoice]

        if setupPreference == .local || setupPreference == .both {
            result.append(.localModels)
        }

        if setupPreference == .api || setupPreference == .both {
            result.append(.apiModels)
        }

        result.append(contentsOf: [.shortcuts, .test, .polishPreview, .ready])
        return result
    }
    
    /// Navigate to next screen
    private func nextScreen() {
        withAnimation(.easeInOut(duration: 0.3)) {
            if currentScreen < routes.count - 1 {
                currentScreen += 1
            }
        }
    }
    
    /// Navigate to previous screen
    private func previousScreen() {
        withAnimation(.easeInOut(duration: 0.3)) {
            if currentScreen > 0 {
                currentScreen -= 1
            }
        }
    }
    
    /// Complete onboarding and close
    private func finishOnboarding() {
        if previewMode {
            closeOnboardingWindow()
            return
        }

        hasCompletedOnboarding = true

        // Auto-load preferred model
        Task {
            await appState.loadPreferredModel()
        }

        // Close onboarding window
        if let window = NSApp.windows.first(where: { $0.identifier?.rawValue == "onboarding" }) {
            window.close()
        }

        // Open Settings window
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
            openWindow(id: "settings")
            SettingsWindowFocus.focus()
        }
    }

    private func closeOnboardingWindow() {
        if let window = NSApp.windows.first(where: { $0.identifier?.rawValue == "onboarding-preview" || $0.identifier?.rawValue == "onboarding" }) {
            window.close()
        }
    }
}

enum OnboardingRoute {
    case welcome
    case microphone
    case setupChoice
    case localModels
    case apiModels
    case shortcuts
    case test
    case polishPreview
    case ready

    var id: String {
        switch self {
        case .welcome: return "welcome"
        case .microphone: return "microphone"
        case .setupChoice: return "setup-choice"
        case .localModels: return "local-models"
        case .apiModels: return "api-models"
        case .shortcuts: return "shortcuts"
        case .test: return "test"
        case .polishPreview: return "polish-preview"
        case .ready: return "ready"
        }
    }

    var stepTitle: String {
        switch self {
        case .welcome: return "Start"
        case .microphone: return "Mic"
        case .setupChoice: return "Mode"
        case .localModels: return "Local"
        case .apiModels: return "API"
        case .shortcuts: return "Keys"
        case .test: return "Test"
        case .polishPreview: return "Next"
        case .ready: return "Ready"
        }
    }
}

enum OnboardingSetupPreference: String, CaseIterable, Identifiable {
    case local
    case api
    case both

    var id: String { rawValue }

    var title: String {
        switch self {
        case .local: return "Local"
        case .api: return "API"
        case .both: return "Both"
        }
    }

    var subtitle: String {
        switch self {
        case .local: return "Private by default"
        case .api: return "Fast public providers"
        case .both: return "Privacy and speed"
        }
    }

    var icon: String {
        switch self {
        case .local: return "lock.shield"
        case .api: return "globe"
        case .both: return "arrow.triangle.2.circlepath"
        }
    }
}

// MARK: - Progress Bar

struct ProgressBar: View {
    let progress: Double
    
    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                // Background track
                Rectangle()
                    .fill(Color.gray.opacity(0.2))
                
                // Progress fill
                Rectangle()
                    .fill(Color.accentColor)
                    .frame(width: geometry.size.width * progress)
            }
        }
        .frame(height: 4)
        .animation(.easeInOut(duration: 0.3), value: progress)
    }
}

// MARK: - Shared Onboarding UI

struct OnboardingTransparentWindow: NSViewRepresentable {
    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        DispatchQueue.main.async {
            guard let window = view.window else { return }
            window.isOpaque = false
            window.backgroundColor = .clear
            window.titleVisibility = .hidden
            window.titlebarAppearsTransparent = true
            window.titlebarSeparatorStyle = .none
            window.toolbar = nil
            window.isMovableByWindowBackground = true
            window.standardWindowButton(.closeButton)?.isHidden = true
            window.standardWindowButton(.miniaturizeButton)?.isHidden = true
            window.standardWindowButton(.zoomButton)?.isHidden = true
        }
        return view
    }

    func updateNSView(_ nsView: NSView, context: Context) {}
}

struct OnboardingBackground: View {
    var body: some View {
        ZStack {
            LinearGradient(
                colors: [
                    Color(red: 0.05, green: 0.08, blue: 0.13),
                    Color(red: 0.09, green: 0.17, blue: 0.27),
                    Color(red: 0.04, green: 0.05, blue: 0.08)
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )

            Circle()
                .fill(.blue.opacity(0.24))
                .frame(width: 360, height: 360)
                .blur(radius: 90)
                .offset(x: -310, y: -230)

            Circle()
                .fill(.teal.opacity(0.18))
                .frame(width: 320, height: 320)
                .blur(radius: 100)
                .offset(x: 330, y: 240)
        }
        .ignoresSafeArea()
    }
}

struct OnboardingStepRail: View {
    let routes: [OnboardingRoute]
    let currentIndex: Int
    let closeAction: () -> Void

    var body: some View {
        HStack(spacing: 0) {
            OnboardingPreviewCloseButton(action: closeAction)

            connector(isComplete: false)

            ForEach(Array(routes.enumerated()), id: \.element.id) { index, route in
                step(route: route, index: index)

                if index < routes.count - 1 {
                    connector(isComplete: index < currentIndex)
                }
            }
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Onboarding step \(currentIndex + 1) of \(routes.count)")
    }

    private func step(route: OnboardingRoute, index: Int) -> some View {
        let isComplete = index < currentIndex
        let isCurrent = index == currentIndex

        return VStack(spacing: 7) {
            ZStack {
                Circle()
                    .fill(isCurrent ? Color.accentColor : (isComplete ? Color.accentColor.opacity(0.24) : Color.white.opacity(0.08)))
                    .frame(width: isCurrent ? 28 : 22, height: isCurrent ? 28 : 22)
                    .overlay(
                        Circle()
                            .strokeBorder(isCurrent ? Color.white.opacity(0.28) : Color.white.opacity(0.10), lineWidth: 1)
                    )

                if isComplete {
                    Image(systemName: "checkmark")
                        .font(.system(size: 10, weight: .bold))
                        .foregroundStyle(.white)
                } else {
                    Circle()
                        .fill(isCurrent ? Color.white : Color.white.opacity(0.45))
                        .frame(width: isCurrent ? 7 : 5, height: isCurrent ? 7 : 5)
                }
            }

            Text(route.stepTitle)
                .font(.system(size: 10, weight: isCurrent ? .bold : .semibold, design: .rounded))
                .foregroundStyle(isCurrent ? .primary : .secondary)
                .lineLimit(1)
        }
        .frame(width: 54)
        .animation(.spring(duration: 0.35, bounce: 0.08), value: currentIndex)
    }

    private func connector(isComplete: Bool) -> some View {
        Capsule()
            .fill(isComplete ? Color.accentColor.opacity(0.7) : Color.white.opacity(0.12))
            .frame(height: 2)
            .frame(maxWidth: .infinity)
            .padding(.horizontal, 8)
            .padding(.bottom, 17)
            .animation(.spring(duration: 0.35, bounce: 0.08), value: currentIndex)
    }
}

struct OnboardingPreviewCloseButton: View {
    let action: () -> Void
    @State private var isHovering = false

    var body: some View {
        Button(action: action) {
            VStack(spacing: 7) {
                ZStack {
                    Circle()
                        .fill(isHovering ? Color.red.opacity(0.48) : Color.red.opacity(0.28))
                        .frame(width: 22, height: 22)
                        .overlay(
                            Circle()
                                .strokeBorder(.white.opacity(isHovering ? 0.24 : 0.13), lineWidth: 1)
                        )

                    Image(systemName: "xmark")
                        .font(.system(size: 8, weight: .bold))
                        .foregroundStyle(.white.opacity(0.92))
                }

                Text("Close")
                    .font(.system(size: 10, weight: .semibold, design: .rounded))
                    .foregroundStyle(isHovering ? .primary : .secondary)
                    .lineLimit(1)
            }
            .frame(width: 54)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { hovering in
            withAnimation(.spring(duration: 0.24, bounce: 0.10)) {
                isHovering = hovering
            }
        }
        .accessibilityLabel("Close introduction preview")
    }
}

struct OnboardingHeader: View {
    let eyebrow: String
    let title: String
    let subtitle: String

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(eyebrow.uppercased())
                .font(.system(size: 11, weight: .bold, design: .rounded))
                .tracking(1.4)
                .foregroundStyle(.secondary)

            Text(title)
                .font(.system(size: 34, weight: .bold, design: .rounded))
                .foregroundStyle(.primary)

            Text(subtitle)
                .font(.system(size: 15, weight: .medium))
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

struct OnboardingNavBar: View {
    let backTitle: String
    let nextTitle: String
    let nextIcon: String
    let nextDisabled: Bool
    let onBack: () -> Void
    let onNext: () -> Void

    init(
        backTitle: String = "Back",
        nextTitle: String = "Continue",
        nextIcon: String = "arrow.right",
        nextDisabled: Bool = false,
        onBack: @escaping () -> Void,
        onNext: @escaping () -> Void
    ) {
        self.backTitle = backTitle
        self.nextTitle = nextTitle
        self.nextIcon = nextIcon
        self.nextDisabled = nextDisabled
        self.onBack = onBack
        self.onNext = onNext
    }

    var body: some View {
        HStack {
            OnboardingSecondaryActionButton(
                title: backTitle,
                systemImage: "arrow.left",
                action: onBack
            )

            Spacer()

            OnboardingPrimaryActionButton(
                title: nextTitle,
                systemImage: nextIcon,
                isDisabled: nextDisabled,
                action: onNext
            )
        }
    }
}

struct OnboardingPrimaryActionButton: View {
    let title: String
    let systemImage: String
    let isDisabled: Bool
    let isLoading: Bool
    let minWidth: CGFloat
    let action: () -> Void

    init(
        title: String,
        systemImage: String,
        isDisabled: Bool = false,
        isLoading: Bool = false,
        minWidth: CGFloat = 190,
        action: @escaping () -> Void
    ) {
        self.title = title
        self.systemImage = systemImage
        self.isDisabled = isDisabled
        self.isLoading = isLoading
        self.minWidth = minWidth
        self.action = action
    }

    var body: some View {
        Button(action: action) {
            HStack(spacing: 8) {
                if isLoading {
                    ProgressView()
                        .controlSize(.small)
                        .scaleEffect(0.82)
                }

                Text(title)

                if !isLoading {
                    Image(systemName: systemImage)
                }
            }
            .font(.system(size: 14, weight: .semibold, design: .rounded))
            .frame(minWidth: minWidth)
        }
        .buttonStyle(.borderedProminent)
        .controlSize(.large)
        .disabled(isDisabled || isLoading)
    }
}

struct OnboardingSecondaryActionButton: View {
    let title: String
    let systemImage: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Label(title, systemImage: systemImage)
                .font(.system(size: 14, weight: .semibold, design: .rounded))
                .frame(minWidth: 128)
        }
        .buttonStyle(.bordered)
        .controlSize(.large)
    }
}

struct OnboardingGlassCard<Content: View>: View {
    let content: Content

    init(@ViewBuilder content: () -> Content) {
        self.content = content()
    }

    var body: some View {
        content
            .padding(18)
            .background(.white.opacity(0.055), in: RoundedRectangle(cornerRadius: 20, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 20, style: .continuous)
                    .strokeBorder(.white.opacity(0.10), lineWidth: 1)
            )
    }
}

// MARK: - Preview

#Preview {
    OnboardingContainerView(previewMode: true)
        .environmentObject(AppState())
        .environmentObject(BackendManager())
}
