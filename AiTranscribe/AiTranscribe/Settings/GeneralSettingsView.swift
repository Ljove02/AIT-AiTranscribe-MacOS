import SwiftUI
import AVFoundation

// MARK: - General Settings View

struct GeneralSettingsView: View {
    @EnvironmentObject var appState: AppState
    let hasAnimated: Bool
    let onAnimated: () -> Void

    @AppStorage("autoPasteAfterTranscription") private var autoPasteAfterTranscription = false
    @AppStorage("playSounds") private var playSounds = true
    @AppStorage("indicatorPosition") private var indicatorPosition = "topCenter"
    @AppStorage("recordingIndicatorStyle") private var recordingIndicatorStyle = "classic"
    @AppStorage("recordingIndicatorNoiseFloorRMS") private var recordingIndicatorNoiseFloorRMS = 0.0
    @AppStorage("basicTranscriptionBatchingEnabled") private var basicBatchingEnabled = true
    @AppStorage("basicTranscriptionBatchingThresholdSeconds") private var basicBatchingThresholdSeconds = 120.0
    @AppStorage("basicTranscriptionBatchingRamBudgetMB") private var basicBatchingRamBudgetMB = 4096

    // Init from hasAnimated so the very first frame is correct — no jump
    @State private var appeared: Bool
    @State private var isCalibratingIndicatorNoise = false
    @State private var noiseCalibrationMessage: String?

    /// Cached TCC status — reading AVCaptureDevice.authorizationStatus is an
    /// IPC-backed call and the body consults it ~8 times per render. Refreshed
    /// on appear and after the request-permission action.
    @State private var microphoneStatus = AVCaptureDevice.authorizationStatus(for: .audio)

    init(hasAnimated: Bool, onAnimated: @escaping () -> Void) {
        self.hasAnimated = hasAnimated
        self.onAnimated = onAnimated
        _appeared = State(initialValue: hasAnimated)
    }

    var body: some View {
        ScrollView {
            VStack(spacing: 0) {
                // ── Title Header ─────────────────────────────────
                HStack(alignment: .firstTextBaseline) {
                    VStack(alignment: .leading, spacing: 3) {
                        Text("General")
                            .font(.system(size: 22, weight: .bold, design: .rounded))

                        Text("10 options")
                            .font(.system(size: 11))
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                }
                .staggerIn(index: 0, appeared: appeared)

                // Transcription section
                Group {
                    SettingsSectionHeader(title: "Transcription")

                    SettingsRow(
                        title: "Auto-paste after transcription",
                        description: "Transcribed text is copied to clipboard. When enabled, it also pastes automatically at cursor position."
                    ) {
                        Toggle("", isOn: $autoPasteAfterTranscription)
                            .toggleStyle(.switch)
                            .labelsHidden()
                            .onChange(of: autoPasteAfterTranscription) { _, newValue in
                                if newValue {
                                    appState.checkAccessibilityPermissionsIfNeeded()
                                }
                            }
                    }
                }
                .staggerIn(index: 1, appeared: appeared)

                SettingsDivider()
                    .staggerIn(index: 2, appeared: appeared)

                // Batch Processing section
                Group {
                    SettingsSectionHeader(title: "Batch Processing")

                    SettingsRow(
                        title: "Rolling batch processing",
                        description: "Transcribe completed chunks while you keep speaking. This keeps long recordings fast when you stop."
                    ) {
                        Toggle("", isOn: $basicBatchingEnabled)
                            .toggleStyle(.switch)
                            .labelsHidden()
                    }

                    SettingsRow(
                        title: "Chunk length",
                        description: "Shorter chunks finish sooner and use less memory. 30 seconds is experimental; two minutes is the validated stable default."
                    ) {
                        batchLengthMenu
                    }

                    SettingsRow(
                        title: "Local memory budget",
                        description: "Used by fallback post-stop batching. Rolling batches usually stay below this because chunks are small."
                    ) {
                        ramBudgetMenu
                    }

                    batchInfoPanel
                }
                .staggerIn(index: 3, appeared: appeared)

                SettingsDivider()
                    .staggerIn(index: 4, appeared: appeared)

                // Recording Indicator section
                Group {
                    SettingsSectionHeader(title: "Recording Indicator")

                    SettingsRow(
                        title: "Position",
                        description: "Where the recording indicator appears on screen."
                    ) {
                        indicatorPositionMenu
                    }

                    indicatorStyleSelector

                    indicatorNoiseCalibrationPanel
                }
                .staggerIn(index: 5, appeared: appeared)

                SettingsDivider()
                    .staggerIn(index: 6, appeared: appeared)

                // Feedback section
                Group {
                    SettingsSectionHeader(title: "Feedback")

                    SettingsRow(
                        title: "Play sounds",
                        description: "Play audio feedback when recording starts and stops."
                    ) {
                        Toggle("", isOn: $playSounds)
                            .toggleStyle(.switch)
                            .labelsHidden()
                    }
                }
                .staggerIn(index: 7, appeared: appeared)

                SettingsDivider()
                    .staggerIn(index: 8, appeared: appeared)

                // Audio section
                Group {
                    SettingsSectionHeader(title: "Audio")

                    AudioDuckingSettings()
                }
                .staggerIn(index: 9, appeared: appeared)

                SettingsDivider()
                    .staggerIn(index: 10, appeared: appeared)

                // Permissions section
                Group {
                    SettingsSectionHeader(title: "Permissions")

                    SettingsRow(
                        title: "Microphone Access",
                        description: microphoneDescription
                    ) {
                        microphoneStatusBadge
                    }

                    if microphoneStatus != .authorized {
                        SettingsRow(
                            title: "Request Permission",
                            description: "If the app doesn't appear in Microphone settings, click this first."
                        ) {
                            Button("Request") {
                                requestMicrophonePermission()
                            }
                            .buttonStyle(.plain)
                            .labelStyle(.titleOnly)
                            .modifier(SettingsActionPillStyle(tint: .accentColor, prominent: true))
                        }
                    }

                    SettingsRow(
                        title: "System Settings",
                        description: "Open macOS Privacy & Security to manage microphone access."
                    ) {
                        Button("Open") {
                            openMicrophoneSettings()
                        }
                        .buttonStyle(.plain)
                        .modifier(SettingsActionPillStyle(tint: .secondary, prominent: false))
                    }
                }
                .staggerIn(index: 11, appeared: appeared)

                // Bottom spacing
                Spacer(minLength: 20)
            }
            .padding(.horizontal, 28)
            .padding(.top, 8)
        }
        .scrollIndicators(.automatic)
        .onAppear {
            microphoneStatus = AVCaptureDevice.authorizationStatus(for: .audio)
        }
        .task(id: "stagger") {
            guard !hasAnimated else { return }
            try? await Task.sleep(for: .milliseconds(80))
            appeared = true
            onAnimated()
        }
    }

    // MARK: - Microphone Permission

    private var microphoneDescription: String {
        switch microphoneStatus {
        case .authorized:
            return "Microphone access is granted."
        case .denied, .restricted:
            return "Microphone access was denied. Enable it in System Settings."
        case .notDetermined:
            return "Microphone permission has not been requested yet."
        @unknown default:
            return "Unknown status."
        }
    }

    private var microphoneStatusBadge: some View {
        HStack(spacing: 5) {
            Circle()
                .fill(microphoneStatus == .authorized ? .green : (microphoneStatus == .notDetermined ? .orange : .red))
                .frame(width: 6, height: 6)

            Text(microphoneStatus == .authorized ? "Granted" : (microphoneStatus == .notDetermined ? "Not Requested" : "Denied"))
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(microphoneStatus == .authorized ? .green : (microphoneStatus == .notDetermined ? .orange : .red))
        }
        .padding(.horizontal, 10)
        .frame(height: 28)
        .background(.thinMaterial, in: Capsule())
        .overlay(Capsule().strokeBorder(.white.opacity(0.06), lineWidth: 0.5))
    }

    private func requestMicrophonePermission() {
        AVCaptureDevice.requestAccess(for: .audio) { granted in
            DispatchQueue.main.async {
                microphoneStatus = AVCaptureDevice.authorizationStatus(for: .audio)
                if granted {
                    appState.statusMessage = "Microphone permission granted"
                } else {
                    appState.statusMessage = "Microphone permission denied - enable in System Settings"
                }
            }
        }
    }

    private func openMicrophoneSettings() {
        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone") {
            NSWorkspace.shared.open(url)
        }
    }

    // MARK: - Pill Controls

    private var batchLengthMenu: some View {
        Menu {
            ForEach(batchLengthPresets) { preset in
                Button(preset.title) {
                    basicBatchingThresholdSeconds = preset.seconds
                }
            }
        } label: {
            settingsSelectPill(
                title: batchLengthPresets.first(where: { $0.seconds == basicBatchingThresholdSeconds })?.title ?? formattedBatchLength,
                icon: "timer",
                width: 224
            )
        }
        .menuStyle(.borderlessButton)
        .disabled(!basicBatchingEnabled)
        .opacity(basicBatchingEnabled ? 1 : 0.55)
    }

    private var ramBudgetMenu: some View {
        Menu {
            ForEach(ramBudgetPresets) { preset in
                Button(preset.title) {
                    basicBatchingRamBudgetMB = preset.megabytes
                }
            }
        } label: {
            settingsSelectPill(
                title: ramBudgetPresets.first(where: { $0.megabytes == basicBatchingRamBudgetMB })?.title ?? formattedRAMBudget,
                icon: "memorychip",
                width: 132
            )
        }
        .menuStyle(.borderlessButton)
        .disabled(!basicBatchingEnabled)
        .opacity(basicBatchingEnabled ? 1 : 0.55)
    }

    private var indicatorPositionMenu: some View {
        Menu {
            ForEach(RecordingIndicatorController.ScreenPosition.allCases, id: \.rawValue) { pos in
                Button(pos.displayName) {
                    indicatorPosition = pos.rawValue
                    appState.recordingIndicator.setPosition(pos)
                }
            }
        } label: {
            settingsSelectPill(
                title: RecordingIndicatorController.ScreenPosition(rawValue: indicatorPosition)?.displayName ?? "Top Center",
                icon: "rectangle.inset.filled",
                width: 160
            )
        }
        .menuStyle(.borderlessButton)
    }

    private func settingsSelectPill(title: String, icon: String, width: CGFloat) -> some View {
        HStack(spacing: 7) {
            Image(systemName: icon)
                .font(.system(size: 9, weight: .semibold))
                .foregroundStyle(.secondary)

            Text(title)
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(.primary)
                .lineLimit(1)
                .truncationMode(.tail)

            Spacer(minLength: 8)

            Image(systemName: "chevron.down")
                .font(.system(size: 8, weight: .bold))
                .foregroundStyle(.tertiary)
        }
        .padding(.horizontal, 11)
        .frame(width: width, height: 30)
        .background(Color.primary.opacity(0.075), in: Capsule())
        .overlay(Capsule().strokeBorder(.white.opacity(0.06), lineWidth: 0.5))
        .contentShape(Capsule())
    }

    // MARK: - Recording Indicator

    private var selectedIndicatorStyle: RecordingIndicatorStyle {
        RecordingIndicatorStyle(rawValue: recordingIndicatorStyle) ?? .classic
    }

    private var indicatorStyleColumns: [GridItem] {
        [
            GridItem(.flexible(), spacing: 10),
            GridItem(.flexible(), spacing: 10)
        ]
    }

    private var indicatorStyleSelector: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Style")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(.primary)

                    Text("Choose the floating indicator shown while recording.")
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                }

                Spacer()
            }

            LazyVGrid(columns: indicatorStyleColumns, alignment: .leading, spacing: 10) {
                ForEach(RecordingIndicatorStyle.allCases) { style in
                    RecordingIndicatorStyleOption(
                        style: style,
                        isSelected: selectedIndicatorStyle == style
                    ) {
                        recordingIndicatorStyle = style.rawValue
                        appState.recordingIndicator.setStyle(style)
                    }
                }
            }
        }
        .padding(12)
        .background(.thinMaterial, in: .rect(cornerRadius: 10, style: .continuous))
        .padding(.vertical, 2)
    }

    private var indicatorNoiseCalibrationPanel: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Noise sensitivity")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(.primary)

                    Text(noiseCalibrationDescription)
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }

                Spacer(minLength: 14)

                HStack(spacing: 8) {
                    if recordingIndicatorNoiseFloorRMS > 0 {
                        Button("Reset") {
                            appState.recordingIndicator.resetNoiseCalibration()
                            recordingIndicatorNoiseFloorRMS = 0
                            noiseCalibrationMessage = "Automatic noise floor restored."
                        }
                        .buttonStyle(.plain)
                        .modifier(SettingsActionPillStyle(tint: .secondary, prominent: false))
                        .disabled(isCalibratingIndicatorNoise)
                    }

                    Button {
                        calibrateIndicatorNoiseFloor()
                    } label: {
                        Label(isCalibratingIndicatorNoise ? "Listening..." : "Calibrate", systemImage: "waveform")
                    }
                    .buttonStyle(.plain)
                    .modifier(SettingsActionPillStyle(tint: .accentColor, prominent: true))
                    .disabled(isCalibratingIndicatorNoise || microphoneStatus != .authorized)
                }
            }

            if let noiseCalibrationMessage {
                Text(noiseCalibrationMessage)
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(.secondary)
            }
        }
        .padding(12)
        .background(.thinMaterial, in: .rect(cornerRadius: 10, style: .continuous))
        .padding(.vertical, 2)
    }

    private var noiseCalibrationDescription: String {
        if microphoneStatus != .authorized {
            return "Grant microphone access before calibrating the indicator noise floor."
        }

        if recordingIndicatorNoiseFloorRMS > 0 {
            return "Saved quiet-room floor: \(formattedNoiseFloor). This is subtracted before voice is mapped to the glow."
        }

        return "Measure the current room while quiet so background noise does not drive the glow."
    }

    private var formattedNoiseFloor: String {
        String(format: "%.5f RMS", recordingIndicatorNoiseFloorRMS)
    }

    private func calibrateIndicatorNoiseFloor() {
        guard !isCalibratingIndicatorNoise else { return }

        isCalibratingIndicatorNoise = true
        noiseCalibrationMessage = "Stay quiet for a moment..."

        appState.recordingIndicator.calibrateNoiseFloor { floor in
            isCalibratingIndicatorNoise = false
            if let floor {
                recordingIndicatorNoiseFloorRMS = floor
                noiseCalibrationMessage = "Noise floor saved: \(String(format: "%.5f RMS", floor))."
            } else {
                noiseCalibrationMessage = "Could not read microphone noise. Check microphone access."
            }
        }
    }

    // MARK: - Batch Settings

    private struct BatchLengthPreset: Identifiable {
        let seconds: Double
        let title: String

        var id: Double { seconds }
    }

    private struct RAMBudgetPreset: Identifiable {
        let megabytes: Int
        let title: String

        var id: Int { megabytes }
    }

    private var batchLengthPresets: [BatchLengthPreset] {
        [
            BatchLengthPreset(seconds: 30, title: "30 seconds - Experimental"),
            BatchLengthPreset(seconds: 60, title: "1 min"),
            BatchLengthPreset(seconds: 90, title: "1.5 min"),
            BatchLengthPreset(seconds: 120, title: "2 min")
        ]
    }

    private var ramBudgetPresets: [RAMBudgetPreset] {
        [
            RAMBudgetPreset(megabytes: 3072, title: "3 GB"),
            RAMBudgetPreset(megabytes: 4096, title: "4 GB"),
            RAMBudgetPreset(megabytes: 6144, title: "6 GB"),
            RAMBudgetPreset(megabytes: 8192, title: "8 GB")
        ]
    }

    private var batchInfoPanel: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: basicBatchingEnabled ? "bolt.horizontal.circle.fill" : "pause.circle.fill")
                .font(.system(size: 18, weight: .semibold))
                .foregroundStyle(basicBatchingEnabled ? .green : .secondary)

            VStack(alignment: .leading, spacing: 4) {
                Text(basicBatchingEnabled ? "Recommended for long dictation" : "Long recordings will process after you stop")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(.primary)

                Text(batchDescription)
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Spacer(minLength: 0)
        }
        .padding(12)
        .background(.thinMaterial, in: .rect(cornerRadius: 12, style: .continuous))
        .padding(.top, 4)
    }

    private var batchDescription: String {
        if basicBatchingEnabled {
            let suffix = basicBatchingThresholdSeconds <= 30
                ? " Watch the debug console for chunk timing and boundary quality."
                : ""
            return "Current setting: a new local chunk starts about every \(formattedBatchLength). Fallback memory budget: \(formattedRAMBudget).\(suffix)"
        }

        return "Use this only for troubleshooting. The validated default is enabled with 2-minute rolling chunks."
    }

    private var formattedBatchLength: String {
        if basicBatchingThresholdSeconds < 60 {
            return "\(Int(basicBatchingThresholdSeconds)) seconds"
        }

        let minutes = basicBatchingThresholdSeconds / 60
        if minutes.rounded() == minutes {
            return "\(Int(minutes)) minute\(minutes == 1 ? "" : "s")"
        }

        return String(format: "%.1f minutes", minutes)
    }

    private var formattedRAMBudget: String {
        let gb = Double(basicBatchingRamBudgetMB) / 1024
        if gb.rounded() == gb {
            return "\(Int(gb)) GB"
        }

        return String(format: "%.1f GB", gb)
    }
}

// MARK: - Reusable Settings Components

private struct SettingsActionPillStyle: ViewModifier {
    let tint: Color
    let prominent: Bool

    func body(content: Content) -> some View {
        content
            .font(.system(size: 10, weight: .semibold))
            .foregroundStyle(prominent ? .white : tint)
            .padding(.horizontal, 12)
            .frame(height: 28)
            .background(prominent ? AnyShapeStyle(tint) : AnyShapeStyle(Color.primary.opacity(0.075)), in: Capsule())
            .overlay(Capsule().strokeBorder(prominent ? tint.opacity(0.28) : .white.opacity(0.06), lineWidth: 0.6))
            .contentShape(Capsule())
    }
}

private struct RecordingIndicatorStyleOption: View {
    let style: RecordingIndicatorStyle
    let isSelected: Bool
    let action: () -> Void
    @State private var isHoveringPreview = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 12) {
                RecordingIndicatorStylePreview(
                    style: style,
                    isActive: isSelected || isHoveringPreview
                )
                    .frame(width: 70, height: 42)

                VStack(alignment: .leading, spacing: 3) {
                    Text(style.displayName)
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(.primary)

                    Text(style.description)
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(.secondary)
                }

                Spacer(minLength: 0)

                Image(systemName: selectionIconName)
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(selectionIconColor)
            }
            .padding(10)
            .frame(maxWidth: .infinity, minHeight: 66, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(isSelected ? Color.green.opacity(0.10) : Color.primary.opacity(0.045))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .strokeBorder(isSelected ? Color.green.opacity(0.42) : Color.white.opacity(0.08), lineWidth: 1)
            )
        }
        .buttonStyle(.plain)
        .onHover { isHoveringPreview = $0 }
        .accessibilityLabel("\(style.displayName) recording indicator style")
        .accessibilityValue(isSelected ? "Selected" : "Not selected")
    }

    private var selectionIconName: String {
        isSelected ? "checkmark.circle.fill" : "circle"
    }

    private var selectionIconColor: Color {
        isSelected ? .green : .secondary.opacity(0.55)
    }
}

private struct RecordingIndicatorStylePreview: View {
    let style: RecordingIndicatorStyle
    let isActive: Bool
    @Environment(\.colorScheme) private var colorScheme
    @Environment(\.controlActiveState) private var controlActiveState

    /// The 18fps timeline only runs while the settings window is frontmost —
    /// a selected card would otherwise keep redrawing behind other windows.
    private var isWindowActive: Bool {
        controlActiveState == .key || controlActiveState == .active
    }

    var body: some View {
        Group {
            if isActive && isWindowActive {
                TimelineView(.periodic(from: Self.animationStartTime, by: 1.0 / 18.0)) { timeline in
                    previewContent(date: timeline.date)
                }
            } else {
                previewContent(date: inactivePreviewDate)
            }
        }
        .accessibilityHidden(true)
    }

    private static let animationStartTime = Date(timeIntervalSinceReferenceDate: 0)

    private var inactivePreviewDate: Date {
        Date(timeIntervalSinceReferenceDate: Double(styleIndex) * 0.23 + 0.42)
    }

    private var styleIndex: Int {
        RecordingIndicatorStyle.allCases.firstIndex(of: style) ?? 0
    }

    @ViewBuilder
    private func previewContent(date: Date) -> some View {
        let level = isActive ? previewLevel(at: date) : 0.58

        ZStack {
            Capsule()
                .fill(
                    colorScheme == .dark
                        ? Color(white: 0.12, opacity: 0.72)
                        : Color(white: 0.92, opacity: 0.85)
                )
                .overlay(
                    Capsule()
                        .fill(
                            LinearGradient(
                                colors: colorScheme == .dark
                                    ? [Color.white.opacity(0.12), Color.clear]
                                    : [Color.white.opacity(0.5), Color.clear],
                                startPoint: .top,
                                endPoint: .center
                            )
                        )
                )
                .overlay(
                    Capsule()
                        .strokeBorder(Color.white.opacity(colorScheme == .dark ? 0.22 : 0.56), lineWidth: 0.5)
                )

            RecordingIndicatorMiniVisual(
                style: style,
                level: level,
                now: date,
                animationStartTime: Self.animationStartTime,
                fluidPhase: date.timeIntervalSinceReferenceDate * 0.72,
                waveProgresses: previewWaveProgresses(at: date),
                showsTranscriptionDots: false,
                showsOnlyTranscriptionDots: false
            )
        }
    }

    private func previewLevel(at date: Date) -> CGFloat {
        let phase = (sin(date.timeIntervalSinceReferenceDate * 2.2) + 1) / 2
        return CGFloat(0.38 + phase * 0.42)
    }

    private func previewWaveProgresses(at date: Date) -> [CGFloat] {
        let cycle = date.timeIntervalSinceReferenceDate.truncatingRemainder(dividingBy: 1.4) / 1.4
        let first = CGFloat(cycle)
        let second = CGFloat((cycle + 0.52).truncatingRemainder(dividingBy: 1))
        return [first, second]
    }
}

/// Section header — small caps title above a group of rows
struct SettingsSectionHeader: View {
    let title: String

    var body: some View {
        HStack {
            Text(title.uppercased())
                .font(.system(size: 11, weight: .semibold, design: .rounded))
                .foregroundStyle(.secondary)
                .tracking(0.8)
            Spacer()
        }
        .padding(.top, 20)
        .padding(.bottom, 8)
    }
}

/// A single settings row: title + description on the left, control on the right
struct SettingsRow<Control: View>: View {
    let title: String
    let description: String
    @ViewBuilder let control: Control

    var body: some View {
        HStack(alignment: .center, spacing: 16) {
            VStack(alignment: .leading, spacing: 3) {
                Text(title)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(.primary)

                Text(description)
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
                    .lineLimit(3)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Spacer()

            control
        }
        .padding(.vertical, 10)
        .padding(.horizontal, 14)
        .background(.thinMaterial, in: .rect(cornerRadius: 10, style: .continuous))
        .padding(.vertical, 2)
    }
}

/// Transparent divider between sections
struct SettingsDivider: View {
    var body: some View {
        Rectangle()
            .fill(.white.opacity(0.06))
            .frame(height: 0.5)
            .padding(.horizontal, 4)
            .padding(.vertical, 4)
    }
}

// MARK: - Staggered Appear Modifier

/// Cascading slide-up animation for settings rows.
/// Uses offset + clip only (no opacity) — opacity breaks material compositing
/// and causes system controls (Toggle, Picker, Slider) to flash black.
struct StaggerInModifier: ViewModifier {
    let index: Int
    let appeared: Bool

    func body(content: Content) -> some View {
        content
            .offset(y: appeared ? 0 : 16)
            .animation(
                .spring(duration: 0.4, bounce: 0.12).delay(Double(index) * 0.05),
                value: appeared
            )
    }
}

extension View {
    func staggerIn(index: Int, appeared: Bool) -> some View {
        modifier(StaggerInModifier(index: index, appeared: appeared))
    }
}
