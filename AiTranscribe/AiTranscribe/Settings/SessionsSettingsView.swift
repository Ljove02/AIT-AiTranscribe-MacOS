import SwiftUI
import AppKit
import ScreenCaptureKit

// MARK: - Sessions Settings View

struct SessionsSettingsView: View {
    @EnvironmentObject var sessionManager: SessionManager
    var onNavigateToModels: ((ModelMode) -> Void)? = nil

    let hasAnimated: Bool
    let onAnimated: () -> Void
    let initialSelectedSessionId: UUID?

    @State private var selectedSessionId: UUID? = nil
    @State private var showDeleteAllAlert = false
    @State private var showDeleteTranscribedAlert = false
    @State private var showDeleteSelectedSessionsAlert = false
    @State private var showClearTranscriptionAlert = false
    @State private var screenRecordingPermission = false
    @State private var isSelectingSessions = false
    @State private var selectedSessionIds: Set<UUID> = []
    @State private var showAudioMixSheet = false
    @State private var selectedDate: Date? = nil
    @State private var showDatePicker = false

    @AppStorage("hideSessionInfoBanner") private var hideBanner = false
    @AppStorage(SessionAudioMixPreferences.micTrimDBKey) private var sessionMicTrimDB = SessionAudioMixPreferences.defaultMicTrimDB
    @AppStorage(SessionAudioMixPreferences.systemTrimDBKey) private var sessionSystemTrimDB = SessionAudioMixPreferences.defaultSystemTrimDB

    /// Animation state
    @State private var appeared: Bool

    init(hasAnimated: Bool, onAnimated: @escaping () -> Void, onNavigateToModels: ((ModelMode) -> Void)? = nil, initialSelectedSessionId: UUID? = nil) {
        self.hasAnimated = hasAnimated
        self.onAnimated = onAnimated
        self.onNavigateToModels = onNavigateToModels
        self.initialSelectedSessionId = initialSelectedSessionId
        _appeared = State(initialValue: hasAnimated)
        _selectedSessionId = State(initialValue: initialSelectedSessionId)
    }

    // MARK: - Computed

    private var filteredSessions: [Session] {
        guard let date = selectedDate else { return sessionManager.sessions }
        return sessionManager.sessions.filter { Calendar.current.isDate($0.createdAt, inSameDayAs: date) }
    }

    /// Group sessions by date (newest first)
    private var groupedSessions: [(date: Date, sessions: [Session])] {
        let grouped = Dictionary(grouping: filteredSessions) { session in
            Calendar.current.startOfDay(for: session.createdAt)
        }
        return grouped
            .map { (date: $0.key, sessions: $0.value.sorted { $0.createdAt > $1.createdAt }) }
            .sorted { $0.date > $1.date }
    }

    private var totalStorageSize: Int64 {
        sessionManager.getTotalStorageSize()
    }

    private var uniqueDates: [Date] {
        let dates = Set(sessionManager.sessions.map { Calendar.current.startOfDay(for: $0.createdAt) })
        return dates.sorted(by: >)
    }

    var body: some View {
        VStack(spacing: 0) {
            if let selectedId = selectedSessionId,
               sessionManager.sessions.contains(where: { $0.id == selectedId }) {
                SessionDetailView(sessionId: selectedId, onBack: {
                    selectedSessionId = nil
                }, onNavigateToModels: onNavigateToModels)
            } else {
                sessionsListView
            }
        }
        .task {
            sessionManager.loadSessions()
            screenRecordingPermission = await SystemAudioCapture.checkPermission()
        }
        .task(id: "stagger") {
            guard !hasAnimated else { return }
            try? await Task.sleep(for: .milliseconds(80))
            appeared = true
            onAnimated()
        }
        .onChange(of: sessionManager.sessions) { _, sessions in
            let validIds = Set(sessions.map(\.id))
            selectedSessionIds = selectedSessionIds.intersection(validIds)
            if sessions.isEmpty { isSelectingSessions = false }
        }
    }

    // MARK: - Sessions List

    private var sessionsListView: some View {
        ScrollView {
            VStack(spacing: 0) {
                // Header
                sessionsHeader
                    .staggerIn(index: 0, appeared: appeared)

                // Permission pills
                permissionPills
                    .padding(.top, 8)
                    .staggerIn(index: 1, appeared: appeared)

                SettingsDivider()
                    .padding(.top, 12)
                    .staggerIn(index: 2, appeared: appeared)

                // Info banner
                if !hideBanner {
                    infoBanner
                        .padding(.top, 12)
                        .staggerIn(index: 3, appeared: appeared)
                }

                // Date filter chips
                if !uniqueDates.isEmpty {
                    dateFilterRow
                        .padding(.top, 12)
                        .staggerIn(index: 4, appeared: appeared)
                }

                SettingsDivider()
                    .padding(.top, 12)
                    .staggerIn(index: 5, appeared: appeared)

                // Session list or empty state
                if sessionManager.sessions.isEmpty {
                    emptyStateView
                        .staggerIn(index: 6, appeared: appeared)
                } else if filteredSessions.isEmpty {
                    noMatchView
                        .staggerIn(index: 6, appeared: appeared)
                } else {
                    sessionGroups
                        .staggerIn(index: 6, appeared: appeared)
                }

                // Footer actions
                if !sessionManager.sessions.isEmpty {
                    SettingsDivider()
                        .padding(.top, 8)
                    footerActions
                        .padding(.top, 8)
                        .staggerIn(index: 7, appeared: appeared)
                }

                Spacer(minLength: 20)
            }
            .padding(.horizontal, 28)
            .padding(.top, 8)
        }
        .scrollIndicators(.automatic)
        .sheet(isPresented: $showAudioMixSheet) {
            AudioMixSheetView(
                micTrimDB: $sessionMicTrimDB,
                systemTrimDB: $sessionSystemTrimDB,
                latestSessionName: latestSessionWithAudio?.name,
                latestAudioURL: latestSessionAudioURL
            )
        }
    }

    // MARK: - Header

    private var sessionsHeader: some View {
        VStack(spacing: 12) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Sessions")
                        .font(.system(size: 22, weight: .bold, design: .rounded))

                    HStack(spacing: 6) {
                        Text("\(sessionManager.sessions.count) sessions")
                            .font(.system(size: 11))
                            .foregroundStyle(.secondary)
                        if totalStorageSize > 0 {
                            Text("·")
                                .foregroundStyle(.tertiary)
                            Text(SessionManager.formatFileSize(totalStorageSize))
                                .font(.system(size: 11))
                                .foregroundStyle(.secondary)
                        }
                    }
                }

                Spacer()

                // Audio mix button
                Button {
                    showAudioMixSheet = true
                } label: {
                    HStack(spacing: 5) {
                        Image(systemName: "slider.horizontal.3")
                            .font(.system(size: 10, weight: .medium))
                        Text("Audio Mix")
                            .font(.system(size: 11, weight: .medium))
                    }
                    .foregroundStyle(.primary)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 7)
                    .background(.thinMaterial, in: Capsule())
                    .overlay(Capsule().strokeBorder(.white.opacity(0.06), lineWidth: 0.5))
                }
                .buttonStyle(.plain)

                // Select button
                Button {
                    if isSelectingSessions {
                        exitSelectionMode()
                    } else {
                        isSelectingSessions = true
                    }
                } label: {
                    Text(isSelectingSessions ? "Done" : "Select")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(isSelectingSessions ? .white : .secondary)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 7)
                        .background(
                            isSelectingSessions
                                ? AnyShapeStyle(Color.accentColor)
                                : AnyShapeStyle(Color.primary.opacity(0.06))
                        )
                        .clipShape(Capsule())
                }
                .buttonStyle(.plain)
                .disabled(sessionManager.sessions.isEmpty)
            }
        }
    }

    // MARK: - Permission Pills

    private var permissionPills: some View {
        HStack(spacing: 8) {
            permissionPill(
                label: "Microphone",
                granted: AudioRecorder.hasMicrophonePermission
            )
            permissionPill(
                label: "Screen Recording",
                granted: screenRecordingPermission,
                onRequest: !screenRecordingPermission ? {
                    SystemAudioCapture.requestPermission()
                    Task {
                        try? await Task.sleep(for: .seconds(2))
                        screenRecordingPermission = await SystemAudioCapture.checkPermission()
                    }
                } : nil
            )

            if isSelectingSessions {
                footerPill(label: "Delete Selected", tint: .red) {
                    showDeleteSelectedSessionsAlert = true
                }
                .disabled(selectedSessionIds.isEmpty)
                .alert("Delete Selected Sessions?", isPresented: $showDeleteSelectedSessionsAlert) {
                    Button("Cancel", role: .cancel) {}
                    Button("Delete", role: .destructive) {
                        sessionManager.bulkDeleteSessions(ids: selectedSessionIds)
                        exitSelectionMode()
                    }
                } message: {
                    Text("Selected sessions and their data will be permanently deleted.")
                }

                footerPill(label: "Clear Transcription", tint: .orange) {
                    showClearTranscriptionAlert = true
                }
                .disabled(selectedSessionIds.isEmpty)
                .alert("Clear Transcription?", isPresented: $showClearTranscriptionAlert) {
                    Button("Cancel", role: .cancel) {}
                    Button("Clear", role: .destructive) {
                        sessionManager.bulkResetTranscription(ids: selectedSessionIds)
                        exitSelectionMode()
                    }
                } message: {
                    Text("Transcriptions for selected sessions will be deleted. Audio files are kept.")
                }
            }

            Spacer()
        }
    }

    private func permissionPill(label: String, granted: Bool, onRequest: (() -> Void)? = nil) -> some View {
        Button {
            onRequest?()
        } label: {
            HStack(spacing: 5) {
                Circle()
                    .fill(granted ? .green : .red)
                    .frame(width: 6, height: 6)
                Text(label)
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(.primary)
                if !granted && onRequest != nil {
                    Text("Grant")
                        .font(.system(size: 9, weight: .semibold))
                        .foregroundStyle(.blue)
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .background(.thinMaterial, in: Capsule())
            .overlay(Capsule().strokeBorder(.white.opacity(0.06), lineWidth: 0.5))
        }
        .buttonStyle(.plain)
        .disabled(granted || onRequest == nil)
    }

    // MARK: - Info Banner

    private var infoBanner: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "info.circle.fill")
                .font(.system(size: 14))
                .foregroundStyle(.blue)

            VStack(alignment: .leading, spacing: 4) {
                Text("Record long-form audio sessions")
                    .font(.system(size: 12, weight: .semibold))

                Text("Capture meetings, lectures, or interviews from the menu bar, then transcribe here.")
                    .font(.system(size: 10))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Spacer()

            Button {
                withAnimation { hideBanner = true }
            } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 9, weight: .medium))
                    .foregroundStyle(.tertiary)
                    .frame(width: 20, height: 20)
                    .background(.quaternary, in: .circle)
            }
            .buttonStyle(.plain)
        }
        .padding(12)
        .background(.thinMaterial, in: .rect(cornerRadius: 12, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(.blue.opacity(0.15), lineWidth: 0.5)
        )
    }

    // MARK: - Date Filter

    private static let dateChipFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "MMM d"
        return f
    }()

    private var dateFilterRow: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 4) {
                // "All" chip
                dateChip(label: "All", isSelected: selectedDate == nil) {
                    selectedDate = nil
                }

                ForEach(Array(uniqueDates.prefix(10)), id: \.self) { date in
                    let label = Calendar.current.isDateInToday(date) ? "Today"
                        : Calendar.current.isDateInYesterday(date) ? "Yesterday"
                        : Self.dateChipFormatter.string(from: date)

                    let count = sessionManager.sessions.filter {
                        Calendar.current.isDate($0.createdAt, inSameDayAs: date)
                    }.count

                    dateChip(
                        label: "\(label) (\(count))",
                        isSelected: selectedDate.map { Calendar.current.isDate($0, inSameDayAs: date) } ?? false
                    ) {
                        selectedDate = (selectedDate.map { Calendar.current.isDate($0, inSameDayAs: date) } ?? false) ? nil : date
                    }
                }

                // Calendar picker
                Button {
                    showDatePicker = true
                } label: {
                    Image(systemName: "calendar")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 6)
                        .background(
                            showDatePicker
                                ? AnyShapeStyle(Color.accentColor.opacity(0.1))
                                : AnyShapeStyle(Color.primary.opacity(0.06))
                        )
                        .clipShape(Capsule())
                }
                .buttonStyle(.plain)
                .popover(isPresented: $showDatePicker, arrowEdge: .bottom) {
                    SessionCalendarPopover(
                        sessionDates: sessionDateCounts,
                        selectedDate: $selectedDate,
                        isPresented: $showDatePicker
                    )
                }
            }
        }
    }

    private var sessionDateCounts: [Date: Int] {
        var counts: [Date: Int] = [:]
        for session in sessionManager.sessions {
            let day = Calendar.current.startOfDay(for: session.createdAt)
            counts[day, default: 0] += 1
        }
        return counts
    }

    private func dateChip(label: String, isSelected: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(label)
                .font(.system(size: 10, weight: isSelected ? .semibold : .regular))
                .foregroundStyle(isSelected ? .primary : .secondary)
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(
                    isSelected
                        ? AnyShapeStyle(Color.primary.opacity(0.08))
                        : AnyShapeStyle(.clear)
                )
                .clipShape(Capsule())
                .animation(.easeOut(duration: 0.15), value: isSelected)
        }
        .buttonStyle(.plain)
    }

    // MARK: - Session Groups

    private static let groupHeaderFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "EEEE, MMMM d"
        return f
    }()

    private var sessionGroups: some View {
        VStack(spacing: 16) {
            ForEach(groupedSessions, id: \.date) { group in
                VStack(alignment: .leading, spacing: 8) {
                    // Date group header
                    HStack(spacing: 6) {
                        let label = Calendar.current.isDateInToday(group.date) ? "Today"
                            : Calendar.current.isDateInYesterday(group.date) ? "Yesterday"
                            : Self.groupHeaderFormatter.string(from: group.date)
                        Text(label.uppercased())
                            .font(.system(size: 10, weight: .semibold, design: .rounded))
                            .foregroundStyle(.tertiary)
                            .tracking(0.6)

                        Text("(\(group.sessions.count))")
                            .font(.system(size: 9))
                            .foregroundStyle(.quaternary)
                    }
                    .padding(.leading, 4)

                    // Session cards
                    VStack(spacing: 6) {
                        ForEach(group.sessions) { session in
                            SessionCardView(
                                session: session,
                                selectionMode: isSelectingSessions,
                                isSelected: selectedSessionIds.contains(session.id),
                                onRename: { newName in
                                    sessionManager.renameSession(id: session.id, newName: newName)
                                }
                            )
                            .onTapGesture {
                                if isSelectingSessions {
                                    toggleSessionSelection(session.id)
                                } else {
                                    withAnimation(.easeInOut(duration: 0.15)) {
                                        selectedSessionId = session.id
                                    }
                                }
                            }
                            .contextMenu {
                                if session.hasTranscription, let text = session.transcriptionText {
                                    Button("Copy Transcription") {
                                        NSPasteboard.general.clearContents()
                                        NSPasteboard.general.setString(text, forType: .string)
                                    }
                                    Divider()
                                }
                                if session.hasAudio {
                                    Button("Show in Finder") {
                                        let audioURL = sessionManager.getAudioURL(for: session)
                                        NSWorkspace.shared.activateFileViewerSelecting([audioURL])
                                    }
                                    Button("Delete Audio Only") {
                                        sessionManager.deleteSessionAudio(id: session.id)
                                    }
                                    Divider()
                                }
                                Button("Delete Session", role: .destructive) {
                                    sessionManager.deleteSession(id: session.id)
                                }
                            }
                        }
                    }
                }
            }
        }
        .padding(.top, 12)
    }

    // MARK: - Empty / No Match

    private var emptyStateView: some View {
        VStack(spacing: 12) {
            Image(systemName: "waveform.circle")
                .font(.system(size: 36))
                .foregroundStyle(.tertiary)
            Text("No sessions yet")
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(.secondary)
            Text("Record your first session from the menu bar.")
                .font(.system(size: 11))
                .foregroundStyle(.tertiary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 60)
    }

    private var noMatchView: some View {
        VStack(spacing: 8) {
            Image(systemName: "calendar.badge.exclamationmark")
                .font(.system(size: 24))
                .foregroundStyle(.tertiary)
            Text("No sessions on this date")
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(.secondary)
            Button("Show all") { selectedDate = nil }
                .font(.system(size: 11))
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 40)
    }

    // MARK: - Footer Actions

    private var footerActions: some View {
        HStack(spacing: 8) {
            Text(isSelectingSessions ? "\(selectedSessionIds.count) of \(sessionManager.sessions.count) selected" : "")
                .font(.system(size: 10))
                .foregroundStyle(.tertiary)

            Spacer()

            if isSelectingSessions {
                footerPill(label: allSessionsSelected ? "Clear All" : "Select All", tint: .secondary) {
                    toggleSelectAllSessions()
                }
            }
        }
    }

    private func footerPill(label: String, tint: Color, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(label)
                .font(.system(size: 10, weight: .medium))
                .foregroundStyle(tint)
                .padding(.horizontal, 10)
                .padding(.vertical, 5)
                .background(.thinMaterial, in: Capsule())
        }
        .buttonStyle(.plain)
    }

    // MARK: - Audio Mix Helpers

    private var latestSessionWithAudio: Session? {
        sessionManager.sessions.first { $0.hasAudio }
    }

    private var latestSessionAudioURL: URL? {
        guard let session = latestSessionWithAudio else { return nil }
        let url = sessionManager.getAudioURL(for: session)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    // MARK: - Selection Helpers

    private var allSessionsSelected: Bool {
        !sessionManager.sessions.isEmpty && selectedSessionIds.count == sessionManager.sessions.count
    }

    private func toggleSessionSelection(_ sessionId: UUID) {
        if selectedSessionIds.contains(sessionId) {
            selectedSessionIds.remove(sessionId)
        } else {
            selectedSessionIds.insert(sessionId)
        }
    }

    private func toggleSelectAllSessions() {
        if allSessionsSelected {
            selectedSessionIds.removeAll()
        } else {
            selectedSessionIds = Set(sessionManager.sessions.map(\.id))
        }
    }

    private func exitSelectionMode() {
        isSelectingSessions = false
        selectedSessionIds.removeAll()
    }
}

// MARK: - Audio Mix Sheet

struct AudioMixSheetView: View {
    @Binding var micTrimDB: Double
    @Binding var systemTrimDB: Double
    let latestSessionName: String?
    let latestAudioURL: URL?
    @Environment(\.dismiss) var dismiss

    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack(spacing: 16) {
                Image(systemName: "slider.horizontal.3")
                    .font(.system(size: 36))
                    .foregroundStyle(.orange.gradient)

                VStack(alignment: .leading, spacing: 4) {
                    Text("Session Audio Mix")
                        .font(.title2)
                        .fontWeight(.semibold)
                    Text("Adjust mic and desktop levels for session recordings")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                Spacer()
            }
            .padding(20)

            Divider()

            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    // Mic slider
                    mixSlider(
                        title: "Microphone Boost",
                        icon: "mic.fill",
                        value: $micTrimDB,
                        range: SessionAudioMixPreferences.micTrimRange,
                        valueStr: SessionAudioMixPreferences.displayString(for: micTrimDB),
                        gainStr: SessionAudioMixPreferences.multiplierString(
                            for: SessionAudioMixPreferences.effectiveMicGain(forTrimDB: micTrimDB)
                        ),
                        minLabel: "Less",
                        maxLabel: "More",
                        tint: .blue
                    )

                    // System slider
                    mixSlider(
                        title: "Desktop Audio",
                        icon: "desktopcomputer",
                        value: $systemTrimDB,
                        range: SessionAudioMixPreferences.systemTrimRange,
                        valueStr: SessionAudioMixPreferences.displayString(for: systemTrimDB),
                        gainStr: SessionAudioMixPreferences.multiplierString(
                            for: SessionAudioMixPreferences.effectiveSystemGain(forTrimDB: systemTrimDB)
                        ),
                        minLabel: "Quieter",
                        maxLabel: "Louder",
                        tint: .purple
                    )

                    Divider()

                    // Quick check
                    VStack(alignment: .leading, spacing: 10) {
                        Text("QUICK CHECK")
                            .font(.system(size: 10, weight: .semibold, design: .rounded))
                            .foregroundStyle(.tertiary)
                            .tracking(0.6)

                        if let audioURL = latestAudioURL, let name = latestSessionName {
                            HStack(spacing: 8) {
                                Button {
                                    NSWorkspace.shared.open(audioURL)
                                } label: {
                                    HStack(spacing: 4) {
                                        Image(systemName: "play.fill")
                                            .font(.system(size: 9))
                                        Text("Play Latest")
                                            .font(.system(size: 11, weight: .medium))
                                    }
                                    .foregroundStyle(.white)
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 7)
                                    .background(.orange, in: Capsule())
                                }
                                .buttonStyle(.plain)

                                Button {
                                    NSWorkspace.shared.activateFileViewerSelecting([audioURL])
                                } label: {
                                    HStack(spacing: 4) {
                                        Image(systemName: "folder")
                                            .font(.system(size: 9))
                                        Text("Show in Finder")
                                            .font(.system(size: 11, weight: .medium))
                                    }
                                    .foregroundStyle(.primary)
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 7)
                                    .background(.quaternary, in: Capsule())
                                }
                                .buttonStyle(.plain)

                                Spacer()
                            }

                            Text(name)
                                .font(.system(size: 10))
                                .foregroundStyle(.tertiary)
                        } else {
                            Text("Record a session to audition your mix here.")
                                .font(.system(size: 11))
                                .foregroundStyle(.tertiary)
                        }
                    }

                    Text("These settings only affect session recordings. Basic recording is not affected.")
                        .font(.system(size: 10))
                        .foregroundStyle(.quaternary)
                }
                .padding(24)
            }

            Divider()

            // Footer
            HStack {
                Button("Reset") {
                    micTrimDB = SessionAudioMixPreferences.defaultMicTrimDB
                    systemTrimDB = SessionAudioMixPreferences.defaultSystemTrimDB
                }
                .disabled(abs(micTrimDB) < 0.05 && abs(systemTrimDB) < 0.05)

                Spacer()

                Button("Done") { dismiss() }
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.defaultAction)
            }
            .padding(20)
        }
        .frame(width: 500, height: 480)
    }

    private func mixSlider(
        title: String,
        icon: String,
        value: Binding<Double>,
        range: ClosedRange<Double>,
        valueStr: String,
        gainStr: String,
        minLabel: String,
        maxLabel: String,
        tint: Color
    ) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: icon)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(tint)
                Text(title)
                    .font(.system(size: 13, weight: .semibold))
                Spacer()
                Text(valueStr)
                    .font(.system(size: 12, weight: .medium, design: .monospaced))
                Text("· \(gainStr)")
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(.secondary)
            }

            Slider(
                value: value,
                in: range,
                step: 0.5,
                minimumValueLabel: Text(minLabel).font(.system(size: 9)).foregroundStyle(.tertiary),
                maximumValueLabel: Text(maxLabel).font(.system(size: 9)).foregroundStyle(.tertiary)
            ) { Text(title) }
            .tint(tint)
        }
    }
}

// MARK: - Session Calendar Popover

struct SessionCalendarPopover: View {
    let sessionDates: [Date: Int]
    @Binding var selectedDate: Date?
    @Binding var isPresented: Bool

    @State private var displayedMonth: Date

    private let calendar = Calendar.current

    private static let monthFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "MMMM yyyy"
        return f
    }()

    private static let infoDateFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "MMM d, yyyy"
        return f
    }()

    init(sessionDates: [Date: Int], selectedDate: Binding<Date?>, isPresented: Binding<Bool>) {
        self.sessionDates = sessionDates
        self._selectedDate = selectedDate
        self._isPresented = isPresented
        let month = selectedDate.wrappedValue ?? Date()
        self._displayedMonth = State(initialValue: Calendar.current.startOfDay(for: month))
    }

    private var monthStart: Date {
        let comps = calendar.dateComponents([.year, .month], from: displayedMonth)
        return calendar.date(from: comps)!
    }

    private var daysInMonth: Int {
        calendar.range(of: .day, in: .month, for: monthStart)!.count
    }

    private var firstWeekdayOffset: Int {
        (calendar.component(.weekday, from: monthStart) - calendar.firstWeekday + 7) % 7
    }

    private var adjustedWeekdaySymbols: [String] {
        let symbols = calendar.veryShortWeekdaySymbols
        let first = calendar.firstWeekday - 1
        return Array(symbols[first...]) + Array(symbols[..<first])
    }

    private var sessionCountForMonth: Int {
        (1...daysInMonth).reduce(0) { total, day in
            let date = calendar.date(byAdding: .day, value: day - 1, to: monthStart)!
            return total + (sessionDates[calendar.startOfDay(for: date)] ?? 0)
        }
    }

    private var isCurrentMonth: Bool {
        calendar.isDate(monthStart, equalTo: Date(), toGranularity: .month)
    }

    var body: some View {
        VStack(spacing: 12) {
            // Month navigation
            HStack {
                Button { goToPreviousMonth() } label: {
                    Image(systemName: "chevron.left")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(.secondary)
                        .frame(width: 28, height: 28)
                        .background(.quaternary, in: .circle)
                }
                .buttonStyle(.plain)

                Spacer()

                VStack(spacing: 2) {
                    Text(Self.monthFormatter.string(from: displayedMonth))
                        .font(.system(size: 13, weight: .semibold))
                    if sessionCountForMonth > 0 {
                        Text("\(sessionCountForMonth) session\(sessionCountForMonth == 1 ? "" : "s")")
                            .font(.system(size: 9))
                            .foregroundStyle(.secondary)
                    }
                }

                Spacer()

                Button { goToNextMonth() } label: {
                    Image(systemName: "chevron.right")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(.secondary)
                        .frame(width: 28, height: 28)
                        .background(.quaternary, in: .circle)
                }
                .buttonStyle(.plain)
                .disabled(isCurrentMonth || monthStart > Date())
            }

            // Weekday headers
            HStack(spacing: 0) {
                ForEach(adjustedWeekdaySymbols, id: \.self) { symbol in
                    Text(symbol)
                        .font(.system(size: 9, weight: .medium))
                        .foregroundStyle(.tertiary)
                        .frame(maxWidth: .infinity)
                }
            }

            // Day grid
            let columns = Array(repeating: GridItem(.flexible(), spacing: 2), count: 7)
            LazyVGrid(columns: columns, spacing: 4) {
                ForEach(0..<firstWeekdayOffset, id: \.self) { _ in
                    Color.clear.frame(height: 34)
                }

                ForEach(1...daysInMonth, id: \.self) { day in
                    let date = calendar.date(byAdding: .day, value: day - 1, to: monthStart)!
                    dayCell(day: day, date: date)
                }
            }

            // Selected date info
            if let sel = selectedDate {
                let count = sessionDates[calendar.startOfDay(for: sel)] ?? 0
                HStack(spacing: 6) {
                    Text(Self.infoDateFormatter.string(from: sel))
                        .font(.system(size: 10, weight: .semibold))
                    Text("·")
                        .foregroundStyle(.tertiary)
                    Text("\(count) session\(count == 1 ? "" : "s")")
                        .font(.system(size: 10))
                        .foregroundStyle(count > 0 ? .green : .secondary)
                }
            }
        }
        .padding(16)
        .frame(width: 270)
    }

    private func dayCell(day: Int, date: Date) -> some View {
        let isToday = calendar.isDateInToday(date)
        let isSelected = selectedDate.map { calendar.isDate($0, inSameDayAs: date) } ?? false
        let count = sessionDates[calendar.startOfDay(for: date)] ?? 0
        let hasSession = count > 0
        let isFuture = date > Date() && !isToday

        return Button {
            if isSelected {
                selectedDate = nil
            } else {
                selectedDate = date
            }
            isPresented = false
        } label: {
            VStack(spacing: 2) {
                Text("\(day)")
                    .font(.system(size: 12, weight: isToday ? .bold : (hasSession ? .semibold : .regular)))
                    .foregroundStyle(
                        isSelected ? .white :
                        isFuture ? .secondary.opacity(0.4) :
                        hasSession ? .primary : .secondary
                    )

                Circle()
                    .fill(hasSession ? .green : .clear)
                    .frame(width: 4, height: 4)
            }
            .frame(maxWidth: .infinity)
            .frame(height: 34)
            .background(
                isSelected
                    ? AnyShapeStyle(Color.accentColor)
                    : isToday
                        ? AnyShapeStyle(Color.accentColor.opacity(0.1))
                        : AnyShapeStyle(.clear)
            )
            .clipShape(.rect(cornerRadius: 6, style: .continuous))
        }
        .buttonStyle(.plain)
        .disabled(isFuture)
    }

    private func goToPreviousMonth() {
        displayedMonth = calendar.date(byAdding: .month, value: -1, to: displayedMonth) ?? displayedMonth
    }

    private func goToNextMonth() {
        displayedMonth = calendar.date(byAdding: .month, value: 1, to: displayedMonth) ?? displayedMonth
    }
}
