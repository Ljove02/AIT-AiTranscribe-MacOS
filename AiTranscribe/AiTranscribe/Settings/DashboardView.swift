import SwiftUI

// MARK: - Dashboard View

struct DashboardView: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var sessionManager: SessionManager
    @ObservedObject var statsManager: DashboardStatsManager

    private let activityGridHeight: CGFloat = 252
    private let insightCardHeight: CGFloat = 132
    private let chartRowHeight: CGFloat = 112

    let hasAnimated: Bool
    let onAnimated: () -> Void
    let onNavigateToHistory: (UUID?) -> Void
    let onNavigateToSessions: (UUID?) -> Void

    @State private var appeared: Bool
    @State private var chartsAnimated = false
    @State private var showResetAPICostAlert = false

    init(
        statsManager: DashboardStatsManager,
        hasAnimated: Bool,
        onAnimated: @escaping () -> Void,
        onNavigateToHistory: @escaping (UUID?) -> Void,
        onNavigateToSessions: @escaping (UUID?) -> Void
    ) {
        self.statsManager = statsManager
        self.hasAnimated = hasAnimated
        self.onAnimated = onAnimated
        self.onNavigateToHistory = onNavigateToHistory
        self.onNavigateToSessions = onNavigateToSessions
        _appeared = State(initialValue: hasAnimated)
        _chartsAnimated = State(initialValue: hasAnimated)
    }

    var body: some View {
        ScrollView {
            VStack(spacing: 0) {
                if !statsManager.hasLoaded {
                    loadingState
                } else if statsManager.stats.totalTranscriptions == 0 && statsManager.hasLoaded {
                    emptyState
                } else {
                    statsContent
                }
            }
            .padding(.horizontal, 28)
            .padding(.top, 8)
        }
        .scrollIndicators(.automatic)
        .task {
            if statsManager.hasLoaded {
                // Already cached — show instantly
                chartsAnimated = true
                return
            }
            chartsAnimated = false
            let entries = await appState.loadAllHistoryEntries()
            await statsManager.computeStats(from: entries, sessions: sessionManager.sessions)

            await Task.yield()
            try? await Task.sleep(for: .milliseconds(120))
            withAnimation(.easeOut(duration: 0.42)) {
                chartsAnimated = true
            }
        }
        .task(id: "stagger") {
            guard !hasAnimated else { return }
            try? await Task.sleep(for: .milliseconds(80))
            appeared = true
            onAnimated()
        }
    }

    // MARK: - Loading State

    private var loadingState: some View {
        Color.clear
            .frame(maxWidth: .infinity)
            .frame(height: 360)
    }

    // MARK: - Empty State

    private var emptyState: some View {
        VStack(spacing: 16) {
            Spacer().frame(height: 200)
            Image(systemName: "waveform.badge.mic")
                .font(.system(size: 40, weight: .light))
                .foregroundStyle(.tertiary)

            VStack(spacing: 4) {
                Text("No transcriptions yet")
                    .font(.system(size: 16, weight: .semibold, design: .rounded))

                Text("Start recording to see your stats here")
                    .font(.system(size: 13))
                    .foregroundStyle(.secondary)
            }
            Spacer()
        }
        .frame(maxWidth: .infinity)
    }

    // MARK: - Stats Content

    private var statsContent: some View {
        VStack(spacing: 0) {
            headerSection
                .staggerIn(index: 0, appeared: appeared)

            Spacer().frame(height: 12)

            heroStatsRow
                .staggerIn(index: 1, appeared: appeared)

            Spacer().frame(height: 12)

            activityContributionCard
                .staggerIn(index: 2, appeared: appeared)
                .zIndex(1)

            Spacer().frame(height: 12)

            insightCardsRow
                .staggerIn(index: 3, appeared: appeared)

            Spacer().frame(height: 12)

            bottomAnalyticsRow
                .staggerIn(index: 4, appeared: appeared)

            Spacer(minLength: 16)
        }
    }

    // MARK: - Header

    private var headerSection: some View {
        HStack(alignment: .center, spacing: 16) {
            VStack(alignment: .leading, spacing: 3) {
                Text("Dashboard")
                    .font(.system(size: 22, weight: .bold, design: .rounded))

                Text(dashboardMetadataText)
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
            }

            Spacer(minLength: 12)

            HStack(spacing: 8) {
                QuickActionPill(icon: "clock.arrow.circlepath", label: "Last Transcription") {
                    let lastId = appState.transcriptionHistory.first?.id
                    onNavigateToHistory(lastId)
                }

                QuickActionPill(icon: "waveform.circle", label: "Last Session") {
                    let lastId = sessionManager.sessions.first?.id
                    onNavigateToSessions(lastId)
                }

                QuickActionPill(icon: "ladybug", label: "Report a Bug") {
                    if let url = URL(string: "https://github.com/Ljove02/AIT-AiTranscribe-MacOS/issues") {
                        NSWorkspace.shared.open(url)
                    }
                }
            }
        }
    }

    // MARK: - Row 1: Hero Stats (equal width)

    private var heroStatsRow: some View {
        HStack(spacing: 10) {
            HeroStatCard(
                value: formatTimeSaved(statsManager.stats.timeSavedMinutes),
                label: "Time Saved",
                icon: "clock.badge.checkmark",
                tint: .green
            )
            .frame(maxWidth: .infinity)

            HeroStatCard(
                value: formatLargeNumber(statsManager.stats.totalWords),
                label: "Words Transcribed",
                icon: "text.word.spacing",
                tint: .blue
            )
            .frame(maxWidth: .infinity)

            HeroStatCard(
                value: "\(statsManager.stats.totalActiveDays)",
                label: "Active Days",
                icon: "calendar.badge.checkmark",
                tint: .teal
            )
            .frame(maxWidth: .infinity)

            HeroStatCard(
                value: "\(statsManager.stats.totalTranscriptions)",
                label: "Transcriptions",
                icon: "number",
                tint: .purple
            )
            .frame(maxWidth: .infinity)
        }
    }

    // MARK: - Row 2: Activity Contribution Grid

    private var activityContributionCard: some View {
        DashboardCard2(title: "Day Streak", icon: "square.grid.3x3.fill", fixedHeight: activityGridHeight) {
            ActivityContributionPanel(
                days: statsManager.stats.activityDays,
                currentStreak: statsManager.stats.currentStreak,
                longestStreak: statsManager.stats.longestStreak,
                activeDays: statsManager.stats.totalActiveDays,
                maxCount: statsManager.stats.activityMaxCount,
                animated: chartsAnimated
            )
        }
    }

    // MARK: - Row 3: Supporting Insights

    private var insightCardsRow: some View {
        HStack(spacing: 10) {
            topModelCard
                .frame(maxWidth: .infinity)
            topWordCard
                .frame(maxWidth: .infinity)
            speakingRhythmCard
                .frame(maxWidth: .infinity)
        }
    }

    private var topModelCard: some View {
        DashboardCard2(title: "Top Model", icon: "cube.box.fill", fixedHeight: insightCardHeight) {
            VStack(alignment: .leading, spacing: 8) {
                if let model = statsManager.stats.favoriteModel {
                    Text(model.name)
                        .font(.system(size: 13, weight: .semibold, design: .rounded))
                        .lineLimit(2)
                        .minimumScaleFactor(0.8)

                    GeometryReader { geo in
                        ZStack(alignment: .leading) {
                            RoundedRectangle(cornerRadius: 3, style: .continuous)
                                .fill(.white.opacity(0.06))

                            RoundedRectangle(cornerRadius: 3, style: .continuous)
                                .fill(Color.accentColor.opacity(0.7))
                                .frame(width: geo.size.width * CGFloat(model.percentage) / 100)
                        }
                    }
                    .frame(height: 6)

                    Text("\(model.percentage)% of all")
                        .font(.system(size: 10))
                        .foregroundStyle(.secondary)

                    if statsManager.stats.allModels.count > 1 {
                        VStack(alignment: .leading, spacing: 3) {
                            ForEach(Array(statsManager.stats.allModels.dropFirst().prefix(2).enumerated()), id: \.offset) { _, m in
                                HStack(spacing: 4) {
                                    Circle()
                                        .fill(.white.opacity(0.15))
                                        .frame(width: 4, height: 4)
                                    Text(shortenModelName(m.name))
                                        .font(.system(size: 9))
                                        .foregroundStyle(.tertiary)
                                        .lineLimit(1)
                                    Spacer()
                                    Text("\(m.percentage)%")
                                        .font(.system(size: 9, weight: .medium))
                                        .foregroundStyle(.tertiary)
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private var topWordCard: some View {
        DashboardCard2(title: "Top Words", icon: "textformat", fixedHeight: insightCardHeight) {
            VStack(alignment: .leading, spacing: 6) {
                if let topWord = statsManager.stats.topWords.first {
                    HStack(alignment: .firstTextBaseline, spacing: 6) {
                        Text("\"\(topWord.word)\"")
                            .font(.system(size: 16, weight: .bold, design: .rounded))
                            .lineLimit(1)

                        Text("\(topWord.count)×")
                            .font(.system(size: 11, weight: .medium))
                            .foregroundStyle(.secondary)
                    }
                }

                let remaining = Array(statsManager.stats.topWords.dropFirst().prefix(4))
                if !remaining.isEmpty {
                    let maxCount = statsManager.stats.topWords.first?.count ?? 1
                    VStack(spacing: 4) {
                        ForEach(Array(remaining.enumerated()), id: \.offset) { _, item in
                            HStack(spacing: 6) {
                                Text(item.word)
                                    .font(.system(size: 10, weight: .medium))
                                    .foregroundStyle(.secondary)
                                    .frame(width: 55, alignment: .leading)
                                    .lineLimit(1)

                                GeometryReader { geo in
                                    RoundedRectangle(cornerRadius: 2, style: .continuous)
                                        .fill(Color.accentColor.opacity(0.4))
                                        .frame(width: max(4, geo.size.width * CGFloat(item.count) / CGFloat(maxCount)))
                                }
                                .frame(height: 4)

                                Text("\(item.count)")
                                    .font(.system(size: 9, weight: .medium, design: .monospaced))
                                    .foregroundStyle(.tertiary)
                            }
                        }
                    }
                }
            }
        }
    }

    private var speakingRhythmCard: some View {
        DashboardCard2(title: "Speaking Rhythm", icon: "gauge.with.needle", fixedHeight: insightCardHeight) {
            VStack(alignment: .leading, spacing: 12) {
                HStack(spacing: 10) {
                    RhythmMetric(value: "\(Int(statsManager.stats.averageWPM))", label: "Avg WPM", tint: .orange)

                    if let peak = peakHour {
                        RhythmMetric(value: formatHour(peak.hour), label: "Peak Hour", tint: .purple)
                    } else {
                        RhythmMetric(value: "-", label: "Peak Hour", tint: .purple)
                    }

                    RhythmMetric(value: formatDuration(statsManager.stats.totalDurationSeconds), label: "Spoken", tint: .teal)
                }

                if let busiest = statsManager.stats.busiestDay {
                    HStack(spacing: 6) {
                        Image(systemName: "flame.fill")
                            .font(.system(size: 10, weight: .semibold))
                            .foregroundStyle(.orange.opacity(0.85))

                        Text("Busiest day \(formatDateString(busiest.date))")
                            .font(.system(size: 10, weight: .medium))
                            .foregroundStyle(.secondary)

                        Spacer(minLength: 4)

                        Text("\(busiest.count) entries")
                            .font(.system(size: 10, weight: .medium, design: .monospaced))
                            .foregroundStyle(.tertiary)
                    }
                }
            }
        }
    }

    // MARK: - Row 4: Chart Grid

    private var bottomAnalyticsRow: some View {
        HStack(spacing: 10) {
            hourlyCard
                .frame(maxWidth: .infinity)

            recordingDistributionCard
                .frame(maxWidth: .infinity)

            if statsManager.stats.apiCostItemCount > 0 {
                apiCostCard
                    .frame(maxWidth: .infinity)
            }
        }
    }

    private var row4_charts: some View {
        VStack(spacing: 10) {
            HStack(spacing: 10) {
                hourlyCard
                    .frame(maxWidth: .infinity)
                wpmTrendCard
                    .frame(maxWidth: .infinity)
            }

            HStack(spacing: 10) {
                apiCostCard
                    .frame(maxWidth: .infinity)
                recordingDistributionCard
                    .frame(maxWidth: .infinity)
            }
        }
    }

    private var hourlyCard: some View {
        DashboardCard2(title: "Hourly Activity", icon: "clock", fixedHeight: chartRowHeight) {
            let maxCount = statsManager.stats.hourlyBreakdown.map(\.count).max() ?? 1
            HStack(alignment: .bottom, spacing: 1) {
                ForEach(Array(statsManager.stats.hourlyBreakdown.enumerated()), id: \.offset) { _, item in
                    RoundedRectangle(cornerRadius: 1.5, style: .continuous)
                        .fill(hourColor(item.hour).opacity(item.count > 0 ? 0.7 : 0.1))
                        .frame(height: chartsAnimated ? max(2, CGFloat(item.count) / CGFloat(maxCount) * 44) : 1)
                }
            }
            .frame(height: 44)

            HStack {
                Text("12am")
                Spacer()
                Text("6am")
                Spacer()
                Text("12pm")
                Spacer()
                Text("6pm")
                Spacer()
                Text("12")
            }
            .font(.system(size: 7, weight: .medium))
            .foregroundStyle(.quaternary)
        }
    }

    private func hourColor(_ hour: Int) -> Color {
        switch hour {
        case 6..<12:  return .orange
        case 12..<18: return .accentColor
        case 18..<22: return .purple
        default:      return .gray
        }
    }

    private var wpmTrendCard: some View {
        DashboardCard2(title: "WPM Trend", icon: "gauge.with.needle", fixedHeight: chartRowHeight) {
            if statsManager.stats.weeklyWPM.count >= 2 {
                VStack(spacing: 4) {
                    WPMSparkline(data: statsManager.stats.weeklyWPM.map(\.wpm), animated: chartsAnimated)
                        .frame(height: 44)

                    HStack {
                        if let first = statsManager.stats.weeklyWPM.first {
                            Text(first.week)
                                .font(.system(size: 7, weight: .medium))
                                .foregroundStyle(.quaternary)
                        }
                        Spacer()
                        if let last = statsManager.stats.weeklyWPM.last {
                            Text(last.week)
                                .font(.system(size: 7, weight: .medium))
                                .foregroundStyle(.quaternary)
                        }
                    }
                }
            } else {
                VStack(spacing: 4) {
                    Spacer()
                    Text("Not enough data")
                        .font(.system(size: 10))
                        .foregroundStyle(.tertiary)
                    Spacer()
                }
                .frame(height: 50)
            }
        }
    }

    private var apiCostCard: some View {
        DashboardCard2(title: "API Cost", icon: "dollarsign.circle", fixedHeight: chartRowHeight) {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text(formatCurrency(statsManager.stats.estimatedAPICostUSD))
                    .font(.system(size: 18, weight: .bold, design: .rounded))
                    .foregroundStyle(statsManager.stats.estimatedAPICostUSD > 0 ? Color.green : Color.secondary)
                    .lineLimit(1)
                    .minimumScaleFactor(0.75)

                Spacer()

                Text("\(statsManager.stats.apiCostItemCount) priced")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(.secondary)
            }

            HStack(spacing: 8) {
                Text("Manual model pricing estimate")
                    .font(.system(size: 9))
                    .foregroundStyle(.tertiary)

                Spacer()

                Button {
                    showResetAPICostAlert = true
                } label: {
                    Text("Reset")
                        .font(.system(size: 9, weight: .semibold))
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3)
                        .background(.white.opacity(0.05), in: Capsule(style: .continuous))
                }
                .buttonStyle(.plain)
                .disabled(statsManager.stats.apiCostItemCount == 0)
            }
            // Attached to the row rather than the button: `.disabled` propagates
            // into its subtree, which would reach the alert's own buttons.
            .alert("Reset API cost tracking?", isPresented: $showResetAPICostAlert) {
                Button("Cancel", role: .cancel) {}
                Button("Reset", role: .destructive) {
                    Task {
                        let entries = await appState.loadAllHistoryEntries()
                        await statsManager.resetAPICosts(from: entries, sessions: sessionManager.sessions)
                    }
                }
            } message: {
                Text(resetAPICostMessage)
            }
        }
    }

    private var recordingDistributionCard: some View {
        DashboardCard2(title: "Recording Distribution", icon: "ruler", fixedHeight: chartRowHeight) {
            let maxCount = statsManager.stats.lengthDistribution.map(\.count).max() ?? 1
            VStack(spacing: 6) {
                ForEach(Array(statsManager.stats.lengthDistribution.enumerated()), id: \.offset) { _, bucket in
                    HStack(spacing: 6) {
                        Text(bucket.label)
                            .font(.system(size: 9, weight: .medium))
                            .foregroundStyle(.secondary)
                            .frame(width: 52, alignment: .trailing)
                            .lineLimit(1)
                            .fixedSize()

                        GeometryReader { geo in
                            RoundedRectangle(cornerRadius: 2, style: .continuous)
                                .fill(Color.accentColor.opacity(0.5))
                                .frame(width: chartsAnimated ? max(4, geo.size.width * CGFloat(bucket.count) / CGFloat(maxCount)) : 4)
                        }
                        .frame(height: 10)

                        Text("\(bucket.count)")
                            .font(.system(size: 9, weight: .medium, design: .monospaced))
                            .foregroundStyle(.tertiary)
                            .frame(width: 32, alignment: .leading)
                    }
                }
            }
        }
    }

    // MARK: - Formatting Helpers

    private func formatTimeSaved(_ minutes: Double) -> String {
        let hours = minutes / 60
        if hours >= 1 {
            return String(format: "%.1fh", hours)
        }
        return "\(Int(minutes))m"
    }

    private func formatLargeNumber(_ n: Int) -> String {
        if n >= 1_000_000 {
            return String(format: "%.1fM", Double(n) / 1_000_000)
        } else if n >= 1_000 {
            return String(format: "%.1fK", Double(n) / 1_000)
        }
        return "\(n)"
    }

    /// Spells out exactly what the running total is before it disappears, and
    /// makes clear this is a counting cut-off rather than deleted history.
    private var resetAPICostMessage: String {
        let total = formatCurrency(statsManager.stats.estimatedAPICostUSD)
        let count = statsManager.stats.apiCostItemCount
        let noun = count == 1 ? "transcription" : "transcriptions"

        return """
        The running total of \(total) across \(count.formatted()) priced \(noun) will stop counting, and the estimate starts again from now.

        Nothing is deleted — your transcriptions and their history are untouched. This only changes what the estimate counts, and it cannot be undone.
        """
    }

    private func formatCurrency(_ value: Double) -> String {
        if value <= 0 { return "$0.00" }
        if value < 0.01 { return String(format: "$%.4f", value) }
        return String(format: "$%.2f", value)
    }

    private var dashboardMetadataText: String {
        var parts: [String] = ["\(statsManager.stats.totalTranscriptions) transcriptions"]

        if let since = statsManager.stats.usingSinceDate {
            parts.append("Using since \(formatMonthDay(since))")
        }

        parts.append("\(statsManager.stats.totalActiveDays) active days")

        if let peak = peakHour {
            parts.append("Peak \(formatHour(peak.hour))")
        }

        return parts.joined(separator: " / ")
    }

    private var peakHour: (hour: Int, count: Int)? {
        guard let peak = statsManager.stats.hourlyBreakdown.max(by: { $0.count < $1.count }),
              peak.count > 0 else {
            return nil
        }
        return peak
    }

    private func formatDateString(_ dateStr: String) -> String {
        let inputFormatter = DateFormatter()
        inputFormatter.dateFormat = "yyyy-MM-dd"
        guard let date = inputFormatter.date(from: dateStr) else { return dateStr }
        let outputFormatter = DateFormatter()
        outputFormatter.dateFormat = "MMM d"
        return outputFormatter.string(from: date)
    }

    private func formatMonthDay(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "MMM d"
        return formatter.string(from: date)
    }

    private func shortenModelName(_ name: String) -> String {
        let parts = name.split(separator: " ")
        if parts.count > 2 {
            return "\(parts.first ?? "") \(parts.last ?? "")"
        }
        return name
    }

    private func formatHour(_ hour: Int) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "h a"
        let cal = Calendar.current
        let date = cal.date(bySettingHour: hour, minute: 0, second: 0, of: Date()) ?? Date()
        return formatter.string(from: date)
    }

    private func formatDuration(_ seconds: Double) -> String {
        let hours = seconds / 3600
        if hours >= 1 {
            return String(format: "%.1fh", hours)
        }
        return "\(Int(seconds / 60))m"
    }

    private func timeAgoString(from date: Date) -> String {
        let components = Calendar.current.dateComponents([.month, .day], from: date, to: Date())
        let months = components.month ?? 0
        let days = components.day ?? 0

        if months > 0 {
            return "\(months)mo, \(days)d ago"
        }
        return "\(days) day\(days == 1 ? "" : "s") ago"
    }
}

// MARK: - Activity Contribution Panel

struct ActivityContributionPanel: View {
    let days: [DashboardActivityDay]
    let currentStreak: Int
    let longestStreak: Int
    let activeDays: Int
    let maxCount: Int
    let animated: Bool

    /// Chunked weeks and month labels depend only on the activity data, never
    /// the cursor. Hover moves re-run `body` on this same struct value 60-120x/s,
    /// so they're computed once in `init` (i.e. only when the parent re-renders
    /// with new data) instead of on every mouse-move pass.
    private let weekGroups: [[DashboardActivityDay]]
    private let monthLabelItems: [(weekIndex: Int, label: String)]

    init(
        days: [DashboardActivityDay],
        currentStreak: Int,
        longestStreak: Int,
        activeDays: Int,
        maxCount: Int,
        animated: Bool
    ) {
        self.days = days
        self.currentStreak = currentStreak
        self.longestStreak = longestStreak
        self.activeDays = activeDays
        self.maxCount = maxCount
        self.animated = animated

        // Locals mirror `dayCount`/`weekCount` — self isn't usable yet here.
        let dayCount = 7
        let groups: [[DashboardActivityDay]] = days.isEmpty ? [] : stride(from: 0, to: days.count, by: dayCount).map { start in
            Array(days[start..<min(start + dayCount, days.count)])
        }
        self.weekGroups = groups
        self.monthLabelItems = Self.contributionMonthLabels(weeks: groups, weekCount: 52)
    }

    private let weekCount = 52
    private let dayCount = 7
    private let labelWidth: CGFloat = 25
    private let cellSpacing: CGFloat = 4

    /// Dock-style magnification field, measured in grid cells.
    ///
    /// `magnifyPlateau` pins the cell actually under the pointer at full lift
    /// rather than letting it ease off as the cursor crosses it, then
    /// `magnifyDecay` steps the rings down around it:
    ///
    ///   hovered  100%  ·  ring 1  40%  ·  ring 2  10%  ·  ring 3  2%
    ///
    /// `magnifyRadius` cuts the tail off where it rounds to nothing, which caps
    /// a mouse move at ~45 redrawn cells out of 364.
    private let magnifyPlateau: Double = 0.35
    private let magnifyDecay: Double = 1.41
    private let magnifyRadius: Double = 3.5

    /// How long the cursor has to sit still before the detail card appears.
    /// Sweeping across the grid should not fire fifty tooltips.
    private let dwellDelay: Duration = .milliseconds(500)

    /// Cursor position in *grid units* — 1.0 is one cell plus one gap, not one
    /// point. Keeping it unitless means the falloff math is independent of the
    /// cell size the layout settled on.
    @State private var cursor: CGPoint?
    @State private var focused: GridCell?

    /// The tooltip keeps its last day and anchor after the cursor leaves so it
    /// can fade out with its content intact, rather than being torn out of the
    /// hierarchy mid-transition.
    @State private var tooltipDay: DashboardActivityDay?
    @State private var tooltipCell: GridCell?
    @State private var tooltipSize: CGSize = .zero
    @State private var isTooltipVisible = false
    @State private var dwellTask: Task<Void, Never>?

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    struct GridCell: Equatable {
        let week: Int
        let day: Int
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .firstTextBaseline, spacing: 16) {
                HStack(alignment: .firstTextBaseline, spacing: 7) {
                    Text("\(currentStreak)")
                        .font(.system(size: 30, weight: .bold, design: .rounded))
                        .foregroundStyle(.primary)
                        .contentTransition(.numericText())

                    Text("day streak")
                        .font(.system(size: 18, weight: .semibold, design: .rounded))
                        .foregroundStyle(.primary.opacity(0.9))
                }

                Spacer()

                HStack(spacing: 16) {
                    ActivityInlineStat(value: "\(longestStreak)", label: "Longest")
                    ActivityInlineStat(value: "\(activeDays)", label: "Active days")
                    ActivityInlineStat(value: "\(maxCount)", label: "Best day")
                }
            }

            GeometryReader { proxy in
                let cellSize = contributionCellSize(for: proxy.size.width)
                let gridWidth = CGFloat(weekCount) * cellSize + CGFloat(weekCount - 1) * cellSpacing

                HStack {
                    Spacer(minLength: 0)

                    VStack(alignment: .leading, spacing: 6) {
                        monthLabels(cellSize: cellSize)
                            .frame(width: labelWidth + gridWidth, height: 14, alignment: .leading)

                        HStack(alignment: .top, spacing: 7) {
                            weekdayLabels(cellSize: cellSize)

                            contributionGrid(
                                weekGroups: weekGroups,
                                cellSize: cellSize,
                                gridWidth: gridWidth
                            )
                        }
                        // The tooltip rides in the grid's overlay, so the whole
                        // row has to outrank the legend underneath it. Without
                        // this, "N days with activity" paints straight through
                        // the glass.
                        .zIndex(1)

                        HStack(spacing: 8) {
                            Text("Less")
                                .font(.system(size: 10, weight: .medium))
                                .foregroundStyle(.tertiary)

                            HStack(spacing: 4) {
                                ForEach(0..<5, id: \.self) { level in
                                    RoundedRectangle(cornerRadius: 2.5, style: .continuous)
                                        .fill(ContributionCell.color(for: level, maxLevel: 4))
                                        .frame(width: 10, height: 10)
                                }
                            }

                            Text("More")
                                .font(.system(size: 10, weight: .medium))
                                .foregroundStyle(.tertiary)

                            Spacer()

                            Text("\(activeDays) days with activity")
                                .font(.system(size: 10, weight: .medium))
                                .foregroundStyle(.tertiary)
                                .lineLimit(1)
                        }
                        .padding(.leading, labelWidth + 7)
                        .padding(.top, 1)
                    }
                    .frame(width: labelWidth + 7 + gridWidth, alignment: .leading)

                    Spacer(minLength: 0)
                }
            }
            .frame(height: 150)
        }
        .onDisappear {
            dwellTask?.cancel()
            dwellTask = nil
        }
    }

    // MARK: Grid

    /// The cell matrix, plus the single hover tracker that drives both the
    /// magnification and the tooltip. One tracking area for the whole grid
    /// replaces the 364 per-cell `onHover` regions this used to install —
    /// which is why hover used to register only intermittently.
    private func contributionGrid(
        weekGroups: [[DashboardActivityDay]],
        cellSize: CGFloat,
        gridWidth: CGFloat
    ) -> some View {
        let pitch = cellSize + cellSpacing

        return HStack(alignment: .top, spacing: cellSpacing) {
            ForEach(Array(weekGroups.prefix(weekCount).enumerated()), id: \.offset) { weekIndex, week in
                VStack(spacing: cellSpacing) {
                    ForEach(Array(week.enumerated()), id: \.element.id) { dayIndex, day in
                        let lift = lift(week: weekIndex, day: dayIndex)

                        ContributionCell(
                            day: day,
                            maxCount: maxCount,
                            size: cellSize,
                            animated: animated,
                            index: weekIndex * dayCount + dayIndex,
                            lift: lift,
                            isFocused: focused == GridCell(week: weekIndex, day: dayIndex),
                            isTracking: cursor != nil
                        )
                        .equatable()
                        .zIndex(lift)
                    }
                }
                .zIndex(columnLift(week: weekIndex))
            }
        }
        .contentShape(Rectangle())
        .onContinuousHover { phase in
            handleHover(phase, pitch: pitch, weekGroups: weekGroups)
        }
        .overlay(alignment: .topLeading) {
            tooltipOverlay(cellSize: cellSize, gridWidth: gridWidth)
        }
        .onChange(of: focused) { _, newValue in
            if let newValue {
                scheduleTooltip(for: newValue)
            } else {
                hideTooltip()
            }
        }
    }

    private func handleHover(_ phase: HoverPhase, pitch: CGFloat, weekGroups: [[DashboardActivityDay]]) {
        guard animated, pitch > 0 else { return }

        switch phase {
        case .active(let point):
            cursor = CGPoint(x: point.x / pitch, y: point.y / pitch)

            let week = Int(floor(point.x / pitch))
            let day = Int(floor(point.y / pitch))
            guard week >= 0, day >= 0, day < dayCount,
                  week < min(weekGroups.count, weekCount),
                  day < weekGroups[week].count
            else {
                focused = nil
                return
            }
            focused = GridCell(week: week, day: day)

        case .ended:
            cursor = nil
            focused = nil
        }
    }

    /// How much this cell rises, 0 at rest through 1 directly under the cursor.
    private func lift(week: Int, day: Int) -> Double {
        guard let cursor else { return 0 }
        let dx = Double(week) + 0.5 - cursor.x
        let dy = Double(day) + 0.5 - cursor.y
        return falloff(ringDistance(dx, dy))
    }

    /// A p = 4 norm rather than a plain circle. On a square grid this pulls the
    /// diagonal neighbours in towards their orthogonal siblings, so each ring
    /// of cells lifts as a ring instead of as a cross.
    private func ringDistance(_ dx: Double, _ dy: Double) -> Double {
        let x2 = dx * dx
        let y2 = dy * dy
        return (x2 * x2 + y2 * y2).squareRoot().squareRoot()
    }

    /// Columns need their own z-order so a magnified cell is never overlapped
    /// by the column to its right.
    private func columnLift(week: Int) -> Double {
        guard let cursor else { return 0 }
        return falloff(abs(Double(week) + 0.5 - cursor.x))
    }

    /// Exponential decay past the plateau, so the field tapers smoothly out to
    /// nothing instead of ending on a hard edge. Quantised to two decimals so
    /// cells whose lift did not visibly change compare equal and skip their
    /// body entirely.
    private func falloff(_ distance: Double) -> Double {
        // Magnification is motion. With Reduce Motion on, the focus ring and
        // the tooltip still do the job of saying which day is under the cursor.
        guard !reduceMotion, distance < magnifyRadius else { return 0 }
        return (exp(-magnifyDecay * max(0, distance - magnifyPlateau)) * 100).rounded() / 100
    }

    // MARK: Tooltip

    @ViewBuilder
    private func tooltipOverlay(cellSize: CGFloat, gridWidth: CGFloat) -> some View {
        if let day = tooltipDay, let cell = tooltipCell {
            let pitch = cellSize + cellSpacing
            let cellTop = CGFloat(cell.day) * pitch
            let above = cellTop - tooltipSize.height - 9
            let placeAbove = above >= 0
            let x = min(
                max(CGFloat(cell.week) * pitch + cellSize / 2 - tooltipSize.width / 2, 0),
                max(gridWidth - tooltipSize.width, 0)
            )

            StreakTooltip(day: day, accent: ContributionCell.color(count: day.transcriptionCount, maxCount: maxCount))
                .onGeometryChange(for: CGSize.self) { $0.size } action: { tooltipSize = $0 }
                .offset(x: x, y: placeAbove ? above : cellTop + cellSize + 9)
                .scaleEffect(
                    isTooltipVisible || reduceMotion ? 1 : 0.94,
                    anchor: placeAbove ? .bottom : .top
                )
                .opacity(isTooltipVisible ? 1 : 0)
                .allowsHitTesting(false)
        }
    }

    private func scheduleTooltip(for cell: GridCell) {
        guard let day = day(at: cell) else {
            hideTooltip()
            return
        }

        dwellTask?.cancel()

        // Every cell earns its own dwell. Moving to a neighbour drops the card
        // at once and starts the wait over, so a card is only ever on screen
        // for a day the cursor actually settled on.
        isTooltipVisible = false

        // Content is swapped now, while the card is invisible, so it re-measures
        // and re-anchors before it is ever seen. Updating it on the way in was
        // what left the glass mid-resize behind already-swapped text.
        tooltipDay = day
        tooltipCell = cell

        dwellTask = Task { @MainActor in
            try? await Task.sleep(for: dwellDelay)
            guard !Task.isCancelled else { return }
            withAnimation(reduceMotion ? .easeOut(duration: 0.16) : .spring(duration: 0.26, bounce: 0.18)) {
                isTooltipVisible = true
            }
        }
    }

    /// Leaving the grid altogether fades the card out — nothing is changing
    /// underneath it, so a fade reads as calm rather than glitchy.
    private func hideTooltip() {
        dwellTask?.cancel()
        dwellTask = nil
        withAnimation(.easeOut(duration: 0.14)) {
            isTooltipVisible = false
        }
    }

    private func day(at cell: GridCell) -> DashboardActivityDay? {
        let groups = weekGroups
        guard cell.week >= 0, cell.week < groups.count else { return nil }
        let week = groups[cell.week]
        guard cell.day >= 0, cell.day < week.count else { return nil }
        return week[cell.day]
    }

    // MARK: Layout helpers

    private func contributionCellSize(for width: CGFloat) -> CGFloat {
        let availableWidth = max(0, width - labelWidth - 7)
        let spacingWidth = CGFloat(weekCount - 1) * cellSpacing
        let rawSize = (availableWidth - spacingWidth) / CGFloat(weekCount)
        return min(13, max(6, floor(rawSize)))
    }

    private func weekdayLabels(cellSize: CGFloat) -> some View {
        let labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        return VStack(alignment: .trailing, spacing: cellSpacing) {
            ForEach(Array(labels.enumerated()), id: \.offset) { _, label in
                Text(label)
                    .font(.system(size: 9, weight: .medium))
                    .foregroundStyle(.tertiary)
                    .frame(width: labelWidth, height: cellSize, alignment: .trailing)
            }
        }
    }

    private func monthLabels(cellSize: CGFloat) -> some View {
        ZStack(alignment: .topLeading) {
            ForEach(monthLabelItems, id: \.weekIndex) { item in
                Text(item.label)
                    .font(.system(size: 9, weight: .medium))
                    .foregroundStyle(.secondary)
                    .offset(x: labelWidth + 7 + CGFloat(item.weekIndex) * (cellSize + cellSpacing))
            }
        }
    }

    private static let monthLabelFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "MMM"
        return formatter
    }()

    private static func contributionMonthLabels(weeks: [[DashboardActivityDay]], weekCount: Int) -> [(weekIndex: Int, label: String)] {
        let calendar = Calendar.current

        return weeks.prefix(weekCount).enumerated().compactMap { index, week in
            guard let firstDay = week.first else { return nil }
            let day = calendar.component(.day, from: firstDay.date)
            if index == 0 || day <= 7 {
                return (index, monthLabelFormatter.string(from: firstDay.date))
            }
            return nil
        }
    }
}

struct ContributionCell: View, Equatable {
    let day: DashboardActivityDay
    let maxCount: Int
    let size: CGFloat
    let animated: Bool
    let index: Int

    /// 0 at rest, 1 directly under the cursor. Quantised by the panel so cells
    /// outside the magnification radius stay at exactly 0, compare equal, and
    /// let `.equatable()` skip this body on every mouse move.
    let lift: Double
    let isFocused: Bool

    /// True while the cursor is anywhere over the grid.
    let isTracking: Bool

    /// Peak magnification. 1.55 reads as a clear pick without the cell
    /// swallowing its neighbours.
    private let maxScale: Double = 0.55

    var body: some View {
        let shape = RoundedRectangle(cornerRadius: 2.5, style: .continuous)

        shape
            .fill(fillColor)
            .frame(width: size, height: size)
            .overlay(
                shape.strokeBorder(.white.opacity(borderOpacity), lineWidth: isFocused ? 0.9 : 0.5)
            )
            .scaleEffect(animated ? 1 + lift * maxScale : 0.65)
            .opacity(animated ? 1 : 0)
            .shadow(
                color: day.hasActivity ? fillColor.opacity(0.18 + lift * 0.4) : .clear,
                radius: 3 + lift * 6
            )
            // While the cursor is over the grid the lift is applied with no
            // animation at all, so the bump sits exactly under the pointer no
            // matter how fast you sweep. A spring here lagged behind a quick
            // swipe badly enough that nothing appeared to rise. The spring is
            // only used on the way out, when the whole grid settles back down.
            .animation(isTracking ? nil : .spring(duration: 0.24, bounce: 0.1), value: lift)
            .animation(.spring(duration: 0.55, bounce: 0.12).delay(Double(index % 42) * 0.01), value: animated)
            .accessibilityLabel(accessibilityText)
    }

    private var borderOpacity: Double {
        if isFocused { return 0.5 }
        return day.hasActivity ? 0.08 : 0.025
    }

    private var fillColor: Color {
        Self.color(count: day.transcriptionCount, maxCount: maxCount)
    }

    private var accessibilityText: String {
        guard day.hasActivity else {
            return "\(day.displayDate), no transcriptions"
        }

        let transcriptionLabel = day.transcriptionCount == 1 ? "transcription" : "transcriptions"
        return "\(day.displayDate), \(day.transcriptionCount) \(transcriptionLabel), \(day.wordCount) words"
    }

    // MARK: Palette

    static let emptyColor = Color.white.opacity(0.055)

    /// Legend swatches, which think in levels rather than counts.
    static func color(for level: Int, maxLevel: Int) -> Color {
        guard level > 0 else { return emptyColor }
        return color(ratio: Double(level) / Double(max(maxLevel, 1)))
    }

    static func color(count: Int, maxCount: Int) -> Color {
        guard count > 0, maxCount > 0 else { return emptyColor }
        return color(ratio: Double(count) / Double(maxCount))
    }

    private static func color(ratio: Double) -> Color {
        switch ratio {
        case 0.75...:
            return Color(red: 0.20, green: 0.82, blue: 0.35)
        case 0.50..<0.75:
            return Color(red: 0.12, green: 0.62, blue: 0.30)
        case 0.25..<0.50:
            return Color(red: 0.08, green: 0.42, blue: 0.25)
        default:
            return Color(red: 0.05, green: 0.25, blue: 0.18)
        }
    }
}

// MARK: - Streak Tooltip

/// The dwell card for a single day. Uses the same glass recipe as the meeting
/// reminder notch so the two floating surfaces in the app read as one material.
private struct StreakTooltip: View {
    let day: DashboardActivityDay
    let accent: Color

    var body: some View {
        VStack(alignment: .leading, spacing: day.hasActivity ? 6 : 4) {
            HStack(spacing: 6) {
                RoundedRectangle(cornerRadius: 1.5, style: .continuous)
                    .fill(accent)
                    .frame(width: 7, height: 7)

                Text(day.displayDate)
                    .font(.system(size: 11, weight: .semibold, design: .rounded))
                    .foregroundStyle(.primary)
                    .lineLimit(1)
            }

            if day.hasActivity {
                VStack(alignment: .leading, spacing: 2) {
                    stat(
                        day.transcriptionCount.formatted(),
                        day.transcriptionCount == 1 ? "transcription" : "transcriptions"
                    )
                    stat(day.wordCount.formatted(), "words")

                    if day.durationSeconds >= 1 {
                        stat(durationText, "recorded")
                    }
                }
            } else {
                Text("No transcriptions")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(.tertiary)
                    .lineLimit(1)
            }
        }
        // Lock the stack to its ideal size *before* the padding and glass wrap
        // it, so the background can never be laid out against a width the text
        // then overflows.
        .fixedSize()
        .padding(.horizontal, 11)
        .padding(.vertical, 9)
        .background(GlassPanelBackground(cornerRadius: 10))
        .shadow(color: .black.opacity(0.34), radius: 12, y: 4)
    }

    private func stat(_ value: String, _ label: String) -> some View {
        HStack(spacing: 4) {
            Text(value)
                .font(.system(size: 11, weight: .semibold, design: .rounded))
                .foregroundStyle(.primary.opacity(0.92))

            Text(label)
                .font(.system(size: 10, weight: .medium))
                .foregroundStyle(.secondary)
        }
        .lineLimit(1)
    }

    private var durationText: String {
        let total = Int(day.durationSeconds.rounded())
        let minutes = total / 60

        if minutes >= 60 {
            return "\(minutes / 60)h \(minutes % 60)m"
        }
        return minutes > 0 ? "\(minutes)m \(total % 60)s" : "\(total)s"
    }
}

/// Liquid Glass surface with the pre-Tahoe material fallback, matching
/// `MeetingReminderGlassNotchShape`.
private struct GlassPanelBackground: View {
    let cornerRadius: CGFloat

    var body: some View {
        let shape = RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)

        if #available(macOS 26.0, *) {
            shape
                .fill(.clear)
                .glassEffect(.regular.tint(.white.opacity(0.045)), in: shape)
                .overlay(shape.stroke(.white.opacity(0.16), lineWidth: 0.6))
        } else {
            shape
                .fill(.ultraThinMaterial)
                .overlay(shape.fill(.white.opacity(0.055)))
                .overlay(shape.stroke(.white.opacity(0.16), lineWidth: 0.6))
        }
    }
}

struct ActivityInlineStat: View {
    let value: String
    let label: String

    var body: some View {
        VStack(alignment: .trailing, spacing: 2) {
            Text(value)
                .font(.system(size: 15, weight: .bold, design: .rounded))
                .foregroundStyle(.primary)

            Text(label)
                .font(.system(size: 9, weight: .medium))
                .foregroundStyle(.tertiary)
        }
    }
}

struct RhythmMetric: View {
    let value: String
    let label: String
    let tint: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(value)
                .font(.system(size: 16, weight: .bold, design: .rounded))
                .foregroundStyle(.primary)
                .lineLimit(1)
                .minimumScaleFactor(0.72)

            HStack(spacing: 4) {
                Circle()
                    .fill(tint.opacity(0.75))
                    .frame(width: 5, height: 5)

                Text(label)
                    .font(.system(size: 9, weight: .medium))
                    .foregroundStyle(.tertiary)
                    .lineLimit(1)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

// MARK: - Overview Mini Box

struct OverviewMiniBox<Content: View>: View {
    let title: String
    let icon: String
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.system(size: 8, weight: .medium))
                    .foregroundStyle(.tertiary)

                Text(title.uppercased())
                    .font(.system(size: 8, weight: .semibold))
                    .foregroundStyle(.tertiary)
                    .tracking(0.4)
            }

            content
        }
        .padding(10)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .leading)
        .background(.thinMaterial, in: .rect(cornerRadius: 10, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(.white.opacity(0.04), lineWidth: 0.5)
        )
    }
}

// MARK: - WPM Sparkline

struct WPMSparkline: View {
    let data: [Double]
    var animated: Bool = true

    var body: some View {
        GeometryReader { geo in
            let minVal = (data.min() ?? 0) * 0.9
            let maxVal = (data.max() ?? 1) * 1.1
            let range = max(maxVal - minVal, 1)
            let stepX = geo.size.width / CGFloat(max(data.count - 1, 1))

            // Line
            Path { path in
                for (i, val) in data.enumerated() {
                    let x = CGFloat(i) * stepX
                    let targetY = geo.size.height - (CGFloat(val - minVal) / CGFloat(range) * geo.size.height)
                    let y = animated ? targetY : geo.size.height * 0.5
                    if i == 0 {
                        path.move(to: CGPoint(x: x, y: y))
                    } else {
                        path.addLine(to: CGPoint(x: x, y: y))
                    }
                }
            }
            .trim(from: 0, to: animated ? 1 : 0)
            .stroke(Color.orange.opacity(0.8), lineWidth: 1.5)

            // Dots
            if animated {
                ForEach(Array(data.enumerated()), id: \.offset) { i, val in
                    let x = CGFloat(i) * stepX
                    let y = geo.size.height - (CGFloat(val - minVal) / CGFloat(range) * geo.size.height)

                    Circle()
                        .fill(Color.orange)
                        .frame(width: 4, height: 4)
                        .position(x: x, y: y)
                }
            }
        }
    }
}

// MARK: - Quick Action Pill

struct QuickActionPill: View {
    let icon: String
    let label: String
    let action: () -> Void

    @State private var isHovered = false
    @State private var isPressed = false

    var body: some View {
        HStack(spacing: 5) {
            Image(systemName: icon)
                .font(.system(size: 10, weight: .medium))
                .foregroundColor(isHovered ? .white : .white.opacity(0.55))

            Text(label)
                .font(.system(size: 11, weight: .medium))
                .foregroundColor(isHovered ? .white : .white.opacity(0.55))
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(
            Capsule(style: .continuous)
                .fill(.white.opacity(isHovered ? 0.08 : 0.04))
                .overlay(
                    Capsule(style: .continuous)
                        .strokeBorder(.white.opacity(isHovered ? 0.14 : 0.06), lineWidth: 0.5)
                )
        )
        .scaleEffect(isPressed ? 0.97 : 1.0)
        .animation(.easeOut(duration: 0.1), value: isPressed)
        .onHover { hovering in
            withAnimation(.easeOut(duration: 0.12)) {
                isHovered = hovering
            }
        }
        .onTapGesture {
            isPressed = true
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                isPressed = false
            }
            action()
        }
        .contentShape(Capsule())
    }
}

// MARK: - Hero Stat Card

struct HeroStatCard: View {
    let value: String
    let label: String
    let icon: String
    let tint: Color

    @State private var isHovered = false

    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: icon)
                .font(.system(size: 14, weight: .medium))
                .foregroundStyle(tint.opacity(0.8))

            Text(value)
                .font(.system(size: 22, weight: .bold, design: .rounded))
                .foregroundStyle(.primary)
                .lineLimit(1)
                .minimumScaleFactor(0.7)

            Text(label)
                .font(.system(size: 10, weight: .medium))
                .foregroundStyle(.secondary)
                .lineLimit(1)
        }
        .padding(.vertical, 14)
        .padding(.horizontal, 8)
        .frame(maxWidth: .infinity)
        .background(.thinMaterial, in: .rect(cornerRadius: 12, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(.white.opacity(isHovered ? 0.1 : 0.04), lineWidth: 0.5)
        )
        .scaleEffect(isHovered ? 1.02 : 1.0)
        .animation(.easeOut(duration: 0.15), value: isHovered)
        .onHover { hovering in isHovered = hovering }
    }
}

// MARK: - Dashboard Card (Container)

struct DashboardCard2<Content: View>: View {
    let title: String
    let icon: String
    let fixedHeight: CGFloat?
    let content: Content

    init(
        title: String,
        icon: String,
        fixedHeight: CGFloat? = nil,
        @ViewBuilder content: () -> Content
    ) {
        self.title = title
        self.icon = icon
        self.fixedHeight = fixedHeight
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 5) {
                Image(systemName: icon)
                    .font(.system(size: 9, weight: .medium))
                    .foregroundStyle(.secondary)

                Text(title.uppercased())
                    .font(.system(size: 9, weight: .semibold, design: .rounded))
                    .foregroundStyle(.secondary)
                    .tracking(0.5)

                Spacer()
            }

            content
        }
        .padding(12)
        .frame(
            maxWidth: .infinity,
            minHeight: fixedHeight,
            maxHeight: fixedHeight ?? .infinity,
            alignment: .topLeading
        )
        .background(.thinMaterial, in: .rect(cornerRadius: 12, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(.white.opacity(0.04), lineWidth: 0.5)
        )
    }
}

// MARK: - Mini Stat

struct MiniStat: View {
    let value: String
    let label: String

    var body: some View {
        VStack(spacing: 2) {
            Text(value)
                .font(.system(size: 14, weight: .bold, design: .rounded))
                .foregroundStyle(.primary)

            Text(label)
                .font(.system(size: 8, weight: .medium))
                .foregroundStyle(.tertiary)
        }
    }
}
