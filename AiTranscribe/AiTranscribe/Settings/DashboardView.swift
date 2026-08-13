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
                    Task {
                        let entries = await appState.loadAllHistoryEntries()
                        await statsManager.resetAPICosts(from: entries, sessions: sessionManager.sessions)
                    }
                } label: {
                    Text("Reset")
                        .font(.system(size: 9, weight: .semibold))
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3)
                        .background(.white.opacity(0.05), in: Capsule(style: .continuous))
                }
                .buttonStyle(.plain)
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

    private let weekCount = 52
    private let dayCount = 7
    private let labelWidth: CGFloat = 25
    private let cellSpacing: CGFloat = 4

    private var weeks: [[DashboardActivityDay]] {
        guard !days.isEmpty else { return [] }
        return stride(from: 0, to: days.count, by: dayCount).map { start in
            Array(days[start..<min(start + dayCount, days.count)])
        }
    }

    var body: some View {
        let weekGroups = weeks

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
                        monthLabels(cellSize: cellSize, weeks: weekGroups)
                            .frame(width: labelWidth + gridWidth, height: 14, alignment: .leading)

                        HStack(alignment: .top, spacing: 7) {
                            weekdayLabels(cellSize: cellSize)

                            HStack(alignment: .top, spacing: cellSpacing) {
                                ForEach(Array(weekGroups.prefix(weekCount).enumerated()), id: \.offset) { weekIndex, week in
                                    VStack(spacing: cellSpacing) {
                                        ForEach(Array(week.enumerated()), id: \.element.id) { dayIndex, day in
                                            ContributionCell(
                                                day: day,
                                                maxCount: maxCount,
                                                size: cellSize,
                                                animated: animated,
                                                index: weekIndex * dayCount + dayIndex
                                            )
                                        }
                                    }
                                }
                            }
                        }

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
    }

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

    private func monthLabels(cellSize: CGFloat, weeks: [[DashboardActivityDay]]) -> some View {
        let labels = contributionMonthLabels(weeks: weeks)

        return ZStack(alignment: .topLeading) {
            ForEach(labels, id: \.weekIndex) { item in
                Text(item.label)
                    .font(.system(size: 9, weight: .medium))
                    .foregroundStyle(.secondary)
                    .offset(x: labelWidth + 7 + CGFloat(item.weekIndex) * (cellSize + cellSpacing))
            }
        }
    }

    private func contributionMonthLabels(weeks: [[DashboardActivityDay]]) -> [(weekIndex: Int, label: String)] {
        let calendar = Calendar.current
        let formatter = DateFormatter()
        formatter.dateFormat = "MMM"

        return weeks.prefix(weekCount).enumerated().compactMap { index, week in
            guard let firstDay = week.first else { return nil }
            let day = calendar.component(.day, from: firstDay.date)
            if index == 0 || day <= 7 {
                return (index, formatter.string(from: firstDay.date))
            }
            return nil
        }
    }
}

struct ContributionCell: View {
    let day: DashboardActivityDay
    let maxCount: Int
    let size: CGFloat
    let animated: Bool
    let index: Int

    @State private var isHovered = false

    var body: some View {
        RoundedRectangle(cornerRadius: 2.5, style: .continuous)
            .fill(fillColor)
            .frame(width: size, height: size)
            .overlay(
                RoundedRectangle(cornerRadius: 2.5, style: .continuous)
                    .strokeBorder(.white.opacity(day.hasActivity ? 0.08 : 0.025), lineWidth: 0.5)
            )
            .scaleEffect(isHovered ? 1.28 : (animated ? 1 : 0.65))
            .opacity(animated ? 1 : 0)
            .shadow(color: day.hasActivity ? fillColor.opacity(isHovered ? 0.45 : 0.18) : .clear, radius: isHovered ? 7 : 3)
            .animation(.spring(duration: 0.18, bounce: 0.16), value: isHovered)
            .animation(.spring(duration: 0.55, bounce: 0.12).delay(Double(index % 42) * 0.01), value: animated)
            .onHover { hovering in
                isHovered = hovering
            }
            .help(helpText)
            .accessibilityLabel(accessibilityText)
    }

    private var fillColor: Color {
        guard day.hasActivity, maxCount > 0 else {
            return Color.white.opacity(0.055)
        }

        let ratio = Double(day.transcriptionCount) / Double(maxCount)
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

    private var helpText: String {
        guard day.hasActivity else {
            return "\(day.displayDate)\nNo transcriptions"
        }

        let transcriptionLabel = day.transcriptionCount == 1 ? "transcription" : "transcriptions"
        return "\(day.displayDate)\n\(day.transcriptionCount) \(transcriptionLabel)\n\(day.wordCount) words"
    }

    private var accessibilityText: String {
        day.hasActivity ? helpText.replacingOccurrences(of: "\n", with: ", ") : helpText
    }

    static func color(for level: Int, maxLevel: Int) -> Color {
        guard level > 0 else { return Color.white.opacity(0.055) }
        let ratio = Double(level) / Double(max(maxLevel, 1))
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
