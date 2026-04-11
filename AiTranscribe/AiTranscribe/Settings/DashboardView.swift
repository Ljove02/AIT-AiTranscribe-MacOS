import SwiftUI

// MARK: - Dashboard View

struct DashboardView: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var sessionManager: SessionManager
    @StateObject private var statsManager = DashboardStatsManager()

    let hasAnimated: Bool
    let onAnimated: () -> Void
    let onNavigateToHistory: (UUID?) -> Void
    let onNavigateToSessions: (UUID?) -> Void

    @State private var appeared: Bool
    @State private var chartsAnimated = false

    init(
        hasAnimated: Bool,
        onAnimated: @escaping () -> Void,
        onNavigateToHistory: @escaping (UUID?) -> Void,
        onNavigateToSessions: @escaping (UUID?) -> Void
    ) {
        self.hasAnimated = hasAnimated
        self.onAnimated = onAnimated
        self.onNavigateToHistory = onNavigateToHistory
        self.onNavigateToSessions = onNavigateToSessions
        _appeared = State(initialValue: hasAnimated)
    }

    var body: some View {
        ScrollView {
            VStack(spacing: 0) {
                if statsManager.isLoading && !statsManager.hasLoaded {
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
            let entries = await appState.loadAllHistoryEntries()
            await statsManager.computeStats(from: entries)
            // Trigger chart grow-in — slow enough to appreciate
            try? await Task.sleep(for: .milliseconds(250))
            withAnimation(.spring(duration: 1.2, bounce: 0.08)) {
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
        VStack(spacing: 16) {
            Spacer().frame(height: 200)
            ProgressView()
                .controlSize(.small)
            Text("Crunching your stats…")
                .font(.system(size: 13))
                .foregroundStyle(.secondary)
            Spacer()
        }
        .frame(maxWidth: .infinity)
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

            Spacer().frame(height: 16)

            heroStatsRow
                .staggerIn(index: 1, appeared: appeared)

            Spacer().frame(height: 12)

            row2_topCards
                .staggerIn(index: 2, appeared: appeared)

            Spacer().frame(height: 12)

            row3_weeklyAndInfo
                .staggerIn(index: 3, appeared: appeared)

            Spacer().frame(height: 12)

            row4_charts
                .staggerIn(index: 4, appeared: appeared)

            Spacer(minLength: 16)
        }
    }

    // MARK: - Header

    private var headerSection: some View {
        VStack(spacing: 12) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Dashboard")
                        .font(.system(size: 22, weight: .bold, design: .rounded))

                    Text("\(statsManager.stats.totalTranscriptions) transcriptions")
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }

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

                Spacer()
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
                value: "\(Int(statsManager.stats.averageWPM))",
                label: "Avg WPM",
                icon: "gauge.with.needle",
                tint: .orange
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

    // MARK: - Row 2: Top Model + Top Word + Activity (equal thirds)

    private var row2_topCards: some View {
        HStack(spacing: 10) {
            topModelCard.frame(maxWidth: .infinity)
            topWordCard.frame(maxWidth: .infinity)
            activityCard.frame(maxWidth: .infinity)
        }
        .fixedSize(horizontal: false, vertical: true)
    }

    private var topModelCard: some View {
        DashboardCard2(title: "Top Model", icon: "cube.box.fill") {
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
        DashboardCard2(title: "Top Words", icon: "textformat") {
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

    private var activityCard: some View {
        DashboardCard2(title: "Activity", icon: "chart.bar.fill") {
            VStack(spacing: 10) {
                if !statsManager.stats.monthlyBreakdown.isEmpty {
                    monthlyBarsView
                }

                HStack(spacing: 0) {
                    MiniStat(value: "\(statsManager.stats.longestStreak)", label: "Streak")
                    Spacer()
                    MiniStat(value: "\(statsManager.stats.totalActiveDays)", label: "Days")
                    if statsManager.stats.currentStreak > 0 {
                        Spacer()
                        MiniStat(value: "\(statsManager.stats.currentStreak)", label: "Current")
                    }
                }
            }
        }
    }

    private var monthlyBarsView: some View {
        let maxCount = statsManager.stats.monthlyBreakdown.map(\.count).max() ?? 1
        return HStack(alignment: .bottom, spacing: 6) {
            ForEach(Array(statsManager.stats.monthlyBreakdown.enumerated()), id: \.offset) { _, item in
                VStack(spacing: 3) {
                    RoundedRectangle(cornerRadius: 3, style: .continuous)
                        .fill(Color.accentColor.opacity(0.7))
                        .frame(height: chartsAnimated ? max(4, CGFloat(item.count) / CGFloat(maxCount) * 36) : 2)

                    Text(item.month)
                        .font(.system(size: 8, weight: .medium))
                        .foregroundStyle(.tertiary)
                }
                .frame(maxWidth: .infinity)
            }
        }
        .frame(height: 48)
    }

    // MARK: - Row 3: Weekly Pattern + 4 Overview Boxes (equal halves)

    private var row3_weeklyAndInfo: some View {
        HStack(spacing: 10) {
            weeklyPatternCard.frame(maxWidth: .infinity)
            overviewGrid.frame(maxWidth: .infinity)
        }
        .fixedSize(horizontal: false, vertical: true)
    }

    private var weeklyPatternCard: some View {
        DashboardCard2(title: "Weekly Pattern", icon: "calendar") {
            let maxCount = statsManager.stats.dayOfWeekCounts.map(\.count).max() ?? 1
            HStack(alignment: .bottom, spacing: 5) {
                ForEach(Array(statsManager.stats.dayOfWeekCounts.enumerated()), id: \.offset) { idx, item in
                    VStack(spacing: 4) {
                        Text("\(item.count)")
                            .font(.system(size: 8, weight: .medium, design: .monospaced))
                            .foregroundStyle(.tertiary)

                        RoundedRectangle(cornerRadius: 3, style: .continuous)
                            .fill(weekdayColor(idx).opacity(0.7))
                            .frame(height: chartsAnimated ? max(6, CGFloat(item.count) / CGFloat(maxCount) * 50) : 2)

                        Text(item.day)
                            .font(.system(size: 9, weight: .medium))
                            .foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity)
                }
            }
            .frame(height: 76)
        }
    }

    private func weekdayColor(_ index: Int) -> Color {
        index >= 5 ? .orange : .accentColor
    }

    /// 4 separate mini boxes in a 2×2 grid
    private var overviewGrid: some View {
        VStack(spacing: 8) {
            HStack(spacing: 8) {
                // Using Since
                OverviewMiniBox(
                    title: "Using Since",
                    icon: "calendar.badge.clock"
                ) {
                    if let since = statsManager.stats.usingSinceDate {
                        Text(since, style: .date)
                            .font(.system(size: 12, weight: .semibold, design: .rounded))
                            .lineLimit(1)
                            .minimumScaleFactor(0.8)

                        Text(timeAgoString(from: since))
                            .font(.system(size: 9))
                            .foregroundStyle(.secondary)
                    }
                }

                // Peak Hour
                OverviewMiniBox(
                    title: "Peak Hour",
                    icon: "clock.fill"
                ) {
                    if let peak = statsManager.stats.hourlyBreakdown.max(by: { $0.count < $1.count }), peak.count > 0 {
                        Text(formatHour(peak.hour))
                            .font(.system(size: 12, weight: .semibold, design: .rounded))

                        Text("\(peak.count) transcriptions")
                            .font(.system(size: 9))
                            .foregroundStyle(.secondary)
                    }
                }
            }

            HStack(spacing: 8) {
                // Busiest Day
                OverviewMiniBox(
                    title: "Busiest Day",
                    icon: "flame.fill"
                ) {
                    if let busiest = statsManager.stats.busiestDay {
                        Text(formatDateString(busiest.date))
                            .font(.system(size: 12, weight: .semibold, design: .rounded))

                        Text("\(busiest.count) transcriptions")
                            .font(.system(size: 9))
                            .foregroundStyle(.secondary)
                    }
                }

                // Total Spoken
                OverviewMiniBox(
                    title: "Total Spoken",
                    icon: "waveform"
                ) {
                    Text(formatDuration(statsManager.stats.totalDurationSeconds))
                        .font(.system(size: 12, weight: .semibold, design: .rounded))

                    Text("recording time")
                        .font(.system(size: 9))
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    // MARK: - Row 4: Hourly + WPM Trend + Length

    private var row4_charts: some View {
        HStack(spacing: 10) {
            hourlyCard.frame(maxWidth: .infinity)
            wpmTrendCard.frame(maxWidth: .infinity)
            lengthCard.frame(maxWidth: .infinity)
        }
        .fixedSize(horizontal: false, vertical: true)
    }

    private var hourlyCard: some View {
        DashboardCard2(title: "Hourly Activity", icon: "clock") {
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
        DashboardCard2(title: "WPM Trend", icon: "gauge.with.needle") {
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

    private var lengthCard: some View {
        DashboardCard2(title: "Length Distribution", icon: "ruler") {
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

    private func formatDateString(_ dateStr: String) -> String {
        let inputFormatter = DateFormatter()
        inputFormatter.dateFormat = "yyyy-MM-dd"
        guard let date = inputFormatter.date(from: dateStr) else { return dateStr }
        let outputFormatter = DateFormatter()
        outputFormatter.dateFormat = "MMM d"
        return outputFormatter.string(from: date)
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
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(10)
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
    @ViewBuilder let content: Content

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
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
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
