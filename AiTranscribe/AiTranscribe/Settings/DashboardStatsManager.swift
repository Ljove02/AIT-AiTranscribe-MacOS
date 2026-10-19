/*
 DashboardStatsManager.swift
 ============================

 Computes and caches dashboard statistics from transcription history.

 - Loaded once on first dashboard visit, cached for the session
 - Incrementally updated when new transcriptions are added
 - All heavy work done off-main via async
 */

import Foundation
import Combine

nonisolated struct DashboardActivityDay: Equatable, Identifiable, Sendable {
    let date: Date
    let key: String
    let transcriptionCount: Int
    let wordCount: Int
    let durationSeconds: Double
    let displayDate: String

    var id: String { key }
    var hasActivity: Bool { transcriptionCount > 0 }
}

/// All the stats the dashboard needs, computed once and cached.
nonisolated struct DashboardStats: Equatable, Sendable {
    // Hero numbers
    var totalTranscriptions: Int = 0
    var totalWords: Int = 0
    var totalDurationSeconds: Double = 0
    var averageWPM: Double = 0
    var timeSavedMinutes: Double = 0  // vs typing at 40 WPM
    var estimatedAPICostUSD: Double = 0
    var apiCostItemCount: Int = 0

    // Activity
    var usingSinceDate: Date?
    var totalActiveDays: Int = 0
    var longestStreak: Int = 0
    var currentStreak: Int = 0
    var busiestDay: (date: String, count: Int)?
    var monthlyBreakdown: [(month: String, count: Int)] = []
    var activityDays: [DashboardActivityDay] = []
    var activityMaxCount: Int = 0

    // Charts
    var dayOfWeekCounts: [(day: String, count: Int)] = []  // Mon-Sun
    var hourlyBreakdown: [(hour: Int, count: Int)] = []     // 0-23

    // Insights
    var favoriteModel: (name: String, percentage: Int)?
    var allModels: [(name: String, count: Int, percentage: Int)] = []
    var topWords: [(word: String, count: Int)] = []
    var averageWordsPerTranscription: Int = 0
    var longestTranscriptionWords: Int = 0

    // WPM trend (weekly averages)
    var weeklyWPM: [(week: String, wpm: Double)] = []

    // Transcription length buckets
    var lengthDistribution: [(label: String, count: Int)] = []

    // For Equatable — ignore tuples
    static func == (lhs: DashboardStats, rhs: DashboardStats) -> Bool {
        lhs.totalTranscriptions == rhs.totalTranscriptions &&
        lhs.totalWords == rhs.totalWords
    }
}

/// Manages stats computation and caching for the dashboard.
@MainActor
final class DashboardStatsManager: ObservableObject {
    @Published var stats = DashboardStats()
    @Published var isLoading = false
    @Published var hasLoaded = false

    private var apiCostResetDate = DashboardStatsManager.defaultAPICostResetDate()

    /// Compute stats from an array of transcription entries.
    /// Call this once when the dashboard first appears.
    func computeStats(
        from entries: [TranscriptionEntry],
        sessions: [Session] = [],
        apiCostResetDate: Date? = DashboardStatsManager.defaultAPICostResetDate()
    ) async {
        guard !entries.isEmpty || !sessions.isEmpty else {
            self.apiCostResetDate = apiCostResetDate
            hasLoaded = true
            return
        }

        self.apiCostResetDate = apiCostResetDate
        isLoading = true

        // Do heavy work off main thread
        let computed = await Task.detached(priority: .userInitiated) {
            Self.calculate(entries: entries, sessions: sessions, apiCostResetDate: apiCostResetDate)
        }.value

        stats = computed
        isLoading = false
        hasLoaded = true
    }

    func resetAPICosts(from entries: [TranscriptionEntry], sessions: [Session] = []) async {
        let resetDate = Date()
        UserDefaults.standard.set(resetDate.timeIntervalSince1970, forKey: Self.apiCostResetTimestampKey)
        UserDefaults.standard.set(true, forKey: Self.apiCostPricingMigrationKey)
        await computeStats(from: entries, sessions: sessions, apiCostResetDate: resetDate)
    }

    /// Lightweight update when a single new entry is added.
    func incorporate(entry: TranscriptionEntry) {
        stats.totalTranscriptions += 1
        stats.totalWords += entry.wordCount
        stats.totalDurationSeconds += entry.duration

        let totalWPMs = stats.averageWPM * Double(stats.totalTranscriptions - 1)
        if entry.wordsPerMinute > 0 {
            stats.averageWPM = (totalWPMs + entry.wordsPerMinute) / Double(stats.totalTranscriptions)
        }

        let typingTime = Double(stats.totalWords) / 40.0
        let speakingTime = stats.totalDurationSeconds / 60.0
        stats.timeSavedMinutes = typingTime - speakingTime

        stats.averageWordsPerTranscription = stats.totalWords / max(stats.totalTranscriptions, 1)
        if entry.wordCount > stats.longestTranscriptionWords {
            stats.longestTranscriptionWords = entry.wordCount
        }

        if let cost = entry.estimatedCostUSD,
           Self.shouldIncludeAPICost(date: entry.timestamp, resetDate: apiCostResetDate) {
            stats.estimatedAPICostUSD += cost
            stats.apiCostItemCount += 1
        }
    }

    // MARK: - Heavy Computation (off-main)

    private nonisolated static let apiCostResetTimestampKey = "dashboardAPICostResetTimestamp"
    private nonisolated static let apiCostPricingMigrationKey = "dashboardAPICostPricingMigration20260622"
    private nonisolated static let activityWeekCount = 52

    private nonisolated static func defaultAPICostResetDate() -> Date? {
        let defaults = UserDefaults.standard
        let storedTimestamp = defaults.double(forKey: apiCostResetTimestampKey)
        if storedTimestamp > 0 {
            return Date(timeIntervalSince1970: storedTimestamp)
        }

        guard !defaults.bool(forKey: apiCostPricingMigrationKey) else {
            return nil
        }

        let resetDate = Date()
        defaults.set(resetDate.timeIntervalSince1970, forKey: apiCostResetTimestampKey)
        defaults.set(true, forKey: apiCostPricingMigrationKey)
        return resetDate
    }

    private nonisolated static func shouldIncludeAPICost(date: Date?, resetDate: Date?) -> Bool {
        guard let resetDate else { return true }
        guard let date else { return false }
        return date >= resetDate
    }

    private nonisolated static func calculate(
        entries: [TranscriptionEntry],
        sessions: [Session],
        apiCostResetDate: Date?
    ) -> DashboardStats {
        var s = DashboardStats()

        s.totalTranscriptions = entries.count
        s.totalWords = entries.reduce(0) { $0 + $1.wordCount }
        s.totalDurationSeconds = entries.reduce(0) { $0 + $1.duration }

        let validWPMs = entries.filter { $0.wordsPerMinute > 0 }.map(\.wordsPerMinute)
        s.averageWPM = validWPMs.isEmpty ? 0 : validWPMs.reduce(0, +) / Double(validWPMs.count)

        let typingTime = Double(s.totalWords) / 40.0
        let speakingTime = s.totalDurationSeconds / 60.0
        s.timeSavedMinutes = typingTime - speakingTime

        s.averageWordsPerTranscription = s.totalWords / max(entries.count, 1)
        s.longestTranscriptionWords = entries.map(\.wordCount).max() ?? 0
        let pricedEntries = entries.filter {
            $0.estimatedCostUSD != nil && shouldIncludeAPICost(date: $0.timestamp, resetDate: apiCostResetDate)
        }
        s.estimatedAPICostUSD = pricedEntries.compactMap(\.estimatedCostUSD).reduce(0, +)
        s.apiCostItemCount = pricedEntries.count

        for session in sessions {
            if let cost = session.transcriptionEstimatedCostUSD,
               shouldIncludeAPICost(date: session.createdAt, resetDate: apiCostResetDate) {
                s.estimatedAPICostUSD += cost
                s.apiCostItemCount += 1
            }
            for summary in session.summaries.values {
                if let cost = summary.estimatedCostUSD,
                   shouldIncludeAPICost(date: summary.generatedAt, resetDate: apiCostResetDate) {
                    s.estimatedAPICostUSD += cost
                    s.apiCostItemCount += 1
                }
            }
        }

        // Date analysis
        let calendar = Calendar.current
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd"
        dateFormatter.locale = Locale(identifier: "en_US_POSIX")

        if let firstEntry = entries.min(by: { $0.timestamp < $1.timestamp }) {
            s.usingSinceDate = firstEntry.timestamp
        }

        let activity = buildActivityDays(from: entries, calendar: calendar, dateFormatter: dateFormatter)
        s.activityDays = activity.days
        s.activityMaxCount = activity.maxCount
        s.totalActiveDays = activity.activeKeys.count
        s.longestStreak = longestStreak(from: activity.activeDates, calendar: calendar)
        s.currentStreak = currentStreak(from: activity.activeKeys, calendar: calendar, dateFormatter: dateFormatter)
        if let busiest = activity.dayCounts.max(by: {
            if $0.value.transcriptionCount == $1.value.transcriptionCount {
                return $0.value.date < $1.value.date
            }
            return $0.value.transcriptionCount < $1.value.transcriptionCount
        }) {
            s.busiestDay = (busiest.key, busiest.value.transcriptionCount)
        }

        // Monthly breakdown (last 4 months)
        let monthFormatter = DateFormatter()
        monthFormatter.dateFormat = "yyyy-MM"
        let monthDisplayFormatter = DateFormatter()
        monthDisplayFormatter.dateFormat = "MMM"

        var monthlyCounts: [String: (display: String, count: Int, sortKey: String)] = [:]
        for entry in entries {
            let key = monthFormatter.string(from: entry.timestamp)
            let display = monthDisplayFormatter.string(from: entry.timestamp)
            monthlyCounts[key, default: (display, 0, key)].count += 1
        }
        s.monthlyBreakdown = monthlyCounts.values
            .sorted { $0.sortKey < $1.sortKey }
            .suffix(4)
            .map { ($0.display, $0.count) }

        // Day of week breakdown
        let dowNames = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        var dowCounts: [Int: Int] = [:]
        for entry in entries {
            // weekday: 1=Sun, 2=Mon, ..., 7=Sat → convert to 0=Mon..6=Sun
            let wd = calendar.component(.weekday, from: entry.timestamp)
            let idx = (wd + 5) % 7  // Mon=0, Tue=1, ..., Sun=6
            dowCounts[idx, default: 0] += 1
        }
        s.dayOfWeekCounts = (0..<7).map { (dowNames[$0], dowCounts[$0] ?? 0) }

        // Hourly breakdown (all 24 hours)
        var hourCounts: [Int: Int] = [:]
        for entry in entries {
            let hour = calendar.component(.hour, from: entry.timestamp)
            hourCounts[hour, default: 0] += 1
        }
        s.hourlyBreakdown = (0..<24).map { ($0, hourCounts[$0] ?? 0) }

        // Model breakdown
        var modelCounts: [String: Int] = [:]
        for entry in entries {
            modelCounts[entry.modelName, default: 0] += 1
        }
        if let topModel = modelCounts.max(by: { $0.value < $1.value }) {
            let pct = Int(Double(topModel.value) / Double(entries.count) * 100)
            s.favoriteModel = (topModel.key, pct)
        }
        s.allModels = modelCounts
            .sorted { $0.value > $1.value }
            .map { ($0.key, $0.value, Int(Double($0.value) / Double(entries.count) * 100)) }

        // Top words (exclude common stop words)
        let stopWords: Set<String> = [
            "the","a","an","is","are","was","were","be","been","being","have","has","had",
            "do","does","did","will","would","shall","should","may","might","must","can",
            "could","i","me","my","you","your","he","him","his","she","her","it","its",
            "we","our","they","them","their","this","that","these","those","in","on","at",
            "to","for","of","and","or","but","not","with","from","by","as","if","so","no",
            "up","out","just","then","than","into","over","after","before","about","very",
            "also","back","only","now","here","there","when","where","how","all","each",
            "every","both","few","more","most","some","any","other","what","which","who",
            "whom","like","um","uh","oh","ah","yeah","okay","know","going","think","want",
            "get","got","make","well","right","really","because","actually","something",
            "things","kind","mean","need","see","thing","way","time","one","two","three",
            "said","say","go","come","take","look","don't","that's","it's","i'm","i've",
            "we've","they're","you're","can't","won't","didn't","doesn't","isn't","aren't",
            "wasn't","weren't","haven't","hasn't","hadn't","shouldn't","wouldn't","couldn't",
            "let","us","much","still","even","too","many","such","own","been","through",
            "being","same","an","could","should","would","much","still","between","down",
            "off","then","once","again","new","already","while","though"
        ]

        var wordFreq: [String: Int] = [:]
        for entry in entries {
            let sample = String(entry.text.prefix(500)).lowercased()
            let words = sample.split(whereSeparator: { !$0.isLetter && $0 != "'" })
                .map(String.init)
                .filter { $0.count > 2 && !stopWords.contains($0) }
            for word in words {
                wordFreq[word, default: 0] += 1
            }
        }
        s.topWords = wordFreq
            .sorted { $0.value > $1.value }
            .prefix(5)
            .map { ($0.key, $0.value) }

        // Weekly WPM trend (last 8 weeks)
        let weekFormatter = DateFormatter()
        weekFormatter.dateFormat = "yyyy-'W'ww"
        let weekDisplayFormatter = DateFormatter()
        weekDisplayFormatter.dateFormat = "'W'ww"

        var weeklyWPMs: [String: (display: String, wpms: [Double], sortKey: String)] = [:]
        for entry in entries where entry.wordsPerMinute > 0 {
            let key = weekFormatter.string(from: entry.timestamp)
            let display = weekDisplayFormatter.string(from: entry.timestamp)
            weeklyWPMs[key, default: (display, [], key)].wpms.append(entry.wordsPerMinute)
        }
        s.weeklyWPM = weeklyWPMs.values
            .sorted { $0.sortKey < $1.sortKey }
            .suffix(8)
            .map { ($0.display, $0.wpms.reduce(0, +) / Double($0.wpms.count)) }

        // Transcription length distribution
        var short = 0, medium = 0, long = 0, veryLong = 0
        for entry in entries {
            switch entry.wordCount {
            case 0..<50:    short += 1
            case 50..<200:  medium += 1
            case 200..<500: long += 1
            default:        veryLong += 1
            }
        }
        s.lengthDistribution = [
            ("< 50w", short),
            ("50-200w", medium),
            ("200-500w", long),
            ("500w+", veryLong)
        ]

        return s
    }

    private nonisolated static func buildActivityDays(
        from entries: [TranscriptionEntry],
        calendar: Calendar,
        dateFormatter: DateFormatter
    ) -> (
        days: [DashboardActivityDay],
        maxCount: Int,
        activeKeys: Set<String>,
        activeDates: [Date],
        dayCounts: [String: (date: Date, transcriptionCount: Int, wordCount: Int, durationSeconds: Double)]
    ) {
        var dayCounts: [String: (date: Date, transcriptionCount: Int, wordCount: Int, durationSeconds: Double)] = [:]
        let displayFormatter = DateFormatter()
        displayFormatter.dateFormat = "MMM d, yyyy"
        displayFormatter.locale = Locale(identifier: "en_US_POSIX")

        for entry in entries {
            let day = calendar.startOfDay(for: entry.timestamp)
            let key = dateFormatter.string(from: day)
            var aggregate = dayCounts[key] ?? (day, 0, 0, 0)
            aggregate.transcriptionCount += 1
            aggregate.wordCount += entry.wordCount
            aggregate.durationSeconds += entry.duration
            dayCounts[key] = aggregate
        }

        let today = calendar.startOfDay(for: Date())
        let weekdayIndex = (calendar.component(.weekday, from: today) + 5) % 7
        let currentWeekStart = calendar.date(byAdding: .day, value: -weekdayIndex, to: today) ?? today
        let firstGridDay = calendar.date(
            byAdding: .day,
            value: -(activityWeekCount - 1) * 7,
            to: currentWeekStart
        ) ?? currentWeekStart

        let gridDayCount = activityWeekCount * 7
        let days = (0..<gridDayCount).compactMap { offset -> DashboardActivityDay? in
            guard let date = calendar.date(byAdding: .day, value: offset, to: firstGridDay) else {
                return nil
            }
            let key = dateFormatter.string(from: date)
            let aggregate = dayCounts[key]
            return DashboardActivityDay(
                date: date,
                key: key,
                transcriptionCount: aggregate?.transcriptionCount ?? 0,
                wordCount: aggregate?.wordCount ?? 0,
                durationSeconds: aggregate?.durationSeconds ?? 0,
                displayDate: displayFormatter.string(from: date)
            )
        }

        let activeKeys = Set(dayCounts.keys)
        let activeDates = dayCounts.values
            .map(\.date)
            .sorted()

        return (
            days,
            days.map(\.transcriptionCount).max() ?? 0,
            activeKeys,
            activeDates,
            dayCounts
        )
    }

    private nonisolated static func longestStreak(from activeDates: [Date], calendar: Calendar) -> Int {
        guard !activeDates.isEmpty else { return 0 }
        guard activeDates.count > 1 else { return 1 }

        var longest = 1
        var running = 1

        for index in 1..<activeDates.count {
            let diff = calendar.dateComponents([.day], from: activeDates[index - 1], to: activeDates[index]).day ?? 0
            if diff == 1 {
                running += 1
                longest = max(longest, running)
            } else if diff > 1 {
                running = 1
            }
        }

        return longest
    }

    private nonisolated static func currentStreak(
        from activeKeys: Set<String>,
        calendar: Calendar,
        dateFormatter: DateFormatter
    ) -> Int {
        var date = calendar.startOfDay(for: Date())
        var key = dateFormatter.string(from: date)

        if !activeKeys.contains(key),
           let yesterday = calendar.date(byAdding: .day, value: -1, to: date) {
            date = yesterday
            key = dateFormatter.string(from: date)
        }

        guard activeKeys.contains(key) else { return 0 }

        var streak = 0
        var cursor = date

        while activeKeys.contains(dateFormatter.string(from: cursor)) {
            streak += 1
            guard let previous = calendar.date(byAdding: .day, value: -1, to: cursor) else {
                break
            }
            cursor = previous
        }

        return streak
    }
}
