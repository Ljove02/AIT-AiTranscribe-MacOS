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

/// All the stats the dashboard needs, computed once and cached.
struct DashboardStats: Equatable {
    // Hero numbers
    var totalTranscriptions: Int = 0
    var totalWords: Int = 0
    var totalDurationSeconds: Double = 0
    var averageWPM: Double = 0
    var timeSavedMinutes: Double = 0  // vs typing at 40 WPM

    // Activity
    var usingSinceDate: Date?
    var totalActiveDays: Int = 0
    var longestStreak: Int = 0
    var currentStreak: Int = 0
    var busiestDay: (date: String, count: Int)?
    var monthlyBreakdown: [(month: String, count: Int)] = []

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

    /// Compute stats from an array of transcription entries.
    /// Call this once when the dashboard first appears.
    func computeStats(from entries: [TranscriptionEntry]) async {
        guard !entries.isEmpty else {
            hasLoaded = true
            return
        }

        isLoading = true

        // Do heavy work off main thread
        let computed = await Task.detached(priority: .userInitiated) {
            Self.calculate(entries: entries)
        }.value

        stats = computed
        isLoading = false
        hasLoaded = true
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
    }

    // MARK: - Heavy Computation (off-main)

    private nonisolated static func calculate(entries: [TranscriptionEntry]) -> DashboardStats {
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

        // Date analysis
        let calendar = Calendar.current
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd"

        let dates = entries.map { dateFormatter.string(from: $0.timestamp) }
        let uniqueDates = Set(dates).sorted()
        s.totalActiveDays = uniqueDates.count

        if let firstEntry = entries.min(by: { $0.timestamp < $1.timestamp }) {
            s.usingSinceDate = firstEntry.timestamp
        }

        // Busiest day
        var dayCounts: [String: Int] = [:]
        for d in dates { dayCounts[d, default: 0] += 1 }
        if let busiest = dayCounts.max(by: { $0.value < $1.value }) {
            s.busiestDay = (busiest.key, busiest.value)
        }

        // Streak calculation
        let dateObjects = uniqueDates.compactMap { dateFormatter.date(from: $0) }
        if dateObjects.count > 1 {
            var maxStreak = 1
            var currentStreak = 1

            for i in 1..<dateObjects.count {
                let diff = calendar.dateComponents([.day], from: dateObjects[i-1], to: dateObjects[i]).day ?? 0
                if diff == 1 {
                    currentStreak += 1
                    maxStreak = max(maxStreak, currentStreak)
                } else {
                    currentStreak = 1
                }
            }
            s.longestStreak = maxStreak

            // Current streak (from today backwards)
            let today = dateFormatter.string(from: Date())
            let yesterday = dateFormatter.string(from: calendar.date(byAdding: .day, value: -1, to: Date()) ?? Date())

            if uniqueDates.last == today || uniqueDates.last == yesterday {
                var streak = 1
                let reversed = dateObjects.reversed().map { $0 }
                for i in 1..<reversed.count {
                    let diff = calendar.dateComponents([.day], from: reversed[i], to: reversed[i-1]).day ?? 0
                    if diff == 1 {
                        streak += 1
                    } else {
                        break
                    }
                }
                s.currentStreak = streak
            }
        } else if dateObjects.count == 1 {
            s.longestStreak = 1
            s.currentStreak = 1
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
}
