import SwiftUI

// MARK: - History Settings View

struct HistorySettingsView: View {
    @EnvironmentObject var appState: AppState

    let hasAnimated: Bool
    let onAnimated: () -> Void
    let initialSelectedEntryId: UUID?

    @State private var searchText = ""
    @State private var selectedEntryId: UUID? = nil
    @State private var showCopied = false
    @State private var copiedEntryId: UUID? = nil
    @State private var showClearAllAlert = false
    @State private var searchResults: [TranscriptionEntry] = []
    @State private var isSearching = false
    @State private var isIndexing = false

    /// Animation state
    @State private var appeared: Bool

    init(hasAnimated: Bool, onAnimated: @escaping () -> Void, initialSelectedEntryId: UUID? = nil) {
        self.hasAnimated = hasAnimated
        self.onAnimated = onAnimated
        self.initialSelectedEntryId = initialSelectedEntryId
        _appeared = State(initialValue: hasAnimated)
        _selectedEntryId = State(initialValue: initialSelectedEntryId)
    }

    /// When not searching, show loaded entries. When searching, show results from ALL entries.
    var displayedHistory: [TranscriptionEntry] {
        searchText.isEmpty ? appState.transcriptionHistory : searchResults
    }

    var body: some View {
        VStack(spacing: 0) {
            if let entryId = selectedEntryId,
               let entry = appState.transcriptionHistory.first(where: { $0.id == entryId })
                            ?? searchResults.first(where: { $0.id == entryId }) {
                historyDetailView(entry: entry)
            } else {
                historyListView
            }
        }
        .task(id: "stagger") {
            guard !hasAnimated else { return }
            try? await Task.sleep(for: .milliseconds(80))
            appeared = true
            onAnimated()
        }
        .task(id: searchText) {
            guard !searchText.isEmpty else {
                searchResults = []
                isSearching = false
                return
            }
            isSearching = true
            // Short debounce — text search is fast, semantic is cached
            try? await Task.sleep(for: .milliseconds(400))
            guard !Task.isCancelled else { return }
            searchResults = await appState.searchHistory(query: searchText)
            isSearching = false
        }
    }

    // MARK: - List View

    private var historyListView: some View {
        ZStack(alignment: .bottom) {
            ScrollView {
                VStack(spacing: 0) {
                    // Header
                    historyHeader
                        .staggerIn(index: 0, appeared: appeared)

                    // Search bar
                    historySearchBar
                        .padding(.top, 10)
                        .staggerIn(index: 1, appeared: appeared)

                    SettingsDivider()
                        .padding(.top, 12)
                        .staggerIn(index: 2, appeared: appeared)

                    // History list or empty state
                    if displayedHistory.isEmpty {
                        emptyStateView
                            .staggerIn(index: 3, appeared: appeared)
                    } else {
                        historyCards
                            .staggerIn(index: 3, appeared: appeared)
                    }

                    // Footer
                    if !appState.transcriptionHistory.isEmpty {
                        SettingsDivider()
                            .padding(.top, 8)
                        footerActions
                            .padding(.top, 8)
                            .staggerIn(index: 4, appeared: appeared)
                    }

                    // Extra space for overlay load-more pill
                    if appState.hasMoreHistory && searchText.isEmpty {
                        Spacer(minLength: 56)
                    } else {
                        Spacer(minLength: 20)
                    }
                }
                .padding(.horizontal, 28)
                .padding(.top, 8)
            }
            .scrollIndicators(.automatic)

            // Floating load-more pill
            if appState.hasMoreHistory && searchText.isEmpty {
                loadMoreOverlay
            }
        }
    }

    // MARK: - Header

    private var historyHeader: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 3) {
                Text("History")
                    .font(.system(size: 22, weight: .bold, design: .rounded))

                HStack(spacing: 6) {
                    Text("\(appState.totalHistoryCount) transcriptions")
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)

                    if appState.transcriptionHistory.count < appState.totalHistoryCount {
                        Text("·")
                            .foregroundStyle(.tertiary)
                        Text("\(appState.transcriptionHistory.count) loaded")
                            .font(.system(size: 11))
                            .foregroundStyle(.tertiary)
                    }
                }
            }

            Spacer()
        }
    }

    // MARK: - Search Bar

    private var historySearchBar: some View {
        VStack(spacing: 8) {
            HStack(spacing: 8) {
                // Search field
                HStack(spacing: 8) {
                    Image(systemName: "magnifyingglass")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(.tertiary)

                    TextField(appState.isHistoryIndexed
                              ? "Search transcriptions (similarity)…"
                              : "Search transcriptions (keyword)…",
                              text: $searchText)
                        .textFieldStyle(.plain)
                        .font(.system(size: 12))

                    if isSearching {
                        ProgressView()
                            .scaleEffect(0.5)
                            .frame(width: 12, height: 12)
                    } else if !searchText.isEmpty {
                        Button {
                            searchText = ""
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .font(.system(size: 12))
                                .foregroundStyle(.tertiary)
                        }
                        .buttonStyle(.plain)
                    }
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(.thinMaterial, in: Capsule())
                .overlay(Capsule().strokeBorder(.white.opacity(0.06), lineWidth: 0.5))

                // Index button
                Button {
                    guard !isIndexing, !appState.isHistoryIndexed else { return }
                    isIndexing = true
                    Task {
                        await appState.buildHistoryIndex()
                        isIndexing = false
                        // Re-trigger search if there's a query
                        if !searchText.isEmpty {
                            isSearching = true
                            searchResults = await appState.searchHistory(query: searchText)
                            isSearching = false
                        }
                    }
                } label: {
                    HStack(spacing: 4) {
                        if isIndexing {
                            ProgressView()
                                .scaleEffect(0.5)
                                .frame(width: 10, height: 10)
                        } else {
                            Image(systemName: appState.isHistoryIndexed ? "checkmark" : "brain")
                                .font(.system(size: 10, weight: .medium))
                        }
                        Text(isIndexing ? "Indexing…"
                             : appState.isHistoryIndexed ? "Indexed" : "Index")
                            .font(.system(size: 10, weight: .medium))
                    }
                    .foregroundStyle(appState.isHistoryIndexed ? .green : .primary)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .background(.thinMaterial, in: Capsule())
                    .overlay(Capsule().strokeBorder(
                        appState.isHistoryIndexed ? Color.green.opacity(0.2) : .white.opacity(0.06),
                        lineWidth: 0.5
                    ))
                }
                .buttonStyle(.plain)
                .disabled(isIndexing || appState.isHistoryIndexed)
            }

            // Info text when not indexed
            if !appState.isHistoryIndexed && !isIndexing {
                Text("Index transcriptions to enable similarity search across all history")
                    .font(.system(size: 9))
                    .foregroundStyle(.tertiary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.leading, 4)
            }
        }
    }

    // MARK: - History Cards

    private var historyCards: some View {
        LazyVStack(spacing: 6) {
            ForEach(displayedHistory) { entry in
                HistoryCardView(
                    entry: entry,
                    isCopied: copiedEntryId == entry.id,
                    onCopy: {
                        copyEntry(entry)
                    },
                    onTap: {
                        withAnimation(.easeInOut(duration: 0.15)) {
                            selectedEntryId = entry.id
                        }
                    }
                )
                .contextMenu {
                    Button("Copy") { copyEntry(entry) }
                    Button("Open") { selectedEntryId = entry.id }
                    Divider()
                    Button("Delete", role: .destructive) {
                        appState.deleteFromHistory(entry)
                    }
                }
            }
        }
        .padding(.top, 12)
    }

    // MARK: - Empty State

    private var emptyStateView: some View {
        VStack(spacing: 12) {
            Image(systemName: searchText.isEmpty ? "clock.arrow.circlepath" : "magnifyingglass")
                .font(.system(size: 36))
                .foregroundStyle(.tertiary)

            Text(searchText.isEmpty ? "No transcriptions yet" : "No results found")
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(.secondary)

            Text(searchText.isEmpty
                 ? "Your transcription history will appear here."
                 : "Try a different search term.")
                .font(.system(size: 11))
                .foregroundStyle(.tertiary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 60)
    }

    // MARK: - Floating Load More

    private var loadMoreOverlay: some View {
        Button {
            Task { await appState.loadMoreHistory() }
        } label: {
            HStack(spacing: 6) {
                if appState.isLoadingMoreHistory {
                    ProgressView()
                        .scaleEffect(0.6)
                        .frame(width: 10, height: 10)
                    Text("Loading…")
                        .font(.system(size: 10, weight: .medium))
                } else {
                    Image(systemName: "chevron.down")
                        .font(.system(size: 9, weight: .semibold))
                    Text("Load More")
                        .font(.system(size: 10, weight: .medium))
                }
            }
            .foregroundStyle(.primary)
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
            .background(.thinMaterial, in: Capsule())
            .overlay(Capsule().strokeBorder(.white.opacity(0.08), lineWidth: 0.5))
            .shadow(color: .black.opacity(0.2), radius: 8, y: 4)
        }
        .buttonStyle(.plain)
        .disabled(appState.isLoadingMoreHistory)
        .padding(.bottom, 16)
    }

    // MARK: - Footer

    private var footerActions: some View {
        HStack(spacing: 8) {
            Text(searchText.isEmpty
                 ? "\(displayedHistory.count) loaded"
                 : "\(displayedHistory.count) results")
                .font(.system(size: 10))
                .foregroundStyle(.tertiary)

            Spacer()

            footerPill(label: "Clear All", tint: .red) {
                showClearAllAlert = true
            }
            .disabled(appState.transcriptionHistory.isEmpty)
            .alert("Clear All History?", isPresented: $showClearAllAlert) {
                Button("Cancel", role: .cancel) {}
                Button("Clear All", role: .destructive) {
                    appState.clearHistory()
                }
            } message: {
                Text("All transcription history will be permanently deleted.")
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

    // MARK: - Detail View

    private func historyDetailView(entry: TranscriptionEntry) -> some View {
        ScrollView {
            VStack(spacing: 0) {
                // Back + actions
                detailHeader(entry: entry)

                SettingsDivider()
                    .padding(.top, 8)

                // Meta pills row
                detailMetaRow(entry: entry)
                    .padding(.top, 12)

                SettingsDivider()
                    .padding(.top, 12)

                // Transcription text
                Text(entry.text)
                    .font(.system(size: 13))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .lineSpacing(4)
                    .padding(.top, 16)

                SettingsDivider()
                    .padding(.top, 16)

                // Footer stats + delete
                detailFooter(entry: entry)
                    .padding(.top, 12)

                Spacer(minLength: 20)
            }
            .padding(.horizontal, 28)
            .padding(.top, 8)
        }
        .scrollIndicators(.automatic)
    }

    private func detailHeader(entry: TranscriptionEntry) -> some View {
        HStack(spacing: 10) {
            // Back button
            Button {
                withAnimation(.easeInOut(duration: 0.15)) {
                    selectedEntryId = nil
                }
            } label: {
                HStack(spacing: 4) {
                    Image(systemName: "chevron.left")
                        .font(.system(size: 10, weight: .semibold))
                    Text("History")
                        .font(.system(size: 11, weight: .medium))
                }
                .foregroundStyle(.secondary)
                .padding(.horizontal, 12)
                .padding(.vertical, 7)
                .background(.thinMaterial, in: Capsule())
                .overlay(Capsule().strokeBorder(.white.opacity(0.06), lineWidth: 0.5))
            }
            .buttonStyle(.plain)

            Spacer()

            // Copy pill
            Button {
                appState.copyToClipboard(entry.text)
                showCopied = true
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                    showCopied = false
                }
            } label: {
                HStack(spacing: 4) {
                    Image(systemName: showCopied ? "checkmark" : "doc.on.doc")
                        .font(.system(size: 10, weight: .medium))
                    Text(showCopied ? "Copied" : "Copy")
                        .font(.system(size: 11, weight: .medium))
                }
                .foregroundStyle(showCopied ? .green : .primary)
                .padding(.horizontal, 12)
                .padding(.vertical, 7)
                .background(.thinMaterial, in: Capsule())
                .overlay(Capsule().strokeBorder(.white.opacity(0.06), lineWidth: 0.5))
            }
            .buttonStyle(.plain)
            .animation(.easeOut(duration: 0.15), value: showCopied)
        }
    }

    private static let detailDateFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateStyle = .medium
        f.timeStyle = .short
        return f
    }()

    private func detailMetaRow(entry: TranscriptionEntry) -> some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                metaPill(icon: "calendar", text: Self.detailDateFormatter.string(from: entry.timestamp))
                metaPill(icon: "timer", text: formatDuration(entry.duration))
                metaPill(icon: "text.word.spacing", text: "\(entry.wordCount) words")
                metaPill(icon: "cpu", text: entry.modelName, tint: .accentColor)
                if !entry.language.isEmpty && entry.language != "unknown" {
                    metaPill(icon: "globe", text: entry.language)
                }
            }
        }
    }

    private func metaPill(icon: String, text: String, tint: Color = .secondary) -> some View {
        HStack(spacing: 3) {
            Image(systemName: icon)
                .font(.system(size: 8))
            Text(text)
                .font(.system(size: 9))
        }
        .foregroundStyle(tint)
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(.quaternary, in: Capsule())
    }

    private func detailFooter(entry: TranscriptionEntry) -> some View {
        HStack(spacing: 8) {
            HStack(spacing: 12) {
                Text("\(entry.charCount) characters")
                Text("\(entry.phrasesCount) sentences")
                if entry.wordsPerMinute > 0 {
                    Text("\(Int(entry.wordsPerMinute)) WPM")
                }
            }
            .font(.system(size: 10))
            .foregroundStyle(.tertiary)

            Spacer()

            footerPill(label: "Delete", tint: .red) {
                appState.deleteFromHistory(entry)
                withAnimation(.easeInOut(duration: 0.15)) {
                    selectedEntryId = nil
                }
            }
        }
    }

    // MARK: - Helpers

    private func copyEntry(_ entry: TranscriptionEntry) {
        appState.copyToClipboard(entry.text)
        copiedEntryId = entry.id
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
            if copiedEntryId == entry.id {
                copiedEntryId = nil
            }
        }
    }

    private func formatDuration(_ seconds: Double) -> String {
        if seconds < 60 {
            return String(format: "%.0fs", seconds)
        }
        let mins = Int(seconds) / 60
        let secs = Int(seconds) % 60
        return "\(mins)m \(secs)s"
    }
}

// MARK: - History Card View

struct HistoryCardView: View {
    let entry: TranscriptionEntry
    var isCopied: Bool = false
    var onCopy: () -> Void
    var onTap: () -> Void

    @State private var isHovered = false

    private static let dateFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateStyle = .short
        f.timeStyle = .short
        return f
    }()

    var body: some View {
        HStack(spacing: 12) {
            // Left: text preview + meta
            VStack(alignment: .leading, spacing: 6) {
                Text(entry.text)
                    .font(.system(size: 12))
                    .lineLimit(2)
                    .frame(maxWidth: .infinity, alignment: .leading)

                HStack(spacing: 5) {
                    metaPill(icon: "clock", text: Self.dateFormatter.string(from: entry.timestamp))
                    metaPill(icon: "timer", text: String(format: "%.0fs", entry.duration))
                    metaPill(icon: "text.word.spacing", text: "\(entry.wordCount)")
                    metaPill(icon: "cpu", text: entry.modelName, tint: .accentColor)
                }
            }

            // Copy button
            Button(action: onCopy) {
                Image(systemName: isCopied ? "checkmark" : "doc.on.doc")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(isCopied ? .green : .secondary)
                    .frame(width: 28, height: 28)
                    .background(.quaternary, in: .circle)
            }
            .buttonStyle(.plain)
            .animation(.easeOut(duration: 0.15), value: isCopied)

            // Chevron
            Image(systemName: "chevron.right")
                .font(.system(size: 10, weight: .medium))
                .foregroundStyle(.quaternary)
        }
        .padding(.vertical, 10)
        .padding(.horizontal, 14)
        .background(.thinMaterial, in: .rect(cornerRadius: 12, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(.white.opacity(isHovered ? 0.1 : 0.06), lineWidth: 0.5)
        )
        .scaleEffect(isHovered ? 1.008 : 1.0)
        .animation(.easeOut(duration: 0.15), value: isHovered)
        .onHover { isHovered = $0 }
        .contentShape(Rectangle())
        .onTapGesture { onTap() }
    }

    private func metaPill(icon: String, text: String, tint: Color = .secondary) -> some View {
        HStack(spacing: 2) {
            Image(systemName: icon)
                .font(.system(size: 7))
            Text(text)
                .font(.system(size: 8))
        }
        .foregroundStyle(tint)
        .padding(.horizontal, 5)
        .padding(.vertical, 2)
        .background(.quaternary, in: Capsule())
    }
}
