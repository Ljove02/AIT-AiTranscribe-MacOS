import SwiftUI

// MARK: - History Settings View

struct HistorySettingsView: View {
    @EnvironmentObject var appState: AppState
    @State private var searchText = ""
    @State private var selectedEntryId: UUID? = nil
    @State private var showCopied = false

    var filteredHistory: [TranscriptionEntry] {
        if searchText.isEmpty {
            return appState.transcriptionHistory
        }
        return appState.transcriptionHistory.filter {
            $0.text.localizedCaseInsensitiveContains(searchText)
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            if let entryId = selectedEntryId,
               let entry = appState.transcriptionHistory.first(where: { $0.id == entryId }) {
                historyDetailView(entry: entry)
            } else {
                historyListView
            }
        }
    }

    // MARK: - Detail View

    private func historyDetailView(entry: TranscriptionEntry) -> some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Button {
                    withAnimation(.easeInOut(duration: 0.15)) {
                        selectedEntryId = nil
                    }
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: "chevron.left")
                        Text("History")
                    }
                    .font(.subheadline)
                }
                .buttonStyle(.plain)
                .foregroundColor(.accentColor)

                Spacer()

                // Copy button
                Button {
                    appState.copyToClipboard(entry.text)
                    showCopied = true
                    DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                        showCopied = false
                    }
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: showCopied ? "checkmark" : "doc.on.doc")
                        Text(showCopied ? "Copied" : "Copy")
                    }
                    .font(.subheadline)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 10)

            Divider()

            // Metadata bar
            HStack(spacing: 12) {
                Label(detailDateFormatter.string(from: entry.timestamp), systemImage: "calendar")
                Label(formatDuration(entry.duration), systemImage: "timer")
                Label("\(entry.wordCount) words", systemImage: "text.word.spacing")
                Label(entry.modelName, systemImage: "cpu")
                    .foregroundColor(.accentColor)
                Spacer()
            }
            .font(.caption)
            .foregroundColor(.secondary)
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
            .background(Color(NSColor.controlBackgroundColor))

            Divider()

            // Full transcription text
            ScrollView {
                Text(entry.text)
                    .font(.body)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(16)
            }

            Divider()

            // Footer stats
            HStack(spacing: 16) {
                Text("\(entry.charCount) characters")
                Text("\(entry.phrasesCount) sentences")
                if entry.wordsPerMinute > 0 {
                    Text("\(Int(entry.wordsPerMinute)) WPM")
                }
                Spacer()

                Button(role: .destructive) {
                    appState.deleteFromHistory(entry)
                    selectedEntryId = nil
                } label: {
                    Label("Delete", systemImage: "trash")
                }
                .buttonStyle(.borderless)
                .foregroundColor(.red)
            }
            .font(.caption)
            .foregroundColor(.secondary)
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
        }
    }

    private var detailDateFormatter: DateFormatter {
        let f = DateFormatter()
        f.dateStyle = .medium
        f.timeStyle = .short
        return f
    }

    private func formatDuration(_ seconds: Double) -> String {
        if seconds < 60 {
            return String(format: "%.0fs", seconds)
        }
        let mins = Int(seconds) / 60
        let secs = Int(seconds) % 60
        return "\(mins)m \(secs)s"
    }

    // MARK: - List View

    private var historyListView: some View {
        VStack(spacing: 0) {
            // Search bar
            HStack {
                Image(systemName: "magnifyingglass")
                    .foregroundColor(.secondary)
                TextField("Search transcriptions...", text: $searchText)
                    .textFieldStyle(.plain)
                if !searchText.isEmpty {
                    Button {
                        searchText = ""
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundColor(.secondary)
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(8)
            .background(Color(NSColor.textBackgroundColor))

            Divider()

            // History list with pagination
            if filteredHistory.isEmpty {
                VStack(spacing: 12) {
                    Image(systemName: "clock")
                        .font(.system(size: 40))
                        .foregroundColor(.secondary)
                    Text(searchText.isEmpty ? "No transcriptions yet" : "No results found")
                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    LazyVStack(spacing: 0) {
                        ForEach(filteredHistory) { entry in
                            HistoryEntryRow(entry: entry)
                                .contentShape(Rectangle())
                                .onTapGesture {
                                    withAnimation(.easeInOut(duration: 0.15)) {
                                        selectedEntryId = entry.id
                                    }
                                }
                                .contextMenu {
                                    Button("Copy") {
                                        appState.copyToClipboard(entry.text)
                                    }
                                    Button("Open") {
                                        selectedEntryId = entry.id
                                    }
                                    Divider()
                                    Button("Delete", role: .destructive) {
                                        appState.deleteFromHistory(entry)
                                    }
                                }

                            Divider()
                        }

                        // Load more button - compact and centered
                        if !searchText.isEmpty {
                            // In search mode, show all results (no pagination)
                            Color.clear
                                .frame(height: 1)
                        } else if appState.hasMoreHistory {
                            HStack {
                                Spacer()
                                Button {
                                    Task {
                                        await appState.loadMoreHistory()
                                    }
                                } label: {
                                    if appState.isLoadingMoreHistory {
                                        HStack(spacing: 6) {
                                            ProgressView()
                                                .scaleEffect(0.7)
                                                .frame(width: 12, height: 12)
                                            Text("Loading...")
                                                .font(.caption)
                                        }
                                        .padding(.horizontal, 16)
                                        .padding(.vertical, 6)
                                    } else {
                                        HStack(spacing: 6) {
                                            Text("Load More")
                                                .font(.caption)
                                            Image(systemName: "chevron.down")
                                                .font(.caption2)
                                        }
                                        .padding(.horizontal, 16)
                                        .padding(.vertical, 6)
                                    }
                                }
                                .buttonStyle(.bordered)
                                .disabled(appState.isLoadingMoreHistory)
                                .padding(.vertical, 12)
                                Spacer()
                            }
                        } else {
                            Color.clear
                                .frame(height: 1)
                        }
                    }
                }
            }

            Divider()

            // Footer with count and clear button
            HStack {
                Text("\(filteredHistory.count) transcriptions")
                    .font(.caption)
                    .foregroundColor(.secondary)

                Spacer()

                Button("Clear All") {
                    appState.clearHistory()
                }
                .disabled(filteredHistory.isEmpty)
            }
            .padding(8)
        }
    }
}

// MARK: - History Entry Row

struct HistoryEntryRow: View {
    let entry: TranscriptionEntry

    private var dateFormatter: DateFormatter {
        let formatter = DateFormatter()
        formatter.dateStyle = .short
        formatter.timeStyle = .short
        return formatter
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            // Main text - left aligned
            Text(entry.text)
                .font(.body)
                .lineLimit(3)
                .multilineTextAlignment(.leading)
                .frame(maxWidth: .infinity, alignment: .leading)

            // Metadata row - left aligned
            HStack(spacing: 8) {
                // Timestamp
                Text(dateFormatter.string(from: entry.timestamp))
                    .font(.caption2)
                    .foregroundColor(.secondary)

                Text("•")
                    .font(.caption2)
                    .foregroundColor(.secondary)

                // Duration
                Text(String(format: "%.0fs", entry.duration))
                    .font(.caption2)
                    .foregroundColor(.secondary)

                // Word count
                Text("•")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                Text("\(entry.wordCount) words")
                    .font(.caption2)
                    .foregroundColor(.secondary)

                // Model name
                Text("•")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                Text(entry.modelName)
                    .font(.caption2)
                    .foregroundColor(.accentColor)

                Spacer()
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.vertical, 8)
        .padding(.horizontal, 12)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
