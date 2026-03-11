import SwiftUI

// MARK: - History Settings View

struct HistorySettingsView: View {
    @EnvironmentObject var appState: AppState
    @State private var searchText = ""

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
                                .contextMenu {
                                    Button("Copy") {
                                        appState.copyToClipboard(entry.text)
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
