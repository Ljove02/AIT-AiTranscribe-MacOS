import SwiftUI

// MARK: - Session Card View

/// A card displaying a single session in the sessions list
struct SessionCardView: View {
    let session: Session
    var selectionMode = false
    var isSelected = false

    private var statusColor: Color {
        switch session.status {
        case .completed: return .green
        case .recording: return .blue
        case .transcribing: return .orange
        case .failed: return .red
        case .idle: return .secondary
        }
    }

    private var statusText: String {
        switch session.status {
        case .completed: return "Transcribed"
        case .recording: return "Recording"
        case .transcribing: return "Transcribing..."
        case .failed: return "Failed"
        case .idle:
            return session.hasTranscription ? "Transcribed" : "Not transcribed"
        }
    }

    private var dateFormatter: DateFormatter {
        let formatter = DateFormatter()
        formatter.dateFormat = "MMM d, EEE  h:mm a"
        return formatter
    }

    var body: some View {
        HStack(spacing: 12) {
            if selectionMode {
                Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                    .font(.system(size: 18))
                    .foregroundColor(isSelected ? .accentColor : .secondary.opacity(0.8))
            }

            // Audio icon
            Image(systemName: session.hasAudio ? "waveform" : "waveform.slash")
                .font(.system(size: 16))
                .foregroundColor(session.hasAudio ? .accentColor : .secondary)
                .frame(width: 28, height: 28)

            // Session info
            VStack(alignment: .leading, spacing: 4) {
                Text(session.name)
                    .font(.system(size: 13, weight: .medium))
                    .lineLimit(1)

                HStack(spacing: 6) {
                    Text(dateFormatter.string(from: session.createdAt))
                        .font(.caption2)
                        .foregroundColor(.secondary)

                    if session.duration > 0 {
                        Text("•")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                        Text(SessionManager.formatDuration(session.duration))
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }

                    if session.fileSize > 0 {
                        Text("•")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                        Text(SessionManager.formatFileSize(session.fileSize))
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                }
            }

            Spacer()

            // Status badge
            HStack(spacing: 4) {
                Circle()
                    .fill(statusColor)
                    .frame(width: 6, height: 6)
                Text(statusText)
                    .font(.caption2)
                    .foregroundColor(statusColor)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(
                Capsule()
                    .fill(statusColor.opacity(0.1))
            )
        }
        .padding(.vertical, 10)
        .padding(.horizontal, 14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(isSelected ? Color.accentColor.opacity(0.08) : Color.clear)
        )
        .contentShape(Rectangle())
    }
}
