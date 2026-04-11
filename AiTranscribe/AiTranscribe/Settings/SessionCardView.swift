import SwiftUI

// MARK: - Session Card View

/// A glass-material card for a single session — compact pill layout with metadata badges.
struct SessionCardView: View {
    let session: Session
    var selectionMode = false
    var isSelected = false
    var onRename: ((String) -> Void)? = nil

    @State private var isHovered = false
    @State private var isEditing = false
    @State private var editedName = ""
    @State private var cardAppeared = false

    private var statusColor: Color {
        switch session.status {
        case .completed: return .green
        case .recording: return .blue
        case .transcribing: return .orange
        case .failed: return .red
        case .idle:
            return session.hasTranscription ? .green : .secondary
        }
    }

    private var statusText: String {
        switch session.status {
        case .completed: return "Transcribed"
        case .recording: return "Recording"
        case .transcribing: return "Transcribing"
        case .failed: return "Failed"
        case .idle:
            return session.hasTranscription ? "Transcribed" : "Pending"
        }
    }

    private var statusIcon: String {
        switch session.status {
        case .completed: return "checkmark.circle.fill"
        case .recording: return "record.circle"
        case .transcribing: return "arrow.triangle.2.circlepath"
        case .failed: return "exclamationmark.circle.fill"
        case .idle:
            return session.hasTranscription ? "checkmark.circle.fill" : "circle.dashed"
        }
    }

    private static let timeFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "h:mm a"
        return f
    }()

    var body: some View {
        HStack(spacing: 12) {
            // Selection checkbox
            if selectionMode {
                Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                    .font(.system(size: 16))
                    .foregroundStyle(isSelected ? Color.accentColor : .secondary)
                    .animation(.easeOut(duration: 0.15), value: isSelected)
            }

            // Audio waveform icon
            ZStack {
                Circle()
                    .fill(statusColor.opacity(0.12))
                    .frame(width: 36, height: 36)
                Image(systemName: session.hasAudio ? "waveform" : "waveform.slash")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(statusColor)
            }

            // Name + time
            VStack(alignment: .leading, spacing: 4) {
                if isEditing {
                    TextField("Session name", text: $editedName, onCommit: {
                        onRename?(editedName)
                        isEditing = false
                    })
                    .textFieldStyle(.plain)
                    .font(.system(size: 13, weight: .semibold))
                    .onExitCommand { isEditing = false }
                } else {
                    Text(session.name)
                        .font(.system(size: 13, weight: .semibold))
                        .lineLimit(1)
                        .onTapGesture(count: 2) {
                            editedName = session.name
                            isEditing = true
                        }
                }

                Text(Self.timeFormatter.string(from: session.createdAt))
                    .font(.system(size: 10))
                    .foregroundStyle(.tertiary)
            }

            Spacer()

            // Metadata pills
            HStack(spacing: 5) {
                if session.duration > 0 {
                    metaPill(icon: "clock", text: SessionManager.formatDuration(session.duration))
                }
                if session.fileSize > 0 {
                    metaPill(icon: "doc", text: SessionManager.formatFileSize(session.fileSize))
                }
            }

            // Status badge
            HStack(spacing: 4) {
                Image(systemName: statusIcon)
                    .font(.system(size: 8))
                Text(statusText)
                    .font(.system(size: 9, weight: .medium))
            }
            .foregroundStyle(statusColor)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(statusColor.opacity(0.1), in: Capsule())
        }
        .padding(.vertical, 10)
        .padding(.horizontal, 14)
        .background(.thinMaterial, in: .rect(cornerRadius: 12, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(
                    isSelected ? Color.accentColor.opacity(0.4) : .white.opacity(isHovered ? 0.1 : 0.06),
                    lineWidth: isSelected ? 1.5 : 0.5
                )
        )
        .scaleEffect(isHovered ? 1.008 : 1.0)
        .scaleEffect(cardAppeared ? 1.0 : 0.96)
        .offset(y: cardAppeared ? 0 : 6)
        .animation(.easeOut(duration: 0.15), value: isHovered)
        .animation(.spring(duration: 0.4, bounce: 0.15), value: cardAppeared)
        .onHover { isHovered = $0 }
        .onAppear {
            if !cardAppeared { cardAppeared = true }
        }
        .contentShape(Rectangle())
    }

    private func metaPill(icon: String, text: String) -> some View {
        HStack(spacing: 3) {
            Image(systemName: icon)
                .font(.system(size: 8))
            Text(text)
                .font(.system(size: 9))
        }
        .foregroundStyle(.secondary)
        .padding(.horizontal, 6)
        .padding(.vertical, 3)
        .background(.quaternary, in: Capsule())
    }
}
