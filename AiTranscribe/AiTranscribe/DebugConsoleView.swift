/*
 DebugConsoleView.swift
 ======================

 A window that displays real-time logs from the backend server.

 FEATURES:
 - Scrollable log output with timestamps
 - Color-coded by log level (info, warning, error)
 - Clear logs button
 - Copy all logs button
 - Auto-scroll to bottom (toggleable)
 - Restart backend button
 */

import SwiftUI

struct DebugConsoleView: View {
    @EnvironmentObject var backendManager: BackendManager
    @State private var autoScroll = true
    @State private var filterLevel: LogEntry.LogLevel? = nil

    /// Filtered logs based on selected level
    var filteredLogs: [LogEntry] {
        if let level = filterLevel {
            return backendManager.logs.filter { $0.level == level }
        }
        return backendManager.logs
    }

    var body: some View {
        VStack(spacing: 0) {
            // Toolbar
            HStack {
                // Status indicator
                HStack(spacing: 6) {
                    Circle()
                        .fill(backendManager.isRunning ? Color.green : Color.red)
                        .frame(width: 8, height: 8)
                    Text(backendManager.statusMessage)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                Spacer()

                // Filter picker
                Picker("Filter", selection: $filterLevel) {
                    Text("All").tag(nil as LogEntry.LogLevel?)
                    Text("Info").tag(LogEntry.LogLevel.info as LogEntry.LogLevel?)
                    Text("Warning").tag(LogEntry.LogLevel.warning as LogEntry.LogLevel?)
                    Text("Error").tag(LogEntry.LogLevel.error as LogEntry.LogLevel?)
                }
                .pickerStyle(.segmented)
                .frame(width: 200)

                Spacer()

                // Action buttons
                Button {
                    backendManager.clearLogs()
                } label: {
                    Image(systemName: "trash")
                }
                .help("Clear logs")

                Button {
                    copyLogs()
                } label: {
                    Image(systemName: "doc.on.doc")
                }
                .help("Copy logs to clipboard")

                Toggle(isOn: $autoScroll) {
                    Image(systemName: "arrow.down.to.line")
                }
                .toggleStyle(.button)
                .help("Auto-scroll to bottom")
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(Color(NSColor.windowBackgroundColor))

            Divider()

            // Log list
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 2) {
                        ForEach(filteredLogs) { entry in
                            LogEntryRow(entry: entry)
                                .id(entry.id)
                        }
                    }
                    .padding(8)
                }
                .background(Color(NSColor.textBackgroundColor))
                .onChange(of: backendManager.logs.count) { _, _ in
                    if autoScroll, let lastLog = filteredLogs.last {
                        withAnimation(.easeOut(duration: 0.1)) {
                            proxy.scrollTo(lastLog.id, anchor: .bottom)
                        }
                    }
                }
            }

            Divider()

            // Footer
            HStack {
                Text("\(filteredLogs.count) log entries")
                    .font(.caption)
                    .foregroundColor(.secondary)

                if backendManager.restartCount > 0 {
                    Text("• \(backendManager.restartCount) restarts")
                        .font(.caption)
                        .foregroundColor(.orange)
                }

                Spacer()

                // Backend controls
                if backendManager.isRunning {
                    Button("Stop") {
                        backendManager.stop()
                    }
                    .buttonStyle(.bordered)

                    Button("Restart") {
                        backendManager.restart()
                    }
                    .buttonStyle(.bordered)
                } else {
                    Button("Start") {
                        backendManager.start()
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(backendManager.restartCount >= 5)

                    if backendManager.restartCount >= 5 {
                        Button("Reset & Retry") {
                            backendManager.resetAndStart()
                        }
                        .buttonStyle(.bordered)
                        .help("Reset restart counter and try again")
                    }
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(Color(NSColor.windowBackgroundColor))
        }
        .frame(minWidth: 600, minHeight: 400)
    }

    /// Copy all logs to clipboard
    private func copyLogs() {
        let logText = backendManager.logsAsString()
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(logText, forType: .string)
    }
}


/// A single row in the log list
struct LogEntryRow: View {
    let entry: LogEntry

    private var timeString: String {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss.SSS"
        return formatter.string(from: entry.timestamp)
    }

    private var levelColor: Color {
        switch entry.level {
        case .info: return .secondary
        case .warning: return .orange
        case .error: return .red
        case .debug: return .gray
        }
    }

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            // Timestamp
            Text(timeString)
                .font(.system(.caption, design: .monospaced))
                .foregroundColor(.secondary)
                .frame(width: 80, alignment: .leading)

            // Level badge
            Text(entry.level.rawValue)
                .font(.system(.caption2, design: .monospaced))
                .fontWeight(.medium)
                .foregroundColor(levelColor)
                .frame(width: 40, alignment: .leading)

            // Message
            Text(entry.message)
                .font(.system(.caption, design: .monospaced))
                .foregroundColor(entry.level == .error ? .red : .primary)
                .textSelection(.enabled)
        }
        .padding(.vertical, 2)
        .padding(.horizontal, 4)
        .background(entry.level == .error ? Color.red.opacity(0.1) : Color.clear)
        .cornerRadius(4)
    }
}


/// Controller for opening the debug console window
class DebugConsoleWindowController {
    static let shared = DebugConsoleWindowController()

    private var window: NSWindow?

    private init() {}

    func showWindow(backendManager: BackendManager) {
        if let existingWindow = window {
            existingWindow.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
            return
        }

        let contentView = DebugConsoleView()
            .environmentObject(backendManager)

        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 700, height: 500),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )

        window.title = "Backend Console"
        window.contentView = NSHostingView(rootView: contentView)
        window.center()
        window.setFrameAutosaveName("DebugConsoleWindow")
        window.isReleasedWhenClosed = false

        self.window = window

        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }
}


#Preview {
    DebugConsoleView()
        .environmentObject(BackendManager())
}
