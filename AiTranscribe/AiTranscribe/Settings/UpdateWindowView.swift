import SwiftUI
import WebKit

// MARK: - Update Window View

struct UpdateWindowView: View {
    @EnvironmentObject var updateChecker: UpdateChecker
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        ZStack {
            TransparentWindow()
                .frame(width: 0, height: 0)

            VStack(spacing: 0) {
                // Top bar
                updateTopBar

                // Content
                VStack(spacing: 16) {
                    // Version badge
                    versionHeader

                    // Release notes
                    if let body = updateChecker.latestRelease?.body, !body.isEmpty {
                        MarkdownWebView(markdown: body)
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                            .clipShape(.rect(cornerRadius: 10, style: .continuous))
                    } else {
                        noNotesPlaceholder
                    }

                    // Action buttons
                    actionButtons
                }
                .padding(.horizontal, 24)
                .padding(.bottom, 20)
            }
            .background(.ultraThinMaterial)
            .clipShape(.rect(cornerRadius: 14, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .strokeBorder(.white.opacity(0.08), lineWidth: 0.5)
            )
            .shadow(color: .black.opacity(0.3), radius: 20, x: 0, y: 8)
        }
        .ignoresSafeArea()
        .frame(width: 520, height: 480)
    }

    // MARK: - Top Bar

    private var updateTopBar: some View {
        HStack(spacing: 10) {
            // Close button
            Button {
                dismiss()
            } label: {
                Circle()
                    .fill(Color.red)
                    .frame(width: 12, height: 12)
            }
            .buttonStyle(.plain)
            .padding(.leading, 16)

            Spacer()

            Text("Software Update")
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(.secondary)

            Spacer()

            // Balance the close button
            Color.clear.frame(width: 12, height: 12)
                .padding(.trailing, 16)
        }
        .frame(height: 44)
    }

    // MARK: - Version Header

    private var versionHeader: some View {
        HStack(spacing: 14) {
            // App icon
            Group {
                if let nsImage = loadAppIconFromBundle() {
                    Image(nsImage: nsImage)
                        .resizable()
                        .scaledToFit()
                } else {
                    Image(systemName: "mic.circle.fill")
                        .resizable()
                        .scaledToFit()
                        .foregroundStyle(.primary.opacity(0.6))
                }
            }
            .frame(width: 48, height: 48)
            .clipShape(.rect(cornerRadius: 11, style: .continuous))
            .shadow(color: .black.opacity(0.2), radius: 6, y: 3)

            VStack(alignment: .leading, spacing: 4) {
                Text("AiTranscribe")
                    .font(.system(size: 16, weight: .bold, design: .rounded))

                HStack(spacing: 8) {
                    Text("v\(updateChecker.currentVersion)")
                        .font(.system(size: 11, weight: .medium, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 7)
                        .padding(.vertical, 2)
                        .background(.quaternary, in: Capsule())

                    Image(systemName: "arrow.right")
                        .font(.system(size: 9, weight: .bold))
                        .foregroundStyle(.tertiary)

                    if let version = updateChecker.latestVersion {
                        Text("v\(version)")
                            .font(.system(size: 11, weight: .bold, design: .monospaced))
                            .foregroundStyle(.green)
                            .padding(.horizontal, 7)
                            .padding(.vertical, 2)
                            .background(.green.opacity(0.15), in: Capsule())
                    }
                }
            }

            Spacer()
        }
        .padding(.top, 4)
    }

    // MARK: - No Notes Placeholder

    private var noNotesPlaceholder: some View {
        VStack(spacing: 8) {
            Spacer()
            Image(systemName: "doc.text")
                .font(.system(size: 28, weight: .light))
                .foregroundStyle(.tertiary)
            Text("No release notes available")
                .font(.system(size: 13))
                .foregroundStyle(.secondary)
            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(.black.opacity(0.15), in: .rect(cornerRadius: 10, style: .continuous))
    }

    // MARK: - Action Buttons

    private var actionButtons: some View {
        HStack(spacing: 10) {
            // View on GitHub
            Button {
                updateChecker.openReleasePage()
            } label: {
                HStack(spacing: 5) {
                    Image(systemName: "arrow.up.right")
                        .font(.system(size: 10, weight: .medium))
                    Text("View on GitHub")
                        .font(.system(size: 12, weight: .medium))
                }
                .foregroundStyle(.primary)
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .background(.thinMaterial, in: Capsule())
                .overlay(Capsule().strokeBorder(.white.opacity(0.06), lineWidth: 0.5))
            }
            .buttonStyle(.plain)

            Spacer()

            switch updateChecker.state {
            case .downloading(let progress):
                HStack(spacing: 8) {
                    ProgressView()
                        .controlSize(.small)
                    Text("\(Int(progress * 100))%")
                        .font(.system(size: 12, weight: .medium, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 8)

            case .readyToInstall:
                Button {
                    updateChecker.relaunchApp()
                } label: {
                    HStack(spacing: 5) {
                        Image(systemName: "arrow.triangle.2.circlepath")
                            .font(.system(size: 10, weight: .medium))
                        Text("Relaunch")
                            .font(.system(size: 12, weight: .semibold))
                    }
                    .foregroundStyle(.white)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 8)
                    .background(.green, in: Capsule())
                }
                .buttonStyle(.plain)

            case .error(let message):
                HStack(spacing: 6) {
                    Image(systemName: "exclamationmark.triangle")
                        .font(.system(size: 10))
                        .foregroundStyle(.orange)
                    Text(message)
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }

            default:
                // Download / Update button
                Button {
                    Task {
                        await updateChecker.downloadAndInstall()
                    }
                } label: {
                    HStack(spacing: 5) {
                        Image(systemName: "arrow.down.circle")
                            .font(.system(size: 10, weight: .medium))
                        Text("Update Now")
                            .font(.system(size: 12, weight: .semibold))
                    }
                    .foregroundStyle(.white)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 8)
                    .background(.blue, in: Capsule())
                }
                .buttonStyle(.plain)
            }
        }
    }
}

// MARK: - Markdown Web View (renders GitHub-flavored Markdown)

struct MarkdownWebView: NSViewRepresentable {
    let markdown: String

    func makeNSView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        let webView = WKWebView(frame: .zero, configuration: config)
        webView.setValue(false, forKey: "drawsBackground")
        loadMarkdown(in: webView)
        return webView
    }

    func updateNSView(_ webView: WKWebView, context: Context) {
        loadMarkdown(in: webView)
    }

    private func loadMarkdown(in webView: WKWebView) {
        // Convert markdown to basic HTML with styling that matches the app
        let html = """
        <!DOCTYPE html>
        <html>
        <head>
        <meta charset="utf-8">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, sans-serif;
                font-size: 13px;
                line-height: 1.6;
                color: rgba(255,255,255,0.85);
                background: transparent;
                padding: 16px;
                -webkit-font-smoothing: antialiased;
            }
            h1, h2, h3 {
                font-weight: 600;
                margin-top: 16px;
                margin-bottom: 8px;
                color: rgba(255,255,255,0.95);
            }
            h1 { font-size: 18px; }
            h2 { font-size: 15px; }
            h3 { font-size: 13px; }
            p { margin-bottom: 10px; }
            ul, ol {
                padding-left: 20px;
                margin-bottom: 10px;
            }
            li { margin-bottom: 4px; }
            code {
                font-family: 'SF Mono', Menlo, monospace;
                font-size: 12px;
                background: rgba(255,255,255,0.08);
                padding: 2px 6px;
                border-radius: 4px;
            }
            pre {
                background: rgba(255,255,255,0.06);
                padding: 12px;
                border-radius: 8px;
                overflow-x: auto;
                margin-bottom: 12px;
            }
            pre code {
                background: none;
                padding: 0;
            }
            a {
                color: #5ac8fa;
                text-decoration: none;
            }
            a:hover { text-decoration: underline; }
            hr {
                border: none;
                border-top: 1px solid rgba(255,255,255,0.1);
                margin: 16px 0;
            }
            blockquote {
                border-left: 3px solid rgba(255,255,255,0.15);
                padding-left: 12px;
                color: rgba(255,255,255,0.6);
                margin-bottom: 10px;
            }
            strong { color: rgba(255,255,255,0.95); font-weight: 600; }
            img { max-width: 100%; border-radius: 8px; margin: 8px 0; display: block; }
        </style>
        </head>
        <body>
        \(markdownToHTML(markdown))
        </body>
        </html>
        """
        webView.loadHTMLString(html, baseURL: nil)
    }

    /// Basic markdown-to-HTML conversion (handles common GitHub release note patterns)
    private func markdownToHTML(_ md: String) -> String {
        var html = md

        // Escape HTML entities first
        html = html.replacingOccurrences(of: "&", with: "&amp;")
        html = html.replacingOccurrences(of: "<", with: "&lt;")
        html = html.replacingOccurrences(of: ">", with: "&gt;")

        // Code blocks (``` ... ```)
        let codeBlockPattern = "```[\\w]*\\n([\\s\\S]*?)```"
        if let regex = try? NSRegularExpression(pattern: codeBlockPattern) {
            let range = NSRange(html.startIndex..., in: html)
            html = regex.stringByReplacingMatches(in: html, range: range, withTemplate: "<pre><code>$1</code></pre>")
        }

        // Inline code (before other inline patterns so backtick content is protected)
        let inlineCodePattern = "`([^`]+)`"
        if let regex = try? NSRegularExpression(pattern: inlineCodePattern) {
            let range = NSRange(html.startIndex..., in: html)
            html = regex.stringByReplacingMatches(in: html, range: range, withTemplate: "<code>$1</code>")
        }

        // Process line by line for block-level elements
        let lines = html.components(separatedBy: "\n")
        var processedLines: [String] = []
        var inList = false

        for line in lines {
            var processed = line

            if processed.hasPrefix("### ") {
                if inList { processedLines.append("</ul>"); inList = false }
                processed = "<h3>\(applyInlineFormatting(String(processed.dropFirst(4))))</h3>"
            } else if processed.hasPrefix("## ") {
                if inList { processedLines.append("</ul>"); inList = false }
                processed = "<h2>\(applyInlineFormatting(String(processed.dropFirst(3))))</h2>"
            } else if processed.hasPrefix("# ") {
                if inList { processedLines.append("</ul>"); inList = false }
                processed = "<h1>\(applyInlineFormatting(String(processed.dropFirst(2))))</h1>"
            } else if processed.hasPrefix("- ") || processed.hasPrefix("* ") {
                if !inList { processedLines.append("<ul>"); inList = true }
                processed = "<li>\(applyInlineFormatting(String(processed.dropFirst(2))))</li>"
            } else if processed.hasPrefix("---") || processed.hasPrefix("***") {
                if inList { processedLines.append("</ul>"); inList = false }
                processed = "<hr>"
            } else if processed.trimmingCharacters(in: .whitespaces).isEmpty {
                if inList { processedLines.append("</ul>"); inList = false }
                processed = ""
            } else {
                if inList { processedLines.append("</ul>"); inList = false }
                processed = "<p>\(applyInlineFormatting(processed))</p>"
            }

            processedLines.append(processed)
        }
        if inList { processedLines.append("</ul>") }

        return processedLines.joined(separator: "\n")
    }

    /// Apply inline markdown formatting (bold, italic, images, links) to any text
    private func applyInlineFormatting(_ text: String) -> String {
        var result = text

        // Images: ![alt](url) — must be before links since links pattern would partially match
        let imagePattern = "!\\[([^\\]]*)\\]\\(([^)]+)\\)"
        if let regex = try? NSRegularExpression(pattern: imagePattern) {
            let range = NSRange(result.startIndex..., in: result)
            result = regex.stringByReplacingMatches(in: result, range: range, withTemplate: "<img src=\"$2\" alt=\"$1\" style=\"max-width:100%;border-radius:8px;margin:8px 0;\">")
        }

        // Bold: **text**
        let boldPattern = "\\*\\*(.+?)\\*\\*"
        if let regex = try? NSRegularExpression(pattern: boldPattern) {
            let range = NSRange(result.startIndex..., in: result)
            result = regex.stringByReplacingMatches(in: result, range: range, withTemplate: "<strong>$1</strong>")
        }

        // Italic: *text*
        let italicPattern = "\\*(.+?)\\*"
        if let regex = try? NSRegularExpression(pattern: italicPattern) {
            let range = NSRange(result.startIndex..., in: result)
            result = regex.stringByReplacingMatches(in: result, range: range, withTemplate: "<em>$1</em>")
        }

        // Links: [text](url)
        let linkPattern = "\\[([^\\]]+)\\]\\(([^)]+)\\)"
        if let regex = try? NSRegularExpression(pattern: linkPattern) {
            let range = NSRange(result.startIndex..., in: result)
            result = regex.stringByReplacingMatches(in: result, range: range, withTemplate: "<a href=\"$2\" target=\"_blank\">$1</a>")
        }

        return result
    }
}
