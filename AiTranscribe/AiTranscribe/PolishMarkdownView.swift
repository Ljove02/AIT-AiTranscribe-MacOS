import SwiftUI

struct PolishMarkdownView: View {
    let text: String

    private var blocks: [PolishMarkdownBlock] {
        PolishMarkdownParser.parse(text)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            ForEach(Array(blocks.enumerated()), id: \.offset) { _, block in
                blockView(block)
            }
        }
    }

    @ViewBuilder
    private func blockView(_ block: PolishMarkdownBlock) -> some View {
        switch block {
        case .heading(let level, let content):
            inlineText(content)
                .font(.system(size: headingSize(level), weight: .bold, design: .rounded))
                .foregroundStyle(.primary)
                .padding(.top, level == 1 ? 4 : 2)
                .padding(.bottom, 1)
        case .paragraph(let content):
            inlineText(content)
                .font(.system(size: 14))
                .lineSpacing(4)
                .foregroundStyle(.primary)
        case .math(let content):
            PolishMathBlockView(source: content)
        case .bullet(let content, let depth):
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text("•")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(.secondary)
                inlineText(content)
                    .font(.system(size: 14))
                    .lineSpacing(3)
                    .foregroundStyle(.primary)
            }
            .padding(.leading, CGFloat(depth * 14))
        case .numbered(let marker, let content, let depth):
            HStack(alignment: .firstTextBaseline, spacing: 7) {
                Text(marker)
                    .font(.system(size: 13, weight: .semibold, design: .rounded))
                    .foregroundStyle(.secondary)
                inlineText(content)
                    .font(.system(size: 14))
                    .lineSpacing(3)
                    .foregroundStyle(.primary)
            }
            .padding(.leading, CGFloat(depth * 14))
        case .quote(let content):
            inlineText(content)
                .font(.system(size: 13))
                .lineSpacing(3)
                .foregroundStyle(.secondary)
                .padding(.leading, 12)
                .overlay(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 1, style: .continuous)
                        .fill(Color.secondary.opacity(0.35))
                        .frame(width: 3)
                }
        case .code(let content):
            ScrollView(.horizontal, showsIndicators: false) {
                Text(content)
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundStyle(.primary)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(12)
            }
            .background(Color.primary.opacity(0.055), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
        case .divider:
            Rectangle()
                .fill(Color.secondary.opacity(0.18))
                .frame(height: 1)
                .padding(.vertical, 4)
        case .spacer:
            Spacer()
                .frame(height: 2)
        }
    }

    private func inlineText(_ content: String) -> Text {
        let renderedContent = PolishMathRenderer.inline(content)
        if let attributed = try? AttributedString(
            markdown: renderedContent,
            options: AttributedString.MarkdownParsingOptions(interpretedSyntax: .full)
        ) {
            return Text(attributed)
        }
        return Text(renderedContent)
    }

    private func headingSize(_ level: Int) -> CGFloat {
        switch level {
        case 1: return 19
        case 2: return 16
        default: return 14
        }
    }
}

private enum PolishMarkdownBlock {
    case heading(level: Int, content: String)
    case paragraph(String)
    case math(String)
    case bullet(content: String, depth: Int)
    case numbered(marker: String, content: String, depth: Int)
    case quote(String)
    case code(String)
    case divider
    case spacer
}

private enum PolishMarkdownParser {
    static func parse(_ text: String) -> [PolishMarkdownBlock] {
        let normalized = text.replacingOccurrences(of: "\r\n", with: "\n")
        let lines = normalized.components(separatedBy: "\n")
        var blocks: [PolishMarkdownBlock] = []
        var paragraphLines: [String] = []
        var codeLines: [String] = []
        var mathLines: [String] = []
        var isInCodeBlock = false
        var mathEndDelimiter: String?

        func flushParagraph() {
            let paragraph = paragraphLines
                .map { $0.trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty }
                .joined(separator: "\n")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !paragraph.isEmpty {
                blocks.append(.paragraph(paragraph))
            }
            paragraphLines.removeAll()
        }

        func flushMath() {
            let math = mathLines
                .joined(separator: "\n")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !math.isEmpty {
                blocks.append(.math(math))
            }
            mathLines.removeAll()
            mathEndDelimiter = nil
        }

        for rawLine in lines {
            let line = rawLine.trimmingCharacters(in: .whitespaces)

            if line.hasPrefix("```") {
                if isInCodeBlock {
                    blocks.append(.code(codeLines.joined(separator: "\n")))
                    codeLines.removeAll()
                    isInCodeBlock = false
                } else {
                    flushParagraph()
                    isInCodeBlock = true
                }
                continue
            }

            if isInCodeBlock {
                codeLines.append(rawLine)
                continue
            }

            if let delimiter = mathEndDelimiter {
                if line == delimiter {
                    flushMath()
                    continue
                }

                if let range = rawLine.range(of: delimiter) {
                    let beforeDelimiter = String(rawLine[..<range.lowerBound])
                    if !beforeDelimiter.trimmingCharacters(in: .whitespaces).isEmpty {
                        mathLines.append(beforeDelimiter)
                    }
                    flushMath()
                    let afterDelimiter = String(rawLine[range.upperBound...])
                        .trimmingCharacters(in: .whitespaces)
                    if !afterDelimiter.isEmpty {
                        paragraphLines.append(afterDelimiter)
                    }
                    continue
                }

                mathLines.append(rawLine)
                continue
            }

            if line.isEmpty {
                flushParagraph()
                if blocks.last?.isSpacer != true {
                    blocks.append(.spacer)
                }
                continue
            }

            if line == "---" || line == "***" || line == "___" {
                flushParagraph()
                blocks.append(.divider)
                continue
            }

            if let mathBlock = singleLineMathBlock(line) {
                flushParagraph()
                blocks.append(.math(mathBlock))
                continue
            }

            if let opener = mathBlockOpener(line) {
                flushParagraph()
                mathEndDelimiter = opener.endDelimiter
                if !opener.initialContent.isEmpty {
                    mathLines.append(opener.initialContent)
                }
                continue
            }

            if let heading = headingBlock(line) {
                flushParagraph()
                blocks.append(heading)
                continue
            }

            if let bullet = bulletBlock(rawLine) {
                flushParagraph()
                blocks.append(bullet)
                continue
            }

            if let numbered = numberedBlock(rawLine) {
                flushParagraph()
                blocks.append(numbered)
                continue
            }

            if line.hasPrefix("> ") {
                flushParagraph()
                blocks.append(.quote(String(line.dropFirst(2))))
                continue
            }

            paragraphLines.append(rawLine)
        }

        if isInCodeBlock {
            blocks.append(.code(codeLines.joined(separator: "\n")))
        }
        if mathEndDelimiter != nil {
            flushMath()
        }
        flushParagraph()

        return blocks.removingEdgeSpacers()
    }

    private static func singleLineMathBlock(_ line: String) -> String? {
        if line.hasPrefix("$$"), line.hasSuffix("$$"), line.count > 4 {
            return String(line.dropFirst(2).dropLast(2)).trimmingCharacters(in: .whitespaces)
        }

        if line.hasPrefix("\\["), line.hasSuffix("\\]"), line.count > 4 {
            return String(line.dropFirst(2).dropLast(2)).trimmingCharacters(in: .whitespaces)
        }

        return nil
    }

    private static func mathBlockOpener(_ line: String) -> (endDelimiter: String, initialContent: String)? {
        if line.hasPrefix("$$") {
            return ("$$", String(line.dropFirst(2)).trimmingCharacters(in: .whitespaces))
        }

        if line.hasPrefix("\\[") {
            return ("\\]", String(line.dropFirst(2)).trimmingCharacters(in: .whitespaces))
        }

        return nil
    }

    private static func headingBlock(_ line: String) -> PolishMarkdownBlock? {
        let markerCount = line.prefix(while: { $0 == "#" }).count
        guard (1...6).contains(markerCount),
              line.dropFirst(markerCount).first == " " else {
            return nil
        }
        let content = line.dropFirst(markerCount + 1).trimmingCharacters(in: .whitespaces)
        return .heading(level: min(markerCount, 3), content: content)
    }

    private static func bulletBlock(_ rawLine: String) -> PolishMarkdownBlock? {
        let trimmed = rawLine.trimmingCharacters(in: .whitespaces)
        guard trimmed.hasPrefix("- ") || trimmed.hasPrefix("* ") || trimmed.hasPrefix("• ") else {
            return nil
        }
        let depth = leadingSpaceDepth(rawLine)
        return .bullet(content: String(trimmed.dropFirst(2)), depth: depth)
    }

    private static func numberedBlock(_ rawLine: String) -> PolishMarkdownBlock? {
        let trimmed = rawLine.trimmingCharacters(in: .whitespaces)
        guard let range = trimmed.range(of: #"^\d+\.\s+"#, options: .regularExpression) else {
            return nil
        }
        let marker = String(trimmed[range]).trimmingCharacters(in: .whitespaces)
        let content = String(trimmed[range.upperBound...])
        return .numbered(marker: marker, content: content, depth: leadingSpaceDepth(rawLine))
    }

    private static func leadingSpaceDepth(_ line: String) -> Int {
        min(line.prefix(while: { $0 == " " || $0 == "\t" }).count / 2, 4)
    }
}

private extension PolishMarkdownBlock {
    var isSpacer: Bool {
        if case .spacer = self { return true }
        return false
    }
}

private struct PolishMathBlockView: View {
    let source: String

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            Group {
                if let matrix = PolishMathRenderer.matrix(from: source) {
                    HStack(alignment: .center, spacing: 8) {
                        if !matrix.prefix.isEmpty {
                            PolishMathInlineView(latex: matrix.prefix, fontSize: 16)
                        }

                        Text(matrix.leftBracket)
                            .font(.system(size: 34, weight: .light, design: .serif))
                            .foregroundStyle(.secondary)

                        VStack(alignment: .center, spacing: 7) {
                            ForEach(Array(matrix.rows.enumerated()), id: \.offset) { _, row in
                                HStack(alignment: .center, spacing: 14) {
                                    ForEach(Array(row.enumerated()), id: \.offset) { _, cell in
                                        PolishMathInlineView(latex: cell, fontSize: 15)
                                            .frame(minWidth: 22)
                                    }
                                }
                            }
                        }

                        Text(matrix.rightBracket)
                            .font(.system(size: 34, weight: .light, design: .serif))
                            .foregroundStyle(.secondary)

                        if !matrix.suffix.isEmpty {
                            PolishMathInlineView(latex: matrix.suffix, fontSize: 16)
                        }
                    }
                } else {
                    PolishMathInlineView(latex: source, fontSize: 16)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
        }
        .background(Color.primary.opacity(0.045), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
    }
}

private struct PolishRenderedMatrix {
    let rows: [[String]]
    let leftBracket: String
    let rightBracket: String
    let prefix: String
    let suffix: String
}

private struct PolishMathInlineView: View {
    let latex: String
    let fontSize: CGFloat

    private var atoms: [PolishMathAtom] {
        PolishMathParser.parse(latex)
    }

    var body: some View {
        PolishMathRowView(atoms: atoms, fontSize: fontSize)
            .foregroundStyle(.primary)
            .textSelection(.enabled)
    }
}

private struct PolishMathRowView: View {
    let atoms: [PolishMathAtom]
    let fontSize: CGFloat

    var body: some View {
        HStack(alignment: .center, spacing: 2) {
            ForEach(Array(atoms.enumerated()), id: \.offset) { _, atom in
                PolishMathAtomView(atom: atom, fontSize: fontSize)
            }
        }
    }
}

private struct PolishMathAtomView: View {
    let atom: PolishMathAtom
    let fontSize: CGFloat

    var body: some View {
        switch atom {
        case .text(let text, let isBold):
            Text(text)
                .font(.system(size: fontSize, weight: isBold ? .semibold : .medium, design: .serif))
        case .fraction(let numerator, let denominator):
            VStack(spacing: 2) {
                PolishMathRowView(atoms: numerator, fontSize: max(10, fontSize * 0.78))
                Rectangle()
                    .fill(Color.primary.opacity(0.72))
                    .frame(height: 1)
                PolishMathRowView(atoms: denominator, fontSize: max(10, fontSize * 0.78))
            }
            .fixedSize()
            .padding(.horizontal, 1)
        case .sqrt(let radicand):
            HStack(alignment: .center, spacing: 2) {
                Text("√")
                    .font(.system(size: fontSize * 1.05, weight: .regular, design: .serif))
                PolishMathRowView(atoms: radicand, fontSize: fontSize * 0.92)
                    .padding(.top, 2)
                    .overlay(alignment: .top) {
                        Rectangle()
                            .fill(Color.primary.opacity(0.72))
                            .frame(height: 1)
                    }
            }
            .fixedSize()
        case .scripted(let base, let subscriptAtoms, let superscriptAtoms):
            HStack(alignment: .center, spacing: 1) {
                PolishMathAtomView(atom: base, fontSize: fontSize)
                VStack(alignment: .leading, spacing: -1) {
                    if let superscriptAtoms {
                        PolishMathRowView(atoms: superscriptAtoms, fontSize: max(9, fontSize * 0.62))
                    }
                    if superscriptAtoms == nil {
                        Spacer().frame(height: fontSize * 0.12)
                    }
                    if let subscriptAtoms {
                        PolishMathRowView(atoms: subscriptAtoms, fontSize: max(9, fontSize * 0.62))
                    }
                    if subscriptAtoms == nil {
                        Spacer().frame(height: fontSize * 0.12)
                    }
                }
                .fixedSize()
            }
            .fixedSize()
        }
    }
}

private indirect enum PolishMathAtom {
    case text(String, isBold: Bool = false)
    case fraction(numerator: [PolishMathAtom], denominator: [PolishMathAtom])
    case sqrt([PolishMathAtom])
    case scripted(base: PolishMathAtom, subscriptAtoms: [PolishMathAtom]?, superscriptAtoms: [PolishMathAtom]?)

    func applyingBold() -> PolishMathAtom {
        switch self {
        case .text(let text, _):
            return .text(text, isBold: true)
        case .fraction(let numerator, let denominator):
            return .fraction(
                numerator: numerator.map { $0.applyingBold() },
                denominator: denominator.map { $0.applyingBold() }
            )
        case .sqrt(let radicand):
            return .sqrt(radicand.map { $0.applyingBold() })
        case .scripted(let base, let subscriptAtoms, let superscriptAtoms):
            return .scripted(
                base: base.applyingBold(),
                subscriptAtoms: subscriptAtoms?.map { $0.applyingBold() },
                superscriptAtoms: superscriptAtoms?.map { $0.applyingBold() }
            )
        }
    }
}

private enum PolishMathParser {
    static func parse(_ latex: String) -> [PolishMathAtom] {
        var parser = Parser(latex)
        return parser.parse()
    }

    private struct Parser {
        private let source: String
        private var index: String.Index

        init(_ source: String) {
            self.source = source
            self.index = source.startIndex
        }

        mutating func parse(stopAtBrace: Bool = false) -> [PolishMathAtom] {
            var atoms: [PolishMathAtom] = []

            while index < source.endIndex {
                let character = source[index]

                if stopAtBrace, character == "}" {
                    index = source.index(after: index)
                    break
                }

                if character == "\\" {
                    atoms.append(contentsOf: parseCommand())
                    continue
                }

                if character == "^" || character == "_" {
                    applyScript(character, to: &atoms)
                    continue
                }

                if character == "{" {
                    index = source.index(after: index)
                    atoms.append(contentsOf: parse(stopAtBrace: true))
                    continue
                }

                if character == "}" {
                    index = source.index(after: index)
                    continue
                }

                atoms.append(.text(String(character)))
                index = source.index(after: index)
            }

            return coalescedTextAtoms(atoms)
        }

        private mutating func parseCommand() -> [PolishMathAtom] {
            index = source.index(after: index)
            guard index < source.endIndex else { return [.text("\\")] }

            if source[index] == "\\" {
                index = source.index(after: index)
                return [.text(" ")]
            }

            let commandStart = index
            while index < source.endIndex, source[index].isLetter {
                index = source.index(after: index)
            }

            let command = String(source[commandStart..<index])
            if command.isEmpty {
                let symbol = String(source[index])
                index = source.index(after: index)
                return symbol == "," || symbol == ";" || symbol == ":" || symbol == "!" ? [.text(" ")] : [.text(symbol)]
            }

            switch command {
            case "left", "right":
                return []
            case "frac", "dfrac", "tfrac":
                let numerator = parseRequiredGroup()
                let denominator = parseRequiredGroup()
                return [.fraction(numerator: numerator, denominator: denominator)]
            case "sqrt":
                return [.sqrt(parseRequiredGroup())]
            case "mathbf", "boldsymbol", "bm", "mathrm", "text":
                return parseRequiredGroup().map { command == "mathrm" || command == "text" ? $0 : $0.applyingBold() }
            case "begin", "end":
                _ = parseRequiredGroup()
                return []
            default:
                return [.text(symbol(for: command) ?? command)]
            }
        }

        private mutating func applyScript(_ marker: Character, to atoms: inout [PolishMathAtom]) {
            index = source.index(after: index)
            let scriptAtoms = parseScriptArgument()
            guard let base = atoms.popLast() else {
                atoms.append(.text(String(marker)))
                atoms.append(contentsOf: scriptAtoms)
                return
            }

            if case .scripted(let existingBase, let existingSubscript, let existingSuperscript) = base {
                atoms.append(.scripted(
                    base: existingBase,
                    subscriptAtoms: marker == "_" ? scriptAtoms : existingSubscript,
                    superscriptAtoms: marker == "^" ? scriptAtoms : existingSuperscript
                ))
            } else {
                atoms.append(.scripted(
                    base: base,
                    subscriptAtoms: marker == "_" ? scriptAtoms : nil,
                    superscriptAtoms: marker == "^" ? scriptAtoms : nil
                ))
            }
        }

        private mutating func parseScriptArgument() -> [PolishMathAtom] {
            skipWhitespace()
            guard index < source.endIndex else { return [] }

            if source[index] == "{" {
                index = source.index(after: index)
                return parse(stopAtBrace: true)
            }

            if source[index] == "\\" {
                return parseCommand()
            }

            let character = source[index]
            index = source.index(after: index)
            return [.text(String(character))]
        }

        private mutating func parseRequiredGroup() -> [PolishMathAtom] {
            skipWhitespace()
            guard index < source.endIndex, source[index] == "{" else { return [] }
            index = source.index(after: index)
            return parse(stopAtBrace: true)
        }

        private mutating func skipWhitespace() {
            while index < source.endIndex, source[index].isWhitespace {
                index = source.index(after: index)
            }
        }

        private func coalescedTextAtoms(_ atoms: [PolishMathAtom]) -> [PolishMathAtom] {
            var result: [PolishMathAtom] = []

            for atom in atoms {
                if case .text(let text, let isBold) = atom,
                   case .text(let previous, let previousIsBold) = result.last,
                   isBold == previousIsBold {
                    result.removeLast()
                    result.append(.text(previous + text, isBold: isBold))
                } else {
                    result.append(atom)
                }
            }

            return result
        }

        private func symbol(for command: String) -> String? {
            [
                "leqslant": "≤", "geqslant": "≥",
                "rightarrow": "→", "leftarrow": "←",
                "Rightarrow": "⇒", "Leftarrow": "⇐",
                "leftrightarrow": "↔", "Leftrightarrow": "⇔",
                "partial": "∂", "nabla": "∇", "infty": "∞",
                "approx": "≈", "neq": "≠", "leq": "≤", "geq": "≥",
                "times": "×", "cdot": "·", "div": "÷", "pm": "±", "mp": "∓",
                "sum": "∑", "prod": "∏", "int": "∫", "oint": "∮",
                "lim": "lim", "sin": "sin", "cos": "cos", "tan": "tan",
                "log": "log", "ln": "ln", "min": "min", "max": "max",
                "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ",
                "epsilon": "ε", "varepsilon": "ε", "zeta": "ζ", "eta": "η",
                "theta": "θ", "lambda": "λ", "mu": "μ", "pi": "π",
                "rho": "ρ", "sigma": "σ", "tau": "τ", "phi": "φ",
                "varphi": "φ", "omega": "ω",
                "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ",
                "Pi": "Π", "Sigma": "Σ", "Phi": "Φ", "Omega": "Ω"
            ][command]
        }
    }
}

private enum PolishMathRenderer {
    private static let commandReplacements: [(String, String)] = [
        ("\\leqslant", "≤"),
        ("\\geqslant", "≥"),
        ("\\rightarrow", "→"),
        ("\\leftarrow", "←"),
        ("\\Rightarrow", "⇒"),
        ("\\Leftarrow", "⇐"),
        ("\\leftrightarrow", "↔"),
        ("\\Leftrightarrow", "⇔"),
        ("\\partial", "∂"),
        ("\\nabla", "∇"),
        ("\\infty", "∞"),
        ("\\approx", "≈"),
        ("\\neq", "≠"),
        ("\\leq", "≤"),
        ("\\geq", "≥"),
        ("\\times", "×"),
        ("\\cdot", "·"),
        ("\\div", "÷"),
        ("\\pm", "±"),
        ("\\mp", "∓"),
        ("\\sum", "∑"),
        ("\\prod", "∏"),
        ("\\int", "∫"),
        ("\\oint", "∮"),
        ("\\lim", "lim"),
        ("\\sin", "sin"),
        ("\\cos", "cos"),
        ("\\tan", "tan"),
        ("\\log", "log"),
        ("\\ln", "ln"),
        ("\\min", "min"),
        ("\\max", "max"),
        ("\\alpha", "α"),
        ("\\beta", "β"),
        ("\\gamma", "γ"),
        ("\\delta", "δ"),
        ("\\epsilon", "ε"),
        ("\\varepsilon", "ε"),
        ("\\zeta", "ζ"),
        ("\\eta", "η"),
        ("\\theta", "θ"),
        ("\\lambda", "λ"),
        ("\\mu", "μ"),
        ("\\pi", "π"),
        ("\\rho", "ρ"),
        ("\\sigma", "σ"),
        ("\\tau", "τ"),
        ("\\phi", "φ"),
        ("\\varphi", "φ"),
        ("\\omega", "ω"),
        ("\\Gamma", "Γ"),
        ("\\Delta", "Δ"),
        ("\\Theta", "Θ"),
        ("\\Lambda", "Λ"),
        ("\\Pi", "Π"),
        ("\\Sigma", "Σ"),
        ("\\Phi", "Φ"),
        ("\\Omega", "Ω")
    ].sorted { $0.0.count > $1.0.count }

    private static let superscriptMap: [Character: Character] = [
        "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
        "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
        "+": "⁺", "-": "⁻", "=": "⁼", "(": "⁽", ")": "⁾",
        "n": "ⁿ", "i": "ⁱ", "T": "ᵀ", "x": "ˣ", "y": "ʸ", "z": "ᶻ"
    ]

    private static let subscriptMap: [Character: Character] = [
        "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄",
        "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉",
        "+": "₊", "-": "₋", "=": "₌", "(": "₍", ")": "₎",
        "a": "ₐ", "e": "ₑ", "h": "ₕ", "i": "ᵢ", "j": "ⱼ",
        "k": "ₖ", "l": "ₗ", "m": "ₘ", "n": "ₙ", "o": "ₒ",
        "p": "ₚ", "r": "ᵣ", "s": "ₛ", "t": "ₜ", "u": "ᵤ",
        "v": "ᵥ", "x": "ₓ", "y": "ᵧ", "z": "ᶻ"
    ]

    static func inline(_ text: String) -> String {
        var result = ""
        var index = text.startIndex

        while index < text.endIndex {
            if text[index] == "$" {
                let next = text.index(after: index)
                if next < text.endIndex, text[next] == "$" {
                    result.append("$$")
                    index = text.index(after: next)
                    continue
                }

                if let close = text[next...].firstIndex(of: "$") {
                    result.append(expression(String(text[next..<close])))
                    index = text.index(after: close)
                    continue
                }
            }

            if text[index] == "\\" {
                let next = text.index(after: index)
                if next < text.endIndex, text[next] == "(" {
                    let contentStart = text.index(after: next)
                    if let close = text[contentStart...].range(of: "\\)") {
                        result.append(expression(String(text[contentStart..<close.lowerBound])))
                        index = close.upperBound
                        continue
                    }
                }
            }

            result.append(text[index])
            index = text.index(after: index)
        }

        return result
    }

    static func expression(_ latex: String) -> String {
        var output = latex
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "\\left", with: "")
            .replacingOccurrences(of: "\\right", with: "")
            .replacingOccurrences(of: "\\,", with: " ")
            .replacingOccurrences(of: "\\;", with: " ")
            .replacingOccurrences(of: "\\:", with: " ")
            .replacingOccurrences(of: "\\!", with: "")

        output = renderFractions(in: output)
        output = renderSquareRoots(in: output)
        output = renderCommandGroups(in: output)

        for (command, symbol) in commandReplacements {
            output = output.replacingOccurrences(of: command, with: symbol)
        }

        output = renderScripts(in: output)
        output = output.replacingOccurrences(of: "{", with: "")
        output = output.replacingOccurrences(of: "}", with: "")
        output = output.replacingOccurrences(of: #"\\[a-zA-Z]+"#, with: "", options: .regularExpression)
        output = output.replacingOccurrences(of: #"\\ "#, with: " ", options: .regularExpression)
        output = output.replacingOccurrences(of: #" +"#, with: " ", options: .regularExpression)

        return output.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func matrix(from latex: String) -> PolishRenderedMatrix? {
        let environments: [(name: String, left: String, right: String)] = [
            ("bmatrix", "[", "]"),
            ("pmatrix", "(", ")"),
            ("matrix", "[", "]"),
            ("array", "[", "]")
        ]

        for environment in environments {
            let begin = "\\begin{\(environment.name)}"
            let end = "\\end{\(environment.name)}"
            guard let beginRange = latex.range(of: begin),
                  let endRange = latex.range(of: end),
                  beginRange.upperBound <= endRange.lowerBound else {
                continue
            }

            let body = String(latex[beginRange.upperBound..<endRange.lowerBound])
            let prefix = String(latex[..<beginRange.lowerBound])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let suffix = String(latex[endRange.upperBound...])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let rows = body
                .components(separatedBy: "\\\\")
                .map { row in
                    row
                        .components(separatedBy: "&")
                        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                        .filter { !$0.isEmpty }
                }
                .filter { !$0.isEmpty }

            guard !rows.isEmpty else { return nil }
            return PolishRenderedMatrix(
                rows: rows,
                leftBracket: environment.left,
                rightBracket: environment.right,
                prefix: prefix,
                suffix: suffix
            )
        }

        return nil
    }

    private static func renderCommandGroups(in source: String) -> String {
        var output = source
        for command in ["\\mathbf", "\\boldsymbol", "\\bm", "\\mathrm", "\\text"] {
            output = replacingCommandGroup(command, in: output)
        }
        return output
    }

    private static func replacingCommandGroup(_ command: String, in source: String) -> String {
        var output = source
        while let commandRange = output.range(of: command) {
            guard let group = bracedContent(in: output, from: commandRange.upperBound) else {
                output.replaceSubrange(commandRange, with: "")
                continue
            }
            let replacement = expression(group.content)
            output.replaceSubrange(commandRange.lowerBound..<group.endIndex, with: replacement)
        }
        return output
    }

    private static func renderFractions(in source: String) -> String {
        var output = source

        while let commandRange = output.range(of: "\\frac") {
            guard let numerator = bracedContent(in: output, from: commandRange.upperBound),
                  let denominator = bracedContent(in: output, from: numerator.endIndex) else {
                break
            }

            let replacement = "(\(expression(numerator.content)))/(\(expression(denominator.content)))"
            output.replaceSubrange(commandRange.lowerBound..<denominator.endIndex, with: replacement)
        }

        return output
    }

    private static func renderSquareRoots(in source: String) -> String {
        var output = source

        while let commandRange = output.range(of: "\\sqrt") {
            guard let radicand = bracedContent(in: output, from: commandRange.upperBound) else {
                break
            }

            let replacement = "√(\(expression(radicand.content)))"
            output.replaceSubrange(commandRange.lowerBound..<radicand.endIndex, with: replacement)
        }

        return output
    }

    private static func renderScripts(in source: String) -> String {
        var output = ""
        var index = source.startIndex

        while index < source.endIndex {
            let marker = source[index]
            guard marker == "^" || marker == "_" else {
                output.append(marker)
                index = source.index(after: index)
                continue
            }

            let contentStart = source.index(after: index)
            let scriptContent: String
            let nextIndex: String.Index

            if contentStart < source.endIndex, source[contentStart] == "{",
               let group = bracedContent(in: source, from: contentStart) {
                scriptContent = group.content
                nextIndex = group.endIndex
            } else if contentStart < source.endIndex {
                scriptContent = String(source[contentStart])
                nextIndex = source.index(after: contentStart)
            } else {
                output.append(marker)
                break
            }

            let rendered = renderScript(expression(scriptContent), using: marker == "^" ? superscriptMap : subscriptMap)
            if let rendered {
                output.append(rendered)
            } else {
                output.append(marker == "^" ? "^\(scriptContent)" : "_\(scriptContent)")
            }
            index = nextIndex
        }

        return output
    }

    private static func renderScript(_ content: String, using map: [Character: Character]) -> String? {
        let compact = content.trimmingCharacters(in: .whitespaces)
        guard !compact.isEmpty else { return nil }

        var rendered = ""
        for character in compact {
            guard let mapped = map[character] else { return nil }
            rendered.append(mapped)
        }
        return rendered
    }

    private static func bracedContent(in source: String, from start: String.Index) -> (content: String, endIndex: String.Index)? {
        var index = start
        while index < source.endIndex, source[index].isWhitespace {
            index = source.index(after: index)
        }

        guard index < source.endIndex, source[index] == "{" else { return nil }

        let contentStart = source.index(after: index)
        var depth = 1
        index = contentStart

        while index < source.endIndex {
            if source[index] == "{" {
                depth += 1
            } else if source[index] == "}" {
                depth -= 1
                if depth == 0 {
                    return (String(source[contentStart..<index]), source.index(after: index))
                }
            }
            index = source.index(after: index)
        }

        return nil
    }
}

private extension Array where Element == PolishMarkdownBlock {
    func removingEdgeSpacers() -> [PolishMarkdownBlock] {
        var result = self
        while result.first?.isSpacer == true {
            result.removeFirst()
        }
        while result.last?.isSpacer == true {
            result.removeLast()
        }
        return result
    }
}
