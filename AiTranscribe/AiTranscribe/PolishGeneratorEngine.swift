import Foundation

// =============================================================================
// Multi-step Polish generator.
//
// Replaces the single-shot "request in, preset out" call with a pipeline of
// small, single-purpose LLM calls so that small/fast models stay reliable:
//
//   1. understand   — goal summary, or clarification questions
//   2. plan         — component plan + eval criteria (structured JSON)
//      (deterministic validation runs in code between steps)
//   3. write        — the runtime prompt, name, icon
//   4. eval flow    — pass/fail per criterion, judged against the plan
//   5. eval prompt  — pass/fail per criterion, judged against the prompt
//   6. repair loop  — failed criteria fed back, bounded to 2 rounds
//
// The result is a reviewable draft package (plan + prompt + scorecard), not a
// silently saved preset. The view owns save/discard.
// =============================================================================

// MARK: - Progress

enum PolishGeneratorPhase: Equatable {
    case understanding
    case planning
    case writingPrompt
    case evaluatingFlow
    case evaluatingPrompt
    case repairing(round: Int)

    var displayText: String {
        switch self {
        case .understanding: return "Step 1/5 · Understanding the goal"
        case .planning: return "Step 2/5 · Planning components"
        case .writingPrompt: return "Step 3/5 · Writing the prompt"
        case .evaluatingFlow: return "Step 4/5 · Evaluating the flow"
        case .evaluatingPrompt: return "Step 5/5 · Evaluating the prompt"
        case .repairing(let round): return "Repairing draft · round \(round)"
        }
    }
}

// MARK: - Step payloads

private struct GeneratorGoalAnalysis: Decodable {
    let isClear: Bool
    let goalSummary: String?
    let reason: String?
    let questions: [GeneratedPolishClarificationQuestion]?

    enum CodingKeys: String, CodingKey {
        case isClear = "is_clear"
        case goalSummary = "goal_summary"
        case reason
        case questions
    }
}

private struct GeneratorComponentPlan: Decodable {
    let inputModules: [String]
    let outputModules: [String]
    let pasteBehavior: String?
    let needsVision: Bool?
    let rationale: String?
    let evalCriteria: [String]

    enum CodingKeys: String, CodingKey {
        case inputModules = "input_modules"
        case outputModules = "output_modules"
        case pasteBehavior = "paste_behavior"
        case needsVision = "needs_vision"
        case rationale
        case evalCriteria = "eval_criteria"
    }
}

private struct GeneratorPromptDraft: Decodable {
    let displayName: String
    let symbolName: String
    let prompt: String

    enum CodingKeys: String, CodingKey {
        case displayName = "display_name"
        case symbolName = "symbol_name"
        case prompt
    }
}

struct PolishGeneratorEvalCheck: Decodable, Identifiable, Equatable {
    var id: String { criterion }
    let criterion: String
    let pass: Bool
    let reason: String

    enum CodingKeys: String, CodingKey {
        case criterion
        case pass
        case reason
    }
}

private struct GeneratorEvalResult: Decodable {
    let checks: [PolishGeneratorEvalCheck]
}

// MARK: - Result package

/// Everything the review screen needs to show a generated draft before the
/// user decides to keep it.
struct PolishGeneratorDraftPackage: Equatable {
    var displayName: String
    var symbolName: String
    var prompt: String
    var inputModules: [PolishInputModule]
    var outputModules: [PolishOutputModule]
    var pasteBehavior: PolishPasteBehavior
    var rationale: String?
    var goalSummary: String?
    var checks: [PolishGeneratorEvalCheck]
    var deterministicWarnings: [String]
    var repairRounds: Int

    var passedCount: Int { checks.filter(\.pass).count }
    var allPassed: Bool { checks.allSatisfy(\.pass) && deterministicWarnings.isEmpty }
}

enum PolishGeneratorOutcome {
    case clarification(GeneratedPolishClarificationDraft)
    case draft(PolishGeneratorDraftPackage)
}

/// The state of an existing Polish that "Edit with AI" starts from.
struct PolishEditSnapshot {
    var displayName: String
    var symbolName: String
    var prompt: String
    var inputModules: [PolishInputModule]
    var outputModules: [PolishOutputModule]
    var pasteBehavior: PolishPasteBehavior
}

enum PolishGeneratorEngineError: LocalizedError {
    case malformedStepOutput(step: String)

    var errorDescription: String? {
        switch self {
        case .malformedStepOutput(let step):
            return "The model returned unusable output during the \(step) step. Try again, or use a stronger LLM for generation."
        }
    }
}

// MARK: - LLM transport

/// Thin routing wrapper so the engine does not care which provider runs a step.
/// `.local` runs through the bundled backend's /polish/local endpoint, so the
/// generator follows the app's Local mode instead of silently using API models.
enum PolishGeneratorLLM {
    case api(model: APILanguageModel, apiKey: String)
    case local(modelId: String)

    func complete(system: String, user: String, maxTokens: Int) async throws -> String {
        switch self {
        case .api(let model, let apiKey):
            if model.providerId == TranscriptionProviderDescriptor.openRouter.id {
                return try await OpenRouterLanguageModelClient().generateText(
                    system: system,
                    user: user,
                    modelId: model.modelId,
                    apiKey: apiKey,
                    providerOnly: model.openRouterProviderOnly,
                    maxOutputTokens: maxTokens
                )
            }
            return try await OpenAILanguageModelClient().generateText(
                system: system,
                user: user,
                modelId: model.modelId,
                apiKey: apiKey,
                maxOutputTokens: maxTokens
            )

        case .local(let modelId):
            var latestText = ""
            var failureMessage: String?
            try await APIClient().polishLocally(
                inputText: user,
                modelId: modelId,
                presetId: "polish-generator-step",
                displayName: "Polish Generator",
                systemPrompt: system,
                maxOutputTokens: maxTokens
            ) { event in
                switch event.event {
                case "partial":
                    if let text = event.text, !text.isEmpty { latestText = text }
                case "done":
                    latestText = event.text ?? latestText
                case "error":
                    failureMessage = event.message ?? "Local generation failed."
                case "cancelled":
                    failureMessage = "Local generation cancelled."
                default:
                    break
                }
            }
            if let failureMessage {
                throw APIError.serverError(failureMessage)
            }
            let text = latestText.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else {
                throw PolishGeneratorEngineError.malformedStepOutput(step: "local generation")
            }
            return text
        }
    }
}

// MARK: - Engine

struct PolishGeneratorEngine {
    let llm: PolishGeneratorLLM
    /// The model the finished preset will run on, used for deterministic
    /// vision checks. Usually the same as `llm.model`.
    /// The model the finished preset will run on, used for deterministic
    /// vision checks. Display name plus vision flag so both API and local
    /// models fit.
    let targetModelName: String
    let targetSupportsVision: Bool

    private static let maxRepairRounds = 2

    func run(
        request: String,
        onPhase: @escaping @MainActor (PolishGeneratorPhase) -> Void
    ) async throws -> PolishGeneratorOutcome {
        // ---- Step 1: understand the goal, or ask -----------------------------
        await onPhase(.understanding)
        let analysisText = try await llm.complete(
            system: Self.understandInstruction,
            user: request,
            maxTokens: 1_400
        )
        let analysis: GeneratorGoalAnalysis = try Self.decode(analysisText, step: "understanding")
        if !analysis.isClear, let questions = analysis.questions, !questions.isEmpty {
            return .clarification(
                GeneratedPolishClarificationDraft(reason: analysis.reason, questions: questions)
            )
        }
        let goalSummary = analysis.goalSummary?.trimmingCharacters(in: .whitespacesAndNewlines)
        let goal = (goalSummary?.isEmpty == false) ? goalSummary! : request

        // ---- Step 2: plan components + eval criteria -------------------------
        try Task.checkCancellation()
        await onPhase(.planning)
        var (plan, warnings) = try await generatePlan(goal: goal, request: request, feedback: nil)

        // ---- Step 3: write the prompt ----------------------------------------
        try Task.checkCancellation()
        await onPhase(.writingPrompt)
        var draft = try await generatePrompt(goal: goal, request: request, plan: plan, feedback: nil)

        // ---- Steps 4-6: evaluate, repair, re-evaluate ------------------------
        let package = try await evaluateAndRepair(
            goal: goal,
            request: request,
            goalSummary: goalSummary,
            initialPlan: plan,
            initialWarnings: warnings,
            initialDraft: draft,
            onPhase: onPhase,
            regeneratePlan: { feedback in
                try await self.generatePlan(goal: goal, request: request, feedback: feedback)
            },
            regeneratePrompt: { newPlan, feedback in
                try await self.generatePrompt(goal: goal, request: request, plan: newPlan, feedback: feedback)
            }
        )
        return .draft(package)
    }

    /// Edit flow: same eval/repair harness, but planning and prompt writing
    /// start from an existing preset and are instructed to change only what
    /// the user asked for.
    func runEdit(
        current: PolishEditSnapshot,
        instruction: String,
        onPhase: @escaping @MainActor (PolishGeneratorPhase) -> Void
    ) async throws -> PolishGeneratorDraftPackage {
        let goal = """
        Update an existing Polish named "\(current.displayName)". Requested change: \(instruction). Everything the change does not touch must keep working exactly as before.
        """
        let request = """
        Current Polish:
        \(Self.snapshotDescription(current))

        Requested change:
        \(instruction)
        """

        await onPhase(.planning)
        let (plan, warnings) = try await generateEditPlan(current: current, instruction: instruction, feedback: nil)

        try Task.checkCancellation()
        await onPhase(.writingPrompt)
        let draft = try await generateEditedPrompt(current: current, instruction: instruction, plan: plan, feedback: nil)

        return try await evaluateAndRepair(
            goal: goal,
            request: request,
            goalSummary: goal,
            initialPlan: plan,
            initialWarnings: warnings,
            initialDraft: draft,
            onPhase: onPhase,
            regeneratePlan: { feedback in
                try await self.generateEditPlan(current: current, instruction: instruction, feedback: feedback)
            },
            regeneratePrompt: { newPlan, feedback in
                try await self.generateEditedPrompt(current: current, instruction: instruction, plan: newPlan, feedback: feedback)
            }
        )
    }

    // MARK: Shared eval + repair loop

    private func evaluateAndRepair(
        goal: String,
        request: String,
        goalSummary: String?,
        initialPlan: ValidatedPlan,
        initialWarnings: [String],
        initialDraft: GeneratorPromptDraft,
        onPhase: @escaping @MainActor (PolishGeneratorPhase) -> Void,
        regeneratePlan: (String) async throws -> (ValidatedPlan, [String]),
        regeneratePrompt: (ValidatedPlan, String?) async throws -> GeneratorPromptDraft
    ) async throws -> PolishGeneratorDraftPackage {
        var plan = initialPlan
        var warnings = initialWarnings
        var draft = initialDraft
        var round = 0

        while true {
            try Task.checkCancellation()
            await onPhase(.evaluatingFlow)
            let flowChecks = try await evaluate(
                system: Self.flowEvalInstruction,
                goal: goal,
                request: request,
                plan: plan,
                artifactLabel: "Component plan JSON",
                artifact: Self.planDescription(plan)
            )

            try Task.checkCancellation()
            await onPhase(.evaluatingPrompt)
            let promptChecks = try await evaluate(
                system: Self.promptEvalInstruction,
                goal: goal,
                request: request,
                plan: plan,
                artifactLabel: "Runtime prompt",
                artifact: draft.prompt
            )

            let combined = Self.combineChecks(flow: flowChecks, prompt: promptChecks)
            let failed = combined.filter { !$0.pass }
            if failed.isEmpty || round >= Self.maxRepairRounds {
                return PolishGeneratorDraftPackage(
                    displayName: draft.displayName,
                    symbolName: draft.symbolName,
                    prompt: draft.prompt,
                    inputModules: plan.resolvedInputs,
                    outputModules: plan.resolvedOutputs,
                    pasteBehavior: plan.resolvedPaste,
                    rationale: plan.rationale,
                    goalSummary: goalSummary,
                    checks: combined,
                    deterministicWarnings: warnings,
                    repairRounds: round
                )
            }

            // ---- Repair: regenerate only what failed -------------------------
            round += 1
            await onPhase(.repairing(round: round))
            let feedback = Self.repairFeedback(failed: failed)
            if flowChecks.contains(where: { !$0.pass }) {
                (plan, warnings) = try await regeneratePlan(feedback)
            }
            draft = try await regeneratePrompt(plan, feedback)
        }
    }

    // MARK: Step runners

    private func generatePlan(
        goal: String,
        request: String,
        feedback: String?
    ) async throws -> (ValidatedPlan, [String]) {
        var user = """
        Goal summary:
        \(goal)

        Full user request:
        \(request)
        """
        if let feedback {
            user += "\n\nA previous attempt failed evaluation. Fix these specific problems:\n\(feedback)"
        }
        let text = try await llm.complete(system: Self.planInstruction, user: user, maxTokens: 1_800)
        let plan: GeneratorComponentPlan = try Self.decode(text, step: "planning")
        return Self.validate(plan: plan, targetModelName: targetModelName, targetSupportsVision: targetSupportsVision)
    }

    private func generatePrompt(
        goal: String,
        request: String,
        plan: ValidatedPlan,
        feedback: String?
    ) async throws -> GeneratorPromptDraft {
        var user = """
        Goal summary:
        \(goal)

        Full user request:
        \(request)

        Component plan (already fixed, do not change it, write the prompt for it):
        \(Self.planDescription(plan))
        """
        if let feedback {
            user += "\n\nA previous prompt failed evaluation. Fix these specific problems:\n\(feedback)"
        }
        let text = try await llm.complete(system: Self.writeInstruction, user: user, maxTokens: 2_400)
        return try Self.decode(text, step: "prompt writing")
    }

    private func generateEditPlan(
        current: PolishEditSnapshot,
        instruction: String,
        feedback: String?
    ) async throws -> (ValidatedPlan, [String]) {
        var user = """
        Current Polish:
        \(Self.snapshotDescription(current))

        Requested change:
        \(instruction)
        """
        if let feedback {
            user += "\n\nA previous attempt failed evaluation. Fix these specific problems:\n\(feedback)"
        }
        let text = try await llm.complete(system: Self.editPlanInstruction, user: user, maxTokens: 1_800)
        let plan: GeneratorComponentPlan = try Self.decode(text, step: "edit planning")
        return Self.validate(plan: plan, targetModelName: targetModelName, targetSupportsVision: targetSupportsVision)
    }

    private func generateEditedPrompt(
        current: PolishEditSnapshot,
        instruction: String,
        plan: ValidatedPlan,
        feedback: String?
    ) async throws -> GeneratorPromptDraft {
        var user = """
        Current Polish:
        \(Self.snapshotDescription(current))

        Requested change:
        \(instruction)

        New component plan (already fixed, do not change it, adapt the prompt to it):
        \(Self.planDescription(plan))
        """
        if let feedback {
            user += "\n\nA previous prompt failed evaluation. Fix these specific problems:\n\(feedback)"
        }
        let text = try await llm.complete(system: Self.editWriteInstruction, user: user, maxTokens: 2_400)
        return try Self.decode(text, step: "prompt editing")
    }

    private static func snapshotDescription(_ snapshot: PolishEditSnapshot) -> String {
        """
        display_name: \(snapshot.displayName)
        symbol_name: \(snapshot.symbolName)
        input_modules: \(snapshot.inputModules.map(\.rawValue).joined(separator: ", "))
        output_modules: \(snapshot.outputModules.map(\.rawValue).joined(separator: ", "))
        paste_behavior: \(snapshot.pasteBehavior.rawValue)
        runtime prompt:
        \(snapshot.prompt)
        """
    }

    private func evaluate(
        system: String,
        goal: String,
        request: String,
        plan: ValidatedPlan,
        artifactLabel: String,
        artifact: String
    ) async throws -> [PolishGeneratorEvalCheck] {
        let criteriaList = plan.evalCriteria.enumerated()
            .map { "\($0.offset + 1). \($0.element)" }
            .joined(separator: "\n")
        let user = """
        Goal summary:
        \(goal)

        Full user request:
        \(request)

        Eval criteria (judge every one of these, in order):
        \(criteriaList)

        \(artifactLabel):
        \(artifact)
        """
        let text = try await llm.complete(system: system, user: user, maxTokens: 1_800)
        let result: GeneratorEvalResult = try Self.decode(text, step: "evaluation")
        return result.checks
    }

    // MARK: Deterministic validation (code, not LLM)

    struct ValidatedPlan {
        var resolvedInputs: [PolishInputModule]
        var resolvedOutputs: [PolishOutputModule]
        var resolvedPaste: PolishPasteBehavior
        var rationale: String?
        var evalCriteria: [String]
    }

    private static func validate(
        plan: GeneratorComponentPlan,
        targetModelName: String,
        targetSupportsVision: Bool
    ) -> (ValidatedPlan, [String]) {
        var warnings: [String] = []

        var inputs = plan.inputModules.compactMap { PolishInputModule(rawValue: $0) }
        let unknownInputs = plan.inputModules.filter { PolishInputModule(rawValue: $0) == nil }
        if !unknownInputs.isEmpty {
            warnings.append("Dropped unknown input modules: \(unknownInputs.joined(separator: ", ")).")
        }
        if inputs.isEmpty {
            inputs = [.voiceTranscript]
            warnings.append("Plan named no valid input modules, so Voice Transcript was used.")
        }

        var outputs = plan.outputModules.compactMap { PolishOutputModule(rawValue: $0) }
            .filter { PolishOutputModule.editableCases.contains($0) }
        let unknownOutputs = plan.outputModules.filter { PolishOutputModule(rawValue: $0) == nil }
        if !unknownOutputs.isEmpty {
            warnings.append("Dropped unknown output modules: \(unknownOutputs.joined(separator: ", ")).")
        }
        if outputs.isEmpty {
            outputs = [.copyToClipboard]
            warnings.append("Plan named no valid output modules, so Copy to Clipboard was used.")
        }

        var paste = plan.pasteBehavior.flatMap { PolishPasteBehavior(rawValue: $0) } ?? .copyOnly

        // The same constraints the import path enforces.
        if outputs.contains(.copyTextAndScreenshot), outputs.contains(.copyToClipboard) {
            outputs.removeAll { $0 == .copyTextAndScreenshot }
            warnings.append("Copy Text and Screenshot cannot combine with Copy to Clipboard, so it was removed.")
        }
        if outputs.contains(.copyTextAndScreenshot) {
            let prerequisitesMet = inputs.contains(.screenSnapshot)
                && inputs.contains(.activeApp)
                && paste == .pasteIntoApp
            if !prerequisitesMet {
                outputs.removeAll { $0 == .copyTextAndScreenshot }
                if outputs.isEmpty { outputs = [.copyToClipboard] }
                warnings.append("Copy Text and Screenshot needs Screenshot + Active App inputs and auto-paste, so it was removed.")
            }
        }
        if !outputs.contains(where: { $0 == .copyToClipboard || $0 == .copyTextAndScreenshot }),
           paste == .pasteIntoApp {
            paste = .copyOnly
            warnings.append("Auto-paste needs a clipboard output, so paste behavior was set to copy only.")
        }
        if inputs.contains(.screenSnapshot), !targetSupportsVision {
            warnings.append("This plan uses Screenshot input, but \(targetModelName) is not marked vision-capable. Enable Vision on the model card or pick another LLM before running it.")
        }

        var criteria = plan.evalCriteria
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        if criteria.isEmpty {
            criteria = ["The workflow produces what the user asked for."]
            warnings.append("Plan produced no eval criteria, so a generic one was used.")
        }

        return (
            ValidatedPlan(
                resolvedInputs: inputs,
                resolvedOutputs: outputs,
                resolvedPaste: paste,
                rationale: plan.rationale,
                evalCriteria: criteria
            ),
            warnings
        )
    }

    // MARK: Helpers

    private static func planDescription(_ plan: ValidatedPlan) -> String {
        """
        input_modules: \(plan.resolvedInputs.map(\.rawValue).joined(separator: ", "))
        output_modules: \(plan.resolvedOutputs.map(\.rawValue).joined(separator: ", "))
        paste_behavior: \(plan.resolvedPaste.rawValue)
        rationale: \(plan.rationale ?? "none given")
        """
    }

    /// A criterion fails overall if either eval failed it. Reasons come from
    /// the failing side so the repair step sees the actual complaint.
    private static func combineChecks(
        flow: [PolishGeneratorEvalCheck],
        prompt: [PolishGeneratorEvalCheck]
    ) -> [PolishGeneratorEvalCheck] {
        let promptByCriterion = Dictionary(prompt.map { ($0.criterion, $0) }, uniquingKeysWith: { a, _ in a })
        var combined: [PolishGeneratorEvalCheck] = []
        var seen = Set<String>()
        for check in flow {
            seen.insert(check.criterion)
            if let other = promptByCriterion[check.criterion] {
                let pass = check.pass && other.pass
                let reason = pass ? check.reason : (!check.pass ? check.reason : other.reason)
                combined.append(PolishGeneratorEvalCheck(criterion: check.criterion, pass: pass, reason: reason))
            } else {
                combined.append(check)
            }
        }
        for check in prompt where !seen.contains(check.criterion) {
            combined.append(check)
        }
        return combined
    }

    private static func repairFeedback(failed: [PolishGeneratorEvalCheck]) -> String {
        failed.map { "- \($0.criterion): \($0.reason)" }.joined(separator: "\n")
    }

    /// Lenient JSON decode: strips code fences, then falls back to the first
    /// top-level object in the text, because small models wrap JSON in prose.
    private static func decode<T: Decodable>(_ text: String, step: String) throws -> T {
        let stripped = text
            .replacingOccurrences(of: "```json", with: "")
            .replacingOccurrences(of: "```", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if let data = stripped.data(using: .utf8),
           let value = try? JSONDecoder().decode(T.self, from: data) {
            return value
        }
        if let start = stripped.firstIndex(of: "{"),
           let end = stripped.lastIndex(of: "}"),
           start < end {
            let slice = String(stripped[start...end])
            if let data = slice.data(using: .utf8),
               let value = try? JSONDecoder().decode(T.self, from: data) {
                return value
            }
        }
        throw PolishGeneratorEngineError.malformedStepOutput(step: step)
    }
}

// =============================================================================
// MARK: - Step instructions
//
// Decomposed from the old single-shot `polishPresetGeneratorInstruction`. The
// shared module reference keeps the semantics in ONE place; every step prompt
// that needs it embeds it from here.
// =============================================================================

extension PolishGeneratorEngine {

    /// What a Polish IS in the app's runtime, and what is none of its business.
    /// Small models invent triggers, buttons, and file outputs without this.
    static let domainPrimer = """
    What a Polish is:
    A Polish is a PROCESSING RECIPE inside AiTranscribe, a macOS voice app. The app owns the entire outer workflow: the user triggers a recording with the app's own hotkeys, the app records and transcribes the audio, and only THEN the Polish runs. The Polish receives the already-collected context (transcripts and optional extras), processes it with an LLM, and hands the result to one of the app's fixed output surfaces. Then it is done.

    A Polish therefore has NO say in, and you must never ask about or mention:
    - how recording is triggered (hotkeys, buttons, voice commands, automatic detection)
    - how long the recording is, or when it starts and stops
    - saving to files, notifications, reminders, scheduling, or any OS integration
    - which LLM or model runs it, or app settings of any kind
    The only things a Polish controls are: which context inputs it uses, what the LLM is instructed to do with them, and which of the fixed output surfaces shows the result.
    """

    /// Shared vocabulary: what a Polish is made of. Embedded in the planning,
    /// writing, and eval instructions so they can never drift apart.
    static let moduleReference = """
    A Polish preset is a voice workflow with these parts:

    Available input_modules, raw values exactly:
    - voiceTranscript: microphone speech from the user. Usually enabled unless the workflow is clearly context-only.
    - desktopAudio: system/desktop audio transcript, separately tagged from the user's voice.
    - clipboardText: exact text currently copied by the user.
    - screenSnapshot: selected screenshot/image region. Requires a vision-capable LLM.
    - activeApp: name of the focused app, used for formatting and destination-specific behavior.

    Available output_modules, raw values exactly:
    - copyToClipboard: copy final text output only.
    - copyTextAndScreenshot: paste screenshot first, then final text. Only valid when screenSnapshot and activeApp are inputs and paste_behavior is pasteIntoApp.
    - floatingWindow: show the final response in the floating response window. Use for explainers, Q&A, diagnostics, and outputs the user wants to read rather than paste.

    Available paste_behavior values:
    - copyOnly: leave the result on the clipboard or in the floating window. Default and safest.
    - pasteIntoApp: automatically paste into the focused app after the result is ready. Only when the user explicitly asks for auto-paste or insertion.

    Hard constraints:
    - copyTextAndScreenshot requires screenSnapshot and activeApp in input_modules and paste_behavior pasteIntoApp.
    - Never include both copyToClipboard and copyTextAndScreenshot; choose one.
    - screenSnapshot requires a vision-capable LLM at runtime.
    """

    static let understandInstruction = """
    You are step 1 of a Polish preset builder for AiTranscribe. Your ONLY job is to decide whether the user's request is clear enough to build a Polish from, and to summarize the goal. You do not design the workflow and you do not write prompts.

    \(domainPrimer)

    \(moduleReference)

    If the request is clear enough, return strict JSON only:
    {
      "is_clear": true,
      "goal_summary": "2-4 sentences stating what the user wants the Polish to do, which of the available inputs it uses, and which available output surface shows the result, in concrete terms."
    }

    If material workflow details are missing, return strict JSON only:
    {
      "is_clear": false,
      "reason": "Short reason questions are needed",
      "questions": [
        {"id": "q1", "question": "What should this Polish output?", "suggested_answers": ["Clean final text", "A prompt for another AI", "A readable explanation", "Structured notes"]}
      ]
    }

    Clarification rules:
    - Ask at most 5 questions. Prefer 2-4.
    - A question is only allowed if it resolves something a Polish can actually express: what the LLM should produce, which of the available inputs are needed (desktop audio / clipboard / screenshot / active app), which available output surface to use (floating window, copy to clipboard, paste into the active app), tone or format of the result, or a genuine edge case in the processing itself.
    - FORBIDDEN questions, never ask these: how the Polish is triggered, hotkeys, buttons, recording length or timing, saving to files, notifications, model or app settings, or anything else from the "no say" list above. The app handles all of that.
    - Every suggested answer must be something the components above can actually deliver. Never offer "save to a file", "voice command", "automatic detection", or any capability that does not exist in the lists.
    - Do not ask about anything the request already answers or clearly implies. If the user said they want an explanation in a window, do not ask what the output should be.
    - Only ask when the missing detail materially changes the workflow. Do not ask about obvious defaults.
    - If the request already contains a section titled "Clarification answers:", the questions were already asked and answered. Treat it as clear unless the answers still make the workflow impossible to define.
    """

    static let planInstruction = """
    You are step 2 of a Polish preset builder for AiTranscribe. Your ONLY job is to plan the components of the workflow and to write the eval criteria the finished workflow will be judged against. You do not write the runtime prompt.

    \(domainPrimer)

    \(moduleReference)

    Selection guidance:
    - If the user mentions calls, meetings, or what another person is saying, include desktopAudio.
    - If the user mentions copying context, plans, or documents beforehand, include clipboardText.
    - If the user wants to point at something visible on screen, include screenSnapshot and set needs_vision true.
    - If the output is meant to be read in place (explainers, Q&A, meeting help), prefer floatingWindow.
    - If the output is meant to be pasted or reused as text, prefer copyToClipboard.

    Eval criteria rules, this is the important part:
    - Write 5 to 9 criteria derived from the user's OWN words.
    - Each criterion must be a single concrete, checkable requirement, phrased so a yes/no judgment is possible.
    - Cover: every input the user implied, the output surface they asked for, and what the runtime prompt must instruct (tone, task, what to do when context is missing).
    - Bad criterion: "The workflow is good." Good criterion: "The workflow captures desktop audio so the other speaker in the call is transcribed."
    - Never write a criterion about triggering, hotkeys, recording length, files, or anything from the "no say" list. Those are the app's job and cannot be judged here.

    Return strict JSON only:
    {
      "input_modules": ["voiceTranscript"],
      "output_modules": ["floatingWindow"],
      "paste_behavior": "copyOnly",
      "needs_vision": false,
      "rationale": "One short paragraph explaining why these components serve the goal.",
      "eval_criteria": ["..."]
    }
    """

    static let writeInstruction = """
    You are step 3 of a Polish preset builder for AiTranscribe. Your ONLY job is to write the runtime prompt, the display name, and the icon for a workflow whose components are already decided. Do not change the component plan.

    \(domainPrimer)

    \(moduleReference)

    Critical mental model:
    - The prompt you write is executed later on real runtime context. It is NOT a prompt about creating a Polish.
    - If the final output is meant to be pasted into another AI model, the prompt should generate a clean final instruction/prompt.
    - If the final output is meant for the user to read, the prompt should generate the final answer/explanation directly.

    Runtime prompt quality rules:
    - Write the prompt as a direct instruction to the runtime LLM applying the Polish.
    - Reference exactly the context sections that the plan's input_modules provide: voice transcript, desktop/system audio transcript, exact clipboard context, screenshot context, active app. Do not reference context the plan does not include.
    - Some context sections may be missing at runtime; instruct the runtime LLM not to invent missing context.
    - Prioritize the user's voice transcript as intent when present.
    - Treat clipboard as exact source context, not a vague "above/below" reference.
    - Treat desktop audio as separate from the user's voice.
    - If screenSnapshot is in the plan, instruct the runtime LLM to inspect the actual visible image and connect it to the user request.
    - If activeApp is in the plan, instruct the runtime LLM to adapt formatting to the focused app: Markdown for Obsidian, VS Code, Cursor, and markdown-friendly apps; clean plain text for Notes, Messages, Mail, and plain-text fields; Markdown-friendly by default when unknown.
    - For explainer workflows: start with the direct answer, explain simply first, add depth only when helpful, and preserve readable paragraphs. Never one giant paragraph.
    - Return only the final usable result at runtime, with no preamble.

    Return strict JSON only:
    {
      "display_name": "Short title under 24 characters",
      "symbol_name": "a common SF Symbol, for example sparkles, wand.and.stars, text.badge.checkmark, note.text, checklist, doc.text, quote.bubble, brain.head.profile, envelope, lightbulb, target, questionmark.bubble, photo.on.rectangle",
      "prompt": "The detailed runtime instruction"
    }
    """

    static let editPlanInstruction = """
    You are the planning step of a Polish preset EDITOR for AiTranscribe. An existing, working Polish and a requested change are provided. Your ONLY job is to produce the FULL new component plan and the eval criteria the edited Polish will be judged against. You do not write the runtime prompt.

    \(domainPrimer)

    \(moduleReference)

    Editing rules, this is the important part:
    - Change ONLY what the requested change requires. Every component the change does not touch must stay exactly as it is in the current Polish.
    - Return the complete plan, not a diff: all input_modules and output_modules the edited Polish should have, kept ones included.
    - If the requested change is impossible with the available components, keep the current plan and say why in the rationale.

    Eval criteria rules:
    - Write 5 to 9 criteria total.
    - 1-3 criteria must verify the requested change is actually applied.
    - 2-4 criteria must verify the existing behavior is preserved, derived from the current Polish (for example "still shows the result in the floating window").
    - Each criterion must be a single concrete, checkable requirement, phrased so a yes/no judgment is possible.
    - Never write a criterion about triggering, hotkeys, recording length, files, or anything from the "no say" list.

    Return strict JSON only:
    {
      "input_modules": ["voiceTranscript"],
      "output_modules": ["floatingWindow"],
      "paste_behavior": "copyOnly",
      "needs_vision": false,
      "rationale": "One short paragraph explaining what changed and what was kept.",
      "eval_criteria": ["..."]
    }
    """

    static let editWriteInstruction = """
    You are the prompt step of a Polish preset EDITOR for AiTranscribe. An existing Polish, a requested change, and the already-decided new component plan are provided. Your ONLY job is to update the runtime prompt, name, and icon. Do not change the component plan.

    \(domainPrimer)

    \(moduleReference)

    Editing rules:
    - Modify the existing runtime prompt MINIMALLY: keep its wording, tone, and structure wherever the change does not require touching them.
    - Adapt the prompt to the new plan: reference exactly the context sections its input_modules provide, and never context they do not.
    - Keep display_name and symbol_name unchanged unless the requested change explicitly asks to rename or re-icon the Polish.
    - Some context sections may be missing at runtime; keep or add the instruction not to invent missing context.
    - Return only the final usable result at runtime, with no preamble.

    Return strict JSON only:
    {
      "display_name": "the current name unless asked to change it",
      "symbol_name": "the current SF Symbol unless asked to change it",
      "prompt": "The updated runtime instruction"
    }
    """

    static let flowEvalInstruction = """
    You are step 4 of a Polish preset builder: a strict evaluator. Your ONLY job is to judge whether the COMPONENT PLAN satisfies each eval criterion. You are not judging the prompt text here.

    \(domainPrimer)

    \(moduleReference)

    Rules:
    - Judge every criterion you are given, in order, one check per criterion. Copy the criterion text verbatim into "criterion".
    - pass is a strict yes/no: does the component plan (inputs, outputs, paste behavior) support this criterion?
    - If a criterion is only about the runtime prompt's wording and no component could affect it, return pass true with reason "Prompt-stage criterion."
    - Reasons are one short sentence. Failures must name what is missing or wrong.
    - Do not invent new criteria. Do not skip any.

    Return strict JSON only:
    {
      "checks": [
        {"criterion": "...", "pass": true, "reason": "..."}
      ]
    }
    """

    static let promptEvalInstruction = """
    You are step 5 of a Polish preset builder: a strict evaluator. Your ONLY job is to judge whether the RUNTIME PROMPT satisfies each eval criterion. The component plan is provided as context only; it was already judged separately.

    \(domainPrimer)

    \(moduleReference)

    Rules:
    - Judge every criterion you are given, in order, one check per criterion. Copy the criterion text verbatim into "criterion".
    - pass is a strict yes/no: does the prompt text actually instruct the runtime LLM to satisfy this criterion?
    - If a criterion is only about which components are selected and the prompt cannot affect it, return pass true with reason "Flow-stage criterion."
    - Also fail a criterion if the prompt references context the plan does not provide, or invents details the user never asked for.
    - Reasons are one short sentence. Failures must name the missing or wrong instruction.
    - Do not invent new criteria. Do not skip any.

    Return strict JSON only:
    {
      "checks": [
        {"criterion": "...", "pass": true, "reason": "..."}
      ]
    }
    """
}
