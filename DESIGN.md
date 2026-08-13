# AiTranscribe Design Playbook

This file is an agent-facing design guide for AiTranscribe. Treat it like a local skill: read it before changing SwiftUI screens, especially settings, sessions, model controls, automation, onboarding, dashboard, history, and polish flows.

The current target is not a generic macOS settings app. It is a dense, premium, low-friction transcription workspace. The best recent example is the Session Automation and Session Detail control redesign: compact capsules, thin material, clear hierarchy, and controls that feel purpose-built instead of default SwiftUI.

## 1. Product Atmosphere

AiTranscribe should feel like a precise desktop tool for people who record, transcribe, summarize, polish, and reuse spoken work. The interface must be fast to scan, calm under load, and confident enough that advanced workflows do not feel complicated.

Design dials:

- Visual density: 6/10. Dense enough for an operator tool, not so dense that controls collide.
- Layout variance: 5/10. Use structured rows and compact groups, but avoid flat form grids that make every control equal.
- Motion: 3/10. SwiftUI state changes can animate softly, but never distract from recording or processing state.
- Material: thin macOS material with restrained borders. No heavy shadows, no glossy fake depth.
- Personality: technical, quiet, polished. Avoid marketing surfaces inside working screens.

The app should make advanced choices feel natural: Local vs Public API, model selection, batch count, RAM budget, summary length, presets, and automation should all look like first-class product controls.

## 2. Core Principle

The main design move is to replace default form widgets with small, semantic control capsules.

Default SwiftUI controls like plain `Picker`, segmented `Picker`, `Stepper`, and `Slider` are acceptable for prototypes, but they often become visually uneven in this app:

- Model names have different lengths, so native pickers resize or misalign.
- Segmented controls look rectangular and system-default.
- Steppers feel like generic macOS preferences controls, not app-specific command controls.
- Sliders take too much horizontal space for values like RAM that are normally adjusted in predictable increments.

Instead, use the AiTranscribe control grammar:

- Source choices are pill toggles.
- Model choices are fixed-width capsule menus.
- Counts and budgets are compact pill steppers.
- Metadata is shown as small info pills.
- Cards are thin-material containers only when grouping is useful.

## 3. Source Of Truth Examples

Use these files as canonical examples before inventing new controls:

- `AiTranscribe/AiTranscribe/Settings/SessionsSettingsView.swift`
  - `automationSourcePills` around line 620.
  - `automationSelectPill` around line 713.
  - `automationIntegerStepperPill` around line 735.
  - `automationDoubleStepperPill` around line 783.
  - `automationLengthPills` around line 830.
  - `permissionPill` around line 424.
  - `automationPresetChip` around line 1015.
- `AiTranscribe/AiTranscribe/Settings/SessionDetailView.swift`
  - `summaryGenerationCard` around line 1109.
  - `detailSourcePills` around line 1711.
  - `detailSelectPill` around line 1746.
  - `detailIntegerStepperPill` around line 1768.
  - `detailDoubleStepperPill` around line 1816.
  - `glassCard` around line 1863.
  - `capsuleButton` around line 1877.
  - `metaPill` around line 1898.
- `AiTranscribe/AiTranscribe/Settings/SessionCardView.swift`
  - Session row density and metadata pill rhythm.
- `AiTranscribe/AiTranscribe/Settings/DashboardView.swift`
  - Larger visual hierarchy, dashboard cards, stats density, and chart rhythm.

If you are touching another page, first inspect the closest example above and copy the design pattern. Do not rebuild the control language from scratch.

## 4. Color And Material Rules

AiTranscribe mostly uses semantic SwiftUI colors so it survives dark mode, materials, and macOS appearance changes. Keep that. Avoid hard-coded palettes unless there is a real reason.

Preferred tokens:

- `Color.accentColor`: only strong accent. Use for selected pill segments, primary actions, and focused states.
- `.primary`: main labels and selected menu text.
- `.secondary`: normal secondary labels, icons, and unselected pill text.
- `.tertiary`: field labels, helper metadata, weak timestamps.
- `.quaternary`: small metadata pill backgrounds and low-weight affordances.
- `.thinMaterial`: permission/status pills and grouped cards.
- `Color.primary.opacity(0.075)`: capsule control background.
- `Color.primary.opacity(0.08)`: small circular plus/minus button background.
- `.white.opacity(0.06)`: thin capsule/card stroke.
- `Color.green`: success, permission granted, completed.
- `Color.orange`: warning, setup, runtime, in-progress caution.
- `Color.red`: destructive or permission missing.

Banned visual moves:

- No neon glows.
- No purple/blue gradient AI aesthetic.
- No pure black backgrounds.
- No large decorative orbs, bokeh, blobs, or abstract gradients.
- No colorful accents competing with `Color.accentColor`.
- No heavy shadows on settings controls.

## 5. Typography Rules

Use the system font deliberately. This is a native macOS SwiftUI app, so the goal is not to import a web font. The polish comes from size, weight, spacing, and hierarchy.

Use these sizes as defaults:

- Screen title: `font(.system(size: 22-24, weight: .bold))`.
- Section header: `SettingsSectionHeader` where available.
- Card title: `font(.system(size: 13, weight: .semibold))`.
- Control label: `font(.system(size: 9, weight: .semibold, design: .rounded))` with `.tracking(0.5)`.
- Normal control text: `font(.system(size: 11, weight: .semibold))`.
- Button text: `font(.system(size: 10-12, weight: .medium or .semibold))`.
- Metadata pill: `font(.system(size: 9))`.
- Numeric values in compact controls: `font(.system(size: 10, weight: .semibold, design: .monospaced))`.

All labels should be short. Prefer `PUBLIC MODEL`, `LOCAL MODEL`, `SOURCE`, `LENGTH`, `PRESETS`, `RAM`, `Batches`, `Words`. Avoid explanatory labels that wrap unless they are helper text below a control.

## 6. Layout Rules

Design for a working desktop window, including the app's common 50 percent UI scale testing. At that scale, bad spacing becomes obvious.

Rules:

- Keep controls in rows, but give every row a clear label zone and a control zone.
- Use fixed widths for menu pills so long model names truncate instead of resizing the layout.
- Use `lineLimit(1)` and `truncationMode(.tail)` for model names.
- Use `Spacer(minLength: 0)` or explicit spacers to push controls without forcing compression.
- Keep vertical card spacing around `10-14`.
- Keep internal card padding around `14-16`.
- Use `HStack(spacing: 8-18)` depending on hierarchy.
- Use `VStack(alignment: .leading, spacing: 6-12)` for labels plus controls.
- Do not put a large settings panel above a list when it forces scrolling before the user can see primary content.
- Prefer a compact button in the toolbar/header that opens a sheet for advanced settings.

The Auto Process redesign is the model here: the always-visible surface is a compact pill button with a switch and status; the heavy configuration lives in a sheet.

## 7. Component Recipes

### 7.1 Source Pills

Use for binary source choices like Local/Public API or Local LLM/Public LLM. This is better than a segmented picker because it keeps the exact capsule shape and selected fill.

Visual specs:

- Outer `HStack(spacing: 2)`.
- Outer `.padding(3)`.
- Outer background: `Color.primary.opacity(0.075)` in `Capsule()`.
- Outer stroke: `Capsule().strokeBorder(.white.opacity(0.06), lineWidth: 0.5)`.
- Segment height: `26`.
- Text size: `11`, semibold.
- Selected text: white.
- Unselected text: secondary.
- Selected background: `Color.accentColor` in `Capsule()`.
- Animation: `.easeOut(duration: 0.12)`.

Use this structure:

```swift
HStack(spacing: 2) {
    Button {
        withAnimation(.easeOut(duration: 0.12)) {
            selection.wrappedValue = false
        }
    } label: {
        Text("Local")
            .font(.system(size: 11, weight: .semibold))
            .foregroundStyle(!selection.wrappedValue ? .white : .secondary)
            .frame(width: 78, height: 26)
            .background(!selection.wrappedValue ? AnyShapeStyle(Color.accentColor) : AnyShapeStyle(.clear), in: Capsule())
    }
    .buttonStyle(.plain)
}
.padding(3)
.background(Color.primary.opacity(0.075), in: Capsule())
.overlay(Capsule().strokeBorder(.white.opacity(0.06), lineWidth: 0.5))
```

Preferred widths:

- `Local`: 78.
- `Public API`: 96.
- `Local LLM`: 92.
- `Public LLM`: 104.
- Short labels can be 58-78.

Do not use native `.pickerStyle(.segmented)` for these source switches.

### 7.2 Select Pills

Use for model selection, provider selection, preset-like menus, and any long selection with a dropdown.

Visual specs:

- Height: `30`.
- Width: fixed, usually `300` or `330`.
- Horizontal padding: `11`.
- Icon size: `9`, semibold, secondary.
- Text size: `11`, semibold, primary.
- Chevron size: `8`, bold, tertiary.
- Background: `Color.primary.opacity(0.075)` in `Capsule()`.
- Stroke: `.white.opacity(0.06)`, `0.5`.
- Content shape: `Capsule()`.
- Menu style: `.menuStyle(.borderlessButton)`.

Use this structure:

```swift
Menu {
    ForEach(models) { model in
        Button(model.displayName) {
            selectedModelId = model.id
        }
    }
} label: {
    HStack(spacing: 7) {
        Image(systemName: "sparkles")
            .font(.system(size: 9, weight: .semibold))
            .foregroundStyle(.secondary)
        Text(selectedModel?.displayName ?? "Choose model")
            .font(.system(size: 11, weight: .semibold))
            .foregroundStyle(.primary)
            .lineLimit(1)
            .truncationMode(.tail)
        Spacer(minLength: 8)
        Image(systemName: "chevron.down")
            .font(.system(size: 8, weight: .bold))
            .foregroundStyle(.tertiary)
    }
    .padding(.horizontal, 11)
    .frame(width: 330, height: 30)
    .background(Color.primary.opacity(0.075), in: Capsule())
    .overlay(Capsule().strokeBorder(.white.opacity(0.06), lineWidth: 0.5))
    .contentShape(Capsule())
}
.menuStyle(.borderlessButton)
```

Width guidance:

- API model with provider suffix: `330`.
- Local model without provider suffix: `300`.
- Small option menus: `220-260`, but only if text cannot become long.

Do not let menus auto-size based on selected text.

### 7.3 Compact Integer Stepper Pill

Use for batch counts, word counts, chunk counts, retry counts, or any bounded integer setting.

Visual specs:

- Height: `30`.
- Outer horizontal padding: `9`.
- Label width: `48`.
- Label font: `9`, semibold, secondary.
- Plus/minus circles: `18x18`.
- Plus/minus icon: `8`, bold.
- Value font: `10`, semibold, monospaced.
- Value width: `42-44`.
- Outer background: `Color.primary.opacity(0.075)` in `Capsule()`.
- Circle background: `Color.primary.opacity(0.08)`.
- Stroke: `.white.opacity(0.06)`, `0.5`.

Use for public transcription batch target:

```swift
detailIntegerStepperPill(
    label: "Batches",
    value: manualPublicBatchBinding,
    in: 0...30,
    step: 6,
    zeroLabel: "Auto"
)
```

Behavior rule:

- If `0` has semantic meaning, show a word like `Auto`, not `0`.
- Normalize values in the binding setter if the backend requires multiples.
- Disable plus/minus at range boundaries.

### 7.4 Compact Double Stepper Pill

Use for RAM budget and similar bounded decimal values.

Visual specs:

- Same as integer stepper.
- Label width: `32`.
- Value width: `66`.
- Format with one decimal place.
- Use monospaced numeric text.

Use for RAM:

```swift
detailDoubleStepperPill(
    label: "RAM",
    value: ramBudgetBinding(for: session),
    in: minRamBudget...maxRamBudget,
    step: 0.5,
    suffix: "GB"
)
```

Behavior rule:

- Clamp in the binding setter.
- Refresh estimates in the setter when the value affects runtime planning.
- Prefer this over a slider when users need discrete understandable values.

### 7.5 Length Pills

Use for summary length choices: Short, Medium, Long, Custom.

Visual specs:

- Outer capsule same as source pills.
- Segment height: `26`.
- Segment width: `58` for normal options, `70` for Custom.
- Font: `10`, semibold when selected, medium when unselected.
- Selected segment: `Color.accentColor`.
- Do not use native segmented picker.

When Custom is selected, show a compact integer stepper next to the pills. Do not show the custom word count all the time.

### 7.6 Info And Metadata Pills

Use for status, duration, batch count, model RAM, file size, provider, and generated state.

Visual specs:

- Icon size: `8-9`.
- Text size: `9-10`.
- Horizontal padding: `6-10`.
- Vertical padding: `3-5`.
- Background: `.quaternary` for simple metadata, `.thinMaterial` for permission/status controls.
- Shape: `Capsule()`.
- Status dot: `Circle().frame(width: 5-6, height: 5-6)`.

Good examples:

- Permission pill: status dot plus label plus optional `Grant`.
- Session card meta pill: duration/file size/transcribed state.
- Summary metadata: target words plus model memory or API model.

Do not create square badges for metadata unless the surrounding design already uses square chips.

### 7.7 Glass Cards

Use cards only to group a coherent control cluster or status block. Cards should never become nested boxes.

Visual specs:

```swift
VStack(alignment: .leading, spacing: 10) {
    content()
}
.padding(14)
.background(.thinMaterial, in: .rect(cornerRadius: 14, style: .continuous))
.overlay(
    RoundedRectangle(cornerRadius: 14, style: .continuous)
        .strokeBorder(.white.opacity(0.06), lineWidth: 0.5)
)
```

Rules:

- Use `14` radius for working cards.
- Use cards for settings groups, not for every row.
- Avoid nested cards. Use dividers or spacing inside a card.
- If a control group gets tall enough to push primary content away, move it into a sheet.

### 7.8 Primary Actions

Use capsule buttons for primary and secondary actions.

Primary:

- White text.
- `Color.accentColor` fill.
- Horizontal padding `14-16`.
- Vertical padding `8-9`.
- Icon size `10-12`.
- Text size `11-12`, medium or semibold.

Secondary:

- Primary or secondary text.
- `.quaternary` fill.
- Capsule.
- Compact padding.

Destructive:

- Red foreground.
- `Color.red.opacity(0.1)` fill.
- Capsule.

Do not use rectangular bordered buttons for core workflows.

## 8. Page-Level Patterns

### Sessions Page

Sessions should keep the list visible. Avoid placing large configuration panels above the sessions list.

Correct pattern:

- Header area shows title, count, storage, permission/status pills.
- Advanced settings appear as compact pill buttons in the top-right area.
- Configuration opens in a sheet.
- Date chips sit above session rows.
- Session rows remain dense and scannable.

Wrong pattern:

- Large automation controls embedded above date filters.
- Tall cards that require scrolling before users can see sessions.
- Default pickers and steppers mixed with custom pills.

### Session Detail

Manual session controls should mirror automation controls.

Untranscribed state:

- Show Local/Public API source pills first.
- Then show the mode-specific card.
- For Public API: fixed-width API model menu, batch pill stepper, batch estimate meta.
- For Local: fixed-width local model menu, RAM pill stepper, batch/time estimate meta.
- Keep primary action as a capsule below the card.

Summary Studio:

- Keep `Summary Studio` as the product area title.
- Generate row first.
- Divider.
- Source pills plus model pill row.
- Length pills below.
- Presets remain pill/card-like and selectable.

### Dashboard

Dashboard can be more visual, but must stay restrained:

- Use cards for important metric panels.
- Use monospaced numbers where values are dense.
- Keep chart labels secondary or tertiary.
- Do not add decorative gradients just to make metrics feel exciting.
- If using color in charts, keep it functional.

### Onboarding

Onboarding can be more expressive than settings, but still must not become a marketing site.

Good pattern:

- One focused visual idea per step.
- Strong hierarchy.
- Real product controls and previews.
- Microphone preview and audio visuals should feel native and responsive.

Avoid:

- Equal generic feature cards.
- Long explanatory text blocks.
- Decorative gradients as the main content.

## 9. State Design

Every workflow screen needs all relevant states:

- Empty: explain what is missing and show the next action.
- Loading: compact progress, not a giant spinner unless the whole screen is blocked.
- Disabled: visible reason nearby when possible.
- Error: inline message with recovery path.
- Active processing: progress, current stage, cancel if safe.
- Completed: status pill plus open/export actions.

Examples:

- No local models: show icon, short text, `Go to Models`.
- No public API models: show icon, short text, `Go to API Models`.
- Summary unavailable before transcription: show compact warning/info row.
- Runtime missing: show install action and progress.

Do not show a blank disabled UI without telling users what to do next.

## 10. Naming And Copy

Copy should be concrete and short.

Good:

- `After session ends`
- `Auto Process`
- `Public API`
- `Local LLM`
- `Batches`
- `RAM`
- `Runs only after the automatic transcript succeeds.`
- `Public session transcription uploads API batches in parallel and stitches them back in order.`

Avoid:

- Marketing words like `seamless`, `next-gen`, `elevate`, `unlock`.
- Explaining obvious UI mechanics.
- Long labels inside capsules.
- Ambiguous source names like `Cloud` if the product uses `Public API`.

Use `Public API` for transcription source and `Public LLM` for summary source. Use `Local` and `Local LLM` where the distinction matters.

## 11. SwiftUI Implementation Rules

Prefer small private view helpers inside the relevant view file when the pattern is local to one screen. Extract shared components only when the same code appears across multiple screens and the abstraction is clearly stable.

Rules:

- Use `Button(...).buttonStyle(.plain)` for custom capsule controls.
- Use `Menu` plus a capsule label for dropdowns.
- Use `Binding` wrappers to normalize values and trigger side effects.
- Use `@AppStorage` for preferences that should persist.
- Use fixed frame widths for pills with dynamic text.
- Keep comments sparse. Comment behavior only when the reason is not obvious.
- Preserve existing view structure and state ownership unless the task requires architecture changes.

When changing a control from native SwiftUI to custom:

1. Preserve the binding and side effects.
2. Preserve disabled behavior.
3. Preserve task/onChange logic.
4. Check long text truncation.
5. Check local and public modes.
6. Build with Xcode.

## 12. Design Review Checklist

Before finishing any UI change, check:

- Are the main controls capsules rather than default rectangles?
- Does every dropdown have fixed width and truncation?
- Does the selected state use `Color.accentColor` and white text?
- Are unselected states secondary, not low-contrast tertiary?
- Are labels outside controls and aligned consistently?
- Are numbers monospaced in compact controls?
- Are plus/minus buttons disabled at bounds?
- Does the layout still work at 50 percent UI scale?
- Does the list or primary content remain visible without unnecessary scrolling?
- Are cards used only for useful grouping?
- Are there no nested cards?
- Are there no neon glows, purple gradients, or decorative blobs?
- Are empty/loading/error states covered?
- Did you run a Debug build after SwiftUI edits?
- If distribution output matters, did you run `./build_production.sh --skip-backend` for Swift-only changes?

## 13. Common Fixes

If a picker looks uneven:

- Replace it with a fixed-width `Menu` label using the select pill recipe.
- Keep the selected model name line-limited and tail-truncated.

If a segmented control looks too system-default:

- Replace it with source pills or length pills.
- Use capsule backgrounds and explicit selected state.

If a slider makes the layout wide or imprecise:

- Replace it with a double stepper pill.
- Use discrete increments and monospaced value text.

If a settings block pushes content down:

- Move advanced controls into a sheet.
- Keep only a compact status/control pill in the header.

If a row feels visually noisy:

- Reduce label copy.
- Move explanation below as tertiary helper text.
- Convert secondary values into metadata pills.

If a button looks like default macOS:

- Use `buttonStyle(.plain)`.
- Build your own capsule label.
- Use semantic accent/secondary/destructive styles.

## 14. What Not To Do

Do not:

- Add a full-width settings panel above sessions or history lists.
- Use native rectangular segmented pickers in polished settings sections.
- Let long model names resize controls.
- Put custom controls and default controls next to each other in the same row.
- Use large cards inside cards.
- Center every control just because the row has space.
- Hide core status behind icons only.
- Add tooltips as a substitute for clear labels.
- Add visual effects that do not explain product state.
- Introduce a new color language for one feature.

## 15. The Design Standard

The desired result is not "pretty UI." It is operational clarity with product taste.

A good AiTranscribe screen should feel like this:

- The user can identify the current mode in one glance.
- Advanced controls are compact but not cryptic.
- Processing state is visible and calm.
- The default path is obvious.
- Every control feels intentionally designed for this app.
- The screen does not collapse when model names get long.
- The first visible area remains useful.

When in doubt, copy the Auto Process and Session Detail pill patterns, then adjust the label and binding. That is the current design bar.
