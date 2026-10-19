# v0.3.0 — Polish, Cloud Providers, Rolling Transcription & The Performance Release

<!-- HERO IMAGE SLOT — design roadmap banner like v0.2.0's DesignRoad image -->

The biggest release yet. v0.2.0 brought summaries; v0.3.0 brings a full **LLM text-transformation system** (Polish), **cloud provider support** for both transcription and language models, **rolling transcription** that works while you speak, a **session mixer**, **session automation**, a **rebuilt onboarding**, and a performance overhaul that cut recording CPU usage from ~108% to ~14%. Over **30,000 lines** of new code across 82 files.

---

## Showcase

<!-- VIDEO SLOT: main showcase — Polish flow end to end (record → polish → floating response window) -->

### Polish Routes

<!-- VIDEO SLOT: routes — switching apps, polishes firing per app -->

### Session Mixer

<!-- VIDEO SLOT: mixer — channel strips in menu bar, meters on the indicator -->

### New Onboarding

<!-- VIDEO SLOT: onboarding — branching Local / API / Both flow -->

---

## The Polish System

The headline feature. A **Polish** is a custom LLM text transformation that runs on your transcriptions — rewrite dictation as a formal email, turn rambling into bullet points, translate, summarize a call, anything you can describe. Polishes run fully local via MLX or through a cloud provider, your choice per preset.

### Polish Presets

- **Custom presets** — Each polish has its own name, system prompt, model, and optional global keyboard shortcut. Record with a polish shortcut and the result arrives already transformed.
- **Per-preset models** — Any preset can use a local MLX model or an API language model. Mix and match.
- **JSON import** — Share and import polish presets as JSON.
- **Floating response window** — Polish output appears in a floating panel with a **hand-rolled Markdown renderer**, including native SwiftUI **LaTeX math** (fractions, roots, matrices) — no WebView anywhere.

### Polish Generator — describe it, AI builds it

Don't write system prompts by hand. Describe what you want in plain language and a **6-phase AI pipeline** builds the polish for you:

1. **Understand** your description (asks clarification questions if it's vague)
2. **Plan** the transformation with deterministic in-code validation
3. **Write** the system prompt
4. **Evaluate the flow** against your intent
5. **Evaluate the prompt** quality with a scorecard
6. **Repair** up to twice if evaluation finds problems

You review the result on an eval scorecard screen — nothing is saved without your approval. The generator also powers **"Edit with AI"** on existing presets (your shortcut, model, and enabled state are preserved). Has its own model preference and works fully local.

### Polish Routes — per-app automatic routing

Tell AiTranscribe which polish belongs to which app, and the routing is **fully automatic** from then on. Dictate a reply in Slack — your messaging polish formats it and it's ready to send. Switch to your coding agent and hit the same hotkey — now your rambling becomes a clean, structured prompt. No mode switching, no picking from menus; the frontmost app decides. Routes get their own sub-page in Polish settings with a scanner for running and installed apps.

### Runtime Context — polishes that can see

Polishes can optionally pull live context into the prompt:

- **Clipboard** — quick-run mode only picks up a clipboard that *changed during the recording*, so stale copies never leak in
- **Screenshot** — interactive screen capture attached to the polish
- **Active app** — the polish knows where you are
- **Desktop audio** — a quick polish can capture and transcribe *the other side of a call* while you record, so "summarize what we just discussed" actually hears both sides

### Multi-step Modules — build your own pipeline

For truly custom polishes, every preset is a composable pipeline of **input modules** (voice transcript, desktop audio, clipboard, screenshot, active app) feeding **output modules** (copy to clipboard, paste image + text, floating window). Mix any inputs with any outputs. A polish can even **bypass the LLM entirely** — raw transcription straight to its outputs — turning a preset into a pure capture pipeline.

---

## Cloud Providers — Transcription & LLM APIs

Everything in AiTranscribe still runs 100% local if you want it to. But now you can also plug in cloud providers:

### API Transcription

- **OpenAI** — `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`, `whisper-1`, and the diarizing `gpt-4o-transcribe-diarize`
- **ElevenLabs** — Scribe v1 and Scribe v2
- **OpenRouter** — any transcription-capable model behind one key
- **Per-minute cost hints** — known OpenAI models show their audio cost per minute

### API Language Models

- **OpenAI and OpenRouter LLMs** for polishes, the polish generator, and summaries (GPT-4o Mini preconfigured)
- **Custom model entries** — add any model ID a provider supports, with your own display name

### Built right

- **Keychain storage** — API keys live in the macOS Keychain, never in plain text
- **Custom model management** — add, edit, and remove API models from a redesigned Models tab
- **Provider switcher** — capsule pill UI, instant switching

---

## Rolling Transcription — results before you stop talking

Quick recordings no longer wait for you to finish:

- **Rolling batches** — audio is transcribed in ~30-second chunks *while you're still recording*, both locally and through API providers. Stop talking and the final text often lands **in under a second** — only the last slice of audio is left to transcribe.
- **Bulletproof API uploads** — every API upload gets a fresh connection (idle pooled connections were silently killing uploads after the first batch), with protocol and timing metrics logged.
- **Never lose a recording** — if a batch fails it's retried up to 4 times, failed batches are re-sent sequentially at the end, and if the API still won't cooperate the app falls back to a single full-file upload. If *that* fails, the WAV is saved to a FailedRecordings folder and an alert with "Show Recording" appears. Long recordings can no longer silently vanish.
- **Batching controls** — new settings to tune the rolling batch behavior.

---

## Session Studio

### Manual Mixer

The session audio pipeline was rewritten around one principle: **record the truth, let you set the levels.**

- **Auto-gain removed** — amplifying call-app-attenuated audio just raised the noise floor. Tracks now record at true level.
- **Real dBFS meters** — OBS-style ballistics with a −60 dB floor and green/yellow/red zones, for both microphone and desktop audio
- **Two channel strips** — meter + fader for mic and desktop audio, right in the menu bar while a session runs
- **−40 to +40 dB faders** — applied into the recorded file through a gain stage with per-buffer gain slide and a tanh soft limiter, so boosts don't clip
- **Meters on the indicator** — the floating capsule draws live meter arcs: top edge = mic, bottom edge = desktop audio
- **Hardware gain held steady** — input gain is written once at session start and no longer fights call apps' automatic gain control
- **Per-recording diagnostics** — every session writes an `audio_levels.txt` with the mix analysis

### Session Automation

Stop a session and walk away:

- **Auto-transcribe** — local or API, with a real preflight check (including model load) before committing
- **Auto-summary chain** — optionally generate a summary right after transcription, with your chosen preset and model
- **One job at a time** — queued sequentially with persisted metadata and human-readable failure messages
- **Dock control** — manage the automation queue from the Sessions tab

### Session Import

Drag in **MP3, WAV, M4A, AAC, AIFF, MP4, or MOV** files and they become regular sessions — converted to the canonical format, transcribable, summarizable, searchable. All audio tracks in a video are mixed down automatically.

---

## New Onboarding

First launch is a completely rebuilt, branching flow:

- **Welcome → Microphone check** with a live reactive level preview (concentric rings that bounce with your voice)
- **Setup choice** — Local, API, or Both, with hover previews of what each path means
- **Local path** — pick and download transcription models
- **API path** — enter keys (stored in Keychain) with current dated model recommendations
- **Shortcuts → Test transcription** — try it before you're done, with capsule pill model pickers
- **Next steps → Ready** — pointers to Polish and Sessions so the good stuff isn't hidden
- **Live step rail** tracks where you are; finishing opens Settings properly focused

---

## Performance: 108% → ~14% CPU

A full performance audit (sampled live traces, window census, CPU timelines) followed by three fix waves. Recording used to pin a core at ~108%; active use now sits around **14%**, and idle is actually idle.

### Wave 1 — the leaks

- **Indicator window leak fixed** — the floating indicator's window survived `orderOut` and kept its GPU-synced TimelineView rendering every frame *forever*, burning 7–9% CPU at permanent idle. Windows are now properly closed and the TimelineView pauses when hidden. (Side effect: indicator dragging, previously broken, works again.)
- **Heavy stop-path work detached** — stopping a recording or session no longer beachballs the main thread.
- **History saves off the main thread** — no more full decode+rewrite on every save.

### Wave 2 — the architecture

- **10 Hz recording stats isolated** — volume and duration moved off the app-wide observable state into a tiny model only the recording UI observes. Previously every tick re-rendered the entire menu bar and dashboard view trees (~47% of main-thread samples).
- **Keychain checks cached** — the menu bar was doing Keychain IPC per provider *on every render*.
- **Sessions load lazily**, summary streaming publishes in batches instead of 220 times per second, and meeting detection (sync AppleScript on a main-thread timer) was removed outright.

### Wave 3 — instant settings

- **Tab switching is instant** — stagger animations play once per window session, tab fetches are guarded, preset/icon/permission caches added
- **Dashboard hover memoized**, the 18 fps indicator preview pauses when the window is inactive
- **Hotkeys register at launch** instead of on first menu open
- **Backend restart is fast** — no more 2-second launch freeze

---

## Audio & Bluetooth Fixes

- **Bluetooth ducking fixed** — opening a Bluetooth device's mic flips it from A2DP (music) to HFP (call) profile, which has its own volume. Ducking used to restore into the wrong profile and leave your music stuck quiet. Volumes are now tracked and restored **per profile**.
- **Prefer built-in mic over Bluetooth** (new setting, default on) — records from the built-in mic when your selected mic is a Bluetooth device, so your headphones/speaker stay in high-quality A2DP and your music never cuts out.
- **The level meter measures the right mic** — the indicator and onboarding preview used to open their own capture on the system default input, which alone could force a Bluetooth speaker into HFP. They now measure the same device the recording actually uses.
- **Level monitor hardening** — capture format pinned, monitor rebuilds when its device goes stale.

---

## Dashboard Activity Grid

A GitHub-style **52-week × 7-day activity grid** on the Dashboard with dock-style magnification under the cursor and dwell tooltips showing per-day counts. Your transcription streak, at a glance.

---

## Recording Indicator — Six Styles & Noise Calibration

The floating recording indicator got a full restyle:

- **Six styles** — Classic, Glow, Aurora, Prism, Graphite, and Ember, switchable in General settings
- **Noise floor calibration** — one click samples your room's ambient noise so the level display reacts to your *voice*, not your fan. Reset to automatic anytime.
- **Live meter arcs** — during sessions the capsule doubles as a meter: top edge mic, bottom edge desktop audio

---

## UI Consistency Round 2

- **Capsule pill dropdowns everywhere** — every model dropdown across session detail, auto-processing, and general settings now renders the custom capsule style
- **Redesigned Add/Edit model sheets** — segmented pickers replaced with capsule pill switchers, glass text fields, capsule action buttons
- **Redesigned Summary Presets manager** — icon header, glass preset rows with accent selection, capsule controls
- **Instant provider switching** in model sheets — no more full-sheet relayout animation

---

## Removed

- **Ultra Instinct multi-model transcription** — round-robin across models made the transcript's voice and spelling drift from chunk to chunk. Old sessions keep their label.
- **Meeting detection** — synchronous AppleScript polling wasn't worth the cost (−1,419 lines). The associated Apple Events permission request is gone too.

---

## Other Changes

- Backend: new `POST /polish/local` (SSE) endpoint for local polishes and the generator, and `POST /transcribe-batch-stream` (SSE) for RAM-budgeted chunked batch transcription
- Settings window focus fixed — the window actually takes focus when opened
- **Relicensed to GPL-3.0-or-later** — use, modify, and share AiTranscribe freely and contribute back, but any distributed fork must stay open-source under the GPL; nobody can turn it into a closed paid app. Separate commercial terms available from the author. (Replaces the earlier MIT + Commons Clause arrangement.)
- Per-recording audio diagnostics (`audio_levels.txt`)

---

**Full Changelog**: https://github.com/Ljove02/AIT-AiTranscribe-MacOS/compare/v0.2.0...v0.3.0
