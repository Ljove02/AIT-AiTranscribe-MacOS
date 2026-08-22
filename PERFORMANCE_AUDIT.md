# Performance Audit — 2026-08-22

Full-codebase scout for CPU, main-thread stalls, re-render waste, and memory growth.
Method: live process sampling (`trace_cpu.sh`), plus three parallel code sweeps
(settings views / publish-observer map / main-thread blocking).

Already fixed before this audit: 10 Hz `currentVolume`/`recordingDuration` publishing on
AppState (recording CPU 100% → ~25%), Keychain presence cache, model-store decode cache.

**Key architectural fact:** the target builds with `SWIFT_DEFAULT_ACTOR_ISOLATION = MainActor`
(project.pbxproj:608,658). Every unannotated class (HistoryManager, AudioRecorder,
SessionRecorder, SessionManager, AppState…) is main-actor isolated. `async` does NOT mean
off-main here — only `Task.detached` / explicit queues leave the main thread.
Also: zero `import Accelerate` in the target; every DSP loop is scalar.

---

## A. Freezes (main-thread stalls the user can feel)

| # | What | Where | When / cost |
|---|------|-------|-------------|
| A1 | Session-stop audio analysis: `logCapturedLevels` → `analyzeAudioLevels` ×2 files × 2 scans each = 4 full-file per-sample histogram passes, main actor | Sessions/SessionRecorder.swift:1275, 1216, 1143/1147, 1051 | Every session stop. 1 h meeting ≈ 690 M scalar iterations + 4 full file reads → multi-second beachball |
| A2 | Backend kill: 2 × `lsof \| xargs kill` with waitUntilExit + 2 × `Thread.sleep(1.0)`, @MainActor | BackendManager.swift:348-381 | App launch + every backend restart: ≥2.0 s guaranteed freeze |
| A3 | History: every save = full JSON load + decode + insert + encode + atomic rewrite, main actor; page load decodes whole file TWICE (`loadPage` + `getTotalCount`); no cap/rotation; entries carry 3 copies of the text | HistoryManager.swift:68-119, 280-283; AppState.swift:526,542,567 | Every transcription stop, every history page scroll; grows forever |
| A4 | Meeting detection: synchronous `NSAppleScript` to Chrome/Safari + `CGWindowListCopyWindowInfo` + AX IPC on a 0.8–2.5 s main-thread timer, runs whenever a browser/Zoom is frontmost (not only when recording) | MeetingPresenceMonitor.swift:596, 511, 538, 273, 280-295 | Continuous; a busy browser stalls the app until Apple-event timeout |
| A5 | Ducking: osascript spawn + waitUntilExit on main — 2 spawns at record start, 1 at stop, 9 call sites (+ one in `deinit`) | AudioManager.swift:71-103, 269-290, 47-63 | ~150–500 ms at every recording start/stop |
| A6 | `SessionManager.loadSessions`: full disk walk, decodes metadata AND reads every transcript + every summary .md into memory, sync on main | Sessions/SessionManager.swift:859-930, 839-857 | App launch + Sessions tab switches; 100 sessions ≈ ~500 sync reads |
| A7 | Python probing: up to 7 × `Process` + waitUntilExit (`python -c …`, 150–500 ms each), @MainActor | SummarySetupManager.swift:105-153; NemoSetupManager.swift:167-204 | Setup flows; up to ~3.5 s freeze |
| A8 | `createWAVData`: whole-recording scalar clamp/convert on main at stop; plus 2× peak memory from combine buffer | AudioRecorder.swift:783-794, 494-511 | Every quick-recording stop |
| A9 | Update install: `ditto` waitUntilExit + whole-.app copy on main | UpdateChecker.swift:208-243 | Rare, but tens of seconds |
| A10 | Screenshot paste path: TIFF→PNG re-encode of full screenshot on main actor; polish screenshot `Data(contentsOf:)` on main actor | AppState.swift:3098-3110, 2155-2166 | At paste / polish time |

## B. Publish storms (re-render waste; same class as the fixed volume bug)

Amplifier: `AiTranscribeApp.swift:33-37` observes 5 singletons at App level; any publish on
AppState / SessionManager / BackendManager / UpdateChecker / MeetingPresenceMonitor
invalidates the whole scene body + MenuBarView + Settings. `meetingMonitor` and
`updateChecker` are observed but never read in body — pure invalidation cost.
AppState has ~44 @Published properties observed by 17+ views.

| # | Publisher | Rate | Blast radius |
|---|-----------|------|--------------|
| B1 | `BackendManager.addLog` — one publish per backend stdout line, no batching | Unbounded (40+/chunk during startup/MLX load) | Whole app scene + 10 views (BackendManager.swift:722-730, 498-508) |
| B2 | Summary typewriter loop: 2 publishes per 9 ms tick, each mutating the whole `sessions` array (COW of every Session incl. transcripts) | ~220/s for ~1.3 s | Whole app + Sessions views (SessionManager.swift:2214-2233) |
| B3 | Summary SSE "partial": whole-array mutation + partialText, entire local-LLM generation | 10/s for minutes | Same (SessionManager.swift:2402-2416) |
| B4 | MeetingPresenceMonitor unguarded `currentPrompt = nil` re-publishes (refresh():198, apply():248,253) | ~1/s during Zoom/browser after a dismissal | Whole app scene + MenuBarView |
| B5 | `SessionManager.sessionDuration` 1 Hz during session recording — right rate, wrong object (SessionManager is app-level observed) | 1/s | Whole app (SessionManager.swift:1134-1138) |
| B6 | `SessionRecorder` meters 30 Hz ×2 publishes + `micVolume` written per audio buffer AND NEVER READ (dead @Published) | 60/s + buffer rate | SessionMixerView subtree (SessionRecorder.swift:484-492, 805-826) |
| B7 | `AppState.lastTranscription` grows-full-transcript publish during streaming | 1/s | All 17+ AppState observers (AppState.swift:2098-2106) |
| B8 | `fetchAvailableModels` ends with `for window in NSApp.windows { window.display() }` — forced sync redraw of EVERY window | Per fetch (incl. every Models tab switch) | Everything (AppState.swift:1100-1102) |
| B9 | Manual `objectWillChange.send()` for @AppStorage without equality guards — slider drags fire whole-app invalidation per delta | Per drag delta | Whole app (AppState.swift:232-262, 1241-1262) |

## C. Tab switching (the "instant feel" question)

Per-switch costs, in order:
1. Stagger animation replay: `.staggerIn` on `Group`s explodes to ~20 independently animated
   offset layers over material backgrounds; last delay 0.55 s + 0.4 s spring ≈ 1 s of 60 fps
   re-compositing per switch. Cross-navigation deliberately removes tabs from `animatedTabs`
   so it replays (GeneralSettingsView.swift:807-829; SettingsView.swift:188,194,222).
2. Models tab: 4 backend round-trips on EVERY appearance, no hasLoaded guard, then B8's
   all-windows forced redraw (ModelsSettingsView.swift:219-234).
3. Sessions tab: main-actor directory scan + per-folder stat, signature computed twice on
   change, + TCC preflight, per appearance (SessionsSettingsView.swift:176-179).
4. Polish tab: `PolishPresetStore.load()` (UserDefaults + JSON decode + merge, NOT cached)
   + backend fetch per appearance (PolishSettingsView.swift:155-159; PolishPresetStore.swift:793).
5. Sidebar: `loadAppIconFromBundle()` (bundle path lookup + possible PNG decode) per render —
   which includes every tab switch and every hover of a sidebar row
   (SettingsSidebar.swift:118, 195-215).
6. General tab: `AVCaptureDevice.authorizationStatus` (TCC) evaluated ~8× per render
   (GeneralSettingsView.swift:236-238 et al).

Steady-state burners while Settings is open:
- General tab: permanent 18 fps TimelineView on the selected indicator-style preview
  (blurred gradient composite), even scrolled off-screen (GeneralSettingsView.swift:674, 620).
- Dashboard activity grid: per mouse-move (60–120 Hz hover) re-materializes 364 cells +
  ~420 transcendentals + allocates a DateFormatter (DashboardView.swift:735-861, 1030, 1048).

## D. Memory

- `recordedBuffers`: unbounded pre-conversion device-rate copies (~23 MB/min), 2× peak at stop,
  3rd copy for WAV (AudioRecorder.swift:312, 438, 494-511).
- `QuickSystemAudioRecorder.samples`: unbounded (~192 KB/s) (QuickSystemAudioRecorder.swift:10,142).
- `HistoryManager.embeddingIndex` + `indexedEntryCache`: two full in-memory copies of all
  entries + a Double vector per entry, built synchronously on main, never trimmed
  (HistoryManager.swift:156-163).
- `SessionManager.sessions` keeps every transcript + summary text resident (A6).

## E. Micro / by-the-way

- Recomputed-computed-properties per render: `filteredModels` ×4 (ModelsSettingsView.swift:94-130),
  `routeGroups` ×4 (PolishSettingsView.swift:1426-1449), `visibleSummaryPresets` ≥6×
  (SessionDetailView.swift:117-144), `session` linear search ×~40 (SessionDetailView.swift:46),
  `polishShortcutPresets` ×5 + O(n²) conflicts (ShortcutsSettingsView.swift:176-285).
- `ByteCountFormatter`/`DateFormatter` allocated per call in card/dashboard renders
  (SessionManager.swift:1018; DashboardView.swift:622-651).
- `Session` Equatable compares full transcript strings → SwiftUI diffs whole transcripts per card.
- Scalar audio loops (off-main but CPU): 3 passes per mic buffer (RMS, gain-RMS, gain-apply),
  double RMS in SessionRecorder, per-sample `tanhf` in GainStage.softLimit, scalar resampler —
  all trivially vDSP-able (AudioRecorder.swift:441-449, 1039-1061, 206-270;
  SessionRecorder.swift:768-811; Sessions/GainStage.swift:47-130; QuickSystemAudioRecorder.swift:340-361).
- `AppLogger` hops audio-thread log calls onto the main actor into BackendManager.logs (AppLogger.swift:23).
- `SummarySetupManager.appendInstallationLog`: FileHandle open/seek/write/close per pip output line (SummarySetupManager.swift:186-192).

## F. Indicator (phase-2, already queued)

- Window leak REPRODUCED on new build: dictation 19:58–20:00 left indicator alive,
  orderOut'd but retained, TimelineView still rendering → ~10% idle CPU. Fix: `close()`
  (isReleasedWhenClosed=false) + pause TimelineView when hidden.
- `window?.display()` per level tick (RecordingIndicator.swift:1946) — now the top app frame.
- TimelineView uncapped + `.onChange(of: now)` state write = double render per frame.
- Duplicate AVCaptureSession for level metering alongside the recorder's.

## Verified clean (don't touch)

- DashboardStatsManager: properly `Task.detached`, gated once per run.
- Backend health-check timer: no publishes at rest, paused while Settings open.
- SessionIndicatorView 30 fps meters: CALayer pull-based, no @Published — architecturally right.
- SoundManager preloads; no sync URLSession / DispatchQueue.main.sync / semaphores anywhere.
- Static DateFormatters correctly hoisted in session/history list views.

## Suggested order of attack

1. **Indicator phase-2** (kills the reproduced leak + remaining ~20% recording CPU).
2. **Tab-switch pack** (C1–C6 + B8): stagger once-per-window, hasLoaded guards on tab fetches,
   cache PolishPresetStore + sidebar icon, drop the all-windows display() loop.
3. **Freeze pack** (A1, A2, A3, A5): detach session-stop analysis, async backend kill,
   incremental/off-main history store, CoreAudio-API or async ducking.
4. **Publish pack** (B1, B2/B3, B4, B5 + App-level observer diet at AiTranscribeApp.swift:33-37).
5. **vDSP + memory pass** (D + E scalar loops).
