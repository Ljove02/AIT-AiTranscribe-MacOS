# AiTranscribe Agent Notes

## Codebase Memory

This repo is indexed with `codebase-memory-mcp`.

- Binary: `~/.local/bin/codebase-memory-mcp`
- Project name: `Volumes-Kingstone_Local-GitHub-Local-AIT-AiTranscribe-MacOS`
- Local graph/cache: `~/.cache/codebase-memory-mcp/`
- Auto-index: enabled globally with `auto_index=true`

Before broad exploration, prefer graph queries over reading many files.

Useful commands:

```bash
~/.local/bin/codebase-memory-mcp cli list_projects
~/.local/bin/codebase-memory-mcp cli get_architecture '{"project":"Volumes-Kingstone_Local-GitHub-Local-AIT-AiTranscribe-MacOS"}'
~/.local/bin/codebase-memory-mcp cli search_graph '{"project":"Volumes-Kingstone_Local-GitHub-Local-AIT-AiTranscribe-MacOS","name_pattern":".*Polish.*","limit":20}'
~/.local/bin/codebase-memory-mcp cli search_code '{"project":"Volumes-Kingstone_Local-GitHub-Local-AIT-AiTranscribe-MacOS","query":"OnboardingWindowController"}'
~/.local/bin/codebase-memory-mcp cli trace_path '{"project":"Volumes-Kingstone_Local-GitHub-Local-AIT-AiTranscribe-MacOS","function_name":"startRecording","direction":"both"}'
```

Use `rg` or direct file reads after the graph identifies the likely files or symbols.

## Main Domains

- Menu bar and basic recording: `MenuBarView.swift`, `AppState.swift`, `AudioRecorder.swift`, `RecordingIndicator.swift`
- Onboarding: `OnboardingContainerView.swift`, `OnboardingWindowController.swift`, `WelcomeOnboardingView.swift`, `*OnboardingView.swift`
- Public/local providers: `TranscriptionProvider.swift`, `APIClient.swift`, `OpenAI*`, `OpenRouter*`, `ElevenLabs*`, `API*ModelStore.swift`
- Polish pipeline: `PolishPresetStore.swift`, `PolishContextReader.swift`, `PolishMarkdownView.swift`, `PolishSettingsView.swift`, `AppState.swift`
- Sessions: `Sessions/SessionManager.swift`, `Sessions/SessionRecorder.swift`, `Sessions/SystemAudioCapture.swift`, `Settings/SessionDetailView.swift`
- Backend: `backend/server.py`, `backend/model_manager.py`, `backend/batch_transcriber.py`, `backend/summary_manager.py`
- Build/package: `build_production.sh`, `run_backend.sh`, `README.md`

## Product Design Notes

- The onboarding microphone preview has a strong reusable signal pattern: concentric rings plus a subtle green/teal reactive field driven by normalized microphone level. Preserve the bouncy outer-ring fade behavior as a candidate for a future recording/input indicator.
- Onboarding pages should avoid default equal-card grids when the content can be expressed with hierarchy, negative space, rows, rails, or one focused visual panel. Use cards only when the surface itself communicates a real decision or status.

## Build / Run Notes

- Production build entrypoint: `./build_production.sh`
  - Full build rebuilds the Python backend, builds the Swift app in Release, copies to `dist/AiTranscribe.app`, and creates the DMG.
  - Use `./build_production.sh --skip-backend` when only Swift/app code changed and the bundled backend executable does not need rebuilding.
  - The script rewrites `dist/AiTranscribe.app`; avoid running it while the app from `dist/` is open.
- Backend runner: `./run_backend.sh`
  - Starts the backend on `127.0.0.1:8770`.
  - Use `./run_backend.sh --dev` for uvicorn reload mode.
- For a quick compile check without rewriting `dist`, use the Xcode project directly:
  `xcodebuild -project AiTranscribe/AiTranscribe.xcodeproj -scheme AiTranscribe -configuration Debug -destination 'platform=macOS' build`
