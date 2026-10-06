# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `min_segment_chars` (Breeze option, default 0 = off): after sentence splitting, a segment shorter than this joins the next one, and a short final segment joins the previous one, never past `max_chars_per_segment`. "Here's a puzzle." spoken alone had 0.24 s of voiced audio. That was too little for continuity to measure pitch or speaker, and a take about 4 semitones high was accepted.
- `ContinuityConfig.max_adjacent_pitch_semitones` and `max_adjacent_tilt_db` (default `None` = off): also limit the pitch and brightness jump from the most recently accepted clip on its own. The window check lets each sentence sit on either side of the window, so neighbours could differ by twice its limits (up to 6 semitones or 10 dB) and sound stitched together. Misses count toward the fallback ranking. Evidence reports `adjacent_pitch_difference_semitones`, `adjacent_tilt_difference_db` and both limits.

### Changed
- Generation manifests record `min_segment_chars`, so every manifest fingerprint changes once.

## [1.4.0] - 2026-10-06

### Added
- Audio continuity now checks pitch and brightness, which speaker embeddings deliberately ignore. A segment fails when its median F0 is more than `max_pitch_semitones` (default 3) from the neighbouring accepted speech, or its voiced-frame spectral tilt differs by more than `max_tilt_db` (default 5 dB). Until 3 s of speech is accepted, the reference voice tops up the comparison. Brightness is measured on voiced frames only, because whole-clip brightness mostly tracks sibilant count. Evidence reports `pitch_hz`, `tilt_db`, both differences and `voiced_seconds`.
- A segment with at least 0.5 s of speech but no voiced frame fails continuity as unvoiced.
- Breeze `temperature` option for the backbone and depth decoder (upstream default 0.9).
- `segment_level_db`: each generated segment is scaled to this active-speech level before validation and joining, with an 18 dB boost cap and a peak ceiling. Breeze defaults to −23 dBFS; other providers leave it off.

### Changed
- Speaker similarity is reported but no longer judged on segments with under 1.5 s of speech. A 0.8 s sentence scored 0.56–0.64 even against its own sibling sentences, so it failed every attempt regardless of how it sounded.
- When no candidate passes, the fallback is the one whose checks missed by least in total, each miss measured as a fraction of its threshold. Previously any candidate passing accent and text outranked every continuity result, so an 11-semitone pitch jump shipped over a take with only slight accent drift.
- Breeze output loudness changes: segments are levelled to −23 dBFS by default. Pass `segment_level_db=None` for the previous behaviour.

## [1.3.5] - 2026-10-06

### Changed
- Audio continuity judges speaker identity against the provider's reference voice plus at least 3 s of recently accepted speech, instead of the last segment alone. Short segments produced unreliable speaker embeddings, so a 0.8 s opening sentence failed every successor while passing itself unchecked. With a reference voice, the first segment is checked too. Loudness is still compared with recent accepted speech only.
- When accent drift fails, text validation is reported as `skipped` rather than `unavailable` and no longer appears in rejection or fallback reasons.

## [1.3.2] - 2026-09-30

### Added
- Opt-in `cuda_graph_depth=True` for Breeze's depth decoder. It substantially reduces warm voice-cloning latency on the measured RTX 3060; the first request pays a compilation and capture cost.
- Opt-in `compile_depth=True` and `triton_depth_sampling=True` for the eager depth path, plus configurable `codec_chunk_frames` and incremental Breeze `stream()` output.
- Cost-based CPU/GPU selection for Breeze reference resampling.

### Changed
- Breeze defaults to PyTorch SDPA attention and avoids unnecessary full-vocabulary work for its default eager top-k sampling settings.
- Isolated streaming WAV handoff uses SoundFile where available, with a Torchaudio read fallback for consumers without SoundFile installed in the parent environment.

## [1.3.1] - 2026-09-24

### Added
- Opt-in adjacent speech continuity validation with explicit per-call context, shared candidate retries, and structured results through isolated providers.

## [1.3.0] - 2026-09-24

### Fixed
- Strict validation now requires successful speech checks before accepting generated audio; exhausted retries no longer publish audio that failed validation.
- Validation errors from isolated providers are preserved across the worker boundary.

## [1.2.0] - 2026-09-02

### Added
- **Breeze-TTS-2 provider** (`provider="breeze"`) — voice cloning from a reference audio clip plus its exact transcript, and natural-language voice *design*/*direction* via an `instruction` string (unique among the providers here)
- `breeze` optional extra pinning the exact torch/transformers versions upstream requires
- Breeze always runs through the subprocess isolation layer; `VenvManager` provisions its venv and clones the upstream source (breeze-tts ships no packaging metadata, so pip cannot install it)
- Model weights pinned to an explicit revision so an unpinned snapshot download cannot swap the weights — or the licence — underneath a working install
- Breeze wired into the Gradio UI, including the instruction field

### Notes
- `breeze` is deliberately **not** part of the `[all]` extra: its pins conflict with the other providers, and the model weights are under the BreezeBlue Research and Non-Commercial License (upstream code is Apache 2.0). Commercial use requires written authorization from RESONIA, INC.

## [1.1.4] - 2026-03-28

### Changed
- Sound decay validation now runs on the full joined + post-processed audio instead of per-segment, catching decay that only appears when segments are concatenated
- Sound decay retry loop regenerates all segments (with a new seed) when decay is detected, up to `max_decay_retries` (default 3) attempts
- `num2words`, `nemo_text_processing`, and `text2num` promoted from `validation` optional extra to core dependencies — number normalization is now always available
- Number normalizer no longer uses try/except import guards; dependencies are guaranteed present
- Sound decay is no longer a per-segment rejection criterion in the validation retry loop — drift and text checks still run per-segment, but decay is evaluated holistically on the final output

### Added
- `max_decay_retries` attribute on `BaseTTS` (default 3) — controls how many full-regeneration attempts are made when sound decay is detected
- `decay_ratio` field on `GenerationResult` — reports the final sound decay ratio in generation metadata
- Number normalizer now strips digit-group commas (`"1,500"` → `"1500"`) and currency symbols (`"$500"` → `"500"`) before normalization

### Fixed
- Sound decay that only manifested in the joined multi-segment audio (not in individual segments) is now caught and corrected

## [1.1.3] - 2026-03-26

### Added
- Word-level text diff logging on STT validation failures — shows missing, extra, and substituted words to help diagnose generation issues
- Auto-sort documentation section in the README

### Changed
- Post-processing (windowed RMS normalization) now runs once on the final accepted audio instead of on every retry iteration, improving performance and avoiding double-normalization
- Raised max normalization gain cap from 12 dB to 18 dB to better correct quiet endings
- Simplified Qwen generation validation logic with early-exit checks before dispatching to model

### Fixed
- Auto-sort now catches `OSError` (permission denied, disk full) gracefully instead of crashing the generation pipeline
- Suppressed noisy `code_predictor_config is None` log spam from Qwen model loading
- Suppressed `pad_token_id` / `eos_token_id` warnings during Qwen generation
- Suppressed sklearn feature-name warnings in accent drift classifier predictions
- Added missing warning log when accent drift feature extraction fails
- Accent drift probability now logged at the call site with threshold context

## [1.1.2] - 2026-03-25

### Added
- Sound decay detection and correction for voice cloning
  - New `_validate_sound_decay()` validation in the generation retry loop — rejects audio where the final third's RMS energy drops below `sound_decay_threshold` (default 0.3) relative to the first third, and retries with a new seed
  - Windowed RMS normalization in `QwenTTS._post_process_audio()` — computes per-window (2s) gain envelope to correct volume taper, smoothed with a 3-tap moving average to avoid artifacts, with a 12 dB max gain cap
  - `sound_decay_threshold` on `BaseTTS` (was previously defined on `QwenTTS` but never checked)
- Automatic CUDA-to-CPU fallback in `QwenTTS._load_qwen3_model()` — when CUDA initialization fails (e.g. driver too old), the model is retried on CPU with a warning instead of crashing
- Test suite for sound decay validation and windowed normalization (`tests/test_sound_decay.py`)

### Fixed
- Classifier trainer now raises a clear `ValueError` when the dataset has fewer than 5 samples, instead of crashing inside `train_test_split` with an opaque sklearn error
- Test script (`scratch/test_testpypi.sh`) no longer hardcodes a stale version default; `--dataset-dir` defaults to empty (skip) instead of a machine-specific path
- Workflow script (`scratch/workflow.py`) auto-detects CUDA availability and falls back to CPU; new `--device` flag for explicit override

## [1.0.9] - 2026-03-25

### Fixed
- Chatterbox "Faster" model no longer crashes when the installed `chatterbox-tts` version doesn't support `max_cache_len` — unsupported kwargs are detected via signature inspection and dropped with a warning
- Voice dropdown no longer retains stale values after switching models, preventing Gradio validation errors (`Value not in the list of choices`)
- Adding, editing, or deleting a model now correctly syncs the voice dropdown with the auto-selected model instead of leaving it empty
- Models tab now pre-populates model choices for the default provider on initial load

### Changed
- Qwen model display names clarified: "Base (Voice Cloning)" and "CustomVoice (Built-in Speakers)"
- Voice dropdown returns empty when no model is selected, instead of showing all built-in voices across providers
- Version bump to 1.0.9

## [1.0.8] - 2026-03-22

### Changed
- Replaced custom 268-line number normalizer with a production-grade pipeline backed by `nemo_text_processing` and `text2num`
  - NeMo's inverse text normalization (ITN) handles dates (`"march twenty second twenty twenty six"` → `"march 22 2026"`), currency (`"five dollars and ninety nine cents"` → `"$5.99"`), and times (`"three thirty pm"` → `"03:30 p.m."`) in addition to numbers
  - `text2num` (`alpha2digit`) covers remaining single and compound word numbers NeMo doesn't handle standalone
  - Mixed digit-word formats (`"2 hundred"` → `"200"`) handled by a retained regex pre-pass
  - Removed `word2number` dependency; added `nemo_text_processing` and `text2num` to `validation` extra
- Version bump to 1.0.8

## [1.0.7] - 2026-03-21

### Added
- Per-session isolation for multi-user deployments (e.g. HF Spaces)
  - New `SessionContext` dataclass gives each browser tab its own cancellation token, generation history, and temp output directory
  - Voices, Models, and Library tabs all operate on session-scoped state
  - Temp directories are cleaned up automatically on session close
- Training tab is now visible (but disabled) in multi-user mode with an explanatory message
- Server-side guard on training callback to block API-level bypass in multi-user mode
- Test suite for session lifecycle (`tests/test_session.py`)

### Changed
- `AppState` accepts `multi_user` flag; config saves are no-ops in multi-user mode
- `AppState.get_or_create_tts()` accepts an optional per-session `config` override
- Callbacks refactored to accept session context for isolated state
- Version bump to 1.0.7

## [1.0.6] - 2026-03-19

### Added
- Screenshots of all five UI tabs in the README (Generate, Library, Voices, Models, Training)

### Changed
- Version bump to 1.0.6
- README installation instructions updated to reflect PyPI availability (removed "once published" qualifier)

### Fixed
- Synced `__init__.py` version with `pyproject.toml`

## [1.0.5] - 2026-03-15

### Added
- Auto-sort: automatically copy generated audio into good/bad training folders based on accent drift score
  - Four new `BaseTTS` attributes: `auto_sort_good_threshold`, `auto_sort_bad_threshold`, `auto_sort_good_dir`, `auto_sort_bad_dir`
  - Samples with drift below `good_threshold` are copied to `good_dir`; samples above `bad_threshold` go to `bad_dir`
  - Middle-zone samples (ambiguous confidence) are intentionally skipped
  - Works with both `max_iterations=1` (single-pass) and multi-iteration validation loops
  - UI state layer (`AppState`) passes auto-sort parameters through to the TTS instance
- Unit and integration tests covering all auto-sort routing scenarios

## [1.0.4] - 2026-03-14

### Added
- Smart segmentation: auto-compute `max_chars_per_segment` based on available GPU VRAM or system RAM
- `train_drift_classifier()` top-level API for training custom drift-detection models
- `drift_model_path` parameter on providers and `BaseTTS` for using custom classifier models
- Provider-specific `MAX_MODEL_CHARS` and `BYTES_PER_CHAR_ESTIMATE` class constants

### Changed
- `max_chars_per_segment` now defaults to `None` (auto-computed) in both `QwenTTS` and `ChatterboxTTS`
- `QwenTTS` inspects model `max_position_embeddings` to refine segment size limits

### Fixed
- Drift classifier cache key now correctly uses explicit `model_path` when provided

## [1.0.3] - 2026-03-12

### Fixed
- Audio saving now falls back to stdlib `wave` module when torchaudio backend (torchcodec) is unavailable
- STT validator import path fix

## [1.0.0] - 2026-03-12

### Added
- `GenerationResult` dataclass returned by `generate()` with audio tensor, path, duration, and metadata
- In-memory generation: call `generate()` without `output_path` to get audio tensors directly
- Context manager protocol (`with TTSFactory.get_tts_instance(...) as tts:`)
- `ProviderInfo` and `VoiceInfo` introspection via `BaseTTS.provider_info()` and `TTSFactory.get_provider_info()`
- Custom exception hierarchy: `RhoTTSError`, `ProviderNotFoundError`, `ModelLoadError`, `AudioGenerationError`, `FormatConversionError`
- Streaming generation via `generate_stream()` yielding audio chunks
- Async generation via `generate_async()`
- Speed and pitch post-processing via `_apply_speed_pitch()`
- Audio format conversion (MP3, FLAC, OGG) via pydub
- Subprocess-based venv isolation layer (`rho_tts.isolation`) for providers with conflicting dependencies
  - JSON-line IPC protocol, auto-created venvs at `~/.rho_tts/venvs/<provider>/`
  - `ProviderProxy` duck-types `BaseTTS` without importing torch
  - Crash recovery with automatic worker restart (up to 2 retries)
- Gradio-based UI (`rho_tts.ui`) with model selection, voice cloning, and training controls
- Comprehensive test suite (107 tests) covering packaging, isolation, pipeline, streaming, and more

### Changed
- Renamed package from `ralph-tts` to `rho-tts`
- `generate()` now returns `GenerationResult` instead of `Optional[str]`
- Removed `GenerateAudio` wrapper class; generation is now handled directly by `BaseTTS.generate()`
- `TTSFactory` supports isolated provider registration via `_isolated_providers`
- Refactored generation pipeline into `_run_pipeline()` for reuse across `generate()` and `generate_stream()`

### Fixed
- Voice cloning UI no longer allows clone-only options for non-cloning models
- Training workflow bug fixes and UI improvements
- Various QwenTTS generation quality improvements

## [0.1.0] - 2025-02-25

### Added
- Initial release extracted from internal project
- `BaseTTS` abstract base class with audio processing utilities
- `QwenTTS` provider with batch processing and validation
- `ChatterboxTTS` provider with voice cloning support
- `TTSFactory` for provider registration and instantiation
- `GenerateAudio` high-level generator with async support
- Accent drift detection via voice quality classifier
- STT validation via Whisper (faster-whisper + transformers fallback)
- Speaker similarity validation via resemblyzer
- Text preprocessing with phonetic mapping and number normalization
- Audio segment smoothing with crossfading
- Thread-safe `CancellationToken` for cooperative task cancellation
