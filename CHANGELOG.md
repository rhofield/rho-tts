# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
