# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
