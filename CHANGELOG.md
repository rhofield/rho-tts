# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
