# rho-tts

Multi-provider text-to-speech library with voice cloning, accent drift detection, and STT validation.

## Features

- **Multi-provider TTS** — Swap between Qwen3-TTS and Chatterbox with a single parameter
- **Voice cloning** — Clone any voice from a short reference audio sample
- **Accent drift detection** — ML classifier catches when the generated voice drifts from your target accent
- **STT validation** — Whisper-based transcription check ensures the model actually said what you asked it to
- **Speaker similarity** — Cosine similarity scoring between generated and reference voice embeddings
- **Audio post-processing** — Silence trimming, crossfading, DC offset removal, fade-in/out
- **Batch processing** — Generate multiple audio files efficiently with memory management
- **Cooperative cancellation** — Thread-safe cancellation tokens for long-running generation tasks
- **Extensible** — Register custom TTS providers via `TTSFactory.register_provider()`

## Installation

```bash
# Core only (brings torch, torchaudio, numpy, pydub)
pip install rho-tts

# With Qwen3-TTS provider
pip install rho-tts[qwen]

# With Chatterbox provider
pip install rho-tts[chatterbox]

# With validation (accent drift, STT, speaker similarity)
pip install rho-tts[validation]

# Everything
pip install rho-tts[all]
```

### System Dependencies

- **ffmpeg** — Required by pydub for audio file joining
- **CUDA** — GPU recommended for reasonable generation speed (CPU works but is slow)

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

### Hardware Requirements

| Model | VRAM | Notes |
|-------|------|-------|
| Qwen3-TTS 0.6B | ~8 GB | Smaller, faster |
| Qwen3-TTS 1.7B | ~16 GB | Higher quality |
| Chatterbox | ~6 GB | Good for single segments |
| Validation (Whisper) | ~1 GB | Runs on CPU by default |

## Quick Start

```python
from rho_tts import TTSFactory

# Create a TTS instance (requires a reference audio for voice cloning)
tts = TTSFactory.get_tts_instance(
    provider="qwen",
    reference_audio="my_voice.wav",
    reference_text="Transcript of my voice sample.",
)

# Generate a single file
tts.generate_single("Hello world!", "output.wav")

# Generate a batch
files = tts.generate(
    texts=["First sentence.", "Second sentence."],
    output_base_path="batch_output",
)
```

## Providers

### Qwen3-TTS (default)

Best for batch generation with validation. Supports voice cloning via reference audio + text.

```python
tts = TTSFactory.get_tts_instance(
    provider="qwen",
    reference_audio="voice.wav",
    reference_text="What the voice says in the audio file.",
    model_path="Qwen/Qwen3-TTS-12Hz-1.7B-Base",  # or local path
    batch_size=5,
    max_iterations=10,
    accent_drift_threshold=0.17,
    text_similarity_threshold=0.85,
)
```

### Chatterbox

Best for single-segment regeneration with comprehensive validation loops.

```python
tts = TTSFactory.get_tts_instance(
    provider="chatterbox",
    reference_audio="voice.wav",
    implementation="faster",  # rsxdalv optimizations
    max_iterations=50,
    accent_drift_threshold=0.17,
    text_similarity_threshold=0.75,
    speaker_similarity_threshold=0.85,
)
```

## Configuration

All thresholds and parameters can be set via constructor kwargs:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `device` | `"cuda"` | `"cuda"` or `"cpu"` |
| `seed` | `789` | Random seed for reproducibility |
| `deterministic` | `False` | Deterministic CUDA ops (slower) |
| `phonetic_mapping` | `{}` | Word-to-pronunciation overrides |
| `max_iterations` | `10`/`50` | Max validation retry loops |
| `accent_drift_threshold` | `0.17` | Max accent drift probability |
| `text_similarity_threshold` | `0.85`/`0.75` | Min STT text match score |
| `batch_size` | `5` | Texts per batch (Qwen only) |

## Custom Providers

Register your own TTS implementation:

```python
from rho_tts import BaseTTS, TTSFactory

class MyTTS(BaseTTS):
    def _generate_audio(self, text, **kwargs):
        # Your model inference here
        ...

    def generate(self, texts, output_base_path, cancellation_token=None):
        ...

    def generate_single(self, text, output_path, cancellation_token=None):
        ...

    @property
    def sample_rate(self):
        return 24000

TTSFactory.register_provider("my_tts", MyTTS)
tts = TTSFactory.get_tts_instance(provider="my_tts")
```

## Validation Pipeline

When validation deps are installed (`pip install rho-tts[validation]`), generated audio goes through:

1. **Accent drift detection** — A trained classifier predicts the probability that the voice has drifted from the target accent. Samples exceeding the threshold are regenerated.

2. **STT text matching** — Whisper transcribes the audio and compares it against the intended text using fuzzy matching with number normalization.

3. **Speaker similarity** — Cosine similarity between the generated audio's speaker embedding and the reference voice embedding.

### Training the Accent Drift Classifier

```bash
# Prepare a dataset with good/ and bad/ subdirectories containing .wav files
python -m rho_tts.validation.classifier.trainer --dataset-dir /path/to/dataset

# Or specify output path
python -m rho_tts.validation.classifier.trainer \
    --dataset-dir /path/to/dataset \
    --output /path/to/voice_quality_model.pkl
```

Set the model path via environment variable:
```bash
export RHO_TTS_CLASSIFIER_MODEL=/path/to/voice_quality_model.pkl
```

## Web UI

A Gradio-based web interface for interactive TTS generation, voice management, and model configuration.

### Installation

```bash
# From PyPI (once published)
pip install rho-tts[ui]

# From local source
pip install -e ".[ui]"
```

### Launch

```bash
# CLI entry point
rho-tts-ui

# Or as a Python module
python -m rho_tts.ui

# With options
rho-tts-ui --host 0.0.0.0 --port 8080 --device cpu --share
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `~/.rho_tts/config.json` | Path to config JSON file |
| `--host` | `127.0.0.1` | Server bind address |
| `--port` | `7860` | Server port |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--share` | off | Create a public Gradio link |

The config path can also be set via the `RHO_TTS_CONFIG` environment variable.

### Tabs

- **Generate** — Select a model and voice, enter text, and generate audio with real-time playback. Includes phonetic mapping overrides per voice/model pair.
- **Voices** — Upload reference audio and transcripts to create reusable voice profiles (stored in `~/.rho_tts/voices/`).
- **Models** — Configure TTS providers with custom thresholds and parameters.

## Cancellation

For long-running generation in web servers or UIs:

```python
from rho_tts import CancellationToken, TTSFactory

token = CancellationToken()

# In worker thread
tts = TTSFactory.get_tts_instance(provider="qwen", reference_audio="voice.wav", reference_text="...")
result = tts.generate(texts, "output", cancellation_token=token)

# In controller thread (e.g., on user cancel button)
token.cancel()
```

## License

MIT
